"""Wayback Machine historical backfill for NRL.com late-mail articles (#268).

NRL.com publishes a weekly ``nrl-late-mail-round-N`` article (the
official source for injuries, suspensions, late changes, and 21-man
squad reveals). Live NRL.com retains only the current season, so to
populate ``injury_reports`` for 2024–2025 walk-forward training we pull
the snapshots from the Internet Archive.

NB: the #268 issue body called these "injury-list" articles, but NRL.com
has never published anything at that URL slug — the actual slug is
``nrl-late-mail-round-N``. Both 2024 (rounds 1–27) and 2025 (rounds 1–26)
are fully covered on Wayback.

Pipeline:

1. ``search_cdx_year`` — query the CDX API for one year of nrl.com news.
   We don't use the server-side regex ``filter`` because it routinely
   triggers 504s; instead we pull the full year (≈ 5–10k rows, < 3 MB)
   and filter client-side.
2. ``dedupe_latest_per_round`` — collapse to one snapshot per
   ``(season, round)``. Wayback often holds many captures of the same
   article; the latest is the most complete.
3. ``WaybackInjuryListSource`` — drop-in replacement for
   ``HttpInjuryListSource`` that fetches the archived HTML via the
   ``id_`` modifier (suppresses Wayback's banner injection).
4. Hand each ``(season, round, url)`` to ``scrape_injury_list`` exactly
   as the live scraper does.

The storage layer already upserts on ``(player_id, season, round, source)``
so re-running is idempotent.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx

from fantasy_coach.injury_scraper import strip_html_to_text
from fantasy_coach.scraper import DEFAULT_TIMEOUT, USER_AGENT

logger = logging.getLogger(__name__)

CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_FETCH_TEMPLATE = "https://web.archive.org/web/{timestamp}id_/{url}"

# The CDX search over nrl.com/news/* can return many MB; the public
# endpoint is also routinely slow under load. 15 s (the default scraper
# timeout) trips frequently. Backfill runs once per season so a generous
# ceiling is fine.
_CDX_TIMEOUT = 120.0

# NRL.com weekly injury articles live at
#     /news/YYYY/MM/DD/nrl-late-mail-round-N-<trailing-slug>/
#
# Critically: nrl.com ALSO publishes a parallel NRLW (women's competition)
# weekly article at /news/YYYY/MM/DD/nrlw-late-mail-round-N-.../. NRLW has
# its own player IDs / team IDs / fixture list — those reports must NOT
# go into injury_reports for the men's competition. We anchor the slug
# to the `nrl-` prefix specifically (with a `(?<![a-z])` lookbehind so
# `nrl-late-mail` matches but `nrlw-late-mail` does not).
_LATE_MAIL_RE = re.compile(
    r"/news/(?P<season>20\d{2})/\d{1,2}/\d{1,2}/"
    r"[^/]*(?<![a-z])nrl-late-mail[^/]*round[-_](?P<round>\d{1,2})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WaybackSnapshot:
    """One archived capture of a URL."""

    timestamp: str  # YYYYMMDDhhmmss
    original_url: str

    @property
    def fetch_url(self) -> str:
        """Wayback fetch URL using the ``id_`` modifier (no banner injection)."""
        return WAYBACK_FETCH_TEMPLATE.format(timestamp=self.timestamp, url=self.original_url)


def search_cdx_year(
    season: int,
    *,
    client: httpx.Client | None = None,
    timeout: float = _CDX_TIMEOUT,
) -> list[WaybackSnapshot]:
    """Query the Wayback CDX API for one year of ``nrl.com/news/`` snapshots.

    Returns every snapshot under ``nrl.com/news/{season}/`` collapsed to
    one row per unique URL. Callers should filter further (e.g. with
    ``filter_late_mail``) — server-side regex ``filter`` parameters
    routinely cause the CDX endpoint to return 504, so we filter
    client-side instead.
    """
    params = {
        "url": f"nrl.com/news/{season}/",
        "matchType": "prefix",
        "from": f"{season}0101",
        "to": f"{season}1231",
        "collapse": "urlkey",
        "output": "json",
    }
    own_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
    try:
        resp = client.get(CDX_API, params=params)
        resp.raise_for_status()
        rows = resp.json()
    finally:
        if own_client:
            client.close()

    if not rows or len(rows) < 2:
        return []
    header = rows[0]
    try:
        ts_idx = header.index("timestamp")
        url_idx = header.index("original")
    except ValueError:
        logger.warning("wayback: unexpected CDX header columns: %s", header)
        return []
    return [
        WaybackSnapshot(timestamp=str(r[ts_idx]), original_url=str(r[url_idx]))
        for r in rows[1:]
        if len(r) > max(ts_idx, url_idx)
    ]


def filter_late_mail(snapshots: Iterable[WaybackSnapshot]) -> list[WaybackSnapshot]:
    """Keep only snapshots whose URL is a late-mail article."""
    return [s for s in snapshots if _LATE_MAIL_RE.search(s.original_url)]


def parse_season_round(url: str) -> tuple[int, int] | None:
    """Extract ``(season, round)`` from an NRL.com late-mail article URL.

    Returns ``None`` when the URL doesn't match the regular-round pattern
    (e.g. finals articles, mis-tagged news posts, non-late-mail URLs).
    """
    m = _LATE_MAIL_RE.search(url)
    if not m:
        return None
    return int(m.group("season")), int(m.group("round"))


def dedupe_latest_per_round(
    snapshots: Iterable[WaybackSnapshot],
) -> list[WaybackSnapshot]:
    """Keep only the latest snapshot per ``(season, round)``.

    The Wayback Machine often holds many captures of the same article —
    later captures usually include corrections + additions to the live
    article between bot crawls, so they're a strict superset.

    Snapshots whose URL doesn't parse to ``(season, round)`` are dropped.
    """
    by_key: dict[tuple[int, int], WaybackSnapshot] = {}
    for snap in snapshots:
        sr = parse_season_round(snap.original_url)
        if sr is None:
            continue
        existing = by_key.get(sr)
        if existing is None or snap.timestamp > existing.timestamp:
            by_key[sr] = snap
    return [by_key[k] for k in sorted(by_key)]


@dataclass
class WaybackInjuryListSource:
    """``InjuryListSource`` implementation that fetches via web.archive.org.

    Drop-in replacement for ``HttpInjuryListSource`` — same single-method
    Protocol so ``scrape_injury_list`` works unchanged.
    """

    timeout: float = DEFAULT_TIMEOUT
    user_agent: str = USER_AGENT

    def fetch(self, url: str) -> str:
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": self.user_agent})
            resp.raise_for_status()
            return strip_html_to_text(resp.text)


def filter_seasons(
    snapshots: Iterable[WaybackSnapshot],
    seasons: Iterable[int],
) -> list[WaybackSnapshot]:
    """Keep only snapshots whose parsed season is in ``seasons``."""
    wanted = set(seasons)
    out: list[WaybackSnapshot] = []
    for snap in snapshots:
        sr = parse_season_round(snap.original_url)
        if sr is not None and sr[0] in wanted:
            out.append(snap)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class BackfillSummary:
    """Per-snapshot outcome for ``backfill_injuries`` callers + tests."""

    season: int
    round: int
    url: str
    timestamp: str
    reports_written: int
    error: str | None = None


def backfill_injuries(
    *,
    seasons: Iterable[int],
    source: Any,  # WaybackInjuryListSource or test stub
    parser: Any,  # InjuryListParser
    player_index: Any,  # PlayerIndex
    injury_repo: Any,  # InjuryReportRepository
    snapshots: Iterable[WaybackSnapshot] | None = None,
    source_label: str = "wayback:nrl.com",
) -> list[BackfillSummary]:
    """Drive the full backfill: discover snapshots → fetch → parse → store.

    ``snapshots`` lets callers (and tests) supply a pre-discovered list
    instead of hitting the live CDX API. When omitted, we hit CDX once
    per season and filter client-side for late-mail articles.

    Each parsed report is handed to ``injury_repo.record_report`` one at
    a time so a single bad article doesn't lose the whole batch.
    """
    from fantasy_coach.injury_scraper import scrape_injury_list  # noqa: PLC0415

    if snapshots is None:
        all_snaps: list[WaybackSnapshot] = []
        for season in sorted(set(seasons)):
            year_snaps = search_cdx_year(season)
            all_snaps.extend(filter_late_mail(year_snaps))
            logger.info(
                "wayback: season=%d fetched %d snapshots, %d late-mail",
                season,
                len(year_snaps),
                sum(1 for s in year_snaps if _LATE_MAIL_RE.search(s.original_url)),
            )
        snapshots = all_snaps
    deduped = filter_seasons(dedupe_latest_per_round(snapshots), seasons)

    results: list[BackfillSummary] = []
    for snap in deduped:
        sr = parse_season_round(snap.original_url)
        if sr is None:
            continue
        season, round_ = sr
        try:
            reports = scrape_injury_list(
                url=snap.fetch_url,
                season=season,
                round=round_,
                source=source,
                parser=_with_source_label(parser, source_label),
                player_index=player_index,
            )
            for report in reports:
                injury_repo.record_report(report)
            results.append(
                BackfillSummary(
                    season=season,
                    round=round_,
                    url=snap.original_url,
                    timestamp=snap.timestamp,
                    reports_written=len(reports),
                )
            )
        except Exception as exc:  # noqa: BLE001 — per-snapshot isolation
            logger.warning(
                "wayback: backfill failed for %s (snap %s): %s",
                snap.original_url,
                snap.timestamp,
                exc,
            )
            results.append(
                BackfillSummary(
                    season=season,
                    round=round_,
                    url=snap.original_url,
                    timestamp=snap.timestamp,
                    reports_written=0,
                    error=str(exc),
                )
            )
    return results


def _with_source_label(parser: Any, source_label: str) -> Any:
    """Override the parser's ``source_label`` so Wayback rows are tagged
    distinctly from live scrapes (``wayback:nrl.com`` vs ``nrl.com``).

    Doing it here keeps the parser API immutable from the caller's
    perspective — they pass a single ``GeminiInjuryListParser`` and we
    just retag the source field on the way through.
    """
    if hasattr(parser, "source_label"):
        # Frozen dataclass or similar — silently leave the original label.
        with contextlib.suppress(AttributeError):
            parser.source_label = source_label
    return parser
