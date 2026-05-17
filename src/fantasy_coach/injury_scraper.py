"""NRL.com injury-list scraper (#208 PR C).

NRL.com publishes a weekly injury list as a news article in unstructured
prose — there's no JSON endpoint to scrape. We solve this with a Gemini
call that takes the raw page text + a strict JSON schema and returns a
list of ``InjuryReport``-shaped dicts. The parser is split from the
fetcher so tests can exercise the parsing path with a fake Gemini
response and the live source-of-truth lives in one place.

Player-name resolution: the parser returns names; we map them to
``player_id`` + ``team_id`` via ``build_player_index`` over the matches
store. Names that don't resolve are logged + dropped — better to silently
skip than to invent IDs.

Historical backfill (Wayback Machine for 2024+2025) is a follow-up
issue; the MVP supports a single URL per invocation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

from fantasy_coach.injury import InjuryReport, InjuryStatus
from fantasy_coach.scraper import DEFAULT_TIMEOUT, USER_AGENT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source Protocol — abstracts "where do we get the page text from"
# ---------------------------------------------------------------------------


class InjuryListSource(Protocol):
    def fetch(self, url: str) -> str: ...


@dataclass
class HttpInjuryListSource:
    """Fetches the URL via httpx + strips HTML to plain text.

    Raises ``httpx.HTTPError`` on failure — callers decide whether to
    retry or skip. Throttle is not enforced here because injury-list
    scraping is one-call-per-week, far below the hot-path scraper's
    politeness window.
    """

    timeout: float = DEFAULT_TIMEOUT
    user_agent: str = USER_AGENT

    def fetch(self, url: str) -> str:
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": self.user_agent})
            resp.raise_for_status()
            return strip_html_to_text(resp.text)


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def strip_html_to_text(html: str) -> str:
    """Quick-and-dirty HTML → plain text, no BeautifulSoup dependency.

    Drops <script>/<style> blocks, strips remaining tags, decodes the
    handful of common entities, and collapses whitespace. Good enough
    for feeding into Gemini, which is robust to messy input.
    """
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = (
        cleaned.replace("&amp;", "&")
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
        .replace("&ndash;", "-")
        .replace("&mdash;", "-")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _BLANK_LINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Parser Protocol — abstracts "how do we structure the prose"
# ---------------------------------------------------------------------------


class InjuryListParser(Protocol):
    def parse(
        self,
        *,
        text: str,
        season: int,
        round: int,
        scraped_at: datetime,
        player_index: PlayerIndex,
    ) -> list[InjuryReport]: ...


# Status synonyms the LLM might emit — normalised back to InjuryStatus.
_STATUS_ALIASES: dict[str, InjuryStatus] = {
    "out": InjuryStatus.OUT,
    "ruled out": InjuryStatus.OUT,
    "unavailable": InjuryStatus.OUT,
    "test": InjuryStatus.TEST,
    "fitness test": InjuryStatus.TEST,
    "doubtful": InjuryStatus.TEST,
    "21_man_squad": InjuryStatus.SQUAD_21,
    "21-man squad": InjuryStatus.SQUAD_21,
    "21 man squad": InjuryStatus.SQUAD_21,
    "extended squad": InjuryStatus.SQUAD_21,
    "returning": InjuryStatus.RETURNING,
    "return": InjuryStatus.RETURNING,
    "first match back": InjuryStatus.RETURNING,
    "suspended": InjuryStatus.SUSPENDED,
    "suspension": InjuryStatus.SUSPENDED,
}


GEMINI_SYSTEM_PROMPT = """\
You are an information extractor for an NRL injury-list news article.
Return JSON ONLY (no prose, no markdown fences). The JSON must be a
single object of shape:

{
  "reports": [
    {
      "first_name": "string",
      "last_name": "string",
      "team_name": "string (NRL club nickname or full name)",
      "status": "out | test | 21_man_squad | returning | suspended",
      "weeks_out": number | null,
      "body_part": "string | null"
    },
    ...
  ]
}

Map every player named in the article to one entry. If the article
mentions a body part ("hamstring") put it in body_part. If it estimates
a return window ("expected back in 2 weeks") put weeks_out=2; otherwise
null. Use the closest matching status from the enum. Do NOT invent
players who aren't in the article.
"""


@dataclass
class GeminiInjuryListParser:
    """Gemini-backed parser — single LLM call extracts structured rows.

    `client` is anything with a ``generate(prompt, *, system, max_output_tokens) -> obj.text``
    interface. The real ``GeminiClient`` from ``commentary/client.py`` fits.
    """

    client: Any
    source_label: str = "nrl.com"
    max_output_tokens: int = 4096

    def parse(
        self,
        *,
        text: str,
        season: int,
        round: int,
        scraped_at: datetime,
        player_index: PlayerIndex,
    ) -> list[InjuryReport]:
        prompt = f"Round: {round}\nSeason: {season}\n\nArticle text:\n{text}"
        response = self.client.generate(
            prompt,
            system=GEMINI_SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
        )
        raw_rows = _parse_gemini_json(response.text)
        return _materialise_reports(
            raw_rows,
            season=season,
            round=round,
            scraped_at=scraped_at,
            player_index=player_index,
            source=self.source_label,
        )


def _parse_gemini_json(text: str) -> list[dict[str, Any]]:
    """Extract a JSON object from the LLM's text response.

    Tolerates the common failure modes: leading/trailing prose, ```json
    fences, single-line preamble. Returns the ``reports`` array, or
    [] if the structure is unrecognisable.
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else None
    if candidate is None:
        # Find the outermost {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning("injury_parser: no JSON object found in Gemini response")
            return []
        candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.warning("injury_parser: JSON decode failed: %s", exc)
        return []
    if not isinstance(data, dict):
        return []
    rows = data.get("reports") or []
    return [r for r in rows if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Player-name resolution — names → (player_id, team_id)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerIndex:
    """Resolves (first_name, last_name) and team_name → (player_id, team_id).

    Built once per scrape from the matches store. Names are normalised
    by lowercasing + stripping punctuation so "Reece Walsh" matches
    "reece walsh" and "O'Sullivan" matches "OSullivan".
    """

    by_full_name: dict[tuple[str, str], tuple[int, int]]
    team_id_by_name: dict[str, int]

    def resolve(
        self, *, first_name: str, last_name: str, team_name: str | None
    ) -> tuple[int, int] | None:
        key = (_norm_name(first_name), _norm_name(last_name))
        if key in self.by_full_name:
            return self.by_full_name[key]
        # Last-name-only fallback when first-name spelling differs
        # (e.g. "Sam" vs "Samuel"). Only safe when the team is given —
        # otherwise we'd collide on common surnames.
        if team_name:
            team_id = self.team_id_by_name.get(_norm_name(team_name))
            if team_id is None:
                return None
            candidates = [
                (pid, tid)
                for (_, ln), (pid, tid) in self.by_full_name.items()
                if ln == _norm_name(last_name) and tid == team_id
            ]
            if len(candidates) == 1:
                return candidates[0]
        return None


_NAME_RE = re.compile(r"[^\w\s]")


def _norm_name(s: str) -> str:
    return _NAME_RE.sub("", s.casefold()).strip()


def build_player_index(matches: list[Any]) -> PlayerIndex:
    """Build a name → (player_id, team_id) lookup over a list of MatchRow.

    Players appearing in multiple matches keep their *most recent*
    team_id so trade-window moves resolve to the new club. Caller is
    responsible for selecting the right slice of matches (e.g. last
    season + this season).
    """
    by_full_name: dict[tuple[str, str], tuple[int, int]] = {}
    team_id_by_name: dict[str, int] = {}
    for match in sorted(matches, key=lambda m: m.start_time):
        for side_name in ("home", "away"):
            team = getattr(match, side_name)
            team_id = team.team_id
            for nick_or_name in (
                getattr(team, "nick_name", None),
                getattr(team, "name", None),
            ):
                if nick_or_name:
                    team_id_by_name[_norm_name(nick_or_name)] = team_id
            for player in team.players:
                first = player.first_name or ""
                last = player.last_name or ""
                if not first or not last:
                    continue
                by_full_name[(_norm_name(first), _norm_name(last))] = (
                    player.player_id,
                    team_id,
                )
    return PlayerIndex(by_full_name=by_full_name, team_id_by_name=team_id_by_name)


# ---------------------------------------------------------------------------
# Materialisation — Gemini rows → InjuryReport[]
# ---------------------------------------------------------------------------


def _materialise_reports(
    rows: list[dict[str, Any]],
    *,
    season: int,
    round: int,
    scraped_at: datetime,
    player_index: PlayerIndex,
    source: str,
) -> list[InjuryReport]:
    out: list[InjuryReport] = []
    skipped = 0
    for row in rows:
        first = str(row.get("first_name") or "").strip()
        last = str(row.get("last_name") or "").strip()
        team_name = row.get("team_name")
        if not first or not last:
            skipped += 1
            continue
        resolved = player_index.resolve(
            first_name=first,
            last_name=last,
            team_name=str(team_name) if team_name else None,
        )
        if resolved is None:
            logger.info(
                "injury_parser: could not resolve player_id for %s %s (team=%r); skipping",
                first,
                last,
                team_name,
            )
            skipped += 1
            continue
        player_id, team_id = resolved

        status_raw = str(row.get("status") or "").strip().lower()
        status = _STATUS_ALIASES.get(status_raw)
        if status is None:
            try:
                status = InjuryStatus(status_raw)
            except ValueError:
                logger.info("injury_parser: unknown status %r for %s %s", status_raw, first, last)
                skipped += 1
                continue

        weeks_out = _coerce_int(row.get("weeks_out"))
        body_part = row.get("body_part")
        body_part = str(body_part).strip() if body_part else None

        out.append(
            InjuryReport(
                player_id=player_id,
                season=season,
                round=round,
                team_id=team_id,
                status=status,
                weeks_out=weeks_out,
                body_part=body_part,
                source=source,
                scraped_at=scraped_at,
            )
        )
    if skipped:
        logger.info("injury_parser: skipped %d rows (unresolved or invalid)", skipped)
    return out


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    try:
        parsed = int(str(value).strip())
        return parsed if parsed >= 0 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def scrape_injury_list(
    *,
    url: str,
    season: int,
    round: int,
    source: InjuryListSource,
    parser: InjuryListParser,
    player_index: PlayerIndex,
    now: datetime | None = None,
) -> list[InjuryReport]:
    """Fetch + parse an NRL.com injury list page into structured rows.

    Returns the parsed reports. Doesn't write anywhere — caller hands
    them to ``InjuryReportRepository.record_report`` so storage is
    decoupled from scraping.
    """
    text = source.fetch(url)
    if not text:
        logger.warning("injury_scraper: empty body from %s", url)
        return []
    scraped_at = now or datetime.now(UTC)
    return parser.parse(
        text=text,
        season=season,
        round=round,
        scraped_at=scraped_at,
        player_index=player_index,
    )


# ---------------------------------------------------------------------------
# Live URL auto-discovery (#268)
# ---------------------------------------------------------------------------

# NRL.com's weekly injury+changes article is published at
#     /news/YYYY/MM/DD/nrl-late-mail-round-N-<slug>/
# (the #268 issue called these "injury-list" articles but that slug has
# never actually existed on NRL.com — verified across 2024/2025 via CDX).
NEWS_INDEX_URL = "https://www.nrl.com/news/?tag=late-mail"

# Anchored to `nrl-` so the parallel NRLW (women's) `nrlw-late-mail-round-N`
# articles don't match — NRLW has different player IDs and team IDs.
_LATE_MAIL_HREF_RE = re.compile(
    r"/news/(?P<season>20\d{2})/\d{1,2}/\d{1,2}/"
    r"[^\"'<>\s]*(?<![a-z])nrl-late-mail[^\"'<>\s]*round[-_](?P<round>\d{1,2})[^\"'<>\s]*",
    re.IGNORECASE,
)


def discover_injury_list_url(
    season: int,
    round_: int,
    *,
    client: Any | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str | None:
    """Crawl the NRL.com news index and find the late-mail article for ``(season, round_)``.

    The index is a Next.js app — article URLs are embedded both in the
    ``__NEXT_DATA__`` JSON payload and as plain ``<a href="...">`` anchors.
    We scan the raw HTML for any URL matching the canonical
    ``/news/YYYY/.../nrl-late-mail-round-N-...`` shape, then return the
    one whose parsed ``(season, round)`` matches the request.

    Returns ``None`` when no match is found. Callers should fall back
    to the ``--url`` flag in that case.
    """
    own_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
    try:
        resp = client.get(NEWS_INDEX_URL)
        resp.raise_for_status()
        html = resp.text
    finally:
        if own_client:
            client.close()

    candidates: list[str] = []
    for match in _LATE_MAIL_HREF_RE.finditer(html):
        url_season = int(match.group("season"))
        url_round = int(match.group("round"))
        if url_season != season or url_round != round_:
            continue
        path = match.group(0)
        absolute = f"https://www.nrl.com{path}"
        if absolute not in candidates:
            candidates.append(absolute)

    if not candidates:
        logger.info(
            "injury_scraper: discover_injury_list_url found no late-mail article "
            "for season=%d round=%d",
            season,
            round_,
        )
        return None
    # Multiple articles for the same round are rare (publishing fix-ups);
    # prefer the longest URL which usually carries the most-specific slug
    # variant (the corrected one).
    return max(candidates, key=len)
