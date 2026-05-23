"""Tests for the Wayback Machine injury-list backfill (#268)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import pytest

from fantasy_coach.injury import InjuryReport, InjuryStatus
from fantasy_coach.injury_scraper import PlayerIndex
from fantasy_coach.wayback import (
    CDX_API,
    WaybackInjuryListSource,
    WaybackSnapshot,
    backfill_injuries,
    dedupe_latest_per_round,
    filter_late_mail,
    filter_seasons,
    parse_season_round,
    search_cdx_year,
)

# ---------------------------------------------------------------------------
# parse_season_round
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        # Canonical 2024 article from CDX
        (
            "https://www.nrl.com/news/2024/02/29/nrl-late-mail-round-1---all-go-for-vegas/",
            (2024, 1),
        ),
        # 2025, double-digit round
        (
            "https://www.nrl.com/news/2025/03/12/nrl-late-mail-round-2---taumalolo-eyes-return-too-cleared-for-take-off/",
            (2025, 2),
        ),
        # AMP variant
        (
            "https://www.nrl.com/news/2024/02/29/nrl-late-mail-round-1---all-go-for-vegas/amp/",
            (2024, 1),
        ),
        # Case insensitive.
        (
            "https://www.nrl.com/news/2025/07/22/NRL-Late-Mail-Round-15-injuries-mounting/",
            (2025, 15),
        ),
    ],
)
def test_parse_season_round_extracts_pair(url: str, expected: tuple[int, int]) -> None:
    assert parse_season_round(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        # Team-list articles share the round-N suffix but aren't late-mail.
        "https://www.nrl.com/news/2025/02/25/nrl-team-lists-round-1-rugby-league-las-vegas/",
        # Finals articles — by design we don't backfill them (no round-N).
        "https://www.nrl.com/news/2025/09/24/qualifying-final-late-mail/",
        # No /news/ prefix.
        "https://www.nrl.com/draw/2024/05/05/late-mail-round-9/",
        # Missing date segments.
        "https://www.nrl.com/news/late-mail-round-7/",
        # Wrong year format.
        "https://www.nrl.com/news/1999/01/01/nrl-late-mail-round-1/",
        # Article about an injured player but not the weekly late-mail.
        "https://www.nrl.com/news/2025/01/24/hess-with-positive-update-on-return-from-his-acl-injury/",
        # NRLW (women's) weekly late-mail — different competition, different
        # player IDs and team IDs. Real URLs from the 2024 Wayback CDX dump.
        "https://www.nrl.com/news/2024/08/29/nrlw-late-mail-round-6-southwell-a-chance-to-return/",
        "https://www.nrl.com/news/2024/09/05/nrlw-late-mail-round-7-young-guns-set-to-return-for-titans/",
        "https://www.nrl.com/news/2024/09/18/nrlw-late-mail-round-9-brown-set-for-final-wing-fling/amp/",
    ],
)
def test_parse_season_round_returns_none_for_unmatched(url: str) -> None:
    assert parse_season_round(url) is None


# ---------------------------------------------------------------------------
# dedupe_latest_per_round
# ---------------------------------------------------------------------------


def _snap(ts: str, url: str) -> WaybackSnapshot:
    return WaybackSnapshot(timestamp=ts, original_url=url)


_LATE_MAIL_2024_R7 = "https://www.nrl.com/news/2024/04/15/nrl-late-mail-round-7-update/"
_LATE_MAIL_2025_R13 = "https://www.nrl.com/news/2025/06/03/nrl-late-mail-round-13-injuries/"


def test_dedupe_keeps_latest_per_round() -> None:
    snaps = [
        _snap("20240415120000", _LATE_MAIL_2024_R7),
        _snap("20240416140000", _LATE_MAIL_2024_R7),
        _snap("20240416120000", _LATE_MAIL_2024_R7),
        _snap("20250603100000", _LATE_MAIL_2025_R13),
    ]
    deduped = dedupe_latest_per_round(snaps)
    assert len(deduped) == 2
    # Sorted by (season, round)
    assert parse_season_round(deduped[0].original_url) == (2024, 7)
    assert deduped[0].timestamp == "20240416140000"  # latest of the three
    assert parse_season_round(deduped[1].original_url) == (2025, 13)


def test_dedupe_drops_unparseable_urls() -> None:
    snaps = [
        _snap("20240901000000", "https://www.nrl.com/news/2024/09/01/qualifying-final-late-mail/"),
        _snap("20240415000000", _LATE_MAIL_2024_R7),
    ]
    deduped = dedupe_latest_per_round(snaps)
    assert len(deduped) == 1
    assert parse_season_round(deduped[0].original_url) == (2024, 7)


def test_filter_late_mail_drops_other_articles() -> None:
    snaps = [
        _snap("20250225000000", "https://www.nrl.com/news/2025/02/25/nrl-team-lists-round-1/"),
        _snap("20250603000000", _LATE_MAIL_2025_R13),
        _snap(
            "20250124000000",
            "https://www.nrl.com/news/2025/01/24/hess-positive-update-on-return-from-acl-injury/",
        ),
    ]
    kept = filter_late_mail(snaps)
    assert len(kept) == 1
    assert kept[0].original_url == _LATE_MAIL_2025_R13


# ---------------------------------------------------------------------------
# filter_seasons
# ---------------------------------------------------------------------------


def test_filter_seasons_keeps_requested_only() -> None:
    snaps = [
        _snap("20240415000000", _LATE_MAIL_2024_R7),
        _snap("20250603000000", _LATE_MAIL_2025_R13),
        _snap("20260415000000", "https://www.nrl.com/news/2026/04/15/nrl-late-mail-round-7/"),
    ]
    out = filter_seasons(snaps, [2024, 2026])
    assert {parse_season_round(s.original_url)[0] for s in out} == {2024, 2026}


# ---------------------------------------------------------------------------
# WaybackSnapshot.fetch_url
# ---------------------------------------------------------------------------


def test_fetch_url_uses_id_modifier() -> None:
    snap = _snap("20240415120000", _LATE_MAIL_2024_R7)
    assert snap.fetch_url == ("https://web.archive.org/web/20240415120000id_/" + _LATE_MAIL_2024_R7)


# ---------------------------------------------------------------------------
# search_cdx — uses httpx MockTransport so no live network calls
# ---------------------------------------------------------------------------


def test_search_cdx_year_returns_snapshots_from_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(CDX_API)
        # Request is one year — verify the params reflect that.
        assert "nrl.com%2Fnews%2F2024%2F" in str(request.url) or "nrl.com/news/2024/" in str(
            request.url
        )
        body = [
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            [
                "com,nrl)/news/2024/02/29/nrl-late-mail-round-1---all-go-for-vegas",
                "20240301000000",
                "https://www.nrl.com/news/2024/02/29/nrl-late-mail-round-1---all-go-for-vegas/",
                "text/html",
                "200",
                "abc",
                "12345",
            ],
            [
                # Non-late-mail article — caller filters this out.
                "com,nrl)/news/2024/03/01/some-other-news",
                "20240302000000",
                "https://www.nrl.com/news/2024/03/01/some-other-news/",
                "text/html",
                "200",
                "def",
                "1000",
            ],
        ]
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    snaps = search_cdx_year(2024, client=client)
    assert len(snaps) == 2
    assert snaps[0].timestamp == "20240301000000"
    # Caller is expected to pipe through filter_late_mail.
    late_mail_only = filter_late_mail(snaps)
    assert len(late_mail_only) == 1
    assert "late-mail-round-1" in late_mail_only[0].original_url


def test_search_cdx_year_returns_empty_on_empty_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert search_cdx_year(2024, client=client) == []


def test_search_cdx_year_returns_empty_when_only_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[["urlkey", "timestamp", "original"]])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert search_cdx_year(2024, client=client) == []


# ---------------------------------------------------------------------------
# WaybackInjuryListSource — fetches + strips HTML
# ---------------------------------------------------------------------------


def test_wayback_source_strips_html_and_returns_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><body><h1>Round 7 injuries</h1><p>Reece Walsh out</p></body></html>",
        )

    # WaybackInjuryListSource creates its own client internally; monkeypatch
    # httpx.Client to use our MockTransport.
    import httpx as _httpx

    real_client = _httpx.Client

    def _patched_client(*args: Any, **kwargs: Any) -> _httpx.Client:
        kwargs.pop("timeout", None)
        return real_client(transport=_httpx.MockTransport(handler))

    _httpx.Client = _patched_client  # type: ignore[misc]
    try:
        source = WaybackInjuryListSource()
        text = source.fetch(
            "https://web.archive.org/web/20240415120000id_/"
            "https://www.nrl.com/news/2024/04/15/round-7-injury-list/"
        )
    finally:
        _httpx.Client = real_client  # type: ignore[misc]
    assert "Round 7 injuries" in text
    assert "Reece Walsh out" in text
    assert "<" not in text


# ---------------------------------------------------------------------------
# backfill_injuries orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _StubSource:
    article_text: str
    fetched: list[str] = field(default_factory=list)

    def fetch(self, url: str) -> str:
        self.fetched.append(url)
        return self.article_text


@dataclass
class _StubParser:
    source_label: str = "wayback:nrl.com"
    calls: list[dict] = field(default_factory=list)
    fixed_reports: list[InjuryReport] | None = None

    def parse(
        self,
        *,
        text: str,
        season: int,
        round: int,
        scraped_at: datetime,
        player_index: PlayerIndex,
    ) -> list[InjuryReport]:
        self.calls.append({"season": season, "round": round, "label": self.source_label})
        if self.fixed_reports is not None:
            return self.fixed_reports
        return [
            InjuryReport(
                player_id=1001,
                season=season,
                round=round,
                team_id=500001,
                status=InjuryStatus.OUT,
                weeks_out=2,
                body_part="shoulder",
                source=self.source_label,
                scraped_at=scraped_at,
            )
        ]


class _RecordingRepo:
    def __init__(self) -> None:
        self.records: list[InjuryReport] = []

    def record_report(self, report: InjuryReport) -> None:
        self.records.append(report)


def _player_index() -> PlayerIndex:
    return PlayerIndex(
        by_full_name={("reece", "walsh"): (1001, 500001)},
        team_id_by_name={"broncos": 500001},
    )


def test_backfill_dedupes_snapshots_and_writes_reports() -> None:
    snaps = [
        _snap("20240415120000", _LATE_MAIL_2024_R7),
        # Newer snapshot of the same article — should win the dedupe.
        _snap("20240416140000", _LATE_MAIL_2024_R7),
        _snap("20250603100000", _LATE_MAIL_2025_R13),
    ]
    source = _StubSource(article_text="Round X injuries — Reece Walsh OUT")
    parser = _StubParser()
    injury_repo = _RecordingRepo()

    results = backfill_injuries(
        seasons=[2024, 2025],
        source=source,
        parser=parser,
        player_index=_player_index(),
        injury_repo=injury_repo,
        snapshots=snaps,
        source_label="wayback:nrl.com",
    )

    # Two articles after dedupe (one per season+round).
    assert len(results) == 2
    assert [r.reports_written for r in results] == [1, 1]
    assert {(r.season, r.round) for r in results} == {(2024, 7), (2025, 13)}

    # The fetcher saw the latest 2024 capture, not the earlier one.
    fetched_2024 = [u for u in source.fetched if "20240416140000" in u]
    assert fetched_2024, source.fetched
    fetched_old = [u for u in source.fetched if "20240415120000" in u]
    assert not fetched_old

    # All written reports carry the wayback source label.
    assert len(injury_repo.records) == 2
    assert all(r.source == "wayback:nrl.com" for r in injury_repo.records)


def test_backfill_filters_to_requested_seasons() -> None:
    snaps = [
        _snap("20240415000000", _LATE_MAIL_2024_R7),
        _snap("20260415000000", "https://www.nrl.com/news/2026/04/15/nrl-late-mail-round-7/"),
    ]
    source = _StubSource(article_text="x")
    parser = _StubParser()
    repo = _RecordingRepo()

    results = backfill_injuries(
        seasons=[2026],
        source=source,
        parser=parser,
        player_index=_player_index(),
        injury_repo=repo,
        snapshots=snaps,
    )
    assert len(results) == 1
    assert results[0].season == 2026


def test_backfill_records_per_snapshot_errors_without_aborting() -> None:
    snaps = [
        _snap("20240415000000", _LATE_MAIL_2024_R7),
        _snap("20250603000000", _LATE_MAIL_2025_R13),
    ]

    @dataclass
    class _FailingSource:
        fail_on: str
        fetched: list[str] = field(default_factory=list)

        def fetch(self, url: str) -> str:
            self.fetched.append(url)
            if self.fail_on in url:
                raise RuntimeError("simulated wayback 503")
            return "article body"

    source = _FailingSource(fail_on="20240415000000")
    parser = _StubParser()
    repo = _RecordingRepo()

    results = backfill_injuries(
        seasons=[2024, 2025],
        source=source,
        parser=parser,
        player_index=_player_index(),
        injury_repo=repo,
        snapshots=snaps,
    )
    assert len(results) == 2
    # First failed, second succeeded.
    errored = [r for r in results if r.error is not None]
    ok = [r for r in results if r.error is None]
    assert len(errored) == 1 and errored[0].season == 2024
    assert len(ok) == 1 and ok[0].season == 2025 and ok[0].reports_written == 1
    # Only the successful one wrote to the repo.
    assert len(repo.records) == 1


def test_backfill_overrides_parser_source_label() -> None:
    """The orchestrator should retag the parser so wayback reports are
    distinguishable from live nrl.com scrapes — different source field."""
    snaps = [_snap("20240415000000", _LATE_MAIL_2024_R7)]
    source = _StubSource(article_text="x")
    parser = _StubParser(source_label="original-label")
    repo = _RecordingRepo()

    backfill_injuries(
        seasons=[2024],
        source=source,
        parser=parser,
        player_index=_player_index(),
        injury_repo=repo,
        snapshots=snaps,
        source_label="wayback:nrl.com",
    )
    # The parser was mutated to use the wayback label, and the report
    # written to storage carries it.
    assert parser.source_label == "wayback:nrl.com"
    assert repo.records[0].source == "wayback:nrl.com"
