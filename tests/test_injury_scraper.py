"""Tests for the injury-list scraper (#208 PR C)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from fantasy_coach.features import PlayerRow, TeamRow
from fantasy_coach.injury import InjuryStatus
from fantasy_coach.injury_scraper import (
    GEMINI_SYSTEM_PROMPT,
    GeminiInjuryListParser,
    PlayerIndex,
    _materialise_reports,
    _parse_gemini_json,
    build_player_index,
    discover_injury_list_url,
    scrape_injury_list,
    strip_html_to_text,
)

# ---------------------------------------------------------------------------
# strip_html_to_text
# ---------------------------------------------------------------------------


def test_strip_html_drops_tags_scripts_and_entities() -> None:
    html = (
        "<html><head><style>body{}</style></head>"
        "<body>"
        "<script>alert('x')</script>"
        "<h1>Round 8 injury list</h1>"
        "<p>Reece Walsh &mdash; <strong>OUT</strong> &nbsp;(shoulder)</p>"
        "</body></html>"
    )
    text = strip_html_to_text(html)
    # Tag noise gone, text preserved.
    assert "Round 8 injury list" in text
    assert "Reece Walsh - OUT (shoulder)" in text
    assert "<" not in text
    assert "alert" not in text  # script body dropped
    assert "&nbsp;" not in text  # entities decoded


def test_strip_html_collapses_whitespace() -> None:
    html = "<p>hello   world</p>\n\n\n<p>line two</p>"
    text = strip_html_to_text(html)
    # Single spaces inside lines, single blank line between blocks.
    assert "hello world" in text
    assert "\n\n\n" not in text


# ---------------------------------------------------------------------------
# _parse_gemini_json
# ---------------------------------------------------------------------------


def test_parse_gemini_json_with_fence() -> None:
    raw = (
        "Here is the JSON you asked for:\n"
        "```json\n"
        '{"reports": [{"first_name": "Reece", "last_name": "Walsh", '
        '"team_name": "Broncos", "status": "out", "weeks_out": 2, '
        '"body_part": "shoulder"}]}\n'
        "```\n"
    )
    rows = _parse_gemini_json(raw)
    assert len(rows) == 1
    assert rows[0]["first_name"] == "Reece"


def test_parse_gemini_json_without_fence() -> None:
    raw = (
        'Output:\n{"reports": [{"first_name": "James", "last_name": "Tedesco", '
        '"team_name": "Roosters", "status": "test", "weeks_out": null, "body_part": "knee"}]}'
    )
    rows = _parse_gemini_json(raw)
    assert len(rows) == 1
    assert rows[0]["status"] == "test"


def test_parse_gemini_json_invalid_returns_empty() -> None:
    assert _parse_gemini_json("Sorry, I can't help.") == []
    assert _parse_gemini_json("{not really json}") == []


def test_parse_gemini_json_filters_non_dict_rows() -> None:
    raw = '{"reports": [{"first_name": "A", "last_name": "B"}, "garbage", null]}'
    rows = _parse_gemini_json(raw)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# PlayerIndex
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StubMatch:
    """Minimal MatchRow stand-in for build_player_index tests."""

    match_id: int
    start_time: datetime
    home: TeamRow
    away: TeamRow


def _player(player_id: int, first: str, last: str) -> PlayerRow:
    return PlayerRow(
        player_id=player_id,
        jersey_number=1,
        position="Halfback",
        first_name=first,
        last_name=last,
        is_on_field=True,
    )


def _team(*, team_id: int, name: str, nick: str, players: list[PlayerRow]) -> TeamRow:
    return TeamRow(team_id=team_id, name=name, nick_name=nick, score=None, players=players)


def test_player_index_resolves_full_name() -> None:
    match = _StubMatch(
        match_id=1,
        start_time=datetime(2026, 4, 25, 9, 0, tzinfo=UTC),
        home=_team(
            team_id=500001,
            name="Brisbane Broncos",
            nick="Broncos",
            players=[_player(1001, "Reece", "Walsh")],
        ),
        away=_team(
            team_id=500002,
            name="Sydney Roosters",
            nick="Roosters",
            players=[_player(2001, "James", "Tedesco")],
        ),
    )
    idx = build_player_index([match])
    assert idx.resolve(first_name="Reece", last_name="Walsh", team_name=None) == (
        1001,
        500001,
    )
    # Case + punctuation insensitive.
    assert idx.resolve(first_name="REECE", last_name="walsh", team_name=None) == (
        1001,
        500001,
    )


def test_player_index_last_name_fallback_when_team_known() -> None:
    match = _StubMatch(
        match_id=1,
        start_time=datetime(2026, 4, 25, 9, 0, tzinfo=UTC),
        home=_team(
            team_id=500001,
            name="Brisbane Broncos",
            nick="Broncos",
            players=[_player(1001, "Samuel", "Walker")],
        ),
        away=_team(
            team_id=500002,
            name="Sydney Roosters",
            nick="Roosters",
            players=[_player(2001, "Spencer", "Leniu")],
        ),
    )
    idx = build_player_index([match])
    # Article says "Sam Walker" — first name nickname differs but last+team match.
    resolved = idx.resolve(first_name="Sam", last_name="Walker", team_name="Broncos")
    assert resolved == (1001, 500001)


def test_player_index_returns_none_when_unknown() -> None:
    match = _StubMatch(
        match_id=1,
        start_time=datetime(2026, 4, 25, 9, 0, tzinfo=UTC),
        home=_team(team_id=1, name="A", nick="A", players=[_player(1, "Reece", "Walsh")]),
        away=_team(team_id=2, name="B", nick="B", players=[_player(2, "James", "Tedesco")]),
    )
    idx = build_player_index([match])
    assert idx.resolve(first_name="Made", last_name="Up", team_name=None) is None


def test_player_index_uses_most_recent_team_id() -> None:
    """A player who appears for multiple teams resolves to the latest."""
    earlier = _StubMatch(
        match_id=1,
        start_time=datetime(2025, 8, 1, 9, 0, tzinfo=UTC),
        home=_team(team_id=500001, name="Old", nick="Old", players=[_player(1, "X", "Y")]),
        away=_team(team_id=999, name="z", nick="z", players=[]),
    )
    later = _StubMatch(
        match_id=2,
        start_time=datetime(2026, 3, 1, 9, 0, tzinfo=UTC),
        home=_team(team_id=500002, name="New", nick="New", players=[_player(1, "X", "Y")]),
        away=_team(team_id=999, name="z", nick="z", players=[]),
    )
    idx = build_player_index([earlier, later])
    assert idx.resolve(first_name="X", last_name="Y", team_name=None) == (1, 500002)


# ---------------------------------------------------------------------------
# _materialise_reports
# ---------------------------------------------------------------------------


def _index_for(player_id: int = 1001, team_id: int = 500001) -> PlayerIndex:
    return PlayerIndex(
        by_full_name={("reece", "walsh"): (player_id, team_id)},
        team_id_by_name={"broncos": team_id},
    )


def test_materialise_normalises_status_alias() -> None:
    rows = [
        {
            "first_name": "Reece",
            "last_name": "Walsh",
            "team_name": "Broncos",
            "status": "ruled out",  # alias for OUT
            "weeks_out": 2,
            "body_part": "shoulder",
        }
    ]
    out = _materialise_reports(
        rows,
        season=2026,
        round=8,
        scraped_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        player_index=_index_for(),
        source="nrl.com",
    )
    assert len(out) == 1
    assert out[0].status == InjuryStatus.OUT
    assert out[0].player_id == 1001
    assert out[0].team_id == 500001
    assert out[0].weeks_out == 2


def test_materialise_skips_unresolved_player() -> None:
    rows = [{"first_name": "No", "last_name": "Body", "status": "out"}]
    out = _materialise_reports(
        rows,
        season=2026,
        round=8,
        scraped_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        player_index=_index_for(),
        source="nrl.com",
    )
    assert out == []


def test_materialise_skips_unknown_status() -> None:
    rows = [
        {"first_name": "Reece", "last_name": "Walsh", "status": "blue-flu"},
    ]
    out = _materialise_reports(
        rows,
        season=2026,
        round=8,
        scraped_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        player_index=_index_for(),
        source="nrl.com",
    )
    assert out == []


def test_materialise_handles_null_weeks_out_and_body_part() -> None:
    rows = [
        {
            "first_name": "Reece",
            "last_name": "Walsh",
            "team_name": "Broncos",
            "status": "test",
            "weeks_out": None,
            "body_part": None,
        }
    ]
    out = _materialise_reports(
        rows,
        season=2026,
        round=8,
        scraped_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        player_index=_index_for(),
        source="nrl.com",
    )
    assert len(out) == 1
    assert out[0].weeks_out is None
    assert out[0].body_part is None


def test_materialise_drops_negative_weeks_out() -> None:
    rows = [
        {
            "first_name": "Reece",
            "last_name": "Walsh",
            "team_name": "Broncos",
            "status": "out",
            "weeks_out": -3,
        }
    ]
    out = _materialise_reports(
        rows,
        season=2026,
        round=8,
        scraped_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        player_index=_index_for(),
        source="nrl.com",
    )
    assert len(out) == 1
    # Negative weeks_out gets coerced to None rather than blowing up the dataclass invariant.
    assert out[0].weeks_out is None


# ---------------------------------------------------------------------------
# scrape_injury_list — end-to-end with stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubSource:
    text: str
    fetched_urls: list[str] = field(default_factory=list)

    def fetch(self, url: str) -> str:
        self.fetched_urls.append(url)
        return self.text


@dataclass
class _StubGeminiResponse:
    text: str


@dataclass
class _StubGemini:
    text: str
    captured: dict = field(default_factory=dict)

    def generate(self, prompt: str, *, system: str, max_output_tokens: int):
        self.captured = {"prompt": prompt, "system": system, "max": max_output_tokens}
        return _StubGeminiResponse(text=self.text)


def test_scrape_injury_list_end_to_end() -> None:
    article = "Round 8 injuries — Reece Walsh: OUT, shoulder, 2 weeks."
    gemini_json = (
        '{"reports": [{"first_name": "Reece", "last_name": "Walsh", '
        '"team_name": "Broncos", "status": "out", "weeks_out": 2, '
        '"body_part": "shoulder"}]}'
    )
    source = _StubSource(text=article)
    gemini = _StubGemini(text=gemini_json)
    parser = GeminiInjuryListParser(client=gemini)
    idx = _index_for()

    reports = scrape_injury_list(
        url="https://www.nrl.com/news/2026/04/21/round-8-injury-list/",
        season=2026,
        round=8,
        source=source,
        parser=parser,
        player_index=idx,
        now=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
    )

    assert len(reports) == 1
    assert reports[0].player_id == 1001
    assert reports[0].status == InjuryStatus.OUT
    # The fetcher was called with the URL we passed, and the system
    # prompt was the strict-schema one.
    assert source.fetched_urls == ["https://www.nrl.com/news/2026/04/21/round-8-injury-list/"]
    assert gemini.captured["system"] == GEMINI_SYSTEM_PROMPT
    assert "Round 8 injuries" in gemini.captured["prompt"]


def test_scrape_injury_list_returns_empty_on_blank_body() -> None:
    source = _StubSource(text="")
    gemini = _StubGemini(text='{"reports":[]}')
    parser = GeminiInjuryListParser(client=gemini)
    idx = _index_for()

    reports = scrape_injury_list(
        url="https://example.com",
        season=2026,
        round=8,
        source=source,
        parser=parser,
        player_index=idx,
    )
    assert reports == []
    # Gemini wasn't called when the source returned nothing.
    assert gemini.captured == {}


def test_scrape_injury_list_tolerates_gemini_garbage() -> None:
    """If Gemini drops a non-JSON response we degrade to an empty list,
    not a crash — the weekly scrape job retries next round."""
    source = _StubSource(text="Round 8 injuries — Reece Walsh: OUT")
    gemini = _StubGemini(text="I cannot parse this article today.")
    parser = GeminiInjuryListParser(client=gemini)
    idx = _index_for()
    reports = scrape_injury_list(
        url="https://example.com",
        season=2026,
        round=8,
        source=source,
        parser=parser,
        player_index=idx,
    )
    assert reports == []


# ---------------------------------------------------------------------------
# discover_injury_list_url (#268)
#
# The articles we want are NRL's weekly "nrl-late-mail-round-N" posts —
# NRL.com has never used "injury-list" as a URL slug.
# ---------------------------------------------------------------------------


def _index_html_with(*paths: str) -> str:
    """Build a fake news-index HTML page containing the given hrefs."""
    anchors = "\n".join(f'<a class="card" href="{p}">Story</a>' for p in paths)
    return f"<html><body>{anchors}</body></html>"


def test_discover_returns_url_for_matching_round() -> None:
    paths = [
        "/news/2026/03/04/nrl-late-mail-round-2-team-changes/",
        "/news/2026/03/11/nrl-late-mail-round-3-injuries/",
        "/news/2026/03/18/nrl-late-mail-round-4/",
    ]
    html = _index_html_with(*paths)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    url = discover_injury_list_url(2026, 3, client=client)
    assert url == "https://www.nrl.com/news/2026/03/11/nrl-late-mail-round-3-injuries/"


def test_discover_returns_none_when_round_missing() -> None:
    html = _index_html_with(
        "/news/2026/03/04/nrl-late-mail-round-2/",
        "/news/2026/03/18/nrl-late-mail-round-4/",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert discover_injury_list_url(2026, 3, client=client) is None


def test_discover_prefers_longest_url_for_corrected_articles() -> None:
    """When NRL.com publishes a corrected article, the original article
    sometimes stays in the index with a shorter slug. Prefer the longer
    (more-specific) URL."""
    html = _index_html_with(
        "/news/2026/03/11/nrl-late-mail-round-3/",
        "/news/2026/03/11/nrl-late-mail-round-3-update-with-injuries/",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    url = discover_injury_list_url(2026, 3, client=client)
    assert url == "https://www.nrl.com/news/2026/03/11/nrl-late-mail-round-3-update-with-injuries/"


def test_discover_ignores_non_matching_season() -> None:
    html = _index_html_with(
        "/news/2025/03/11/nrl-late-mail-round-3/",
        "/news/2026/03/11/nrl-late-mail-round-3/",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    url = discover_injury_list_url(2026, 3, client=client)
    assert url == "https://www.nrl.com/news/2026/03/11/nrl-late-mail-round-3/"


def test_discover_ignores_team_list_articles_for_same_round() -> None:
    """Team-list articles share the round-N suffix but aren't late-mail —
    they shouldn't match the discovery regex."""
    html = _index_html_with(
        "/news/2026/03/11/nrl-team-lists-round-3/",
        "/news/2026/03/11/nrl-late-mail-round-3/",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    url = discover_injury_list_url(2026, 3, client=client)
    assert url == "https://www.nrl.com/news/2026/03/11/nrl-late-mail-round-3/"
