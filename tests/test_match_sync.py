"""Tests for :mod:`fantasy_coach.match_sync`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.match_sync import refresh_stale_matches


def _match(
    *,
    match_id: int,
    season: int = 2026,
    round: int = 8,
    when: datetime,
    state: str = "Upcoming",
    home_score: int | None = None,
    away_score: int | None = None,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round,
        start_time=when,
        match_state=state,
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=1,
            name=f"home-{match_id}",
            nick_name="H",
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=2,
            name=f"away-{match_id}",
            nick_name="A",
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


class _FakeRepo:
    def __init__(self, matches: list[MatchRow]) -> None:
        self._rows: dict[int, MatchRow] = {m.match_id: m for m in matches}
        self.upserts: list[MatchRow] = []

    def list_matches(self, season: int, round: int | None = None) -> list[MatchRow]:
        return [
            m
            for m in self._rows.values()
            if m.season == season and (round is None or m.round == round)
        ]

    def upsert_match(self, row: MatchRow) -> None:
        self._rows[row.match_id] = row
        self.upserts.append(row)

    def get_match(self, match_id: int) -> MatchRow | None:  # pragma: no cover
        return self._rows.get(match_id)


def _raw_match_payload(match_id: int, state: str, h_score: int, a_score: int) -> dict:
    """Minimal dict that ``extract_match_features`` will accept."""
    return {
        "matchId": str(match_id),
        "matchState": state,
        "startTime": "2026-04-23T09:50:00Z",
        "roundNumber": 8,
        "homeTeam": {
            "teamId": 1,
            "name": f"home-{match_id}",
            "nickName": "H",
            "score": h_score,
            "players": [],
        },
        "awayTeam": {
            "teamId": 2,
            "name": f"away-{match_id}",
            "nickName": "A",
            "score": a_score,
            "players": [],
        },
    }


def test_refresh_stale_matches_upserts_past_start_upcoming_matches():
    past = datetime(2026, 4, 23, 9, 50, tzinfo=UTC)
    now = datetime(2026, 4, 24, 2, 0, tzinfo=UTC)
    repo = _FakeRepo([_match(match_id=1, when=past, state="Upcoming")])

    round_payload = {
        "fixtures": [
            {"matchCentreUrl": "/draw/match-1/"},
        ]
    }
    per_match = {"/draw/match-1/": _raw_match_payload(1, "FullTime", 33, 14)}

    updated = refresh_stale_matches(
        repo,
        season=2026,
        now=now,
        fetch_round_fn=lambda s, r: round_payload,
        fetch_match_fn=lambda url: per_match.get(url),
    )
    assert updated == 1
    assert repo._rows[1].match_state == "FullTime"
    assert repo._rows[1].home.score == 33
    assert repo._rows[1].away.score == 14


def test_refresh_stale_matches_skips_future_matches():
    future = datetime(2026, 5, 1, 9, 50, tzinfo=UTC)
    now = datetime(2026, 4, 24, 2, 0, tzinfo=UTC)
    repo = _FakeRepo([_match(match_id=1, when=future, state="Upcoming")])
    calls: list[tuple] = []

    updated = refresh_stale_matches(
        repo,
        season=2026,
        now=now,
        fetch_round_fn=lambda s, r: (calls.append((s, r)), {"fixtures": []})[1],
        fetch_match_fn=lambda url: None,
    )
    assert updated == 0
    assert calls == []  # fetch_round never invoked — nothing to refresh


def test_refresh_stale_matches_skips_fulltime_matches():
    past = datetime(2026, 4, 23, 9, 50, tzinfo=UTC)
    now = datetime(2026, 4, 24, 2, 0, tzinfo=UTC)
    repo = _FakeRepo(
        [_match(match_id=1, when=past, state="FullTime", home_score=33, away_score=14)]
    )
    calls: list[tuple] = []

    updated = refresh_stale_matches(
        repo,
        season=2026,
        now=now,
        fetch_round_fn=lambda s, r: (calls.append((s, r)), {"fixtures": []})[1],
        fetch_match_fn=lambda url: None,
    )
    assert updated == 0
    assert calls == []


def test_refresh_stale_matches_survives_fetch_round_failure():
    past = datetime(2026, 4, 23, 9, 50, tzinfo=UTC)
    now = datetime(2026, 4, 24, 2, 0, tzinfo=UTC)
    repo = _FakeRepo([_match(match_id=1, when=past, state="Upcoming")])

    def boom(season, round_):
        raise RuntimeError("nrl.com down")

    updated = refresh_stale_matches(
        repo,
        season=2026,
        now=now,
        fetch_round_fn=boom,
        fetch_match_fn=lambda url: None,
    )
    assert updated == 0
    assert repo._rows[1].match_state == "Upcoming"  # unchanged


def test_refresh_stale_matches_survives_fetch_match_failure():
    past = datetime(2026, 4, 23, 9, 50, tzinfo=UTC)
    now = datetime(2026, 4, 24, 2, 0, tzinfo=UTC)
    repo = _FakeRepo([_match(match_id=1, when=past, state="Upcoming")])

    def boom(url):
        raise RuntimeError("timeout")

    updated = refresh_stale_matches(
        repo,
        season=2026,
        now=now,
        fetch_round_fn=lambda s, r: {"fixtures": [{"matchCentreUrl": "/draw/match-1/"}]},
        fetch_match_fn=boom,
    )
    assert updated == 0
    assert repo._rows[1].match_state == "Upcoming"


def test_refresh_stale_matches_only_touches_stale_ids_in_returned_round():
    # Round has two fixtures; only one is stale in repo. Second fixture's
    # details are never fetched — we short-circuit the "not stale" case.
    past = datetime(2026, 4, 23, 9, 50, tzinfo=UTC)
    future = datetime(2026, 4, 30, 9, 50, tzinfo=UTC)
    now = datetime(2026, 4, 24, 2, 0, tzinfo=UTC)
    repo = _FakeRepo(
        [
            _match(match_id=1, when=past, state="Upcoming"),
            _match(match_id=2, when=future, state="Upcoming"),
        ]
    )

    fetched_urls: list[str] = []

    def fetch_match(url):
        fetched_urls.append(url)
        if url == "/draw/match-1/":
            return _raw_match_payload(1, "FullTime", 33, 14)
        return _raw_match_payload(2, "Upcoming", 0, 0)

    updated = refresh_stale_matches(
        repo,
        season=2026,
        now=now,
        fetch_round_fn=lambda s, r: {
            "fixtures": [
                {"matchCentreUrl": "/draw/match-1/"},
                {"matchCentreUrl": "/draw/match-2/"},
            ]
        },
        fetch_match_fn=fetch_match,
    )
    # Fetches both (only API-level call is round-level), but only
    # upserts the stale one.
    assert updated == 1
    assert repo._rows[1].match_state == "FullTime"
    assert repo._rows[2].match_state == "Upcoming"


def test_refresh_stale_matches_caps_at_max_rounds_back():
    past = datetime(2026, 4, 1, tzinfo=UTC)
    now = datetime(2026, 4, 24, tzinfo=UTC)
    # 10 stale rounds; max_rounds_back=4 → only last 4 are re-fetched.
    matches = [
        _match(match_id=i, round=i, when=past + timedelta(days=i), state="Upcoming")
        for i in range(1, 11)
    ]
    repo = _FakeRepo(matches)

    touched_rounds: list[int] = []

    def fetch_round(season, r):
        touched_rounds.append(r)
        return {"fixtures": []}

    refresh_stale_matches(
        repo,
        season=2026,
        now=now,
        fetch_round_fn=fetch_round,
        fetch_match_fn=lambda url: None,
        max_rounds_back=4,
    )
    assert touched_rounds == [7, 8, 9, 10]
