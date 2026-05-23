"""Tests for the late team-list watcher (#208 PR B)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fantasy_coach.features import PlayerRow
from fantasy_coach.injury import LateTeamChange
from fantasy_coach.late_change_watcher import (
    compute_late_changes,
    find_matches_in_window,
    select_reference_snapshot,
    watch_match,
)
from fantasy_coach.storage.injury import SQLiteLateTeamChangeRepository
from fantasy_coach.storage.sqlite import SQLiteRepository
from fantasy_coach.storage.team_list import SQLiteTeamListRepository
from fantasy_coach.team_lists import TeamListSnapshot


def _player(
    player_id: int,
    *,
    jersey: int,
    position: str = "Halfback",
    is_on_field: bool | None = True,
) -> PlayerRow:
    return PlayerRow(
        player_id=player_id,
        jersey_number=jersey,
        position=position,
        first_name=f"First{player_id}",
        last_name=f"Last{player_id}",
        is_on_field=is_on_field,
    )


def _snapshot(
    *,
    match_id: int = 1000,
    team_id: int = 500001,
    scraped_at: datetime,
    players: tuple[PlayerRow, ...],
) -> TeamListSnapshot:
    return TeamListSnapshot(
        season=2026,
        round=8,
        match_id=match_id,
        team_id=team_id,
        scraped_at=scraped_at,
        players=players,
    )


# ---------------------------------------------------------------------------
# compute_late_changes
# ---------------------------------------------------------------------------


def test_compute_returns_empty_when_identical() -> None:
    players = (_player(1, jersey=7), _player(2, jersey=14, is_on_field=False))
    early = _snapshot(scraped_at=datetime(2026, 4, 25, 6, 0, tzinfo=UTC), players=players)
    late = _snapshot(scraped_at=datetime(2026, 4, 25, 9, 30, tzinfo=UTC), players=players)
    assert compute_late_changes(earlier=early, later=late) == []


def test_compute_emits_withdrawn_and_added_for_swap() -> None:
    early = _snapshot(
        scraped_at=datetime(2026, 4, 25, 6, 0, tzinfo=UTC),
        players=(
            _player(1, jersey=7, is_on_field=True),
            _player(2, jersey=14, is_on_field=False),
        ),
    )
    late = _snapshot(
        scraped_at=datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
        players=(
            _player(1, jersey=14, is_on_field=False),
            _player(2, jersey=7, is_on_field=True),
        ),
    )
    rows = compute_late_changes(earlier=early, later=late)
    assert len(rows) == 2
    by_kind = {r.change_type: r for r in rows}
    assert by_kind["withdrawn"].player_id == 1
    assert by_kind["withdrawn"].was_starting is True
    assert by_kind["withdrawn"].is_starting is False
    assert by_kind["added"].player_id == 2
    assert by_kind["added"].was_starting is False
    assert by_kind["added"].is_starting is True
    # detected_at is the LATER snapshot's scraped_at.
    assert by_kind["withdrawn"].detected_at == late.scraped_at


def test_compute_emits_position_change_when_starter_moves() -> None:
    early = _snapshot(
        scraped_at=datetime(2026, 4, 25, 6, 0, tzinfo=UTC),
        players=(_player(1, jersey=7, position="Halfback"),),
    )
    late = _snapshot(
        scraped_at=datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
        players=(_player(1, jersey=6, position="Five-Eighth"),),
    )
    rows = compute_late_changes(earlier=early, later=late)
    assert len(rows) == 1
    assert rows[0].change_type == "position_change"
    assert rows[0].position == "Five-Eighth"  # the *new* position


def test_compute_skips_players_with_none_is_on_field() -> None:
    # A pre-team-list-drop snapshot has is_on_field=None for everyone — we
    # can't derive a starter set so the player is invisible to the diff.
    early = _snapshot(
        scraped_at=datetime(2026, 4, 25, 6, 0, tzinfo=UTC),
        players=(_player(1, jersey=7, is_on_field=None),),
    )
    late = _snapshot(
        scraped_at=datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
        players=(_player(1, jersey=7, is_on_field=True),),
    )
    assert compute_late_changes(earlier=early, later=late) == []


def test_compute_rejects_mismatched_match_or_team() -> None:
    a = _snapshot(
        match_id=1, team_id=10, scraped_at=datetime(2026, 4, 25, 6, 0, tzinfo=UTC), players=()
    )
    b = _snapshot(
        match_id=2, team_id=10, scraped_at=datetime(2026, 4, 25, 9, 0, tzinfo=UTC), players=()
    )
    with pytest.raises(ValueError):
        compute_late_changes(earlier=a, later=b)


# ---------------------------------------------------------------------------
# select_reference_snapshot
# ---------------------------------------------------------------------------


def test_select_reference_picks_latest_before_cutoff() -> None:
    kickoff = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    candidates = [
        _snapshot(scraped_at=kickoff - timedelta(days=4), players=()),  # Tuesday
        _snapshot(scraped_at=kickoff - timedelta(days=2), players=()),  # Thursday
        _snapshot(scraped_at=kickoff - timedelta(minutes=30), players=()),  # inside window
    ]
    chosen = select_reference_snapshot(
        candidates, kickoff=kickoff, cutoff_minutes_before_kickoff=60
    )
    assert chosen is not None
    # Latest snapshot at LEAST 60 min before kickoff = the Thursday one.
    assert chosen.scraped_at == kickoff - timedelta(days=2)


def test_select_reference_returns_none_when_all_inside_window() -> None:
    kickoff = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    candidates = [
        _snapshot(scraped_at=kickoff - timedelta(minutes=45), players=()),
        _snapshot(scraped_at=kickoff - timedelta(minutes=15), players=()),
    ]
    assert (
        select_reference_snapshot(candidates, kickoff=kickoff, cutoff_minutes_before_kickoff=60)
        is None
    )


def test_select_reference_returns_none_when_no_candidates() -> None:
    assert select_reference_snapshot([], kickoff=datetime(2026, 4, 25, 10, 0, tzinfo=UTC)) is None


# ---------------------------------------------------------------------------
# find_matches_in_window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StubMatch:
    match_id: int
    start_time: datetime


def test_find_matches_in_window_filters_by_kickoff() -> None:
    now = datetime(2026, 4, 25, 9, 0, tzinfo=UTC)
    matches = [
        _StubMatch(1, now + timedelta(minutes=30)),  # in window
        _StubMatch(2, now + timedelta(minutes=89)),  # in window
        _StubMatch(3, now + timedelta(minutes=120)),  # too far away
        _StubMatch(4, now - timedelta(minutes=10)),  # already kicked off
    ]
    chosen = find_matches_in_window(matches=matches, now=now, lead_minutes=90)
    assert [m.match_id for m in chosen] == [1, 2]


# ---------------------------------------------------------------------------
# watch_match — end-to-end with fake fetcher + real SQLite repos
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_conn(tmp_path: Path) -> sqlite3.Connection:
    repo = SQLiteRepository(tmp_path / "watch.db")
    return repo._conn  # noqa: SLF001 — share schema-bootstrapped conn


@dataclass(frozen=True)
class _StubTeam:
    team_id: int


@dataclass(frozen=True)
class _StubMatchRow:
    match_id: int
    season: int
    round: int
    start_time: datetime
    home: _StubTeam
    away: _StubTeam


def _make_payload(
    *,
    home_team_id: int,
    away_team_id: int,
    home_players: list[dict[str, Any]],
    away_players: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "homeTeam": {"teamId": home_team_id, "players": home_players},
        "awayTeam": {"teamId": away_team_id, "players": away_players},
    }


def _player_payload(player_id: int, *, position: str, is_on_field: bool, jersey: int) -> dict:
    return {
        "playerId": player_id,
        "position": position,
        "isOnField": is_on_field,
        "number": jersey,
        "firstName": f"First{player_id}",
        "lastName": f"Last{player_id}",
    }


def test_watch_match_writes_late_change_when_starter_dropped(
    sqlite_conn: sqlite3.Connection,
) -> None:
    team_list_repo = SQLiteTeamListRepository(sqlite_conn)
    late_repo = SQLiteLateTeamChangeRepository(sqlite_conn)

    kickoff = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    match = _StubMatchRow(
        match_id=1000,
        season=2026,
        round=8,
        start_time=kickoff,
        home=_StubTeam(team_id=500001),
        away=_StubTeam(team_id=500002),
    )

    # Pre-window reference snapshot (named team list dropped Thursday).
    earlier = TeamListSnapshot(
        season=2026,
        round=8,
        match_id=1000,
        team_id=500001,
        scraped_at=kickoff - timedelta(days=2),
        players=(
            _player(1, jersey=7, position="Halfback", is_on_field=True),
            _player(2, jersey=14, position="Bench", is_on_field=False),
        ),
    )
    team_list_repo.record_snapshot(earlier)
    # Away team has its named-team-list snapshot but no late changes
    # — still recorded so list_snapshots(team_id=away) is non-empty.
    earlier_away = TeamListSnapshot(
        season=2026,
        round=8,
        match_id=1000,
        team_id=500002,
        scraped_at=kickoff - timedelta(days=2),
        players=(_player(10, jersey=7, position="Halfback", is_on_field=True),),
    )
    team_list_repo.record_snapshot(earlier_away)

    # Watcher payload — player 1 dropped from starting XIII inside kickoff hour.
    fresh = _make_payload(
        home_team_id=500001,
        away_team_id=500002,
        home_players=[
            _player_payload(1, position="Bench", is_on_field=False, jersey=14),
            _player_payload(2, position="Halfback", is_on_field=True, jersey=7),
        ],
        away_players=[
            _player_payload(10, position="Halfback", is_on_field=True, jersey=7),
        ],
    )

    written = watch_match(
        match=match,
        fetch_match=lambda _m: fresh,
        team_list_repo=team_list_repo,
        late_repo=late_repo,
        now=kickoff - timedelta(minutes=30),
        cutoff_minutes_before_kickoff=60,
    )

    # One swap on the home team: player 1 dropped, player 2 promoted.
    kinds = sorted(c.change_type for c in written)
    assert kinds == ["added", "withdrawn"]
    home_rows = late_repo.list_changes(match_id=1000, team_id=500001)
    assert len(home_rows) == 2
    away_rows = late_repo.list_changes(match_id=1000, team_id=500002)
    assert away_rows == []


def test_watch_match_records_snapshot_even_with_no_diff(
    sqlite_conn: sqlite3.Connection,
) -> None:
    team_list_repo = SQLiteTeamListRepository(sqlite_conn)
    late_repo = SQLiteLateTeamChangeRepository(sqlite_conn)

    kickoff = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    match = _StubMatchRow(
        match_id=2000,
        season=2026,
        round=8,
        start_time=kickoff,
        home=_StubTeam(team_id=600001),
        away=_StubTeam(team_id=600002),
    )

    earlier = TeamListSnapshot(
        season=2026,
        round=8,
        match_id=2000,
        team_id=600001,
        scraped_at=kickoff - timedelta(days=2),
        players=(_player(1, jersey=7, position="Halfback", is_on_field=True),),
    )
    team_list_repo.record_snapshot(earlier)

    fresh = _make_payload(
        home_team_id=600001,
        away_team_id=600002,
        home_players=[_player_payload(1, position="Halfback", is_on_field=True, jersey=7)],
        away_players=[],
    )

    now = kickoff - timedelta(minutes=30)
    written = watch_match(
        match=match,
        fetch_match=lambda _m: fresh,
        team_list_repo=team_list_repo,
        late_repo=late_repo,
        now=now,
        cutoff_minutes_before_kickoff=60,
    )
    assert written == []
    # The fresh snapshot should have been recorded for both teams.
    home = team_list_repo.list_snapshots(match_id=2000, team_id=600001)
    away = team_list_repo.list_snapshots(match_id=2000, team_id=600002)
    assert any(s.scraped_at == now for s in home)
    assert any(s.scraped_at == now for s in away)


def test_watch_match_skips_when_fetcher_returns_none(
    sqlite_conn: sqlite3.Connection,
) -> None:
    team_list_repo = SQLiteTeamListRepository(sqlite_conn)
    late_repo = SQLiteLateTeamChangeRepository(sqlite_conn)

    match = _StubMatchRow(
        match_id=3000,
        season=2026,
        round=8,
        start_time=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        home=_StubTeam(team_id=700001),
        away=_StubTeam(team_id=700002),
    )

    written = watch_match(
        match=match,
        fetch_match=lambda _m: None,  # cache hit / 304 / 404
        team_list_repo=team_list_repo,
        late_repo=late_repo,
        now=datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
    )
    assert written == []
    # Nothing should have been recorded on either side.
    assert team_list_repo.list_snapshots(match_id=3000) == []


def test_watch_match_no_reference_writes_snapshot_only(
    sqlite_conn: sqlite3.Connection,
) -> None:
    """When there's no pre-window snapshot we still record the fresh one
    but emit no LateTeamChange rows — there's nothing to diff against."""
    team_list_repo = SQLiteTeamListRepository(sqlite_conn)
    late_repo = SQLiteLateTeamChangeRepository(sqlite_conn)

    kickoff = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    match = _StubMatchRow(
        match_id=4000,
        season=2026,
        round=8,
        start_time=kickoff,
        home=_StubTeam(team_id=800001),
        away=_StubTeam(team_id=800002),
    )

    fresh = _make_payload(
        home_team_id=800001,
        away_team_id=800002,
        home_players=[_player_payload(1, position="Halfback", is_on_field=True, jersey=7)],
        away_players=[_player_payload(2, position="Halfback", is_on_field=True, jersey=7)],
    )
    written: list[LateTeamChange] = watch_match(
        match=match,
        fetch_match=lambda _m: fresh,
        team_list_repo=team_list_repo,
        late_repo=late_repo,
        now=kickoff - timedelta(minutes=30),
    )
    assert written == []
    assert team_list_repo.list_snapshots(match_id=4000) != []
    assert late_repo.list_changes(match_id=4000) == []
