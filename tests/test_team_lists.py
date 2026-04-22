"""Tests for the team-list snapshot data model + storage."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fantasy_coach.features import PlayerRow
from fantasy_coach.storage.sqlite import SQLiteRepository
from fantasy_coach.storage.team_list import (
    FirestoreTeamListRepository,
    SQLiteTeamListRepository,
    _dict_to_player,
    _player_to_dict,
)
from fantasy_coach.team_lists import (
    TeamListSnapshot,
    compute_team_list_changes,
)


def _player(
    player_id: int,
    *,
    jersey: int,
    position: str = "Fullback",
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
    team_id: int = 10,
    scraped_at: datetime | None = None,
    players: tuple[PlayerRow, ...] = (),
) -> TeamListSnapshot:
    return TeamListSnapshot(
        season=2026,
        round=8,
        match_id=match_id,
        team_id=team_id,
        scraped_at=scraped_at or datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        players=players,
    )


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_snapshot_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        TeamListSnapshot(
            season=2026,
            round=8,
            match_id=1,
            team_id=10,
            scraped_at=datetime(2026, 4, 21, 9, 0),  # no tz
            players=(),
        )


# ---------------------------------------------------------------------------
# compute_team_list_changes
# ---------------------------------------------------------------------------


def test_change_detects_none_when_identical() -> None:
    players = (_player(1, jersey=1), _player(2, jersey=2))
    earlier = _snapshot(players=players)
    later = _snapshot(
        players=players,
        scraped_at=earlier.scraped_at + timedelta(days=2),
    )
    change = compute_team_list_changes(earlier, later)
    assert change.dropped == ()
    assert change.added == ()
    assert change.has_changes is False


def test_change_detects_single_swap() -> None:
    earlier = _snapshot(
        players=(_player(1, jersey=7, is_on_field=True), _player(2, jersey=14, is_on_field=False)),
    )
    later = _snapshot(
        players=(_player(1, jersey=14, is_on_field=False), _player(2, jersey=7, is_on_field=True)),
        scraped_at=earlier.scraped_at + timedelta(days=2),
    )
    change = compute_team_list_changes(earlier, later)
    # Player 1 was starting, now on bench → dropped.
    # Player 2 was on bench, now starting → added.
    assert [p.player_id for p in change.dropped] == [1]
    assert [p.player_id for p in change.added] == [2]
    assert change.has_changes is True


def test_change_ignores_players_with_none_is_on_field() -> None:
    # A pre-team-list-drop scrape won't carry is_on_field at all.
    earlier = _snapshot(players=(_player(1, jersey=1, is_on_field=None),))
    later = _snapshot(
        players=(_player(1, jersey=1, is_on_field=True),),
        scraped_at=earlier.scraped_at + timedelta(days=2),
    )
    # Starting sets: earlier={} (None filtered), later={1}. → player 1 added.
    change = compute_team_list_changes(earlier, later)
    assert [p.player_id for p in change.added] == [1]
    assert change.dropped == ()


def test_change_rejects_mismatched_match_team() -> None:
    a = _snapshot(match_id=1, team_id=10)
    b = _snapshot(match_id=2, team_id=10)
    with pytest.raises(ValueError):
        compute_team_list_changes(a, b)


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_conn(tmp_path: Path) -> sqlite3.Connection:
    # Use the main SQLiteRepository to initialise the schema (creates the
    # team_list_snapshots table via schema.sql + bumps schema_version).
    repo = SQLiteRepository(tmp_path / "tl.db")
    return repo._conn  # noqa: SLF001 — repository owns the connection


def test_sqlite_record_and_list(sqlite_conn: sqlite3.Connection) -> None:
    repo = SQLiteTeamListRepository(sqlite_conn)
    first = _snapshot(
        scraped_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        players=(_player(1, jersey=1), _player(2, jersey=14, is_on_field=False)),
    )
    second = _snapshot(
        scraped_at=datetime(2026, 4, 23, 6, 0, tzinfo=UTC),
        players=(_player(1, jersey=1), _player(2, jersey=7, is_on_field=True)),
    )
    repo.record_snapshot(first)
    repo.record_snapshot(second)

    snapshots = repo.list_snapshots(match_id=1000)
    assert len(snapshots) == 2
    # Ordered by scraped_at ASC.
    assert snapshots[0].scraped_at < snapshots[1].scraped_at
    assert snapshots[0].players[1].is_on_field is False
    assert snapshots[1].players[1].is_on_field is True


def test_sqlite_list_filters_by_team(sqlite_conn: sqlite3.Connection) -> None:
    repo = SQLiteTeamListRepository(sqlite_conn)
    home = _snapshot(team_id=10, players=(_player(1, jersey=1),))
    away = _snapshot(team_id=20, players=(_player(2, jersey=1),))
    repo.record_snapshot(home)
    repo.record_snapshot(away)

    home_only = repo.list_snapshots(match_id=1000, team_id=10)
    assert len(home_only) == 1
    assert home_only[0].team_id == 10


# ---------------------------------------------------------------------------
# Firestore storage (with a fake client)
# ---------------------------------------------------------------------------


class _FakeFirestoreCollection:
    def __init__(self) -> None:
        self._docs: list[dict] = []
        self._filters: list[tuple[str, str, object]] = []
        self._order_by: str | None = None

    def add(self, data: dict) -> None:
        self._docs.append(data)

    def where(self, field: str, op: str, value: object) -> _FakeFirestoreCollection:
        # Return a child query with this filter added; share the docs list.
        child = _FakeFirestoreCollection()
        child._docs = self._docs
        child._filters = [*self._filters, (field, op, value)]
        child._order_by = self._order_by
        return child

    def order_by(self, field: str) -> _FakeFirestoreCollection:
        child = _FakeFirestoreCollection()
        child._docs = self._docs
        child._filters = self._filters
        child._order_by = field
        return child

    def stream(self):
        filtered = [
            d for d in self._docs if all(_apply_op(d.get(f), op, v) for f, op, v in self._filters)
        ]
        if self._order_by is not None:
            filtered.sort(key=lambda d: d.get(self._order_by))  # type: ignore[arg-type]

        class _Doc:
            def __init__(self, d: dict) -> None:
                self._d = d

            def to_dict(self) -> dict:
                return self._d

        return [_Doc(d) for d in filtered]


def _apply_op(left: object, op: str, right: object) -> bool:
    if op == "==":
        return left == right
    raise NotImplementedError(op)


class _FakeFirestoreClient:
    def __init__(self) -> None:
        self._collection = _FakeFirestoreCollection()

    def collection(self, name: str) -> _FakeFirestoreCollection:
        assert name == "team_list_snapshots"
        return self._collection


def test_firestore_record_and_list() -> None:
    repo = FirestoreTeamListRepository(client=_FakeFirestoreClient())
    a = _snapshot(
        scraped_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        players=(_player(1, jersey=1),),
    )
    b = _snapshot(
        scraped_at=datetime(2026, 4, 23, 6, 0, tzinfo=UTC),
        players=(_player(1, jersey=1, is_on_field=False),),
    )
    repo.record_snapshot(a)
    repo.record_snapshot(b)

    loaded = repo.list_snapshots(match_id=1000)
    assert [s.scraped_at for s in loaded] == [a.scraped_at, b.scraped_at]
    assert loaded[1].players[0].is_on_field is False


# ---------------------------------------------------------------------------
# Shared serialisation helpers
# ---------------------------------------------------------------------------


def test_player_roundtrip_preserves_is_on_field() -> None:
    original = _player(1, jersey=1, is_on_field=False)
    assert _dict_to_player(_player_to_dict(original)) == original


def test_player_roundtrip_handles_none() -> None:
    original = _player(1, jersey=1, is_on_field=None)
    assert _dict_to_player(_player_to_dict(original)) == original
