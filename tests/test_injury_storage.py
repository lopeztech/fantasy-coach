"""Tests for the injury-report + late-team-change storage layer (#208)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fantasy_coach.injury import InjuryReport, InjuryStatus, LateTeamChange
from fantasy_coach.storage.injury import (
    FirestoreInjuryReportRepository,
    FirestoreLateTeamChangeRepository,
    SQLiteInjuryReportRepository,
    SQLiteLateTeamChangeRepository,
)
from fantasy_coach.storage.sqlite import SQLiteRepository


def _report(
    *,
    player_id: int = 4001,
    season: int = 2026,
    round: int = 8,
    team_id: int = 500001,
    status: InjuryStatus = InjuryStatus.OUT,
    weeks_out: int | None = 2,
    body_part: str | None = "shoulder",
    source: str = "nrl.com",
    scraped_at: datetime | None = None,
) -> InjuryReport:
    return InjuryReport(
        player_id=player_id,
        season=season,
        round=round,
        team_id=team_id,
        status=status,
        weeks_out=weeks_out,
        body_part=body_part,
        source=source,
        scraped_at=scraped_at or datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
    )


def _change(
    *,
    match_id: int = 1000,
    team_id: int = 500001,
    player_id: int = 4001,
    change_type: str = "withdrawn",
    position: str | None = "Halfback",
    was_starting: bool = True,
    is_starting: bool = False,
    detected_at: datetime | None = None,
) -> LateTeamChange:
    return LateTeamChange(
        match_id=match_id,
        team_id=team_id,
        player_id=player_id,
        change_type=change_type,
        position=position,
        was_starting=was_starting,
        is_starting=is_starting,
        detected_at=detected_at or datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_injury_report_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        InjuryReport(
            player_id=1,
            season=2026,
            round=1,
            team_id=1,
            status=InjuryStatus.OUT,
            weeks_out=None,
            body_part=None,
            source="x",
            scraped_at=datetime(2026, 4, 21, 9, 0),  # naive
        )


def test_injury_report_rejects_negative_weeks_out() -> None:
    with pytest.raises(ValueError):
        _report(weeks_out=-1)


def test_late_change_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        LateTeamChange(
            match_id=1,
            team_id=1,
            player_id=1,
            change_type="withdrawn",
            position=None,
            was_starting=True,
            is_starting=False,
            detected_at=datetime(2026, 4, 21, 9, 0),  # naive
        )


def test_late_change_rejects_unknown_change_type() -> None:
    with pytest.raises(ValueError):
        LateTeamChange(
            match_id=1,
            team_id=1,
            player_id=1,
            change_type="bench_swap",  # not in the allowed set
            position=None,
            was_starting=True,
            is_starting=False,
            detected_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------
# SQLite — shared connection fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_conn(tmp_path: Path) -> sqlite3.Connection:
    repo = SQLiteRepository(tmp_path / "injury.db")
    return repo._conn  # noqa: SLF001 — tests share the schema-bootstrapped conn


# ---------------------------------------------------------------------------
# SQLite — InjuryReport
# ---------------------------------------------------------------------------


def test_sqlite_injury_record_and_list(sqlite_conn: sqlite3.Connection) -> None:
    repo = SQLiteInjuryReportRepository(sqlite_conn)
    repo.record_report(_report(player_id=1, status=InjuryStatus.OUT, weeks_out=2))
    repo.record_report(_report(player_id=2, status=InjuryStatus.TEST, weeks_out=None))

    rows = repo.list_reports(season=2026, round=8)
    assert len(rows) == 2
    assert {r.player_id for r in rows} == {1, 2}


def test_sqlite_injury_list_filters_by_team(sqlite_conn: sqlite3.Connection) -> None:
    repo = SQLiteInjuryReportRepository(sqlite_conn)
    repo.record_report(_report(player_id=1, team_id=500001))
    repo.record_report(_report(player_id=2, team_id=500002))

    home_only = repo.list_reports(season=2026, round=8, team_id=500001)
    assert len(home_only) == 1
    assert home_only[0].team_id == 500001


def test_sqlite_injury_upsert_overwrites_same_pk(sqlite_conn: sqlite3.Connection) -> None:
    repo = SQLiteInjuryReportRepository(sqlite_conn)
    repo.record_report(_report(weeks_out=4, body_part="shoulder"))
    # Same PK (player_id, season, round, source) — should overwrite, not insert.
    repo.record_report(
        _report(
            weeks_out=1,
            body_part="shoulder (recovered)",
            scraped_at=datetime(2026, 4, 24, 9, 0, tzinfo=UTC),
        )
    )

    rows = repo.list_reports(season=2026, round=8)
    assert len(rows) == 1
    assert rows[0].weeks_out == 1
    assert rows[0].body_part == "shoulder (recovered)"


def test_sqlite_injury_distinct_sources_coexist(sqlite_conn: sqlite3.Connection) -> None:
    repo = SQLiteInjuryReportRepository(sqlite_conn)
    repo.record_report(_report(source="nrl.com", weeks_out=2))
    repo.record_report(_report(source="nrl_physio", weeks_out=3))

    rows = repo.list_reports(season=2026, round=8)
    assert len(rows) == 2
    assert {r.source for r in rows} == {"nrl.com", "nrl_physio"}


def test_sqlite_injury_check_constraint_rejects_unknown_status(
    sqlite_conn: sqlite3.Connection,
) -> None:
    # The dataclass blocks unknown statuses, so to exercise the SQL CHECK
    # constraint we have to drop down to raw SQL.
    with pytest.raises(sqlite3.IntegrityError):
        sqlite_conn.execute(
            """
            INSERT INTO injury_reports
                (player_id, season, round, team_id, status,
                 weeks_out, body_part, source, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, 2026, 1, 1, "doubtful", None, None, "x", "2026-04-21T09:00:00+00:00"),
        )


# ---------------------------------------------------------------------------
# SQLite — LateTeamChange
# ---------------------------------------------------------------------------


def test_sqlite_late_change_record_and_list(sqlite_conn: sqlite3.Connection) -> None:
    repo = SQLiteLateTeamChangeRepository(sqlite_conn)
    earlier = _change(detected_at=datetime(2026, 4, 25, 9, 0, tzinfo=UTC))
    later = _change(
        player_id=4002,
        detected_at=datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
    )
    repo.record_change(earlier)
    repo.record_change(later)

    rows = repo.list_changes(match_id=1000)
    assert len(rows) == 2
    assert rows[0].detected_at < rows[1].detected_at
    # Booleans roundtrip correctly through 0/1.
    assert rows[0].was_starting is True
    assert rows[0].is_starting is False


def test_sqlite_late_change_filters_by_team(sqlite_conn: sqlite3.Connection) -> None:
    repo = SQLiteLateTeamChangeRepository(sqlite_conn)
    repo.record_change(_change(team_id=500001, player_id=4001))
    repo.record_change(_change(team_id=500002, player_id=4002))

    home_only = repo.list_changes(match_id=1000, team_id=500001)
    assert len(home_only) == 1
    assert home_only[0].team_id == 500001


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_schema_migrates_v7_to_v8(tmp_path: Path) -> None:
    """An existing v7 DB should auto-migrate to v8 + carry working tables."""
    db_path = tmp_path / "legacy.db"

    raw = sqlite3.connect(db_path)
    raw.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (7);
        """
    )
    raw.close()

    repo = SQLiteRepository(db_path)
    try:
        version = repo._conn.execute(  # noqa: SLF001
            "SELECT version FROM schema_version"
        ).fetchone()["version"]
        assert version == 8
        # Table exists + accepts an insert.
        injuries = SQLiteInjuryReportRepository(repo._conn)  # noqa: SLF001
        injuries.record_report(_report())
        assert injuries.list_reports(season=2026, round=8) != []
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Firestore — fake client mirrors test_team_lists.py shape
# ---------------------------------------------------------------------------


class _FakeFirestoreCollection:
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}
        self._filters: list[tuple[str, str, object]] = []
        self._order_by: str | None = None

    # add() and document().set() both write to the same backing store.
    def add(self, data: dict) -> None:
        self._docs[f"_auto_{len(self._docs)}"] = data

    def document(self, doc_id: str) -> _FakeDocument:
        return _FakeDocument(self, doc_id)

    def where(self, field: str, op: str, value: object) -> _FakeFirestoreCollection:
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
            d
            for d in self._docs.values()
            if all(_apply_op(d.get(f), op, v) for f, op, v in self._filters)
        ]
        if self._order_by is not None:
            filtered.sort(key=lambda d: d.get(self._order_by))  # type: ignore[arg-type]

        class _Doc:
            def __init__(self, d: dict) -> None:
                self._d = d

            def to_dict(self) -> dict:
                return self._d

        return [_Doc(d) for d in filtered]


class _FakeDocument:
    def __init__(self, parent: _FakeFirestoreCollection, doc_id: str) -> None:
        self._parent = parent
        self._doc_id = doc_id

    def set(self, data: dict) -> None:
        self._parent._docs[self._doc_id] = data


def _apply_op(left: object, op: str, right: object) -> bool:
    if op == "==":
        return left == right
    raise NotImplementedError(op)


class _FakeFirestoreClient:
    def __init__(self, expected_collection: str) -> None:
        self._expected = expected_collection
        self._collection = _FakeFirestoreCollection()

    def collection(self, name: str) -> _FakeFirestoreCollection:
        assert name == self._expected
        return self._collection


def test_firestore_injury_record_and_list() -> None:
    repo = FirestoreInjuryReportRepository(client=_FakeFirestoreClient("injury_reports"))
    repo.record_report(_report(player_id=1, status=InjuryStatus.OUT, round=7))
    repo.record_report(_report(player_id=2, status=InjuryStatus.TEST, round=8))

    rows = repo.list_reports(season=2026)
    assert len(rows) == 2
    assert [r.round for r in rows] == [7, 8]  # ordered by round


def test_firestore_injury_doc_id_overwrites_in_place() -> None:
    repo = FirestoreInjuryReportRepository(client=_FakeFirestoreClient("injury_reports"))
    repo.record_report(_report(weeks_out=4))
    repo.record_report(
        _report(
            weeks_out=1,
            scraped_at=datetime(2026, 4, 24, 9, 0, tzinfo=UTC),
        )
    )

    rows = repo.list_reports(season=2026, round=8)
    assert len(rows) == 1
    assert rows[0].weeks_out == 1


def test_firestore_late_change_record_and_list() -> None:
    repo = FirestoreLateTeamChangeRepository(client=_FakeFirestoreClient("late_team_changes"))
    early = _change(detected_at=datetime(2026, 4, 25, 9, 0, tzinfo=UTC))
    late = _change(
        player_id=4002,
        detected_at=datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
    )
    repo.record_change(early)
    repo.record_change(late)

    rows = repo.list_changes(match_id=1000)
    assert [r.detected_at for r in rows] == [early.detected_at, late.detected_at]


def test_firestore_injury_ttl_present() -> None:
    """Firestore docs need a ttl_timestamp for the platform-infra TTL policy."""
    client = _FakeFirestoreClient("injury_reports")
    repo = FirestoreInjuryReportRepository(client=client)
    repo.record_report(_report(scraped_at=datetime(2026, 4, 21, 9, 0, tzinfo=UTC)))
    [(_, doc)] = client._collection._docs.items()
    assert "ttl_timestamp" in doc
    assert doc["ttl_timestamp"] == datetime(2026, 4, 21, 9, 0, tzinfo=UTC) + timedelta(days=365)
