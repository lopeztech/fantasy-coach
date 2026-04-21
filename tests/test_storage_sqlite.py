from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from fantasy_coach.features import extract_match_features
from fantasy_coach.storage import SQLiteRepository
from fantasy_coach.storage.sqlite import SCHEMA_VERSION

FIXTURES = Path(__file__).parent / "fixtures"
FULLTIME_FIXTURE = "match-2024-rd1-sea-eagles-v-rabbitohs.json"
UPCOMING_FIXTURE = "match-2026-rd8-wests-tigers-v-raiders.json"


def _match(fixture: str):
    return extract_match_features(json.loads((FIXTURES / fixture).read_text()))


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SQLiteRepository]:
    db = SQLiteRepository(tmp_path / "test.db")
    yield db
    db.close()


def test_sqlite_repo_exposes_protocol_methods(repo: SQLiteRepository) -> None:
    for name in ("upsert_match", "get_match", "list_matches"):
        assert callable(getattr(repo, name))


def test_round_trip_full_time_match(repo: SQLiteRepository) -> None:
    row = _match(FULLTIME_FIXTURE)

    repo.upsert_match(row)
    loaded = repo.get_match(row.match_id)

    assert loaded is not None
    assert loaded == row


def test_round_trip_upcoming_match(repo: SQLiteRepository) -> None:
    row = _match(UPCOMING_FIXTURE)

    repo.upsert_match(row)
    loaded = repo.get_match(row.match_id)

    assert loaded is not None
    assert loaded == row
    assert loaded.home.score is None
    assert loaded.home.players == []


def test_get_match_missing_returns_none(repo: SQLiteRepository) -> None:
    assert repo.get_match(999_999) is None


def test_upsert_is_idempotent(repo: SQLiteRepository) -> None:
    row = _match(FULLTIME_FIXTURE)

    repo.upsert_match(row)
    repo.upsert_match(row)
    repo.upsert_match(row)

    # Should still only have one row — and child-row counts should match the source.
    loaded = repo.get_match(row.match_id)
    assert loaded == row


def test_upsert_replaces_stale_children_when_match_transitions(
    repo: SQLiteRepository,
) -> None:
    upcoming = _match(UPCOMING_FIXTURE)
    # Pretend the same match later comes back with roster + stats
    # (just reuse the 2024 row but patch its id/season/round to match).
    fulltime = _match(FULLTIME_FIXTURE).model_copy(
        update={
            "match_id": upcoming.match_id,
            "season": upcoming.season,
            "round": upcoming.round,
        }
    )

    repo.upsert_match(upcoming)
    repo.upsert_match(fulltime)

    loaded = repo.get_match(upcoming.match_id)
    assert loaded == fulltime
    assert len(loaded.home.players) == len(fulltime.home.players)
    assert len(loaded.team_stats) == len(fulltime.team_stats)


def test_list_matches_by_season(repo: SQLiteRepository) -> None:
    a = _match(FULLTIME_FIXTURE)  # 2024
    b = _match(UPCOMING_FIXTURE)  # 2026
    repo.upsert_match(a)
    repo.upsert_match(b)

    assert [m.match_id for m in repo.list_matches(2024)] == [a.match_id]
    assert [m.match_id for m in repo.list_matches(2026)] == [b.match_id]
    assert repo.list_matches(2025) == []


def test_list_matches_by_season_and_round(repo: SQLiteRepository) -> None:
    a = _match(FULLTIME_FIXTURE)  # season=2024 round=1
    repo.upsert_match(a)

    assert repo.list_matches(2024, round=1) == [a]
    assert repo.list_matches(2024, round=2) == []


def test_schema_version_is_recorded(tmp_path: Path) -> None:
    db_path = tmp_path / "v.db"
    SQLiteRepository(db_path).close()

    with sqlite3.connect(db_path) as conn:
        (version,) = conn.execute("SELECT version FROM schema_version").fetchone()

    assert version == SCHEMA_VERSION


def test_mismatched_schema_version_rejects_open(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    SQLiteRepository(db_path).close()

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE schema_version SET version = 999")
        conn.commit()

    with pytest.raises(RuntimeError, match="schema v999"):
        SQLiteRepository(db_path)
