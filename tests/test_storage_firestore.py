"""Tests for FirestoreRepository using an in-memory fake client.

No real GCP calls are made — the fake client stores documents in a plain
dict and implements the same chained-call interface as the real SDK.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fantasy_coach.features import extract_match_features
from fantasy_coach.storage.firestore import FirestoreRepository

FIXTURES = Path(__file__).parent / "fixtures"
FULLTIME_FIXTURE = "match-2024-rd1-sea-eagles-v-rabbitohs.json"
UPCOMING_FIXTURE = "match-2026-rd8-wests-tigers-v-raiders.json"


def _match(fixture: str):
    return extract_match_features(json.loads((FIXTURES / fixture).read_text()))


# ---------------------------------------------------------------------------
# In-memory Firestore fake
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    def __init__(self, data: dict | None) -> None:
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict | None:
        return self._data


class _FakeDocument:
    def __init__(self, store: dict, collection: str, doc_id: str) -> None:
        self._store = store
        self._key = (collection, doc_id)

    def set(self, data: dict) -> None:
        self._store[self._key] = dict(data)

    def get(self) -> _FakeSnapshot:
        return _FakeSnapshot(self._store.get(self._key))


class _FakeQuery:
    def __init__(self, store: dict, collection: str, filters: list) -> None:
        self._store = store
        self._collection = collection
        self._filters = filters  # [(field, op, value), ...]
        self._order_fields: list[str] = []

    def where(self, field: str, op: str, value: object) -> _FakeQuery:
        q = _FakeQuery(self._store, self._collection, [*self._filters, (field, op, value)])
        q._order_fields = list(self._order_fields)
        return q

    def order_by(self, field: str) -> _FakeQuery:
        q = _FakeQuery(self._store, self._collection, list(self._filters))
        q._order_fields = [*self._order_fields, field]
        return q

    def stream(self) -> list[_FakeSnapshot]:
        results = [
            data
            for (col, _), data in self._store.items()
            if col == self._collection and self._matches(data)
        ]
        # Stable multi-key sort: iterate order fields in reverse so the first
        # field ends up as the primary sort key (Python sort is stable).
        for field in reversed(self._order_fields):
            results.sort(key=lambda d, f=field: (d.get(f) is None, d.get(f, "")))
        return [_FakeSnapshot(d) for d in results]

    def _matches(self, data: dict) -> bool:
        for field, op, value in self._filters:
            if op == "==" and data.get(field) != value:
                return False
        return True


class _FakeCollection:
    def __init__(self, store: dict, name: str) -> None:
        self._store = store
        self._name = name

    def document(self, doc_id: str) -> _FakeDocument:
        return _FakeDocument(self._store, self._name, doc_id)

    def where(self, field: str, op: str, value: object) -> _FakeQuery:
        return _FakeQuery(self._store, self._name, [(field, op, value)])


class _FakeBatch:
    """Minimal WriteBatch shim — accumulates set() ops and flushes on commit()."""

    def __init__(self, store: dict) -> None:
        self._store = store
        self._ops: list[tuple[tuple[str, str], dict]] = []

    def set(self, doc_ref: _FakeDocument, data: dict) -> None:
        self._ops.append((doc_ref._key, dict(data)))

    def commit(self) -> None:
        for key, data in self._ops:
            self._store[key] = data
        self._ops.clear()


class _FakeFirestoreClient:
    def __init__(self) -> None:
        self._store: dict = {}

    def collection(self, name: str) -> _FakeCollection:
        return _FakeCollection(self._store, name)

    def batch(self) -> _FakeBatch:
        return _FakeBatch(self._store)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo() -> FirestoreRepository:
    return FirestoreRepository(client=_FakeFirestoreClient())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_firestore_repo_exposes_protocol_methods(repo: FirestoreRepository) -> None:
    for name in ("upsert_match", "get_match", "list_matches"):
        assert callable(getattr(repo, name))


def test_round_trip_full_time_match(repo: FirestoreRepository) -> None:
    row = _match(FULLTIME_FIXTURE)
    repo.upsert_match(row)
    loaded = repo.get_match(row.match_id)
    assert loaded is not None
    assert loaded == row


def test_round_trip_upcoming_match(repo: FirestoreRepository) -> None:
    row = _match(UPCOMING_FIXTURE)
    repo.upsert_match(row)
    loaded = repo.get_match(row.match_id)
    assert loaded is not None
    assert loaded == row
    assert loaded.home.score is None
    assert loaded.home.players == []


def test_get_match_missing_returns_none(repo: FirestoreRepository) -> None:
    assert repo.get_match(999_999) is None


def test_upsert_is_idempotent(repo: FirestoreRepository) -> None:
    row = _match(FULLTIME_FIXTURE)
    repo.upsert_match(row)
    repo.upsert_match(row)
    repo.upsert_match(row)
    loaded = repo.get_match(row.match_id)
    assert loaded == row


def test_upsert_replaces_stale_state(repo: FirestoreRepository) -> None:
    upcoming = _match(UPCOMING_FIXTURE)
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
    assert len(loaded.home_player_stats) == len(fulltime.home_player_stats)


def test_player_stats_round_trip(repo: FirestoreRepository) -> None:
    row = _match(FULLTIME_FIXTURE)
    repo.upsert_match(row)
    loaded = repo.get_match(row.match_id)
    assert loaded is not None
    assert loaded.home_player_stats == row.home_player_stats
    assert loaded.away_player_stats == row.away_player_stats


def test_list_matches_by_season(repo: FirestoreRepository) -> None:
    a = _match(FULLTIME_FIXTURE)  # season 2024
    b = _match(UPCOMING_FIXTURE)  # season 2026
    repo.upsert_match(a)
    repo.upsert_match(b)
    assert [m.match_id for m in repo.list_matches(2024)] == [a.match_id]
    assert [m.match_id for m in repo.list_matches(2026)] == [b.match_id]
    assert repo.list_matches(2025) == []


def test_list_matches_by_season_and_round(repo: FirestoreRepository) -> None:
    a = _match(FULLTIME_FIXTURE)  # season=2024 round=1
    repo.upsert_match(a)
    assert repo.list_matches(2024, round=1) == [a]
    assert repo.list_matches(2024, round=2) == []


# ---------------------------------------------------------------------------
# Batched writes (#181)
# ---------------------------------------------------------------------------


def test_upsert_matches_batch_writes_all_docs(repo: FirestoreRepository) -> None:
    a = _match(FULLTIME_FIXTURE)
    b = _match(UPCOMING_FIXTURE)
    repo.upsert_matches_batch([a, b])
    assert repo.get_match(a.match_id) == a
    assert repo.get_match(b.match_id) == b


def test_upsert_matches_batch_empty_is_noop(repo: FirestoreRepository) -> None:
    repo.upsert_matches_batch([])  # must not raise


def test_upsert_matches_batch_is_idempotent(repo: FirestoreRepository) -> None:
    a = _match(FULLTIME_FIXTURE)
    repo.upsert_matches_batch([a])
    repo.upsert_matches_batch([a])
    assert repo.get_match(a.match_id) == a


def test_upsert_matches_batch_chunks_large_input(repo: FirestoreRepository) -> None:
    # Build 600 pseudo-unique matches to force two batch commits (>500 per chunk).
    base = _match(FULLTIME_FIXTURE)
    rows = [base.model_copy(update={"match_id": i}) for i in range(600)]
    repo.upsert_matches_batch(rows)
    for row in rows:
        assert repo.get_match(row.match_id) is not None
