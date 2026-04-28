"""Tests for GET /leaderboard, including the group-filtered branch (#173)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fantasy_coach.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _stat_snap(uid: str, *, accuracy: float, wins: int, losses: int) -> MagicMock:
    """Build a DocumentSnapshot mock matching what `users/{uid}/stats/{season}` looks like."""
    snap = MagicMock()
    snap.exists = True
    snap.to_dict.return_value = {
        "wins": wins,
        "losses": losses,
        "total_tips": wins + losses,
        "accuracy": accuracy,
        "margin_points": 0.0,
        "current_streak": 0,
        "longest_streak": 0,
        "display_name": f"User {uid}",
    }
    parent = MagicMock()
    parent.parent.id = uid
    snap.reference.parent = parent
    return snap


def _fake_db_with_group(member_uids: list[str], stats_by_uid: dict[str, MagicMock]) -> MagicMock:
    """Build a fake firestore.Client where:
    - `groups/{gid}/members` enumerates `member_uids` (each as a doc with id=uid)
    - `db.get_all([users/{uid}/stats/{season} refs])` returns the matching stat snapshots
    """
    db = MagicMock()
    member_doc_mocks = []
    for uid in member_uids:
        m = MagicMock()
        m.id = uid
        member_doc_mocks.append(m)
    db.collection.return_value.document.return_value.collection.return_value.stream.return_value = (
        iter(member_doc_mocks)
    )

    def fake_get_all(refs: list[Any]) -> list[MagicMock]:
        # We can't introspect the chained MagicMock refs, but order doesn't matter
        # because `_compute_leaderboard` re-sorts by accuracy.
        return list(stats_by_uid.values())

    db.get_all.side_effect = fake_get_all
    return db


def test_leaderboard_group_filter_returns_member_stats(client: TestClient) -> None:
    """Group-filtered leaderboard fetches each member's stats by direct path,
    not via collection_group + __name__ (which raises 'must be a Key')."""
    stats = {
        "uid_a": _stat_snap("uid_a", accuracy=0.7, wins=7, losses=3),
        "uid_b": _stat_snap("uid_b", accuracy=0.5, wins=5, losses=5),
    }
    db = _fake_db_with_group(["uid_a", "uid_b"], stats)

    with (
        patch.dict("os.environ", {"STORAGE_BACKEND": "firestore"}),
        patch("fantasy_coach.app._get_firestore_client", return_value=db),
    ):
        resp = client.get("/leaderboard?season=2026&group_id=g1")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["season"] == 2026
    assert body["groupId"] == "g1"
    # The endpoint sorts entries by accuracy DESC and assigns rank.
    assert [e["rank"] for e in body["entries"]] == [1, 2]
    assert body["entries"][0]["accuracy"] == pytest.approx(0.7)
    # Verify get_all was used (the bug we're regression-testing was a
    # collection_group + where('__name__', 'in', ...) call that raised
    # "__key__ filter value must be a Key").
    assert db.get_all.called


def test_leaderboard_group_filter_empty_group(client: TestClient) -> None:
    db = _fake_db_with_group([], {})
    with (
        patch.dict("os.environ", {"STORAGE_BACKEND": "firestore"}),
        patch("fantasy_coach.app._get_firestore_client", return_value=db),
    ):
        resp = client.get("/leaderboard?season=2026&group_id=empty")
    assert resp.status_code == 200
    assert resp.json()["entries"] == []
    db.get_all.assert_not_called()
