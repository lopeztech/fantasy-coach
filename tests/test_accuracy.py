"""Tests for the /accuracy endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fantasy_coach.app import app
from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.predictions import PredictionOut, PredictionStore, TeamInfo


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_app_singletons():
    import fantasy_coach.app as app_module

    app_module._store = None
    app_module._repo = None
    yield
    app_module._store = None
    app_module._repo = None


def _make_match(
    match_id: int,
    round_: int,
    home_score: int,
    away_score: int,
    state: str = "FullTime",
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=2026,
        round=round_,
        start_time=datetime(2026, 4, 1, 9, 0, tzinfo=UTC),
        match_state=state,
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(team_id=1, name="Home", nick_name="H", score=home_score, players=[]),
        away=TeamRow(team_id=2, name="Away", nick_name="A", score=away_score, players=[]),
        team_stats=[],
    )


def _make_pred(
    match_id: int,
    predicted_winner: str,
    model_version: str = "abc12345",
) -> PredictionOut:
    return PredictionOut(
        matchId=match_id,
        home=TeamInfo(id=1, name="Home"),
        away=TeamInfo(id=2, name="Away"),
        kickoff="2026-04-01T09:00:00+00:00",
        predictedWinner=predicted_winner,
        homeWinProbability=0.65 if predicted_winner == "home" else 0.35,
        modelVersion=model_version,
        featureHash="fh1",
    )


# ---------------------------------------------------------------------------
# /accuracy endpoint — happy path
# ---------------------------------------------------------------------------


def test_accuracy_returns_correct_round_stats(client: TestClient, tmp_path: Path) -> None:
    matches = [
        _make_match(1, round_=1, home_score=20, away_score=10),  # home wins
        _make_match(2, round_=1, home_score=10, away_score=20),  # away wins
        _make_match(3, round_=1, home_score=14, away_score=14),  # tie -> away (not >)
    ]
    # Predictions: 2/3 correct (match 1 correct, match 2 correct, match 3 wrong)
    preds = [
        _make_pred(1, "home"),   # correct
        _make_pred(2, "away"),   # correct
        _make_pred(3, "home"),   # wrong (tie counted as away)
    ]

    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")
    store.put(2026, 1, preds)

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/accuracy?season=2026&last_n_rounds=10")

    assert resp.status_code == 200
    body = resp.json()
    assert body["scoredMatches"] == 3
    assert body["threshold"] == pytest.approx(0.55)
    assert len(body["rounds"]) == 1
    r = body["rounds"][0]
    assert r["round"] == 1
    assert r["total"] == 3
    assert r["correct"] == 2
    assert r["accuracy"] == pytest.approx(2 / 3)


def test_accuracy_overall_accuracy_and_below_threshold_flag(
    client: TestClient, tmp_path: Path
) -> None:
    matches = [_make_match(i, round_=1, home_score=20, away_score=10) for i in range(1, 6)]
    # Only 2 of 5 predicted correctly → 0.4 < threshold 0.55
    preds = [_make_pred(i, "home" if i <= 2 else "away") for i in range(1, 6)]

    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")
    store.put(2026, 1, preds)

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/accuracy?season=2026&last_n_rounds=10")

    body = resp.json()
    assert body["overallAccuracy"] == pytest.approx(2 / 5)
    assert body["belowThreshold"] is True


def test_accuracy_above_threshold(client: TestClient, tmp_path: Path) -> None:
    matches = [_make_match(i, round_=1, home_score=20, away_score=10) for i in range(1, 5)]
    preds = [_make_pred(i, "home") for i in range(1, 5)]  # 4/4 correct

    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")
    store.put(2026, 1, preds)

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/accuracy?season=2026&last_n_rounds=10")

    body = resp.json()
    assert body["overallAccuracy"] == pytest.approx(1.0)
    assert body["belowThreshold"] is False


def test_accuracy_no_scored_matches(client: TestClient, tmp_path: Path) -> None:
    matches = [_make_match(1, round_=1, home_score=0, away_score=0, state="Upcoming")]

    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/accuracy?season=2026&last_n_rounds=10")

    body = resp.json()
    assert resp.status_code == 200
    assert body["scoredMatches"] == 0
    assert body["overallAccuracy"] is None
    assert body["belowThreshold"] is False
    assert body["rounds"] == []


def test_accuracy_multiple_rounds_last_n_respected(client: TestClient, tmp_path: Path) -> None:
    matches = [
        _make_match(i, round_=r, home_score=20, away_score=10)
        for r in range(1, 6)
        for i in [r * 10]
    ]
    preds_by_round = {
        r: [_make_pred(r * 10, "home")] for r in range(1, 6)
    }

    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")
    for r, preds in preds_by_round.items():
        store.put(2026, r, preds)

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/accuracy?season=2026&last_n_rounds=3")

    body = resp.json()
    # Only last 3 rounds (3, 4, 5) should be returned
    assert len(body["rounds"]) == 3
    returned_rounds = {r["round"] for r in body["rounds"]}
    assert returned_rounds == {3, 4, 5}


def test_accuracy_by_model_version_breakdown(client: TestClient, tmp_path: Path) -> None:
    matches = [
        _make_match(1, round_=1, home_score=20, away_score=10),
        _make_match(2, round_=1, home_score=20, away_score=10),
        _make_match(3, round_=2, home_score=20, away_score=10),
    ]
    preds_r1 = [
        _make_pred(1, "home", model_version="aaa"),
        _make_pred(2, "away", model_version="aaa"),  # wrong
    ]
    preds_r2 = [
        _make_pred(3, "home", model_version="bbb"),
    ]

    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")
    store.put(2026, 1, preds_r1)
    store.put(2026, 2, preds_r2)

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/accuracy?season=2026&last_n_rounds=10")

    body = resp.json()
    by_mv = {mv["modelVersion"]: mv for mv in body["byModelVersion"]}
    assert "aaa" in by_mv
    assert "bbb" in by_mv
    assert by_mv["aaa"]["total"] == 2
    assert by_mv["aaa"]["correct"] == 1
    assert by_mv["bbb"]["total"] == 1
    assert by_mv["bbb"]["correct"] == 1


def test_accuracy_returns_503_when_repo_unavailable(client: TestClient) -> None:
    mock_repo = MagicMock()
    mock_repo.list_matches.side_effect = RuntimeError("db gone")

    with patch("fantasy_coach.app._get_repo", return_value=mock_repo):
        resp = client.get("/accuracy?season=2026&last_n_rounds=10")

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()
