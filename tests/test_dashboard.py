"""Tests for the GET /me/dashboard endpoint (#148)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fantasy_coach.app import app
from fantasy_coach.features import MatchRow, TeamRow


def _make_match(
    match_id: int,
    round_: int,
    home_id: int,
    away_id: int,
    home_score: int | None,
    away_score: int | None,
    home_name: str = "Home Team",
    away_name: str = "Away Team",
    season: int = 2026,
    match_state: str = "FullTime",
    offset_seconds: int = 0,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round_,
        start_time=datetime(2026, 3, 1, 9, 0, offset_seconds, tzinfo=UTC),
        match_state=match_state,
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name=home_name,
            nick_name=home_name,
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=away_id,
            name=away_name,
            nick_name=away_name,
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


def _mock_repo(matches: list[MatchRow]) -> MagicMock:
    repo = MagicMock()
    repo.list_matches.return_value = matches
    return repo


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_app_singletons():
    import fantasy_coach.app as app_module

    app_module._repo = None
    app_module._store = None
    app_module._dashboard_cache.clear()
    yield
    app_module._repo = None
    app_module._store = None
    app_module._dashboard_cache.clear()


# ── Basic shape ──────────────────────────────────────────────────────────────


def test_dashboard_returns_200_with_correct_shape(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10),
        _make_match(2, 2, home_id=20, away_id=10, home_score=None, away_score=None,
                    match_state="Pre Game"),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026")
    assert resp.status_code == 200
    data = resp.json()
    assert "season" in data
    assert "currentRound" in data
    assert "untippedMatchIds" in data
    assert "totalTips" in data
    assert "correctTips" in data


def test_dashboard_current_round_is_lowest_upcoming(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, 10, 20, 20, 10),
        _make_match(2, 2, 20, 10, None, None, match_state="Pre Game"),
        _make_match(3, 3, 10, 30, None, None, match_state="Pre Game"),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026")
    assert resp.json()["currentRound"] == 2


def test_dashboard_current_round_falls_back_to_max_completed(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, 10, 20, 20, 10),
        _make_match(2, 2, 20, 10, 14, 22),
        _make_match(3, 3, 10, 30, 30, 12),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026")
    assert resp.json()["currentRound"] == 3


def test_dashboard_current_round_none_when_no_matches(client: TestClient) -> None:
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo([])), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026")
    assert resp.json()["currentRound"] is None


# ── Untipped match IDs ───────────────────────────────────────────────────────


def test_dashboard_untipped_match_ids_contains_current_round(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, 10, 20, 20, 10),
        _make_match(2, 2, 20, 10, None, None, match_state="Pre Game"),
        _make_match(3, 2, 10, 30, None, None, match_state="Pre Game"),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026")
    data = resp.json()
    assert set(data["untippedMatchIds"]) == {2, 3}


# ── Next fixture ─────────────────────────────────────────────────────────────


def test_dashboard_next_fixture_for_favourite_team(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, 10, 20, None, None, home_name="Tigers", away_name="Roosters",
                    match_state="Pre Game"),
        _make_match(2, 2, 10, 30, None, None, home_name="Tigers", away_name="Broncos",
                    match_state="Pre Game"),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026&favourite_team_id=10")
    data = resp.json()
    assert data["nextFixture"] is not None
    assert data["nextFixture"]["matchId"] == 1
    assert data["nextFixture"]["isHome"] is True
    assert data["nextFixture"]["opponent"] == "Roosters"


def test_dashboard_next_fixture_away_team(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, 20, 10, None, None, home_name="Roosters", away_name="Tigers",
                    match_state="Pre Game"),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026&favourite_team_id=10")
    data = resp.json()
    assert data["nextFixture"]["isHome"] is False
    assert data["nextFixture"]["opponent"] == "Roosters"


def test_dashboard_no_next_fixture_when_all_fulltime(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, 10, 20, 20, 10),
        _make_match(2, 2, 10, 30, 14, 20),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026&favourite_team_id=10")
    assert resp.json()["nextFixture"] is None


def test_dashboard_no_favourite_team_when_not_set(client: TestClient) -> None:
    matches = [_make_match(1, 1, 10, 20, None, None, match_state="Pre Game")]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026")
    data = resp.json()
    assert data["favouriteTeamId"] is None
    assert data["nextFixture"] is None


# ── Prediction attachment ─────────────────────────────────────────────────────


def test_dashboard_attaches_prediction_to_next_fixture(client: TestClient) -> None:
    from fantasy_coach.predictions import PredictionOut

    matches = [
        _make_match(1, 1, 10, 20, None, None, home_name="Tigers", away_name="Roosters",
                    match_state="Pre Game"),
    ]
    pred = PredictionOut(
        matchId=1,
        home={"id": 10, "name": "Tigers"},
        away={"id": 20, "name": "Roosters"},
        kickoff="2026-03-01T09:00:00",
        predictedWinner="home",
        homeWinProbability=0.65,
        modelVersion="abc123",
        featureHash="xyz",
    )
    mock_store = MagicMock()
    mock_store.get.return_value = [pred]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)), \
         patch("fantasy_coach.app._get_store", return_value=mock_store):
        resp = client.get("/me/dashboard?season=2026&favourite_team_id=10")
    data = resp.json()
    assert data["nextFixture"]["predWinner"] == "home"
    assert abs(data["nextFixture"]["predProb"] - 0.65) < 0.001


# ── Accuracy defaults ────────────────────────────────────────────────────────


def test_dashboard_season_accuracy_and_tips_default_to_zero(client: TestClient) -> None:
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo([])), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2026")
    data = resp.json()
    assert data["totalTips"] == 0
    assert data["correctTips"] == 0
    assert data["seasonAccuracy"] is None


# ── Season passthrough ───────────────────────────────────────────────────────


def test_dashboard_season_value_matches_request(client: TestClient) -> None:
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo([])), \
         patch("fantasy_coach.app._get_store", return_value=MagicMock(get=MagicMock(return_value=[]))):
        resp = client.get("/me/dashboard?season=2024")
    assert resp.json()["season"] == 2024


# ── Error handling ───────────────────────────────────────────────────────────


def test_dashboard_returns_503_when_repo_unavailable(client: TestClient) -> None:
    broken_repo = MagicMock()
    broken_repo.list_matches.side_effect = RuntimeError("DB down")
    with patch("fantasy_coach.app._get_repo", return_value=broken_repo):
        resp = client.get("/me/dashboard?season=2026")
    assert resp.status_code == 503
