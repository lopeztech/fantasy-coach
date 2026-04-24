"""Tests for the GET /me/dashboard endpoint."""

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
    app_module._dashboard_cache.clear()
    yield
    app_module._store = None
    app_module._repo = None
    app_module._dashboard_cache.clear()


def _make_match(
    match_id: int,
    round_: int,
    home_id: int = 1,
    away_id: int = 2,
    home_name: str = "Home",
    away_name: str = "Away",
    match_state: str = "Upcoming",
    home_score: int | None = None,
    away_score: int | None = None,
    season: int = 2026,
    offset_seconds: int = 0,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round_,
        start_time=datetime(2026, 4, 1, 9, 0, offset_seconds, tzinfo=UTC),
        match_state=match_state,
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name=home_name,
            nick_name=home_name[:3],
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=away_id,
            name=away_name,
            nick_name=away_name[:3],
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


def _make_pred(
    match_id: int,
    predicted_winner: str = "home",
    model_version: str = "abc123",
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
# Happy-path tests
# ---------------------------------------------------------------------------


def test_dashboard_returns_200_with_correct_shape(client: TestClient, tmp_path: Path) -> None:
    matches = [
        _make_match(1, round_=1, match_state="Upcoming"),
        _make_match(2, round_=1, match_state="Upcoming", offset_seconds=1),
    ]
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026")

    assert resp.status_code == 200
    body = resp.json()
    # Schema shape check
    assert "season" in body
    assert "currentRound" in body
    assert "favouriteTeamId" in body
    assert "nextFixture" in body
    assert "untippedMatchIds" in body
    assert "seasonAccuracy" in body
    assert "totalTips" in body
    assert "correctTips" in body


def test_dashboard_current_round_is_lowest_upcoming(client: TestClient, tmp_path: Path) -> None:
    """currentRound should be the lowest round with at least one non-FullTime match."""
    matches = [
        _make_match(10, round_=1, match_state="FullTime", home_score=20, away_score=10),
        _make_match(20, round_=2, match_state="Upcoming"),
        _make_match(21, round_=2, match_state="Upcoming", offset_seconds=1),
    ]
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026")

    assert resp.status_code == 200
    assert resp.json()["currentRound"] == 2


def test_dashboard_current_round_falls_back_to_max_completed(
    client: TestClient, tmp_path: Path
) -> None:
    """When season is complete, currentRound = highest round seen."""
    matches = [
        _make_match(1, round_=1, match_state="FullTime", home_score=20, away_score=10),
        _make_match(2, round_=2, match_state="FullTime", home_score=14, away_score=12),
        _make_match(3, round_=3, match_state="FullTime", home_score=8, away_score=6),
    ]
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026")

    assert resp.status_code == 200
    assert resp.json()["currentRound"] == 3


def test_dashboard_untipped_match_ids_contains_current_round(
    client: TestClient, tmp_path: Path
) -> None:
    matches = [
        _make_match(100, round_=5, match_state="Upcoming"),
        _make_match(101, round_=5, match_state="Upcoming", offset_seconds=1),
    ]
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026")

    body = resp.json()
    assert set(body["untippedMatchIds"]) == {100, 101}


def test_dashboard_next_fixture_for_favourite_team(client: TestClient, tmp_path: Path) -> None:
    matches = [
        _make_match(
            10,
            round_=1,
            home_id=7,
            away_id=8,
            home_name="Broncos",
            away_name="Roosters",
            match_state="Upcoming",
        ),
        _make_match(
            11,
            round_=1,
            home_id=9,
            away_id=10,
            home_name="Storm",
            away_name="Panthers",
            match_state="Upcoming",
            offset_seconds=1,
        ),
    ]
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026&favourite_team_id=7")

    body = resp.json()
    assert body["favouriteTeamId"] == 7
    assert body["nextFixture"] is not None
    fixture = body["nextFixture"]
    assert fixture["matchId"] == 10
    assert fixture["isHome"] is True
    assert fixture["opponent"] == "Roosters"
    assert fixture["opponentId"] == 8
    assert fixture["season"] == 2026


def test_dashboard_next_fixture_away_team(client: TestClient, tmp_path: Path) -> None:
    matches = [
        _make_match(
            20,
            round_=3,
            home_id=1,
            away_id=5,
            home_name="Knights",
            away_name="Tigers",
            match_state="Upcoming",
        ),
    ]
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026&favourite_team_id=5")

    body = resp.json()
    fixture = body["nextFixture"]
    assert fixture is not None
    assert fixture["isHome"] is False
    assert fixture["opponent"] == "Knights"


def test_dashboard_no_next_fixture_when_all_fulltime(client: TestClient, tmp_path: Path) -> None:
    matches = [
        _make_match(
            1,
            round_=1,
            home_id=10,
            away_id=20,
            match_state="FullTime",
            home_score=20,
            away_score=10,
        ),
    ]
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026&favourite_team_id=10")

    body = resp.json()
    assert body["nextFixture"] is None


def test_dashboard_season_accuracy_and_tips_default_to_zero(
    client: TestClient, tmp_path: Path
) -> None:
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = []
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026")

    body = resp.json()
    assert body["seasonAccuracy"] is None
    assert body["totalTips"] == 0
    assert body["correctTips"] == 0


def test_dashboard_attaches_prediction_to_next_fixture(client: TestClient, tmp_path: Path) -> None:
    matches = [
        _make_match(
            50,
            round_=4,
            home_id=3,
            away_id=6,
            home_name="Titans",
            away_name="Cowboys",
            match_state="Upcoming",
        ),
    ]
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = matches
    store = PredictionStore(path=tmp_path / "p.db")
    store.put(2026, 4, [_make_pred(50, "home")])

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026&favourite_team_id=3")

    body = resp.json()
    fixture = body["nextFixture"]
    assert fixture is not None
    assert fixture["predWinner"] == "home"
    assert fixture["predProb"] is not None


def test_dashboard_no_favourite_team_when_not_set(client: TestClient, tmp_path: Path) -> None:
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = []
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026")

    body = resp.json()
    assert body["favouriteTeamId"] is None
    assert body["nextFixture"] is None


def test_dashboard_returns_503_when_repo_unavailable(client: TestClient) -> None:
    mock_repo = MagicMock()
    mock_repo.list_matches.side_effect = RuntimeError("db gone")

    with patch("fantasy_coach.app._get_repo", return_value=mock_repo):
        resp = client.get("/me/dashboard?season=2026")

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_dashboard_season_value_matches_request(client: TestClient, tmp_path: Path) -> None:
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = []
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2025")

    assert resp.status_code == 200
    assert resp.json()["season"] == 2025


def test_dashboard_current_round_none_when_no_matches(client: TestClient, tmp_path: Path) -> None:
    mock_repo = MagicMock()
    mock_repo.list_matches.return_value = []
    store = PredictionStore(path=tmp_path / "p.db")

    with (
        patch("fantasy_coach.app._get_repo", return_value=mock_repo),
        patch("fantasy_coach.app._get_store", return_value=store),
    ):
        resp = client.get("/me/dashboard?season=2026")

    assert resp.status_code == 200
    assert resp.json()["currentRound"] is None
