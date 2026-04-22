"""Tests for the GET /teams/{team_id}/form endpoint."""

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
def _reset_app_repo():
    import fantasy_coach.app as app_module

    app_module._repo = None
    yield
    app_module._repo = None


def test_returns_200_with_correct_structure(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=0),
        _make_match(2, 2, home_id=20, away_id=10, home_score=14, away_score=22, offset_seconds=1),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/form?season=2026")

    assert response.status_code == 200
    body = response.json()
    assert body["teamId"] == 10
    assert body["teamName"] == "Home Team"
    assert body["season"] == 2026
    assert len(body["matches"]) == 2


def test_returns_404_when_team_not_in_season(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/99/form?season=2026")

    assert response.status_code == 404
    assert "99" in response.json()["detail"]


def test_last_parameter_limits_returned_matches(client: TestClient) -> None:
    matches = [
        _make_match(i, i, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=i)
        for i in range(1, 6)
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/form?season=2026&last=3")

    assert response.status_code == 200
    assert len(response.json()["matches"]) == 3


def test_result_values_are_win_loss_draw(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=0),
        _make_match(2, 2, home_id=10, away_id=20, home_score=10, away_score=20, offset_seconds=1),
        _make_match(3, 3, home_id=10, away_id=20, home_score=15, away_score=15, offset_seconds=2),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/form?season=2026")

    results = [m["result"] for m in response.json()["matches"]]
    assert results == ["win", "loss", "draw"]


def test_elo_after_values_are_positive(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=0),
        _make_match(2, 2, home_id=20, away_id=10, home_score=10, away_score=14, offset_seconds=1),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/form?season=2026")

    body = response.json()
    for entry in body["matches"]:
        assert entry["eloAfter"] > 0


def test_excludes_non_fulltime_matches(client: TestClient) -> None:
    matches = [
        _make_match(
            1, 1, home_id=10, away_id=20, home_score=20, away_score=10, match_state="FullTime"
        ),
        _make_match(
            2,
            2,
            home_id=10,
            away_id=20,
            home_score=None,
            away_score=None,
            match_state="Upcoming",
            offset_seconds=1,
        ),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/form?season=2026")

    assert response.status_code == 200
    assert len(response.json()["matches"]) == 1


def test_503_when_repo_unavailable(client: TestClient) -> None:
    boom_repo = MagicMock()
    boom_repo.list_matches.side_effect = RuntimeError("db gone")

    with patch("fantasy_coach.app._get_repo", return_value=boom_repo):
        response = client.get("/teams/10/form?season=2026")

    assert response.status_code == 503
