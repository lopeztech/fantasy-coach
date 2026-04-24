"""Tests for the GET /teams/{team_id}/profile endpoint."""

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
    app_module._profile_cache.clear()
    yield
    app_module._repo = None
    app_module._store = None
    app_module._profile_cache.clear()


def test_returns_200_with_correct_structure(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=0),
        _make_match(2, 2, home_id=20, away_id=10, home_score=14, away_score=22, offset_seconds=1),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    assert response.status_code == 200
    body = response.json()
    assert body["teamId"] == 10
    assert body["teamName"] == "Home Team"
    assert body["season"] == 2026
    assert "currentRecord" in body
    assert "currentElo" in body
    assert "eloTrend" in body
    assert "recentForm" in body
    assert "nextFixture" in body
    assert "allFixtures" in body


def test_current_record_reflects_results(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=0),
        _make_match(2, 2, home_id=10, away_id=20, home_score=10, away_score=20, offset_seconds=1),
        _make_match(3, 3, home_id=10, away_id=20, home_score=15, away_score=15, offset_seconds=2),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    assert response.status_code == 200
    rec = response.json()["currentRecord"]
    assert rec == {"wins": 1, "losses": 1, "draws": 1}


def test_recent_form_returns_correct_letters(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=0),
        _make_match(2, 2, home_id=10, away_id=20, home_score=10, away_score=20, offset_seconds=1),
        _make_match(3, 3, home_id=10, away_id=20, home_score=15, away_score=15, offset_seconds=2),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    assert response.json()["recentForm"] == ["W", "L", "D"]


def test_recent_form_capped_at_10(client: TestClient) -> None:
    matches = [
        _make_match(
            i, i, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=i
        )
        for i in range(1, 16)
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    assert len(response.json()["recentForm"]) == 10


def test_next_fixture_is_first_upcoming(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=0),
        _make_match(
            2, 2, home_id=10, away_id=20,
            home_score=None, away_score=None,
            match_state="Upcoming", offset_seconds=1,
        ),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    body = response.json()
    assert body["nextFixture"] is not None
    assert body["nextFixture"]["matchId"] == 2
    assert body["nextFixture"]["round"] == 2


def test_next_fixture_none_when_all_played(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    assert response.json()["nextFixture"] is None


def test_all_fixtures_contains_all_matches(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10, offset_seconds=0),
        _make_match(
            2, 2, home_id=10, away_id=20,
            home_score=None, away_score=None,
            match_state="Upcoming", offset_seconds=1,
        ),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    assert len(response.json()["allFixtures"]) == 2


def test_returns_404_when_team_not_in_season(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10),
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/99/profile?season=2026")

    assert response.status_code == 404
    assert "99" in response.json()["detail"]


def test_elo_trend_up_after_wins(client: TestClient) -> None:
    # Team wins 3 in a row — Elo should trend up.
    matches = [
        _make_match(
            i, i, home_id=10, away_id=20, home_score=30, away_score=10, offset_seconds=i
        )
        for i in range(1, 4)
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    assert response.json()["eloTrend"] == "up"


def test_elo_trend_down_after_losses(client: TestClient) -> None:
    # Team loses 3 in a row — Elo should trend down.
    matches = [
        _make_match(
            i, i, home_id=10, away_id=20, home_score=10, away_score=30, offset_seconds=i
        )
        for i in range(1, 4)
    ]
    with patch("fantasy_coach.app._get_repo", return_value=_mock_repo(matches)):
        response = client.get("/teams/10/profile?season=2026")

    assert response.json()["eloTrend"] == "down"


def test_503_when_repo_unavailable(client: TestClient) -> None:
    boom_repo = MagicMock()
    boom_repo.list_matches.side_effect = RuntimeError("db gone")

    with patch("fantasy_coach.app._get_repo", return_value=boom_repo):
        response = client.get("/teams/10/profile?season=2026")

    assert response.status_code == 503


def test_cache_is_used_on_second_request(client: TestClient) -> None:
    matches = [
        _make_match(1, 1, home_id=10, away_id=20, home_score=20, away_score=10),
    ]
    mock_repo = _mock_repo(matches)

    with patch("fantasy_coach.app._get_repo", return_value=mock_repo):
        r1 = client.get("/teams/10/profile?season=2026")
        r2 = client.get("/teams/10/profile?season=2026")

    assert r1.status_code == 200
    assert r2.status_code == 200
    # list_matches should only be called once (second hit served from cache).
    assert mock_repo.list_matches.call_count == 1
