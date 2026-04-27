"""Tests for POST /predictions/{match_id}/whatif endpoint (#150)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fantasy_coach.app import app
from fantasy_coach.feature_engineering import FEATURE_NAMES
from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.predictions import PredictionOut, PredictionStore, TeamInfo

MATCH_ID = 9001
SEASON = 2026
ROUND = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    import fantasy_coach.app as app_module
    import fantasy_coach.predictions as pred_module

    app_module._store = None
    app_module._repo = None
    app_module._whatif_rate.clear()
    app_module._whatif_result_cache.clear()
    pred_module._whatif_base_cache.clear()
    yield
    app_module._store = None
    app_module._repo = None
    app_module._whatif_rate.clear()
    app_module._whatif_result_cache.clear()
    pred_module._whatif_base_cache.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pred(prob: float = 0.65) -> PredictionOut:
    return PredictionOut(
        matchId=MATCH_ID,
        home=TeamInfo(id=1, name="Tigers"),
        away=TeamInfo(id=2, name="Raiders"),
        kickoff="2026-05-01T09:00:00+00:00",
        predictedWinner="home" if prob >= 0.5 else "away",
        homeWinProbability=prob,
        modelVersion="abc123",
        featureHash="fh1",
    )


def _make_match() -> MatchRow:
    return MatchRow(
        match_id=MATCH_ID,
        season=SEASON,
        round=ROUND,
        start_time=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        match_state="Upcoming",
        venue="Campbelltown",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(team_id=1, name="Tigers", nick_name="TIG", score=None, players=[]),
        away=TeamRow(team_id=2, name="Raiders", nick_name="RAI", score=None, players=[]),
        team_stats=[],
    )


def _make_mock_model(prob: float = 0.70) -> MagicMock:
    model = MagicMock()
    model.predict_home_win_prob.return_value = np.array([prob])
    model.feature_names = FEATURE_NAMES
    return model


def _store_with_pred(tmp_path: Path) -> PredictionStore:
    store = PredictionStore(path=tmp_path / "pred.db")
    store.put(SEASON, ROUND, [_make_pred(0.65)])
    return store


def _mock_repo() -> MagicMock:
    repo = MagicMock()
    repo.get_match.return_value = _make_match()
    repo.list_matches.return_value = []
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_whatif_404_when_no_stored_prediction(client: TestClient, tmp_path: Path) -> None:
    import fantasy_coach.app as app_module

    app_module._store = PredictionStore(path=tmp_path / "empty.db")
    app_module._repo = _mock_repo()

    r = client.post(
        f"/predictions/{MATCH_ID}/whatif",
        json={"season": SEASON, "round": ROUND, "overrides": {}},
    )
    assert r.status_code == 404


def test_whatif_422_on_unknown_override(client: TestClient, tmp_path: Path) -> None:
    import fantasy_coach.app as app_module

    app_module._store = _store_with_pred(tmp_path)

    r = client.post(
        f"/predictions/{MATCH_ID}/whatif",
        json={"season": SEASON, "round": ROUND, "overrides": {"elo_diff": 50.0}},
    )
    assert r.status_code == 422
    assert "elo_diff" in r.json()["detail"]


def test_whatif_returns_200_with_base_prob(client: TestClient, tmp_path: Path) -> None:
    import fantasy_coach.app as app_module

    app_module._store = _store_with_pred(tmp_path)
    app_module._repo = _mock_repo()

    mock_model = _make_mock_model(0.65)
    with (
        patch("fantasy_coach.predictions._ensure_model"),
        patch("fantasy_coach.predictions._model_version", return_value="ver1"),
        patch("fantasy_coach.models.loader.load_model", return_value=mock_model),
    ):
        r = client.post(
            f"/predictions/{MATCH_ID}/whatif",
            json={"season": SEASON, "round": ROUND, "overrides": {}},
        )

    assert r.status_code == 200
    data = r.json()
    assert data["baseHomeWinProbability"] == pytest.approx(0.65, abs=1e-3)
    assert "homeWinProbability" in data
    assert data["predictedWinner"] in ("home", "away")


def test_whatif_weather_override_clears_missing_flag(client: TestClient, tmp_path: Path) -> None:
    """Overriding a weather feature should set missing_weather=0 in the vector sent to the model."""
    import fantasy_coach.app as app_module

    app_module._store = _store_with_pred(tmp_path)
    app_module._repo = _mock_repo()

    captured_x: list[np.ndarray] = []

    def _capture(x: np.ndarray) -> np.ndarray:
        captured_x.append(x.copy())
        return np.array([0.60])

    mock_model = _make_mock_model()
    mock_model.predict_home_win_prob.side_effect = _capture

    with (
        patch("fantasy_coach.predictions._ensure_model"),
        patch("fantasy_coach.predictions._model_version", return_value="ver1"),
        patch("fantasy_coach.models.loader.load_model", return_value=mock_model),
    ):
        r = client.post(
            f"/predictions/{MATCH_ID}/whatif",
            json={
                "season": SEASON,
                "round": ROUND,
                "overrides": {"is_wet": 1.0, "wind_kph": 30.0},
            },
        )

    assert r.status_code == 200
    assert captured_x, "model was not called"
    x = captured_x[0]
    assert x[0, FEATURE_NAMES.index("is_wet")] == pytest.approx(1.0)
    assert x[0, FEATURE_NAMES.index("wind_kph")] == pytest.approx(30.0)
    assert x[0, FEATURE_NAMES.index("missing_weather")] == pytest.approx(0.0)


def test_whatif_rest_override_applied(client: TestClient, tmp_path: Path) -> None:
    """days_rest_diff override replaces the feature value in the model call."""
    import fantasy_coach.app as app_module

    app_module._store = _store_with_pred(tmp_path)
    app_module._repo = _mock_repo()

    captured_x: list[np.ndarray] = []

    def _capture(x: np.ndarray) -> np.ndarray:
        captured_x.append(x.copy())
        return np.array([0.55])

    mock_model = _make_mock_model()
    mock_model.predict_home_win_prob.side_effect = _capture

    with (
        patch("fantasy_coach.predictions._ensure_model"),
        patch("fantasy_coach.predictions._model_version", return_value="ver1"),
        patch("fantasy_coach.models.loader.load_model", return_value=mock_model),
    ):
        r = client.post(
            f"/predictions/{MATCH_ID}/whatif",
            json={
                "season": SEASON,
                "round": ROUND,
                "overrides": {"days_rest_diff": 5.0},
            },
        )

    assert r.status_code == 200
    assert captured_x, "model was not called"
    x = captured_x[0]
    assert x[0, FEATURE_NAMES.index("days_rest_diff")] == pytest.approx(5.0)


def test_whatif_result_is_cached(client: TestClient, tmp_path: Path) -> None:
    """Identical requests should hit the result cache, not call the model twice."""
    import fantasy_coach.app as app_module

    app_module._store = _store_with_pred(tmp_path)
    app_module._repo = _mock_repo()

    mock_model = _make_mock_model(0.65)

    with (
        patch("fantasy_coach.predictions._ensure_model"),
        patch("fantasy_coach.predictions._model_version", return_value="ver1"),
        patch("fantasy_coach.models.loader.load_model", return_value=mock_model),
    ):
        for _ in range(3):
            r = client.post(
                f"/predictions/{MATCH_ID}/whatif",
                json={"season": SEASON, "round": ROUND, "overrides": {"wind_kph": 20.0}},
            )
            assert r.status_code == 200

    # Model is called at most twice: once for the base vector rebuild via
    # compute_whatif_base (which has its own cache), and once for prediction.
    assert mock_model.predict_home_win_prob.call_count <= 2


def test_whatif_rate_limit_429(client: TestClient, tmp_path: Path) -> None:
    """21st+ request from the same user within 60 s should return 429."""
    import fantasy_coach.app as app_module

    app_module._store = _store_with_pred(tmp_path)
    app_module._repo = _mock_repo()

    mock_model = _make_mock_model(0.65)

    with (
        patch("fantasy_coach.predictions._ensure_model"),
        patch("fantasy_coach.predictions._model_version", return_value="ver1"),
        patch("fantasy_coach.models.loader.load_model", return_value=mock_model),
    ):
        # Send 22 requests, each with a unique override so result cache doesn't help.
        responses = [
            client.post(
                f"/predictions/{MATCH_ID}/whatif",
                json={
                    "season": SEASON,
                    "round": ROUND,
                    "overrides": {"wind_kph": float(i)},
                },
            )
            for i in range(22)
        ]

    ok = [r for r in responses if r.status_code == 200]
    limited = [r for r in responses if r.status_code == 429]
    assert len(ok) == 20
    assert len(limited) == 2
