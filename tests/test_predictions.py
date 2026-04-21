"""Tests for the /predictions endpoint and PredictionStore.

Network calls (fetch_round, fetch_match_from_url) are mocked with respx so
no real NRL.com traffic is generated. Model loading is also mocked so no
trained artefact is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fantasy_coach.app import app
from fantasy_coach.features import extract_match_features
from fantasy_coach.predictions import (
    PredictionOut,
    PredictionStore,
    TeamInfo,
    _feature_hash,
    compute_predictions,
)
from fantasy_coach.storage.sqlite import SQLiteRepository

FIXTURES = Path(__file__).parent / "fixtures"
UPCOMING_FIXTURE = "match-2026-rd8-wests-tigers-v-raiders.json"
FULLTIME_FIXTURE = "match-2024-rd1-sea-eagles-v-rabbitohs.json"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> PredictionStore:
    return PredictionStore(path=tmp_path / "pred.db")


@pytest.fixture
def sqlite_repo(tmp_path: Path) -> SQLiteRepository:
    return SQLiteRepository(tmp_path / "matches.db")


def _make_mock_model(prob: float = 0.65) -> MagicMock:
    model = MagicMock()
    model.predict_home_win_prob.return_value = np.array([prob])
    return model


def _sample_round_payload(match_url: str) -> dict:
    return {"fixtures": [{"matchCentreUrl": match_url}]}


# ---------------------------------------------------------------------------
# PredictionStore unit tests
# ---------------------------------------------------------------------------


def test_store_get_empty(store: PredictionStore) -> None:
    assert store.get(2026, 8) == []


def test_store_put_and_get(store: PredictionStore) -> None:
    pred = PredictionOut(
        matchId=12345,
        home=TeamInfo(id=1, name="Tigers"),
        away=TeamInfo(id=2, name="Raiders"),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner="home",
        homeWinProbability=0.65,
        modelVersion="abc123",
        featureHash=_feature_hash(),
    )
    store.put(2026, 8, [pred])
    loaded = store.get(2026, 8)
    assert len(loaded) == 1
    assert loaded[0].matchId == pred.matchId
    assert loaded[0].homeWinProbability == pred.homeWinProbability
    assert loaded[0].predictedWinner == "home"


def test_store_put_is_idempotent(store: PredictionStore) -> None:
    pred = PredictionOut(
        matchId=99,
        home=TeamInfo(id=3, name="A"),
        away=TeamInfo(id=4, name="B"),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner="away",
        homeWinProbability=0.3,
        modelVersion="v1",
        featureHash="fh1",
    )
    store.put(2026, 8, [pred])
    store.put(2026, 8, [pred])  # second put replaces
    assert len(store.get(2026, 8)) == 1


# ---------------------------------------------------------------------------
# compute_predictions unit tests
# ---------------------------------------------------------------------------


def test_compute_returns_cached_without_scraping(
    store: PredictionStore, sqlite_repo: SQLiteRepository
) -> None:
    pred = PredictionOut(
        matchId=1,
        home=TeamInfo(id=1, name="Home"),
        away=TeamInfo(id=2, name="Away"),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner="home",
        homeWinProbability=0.7,
        modelVersion="m1",
        featureHash="f1",
    )
    store.put(2026, 8, [pred])

    mock_fetch_round = MagicMock()
    result = compute_predictions(2026, 8, sqlite_repo, store, fetch_round_fn=mock_fetch_round)
    mock_fetch_round.assert_not_called()
    assert len(result) == 1
    assert result[0].matchId == 1


def test_compute_raises_file_not_found_when_no_model(
    store: PredictionStore, sqlite_repo: SQLiteRepository, tmp_path: Path
) -> None:
    mock_round = MagicMock(return_value=_sample_round_payload("/some/url"))
    mock_match = MagicMock(return_value=_load_fixture(UPCOMING_FIXTURE))
    with pytest.raises(FileNotFoundError):
        compute_predictions(
            2026,
            8,
            sqlite_repo,
            store,
            model_path=tmp_path / "missing.joblib",
            fetch_round_fn=mock_round,
            fetch_match_fn=mock_match,
        )


def test_compute_cache_miss_scrapes_and_stores(
    store: PredictionStore, sqlite_repo: SQLiteRepository, tmp_path: Path
) -> None:
    raw_match = _load_fixture(UPCOMING_FIXTURE)
    mock_round = MagicMock(return_value=_sample_round_payload("/match/url"))
    mock_match = MagicMock(return_value=raw_match)
    mock_model = _make_mock_model(0.72)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"fake-model-bytes")

    with patch("fantasy_coach.predictions.load_model", return_value=mock_model):
        result = compute_predictions(
            2026,
            8,
            sqlite_repo,
            store,
            model_path=model_path,
            fetch_round_fn=mock_round,
            fetch_match_fn=mock_match,
        )

    assert len(result) == 1
    p = result[0]
    match = extract_match_features(raw_match)
    assert p.matchId == match.match_id
    assert p.home.id == match.home.team_id
    assert p.away.id == match.away.team_id
    assert p.homeWinProbability == 0.72
    assert p.predictedWinner == "home"

    # Verify it's now cached
    cached = store.get(2026, 8)
    assert len(cached) == 1
    assert cached[0].matchId == p.matchId


def test_compute_returns_empty_when_no_fixtures(
    store: PredictionStore, sqlite_repo: SQLiteRepository, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"bytes")
    mock_round = MagicMock(return_value={"fixtures": []})

    with patch("fantasy_coach.predictions.load_model", return_value=_make_mock_model()):
        result = compute_predictions(
            2026,
            9,
            sqlite_repo,
            store,
            model_path=model_path,
            fetch_round_fn=mock_round,
        )
    assert result == []


# ---------------------------------------------------------------------------
# /predictions endpoint integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_app_store():
    """Reset module-level store so tests don't share state."""
    import fantasy_coach.app as app_module

    app_module._store = None
    yield
    app_module._store = None


def _endpoint_patches(tmp_path: Path):
    """Patch both IO sinks the endpoint touches so tests stay hermetic."""
    return (
        patch("fantasy_coach.app.get_repository", return_value=MagicMock()),
        patch("fantasy_coach.app._get_store", return_value=PredictionStore(path=tmp_path / "p.db")),
    )


def test_endpoint_returns_503_when_no_model(client: TestClient, tmp_path: Path) -> None:
    p1, p2 = _endpoint_patches(tmp_path)
    with (
        p1,
        p2,
        patch(
            "fantasy_coach.app.compute_predictions",
            side_effect=FileNotFoundError(tmp_path / "missing.joblib"),
        ),
    ):
        response = client.get("/predictions?season=2026&round=8")
    assert response.status_code == 503


def test_endpoint_returns_predictions(client: TestClient, tmp_path: Path) -> None:
    raw = _load_fixture(UPCOMING_FIXTURE)
    match = extract_match_features(raw)
    expected = [
        PredictionOut(
            matchId=match.match_id,
            home=TeamInfo(id=match.home.team_id, name=match.home.name),
            away=TeamInfo(id=match.away.team_id, name=match.away.name),
            kickoff=match.start_time.isoformat(),
            predictedWinner="home",
            homeWinProbability=0.65,
            modelVersion="abc123",
            featureHash=_feature_hash(),
        )
    ]
    p1, p2 = _endpoint_patches(tmp_path)
    with p1, p2, patch("fantasy_coach.app.compute_predictions", return_value=expected):
        response = client.get("/predictions?season=2026&round=8")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1
    pred = body[0]
    assert pred["matchId"] == match.match_id
    assert pred["predictedWinner"] == "home"
    assert pred["homeWinProbability"] == 0.65
    assert pred["modelVersion"] == "abc123"
    assert "featureHash" in pred
