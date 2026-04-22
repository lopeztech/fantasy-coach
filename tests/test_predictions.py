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
    FirestorePredictionStore,
    PredictionOut,
    PredictionStore,
    TeamInfo,
    _ensure_model,
    _feature_hash,
    compute_predictions,
    get_prediction_store,
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


# ---------------------------------------------------------------------------
# _ensure_model — GCS bootstrap for the precompute Job
# ---------------------------------------------------------------------------


def test_ensure_model_noop_when_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "already-here.joblib"
    path.write_bytes(b"x")
    # Should not touch GCS at all (no env var set, no client imported).
    _ensure_model(path)
    assert path.read_bytes() == b"x"


def test_ensure_model_raises_when_missing_and_no_gcs_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FANTASY_COACH_MODEL_GCS_URI", raising=False)
    with pytest.raises(FileNotFoundError):
        _ensure_model(tmp_path / "missing.joblib")


def test_ensure_model_downloads_from_gcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "subdir" / "logistic.joblib"
    monkeypatch.setenv(
        "FANTASY_COACH_MODEL_GCS_URI",
        "gs://fc-models/logistic/latest.joblib",
    )

    fake_blob = MagicMock()

    def _fake_download(dest: str) -> None:
        Path(dest).write_bytes(b"downloaded-bytes")

    fake_blob.download_to_filename.side_effect = _fake_download
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = fake_blob
    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket

    with patch("google.cloud.storage.Client", return_value=fake_client) as client_cls:
        _ensure_model(target)

    client_cls.assert_called_once_with()
    fake_client.bucket.assert_called_once_with("fc-models")
    fake_bucket.blob.assert_called_once_with("logistic/latest.joblib")
    fake_blob.download_to_filename.assert_called_once_with(str(target))
    assert target.read_bytes() == b"downloaded-bytes"


@pytest.mark.parametrize(
    "bad_uri",
    ["s3://nope/path", "gs://just-bucket", "gs:///no-bucket/path"],
)
def test_ensure_model_rejects_malformed_gcs_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_uri: str
) -> None:
    monkeypatch.setenv("FANTASY_COACH_MODEL_GCS_URI", bad_uri)
    with pytest.raises(ValueError):
        _ensure_model(tmp_path / "target.joblib")


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


def test_compute_dispatches_ensemble_artifact_end_to_end(
    store: PredictionStore, sqlite_repo: SQLiteRepository, tmp_path: Path
) -> None:
    """Point FANTASY_COACH_MODEL_PATH at an ensemble artifact (not logistic)
    and verify the returned probability comes from the ensemble combiner.

    Exercises the artifact-sniffing loader end-to-end — no patching of
    ``load_model``; just the real dispatcher path.
    """
    from fantasy_coach.feature_engineering import FEATURE_NAMES, TrainingFrame
    from fantasy_coach.models.ensemble import EnsembleModel, save_ensemble
    from fantasy_coach.models.logistic import train_logistic

    # Two logistic bases trained on the live 18-feature shape, different seeds.
    rng = np.random.default_rng(7)
    n = 150
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(int)
    frame = TrainingFrame(
        X=X,
        y=y,
        match_ids=np.arange(n),
        start_times=np.arange(n, dtype=float),
        feature_names=FEATURE_NAMES,
    )
    base_a = train_logistic(frame, test_fraction=0.0, random_state=0)
    base_b = train_logistic(frame, test_fraction=0.0, random_state=11)
    base_blobs = [
        {
            "model_type": "logistic",
            "pipeline": base_a.pipeline,
            "feature_names": base_a.feature_names,
        },
        {
            "model_type": "logistic",
            "pipeline": base_b.pipeline,
            "feature_names": base_b.feature_names,
        },
    ]
    weights = np.array([0.6, 0.4])
    ensemble = EnsembleModel(
        mode="weighted",
        base_model_names=("a", "b"),
        weights=weights,
    )
    model_path = tmp_path / "ensemble.joblib"
    save_ensemble(model_path, ensemble=ensemble, base_blobs=base_blobs)

    raw_match = _load_fixture(UPCOMING_FIXTURE)
    mock_round = MagicMock(return_value=_sample_round_payload("/match/url"))
    mock_match = MagicMock(return_value=raw_match)

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
    pred = result[0]
    # Probability must be the ensemble-combined value — derive the expected
    # prob from the actual feature row the endpoint would have fed in.
    from fantasy_coach.feature_engineering import FeatureBuilder
    from fantasy_coach.features import extract_match_features

    match = extract_match_features(raw_match)
    feature_row = np.asarray([FeatureBuilder().feature_row(match)], dtype=float)
    pa = base_a.pipeline.predict_proba(feature_row)[0, 1]
    pb = base_b.pipeline.predict_proba(feature_row)[0, 1]
    expected = round(float(weights[0] * pa + weights[1] * pb), 4)
    assert pred.homeWinProbability == expected


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


def test_compute_force_bypasses_cache_and_rescrapes(
    store: PredictionStore, sqlite_repo: SQLiteRepository, tmp_path: Path
) -> None:
    """``force=True`` makes the precompute Job re-scrape even when predictions
    are already cached — so team-list changes between Tue/Thu runs land."""
    raw_match = _load_fixture(UPCOMING_FIXTURE)
    # Pre-populate the cache with a stale prediction for the same round.
    match = extract_match_features(raw_match)
    stale = PredictionOut(
        matchId=match.match_id,
        home=TeamInfo(id=match.home.team_id, name=match.home.name),
        away=TeamInfo(id=match.away.team_id, name=match.away.name),
        kickoff=match.start_time.isoformat(),
        predictedWinner="home",
        homeWinProbability=0.99,  # clearly a stale / sentinel value
        modelVersion="stale",
        featureHash="stale",
    )
    store.put(2026, 8, [stale])

    mock_round = MagicMock(return_value=_sample_round_payload("/match/url"))
    mock_match = MagicMock(return_value=raw_match)
    mock_model = _make_mock_model(0.42)
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
            force=True,
        )

    mock_round.assert_called_once()  # force bypassed the cache and actually scraped
    assert len(result) == 1
    # Fresh computation wins over the stale cache entry.
    assert result[0].homeWinProbability == 0.42
    assert result[0].modelVersion != "stale"


def test_endpoint_returns_503_when_cache_empty(client: TestClient, tmp_path: Path) -> None:
    """The endpoint no longer scrapes; an empty cache is a 503 with retry hint."""
    empty_store = PredictionStore(path=tmp_path / "p.db")
    with patch("fantasy_coach.app._get_store", return_value=empty_store):
        response = client.get("/predictions?season=2026&round=8")
    assert response.status_code == 503
    assert "precompute" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# FirestorePredictionStore — in-memory fake client round-trip
# ---------------------------------------------------------------------------


class _FakeDocRef:
    def __init__(self, store: dict, key: tuple[str, str]) -> None:
        self._store = store
        self._key = key

    def set(self, data: dict) -> None:
        self._store[self._key] = dict(data)

    def get(self):
        class _Snap:
            def __init__(self, data: dict | None) -> None:
                self._data = data

            @property
            def exists(self) -> bool:
                return self._data is not None

            def to_dict(self) -> dict | None:
                return self._data

        return _Snap(self._store.get(self._key))


class _FakeCollection:
    def __init__(self, store: dict, name: str) -> None:
        self._store = store
        self._name = name

    def document(self, doc_id: str) -> _FakeDocRef:
        return _FakeDocRef(self._store, (self._name, doc_id))


class _FakeFirestoreClient:
    def __init__(self) -> None:
        self._store: dict = {}

    def collection(self, name: str) -> _FakeCollection:
        return _FakeCollection(self._store, name)


def test_firestore_prediction_store_round_trip() -> None:
    store = FirestorePredictionStore(client=_FakeFirestoreClient())
    preds = [
        PredictionOut(
            matchId=1,
            home=TeamInfo(id=10, name="Home"),
            away=TeamInfo(id=20, name="Away"),
            kickoff="2026-05-01T08:00:00+00:00",
            predictedWinner="home",
            homeWinProbability=0.6,
            modelVersion="v1",
            featureHash="fh1",
        ),
        PredictionOut(
            matchId=2,
            home=TeamInfo(id=30, name="A"),
            away=TeamInfo(id=40, name="B"),
            kickoff="2026-05-02T08:00:00+00:00",
            predictedWinner="away",
            homeWinProbability=0.3,
            modelVersion="v1",
            featureHash="fh1",
        ),
    ]
    store.put(2026, 8, preds)
    loaded = store.get(2026, 8)
    assert len(loaded) == 2
    assert loaded[0].matchId == 1
    assert loaded[0].homeWinProbability == 0.6
    assert loaded[1].matchId == 2


def test_firestore_prediction_store_get_empty_returns_empty_list() -> None:
    store = FirestorePredictionStore(client=_FakeFirestoreClient())
    assert store.get(2026, 8) == []


def test_firestore_prediction_store_put_overwrites() -> None:
    """A second put for the same (season, round) replaces the prior entry
    — load-bearing for ``--force`` semantics on the Job."""
    store = FirestorePredictionStore(client=_FakeFirestoreClient())
    p1 = PredictionOut(
        matchId=1,
        home=TeamInfo(id=1, name="A"),
        away=TeamInfo(id=2, name="B"),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner="home",
        homeWinProbability=0.9,
        modelVersion="v1",
        featureHash="fh1",
    )
    store.put(2026, 8, [p1])
    # Second put with same key should replace, not append.
    p2 = p1.model_copy(update={"homeWinProbability": 0.1, "predictedWinner": "away"})
    store.put(2026, 8, [p2])
    loaded = store.get(2026, 8)
    assert len(loaded) == 1
    assert loaded[0].homeWinProbability == 0.1


def test_get_prediction_store_factory_defaults_to_sqlite(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("FANTASY_COACH_PREDICTIONS_DB_PATH", str(tmp_path / "p.db"))
    store = get_prediction_store()
    assert isinstance(store, PredictionStore)


def test_get_prediction_store_factory_picks_firestore(monkeypatch) -> None:
    """``STORAGE_BACKEND=firestore`` wires the factory to FirestorePredictionStore."""
    monkeypatch.setenv("STORAGE_BACKEND", "firestore")
    # Avoid actually constructing the real google.cloud.firestore client.
    with patch(
        "fantasy_coach.predictions.FirestorePredictionStore",
        return_value=MagicMock(spec=FirestorePredictionStore),
    ) as cls:
        store = get_prediction_store()
    cls.assert_called_once()
    assert store is cls.return_value


def test_endpoint_returns_cached_predictions(client: TestClient, tmp_path: Path) -> None:
    """Cache is populated by the precompute Job; endpoint just reads it."""
    raw = _load_fixture(UPCOMING_FIXTURE)
    match = extract_match_features(raw)
    cached = PredictionOut(
        matchId=match.match_id,
        home=TeamInfo(id=match.home.team_id, name=match.home.name),
        away=TeamInfo(id=match.away.team_id, name=match.away.name),
        kickoff=match.start_time.isoformat(),
        predictedWinner="home",
        homeWinProbability=0.65,
        modelVersion="abc123",
        featureHash=_feature_hash(),
    )
    populated_store = PredictionStore(path=tmp_path / "p.db")
    populated_store.put(2026, 8, [cached])

    with patch("fantasy_coach.app._get_store", return_value=populated_store):
        response = client.get("/predictions?season=2026&round=8")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    pred = body[0]
    assert pred["matchId"] == match.match_id
    assert pred["predictedWinner"] == "home"
    assert pred["homeWinProbability"] == 0.65
    assert pred["modelVersion"] == "abc123"
    assert "featureHash" in pred
