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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fantasy_coach.app import app
from fantasy_coach.features import extract_match_features
from fantasy_coach.predictions import (
    AlternativeModels,
    FeatureContribution,
    FirestorePredictionStore,
    PickSummary,
    PredictionOut,
    PredictionStore,
    TeamInfo,
    _bookmaker_pick_summary,
    _compute_contributions,
    _ensure_model,
    _feature_hash,
    _record_team_list_snapshots,
    _try_load_secondary_model,
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


def test_ensure_model_downloads_from_gcs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    with patch("fantasy_coach.models.loader.load_model", return_value=mock_model):
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

    # Two logistic bases trained on the live FEATURE_NAMES shape, different seeds.
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


# ---------------------------------------------------------------------------
# _compute_contributions — feature-contribution extraction
# ---------------------------------------------------------------------------


def _make_logistic_loaded(n_features: int = 3):
    """Return a LoadedModel-shaped object with a fitted logistic pipeline."""
    from fantasy_coach.models.logistic import LoadedModel

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, n_features))
    # Make feature 0 dominate so we can assert it sorts to the top.
    y = (X[:, 0] > 0).astype(int)

    pipeline = Pipeline(
        steps=[("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X, y)
    return LoadedModel(
        pipeline=pipeline,
        feature_names=tuple(f"feat_{i}" for i in range(n_features)),
    )


def test_compute_contributions_picks_top_k_by_abs_value() -> None:
    loaded = _make_logistic_loaded(n_features=4)
    x = np.array([[2.5, 0.1, -0.2, 0.05]])  # feat_0 is dominant
    contribs = _compute_contributions(loaded, x, top_k=2)

    assert contribs is not None
    assert len(contribs) == 2
    # Sorted by |contribution| descending; feat_0 must win.
    assert contribs[0].feature == "feat_0"
    assert abs(contribs[0].contribution) >= abs(contribs[1].contribution)
    # Raw (unscaled) value passed through faithfully.
    assert contribs[0].value == pytest.approx(2.5)


def test_compute_contributions_sum_equals_logit_up_to_intercept() -> None:
    loaded = _make_logistic_loaded(n_features=3)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 3))
    # top_k covers all features so the sum should equal the full log-odds.
    contribs = _compute_contributions(loaded, x, top_k=3)

    assert contribs is not None
    intercept = float(loaded.pipeline.named_steps["lr"].intercept_[0])
    total = sum(c.contribution for c in contribs) + intercept

    raw_prob = float(loaded.pipeline.predict_proba(x)[0, 1])
    expected_logit = float(np.log(raw_prob / (1.0 - raw_prob)))
    # Rounded at 4dp, so allow a little slack.
    assert total == pytest.approx(expected_logit, abs=1e-3)


def test_compute_contributions_returns_none_for_non_logistic() -> None:
    """Mock a model with no ``.pipeline`` attribute — the happy path out."""
    mock = MagicMock(spec=["feature_names"])
    contribs = _compute_contributions(mock, np.zeros((1, 3)))
    assert contribs is None


def test_store_roundtrips_contributions(store: PredictionStore) -> None:
    pred = PredictionOut(
        matchId=42,
        home=TeamInfo(id=1, name="A"),
        away=TeamInfo(id=2, name="B"),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner="home",
        homeWinProbability=0.6,
        modelVersion="mv",
        featureHash="fh",
        contributions=[
            FeatureContribution(feature="elo_diff", value=42.3, contribution=0.31),
            FeatureContribution(feature="days_rest_diff", value=3.0, contribution=0.08),
        ],
    )
    store.put(2026, 8, [pred])

    loaded = store.get(2026, 8)
    assert len(loaded) == 1
    assert loaded[0].contributions is not None
    assert len(loaded[0].contributions) == 2
    assert loaded[0].contributions[0].feature == "elo_diff"
    assert loaded[0].contributions[0].contribution == pytest.approx(0.31)


def test_compute_contributions_filters_sentinel_and_flag_features() -> None:
    """Sentinel-valued / constant binary-flag features shouldn't surface in
    the top-K list, even if the coefficient happens to be big.

    ``is_home_field`` is a constant 1.0; ``missing_weather=0`` is "we have
    weather" (data-era flag, not narrative); ``venue_avg_total_points=0`` is
    "no venue history" (sentinel). None should appear in contributions.
    """
    from fantasy_coach.feature_engineering import FEATURE_NAMES
    from fantasy_coach.models.logistic import LoadedModel

    # Train a logistic with the real FEATURE_NAMES shape.
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    # Force ``is_home_field`` to always be 1.0 (its true value) so the trained
    # pipeline sees the same constant as in prod.
    home_idx = FEATURE_NAMES.index("is_home_field")
    X[:, home_idx] = 1.0
    y = (X[:, 0] > 0).astype(int)
    pipeline = Pipeline(
        steps=[("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X, y)
    loaded = LoadedModel(pipeline=pipeline, feature_names=FEATURE_NAMES)

    # Build an inference row where the sentinel/flag features are at their
    # "hide me" values.
    x = np.zeros((1, len(FEATURE_NAMES)))
    x[0, home_idx] = 1.0  # constant flag
    x[0, FEATURE_NAMES.index("missing_weather")] = 0.0  # "we do have weather"
    x[0, FEATURE_NAMES.index("venue_avg_total_points")] = 0.0  # no history
    x[0, FEATURE_NAMES.index("venue_home_win_rate")] = 0.5  # neutral prior
    x[0, FEATURE_NAMES.index("key_absence_diff")] = 0.0  # no missing regulars
    # Make one real feature carry strong signal so something survives.
    x[0, 0] = 2.0

    contribs = _compute_contributions(loaded, x, top_k=10)
    assert contribs is not None
    surfaced = {c.feature for c in contribs}
    assert "is_home_field" not in surfaced
    assert "missing_weather" not in surfaced
    assert "venue_avg_total_points" not in surfaced
    assert "venue_home_win_rate" not in surfaced
    assert "key_absence_diff" not in surfaced


def test_compute_contributions_dispatches_to_xgboost() -> None:
    """XGBoost artefacts use pred_contribs instead of coef × scaled-feature,
    but the returned shape is the same (top-K FeatureContribution list) and
    the sentinel filter still applies."""
    try:
        from xgboost import XGBClassifier

        from fantasy_coach.models.xgboost_model import LoadedModel as XGBLoaded
    except Exception:
        pytest.skip("xgboost / libomp not importable")

    from fantasy_coach.feature_engineering import FEATURE_NAMES

    rng = np.random.default_rng(7)
    n = 120
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    # Make feature 0 carry real signal; seed a few constant sentinel columns.
    y = (X[:, 0] > 0).astype(int)
    X[:, FEATURE_NAMES.index("is_home_field")] = 1.0
    X[:, FEATURE_NAMES.index("missing_weather")] = 0.0

    est = XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", verbosity=0)
    est.fit(X, y)
    loaded = XGBLoaded(estimator=est, feature_names=FEATURE_NAMES)

    # Same inference row the logistic ablation used — strong signal on feat 0.
    x = np.zeros((1, len(FEATURE_NAMES)))
    x[0, 0] = 2.0
    x[0, FEATURE_NAMES.index("is_home_field")] = 1.0
    x[0, FEATURE_NAMES.index("missing_weather")] = 0.0

    contribs = _compute_contributions(loaded, x, top_k=5)
    assert contribs is not None
    surfaced = {c.feature for c in contribs}
    # Sentinel / flag features never surface, no matter how large their
    # coefficient/split contribution happens to be.
    assert "is_home_field" not in surfaced
    assert "missing_weather" not in surfaced
    # Top contribution has the real signal feature at rank 1.
    assert contribs[0].feature == "elo_diff"  # FEATURE_NAMES[0]


def test_compute_contributions_enriches_key_absence_with_missing_players() -> None:
    """When builder + match are provided, ``key_absence_diff`` gets a
    structured detail list of missing-regular players the UI can render."""
    from datetime import UTC, datetime

    from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder
    from fantasy_coach.features import MatchRow, PlayerRow, TeamRow
    from fantasy_coach.models.logistic import LoadedModel

    builder = FeatureBuilder()

    # Seed a "regular" halfback for team 10 by recording two prior matches.
    def _p(pid: int, jersey: int, pos: str, on: bool, first: str, last: str) -> PlayerRow:
        return PlayerRow(
            player_id=pid,
            jersey_number=jersey,
            position=pos,
            first_name=first,
            last_name=last,
            is_on_field=on,
        )

    def _match(match_id: int, start_day: int, home_players, away_players) -> MatchRow:
        return MatchRow(
            match_id=match_id,
            season=2024,
            round=1,
            start_time=datetime(2024, 3, start_day, tzinfo=UTC),
            match_state="FullTime",
            venue="Eden Park",
            venue_city="Auckland",
            weather=None,
            home=TeamRow(team_id=10, name="T10", nick_name="T10", score=20, players=home_players),
            away=TeamRow(team_id=20, name="T20", nick_name="T20", score=14, players=away_players),
            team_stats=[],
        )

    regular_halfback = _p(107, 7, "Halfback", True, "Lachlan", "Ilias")
    home_regular = [regular_halfback, _p(101, 1, "Fullback", True, "F", "Fullback")]
    away_regular = [_p(207, 7, "Halfback", True, "Nathan", "Cleary")]
    for r in range(2):
        builder.record(_match(1000 + r, r + 1, home_regular, away_regular))

    # "Today": regular halfback is missing; fill-in is on field.
    fill_in = _p(999, 7, "Halfback", True, "Rookie", "Halfback")
    today_home = [fill_in, _p(101, 1, "Fullback", True, "F", "Fullback")]
    today_away = away_regular
    today = _match(2000, 10, today_home, today_away)

    # Build a trivial logistic on the live FEATURE_NAMES shape so the
    # detail-enrichment path is exercised on the key_absence_diff row.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)
    pipeline = Pipeline(
        steps=[("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X, y)
    loaded = LoadedModel(pipeline=pipeline, feature_names=FEATURE_NAMES)

    x = np.asarray([builder.feature_row(today)], dtype=float)
    # Use top_k=len(FEATURE_NAMES) so the test isn't sensitive to which random
    # coefficients happen to rank key_absence_diff above/below the cutoff.
    contribs = _compute_contributions(
        loaded, x, builder=builder, match=today, top_k=len(FEATURE_NAMES)
    )
    assert contribs is not None

    absence = next((c for c in contribs if c.feature == "key_absence_diff"), None)
    assert absence is not None, (
        "key_absence_diff should surface when a regular is missing; "
        f"got: {[c.feature for c in contribs]}"
    )
    assert absence.detail is not None
    home_missing = absence.detail.get("home_missing") or []
    away_missing = absence.detail.get("away_missing") or []
    assert len(home_missing) == 1, f"expected home missing the halfback, got {home_missing}"
    assert home_missing[0]["name"] == "Lachlan Ilias"
    assert home_missing[0]["position"] == "Halfback"
    assert home_missing[0]["weight"] > 0
    assert away_missing == []


def test_store_roundtrips_missing_contributions_as_none(store: PredictionStore) -> None:
    pred = PredictionOut(
        matchId=43,
        home=TeamInfo(id=1, name="A"),
        away=TeamInfo(id=2, name="B"),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner="home",
        homeWinProbability=0.6,
        modelVersion="mv",
        featureHash="fh",
        # contributions omitted
    )
    store.put(2026, 8, [pred])
    loaded = store.get(2026, 8)
    assert loaded[0].contributions is None


# ---------------------------------------------------------------------------
# _record_team_list_snapshots — wiring guard
# ---------------------------------------------------------------------------


def _make_match_row_with_team_lists(has_is_on_field: bool = True):
    from datetime import UTC, datetime

    from fantasy_coach.features import MatchRow, PlayerRow, TeamRow

    def _p(pid: int, jersey: int, on_field: bool | None) -> PlayerRow:
        return PlayerRow(
            player_id=pid,
            jersey_number=jersey,
            position="Fullback",
            first_name="F",
            last_name="L",
            is_on_field=on_field if has_is_on_field else None,
        )

    return MatchRow(
        match_id=12345,
        season=2026,
        round=8,
        start_time=datetime(2026, 4, 24, 9, 0, tzinfo=UTC),
        match_state="Upcoming",
        venue="Eden Park",
        venue_city="Auckland",
        weather=None,
        home=TeamRow(
            team_id=10,
            name="Tigers",
            nick_name="Tigers",
            score=None,
            players=[_p(1, 1, True), _p(2, 14, False)],
        ),
        away=TeamRow(
            team_id=20,
            name="Raiders",
            nick_name="Raiders",
            score=None,
            players=[_p(3, 1, True), _p(4, 14, False)],
        ),
        team_stats=[],
    )


def test_record_team_list_snapshots_writes_one_per_team() -> None:
    row = _make_match_row_with_team_lists(has_is_on_field=True)
    repo = MagicMock()

    _record_team_list_snapshots(repo, row)

    assert repo.record_snapshot.call_count == 2
    snapshots = [c.args[0] for c in repo.record_snapshot.call_args_list]
    assert {s.team_id for s in snapshots} == {10, 20}
    assert snapshots[0].season == 2026
    assert snapshots[0].round == 8


def test_record_team_list_snapshots_skips_when_no_is_on_field_flag() -> None:
    # Pre-drop scrape: players present but is_on_field is all None.
    row = _make_match_row_with_team_lists(has_is_on_field=False)
    repo = MagicMock()

    _record_team_list_snapshots(repo, row)

    repo.record_snapshot.assert_not_called()


def test_record_team_list_snapshots_swallows_repo_errors() -> None:
    row = _make_match_row_with_team_lists(has_is_on_field=True)
    repo = MagicMock()
    repo.record_snapshot.side_effect = RuntimeError("firestore down")

    # Should not raise — snapshot failures are best-effort per the wiring.
    _record_team_list_snapshots(repo, row)

    # Both teams attempted (home first, then away) even after the first error.
    assert repo.record_snapshot.call_count == 2


def test_compute_returns_empty_when_no_fixtures(
    store: PredictionStore, sqlite_repo: SQLiteRepository, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"bytes")
    mock_round = MagicMock(return_value={"fixtures": []})

    with patch("fantasy_coach.models.loader.load_model", return_value=_make_mock_model()):
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

    with patch("fantasy_coach.models.loader.load_model", return_value=mock_model):
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


# ---------------------------------------------------------------------------
# _bookmaker_pick_summary — bookmaker implied pick (#140)
# ---------------------------------------------------------------------------


def test_bookmaker_pick_summary_returns_pick_when_odds_available() -> None:
    from fantasy_coach.feature_engineering import FEATURE_NAMES

    x = np.zeros((1, len(FEATURE_NAMES)))
    odds_idx = FEATURE_NAMES.index("odds_home_win_prob")
    missing_idx = FEATURE_NAMES.index("missing_odds")
    x[0, odds_idx] = 0.72
    x[0, missing_idx] = 0.0  # odds available

    pick = _bookmaker_pick_summary(x, FEATURE_NAMES)
    assert pick is not None
    assert pick.predictedWinner == "home"
    assert pick.homeWinProbability == pytest.approx(0.72)


def test_bookmaker_pick_summary_away_when_prob_below_half() -> None:
    from fantasy_coach.feature_engineering import FEATURE_NAMES

    x = np.zeros((1, len(FEATURE_NAMES)))
    x[0, FEATURE_NAMES.index("odds_home_win_prob")] = 0.38
    x[0, FEATURE_NAMES.index("missing_odds")] = 0.0

    pick = _bookmaker_pick_summary(x, FEATURE_NAMES)
    assert pick is not None
    assert pick.predictedWinner == "away"
    assert pick.homeWinProbability == pytest.approx(0.38)


def test_bookmaker_pick_summary_returns_none_when_odds_missing() -> None:
    from fantasy_coach.feature_engineering import FEATURE_NAMES

    x = np.zeros((1, len(FEATURE_NAMES)))
    x[0, FEATURE_NAMES.index("odds_home_win_prob")] = 0.65
    x[0, FEATURE_NAMES.index("missing_odds")] = 1.0  # no odds data

    pick = _bookmaker_pick_summary(x, FEATURE_NAMES)
    assert pick is None


def test_bookmaker_pick_summary_returns_none_for_unknown_feature_names() -> None:
    pick = _bookmaker_pick_summary(np.zeros((1, 3)), ("feat_a", "feat_b", "feat_c"))
    assert pick is None


# ---------------------------------------------------------------------------
# _try_load_secondary_model — graceful loading of logistic artefact (#140)
# ---------------------------------------------------------------------------


def test_try_load_secondary_model_returns_none_when_missing_and_no_gcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FANTASY_COACH_LOGISTIC_GCS_URI", raising=False)
    result = _try_load_secondary_model(
        tmp_path / "missing.joblib", "FANTASY_COACH_LOGISTIC_GCS_URI"
    )
    assert result is None


def test_try_load_secondary_model_loads_when_file_exists(tmp_path: Path) -> None:
    from fantasy_coach.feature_engineering import FEATURE_NAMES, TrainingFrame
    from fantasy_coach.models.logistic import save_model, train_logistic

    rng = np.random.default_rng(0)
    n = 60
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)
    frame = TrainingFrame(
        X=X,
        y=y,
        match_ids=np.arange(n),
        start_times=np.arange(n, dtype=float),
        feature_names=FEATURE_NAMES,
    )
    result = train_logistic(frame, test_fraction=0.0)
    model_path = tmp_path / "logistic.joblib"
    save_model(result, model_path)

    loaded = _try_load_secondary_model(model_path, "FANTASY_COACH_LOGISTIC_GCS_URI")
    assert loaded is not None
    x = np.zeros((1, len(FEATURE_NAMES)))
    prob = loaded.predict_home_win_prob(x)
    assert 0.0 <= prob[0] <= 1.0


# ---------------------------------------------------------------------------
# AlternativeModels round-trip through SQLite store (#140)
# ---------------------------------------------------------------------------


def test_store_roundtrips_alternatives(store: PredictionStore) -> None:
    pred = PredictionOut(
        matchId=55,
        home=TeamInfo(id=1, name="A"),
        away=TeamInfo(id=2, name="B"),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner="home",
        homeWinProbability=0.65,
        modelVersion="mv",
        featureHash="fh",
        alternatives=AlternativeModels(
            logistic=PickSummary(predictedWinner="away", homeWinProbability=0.45),
            bookmaker=PickSummary(predictedWinner="home", homeWinProbability=0.68),
        ),
    )
    store.put(2026, 8, [pred])
    loaded = store.get(2026, 8)
    assert len(loaded) == 1
    alts = loaded[0].alternatives
    assert alts is not None
    assert alts.logistic is not None
    assert alts.logistic.predictedWinner == "away"
    assert alts.logistic.homeWinProbability == pytest.approx(0.45)
    assert alts.bookmaker is not None
    assert alts.bookmaker.predictedWinner == "home"
    assert alts.bookmaker.homeWinProbability == pytest.approx(0.68)


def test_store_roundtrips_null_alternatives(store: PredictionStore) -> None:
    pred = PredictionOut(
        matchId=56,
        home=TeamInfo(id=1, name="A"),
        away=TeamInfo(id=2, name="B"),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner="home",
        homeWinProbability=0.7,
        modelVersion="mv",
        featureHash="fh",
        # alternatives intentionally absent
    )
    store.put(2026, 8, [pred])
    loaded = store.get(2026, 8)
    assert loaded[0].alternatives is None


# ---------------------------------------------------------------------------
# compute_predictions — alternatives populated when logistic model present (#140)
# ---------------------------------------------------------------------------


def test_compute_populates_alternatives_when_logistic_model_provided(
    store: PredictionStore, sqlite_repo: SQLiteRepository, tmp_path: Path
) -> None:
    from fantasy_coach.feature_engineering import FEATURE_NAMES, TrainingFrame
    from fantasy_coach.models.logistic import save_model, train_logistic

    raw_match = _load_fixture(UPCOMING_FIXTURE)
    mock_round = MagicMock(return_value=_sample_round_payload("/match/url"))
    mock_match = MagicMock(return_value=raw_match)

    # Primary model (mock — stands in for XGBoost)
    primary_model = _make_mock_model(0.72)
    primary_path = tmp_path / "primary.joblib"
    primary_path.write_bytes(b"primary-bytes")

    # Secondary logistic model (real — needs proper feature_names for bookmaker pick)
    rng = np.random.default_rng(1)
    n = 60
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)
    frame = TrainingFrame(
        X=X,
        y=y,
        match_ids=np.arange(n),
        start_times=np.arange(n, dtype=float),
        feature_names=FEATURE_NAMES,
    )
    logistic_result = train_logistic(frame, test_fraction=0.0)
    logistic_path = tmp_path / "logistic.joblib"
    save_model(logistic_result, logistic_path)

    with patch("fantasy_coach.models.loader.load_model", return_value=primary_model):
        result = compute_predictions(
            2026,
            8,
            sqlite_repo,
            store,
            model_path=primary_path,
            logistic_path=logistic_path,
            fetch_round_fn=mock_round,
            fetch_match_fn=mock_match,
        )

    assert len(result) == 1
    pred = result[0]
    assert pred.alternatives is not None
    # Logistic pick must be present and valid
    assert pred.alternatives.logistic is not None
    assert pred.alternatives.logistic.predictedWinner in ("home", "away")
    assert 0.0 <= pred.alternatives.logistic.homeWinProbability <= 1.0


def test_compute_alternatives_absent_when_same_path(
    store: PredictionStore, sqlite_repo: SQLiteRepository, tmp_path: Path
) -> None:
    """When primary and logistic paths point at the same artefact, skip secondary load."""
    raw_match = _load_fixture(UPCOMING_FIXTURE)
    mock_round = MagicMock(return_value=_sample_round_payload("/match/url"))
    mock_match = MagicMock(return_value=raw_match)
    mock_model = _make_mock_model(0.55)
    same_path = tmp_path / "model.joblib"
    same_path.write_bytes(b"bytes")

    with patch("fantasy_coach.models.loader.load_model", return_value=mock_model):
        result = compute_predictions(
            2026,
            8,
            sqlite_repo,
            store,
            model_path=same_path,
            logistic_path=same_path,  # same as primary
            fetch_round_fn=mock_round,
            fetch_match_fn=mock_match,
        )

    assert len(result) == 1
    # alternatives may be None or only have bookmaker (odds-based) — logistic must not be set
    pred = result[0]
    if pred.alternatives is not None:
        assert pred.alternatives.logistic is None
