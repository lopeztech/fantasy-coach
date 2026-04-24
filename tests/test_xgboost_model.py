"""Tests for the XGBoost model module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fantasy_coach.feature_engineering import FEATURE_NAMES
from fantasy_coach.models.xgboost_model import (
    MONOTONE_CONSTRAINTS,
    SEASON_WEIGHTS,
    LoadedModel,
    TrainResult,
    _monotone_tuple,
    load_best_params,
    load_model,
    optuna_search,
    recency_weights,
    save_best_params,
    save_model,
    train_xgboost,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(n: int = 80, seed: int = 42):
    """Return a minimal TrainingFrame with `n` rows."""
    from fantasy_coach.feature_engineering import TrainingFrame

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=n)
    base = np.datetime64("2024-01-01", "s")
    start_times = base + np.arange(n) * np.timedelta64(7 * 24 * 3600, "s")
    return TrainingFrame(
        X=X,
        y=y,
        match_ids=np.arange(n),
        start_times=start_times,
    )


# ---------------------------------------------------------------------------
# train_xgboost
# ---------------------------------------------------------------------------


def test_train_xgboost_returns_train_result() -> None:
    frame = _make_frame(80)
    result = train_xgboost(frame, test_fraction=0.2, use_hpo=False)
    assert isinstance(result, TrainResult)
    assert result.n_train == 64
    assert result.n_test == 16


def test_train_xgboost_train_accuracy_reasonable() -> None:
    result = train_xgboost(_make_frame(80), use_hpo=False)
    assert 0.0 <= result.train_accuracy <= 1.0


def test_train_xgboost_test_accuracy_is_nan_when_no_test_set() -> None:
    result = train_xgboost(_make_frame(80), test_fraction=0.0, use_hpo=False)
    assert np.isnan(result.test_accuracy)
    assert result.n_test == 0
    assert result.n_train == 80


def test_train_xgboost_feature_names_preserved() -> None:
    result = train_xgboost(_make_frame(80), use_hpo=False)
    assert result.feature_names == FEATURE_NAMES


def test_train_xgboost_best_params_populated() -> None:
    result = train_xgboost(_make_frame(80), use_hpo=False)
    assert "max_depth" in result.best_params
    assert "n_estimators" in result.best_params
    assert "learning_rate" in result.best_params


def test_train_xgboost_small_dataset_uses_fixed_defaults() -> None:
    # use_hpo=False forces the grid-search fallback; the small-dataset
    # path then uses hardcoded conservative defaults.
    result = train_xgboost(_make_frame(20), test_fraction=0.0, use_hpo=False)
    assert result.best_params["max_depth"] == 3
    assert result.best_params["n_estimators"] == 100


def test_train_xgboost_raises_on_too_few_rows() -> None:
    frame = _make_frame(5)
    with pytest.raises(ValueError, match="Need at least 10 rows"):
        train_xgboost(frame)


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip() -> None:
    result = train_xgboost(_make_frame(80), test_fraction=0.0)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = Path(f.name)

    save_model(result, path)
    loaded = load_model(path)

    assert isinstance(loaded, LoadedModel)
    assert loaded.feature_names == FEATURE_NAMES


def test_loaded_model_predict_shape() -> None:
    result = train_xgboost(_make_frame(80), test_fraction=0.0)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = Path(f.name)

    save_model(result, path)
    loaded = load_model(path)

    X = np.random.default_rng(0).standard_normal((5, len(FEATURE_NAMES)))
    probs = loaded.predict_home_win_prob(X)
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_loaded_model_raises_on_wrong_feature_count() -> None:
    result = train_xgboost(_make_frame(80), test_fraction=0.0)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = Path(f.name)
    save_model(result, path)
    loaded = load_model(path)

    bad_X = np.zeros((1, 5))
    with pytest.raises(ValueError, match="Expected"):
        loaded.predict_home_win_prob(bad_X)


def test_load_model_raises_on_feature_mismatch(tmp_path: Path) -> None:
    import joblib

    bad_blob = {"estimator": None, "feature_names": ("a", "b"), "model_type": "xgboost"}
    joblib.dump(bad_blob, tmp_path / "bad.joblib")
    with pytest.raises(RuntimeError, match="Retrain"):
        load_model(tmp_path / "bad.joblib")


# ---------------------------------------------------------------------------
# XGBoostPredictor (via predictors.py)
# ---------------------------------------------------------------------------


def test_xgboost_predictor_fits_and_predicts() -> None:
    from datetime import UTC, datetime

    from fantasy_coach.evaluation.predictors import XGBoostPredictor
    from fantasy_coach.features import MatchRow, TeamRow

    def _team(tid: int, score: int | None = None) -> TeamRow:
        return TeamRow(team_id=tid, name=f"T{tid}", nick_name=f"T{tid}", score=score, players=[])

    def _match(mid: int, offset_days: int, hscore: int, ascore: int) -> MatchRow:
        from datetime import timedelta

        return MatchRow(
            match_id=mid,
            season=2025,
            round=1,
            start_time=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=offset_days),
            match_state="FullTime",
            venue="V",
            venue_city="C",
            weather=None,
            home=_team(1, hscore),
            away=_team(2, ascore),
            team_stats=[],
        )

    # alternate home wins / away wins to ensure both classes present in every CV fold
    history = [
        _match(i, i * 7, 22 if i % 2 == 0 else 16, 16 if i % 2 == 0 else 22) for i in range(60)
    ]
    predictor = XGBoostPredictor()
    predictor.fit(history)

    upcoming = MatchRow(
        match_id=999,
        season=2025,
        round=20,
        start_time=datetime(2025, 8, 1, tzinfo=UTC),
        match_state="Upcoming",
        venue="V",
        venue_city="C",
        weather=None,
        home=_team(1),
        away=_team(2),
        team_stats=[],
    )
    p = predictor.predict_home_win_prob(upcoming)
    assert 0.0 <= p <= 1.0


def test_xgboost_predictor_fallback_with_no_history() -> None:
    from datetime import UTC, datetime

    from fantasy_coach.evaluation.predictors import XGBoostPredictor
    from fantasy_coach.features import MatchRow, TeamRow

    predictor = XGBoostPredictor()
    predictor.fit([])
    match = MatchRow(
        match_id=1,
        season=2025,
        round=1,
        start_time=datetime(2025, 1, 1, tzinfo=UTC),
        match_state="Upcoming",
        venue="V",
        venue_city="C",
        weather=None,
        home=TeamRow(team_id=1, name="H", nick_name="H", score=None, players=[]),
        away=TeamRow(team_id=2, name="A", nick_name="A", score=None, players=[]),
        team_stats=[],
    )
    assert predictor.predict_home_win_prob(match) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Monotonic constraints (#165)
# ---------------------------------------------------------------------------


def test_monotone_tuple_length_matches_feature_names() -> None:
    assert len(_monotone_tuple()) == len(FEATURE_NAMES)


def test_monotone_tuple_entries_in_valid_set() -> None:
    assert set(_monotone_tuple()).issubset({-1, 0, 1})


def test_monotone_constraints_keys_all_exist_in_feature_names() -> None:
    # Typo guard — a stale key would silently do nothing without this.
    assert set(MONOTONE_CONSTRAINTS).issubset(set(FEATURE_NAMES))


def test_monotone_tuple_aligns_to_feature_order() -> None:
    t = _monotone_tuple()
    for idx, name in enumerate(FEATURE_NAMES):
        expected = MONOTONE_CONSTRAINTS.get(name, 0)
        assert t[idx] == expected, f"{name} mismatched at index {idx}"


def test_train_xgboost_respects_monotone_increasing_constraint() -> None:
    """A feature declared monotone +1 should produce strictly non-decreasing
    predictions as we sweep that feature up with others held constant.
    """
    from fantasy_coach.feature_engineering import TrainingFrame

    # odds_home_win_prob is monotone +1 in MONOTONE_CONSTRAINTS. Build a
    # training set where the label is strongly correlated with that feature
    # but the model is also shown noise that *could* let it learn perverse
    # splits. With the constraint, it cannot.
    rng = np.random.default_rng(0)
    n = 200
    n_feat = len(FEATURE_NAMES)
    odds_idx = FEATURE_NAMES.index("odds_home_win_prob")

    X = rng.standard_normal((n, n_feat))
    odds = rng.uniform(0.1, 0.9, size=n)
    X[:, odds_idx] = odds
    # Label = 1 roughly when odds > 0.5, with some noise.
    y = (odds > 0.5).astype(int)
    # Force a cluster of ``odds=0.6, label=0`` examples — the spurious
    # pattern XGBoost learned on the R8 Tigers/Raiders match. Without
    # the constraint the model would happily carve out a negative split
    # around odds=0.6; with it, it cannot.
    flip_mask = (odds > 0.55) & (odds < 0.65)
    y[flip_mask] = 0

    base = np.datetime64("2024-01-01", "s")
    start_times = base + np.arange(n) * np.timedelta64(7 * 24 * 3600, "s")
    frame = TrainingFrame(
        X=X,
        y=y,
        match_ids=np.arange(n),
        start_times=start_times,
    )

    result = train_xgboost(frame, test_fraction=0.0)
    est = result.estimator

    # Sweep odds_home_win_prob from 0.1 → 0.9 with everything else at 0.
    sweep = np.linspace(0.1, 0.9, 17)
    probe = np.zeros((len(sweep), n_feat))
    probe[:, odds_idx] = sweep
    probs = est.predict_proba(probe)[:, 1]

    # Monotone non-decreasing — strict-enough after XGBoost rounds to
    # same-leaf identity in flat regions, so we use <= not <.
    diffs = np.diff(probs)
    assert (diffs >= -1e-9).all(), (
        f"constraint violated: probs={probs.tolist()} diffs={diffs.tolist()}"
    )


# ---------------------------------------------------------------------------
# Recency weighting (#167)
# ---------------------------------------------------------------------------


def test_recency_weights_maps_years_to_multipliers() -> None:
    times = np.array(["2024-03-01", "2025-03-01", "2026-03-01"], dtype="datetime64[s]")
    w = recency_weights(times)
    assert w.tolist() == [
        SEASON_WEIGHTS[2024],
        SEASON_WEIGHTS[2025],
        SEASON_WEIGHTS[2026],
    ]


def test_recency_weights_unknown_season_defaults_to_one() -> None:
    times = np.array(["2019-03-01"], dtype="datetime64[s]")
    assert recency_weights(times).tolist() == [1.0]


def test_recency_weights_accepts_custom_mapping() -> None:
    times = np.array(["2026-03-01"], dtype="datetime64[s]")
    assert recency_weights(times, season_weights={2026: 7.5}).tolist() == [7.5]


def test_recency_weights_length_matches_input() -> None:
    times = np.array(
        ["2024-01-01", "2024-06-01", "2025-06-01", "2026-01-01"], dtype="datetime64[s]"
    )
    assert recency_weights(times).shape == (4,)


# ---------------------------------------------------------------------------
# best_params persistence (#167)
# ---------------------------------------------------------------------------


def test_save_and_load_best_params_roundtrip(tmp_path: Path) -> None:
    params = {"max_depth": 5, "learning_rate": 0.07, "n_estimators": 450}
    target = tmp_path / "best_params.json"
    save_best_params(params, target)
    loaded = load_best_params(target)
    assert loaded == params


def test_load_best_params_missing_returns_none(tmp_path: Path) -> None:
    assert load_best_params(tmp_path / "nope.json") is None


def test_train_xgboost_uses_best_params_when_provided() -> None:
    frame = _make_frame(80)
    best = {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.1}
    result = train_xgboost(frame, test_fraction=0.0, best_params=best)
    # best_params flows through to result.
    assert result.best_params["max_depth"] == 3
    assert result.best_params["n_estimators"] == 50
    assert result.best_params["learning_rate"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Optuna search (#167) — smoke test, not a full CV run
# ---------------------------------------------------------------------------


def test_optuna_search_returns_best_params() -> None:
    pytest.importorskip("optuna")
    frame = _make_frame(80)
    best = optuna_search(frame, n_trials=3, random_state=0)
    # Every search-space key must appear in the output.
    for key in (
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_weight",
        "gamma",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ):
        assert key in best


def test_optuna_search_raises_when_dataset_too_small() -> None:
    pytest.importorskip("optuna")
    frame = _make_frame(20)
    with pytest.raises(ValueError, match="at least"):
        optuna_search(frame, n_trials=2)


def test_optuna_search_output_trainable_back_through_train_xgboost() -> None:
    pytest.importorskip("optuna")
    # End-to-end — HPO → feed best back to train_xgboost → predict shape OK.
    frame = _make_frame(80)
    best = optuna_search(frame, n_trials=3, random_state=0)
    result = train_xgboost(frame, test_fraction=0.2, best_params=best)
    assert result.n_train == 64
    probs = result.estimator.predict_proba(frame.X[:5])
    assert probs.shape == (5, 2)
