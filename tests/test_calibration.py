"""Tests for the probability calibration module."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fantasy_coach.models.calibration import (
    CalibrationWrapper,
    ece,
    reliability_bins,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(random_state=0))])


def _synthetic_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    # Simple linearly-separable target so the pipeline can actually fit.
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# CalibrationWrapper
# ---------------------------------------------------------------------------


def test_calibration_wrapper_platt_predict_before_fit():
    """Before .fit(), wrapper should fall back to the raw pipeline."""
    X, y = _synthetic_data()
    pipe = _make_pipeline()
    pipe.fit(X, y)
    wrapper = CalibrationWrapper(pipe, method="platt")
    assert not wrapper.is_fitted
    raw = pipe.predict_proba(X)[:, 1]
    wrapped = wrapper.predict_home_win_prob(X)
    np.testing.assert_array_equal(raw, wrapped)


def test_calibration_wrapper_platt_fit_and_predict():
    """After fitting with Platt, predictions should be in [0, 1]."""
    X, y = _synthetic_data(n=300)
    pipe = _make_pipeline()
    split = 200
    pipe.fit(X[:split], y[:split])
    wrapper = CalibrationWrapper(pipe, method="platt")
    wrapper.fit(X[split:], y[split:])
    assert wrapper.is_fitted
    probs = wrapper.predict_home_win_prob(X[split:])
    assert probs.shape == (100,)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_calibration_wrapper_isotonic_fit_and_predict():
    """After fitting with isotonic, predictions should be in [0, 1]."""
    X, y = _synthetic_data(n=300)
    pipe = _make_pipeline()
    split = 200
    pipe.fit(X[:split], y[:split])
    wrapper = CalibrationWrapper(pipe, method="isotonic")
    wrapper.fit(X[split:], y[split:])
    assert wrapper.is_fitted
    probs = wrapper.predict_home_win_prob(X[split:])
    assert probs.shape == (100,)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_calibration_wrapper_fit_is_fitted():
    X, y = _synthetic_data(n=300)
    pipe = _make_pipeline()
    pipe.fit(X[:200], y[:200])
    wrapper = CalibrationWrapper(pipe)
    assert not wrapper.is_fitted
    wrapper.fit(X[200:], y[200:])
    assert wrapper.is_fitted


def test_calibration_wrapper_platt_reduces_ece_on_overconfident_model():
    """Platt calibration should reduce ECE relative to raw probabilities."""
    X, y = _synthetic_data(n=500, seed=42)
    pipe = _make_pipeline()
    split_train, split_cal = 300, 400
    pipe.fit(X[:split_train], y[:split_train])

    raw_probs = pipe.predict_proba(X[split_cal:])[:, 1].tolist()
    actuals = y[split_cal:].tolist()
    raw_ece = ece(raw_probs, actuals)

    wrapper = CalibrationWrapper(pipe, method="platt")
    wrapper.fit(X[split_train:split_cal], y[split_train:split_cal])
    cal_probs = wrapper.predict_home_win_prob(X[split_cal:]).tolist()
    cal_ece = ece(cal_probs, actuals)

    # Calibrated ECE should be <= raw (not always strictly lower with tiny
    # datasets, but should never be materially worse — allow small tolerance).
    assert cal_ece <= raw_ece + 0.05


# ---------------------------------------------------------------------------
# ece()
# ---------------------------------------------------------------------------


def test_ece_empty():
    assert ece([], []) != ece([], [])  # NaN != NaN


def test_ece_perfect_calibration():
    # Perfect calibration: confidence == actual rate in every bin.
    probs = [0.1] * 10 + [0.5] * 10 + [0.9] * 10
    # Match outcomes at the exact stated probability.
    rng = np.random.default_rng(0)
    outcomes = []
    for p in probs:
        outcomes.append(int(rng.random() < p))
    # ECE can't be exactly 0 with stochastic outcomes, but should be < 0.2.
    result = ece(probs, outcomes)
    assert 0.0 <= result <= 0.5


def test_ece_overconfident():
    # All probabilities near 1 but only 50% correct → high ECE.
    probs = [0.95] * 20
    outcomes = [1, 0] * 10
    result = ece(probs, outcomes)
    # Expected: |0.5 - 0.95| * 1.0 ≈ 0.45
    assert result > 0.3


def test_ece_underconfident():
    # All probabilities near 0.5 but actual rate is 90% → high ECE.
    probs = [0.5] * 20
    outcomes = [1] * 18 + [0] * 2
    result = ece(probs, outcomes)
    # Expected: |0.9 - 0.5| = 0.4
    assert result > 0.3


def test_ece_single_bin():
    probs = [0.6, 0.6, 0.6, 0.6]
    outcomes = [1, 1, 0, 0]
    result = ece(probs, outcomes)
    # One bin: |0.5 - 0.6| * 1 = 0.1
    assert abs(result - 0.1) < 0.01


# ---------------------------------------------------------------------------
# reliability_bins()
# ---------------------------------------------------------------------------


def test_reliability_bins_count():
    probs = list(np.linspace(0.05, 0.95, 50))
    outcomes = [1 if p > 0.5 else 0 for p in probs]
    bins = reliability_bins(probs, outcomes, n_bins=10)
    assert len(bins) == 10


def test_reliability_bins_empty_bin():
    # All predictions in [0.9, 1.0] — first 9 bins should be empty.
    probs = [0.95] * 20
    outcomes = [1] * 20
    bins = reliability_bins(probs, outcomes, n_bins=10)
    for b in bins[:9]:
        assert b["mean_confidence"] is None
        assert b["mean_accuracy"] is None
        assert b["n"] == 0
    assert bins[9]["n"] == 20


def test_reliability_bins_values():
    probs = [0.15, 0.15, 0.15, 0.15]  # all in [0.1, 0.2] bin
    outcomes = [1, 1, 0, 0]
    bins = reliability_bins(probs, outcomes, n_bins=10)
    populated = [b for b in bins if b["n"] > 0]
    assert len(populated) == 1
    b = populated[0]
    assert abs(b["mean_confidence"] - 0.15) < 1e-9
    assert abs(b["mean_accuracy"] - 0.5) < 1e-9
    assert b["n"] == 4


# ---------------------------------------------------------------------------
# save / load round-trip via logistic.save_model / load_model
# ---------------------------------------------------------------------------


def test_save_load_with_calibrator(tmp_path):
    from fantasy_coach.feature_engineering import TrainingFrame
    from fantasy_coach.models.logistic import LoadedModel, load_model, save_model, train_logistic

    rng = np.random.default_rng(0)
    n = 60
    X = rng.standard_normal((n, 6))
    y = (X[:, 0] > 0).astype(int)
    base = np.datetime64("2024-01-01", "s")
    start_times = np.array(
        [base + np.timedelta64(i * 7, "D") for i in range(n)], dtype="datetime64[s]"
    )
    match_ids = np.arange(n)

    from fantasy_coach.feature_engineering import FEATURE_NAMES

    frame = TrainingFrame(
        X=X, y=y, match_ids=match_ids, start_times=start_times, feature_names=FEATURE_NAMES
    )
    result = train_logistic(frame, test_fraction=0.0)

    cal_split = int(n * 0.8)
    wrapper = CalibrationWrapper(result.pipeline, method="platt")
    wrapper.fit(X[cal_split:], y[cal_split:])

    path = tmp_path / "logistic.joblib"
    save_model(result, path, calibration_wrapper=wrapper)

    loaded: LoadedModel = load_model(path)
    assert loaded.calibration_wrapper is not None
    assert loaded.calibration_wrapper.is_fitted

    # Predictions via LoadedModel should use the calibrator.
    probs = loaded.predict_home_win_prob(X)
    assert probs.shape == (n,)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_save_load_without_calibrator(tmp_path):
    from fantasy_coach.feature_engineering import FEATURE_NAMES, TrainingFrame
    from fantasy_coach.models.logistic import load_model, save_model, train_logistic

    rng = np.random.default_rng(1)
    n = 30
    X = rng.standard_normal((n, 6))
    y = (X[:, 0] > 0).astype(int)
    base = np.datetime64("2024-01-01", "s")
    start_times = np.array(
        [base + np.timedelta64(i * 7, "D") for i in range(n)], dtype="datetime64[s]"
    )
    frame = TrainingFrame(
        X=X, y=y, match_ids=np.arange(n), start_times=start_times, feature_names=FEATURE_NAMES
    )
    result = train_logistic(frame, test_fraction=0.0)

    path = tmp_path / "logistic_no_cal.joblib"
    save_model(result, path)

    loaded = load_model(path)
    assert loaded.calibration_wrapper is None

    probs = loaded.predict_home_win_prob(X)
    assert probs.shape == (n,)
