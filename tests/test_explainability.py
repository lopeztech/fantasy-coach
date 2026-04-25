"""Tests for SHAP-based feature attributions in models/explainability.py.

The central correctness invariant for TreeSHAP (Lundberg & Lee 2017):

    sum(shap_contributions(model, x)) + shap_bias(model, x) == raw_log_odds

This must hold within floating-point tolerance for every input.
"""

from __future__ import annotations

import numpy as np
import pytest

from fantasy_coach.feature_engineering import FEATURE_NAMES


@pytest.fixture(scope="module")
def xgb_loaded():
    """Small trained XGBoost artefact using the full FEATURE_NAMES shape."""
    try:
        from xgboost import XGBClassifier

        from fantasy_coach.models.xgboost_model import LoadedModel
    except Exception:
        pytest.skip("xgboost / libomp not importable")

    rng = np.random.default_rng(42)
    n = 150
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    est = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", verbosity=0)
    est.fit(X, y)
    return LoadedModel(estimator=est, feature_names=FEATURE_NAMES)


@pytest.fixture(scope="module")
def sample_x():
    rng = np.random.default_rng(7)
    return rng.standard_normal((1, len(FEATURE_NAMES)))


# ---------------------------------------------------------------------------
# shap_contributions
# ---------------------------------------------------------------------------


def test_shap_contributions_returns_feature_length_array(xgb_loaded, sample_x) -> None:
    from fantasy_coach.models.explainability import shap_contributions

    result = shap_contributions(xgb_loaded, sample_x)
    assert result is not None
    assert result.shape == (len(FEATURE_NAMES),)


def test_shap_contributions_returns_none_for_logistic(sample_x) -> None:
    from fantasy_coach.models.explainability import shap_contributions
    from fantasy_coach.models.logistic import LoadedModel

    # A logistic LoadedModel has no .estimator with get_booster()
    mock = LoadedModel.__new__(LoadedModel)
    result = shap_contributions(mock, sample_x)
    assert result is None


# ---------------------------------------------------------------------------
# shap_bias
# ---------------------------------------------------------------------------


def test_shap_bias_returns_float(xgb_loaded, sample_x) -> None:
    from fantasy_coach.models.explainability import shap_bias

    result = shap_bias(xgb_loaded, sample_x)
    assert result is not None
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Sum invariant — the central correctness guarantee
# ---------------------------------------------------------------------------


def test_shap_sum_invariant(xgb_loaded, sample_x) -> None:
    """sum(shap_contributions) + shap_bias == raw_log_odds exactly."""
    import xgboost as xgb

    from fantasy_coach.models.explainability import shap_bias, shap_contributions

    contribs = shap_contributions(xgb_loaded, sample_x)
    bias = shap_bias(xgb_loaded, sample_x)
    assert contribs is not None and bias is not None

    # Compute the raw margin (log-odds before sigmoid) directly.
    booster = xgb_loaded.estimator.get_booster()
    dmatrix = xgb.DMatrix(
        np.asarray(sample_x, dtype=float),
        feature_names=list(FEATURE_NAMES),
    )
    raw_margin = float(booster.predict(dmatrix, output_margin=True)[0])

    shap_total = float(np.sum(contribs)) + bias
    assert shap_total == pytest.approx(raw_margin, abs=1e-5)


def test_shap_sum_invariant_multiple_samples(xgb_loaded) -> None:
    """Sum invariant holds for 10 different random inputs."""
    import xgboost as xgb

    from fantasy_coach.models.explainability import shap_bias, shap_contributions

    rng = np.random.default_rng(99)
    booster = xgb_loaded.estimator.get_booster()

    for _ in range(10):
        x = rng.standard_normal((1, len(FEATURE_NAMES)))
        contribs = shap_contributions(xgb_loaded, x)
        bias = shap_bias(xgb_loaded, x)
        assert contribs is not None and bias is not None

        dmatrix = xgb.DMatrix(np.asarray(x, dtype=float), feature_names=list(FEATURE_NAMES))
        raw_margin = float(booster.predict(dmatrix, output_margin=True)[0])
        shap_total = float(np.sum(contribs)) + bias
        assert shap_total == pytest.approx(raw_margin, abs=1e-5)


# ---------------------------------------------------------------------------
# shap_interactions
# ---------------------------------------------------------------------------


def test_shap_interactions_returns_dict_with_partners(xgb_loaded, sample_x) -> None:
    from fantasy_coach.models.explainability import shap_interactions

    result = shap_interactions(xgb_loaded, sample_x)
    # May be None if all interaction magnitudes are below threshold;
    # on a trained model with signal this should not happen.
    if result is not None:
        for feature_name, (partner, magnitude) in result.items():
            assert feature_name in FEATURE_NAMES
            assert partner in FEATURE_NAMES
            assert isinstance(magnitude, float)


def test_shap_interactions_partner_differs_from_feature(xgb_loaded, sample_x) -> None:
    from fantasy_coach.models.explainability import shap_interactions

    result = shap_interactions(xgb_loaded, sample_x)
    if result is not None:
        for name, (partner, _) in result.items():
            assert name != partner, f"Feature {name} should not interact with itself"


def test_shap_interactions_returns_none_for_non_xgboost(sample_x) -> None:
    from unittest.mock import MagicMock

    from fantasy_coach.models.explainability import shap_interactions

    mock = MagicMock(spec=[])  # no .estimator attribute
    result = shap_interactions(mock, sample_x)
    assert result is None


# ---------------------------------------------------------------------------
# Round-trip: predict via Booster, predict via SHAP-summing — assert equal
# ---------------------------------------------------------------------------


def test_round_trip_predict_equals_shap_sum(xgb_loaded, sample_x) -> None:
    """Booster.predict(output_margin=False) == sigmoid(sum(SHAP) + bias)."""
    import xgboost as xgb

    from fantasy_coach.models.explainability import shap_bias, shap_contributions

    contribs = shap_contributions(xgb_loaded, sample_x)
    bias = shap_bias(xgb_loaded, sample_x)
    assert contribs is not None and bias is not None

    shap_log_odds = float(np.sum(contribs)) + bias
    shap_prob = 1.0 / (1.0 + np.exp(-shap_log_odds))

    booster = xgb_loaded.estimator.get_booster()
    dmatrix = xgb.DMatrix(np.asarray(sample_x, dtype=float), feature_names=list(FEATURE_NAMES))
    direct_prob = float(booster.predict(dmatrix)[0])

    assert shap_prob == pytest.approx(direct_prob, abs=1e-5)
