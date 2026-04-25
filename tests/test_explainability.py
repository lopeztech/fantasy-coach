"""Tests for models.explainability — TreeSHAP contributions and interactions.

Central correctness invariant (Lundberg & Lee 2017):
    bias + sum(shap_values) == raw_margin    (within floating-point precision)
where raw_margin = logit(predicted_prob) for a binary XGBoost classifier.
"""

from __future__ import annotations

import numpy as np
import pytest


def _build_xgb_loaded(n_features: int = 5, *, seed: int = 0):
    """Return a minimal XGBoost LoadedModel for unit-testing the SHAP path."""
    try:
        from xgboost import XGBClassifier

        from fantasy_coach.models.xgboost_model import LoadedModel
    except Exception:
        pytest.skip("xgboost / libomp not importable")

    rng = np.random.default_rng(seed)
    n = 120
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    feature_names = tuple(f"feat_{i}" for i in range(n_features))
    est = XGBClassifier(
        n_estimators=30,
        max_depth=3,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    est.fit(X, y)
    return LoadedModel(estimator=est, feature_names=feature_names), feature_names


# ---------------------------------------------------------------------------
# shap_contributions — sum invariant
# ---------------------------------------------------------------------------


def test_shap_contributions_returns_per_feature_and_bias() -> None:
    from fantasy_coach.models.explainability import shap_contributions

    loaded, feature_names = _build_xgb_loaded(n_features=5)
    x = np.array([[1.0, -0.5, 0.2, 0.8, -1.2]])

    result = shap_contributions(loaded, x)
    assert result is not None
    per_feature, bias = result
    assert per_feature.shape == (len(feature_names),)
    assert isinstance(bias, float)


def test_shap_sum_equals_raw_margin() -> None:
    """bias + sum(shap) == raw_margin (booster output_margin) within 1e-4.

    Compares against ``output_margin=True`` directly rather than going through
    predict_proba → logit, which introduces a double-precision round-trip error
    large enough to break a tighter tolerance.
    """
    import xgboost as xgb

    from fantasy_coach.models.explainability import shap_contributions

    loaded, feature_names = _build_xgb_loaded(n_features=5)
    booster = loaded.estimator.get_booster()
    rng = np.random.default_rng(42)

    for _ in range(10):
        x_arr = rng.standard_normal((1, 5))
        result = shap_contributions(loaded, x_arr)
        assert result is not None
        per_feature, bias = result

        shap_total = float(per_feature.sum()) + bias

        dmat = xgb.DMatrix(x_arr, feature_names=list(feature_names))
        raw_margin = float(booster.predict(dmat, output_margin=True)[0])

        assert shap_total == pytest.approx(raw_margin, abs=1e-4), (
            f"SHAP sum {shap_total:.6f} != raw margin {raw_margin:.6f}"
        )


def test_shap_contributions_returns_none_for_non_xgboost() -> None:
    from unittest.mock import MagicMock

    from fantasy_coach.models.explainability import shap_contributions

    mock = MagicMock(spec=["feature_names"])  # no .estimator
    assert shap_contributions(mock, np.zeros((1, 5))) is None


def test_shap_contributions_real_feature_names() -> None:
    """Works end-to-end with the production FEATURE_NAMES shape."""
    import xgboost as xgb
    from fantasy_coach.feature_engineering import FEATURE_NAMES
    from fantasy_coach.models.explainability import shap_contributions

    try:
        from xgboost import XGBClassifier

        from fantasy_coach.models.xgboost_model import LoadedModel
    except Exception:
        pytest.skip("xgboost / libomp not importable")

    rng = np.random.default_rng(7)
    n = 100
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)
    est = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    est.fit(X, y)
    loaded = LoadedModel(estimator=est, feature_names=FEATURE_NAMES)

    x = rng.standard_normal((1, len(FEATURE_NAMES)))
    result = shap_contributions(loaded, x)
    assert result is not None
    per_feature, bias = result
    assert per_feature.shape == (len(FEATURE_NAMES),)

    booster = est.get_booster()
    dmat = xgb.DMatrix(x, feature_names=list(FEATURE_NAMES))
    raw_margin = float(booster.predict(dmat, output_margin=True)[0])
    assert float(per_feature.sum()) + bias == pytest.approx(raw_margin, abs=1e-4)


# ---------------------------------------------------------------------------
# shap_interactions
# ---------------------------------------------------------------------------


def test_shap_interactions_returns_top_k_pairs() -> None:
    from fantasy_coach.models.explainability import shap_interactions

    loaded, feature_names = _build_xgb_loaded(n_features=5)
    x = np.array([[1.0, -0.5, 0.2, 0.8, -1.2]])

    result = shap_interactions(loaded, x, top_k=3)
    assert result is not None
    assert len(result) <= 3
    for fi, fj, val in result:
        assert fi in feature_names
        assert fj in feature_names
        assert fi != fj  # off-diagonal only
        assert isinstance(val, float)


def test_shap_interactions_sorted_by_abs_value() -> None:
    from fantasy_coach.models.explainability import shap_interactions

    loaded, _ = _build_xgb_loaded(n_features=6)
    x = np.array([[2.0, -1.5, 0.0, 1.0, -0.5, 0.3]])

    result = shap_interactions(loaded, x, top_k=5)
    assert result is not None
    abs_vals = [abs(v) for _, _, v in result]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_shap_interactions_returns_none_for_non_xgboost() -> None:
    from unittest.mock import MagicMock

    from fantasy_coach.models.explainability import shap_interactions

    mock = MagicMock(spec=["feature_names"])
    assert shap_interactions(mock, np.zeros((1, 5))) is None
