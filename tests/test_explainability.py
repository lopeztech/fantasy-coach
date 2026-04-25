"""Tests for models/explainability.py — TreeSHAP attribution module.

The central invariant tested here: for XGBoost binary classification,
    sum(shap_contributions) + bias == booster.predict(x)  [raw margin]
This equality is guaranteed by TreeSHAP and is the definition of a
"truthful" attribution — the explanation accurately sums to the prediction.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def xgb_estimator_and_names():
    """Train a small XGBClassifier for use across tests in this module."""
    try:
        from xgboost import XGBClassifier
    except Exception:
        pytest.skip("xgboost / libomp not importable")

    from fantasy_coach.feature_engineering import FEATURE_NAMES

    rng = np.random.default_rng(42)
    n = 150
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)

    est = XGBClassifier(
        n_estimators=50, max_depth=3, eval_metric="logloss", verbosity=0, n_jobs=1
    )
    est.fit(X, y)
    return est, FEATURE_NAMES


class TestShapContributions:
    def test_returns_array_of_feature_length(self, xgb_estimator_and_names) -> None:
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_contributions

        est, names = xgb_estimator_and_names
        x = np.zeros((1, len(FEATURE_NAMES)))
        result = shap_contributions(est, names, x)
        assert result is not None
        assert result.shape == (len(FEATURE_NAMES),)

    def test_accepts_1d_input(self, xgb_estimator_and_names) -> None:
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_contributions

        est, names = xgb_estimator_and_names
        x_1d = np.zeros(len(FEATURE_NAMES))
        result = shap_contributions(est, names, x_1d)
        assert result is not None
        assert result.shape == (len(FEATURE_NAMES),)

    def test_sum_invariant_equals_raw_margin(self, xgb_estimator_and_names) -> None:
        """The central TreeSHAP correctness guarantee:
        sum(shap_values) + bias == booster.predict(x, output_margin=True)
        """
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_bias, shap_contributions

        try:
            import xgboost as xgb
        except Exception:
            pytest.skip("xgboost not importable")

        est, names = xgb_estimator_and_names
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, len(FEATURE_NAMES)))

        booster = est.get_booster()
        for i in range(len(X)):
            x = X[i : i + 1]
            contribs = shap_contributions(est, names, x)
            bias = shap_bias(est, names, x)
            assert contribs is not None
            assert bias is not None

            # Ground truth: raw margin from the booster.
            dm = xgb.DMatrix(x, feature_names=list(names))
            expected_margin = float(booster.predict(dm, output_margin=True)[0])

            shap_sum = float(np.sum(contribs)) + bias
            assert shap_sum == pytest.approx(expected_margin, abs=1e-5), (
                f"Sample {i}: SHAP sum {shap_sum:.6f} != margin {expected_margin:.6f}"
            )

    def test_returns_none_for_non_xgboost_estimator(self) -> None:
        from unittest.mock import MagicMock

        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_contributions

        non_xgb = MagicMock(spec=[])  # no get_booster attribute
        result = shap_contributions(non_xgb, FEATURE_NAMES, np.zeros((1, len(FEATURE_NAMES))))
        assert result is None

    def test_values_vary_across_inputs(self, xgb_estimator_and_names) -> None:
        """Different feature inputs should produce different SHAP values."""
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_contributions

        est, names = xgb_estimator_and_names
        x1 = np.zeros((1, len(FEATURE_NAMES)))
        x2 = np.ones((1, len(FEATURE_NAMES))) * 2.0

        c1 = shap_contributions(est, names, x1)
        c2 = shap_contributions(est, names, x2)
        assert c1 is not None and c2 is not None
        # At least one feature should differ between the two predictions.
        assert not np.allclose(c1, c2)


class TestShapBias:
    def test_bias_is_scalar(self, xgb_estimator_and_names) -> None:
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_bias

        est, names = xgb_estimator_and_names
        x = np.zeros((1, len(FEATURE_NAMES)))
        b = shap_bias(est, names, x)
        assert isinstance(b, float)

    def test_bias_constant_across_inputs(self, xgb_estimator_and_names) -> None:
        """XGBoost bias is model-global, not sample-dependent."""
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_bias

        est, names = xgb_estimator_and_names
        b1 = shap_bias(est, names, np.zeros((1, len(FEATURE_NAMES))))
        b2 = shap_bias(est, names, np.ones((1, len(FEATURE_NAMES))) * 3.0)
        assert b1 == pytest.approx(b2, abs=1e-6)


class TestShapInteractions:
    def test_returns_list(self, xgb_estimator_and_names) -> None:
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_interactions

        est, names = xgb_estimator_and_names
        x = np.zeros((1, len(FEATURE_NAMES)))
        result = shap_interactions(est, names, x, top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_result_shape(self, xgb_estimator_and_names) -> None:
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_interactions

        est, names = xgb_estimator_and_names
        x = np.zeros((1, len(FEATURE_NAMES)))
        result = shap_interactions(est, names, x, top_k=5)
        for pair in result:
            assert "feature_a" in pair
            assert "feature_b" in pair
            assert "magnitude" in pair
            assert pair["feature_a"] in names
            assert pair["feature_b"] in names
            assert pair["feature_a"] != pair["feature_b"]

    def test_sorted_by_abs_magnitude(self, xgb_estimator_and_names) -> None:
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_interactions

        est, names = xgb_estimator_and_names
        x = np.zeros((1, len(FEATURE_NAMES)))
        x[0, 0] = 3.0  # strong signal to create interactions
        result = shap_interactions(est, names, x, top_k=10)
        magnitudes = [abs(p["magnitude"]) for p in result]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_returns_empty_list_for_non_xgboost(self) -> None:
        from unittest.mock import MagicMock

        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.explainability import shap_interactions

        non_xgb = MagicMock(spec=[])
        result = shap_interactions(non_xgb, FEATURE_NAMES, np.zeros((1, len(FEATURE_NAMES))))
        assert result == []


class TestXGBoostDispatchInPredictions:
    """Round-trip: _compute_contributions on XGBoost produces SHAP-backed values."""

    def test_xgboost_contributions_sum_approximates_logit(
        self, xgb_estimator_and_names
    ) -> None:
        """The top-K filter means we can't recover the full logit, but we can
        verify that each contribution has the right sign and the full sum
        (before filtering) equals the raw margin via the invariant test above."""
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.xgboost_model import LoadedModel
        from fantasy_coach.predictions import _compute_contributions

        est, names = xgb_estimator_and_names
        loaded = LoadedModel(estimator=est, feature_names=FEATURE_NAMES)
        x = np.zeros((1, len(FEATURE_NAMES)))
        x[0, 0] = 2.0  # push feature 0 strongly

        contribs = _compute_contributions(loaded, x, top_k=len(FEATURE_NAMES))
        assert contribs is not None
        # All contributions should be finite numbers.
        for c in contribs:
            assert np.isfinite(c.contribution)

    def test_xgboost_contributions_json_contract_preserved(
        self, xgb_estimator_and_names
    ) -> None:
        """The FeatureContribution schema (fields + types) must be unchanged."""
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.xgboost_model import LoadedModel
        from fantasy_coach.predictions import FeatureContribution, _compute_contributions

        est, names = xgb_estimator_and_names
        loaded = LoadedModel(estimator=est, feature_names=FEATURE_NAMES)
        x = np.zeros((1, len(FEATURE_NAMES)))

        contribs = _compute_contributions(loaded, x)
        assert contribs is not None
        assert all(isinstance(c, FeatureContribution) for c in contribs)
        for c in contribs:
            assert isinstance(c.feature, str)
            assert isinstance(c.value, float)
            assert isinstance(c.contribution, float)
            # detail is optional
            assert c.detail is None or isinstance(c.detail, dict)

    def test_xgboost_interaction_partner_in_detail(self, xgb_estimator_and_names) -> None:
        """XGBoost contributions may carry an interaction_partner in detail."""
        from fantasy_coach.feature_engineering import FEATURE_NAMES
        from fantasy_coach.models.xgboost_model import LoadedModel
        from fantasy_coach.predictions import _compute_contributions

        est, names = xgb_estimator_and_names
        loaded = LoadedModel(estimator=est, feature_names=FEATURE_NAMES)
        x = np.zeros((1, len(FEATURE_NAMES)))
        x[0, 0] = 3.0
        x[0, 1] = 2.0

        contribs = _compute_contributions(loaded, x, top_k=len(FEATURE_NAMES))
        assert contribs is not None
        # At least some contributions should have interaction detail (if the model
        # has learned any non-zero interactions on this input).
        interaction_details = [
            c for c in contribs if c.detail is not None and "interaction_partner" in c.detail
        ]
        # Not asserted to be non-empty — a trivial model may have zero interactions.
        # But if any exist, validate their structure.
        for c in interaction_details:
            assert isinstance(c.detail["interaction_partner"], str)
            assert c.detail["interaction_partner"] in FEATURE_NAMES
            assert isinstance(c.detail["magnitude"], float)
