"""SHAP-based feature attributions and interaction detection for tree models.

Provides exact TreeSHAP attributions for the production XGBoost artefact.
For binary classification the contributions satisfy:

    sum(shap_contributions(loaded, x)) + shap_bias(loaded, x) == raw_margin

where ``raw_margin`` is the booster's output before the sigmoid — i.e. the
log-odds.  This makes the "Why this pick" panel mathematically consistent with
the displayed probability, which the previous logistic-coefficient approximation
could not guarantee for tree-based models.

No extra dependency — XGBoost ships TreeSHAP natively via
``Booster.predict(pred_contribs=True)`` and
``Booster.predict(pred_interactions=True)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def shap_contributions(loaded: Any, x: Any) -> np.ndarray | None:
    """Return per-feature SHAP contributions for one sample from an XGBoost artefact.

    Uses ``Booster.predict(pred_contribs=True)`` which returns n_features+1
    values per sample; the last value is the bias/intercept.  Returns only the
    n_features slice so the result aligns with ``loaded.feature_names``.

    Returns ``None`` when ``loaded`` is not an XGBoost artefact or xgboost is
    unavailable (e.g. libomp missing on macOS CI).
    """
    booster = _get_booster(loaded)
    if booster is None:
        return None

    feature_names = getattr(loaded, "feature_names", None)
    try:
        import xgboost as xgb  # noqa: PLC0415

        dmatrix = xgb.DMatrix(
            np.asarray(x, dtype=float),
            feature_names=list(feature_names) if feature_names else None,
        )
        contribs = booster.predict(dmatrix, pred_contribs=True)
    except Exception:
        return None

    arr = np.asarray(contribs)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return None
    return arr[0, :-1]  # drop the bias column


def shap_bias(loaded: Any, x: Any) -> float | None:
    """Return the SHAP bias term for one sample.

    Together with ``shap_contributions`` it satisfies the sum invariant:
        sum(shap_contributions(loaded, x)) + shap_bias(loaded, x) == raw_margin
    """
    booster = _get_booster(loaded)
    if booster is None:
        return None

    feature_names = getattr(loaded, "feature_names", None)
    try:
        import xgboost as xgb  # noqa: PLC0415

        dmatrix = xgb.DMatrix(
            np.asarray(x, dtype=float),
            feature_names=list(feature_names) if feature_names else None,
        )
        contribs = booster.predict(dmatrix, pred_contribs=True)
    except Exception:
        return None

    arr = np.asarray(contribs)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return None
    return float(arr[0, -1])


def shap_interactions(
    loaded: Any,
    x: Any,
    *,
    min_magnitude: float = 1e-4,
) -> dict[str, tuple[str, float]] | None:
    """Return the dominant interaction partner per feature for an XGBoost model.

    Uses ``Booster.predict(pred_interactions=True)`` which returns an
    (n_features+1) × (n_features+1) interaction matrix per sample.  For each
    feature i the top partner is ``argmax_{j ≠ i} |interaction[i, j]|``.

    Returns a dict mapping ``feature_name → (partner_name, magnitude)``, or
    ``None`` when the model isn't XGBoost / xgboost is unavailable / no
    interaction exceeds ``min_magnitude``.

    Used by the #150 what-if explorer for interaction-aware sensitivity.
    The ``/predictions`` endpoint calls this for tree-based artefacts to
    enrich ``FeatureContribution.detail`` with the dominant interaction partner.
    """
    booster = _get_booster(loaded)
    if booster is None:
        return None

    feature_names = getattr(loaded, "feature_names", None)
    if not feature_names:
        return None

    try:
        import xgboost as xgb  # noqa: PLC0415

        dmatrix = xgb.DMatrix(
            np.asarray(x, dtype=float),
            feature_names=list(feature_names),
        )
        raw = booster.predict(dmatrix, pred_interactions=True)
    except Exception:
        return None

    arr = np.asarray(raw)
    if arr.ndim != 3 or arr.shape[0] < 1:
        return None

    n = len(feature_names)
    mat = arr[0, :n, :n]  # drop bias row/col

    result: dict[str, tuple[str, float]] = {}
    for i, name in enumerate(feature_names):
        row = mat[i].copy()
        row[i] = 0.0  # exclude self-interaction (diagonal = main effect)
        j = int(np.argmax(np.abs(row)))
        magnitude = float(row[j])
        if abs(magnitude) >= min_magnitude:
            result[name] = (feature_names[j], magnitude)

    return result if result else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_booster(loaded: Any) -> Any | None:
    """Extract the XGBoost ``Booster`` from a loaded artefact, or return None."""
    estimator = getattr(loaded, "estimator", None)
    if estimator is None:
        return None
    get_booster = getattr(estimator, "get_booster", None)
    if not callable(get_booster):
        return None
    try:
        import xgboost  # noqa: PLC0415, F401
    except Exception:
        return None
    try:
        return get_booster()
    except Exception:
        return None
