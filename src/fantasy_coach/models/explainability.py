"""Model-aware feature attribution (explainability) for production models.

Dispatches to the appropriate attribution method based on model type:
- Logistic: linear  ``coef × (x − mean) / scale`` (handled in predictions.py).
- XGBoost / gradient_boosting: TreeSHAP via
  ``xgboost.Booster.predict(pred_contribs=True)``.

The central correctness invariant for XGBoost TreeSHAP is:
    sum(shap_values) + bias == booster.predict(x)  [raw margin / log-odds]
This is guaranteed by the XGBoost implementation and verified in tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def shap_contributions(
    estimator: Any,
    feature_names: tuple[str, ...],
    x: np.ndarray,
) -> np.ndarray | None:
    """Return per-feature TreeSHAP contributions for one sample.

    ``x`` must have shape ``(1, n_features)`` or ``(n_features,)``.
    Returns a 1-D numpy array of length ``n_features`` in the same order as
    ``feature_names``, or ``None`` if the estimator isn't an XGBoost model or
    XGBoost isn't importable (macOS without libomp safety net).

    The values are in raw-margin (log-odds for binary classification) units.
    The sum of all values plus the model's bias equals the model's raw output
    for this sample — that invariant is the basis for truthful attributions.
    """
    try:
        import xgboost as xgb  # noqa: PLC0415
    except Exception:  # pragma: no cover — libomp-missing safety net
        return None

    get_booster = getattr(estimator, "get_booster", None)
    if not callable(get_booster):
        return None
    try:
        booster = get_booster()
    except Exception:
        return None

    try:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        dmatrix = xgb.DMatrix(arr, feature_names=list(feature_names))
        contribs = booster.predict(dmatrix, pred_contribs=True)
    except Exception:
        return None

    result = np.asarray(contribs)
    if result.ndim != 2 or result.shape[0] < 1:
        return None
    # Last column is the global bias — drop to align with feature_names.
    return result[0, :-1]


def shap_bias(
    estimator: Any,
    feature_names: tuple[str, ...],
    x: np.ndarray,
) -> float | None:
    """Return the model bias (intercept contribution) for one sample.

    This is the last column from ``pred_contribs=True``.  Combined with
    ``shap_contributions``, the sum equals the raw margin: bias + sum(shap) == margin.
    Returns ``None`` on any failure (mirrors ``shap_contributions``).
    """
    try:
        import xgboost as xgb  # noqa: PLC0415
    except Exception:  # pragma: no cover
        return None

    get_booster = getattr(estimator, "get_booster", None)
    if not callable(get_booster):
        return None
    try:
        booster = get_booster()
    except Exception:
        return None

    try:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        dmatrix = xgb.DMatrix(arr, feature_names=list(feature_names))
        contribs = booster.predict(dmatrix, pred_contribs=True)
    except Exception:
        return None

    result = np.asarray(contribs)
    if result.ndim != 2 or result.shape[0] < 1:
        return None
    return float(result[0, -1])


def shap_interactions(
    estimator: Any,
    feature_names: tuple[str, ...],
    x: np.ndarray,
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return the top-K pairwise SHAP interaction pairs for one sample.

    Uses ``xgboost.Booster.predict(pred_interactions=True)`` which returns
    a symmetric matrix of shape ``(n_features+1, n_features+1)``.  Off-diagonal
    entries ``[i, j]`` represent the interaction effect between features i and
    j; diagonal entries are the main effects.

    Returns a list of dicts (sorted by descending absolute magnitude):
        {
            "feature_a": str,
            "feature_b": str,
            "magnitude": float,  # log-odds interaction contribution
        }

    Returns an empty list on any failure so callers don't need to handle None.
    """
    try:
        import xgboost as xgb  # noqa: PLC0415
    except Exception:  # pragma: no cover
        return []

    get_booster = getattr(estimator, "get_booster", None)
    if not callable(get_booster):
        return []
    try:
        booster = get_booster()
    except Exception:
        return []

    try:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        dmatrix = xgb.DMatrix(arr, feature_names=list(feature_names))
        interactions = booster.predict(dmatrix, pred_interactions=True)
    except Exception:
        return []

    result = np.asarray(interactions)
    if result.ndim != 3 or result.shape[0] < 1:
        return []

    n = len(feature_names)
    matrix = result[0, :n, :n]  # drop bias row/col

    # Collect upper-triangle off-diagonal pairs (matrix is symmetric).
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            mag = float(matrix[i, j])
            pairs.append((abs(mag), i, j, mag))

    pairs.sort(key=lambda t: t[0], reverse=True)
    return [
        {
            "feature_a": feature_names[i],
            "feature_b": feature_names[j],
            "magnitude": round(mag, 6),
        }
        for _, i, j, mag in pairs[:top_k]
    ]
