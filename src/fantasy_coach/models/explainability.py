"""Model-aware feature attribution using TreeSHAP for XGBoost.

For logistic models the attribution path lives in ``predictions.py``
(linear: ``coef × (x - mean) / scale``). This module owns the tree-based
path: TreeSHAP (Lundberg & Lee 2017) via XGBoost's native ``pred_contribs``
and ``pred_interactions`` modes.

The public contract is:
- ``shap_contributions`` — per-feature SHAP values in raw-margin (log-odds)
  space. Sum of all values plus the bias equals the model's raw margin output.
- ``shap_interactions`` — top-K feature×feature SHAP interaction values
  (off-diagonal only). Exposes how features co-operate to drive the margin.

Both return ``None`` when the supplied artefact is not an XGBoost model or
when xgboost is not importable in the current environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def shap_contributions(
    loaded: Any,
    x: Any,
) -> tuple[Any, float] | None:
    """Return (per_feature_shap, bias) for an XGBoost artefact, or None.

    ``per_feature_shap`` is a 1-D array aligned with ``loaded.feature_names``.
    ``bias`` is the model's base value (last column from ``pred_contribs``).

    The invariant ``bias + per_feature_shap.sum() == raw_margin`` holds within
    floating-point precision, where ``raw_margin = logit(predicted_prob)`` for
    a binary XGBoost classifier.
    """
    estimator = getattr(loaded, "estimator", None)
    if estimator is None:
        return None
    get_booster = getattr(estimator, "get_booster", None)
    if not callable(get_booster):
        return None

    try:
        import numpy as np  # noqa: PLC0415
        import xgboost as xgb  # noqa: PLC0415
    except Exception:  # pragma: no cover — libomp-missing safety net
        return None

    feature_names = getattr(loaded, "feature_names", None)
    try:
        booster = get_booster()
        dmatrix = xgb.DMatrix(
            np.asarray(x, dtype=float).reshape(1, -1),
            feature_names=list(feature_names) if feature_names else None,
        )
        contribs = booster.predict(dmatrix, pred_contribs=True)
    except Exception:
        return None

    arr = np.asarray(contribs)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return None

    # Last column is the bias term / expected value.
    per_feature = arr[0, :-1]
    bias = float(arr[0, -1])
    return per_feature, bias


def shap_interactions(
    loaded: Any,
    x: Any,
    *,
    top_k: int = 3,
) -> list[tuple[str, str, float]] | None:
    """Return top-K SHAP interaction pairs (feature_i, feature_j, value).

    Uses XGBoost's ``pred_interactions=True`` mode. Only off-diagonal entries
    are returned (diagonal = main effects, covered by ``shap_contributions``).
    Results are sorted by descending ``|value|``; ties broken by feature index.

    Returns ``None`` when the artefact is not XGBoost or xgboost is missing.
    """
    estimator = getattr(loaded, "estimator", None)
    if estimator is None:
        return None
    get_booster = getattr(estimator, "get_booster", None)
    if not callable(get_booster):
        return None

    feature_names = getattr(loaded, "feature_names", None)
    if not feature_names:
        return None

    try:
        import numpy as np  # noqa: PLC0415
        import xgboost as xgb  # noqa: PLC0415
    except Exception:  # pragma: no cover
        return None

    try:
        booster = get_booster()
        dmatrix = xgb.DMatrix(
            np.asarray(x, dtype=float).reshape(1, -1),
            feature_names=list(feature_names),
        )
        # Shape: (1, n_features+1, n_features+1) — last row/col is bias.
        interactions = booster.predict(dmatrix, pred_interactions=True)
    except Exception:
        return None

    arr = np.asarray(interactions)
    if arr.ndim != 3 or arr.shape[0] < 1:
        return None

    mat = arr[0]  # (n_features+1, n_features+1)
    n = len(feature_names)

    # Collect upper-triangle off-diagonal entries (each interaction appears
    # twice symmetrically; take one instance, sorted by |value|).
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((abs(mat[i, j]), i, j))

    pairs.sort(key=lambda t: -t[0])
    return [
        (str(feature_names[i]), str(feature_names[j]), round(float(mat[i, j]), 6))
        for _, i, j in pairs[:top_k]
    ]
