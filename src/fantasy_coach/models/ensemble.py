"""Model ensembling over calibrated base-model probabilities.

Two modes:

- ``weighted`` — convex combination ``p = Σ w_i · p_i`` with ``w_i ≥ 0`` and
  ``Σ w_i = 1``. Weights are fit by minimising log loss on a held-out
  validation slice via constrained L-BFGS-B.
- ``stacked`` — sklearn ``LogisticRegression`` meta-learner over the base
  probabilities. More flexible (can learn to invert a miscalibrated base
  or weight interactions) but needs more validation data to not overfit.

Base probabilities are assumed already calibrated — stacking uncalibrated
probabilities makes the meta-learner waste capacity correcting base-model
calibration rather than combining independent signals (see calibration
issue #53 for the rationale).

## Kill switch

If the fitted ensemble doesn't beat the best individual base predictor by
at least ``min_improvement`` log-loss points (default 0.005 nats) on the
same validation slice, ``EnsembleModel.fallback_to_base`` is set to the
winning base predictor's name. Callers should route prediction traffic to
that base predictor instead — the ensemble has learned noise, not signal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from fantasy_coach.feature_engineering import FEATURE_NAMES

if TYPE_CHECKING:
    from fantasy_coach.models.loader import Model

EnsembleMode = Literal["weighted", "stacked"]

# Minimum log-loss improvement (on the validation fold) over the best base
# predictor before we trust the ensemble. Chosen to be well above the noise
# floor on a ~300-match fold but below meaningful signal — 0.005 nats is ~5 %
# of a typical NRL-prediction log-loss gap between the bookmaker and a
# decent model.
DEFAULT_MIN_IMPROVEMENT = 0.005

_EPS = 1e-15


def _log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class EnsembleModel:
    """A fitted ensemble over a fixed set of base predictors.

    Only one of ``weights`` / ``meta_learner`` is populated, depending on
    ``mode``. ``fallback_to_base`` being set means the kill switch fired
    and callers should ignore the ensemble's prediction.
    """

    mode: EnsembleMode
    base_model_names: tuple[str, ...]
    weights: np.ndarray | None = None
    meta_learner: LogisticRegression | None = None

    # Diagnostics — kept so the evaluation harness can surface the kill-
    # switch decision in reports.
    ensemble_log_loss: float = float("nan")
    base_log_losses: dict[str, float] = field(default_factory=dict)
    best_base_name: str = ""
    fallback_to_base: str | None = None

    @property
    def use_fallback(self) -> bool:
        return self.fallback_to_base is not None

    def predict_home_win_prob(self, base_probs: np.ndarray) -> np.ndarray:
        """Combine per-sample base probabilities into ensemble probabilities.

        ``base_probs`` has shape ``(n_samples, n_base)``; columns must be in
        the same order as ``base_model_names``. When the kill switch has
        fired, we return the winning base predictor's column unchanged.
        """
        if base_probs.shape[1] != len(self.base_model_names):
            raise ValueError(
                f"Expected {len(self.base_model_names)} base columns, got {base_probs.shape[1]}"
            )
        if self.use_fallback:
            assert self.fallback_to_base is not None
            idx = self.base_model_names.index(self.fallback_to_base)
            return base_probs[:, idx]
        if self.mode == "weighted":
            assert self.weights is not None
            return np.clip(base_probs @ self.weights, 0.0, 1.0)
        assert self.meta_learner is not None
        return self.meta_learner.predict_proba(base_probs)[:, 1]


def fit_ensemble(
    base_probs: np.ndarray,
    y: np.ndarray,
    *,
    mode: EnsembleMode,
    base_model_names: tuple[str, ...],
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
) -> EnsembleModel:
    """Fit a ``weighted`` or ``stacked`` ensemble on out-of-fold predictions.

    ``base_probs`` is the matrix of base-predictor probabilities on the
    validation slice (shape ``(n_samples, n_base)``); ``y`` is the actual
    outcomes on the same rows.

    The caller is responsible for producing ``base_probs`` without leakage —
    each row must come from base predictors trained on data strictly before
    the match the row represents.
    """
    if base_probs.ndim != 2:
        raise ValueError("base_probs must be 2D (n_samples, n_base)")
    if base_probs.shape[1] != len(base_model_names):
        raise ValueError(
            f"base_probs has {base_probs.shape[1]} columns, "
            f"but {len(base_model_names)} names were given"
        )
    if base_probs.shape[0] != y.shape[0]:
        raise ValueError("base_probs and y must have the same number of rows")

    base_losses = {name: _log_loss(base_probs[:, i], y) for i, name in enumerate(base_model_names)}
    best_base = min(base_losses, key=lambda n: base_losses[n])

    model: EnsembleModel
    if mode == "weighted":
        weights = _fit_convex_weights(base_probs, y)
        ensemble_probs = base_probs @ weights
        model = EnsembleModel(
            mode=mode,
            base_model_names=base_model_names,
            weights=weights,
            base_log_losses=base_losses,
            best_base_name=best_base,
            ensemble_log_loss=_log_loss(ensemble_probs, y),
        )
    elif mode == "stacked":
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(base_probs, y)
        ensemble_probs = meta.predict_proba(base_probs)[:, 1]
        model = EnsembleModel(
            mode=mode,
            base_model_names=base_model_names,
            meta_learner=meta,
            base_log_losses=base_losses,
            best_base_name=best_base,
            ensemble_log_loss=_log_loss(ensemble_probs, y),
        )
    else:  # pragma: no cover — guarded by Literal
        raise ValueError(f"Unknown ensemble mode: {mode!r}")

    improvement = base_losses[best_base] - model.ensemble_log_loss
    if improvement < min_improvement:
        model.fallback_to_base = best_base
    return model


@dataclass(frozen=True)
class LoadedEnsemble:
    """Ensemble artifact wrapper that satisfies the ``Model`` protocol.

    Holds the fitted ``EnsembleModel`` combiner plus one already-loaded base
    model per column. At inference, scores each base on the raw feature
    matrix, stacks their probabilities, and passes the result through the
    ensemble combiner (honouring the kill switch).
    """

    feature_names: tuple[str, ...]
    ensemble: EnsembleModel
    base_models: tuple[Model, ...]

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features "
                f"({self.feature_names}), got {X.shape[1]}"
            )
        base_probs = np.column_stack(
            [np.asarray(base.predict_home_win_prob(X), dtype=float) for base in self.base_models]
        )
        return self.ensemble.predict_home_win_prob(base_probs)


def save_ensemble(
    path: Path | str,
    *,
    ensemble: EnsembleModel,
    base_blobs: Sequence[dict],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> None:
    """Persist an ensemble artifact bundling its fitted combiner + bases.

    ``base_blobs`` is a sequence of per-base dicts already shaped for the
    per-model ``_from_blob`` loaders (each carries its own ``model_type``).
    Column order must match ``ensemble.base_model_names``.
    """
    if len(base_blobs) != len(ensemble.base_model_names):
        raise ValueError(
            f"Got {len(base_blobs)} base blobs but ensemble expects "
            f"{len(ensemble.base_model_names)} columns"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "ensemble",
            "feature_names": tuple(feature_names),
            "ensemble": ensemble,
            "base_blobs": list(base_blobs),
        },
        path,
    )


def _from_blob(blob: dict) -> LoadedEnsemble:
    from fantasy_coach.models.loader import _from_blob as _recurse

    feature_names = tuple(blob.get("feature_names", ()))
    if feature_names != FEATURE_NAMES:
        raise RuntimeError(
            f"Ensemble trained with features {feature_names}, "
            f"current code expects {FEATURE_NAMES}. Retrain before loading."
        )
    ensemble = blob["ensemble"]
    base_blobs = blob.get("base_blobs") or []
    if len(base_blobs) != len(ensemble.base_model_names):
        raise RuntimeError(
            f"Ensemble artifact has {len(base_blobs)} base blobs but "
            f"{len(ensemble.base_model_names)} base columns — mismatch"
        )
    bases = tuple(_recurse(b) for b in base_blobs)
    return LoadedEnsemble(
        feature_names=feature_names,
        ensemble=ensemble,
        base_models=bases,
    )


def _fit_convex_weights(base_probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimise log loss over the unit simplex.

    Parameterised by the raw vector ``w`` projected onto the simplex at each
    evaluation: non-negative clipping + normalisation. L-BFGS-B handles the
    box constraint; the sum-to-one normalisation makes the problem convex
    in the induced probability space.
    """
    n_base = base_probs.shape[1]
    w0 = np.full(n_base, 1.0 / n_base)

    def objective(w: np.ndarray) -> float:
        clipped = np.clip(w, 0.0, None)
        total = clipped.sum()
        if total == 0.0:
            return float("inf")
        normalised = clipped / total
        return _log_loss(base_probs @ normalised, y)

    result = minimize(objective, w0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * n_base)
    clipped = np.clip(result.x, 0.0, None)
    total = clipped.sum()
    # Fallback to uniform if the optimiser returned an all-zero vector —
    # can happen when the base predictors are all wildly miscalibrated.
    if total == 0.0:
        return np.full(n_base, 1.0 / n_base)
    return clipped / total
