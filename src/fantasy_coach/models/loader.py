"""Uniform model-loading dispatch for the prediction API.

The /predictions endpoint points ``FANTASY_COACH_MODEL_PATH`` at a joblib
artifact without knowing or caring which model type produced it. This module
sniffs the artifact's ``model_type`` key and returns an object satisfying
the ``Model`` protocol — ``predict_home_win_prob(X)`` + ``feature_names``.

Supported artifact types: ``logistic``, ``xgboost``, ``ensemble``. Blobs
without a ``model_type`` key are treated as logistic (the pre-#84 format).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np


@runtime_checkable
class Model(Protocol):
    """Inference contract shared across every saved model type."""

    feature_names: tuple[str, ...]

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray: ...


def _from_blob(blob: dict) -> Model:
    """Dispatch a loaded joblib dict to the right model-specific loader.

    Lazy imports avoid circular deps (ensemble nests base blobs and calls
    back into this function).
    """
    kind = blob.get("model_type", "logistic")
    if kind == "logistic":
        from fantasy_coach.models.logistic import _from_blob as _log

        return _log(blob)
    if kind == "xgboost":
        from fantasy_coach.models.xgboost_model import _from_blob as _xgb

        return _xgb(blob)
    if kind == "ensemble":
        from fantasy_coach.models.ensemble import _from_blob as _ens

        return _ens(blob)
    if kind == "bivariate_poisson":
        from fantasy_coach.models.bivariate_poisson import _from_blob as _bp

        return _bp(blob)
    if kind == "multitask":
        from fantasy_coach.models.multitask import _from_blob as _mt

        return _mt(blob)
    raise RuntimeError(f"Unknown model_type in artifact: {kind!r}")


def load_model(path: Path | str) -> Model:
    """Load a joblib artifact and return an object satisfying ``Model``."""
    blob = joblib.load(Path(path))
    if not isinstance(blob, dict):
        raise RuntimeError(f"Expected dict artifact at {path}, got {type(blob).__name__}")
    return _from_blob(blob)
