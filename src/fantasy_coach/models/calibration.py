"""Probability calibration wrapper for sklearn-based predictors.

Supports two methods:
- 'platt' (sigmoid): linear scaling in log-odds space; default for LogReg
- 'isotonic': non-parametric monotonic regression; default for XGBoost

Fitting always uses a **held-out fold** — never the same data the base model
trained on.  Pass the last 20% of the time-ordered training frame as the
calibration set.

Decision: use 'platt' for LogReg, 'isotonic' for XGBoost (tree models have
more extreme confidence near 0/1 which isotonic handles better than Platt's
linear correction).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CalibrationMethod = Literal["platt", "isotonic"]


@dataclass
class CalibrationWrapper:
    """Wraps a fitted sklearn Pipeline and adjusts its probability output.

    Calibration is applied as a post-hoc mapping from raw probability scores
    to calibrated probabilities, fitted on a **held-out** calibration set:

    - ``platt``: logistic regression fit on log-odds(raw_prob).  Cheap,
      assumes the raw scores are monotone and need only a linear correction.
      Default for LogReg.
    - ``isotonic``: non-parametric monotone regression.  More flexible but
      needs a larger calibration set (n_cal ≥ 500 recommended).  Default for
      XGBoost.

    After ``fit(X_cal, y_cal)``, ``predict_home_win_prob`` returns calibrated
    probabilities.  Before fitting it falls back to the raw pipeline.
    """

    pipeline: Pipeline
    method: CalibrationMethod = "platt"
    _calibrator: LogisticRegression | IsotonicRegression | None = field(
        default=None, init=False, repr=False
    )

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> CalibrationWrapper:
        raw = self.pipeline.predict_proba(X_cal)[:, 1]
        if self.method == "platt":
            # Platt scaling: logistic regression on the raw probability scores.
            self._calibrator = LogisticRegression(C=1.0, max_iter=1000)
            self._calibrator.fit(raw.reshape(-1, 1), y_cal)
        else:
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw, y_cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.pipeline.predict_proba(X)[:, 1]
        if self._calibrator is None:
            cal = raw
        elif isinstance(self._calibrator, LogisticRegression):
            cal = self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            cal = self._calibrator.predict(raw)
        proba = np.column_stack([1.0 - cal, cal])
        return proba

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    @property
    def is_fitted(self) -> bool:
        return self._calibrator is not None


def ece(
    probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (equal-width bins, weighted by bin size).

    A value of 0.0 means perfectly calibrated; a well-tuned model should
    achieve ECE < 0.05.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_error = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        bin_size = int(np.sum(mask))
        avg_conf = float(np.mean(p[mask]))
        avg_acc = float(np.mean(y[mask]))
        total_error += (bin_size / n) * abs(avg_acc - avg_conf)

    return total_error


def reliability_bins(
    probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[dict[str, object]]:
    """Per-bin data for a reliability diagram.

    Each entry: ``{lo, hi, mean_confidence, mean_accuracy, n}``.
    Bins with no predictions have ``mean_confidence`` and ``mean_accuracy``
    set to ``None``.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, object]] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (p >= lo) & (p < hi)
        n = int(np.sum(mask))
        if n == 0:
            bins.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "mean_confidence": None,
                    "mean_accuracy": None,
                    "n": 0,
                }
            )
        else:
            bins.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "mean_confidence": float(np.mean(p[mask])),
                    "mean_accuracy": float(np.mean(y[mask])),
                    "n": n,
                }
            )

    return bins
