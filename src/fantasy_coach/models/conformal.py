"""Conformal prediction intervals for distribution-free calibrated uncertainty (#214).

Sits *outside* any model and converts a calibration-set residual distribution
into prediction intervals with distribution-free, finite-sample guaranteed
coverage (Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2021).

Unlike Bayesian or model-internal calibration, conformal prediction requires
*no* distributional assumptions:

    P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 − α

for any miscoverage level α, regardless of model misspecification, as long as
the calibration data is exchangeable with the test data.

Two conformalizers are provided:

ProbabilityConformalizer
    Fits on a held-out calibration set of (model_probability, actual_outcome)
    pairs.  Non-conformity score = |p_model − y| (regression-style).
    Transforms a point probability p into a guaranteed-coverage interval
    [max(0, p−q), min(1, p+q)] where q is the (1−α)(1+1/n) empirical quantile
    of calibration residuals.

MarginConformalizer
    Fits on (predicted_margin, actual_margin) pairs.
    Non-conformity score = |predicted − actual|.
    Produces a symmetric interval [pred−q, pred+q].

MondrianConformalizer
    Conditional (stratified) variant.  Bins the calibration set by a grouping
    function, then uses bin-specific quantiles.  Achieves conditional coverage
    within each bin at the cost of wider intervals where data is thin — which
    is the correct behaviour for out-of-distribution predictions.

References
----------
Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a
    Random World*. Springer.
Angelopoulos, A. N. & Bates, S. (2021). A Gentle Introduction to Conformal
    Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Probability conformalizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbabilityInterval:
    """A distribution-free probability interval for the home-win probability."""

    lo: float   # lower bound in [0, 1]
    hi: float   # upper bound in [0, 1]
    point: float  # original point estimate


@dataclass(frozen=True)
class MarginInterval:
    """A distribution-free interval around the predicted margin."""

    lo: float
    hi: float
    point: float


class ProbabilityConformalizer:
    """Conformal prediction intervals for model win-probability outputs.

    Non-conformity score: s = |p_model − y|  (y ∈ {0, 1}).

    After fit, ``transform(p)`` returns a [lo, hi] interval such that for a
    new match drawn from the same distribution the interval covers the true
    outcome probability at rate ≥ 1 − alpha.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self._quantile: float | None = None
        self._n_calib: int = 0

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit on calibration-set (probability, binary-outcome) pairs.

        Parameters
        ----------
        probabilities:
            Model-predicted P(home wins), shape (n,).
        outcomes:
            Observed binary outcomes (1 = home won, 0 = away won), shape (n,).
        """
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        if p.shape != y.shape or p.ndim != 1:
            raise ValueError("probabilities and outcomes must be 1-D arrays of equal length")
        scores = np.abs(p - y)
        n = len(scores)
        self._n_calib = n
        # Conformal quantile: ceil((n+1)(1−α))/n quantile of the scores.
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._quantile = float(np.quantile(scores, level, method="higher"))

    def transform(self, probability: float) -> ProbabilityInterval:
        """Return the conformal interval around ``probability``."""
        if self._quantile is None:
            raise RuntimeError("Call fit() before transform()")
        q = self._quantile
        return ProbabilityInterval(
            lo=float(np.clip(probability - q, 0.0, 1.0)),
            hi=float(np.clip(probability + q, 0.0, 1.0)),
            point=float(probability),
        )

    def coverage_fraction(self, probabilities: np.ndarray, outcomes: np.ndarray) -> float:
        """Empirical coverage on a held-out set (for evaluation)."""
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        intervals = [self.transform(float(pi)) for pi in p]
        covered = sum(iv.lo <= yi <= iv.hi for iv, yi in zip(intervals, y, strict=True))
        return covered / len(y) if len(y) > 0 else 0.0


# ---------------------------------------------------------------------------
# Margin conformalizer
# ---------------------------------------------------------------------------


class MarginConformalizer:
    """Conformal prediction intervals for the predicted score margin.

    Non-conformity score: s = |predicted_margin − actual_margin|.

    After fit, ``transform(predicted)`` returns a [predicted−q, predicted+q]
    interval with guaranteed coverage ≥ 1 − alpha.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self._quantile: float | None = None
        self._n_calib: int = 0

    def fit(self, predicted_margins: np.ndarray, actual_margins: np.ndarray) -> None:
        """Fit on calibration-set (predicted, actual) margin pairs."""
        pred = np.asarray(predicted_margins, dtype=float)
        actual = np.asarray(actual_margins, dtype=float)
        if pred.shape != actual.shape or pred.ndim != 1:
            raise ValueError("Arrays must be 1-D and equal length")
        scores = np.abs(pred - actual)
        n = len(scores)
        self._n_calib = n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._quantile = float(np.quantile(scores, level, method="higher"))

    def transform(self, predicted_margin: float) -> MarginInterval:
        """Return the conformal interval around ``predicted_margin``."""
        if self._quantile is None:
            raise RuntimeError("Call fit() before transform()")
        q = self._quantile
        return MarginInterval(
            lo=float(predicted_margin - q),
            hi=float(predicted_margin + q),
            point=float(predicted_margin),
        )

    def coverage_fraction(
        self, predicted_margins: np.ndarray, actual_margins: np.ndarray
    ) -> float:
        """Empirical coverage on a held-out set."""
        pred = np.asarray(predicted_margins, dtype=float)
        actual = np.asarray(actual_margins, dtype=float)
        intervals = [self.transform(float(p)) for p in pred]
        covered = sum(iv.lo <= a <= iv.hi for iv, a in zip(intervals, actual, strict=True))
        return covered / len(actual) if len(actual) > 0 else 0.0


# ---------------------------------------------------------------------------
# Mondrian (conditional) conformalizer
# ---------------------------------------------------------------------------


class MondrianConformalizer:
    """Conditional conformalizer that uses bin-specific residual quantiles.

    Achieves bin-conditional coverage ≥ 1 − alpha within each bin.  Intervals
    are wider for bins with few calibration observations — which is the correct
    behaviour: low-data bins should report lower confidence.

    The ``bin_fn`` argument maps a (probability, feature_vector) pair to a
    bin label (str or int).  Typical choices:

    - Split by season half: bin_fn = lambda p, x: "early" if round <= 12 else "late"
    - Home vs away favourite: bin_fn = lambda p, x: "home_fav" if p >= 0.5 else "away_fav"
    """

    def __init__(
        self,
        bin_fn: Callable[[float, np.ndarray | None], Any],
        alpha: float = 0.1,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.bin_fn = bin_fn
        self.alpha = alpha
        self._quantiles: dict[Any, float] = {}
        self._fallback_quantile: float | None = None

    def fit(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        features: np.ndarray | None = None,
    ) -> None:
        """Fit per-bin quantiles.

        features: shape (n, n_features) optional feature matrix used by bin_fn.
        """
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        scores = np.abs(p - y)
        n = len(scores)

        bins: dict[Any, list[float]] = {}
        for i in range(n):
            feat_i = features[i] if features is not None else None
            b = self.bin_fn(float(p[i]), feat_i)
            bins.setdefault(b, []).append(float(scores[i]))

        self._quantiles = {}
        for b, bin_scores in bins.items():
            n_b = len(bin_scores)
            level = np.ceil((n_b + 1) * (1.0 - self.alpha)) / n_b
            level = min(level, 1.0)
            self._quantiles[b] = float(np.quantile(bin_scores, level, method="higher"))

        # Marginal fallback for unseen bins
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._fallback_quantile = float(np.quantile(scores, level, method="higher"))

    def transform(
        self,
        probability: float,
        features: np.ndarray | None = None,
    ) -> ProbabilityInterval:
        """Return the bin-conditional conformal interval."""
        if not self._quantiles and self._fallback_quantile is None:
            raise RuntimeError("Call fit() before transform()")
        b = self.bin_fn(float(probability), features)
        q = self._quantiles.get(b, self._fallback_quantile)
        return ProbabilityInterval(
            lo=float(np.clip(probability - q, 0.0, 1.0)),
            hi=float(np.clip(probability + q, 0.0, 1.0)),
            point=float(probability),
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_conformalizer(path: Path | str, conformalizer: object) -> None:
    """Persist any conformalizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(conformalizer, path)


def load_conformalizer(path: Path | str) -> object:
    """Load a previously saved conformalizer."""
    return joblib.load(Path(path))
