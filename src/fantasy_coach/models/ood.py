"""Out-of-distribution / extrapolation detector for prediction trustworthiness (#216).

Attaches an OOD score to every prediction: a percentile (0–100) expressing
how far this match's feature vector is from the training distribution.  High
percentile = unusual match = model is extrapolating.

Three-level flag derived from the percentile:
    "in_distribution"     percentile < 90 (model has seen many similar matches)
    "edge"                90 ≤ percentile < 99
    "out_of_distribution" percentile ≥ 99

The default implementation uses scikit-learn's ``IsolationForest``, which is
fast, scale-invariant, and well-suited to mixed feature spaces.  The scoring
backend is pluggable via a Protocol so Mahalanobis distance or a normalising
flow can be swapped in later without changing the contract.

References
----------
Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation Forest. ICDM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Percentile thresholds for the three-level flag
_EDGE_THRESHOLD = 90.0
_OOD_THRESHOLD = 99.0


# ---------------------------------------------------------------------------
# Scorer protocol (pluggable backend)
# ---------------------------------------------------------------------------


@runtime_checkable
class OODScorer(Protocol):
    """Minimal interface for OOD scoring backends."""

    def fit(self, X: np.ndarray) -> None: ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores: *lower* = more anomalous (matches sklearn convention)."""
        ...


# ---------------------------------------------------------------------------
# OOD result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OODResult:
    """OOD detection result for a single match."""

    percentile: float  # 0–100; high = unusual
    flag: str          # "in_distribution" | "edge" | "out_of_distribution"
    raw_score: float   # raw anomaly score from the backend (lower = more anomalous)


def _percentile_to_flag(percentile: float) -> str:
    if percentile >= _OOD_THRESHOLD:
        return "out_of_distribution"
    if percentile >= _EDGE_THRESHOLD:
        return "edge"
    return "in_distribution"


# ---------------------------------------------------------------------------
# OOD detector
# ---------------------------------------------------------------------------


class OODDetector:
    """Isolation-Forest-based OOD detector for the NRL prediction feature space.

    Usage::

        detector = OODDetector()
        detector.fit(X_train)                    # fit on training features
        result = detector.score(X_new)           # OODResult per row
        detections = detector.score_batch(X_new) # list[OODResult]
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self._scaler = StandardScaler()
        self._forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
        )
        self._train_scores: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Fit the OOD detector on training feature matrix X (n_train × n_features)."""
        X_scaled = self._scaler.fit_transform(X)
        self._forest.fit(X_scaled)
        # Stash training-set scores to derive empirical percentiles at inference
        raw = self._forest.score_samples(X_scaled)  # lower = more anomalous
        self._train_scores = raw

    def score(self, x: np.ndarray) -> OODResult:
        """Return the OOD result for a single match.

        x: shape (1, n_features) or (n_features,).
        """
        x = np.atleast_2d(x)
        return self.score_batch(x)[0]

    def score_batch(self, X: np.ndarray) -> list[OODResult]:
        """Return OOD results for a batch of matches.

        X: shape (n, n_features).
        """
        if self._train_scores is None:
            raise RuntimeError("Call fit() before scoring")
        X_scaled = self._scaler.transform(X)
        raw_scores = self._forest.score_samples(X_scaled)
        results = []
        for rs in raw_scores:
            # Convert to a "how anomalous" percentile: negate (so higher = more anomalous),
            # then find what fraction of training scores are *less anomalous* (i.e., higher
            # raw score).  A new point with score lower than 90% of training scores is
            # in the 90th anomaly percentile.
            pct = float(np.mean(self._train_scores > rs) * 100.0)
            results.append(OODResult(
                percentile=round(pct, 2),
                flag=_percentile_to_flag(pct),
                raw_score=float(rs),
            ))
        return results


# ---------------------------------------------------------------------------
# Mahalanobis-distance alternative backend (simple, no sklearn dependency)
# ---------------------------------------------------------------------------


class MahalanobisScorer:
    """Mahalanobis-distance OOD scorer.

    Faster than Isolation Forest for small feature sets but less robust to
    high-dimensional, mixed-type feature spaces.
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        self._mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        # Regularise with a small diagonal to avoid singular covariance
        cov += np.eye(cov.shape[0]) * 1e-6
        self._cov_inv = np.linalg.inv(cov)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return *negative* Mahalanobis distance (lower = more anomalous)."""
        if self._mean is None or self._cov_inv is None:
            raise RuntimeError("Call fit() first")
        delta = X - self._mean
        maha_sq = np.einsum("ij,jk,ik->i", delta, self._cov_inv, delta)
        return -np.sqrt(np.maximum(maha_sq, 0.0))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_ood_detector(path: Path | str, detector: OODDetector) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, path)


def load_ood_detector(path: Path | str) -> OODDetector:
    obj = joblib.load(Path(path))
    if not isinstance(obj, OODDetector):
        raise ValueError(f"Expected OODDetector, got {type(obj).__name__}")
    return obj
