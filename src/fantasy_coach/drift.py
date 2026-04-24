"""Model drift detection for the weekly retrain loop (#107).

Three signals surface in the weekly ``DriftReport``:

1. **Past-round accuracy / log-loss / brier** — how the live model did on
   the round that just finished. The fastest signal — one bad week doesn't
   mean drift, but a run of them does.
2. **Rolling log-loss trend** — per-round log-loss across the last N rounds
   so slow decay shows up even when no single round is catastrophic.
3. **Per-feature Population Stability Index (PSI)** — distribution shift
   between the *training* feature matrix and the *recent* one. High PSI on
   a feature means the model was trained on a world that no longer matches
   the one it's predicting.

The retrain loop writes one ``DriftReport`` per run to Firestore collection
``model_drift_reports`` keyed ``{season}-{round:02d}``. The promotion-gate
decision is separate (see ``models.promotion``) — PSI is a *warning* signal
only, never a block. A feature with PSI above ``PSI_WARN`` (0.25) surfaces
in ``DriftReport.psi_warnings`` but doesn't stop a candidate model from
being promoted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# PSI thresholds — industry standard (Siddiqi 2006). The AC pins the warn
# threshold at 0.25.
PSI_WARN = 0.25

# Default bin count for PSI. Quantile binning on the expected distribution;
# 10 bins is the common baseline. Auto-reduced when the smaller sample is
# too thin to support 10 bins — see ``_effective_bins``.
DEFAULT_PSI_BINS = 10

# Expected samples per bin used by ``_effective_bins``. Below ~10 samples
# per bin PSI's null distribution (E[PSI | same population] ≈ (bins−1)/n)
# rises fast enough to exceed ``PSI_WARN`` on pure sampling variance, so
# we cap the bin count at ``n_smaller // 10`` to keep E[PSI | null] under
# ~0.10 at the sample sizes the retrain loop actually sees (~32 recent
# predictions = 4 rounds × 8 matches).

# Floor applied to empty-bin fractions before the log ratio. Keeps PSI
# finite when a bin vanishes in one of the two samples.
_EPS = 1e-4


def _effective_bins(expected_size: int, actual_size: int, requested: int) -> int:
    """Cap the bin count so the smaller sample has ≥ 5 expected per bin.

    The production use case runs shadow-eval on ~4 rounds × 8 matches = 32
    samples. With 10 bins that's ~3 per bin, so pure sampling variance
    exceeds ``PSI_WARN`` on same-population draws. Auto-reducing fixes the
    false-positive rate without moving the warn threshold.
    """
    smaller = min(expected_size, actual_size)
    cap = max(2, smaller // 10)
    return min(requested, cap)


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    bins: int = DEFAULT_PSI_BINS,
) -> float:
    """Population Stability Index between two 1-D samples of a feature.

    Interpretation:
    - 0.0         identical.
    - < 0.1       no meaningful shift.
    - 0.1 – 0.25  minor shift.
    - > 0.25      material shift (``PSI_WARN``).

    Returns 0.0 on empty input. Low-cardinality features (unique value
    count ≤ ``bins``) are handled with a categorical formula rather than
    quantile binning — quantile edges collapse when only a handful of
    distinct values exist, so the continuous path gives misleading zeros
    on binary flags like ``missing_weather``.
    """
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if expected.size == 0 or actual.size == 0:
        return 0.0

    effective_bins = _effective_bins(expected.size, actual.size, bins)

    unique = np.unique(np.concatenate([expected, actual]))
    if unique.size <= effective_bins:
        return _categorical_psi(expected, actual, unique)

    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, effective_bins + 1)))
    if edges.size < 2:
        return 0.0

    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual, bins=edges)

    exp_frac = np.where(exp_hist == 0, _EPS, exp_hist / expected.size)
    act_frac = np.where(act_hist == 0, _EPS, act_hist / actual.size)

    psi = float(np.sum((act_frac - exp_frac) * np.log(act_frac / exp_frac)))
    return max(psi, 0.0)


def _categorical_psi(expected: np.ndarray, actual: np.ndarray, unique: np.ndarray) -> float:
    psi = 0.0
    for v in unique:
        p = max((expected == v).sum() / expected.size, _EPS)
        q = max((actual == v).sum() / actual.size, _EPS)
        psi += (q - p) * np.log(q / p)
    return max(float(psi), 0.0)


def per_feature_psi(
    training_X: np.ndarray,
    recent_X: np.ndarray,
    feature_names: Sequence[str],
    *,
    bins: int = DEFAULT_PSI_BINS,
) -> dict[str, float]:
    """Compute PSI column-wise between two feature matrices.

    ``training_X`` and ``recent_X`` must have the same number of columns,
    matching ``feature_names`` in order.
    """
    if training_X.shape[1] != len(feature_names):
        raise ValueError(
            f"feature matrix has {training_X.shape[1]} cols but "
            f"{len(feature_names)} feature_names given"
        )
    if recent_X.shape[1] != len(feature_names):
        raise ValueError(
            f"recent matrix has {recent_X.shape[1]} cols but "
            f"{len(feature_names)} feature_names given"
        )
    return {
        name: compute_psi(training_X[:, i], recent_X[:, i], bins=bins)
        for i, name in enumerate(feature_names)
    }


def psi_warnings(feature_psi: dict[str, float], *, threshold: float = PSI_WARN) -> list[str]:
    """Return feature names whose PSI exceeds ``threshold``, alphabetically."""
    return sorted(name for name, value in feature_psi.items() if value > threshold)


@dataclass(frozen=True)
class RoundLogLoss:
    """One entry in the rolling log-loss trend."""

    season: int
    round: int
    n: int
    log_loss: float
    accuracy: float


@dataclass(frozen=True)
class DriftReport:
    """Single weekly drift snapshot. Persisted as a Firestore document.

    Schema is stable so future UI (e.g. the Accuracy page) can chart the
    per-round trend without re-deriving it.
    """

    season: int
    round: int
    generated_at: str  # ISO 8601 UTC
    model_version: str  # first 12 hex chars of sha256(artefact)
    past_round_accuracy: float | None
    past_round_log_loss: float | None
    past_round_brier: float | None
    rolling_log_loss: list[RoundLogLoss]
    feature_psi: dict[str, float]
    psi_warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "round": self.round,
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "past_round_accuracy": self.past_round_accuracy,
            "past_round_log_loss": self.past_round_log_loss,
            "past_round_brier": self.past_round_brier,
            "rolling_log_loss": [
                {
                    "season": r.season,
                    "round": r.round,
                    "n": r.n,
                    "log_loss": r.log_loss,
                    "accuracy": r.accuracy,
                }
                for r in self.rolling_log_loss
            ],
            "feature_psi": dict(self.feature_psi),
            "psi_warnings": list(self.psi_warnings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DriftReport:
        return cls(
            season=data["season"],
            round=data["round"],
            generated_at=data["generated_at"],
            model_version=data["model_version"],
            past_round_accuracy=data.get("past_round_accuracy"),
            past_round_log_loss=data.get("past_round_log_loss"),
            past_round_brier=data.get("past_round_brier"),
            rolling_log_loss=[RoundLogLoss(**r) for r in data.get("rolling_log_loss", [])],
            feature_psi=dict(data.get("feature_psi", {})),
            psi_warnings=list(data.get("psi_warnings", [])),
        )
