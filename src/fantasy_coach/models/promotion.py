"""Promotion gate for the weekly retrain loop (#107).

The gate runs a shadow walk-forward evaluation on the most recent N
"holdout" rounds against both the *incumbent* (current production) and
*candidate* (freshly trained) artefacts. If the candidate regresses on
log-loss OR brier by more than ``max_regression_pct`` (2 % by AC), the
gate blocks publication and the retrain loop keeps the incumbent live.

Why log-loss + brier, not accuracy:
- Both are proper scoring rules sensitive to probability *calibration*,
  not just the top pick. A model that pushes probabilities toward 0/1
  can score higher accuracy while scoring worse on log-loss — and the
  API's ``homeWinProbability`` output + the contribution-list UI both
  rely on well-calibrated probabilities.
- Accuracy change is surfaced in ``GateDecision`` for the drift report
  but never blocks on its own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fantasy_coach.evaluation.metrics import accuracy, brier_score, log_loss
from fantasy_coach.feature_engineering import FeatureBuilder
from fantasy_coach.features import MatchRow
from fantasy_coach.models.loader import Model

DEFAULT_MAX_REGRESSION_PCT = 2.0
DEFAULT_HOLDOUT_ROUNDS = 4


@dataclass(frozen=True)
class ShadowMetrics:
    """Walk-forward metrics on a fixed holdout window."""

    n: int
    accuracy: float
    log_loss: float
    brier: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "brier": self.brier,
        }


@dataclass(frozen=True)
class GateDecision:
    """Result of comparing candidate vs incumbent on the holdout window."""

    promote: bool
    reason: str
    incumbent: ShadowMetrics
    candidate: ShadowMetrics
    log_loss_regression_pct: float  # + = candidate worse on log_loss
    brier_regression_pct: float  # + = candidate worse on brier
    accuracy_delta_pct: float  # + = candidate better on accuracy (informational)

    def to_dict(self) -> dict:
        return {
            "promote": self.promote,
            "reason": self.reason,
            "incumbent": self.incumbent.to_dict(),
            "candidate": self.candidate.to_dict(),
            "log_loss_regression_pct": self.log_loss_regression_pct,
            "brier_regression_pct": self.brier_regression_pct,
            "accuracy_delta_pct": self.accuracy_delta_pct,
        }


def _is_complete(match: MatchRow) -> bool:
    return (
        match.match_state in {"FullTime", "FullTimeED"}
        and match.home.score is not None
        and match.away.score is not None
    )


def shadow_evaluate(
    model: Model,
    training_matches: Sequence[MatchRow],
    holdout_matches: Sequence[MatchRow],
) -> ShadowMetrics:
    """Walk-forward score ``model`` on ``holdout_matches``.

    ``training_matches`` is the warmup history fed into a fresh
    ``FeatureBuilder`` so rolling-state features see everything the
    candidate saw at fit time. ``holdout_matches`` are then scored in
    chronological order; each round's matches are scored *before* their
    outcomes are folded into the builder, mirroring the real prediction
    flow. Draws are dropped from scoring (binary metrics want {0, 1}).
    """
    builder = FeatureBuilder()
    for match in sorted(training_matches, key=lambda m: (m.start_time, m.match_id)):
        if not _is_complete(match):
            continue
        builder.advance_season_if_needed(match)
        builder.record(match)

    probs: list[float] = []
    outcomes: list[int] = []
    for match in sorted(holdout_matches, key=lambda m: (m.start_time, m.match_id)):
        if not _is_complete(match):
            continue
        builder.advance_season_if_needed(match)
        h_score = int(match.home.score or 0)
        a_score = int(match.away.score or 0)
        if h_score == a_score:
            builder.record(match)
            continue
        row = np.asarray(builder.feature_row(match)).reshape(1, -1)
        p = float(model.predict_home_win_prob(row)[0])
        probs.append(p)
        outcomes.append(1 if h_score > a_score else 0)
        builder.record(match)

    return ShadowMetrics(
        n=len(probs),
        accuracy=accuracy(probs, outcomes),
        log_loss=log_loss(probs, outcomes),
        brier=brier_score(probs, outcomes),
    )


def _pct_change(baseline: float, new: float) -> float:
    """Signed percent change. Positive = ``new`` larger than ``baseline``."""
    if not math.isfinite(baseline) or baseline == 0.0:
        return 0.0
    return (new - baseline) / baseline * 100.0


def gate_decision(
    incumbent: ShadowMetrics,
    candidate: ShadowMetrics,
    *,
    max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT,
) -> GateDecision:
    """Decide whether ``candidate`` should replace ``incumbent``.

    Block on log-loss OR brier regression above ``max_regression_pct``.
    Accuracy delta is returned for logging but does not gate.
    """
    ll_pct = _pct_change(incumbent.log_loss, candidate.log_loss)
    br_pct = _pct_change(incumbent.brier, candidate.brier)
    acc_pct = _pct_change(incumbent.accuracy, candidate.accuracy)

    if ll_pct > max_regression_pct:
        return GateDecision(
            promote=False,
            reason=(
                f"log_loss regression {ll_pct:+.2f}% exceeds threshold +{max_regression_pct:.2f}%"
            ),
            incumbent=incumbent,
            candidate=candidate,
            log_loss_regression_pct=ll_pct,
            brier_regression_pct=br_pct,
            accuracy_delta_pct=acc_pct,
        )
    if br_pct > max_regression_pct:
        return GateDecision(
            promote=False,
            reason=(
                f"brier regression {br_pct:+.2f}% exceeds threshold +{max_regression_pct:.2f}%"
            ),
            incumbent=incumbent,
            candidate=candidate,
            log_loss_regression_pct=ll_pct,
            brier_regression_pct=br_pct,
            accuracy_delta_pct=acc_pct,
        )
    return GateDecision(
        promote=True,
        reason=(
            f"candidate cleared gate: log_loss {ll_pct:+.2f}%, "
            f"brier {br_pct:+.2f}%, accuracy {acc_pct:+.2f}%"
        ),
        incumbent=incumbent,
        candidate=candidate,
        log_loss_regression_pct=ll_pct,
        brier_regression_pct=br_pct,
        accuracy_delta_pct=acc_pct,
    )


def split_training_holdout(
    matches: Sequence[MatchRow],
    *,
    holdout_rounds: int = DEFAULT_HOLDOUT_ROUNDS,
) -> tuple[list[MatchRow], list[MatchRow]]:
    """Split matches into (training, holdout) by most-recent ``holdout_rounds``.

    Matches are grouped by (season, round) — the last ``holdout_rounds``
    *completed* (season, round) tuples by chronological order form the
    holdout; everything earlier is training. Incomplete matches belong
    to whichever bucket they fall into — they're skipped by
    ``shadow_evaluate``.
    """
    completed_keys = sorted(
        {(m.season, m.round) for m in matches if _is_complete(m)},
    )
    if len(completed_keys) <= holdout_rounds:
        return list(matches), []
    cutoff = set(completed_keys[-holdout_rounds:])
    training = [m for m in matches if (m.season, m.round) not in cutoff]
    holdout = [m for m in matches if (m.season, m.round) in cutoff]
    return training, holdout
