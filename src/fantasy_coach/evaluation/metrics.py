"""Scoring metrics for the walk-forward harness."""

from __future__ import annotations

import math
from collections.abc import Sequence

EPS = 1e-15


def accuracy(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    if not probs:
        return float("nan")
    correct = sum(1 for p, y in zip(probs, outcomes, strict=True) if (p >= 0.5) == (y == 1))
    return correct / len(probs)


def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    if not probs:
        return float("nan")
    total = 0.0
    for p, y in zip(probs, outcomes, strict=True):
        p = min(max(p, EPS), 1.0 - EPS)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(probs)


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes, strict=True)) / len(probs)


def ece(probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error — delegates to the calibration module."""
    from fantasy_coach.models.calibration import ece as _ece

    return _ece(probs, outcomes, n_bins=n_bins)
