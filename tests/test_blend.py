"""Unit tests for the logit-space probability blend (models/blend.py)."""

from __future__ import annotations

import math

import pytest

from fantasy_coach.models.blend import (
    ELO_MOV_BLEND_WEIGHT,
    MARKET_BLEND_WEIGHT,
    MODEL_BLEND_WEIGHT,
    blend_home_win_prob,
)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def test_weights_sum_to_one() -> None:
    assert pytest.approx(1.0) == MODEL_BLEND_WEIGHT + ELO_MOV_BLEND_WEIGHT + MARKET_BLEND_WEIGHT


def test_full_blend_matches_manual_logit_mix() -> None:
    expected = _sigmoid(
        MODEL_BLEND_WEIGHT * _logit(0.55)
        + ELO_MOV_BLEND_WEIGHT * _logit(0.62)
        + MARKET_BLEND_WEIGHT * _logit(0.70)
    )
    assert blend_home_win_prob(0.55, 0.62, 0.70) == pytest.approx(expected)


def test_missing_market_renormalises_over_model_and_elo() -> None:
    total = MODEL_BLEND_WEIGHT + ELO_MOV_BLEND_WEIGHT
    expected = _sigmoid(
        (MODEL_BLEND_WEIGHT * _logit(0.55) + ELO_MOV_BLEND_WEIGHT * _logit(0.62)) / total
    )
    assert blend_home_win_prob(0.55, 0.62, None) == pytest.approx(expected)


def test_all_signals_missing_returns_model_prob() -> None:
    # Renormalisation over the single remaining weight is the identity.
    assert blend_home_win_prob(0.55, None, None) == pytest.approx(0.55)


def test_identical_inputs_are_a_fixed_point() -> None:
    assert blend_home_win_prob(0.65, 0.65, 0.65) == pytest.approx(0.65)


def test_extreme_probabilities_are_clipped_not_infinite() -> None:
    p = blend_home_win_prob(1.0, 0.0, 0.5)
    assert 0.0 < p < 1.0
