"""Tests for evaluation.profit — CLV computation, Kelly sizing, PnL simulation.

All tests use synthetic predictions + odds so no real match data is needed.
"""

from __future__ import annotations

import math

import pytest

from fantasy_coach.evaluation.harness import EvaluationResult, Prediction
from fantasy_coach.evaluation.profit import (
    CLVReport,
    MatchCLV,
    clv_wald_pvalue,
    compute_clv,
    kelly_fraction,
    render_clv_section,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(
    match_id: int,
    p_home: float,
    actual: int,
    *,
    season: int = 2024,
    round_: int = 1,
) -> Prediction:
    return Prediction(
        season=season,
        round=round_,
        match_id=match_id,
        home_id=1,
        away_id=2,
        p_home_win=p_home,
        actual_home_win=actual,
    )


def _result(preds: list[Prediction], name: str = "test") -> EvaluationResult:
    r = EvaluationResult(predictor_name=name)
    r.predictions = preds
    return r


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------


def test_kelly_fraction_positive_edge() -> None:
    # Model says 0.7, odds are 2.0 (p_close = 0.5)
    k = kelly_fraction(0.7, 2.0, kelly_multiplier=0.25)
    full_k = (1.0 * 0.7 - 0.3) / 1.0  # b=1, p=0.7, q=0.3
    assert k == pytest.approx(full_k * 0.25, abs=1e-6)


def test_kelly_fraction_zero_edge_returns_zero() -> None:
    # Model probability equals implied probability (no edge)
    k = kelly_fraction(0.5, 2.0)  # p=0.5, b=1 → f = (1*0.5 - 0.5)/1 = 0
    assert k == 0.0


def test_kelly_fraction_negative_edge_returns_zero() -> None:
    k = kelly_fraction(0.3, 2.0)  # p < implied prob → no bet
    assert k == 0.0


def test_kelly_fraction_never_exceeds_multiplier() -> None:
    # Even with a massive edge, capped at kelly_multiplier.
    k = kelly_fraction(0.99, 2.0, kelly_multiplier=0.25)
    assert k <= 0.25


# ---------------------------------------------------------------------------
# compute_clv — correctness
# ---------------------------------------------------------------------------


def test_compute_clv_returns_nan_when_no_odds() -> None:
    result = _result([_pred(1, 0.6, 1), _pred(2, 0.4, 0)])
    closing_probs: dict[int, float] = {}  # empty — no coverage
    report = compute_clv(result, closing_probs)
    assert report.n_with_odds == 0
    assert math.isnan(report.mean_clv)


def test_compute_clv_signed_correctly() -> None:
    """CLV is positive when model prob > closing prob (model likes home more)."""
    result = _result([_pred(1, 0.65, 1)])  # model says 0.65, market says 0.50
    report = compute_clv(result, {1: 0.50})
    assert report.n_with_odds == 1
    assert report.mean_clv == pytest.approx(0.15, abs=1e-6)


def test_compute_clv_negative_when_model_underestimates_home() -> None:
    result = _result([_pred(1, 0.40, 0)])  # model 0.40, market 0.60
    report = compute_clv(result, {1: 0.60})
    assert report.mean_clv == pytest.approx(-0.20, abs=1e-6)


def test_compute_clv_partial_coverage() -> None:
    """Only matches with odds in closing_probs contribute to CLV."""
    preds = [_pred(1, 0.65, 1), _pred(2, 0.70, 1), _pred(3, 0.55, 0)]
    result = _result(preds)
    closing_probs = {1: 0.50, 3: 0.50}  # match 2 has no closing line
    report = compute_clv(result, closing_probs)
    assert report.n_with_odds == 2
    # CLV for match 1: 0.65 - 0.50 = 0.15; for match 3: 0.55 - 0.50 = 0.05
    assert report.mean_clv == pytest.approx(0.10, abs=1e-6)


def test_compute_clv_cumulative_curve_length_matches_n() -> None:
    preds = [_pred(i, 0.6, 1) for i in range(5)]
    closing_probs = {i: 0.50 for i in range(5)}
    report = compute_clv(_result(preds), closing_probs)
    assert len(report.cumulative_clv) == 5
    assert len(report.bankroll_flat_stake) == 5
    assert len(report.bankroll_quarter_kelly) == 5


def test_compute_clv_pnl_simulation_reproducible() -> None:
    """PnL simulation is deterministic for the same inputs."""
    preds = [_pred(i, 0.6 + 0.01 * i, i % 2) for i in range(10)]
    closing_probs = {i: 0.5 for i in range(10)}
    r1 = compute_clv(_result(preds), closing_probs)
    r2 = compute_clv(_result(preds), closing_probs)
    assert r1.bankroll_flat_stake == r2.bankroll_flat_stake
    assert r1.bankroll_quarter_kelly == r2.bankroll_quarter_kelly


def test_quarter_kelly_stake_bounded_by_quarter_of_current_bankroll() -> None:
    """Kelly stake fraction ≤ 0.25 of *current* bankroll for any single bet."""
    preds = [_pred(i, 0.9, 1) for i in range(20)]  # extreme model confidence
    closing_probs = {i: 0.5 for i in range(20)}
    report = compute_clv(_result(preds), closing_probs, starting_bankroll=100.0)
    # Reconstruct the running bankroll to verify the cap is against current balance.
    running = 100.0
    for record in report.records:
        assert record.kelly_stake <= 0.25 * running + 1e-6, (
            f"Kelly stake {record.kelly_stake:.4f} > 25% of bankroll {running:.4f}"
        )
        running += record.kelly_pnl


def test_compute_clv_flat_stake_correct_win() -> None:
    """A single home win at odds 2.0 with a 1-unit bet returns +1."""
    preds = [_pred(1, 0.7, 1)]  # model likes home, home wins
    closing_probs = {1: 0.5}  # odds = 1/0.5 = 2.0
    report = compute_clv(_result(preds), closing_probs, starting_bankroll=100.0)
    # flat stake: win = home_odds - 1 = 1.0 unit profit
    assert report.bankroll_flat_stake[0] == pytest.approx(101.0, abs=1e-4)


def test_compute_clv_flat_stake_correct_loss() -> None:
    """A single home loss with 1-unit bet on home costs −1."""
    preds = [_pred(1, 0.7, 0)]  # model likes home, home loses
    closing_probs = {1: 0.5}
    report = compute_clv(_result(preds), closing_probs, starting_bankroll=100.0)
    assert report.bankroll_flat_stake[0] == pytest.approx(99.0, abs=1e-4)


# ---------------------------------------------------------------------------
# clv_wald_pvalue
# ---------------------------------------------------------------------------


def test_clv_wald_pvalue_returns_nan_for_small_n() -> None:
    assert math.isnan(clv_wald_pvalue([]))
    assert math.isnan(clv_wald_pvalue([MatchCLV(1, 0.6, 0.5, 1, 0.1, 0.1, 0.01, 0.01)]))


def test_clv_wald_pvalue_zero_variance_returns_nan() -> None:
    records = [MatchCLV(i, 0.6, 0.5, 1, 0.1, 0.1, 0.01, 0.01) for i in range(5)]
    # All CLVs are identical → zero variance → nan
    assert math.isnan(clv_wald_pvalue(records))


def test_clv_wald_pvalue_large_sample_high_power() -> None:
    """With n=400 and a consistent 0.05 edge, p-value should be << 0.05."""
    records = [
        MatchCLV(i, 0.55, 0.50, 1, 0.05 + (0.01 if i % 3 else -0.005), 0.1, 0.01, 0.01)
        for i in range(400)
    ]
    p = clv_wald_pvalue(records)
    assert p < 0.05


# ---------------------------------------------------------------------------
# render_clv_section
# ---------------------------------------------------------------------------


def test_render_clv_section_produces_markdown_table() -> None:
    r = CLVReport(
        model_name="xgboost",
        n_with_odds=200,
        mean_clv=0.0123,
        cumulative_clv=[],
        roi_flat_stake=0.05,
        bankroll_flat_stake=[],
        roi_quarter_kelly=0.08,
        bankroll_quarter_kelly=[105.0],
        records=[],
    )
    md = render_clv_section([r])
    assert "Market efficiency" in md
    assert "xgboost" in md
    assert "+0.0123" in md


def test_render_clv_section_no_odds_row() -> None:
    r = CLVReport(
        model_name="elo",
        n_with_odds=0,
        mean_clv=float("nan"),
        cumulative_clv=[],
        roi_flat_stake=float("nan"),
        bankroll_flat_stake=[],
        roi_quarter_kelly=float("nan"),
        bankroll_quarter_kelly=[],
        records=[],
    )
    md = render_clv_section([r])
    assert "elo" in md
    assert "—" in md
