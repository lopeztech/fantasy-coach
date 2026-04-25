"""Tests for the CLV + profit-based evaluation harness (evaluation/profit.py).

Tests use synthetic predictions and odds so no real match data is required.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from fantasy_coach.evaluation.harness import EvaluationResult
from fantasy_coach.evaluation.harness import Prediction as HarnessPrediction
from fantasy_coach.evaluation.profit import (
    CLVReport,
    MatchCLV,
    compute_clv,
    kelly_stake,
    simulate_pnl,
)
from fantasy_coach.features import MatchRow, TeamRow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_match_row(
    match_id: int,
    home_odds: float | None,
    away_odds: float | None,
    home_score: int = 20,
    away_score: int = 14,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=2024,
        round=1,
        start_time=datetime(2024, 3, 7, 9, 0, tzinfo=UTC),
        match_state="FullTime",
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(
            team_id=1,
            name="Home",
            nick_name="Home",
            score=home_score,
            players=[],
            odds=home_odds,
        ),
        away=TeamRow(
            team_id=2,
            name="Away",
            nick_name="Away",
            score=away_score,
            players=[],
            odds=away_odds,
        ),
        team_stats=[],
    )


def _make_eval_result(
    predictions: list[tuple[int, float, int]],  # (match_id, p_home_win, actual)
    name: str = "test_model",
) -> EvaluationResult:
    result = EvaluationResult(predictor_name=name)
    result.predictions = [
        HarnessPrediction(
            season=2024,
            round=1,
            match_id=mid,
            home_id=1,
            away_id=2,
            p_home_win=p,
            actual_home_win=actual,
        )
        for mid, p, actual in predictions
    ]
    return result


def _make_clv(
    p_model: float,
    p_close: float,
    actual: int,
    home_odds: float = 1.8,
    away_odds: float = 2.1,
) -> MatchCLV:
    return MatchCLV(
        match_id=1,
        season=2024,
        round=1,
        p_model=p_model,
        p_close=p_close,
        actual=actual,
        clv=round(p_model - p_close, 6),
        home_decimal_odds=home_odds,
        away_decimal_odds=away_odds,
    )


# ---------------------------------------------------------------------------
# kelly_stake
# ---------------------------------------------------------------------------


def test_kelly_stake_returns_zero_when_no_edge() -> None:
    # p * odds - 1 <= 0: no edge
    stake = kelly_stake(p_model=0.4, decimal_odds=2.0, bankroll=100.0)
    assert stake == 0.0


def test_kelly_stake_never_exceeds_quarter_bankroll() -> None:
    # Even with absurdly high p_model, quarter-Kelly never stakes > 25%
    stake = kelly_stake(p_model=0.99, decimal_odds=2.0, bankroll=100.0, kelly_fraction=0.25)
    assert stake <= 25.0


def test_kelly_stake_proportional_to_bankroll() -> None:
    s1 = kelly_stake(p_model=0.6, decimal_odds=2.0, bankroll=100.0)
    s2 = kelly_stake(p_model=0.6, decimal_odds=2.0, bankroll=200.0)
    assert s2 == pytest.approx(2.0 * s1, rel=1e-6)


def test_kelly_stake_zero_for_degenerate_odds() -> None:
    assert kelly_stake(p_model=0.8, decimal_odds=0.5, bankroll=100.0) == 0.0
    assert kelly_stake(p_model=0.8, decimal_odds=1.0, bankroll=100.0) == 0.0


def test_kelly_stake_zero_bankroll() -> None:
    assert kelly_stake(p_model=0.7, decimal_odds=2.0, bankroll=0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_clv
# ---------------------------------------------------------------------------


def _make_enough_matches(n: int = 15) -> tuple[EvaluationResult, list[MatchRow]]:
    """Return n predictions + matching MatchRows with odds."""
    preds = [(i, 0.65, 1) for i in range(1, n + 1)]
    eval_result = _make_eval_result(preds)
    matches = [_make_match_row(i, home_odds=1.8, away_odds=2.1) for i in range(1, n + 1)]
    return eval_result, matches


def test_compute_clv_returns_report_with_enough_coverage() -> None:
    eval_result, matches = _make_enough_matches(15)
    report = compute_clv(eval_result, matches)
    assert report is not None
    assert report.n == 15


def test_compute_clv_returns_none_when_too_few_matches() -> None:
    eval_result, matches = _make_enough_matches(5)
    report = compute_clv(eval_result, matches)
    assert report is None


def test_compute_clv_skips_matches_without_odds() -> None:
    eval_result = _make_eval_result([(i, 0.65, 1) for i in range(1, 21)])
    # Only first 15 have odds
    matches = [_make_match_row(i, home_odds=1.8, away_odds=2.1) for i in range(1, 16)]
    matches += [_make_match_row(i, home_odds=None, away_odds=None) for i in range(16, 21)]
    report = compute_clv(eval_result, matches)
    assert report is not None
    assert report.n == 15


def test_compute_clv_correctly_signed() -> None:
    """When model is consistently higher than market, mean CLV is positive."""
    # de-vigged prob for 1.8/2.1 = (1/1.8) / (1/1.8 + 1/2.1)
    p_close = (1 / 1.8) / (1 / 1.8 + 1 / 2.1)
    p_model_high = p_close + 0.05  # model beats market

    eval_result, matches = _make_enough_matches(20)
    for p in eval_result.predictions:
        object.__setattr__(p, "p_home_win", p_model_high)
    report = compute_clv(eval_result, matches)
    assert report is not None
    assert report.mean_clv > 0


def test_compute_clv_negative_when_model_below_market() -> None:
    p_close = (1 / 1.8) / (1 / 1.8 + 1 / 2.1)
    p_model_low = p_close - 0.05

    eval_result, matches = _make_enough_matches(20)
    for p in eval_result.predictions:
        object.__setattr__(p, "p_home_win", p_model_low)
    report = compute_clv(eval_result, matches)
    assert report is not None
    assert report.mean_clv < 0


def test_compute_clv_clv_field_equals_p_model_minus_p_close() -> None:
    eval_result, matches = _make_enough_matches(15)
    report = compute_clv(eval_result, matches)
    assert report is not None
    for c in report.match_clvs:
        assert c.clv == pytest.approx(c.p_model - c.p_close, abs=1e-6)


# ---------------------------------------------------------------------------
# simulate_pnl
# ---------------------------------------------------------------------------


def test_simulate_pnl_returns_series_same_length_as_input() -> None:
    clvs = [_make_clv(0.65, 0.55, 1) for _ in range(10)]
    series = simulate_pnl(clvs)
    assert len(series) == 10


def test_simulate_pnl_bankroll_increases_on_winning_bet() -> None:
    # Single bet, model picks home (p=0.7 > 0.5), home wins.
    clvs = [_make_clv(p_model=0.7, p_close=0.5, actual=1, home_odds=1.8)]
    series = simulate_pnl(clvs, starting_bankroll=100.0, strategy="flat")
    assert series[0] > 100.0


def test_simulate_pnl_bankroll_decreases_on_losing_bet() -> None:
    # Single bet, model picks home (p=0.7 > 0.5), away wins.
    clvs = [_make_clv(p_model=0.7, p_close=0.5, actual=0, home_odds=1.8)]
    series = simulate_pnl(clvs, starting_bankroll=100.0, strategy="flat")
    assert series[0] < 100.0


def test_simulate_pnl_quarter_kelly_never_bets_above_25_percent() -> None:
    """Under quarter-Kelly, no single stake exceeds 25% of current bankroll."""
    rng = __import__("random").Random(42)
    clvs = [
        _make_clv(
            p_model=rng.uniform(0.55, 0.90),
            p_close=0.50,
            actual=1,
            home_odds=1.8,
        )
        for _ in range(50)
    ]
    bankroll = 100.0
    for c in clvs:
        stake = kelly_stake(c.p_model, c.home_decimal_odds, bankroll)
        assert stake <= bankroll * 0.25 + 1e-9
        bankroll += stake * (c.home_decimal_odds - 1.0)  # all win for this test


def test_simulate_pnl_flat_strategy_stakes_one_unit_always() -> None:
    clvs = [_make_clv(p_model=0.7, p_close=0.5, actual=1, home_odds=2.0) for _ in range(5)]
    series = simulate_pnl(clvs, starting_bankroll=1000.0, strategy="flat")
    # Each win returns 1.0 * (2.0 - 1.0) = 1.0; all 5 win
    assert series[-1] == pytest.approx(1005.0, abs=1e-6)


def test_simulate_pnl_reproducible() -> None:
    clvs = [_make_clv(0.65, 0.55, i % 2) for i in range(20)]
    s1 = simulate_pnl(clvs)
    s2 = simulate_pnl(clvs)
    assert s1 == s2


# ---------------------------------------------------------------------------
# CLVReport properties
# ---------------------------------------------------------------------------


def test_clv_report_cumulative_clv_length_matches_n() -> None:
    report = CLVReport(predictor_name="test")
    report.match_clvs = [_make_clv(0.6, 0.55, 1) for _ in range(10)]
    assert len(report.cumulative_clv) == 10


def test_clv_report_cumulative_clv_last_equals_sum() -> None:
    clvs = [_make_clv(0.65, 0.55, 1) for _ in range(10)]
    report = CLVReport(predictor_name="test")
    report.match_clvs = clvs
    expected_total = sum(c.clv for c in clvs)
    assert report.cumulative_clv[-1] == pytest.approx(expected_total, abs=1e-5)


def test_clv_report_empty_is_nan() -> None:
    report = CLVReport(predictor_name="test")
    assert math.isnan(report.mean_clv)
    assert math.isnan(report.win_rate)
    assert math.isnan(report.roi_flat)
