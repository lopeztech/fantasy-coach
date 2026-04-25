"""Tests for evaluation/profit.py — CLV and PnL simulation.

Tests use synthetic predictions + closing-line data to verify:
- CLV is correctly signed (positive when model beats market, negative otherwise)
- Quarter-Kelly never bets > 25% of current bankroll
- PnL simulation is reproducible (deterministic given sorted input)
- CLVReport fields have correct types and relationships
- render_profit_section produces valid markdown
"""

from __future__ import annotations

import pytest

from fantasy_coach.evaluation.profit import (
    CLVEntry,
    CLVReport,
    compute_clv,
    kelly_stake,
    render_profit_section,
    simulate_pnl,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _entry(
    match_id: int,
    p_model: float,
    p_close: float,
    *,
    home_odds: float = 1.9,
    away_odds: float = 1.9,
    actual_home_win: int = 1,
    predicted_home: bool = True,
    season: int = 2024,
    round_: int = 1,
) -> CLVEntry:
    return CLVEntry(
        match_id=match_id,
        season=season,
        round=round_,
        p_model=p_model,
        p_close=p_close,
        home_decimal_odds=home_odds,
        away_decimal_odds=away_odds,
        actual_home_win=actual_home_win,
        predicted_home=predicted_home,
    )


def _entries_all_home_bets(n: int = 50) -> list[CLVEntry]:
    """n synthetic matches where model always bets home and always wins."""
    return [_entry(i, 0.6, 0.5, actual_home_win=1) for i in range(n)]


def _entries_all_model_beats_market(n: int = 50) -> list[CLVEntry]:
    """Model probability exceeds close for every bet."""
    return [_entry(i, 0.65, 0.55, actual_home_win=1) for i in range(n)]


def _entries_model_trails_market(n: int = 50) -> list[CLVEntry]:
    """Market is always better calibrated (p_model < p_close)."""
    return [_entry(i, 0.45, 0.55, actual_home_win=1) for i in range(n)]


# ---------------------------------------------------------------------------
# kelly_stake
# ---------------------------------------------------------------------------


class TestKellyStake:
    def test_positive_ev_yields_positive_stake(self) -> None:
        # p=0.6, decimal_odds=2.0 → full Kelly = (0.6*1 - 0.4)/1 = 0.2 → qk=0.05 * 100
        stake = kelly_stake(0.6, 2.0, 100.0)
        assert stake > 0.0

    def test_zero_ev_yields_zero_stake(self) -> None:
        # p=0.5 on even money (2.0 odds): full Kelly = 0 → stake = 0
        stake = kelly_stake(0.5, 2.0, 100.0)
        assert stake == pytest.approx(0.0, abs=1e-6)

    def test_negative_ev_yields_zero_stake(self) -> None:
        # p=0.4 on even money: Kelly < 0 → clip to 0
        stake = kelly_stake(0.4, 2.0, 100.0)
        assert stake == pytest.approx(0.0)

    def test_capped_at_quarter_kelly_fraction(self) -> None:
        # Extremely high p → full Kelly large → must be capped at kelly_fraction * bankroll
        stake = kelly_stake(0.99, 1.5, 100.0, kelly_fraction=0.25)
        assert stake <= 0.25 * 100.0

    def test_proportional_to_bankroll(self) -> None:
        s1 = kelly_stake(0.6, 2.0, 100.0)
        s2 = kelly_stake(0.6, 2.0, 200.0)
        assert s2 == pytest.approx(2.0 * s1, rel=1e-6)

    def test_invalid_odds_returns_zero(self) -> None:
        assert kelly_stake(0.6, 1.0, 100.0) == 0.0
        assert kelly_stake(0.6, 0.5, 100.0) == 0.0

    def test_custom_kelly_fraction(self) -> None:
        full = kelly_stake(0.6, 2.0, 100.0, kelly_fraction=1.0)
        half = kelly_stake(0.6, 2.0, 100.0, kelly_fraction=0.5)
        qk = kelly_stake(0.6, 2.0, 100.0, kelly_fraction=0.25)
        assert full == pytest.approx(2.0 * half, rel=1e-6)
        assert half == pytest.approx(2.0 * qk, rel=1e-6)


# ---------------------------------------------------------------------------
# simulate_pnl
# ---------------------------------------------------------------------------


class TestSimulatePnl:
    def test_same_length_as_entries(self) -> None:
        entries = _entries_all_home_bets(20)
        trace = simulate_pnl(entries)
        assert len(trace) == 20

    def test_bankroll_grows_on_consecutive_wins(self) -> None:
        entries = [_entry(i, 0.6, 0.5, actual_home_win=1) for i in range(10)]
        trace = simulate_pnl(entries, strategy="quarter_kelly")
        assert trace[-1].cumulative_bankroll > 100.0

    def test_bankroll_shrinks_on_consecutive_losses(self) -> None:
        entries = [_entry(i, 0.6, 0.5, actual_home_win=0) for i in range(10)]
        trace = simulate_pnl(entries, strategy="quarter_kelly")
        assert trace[-1].cumulative_bankroll < 100.0

    def test_quarter_kelly_never_exceeds_25pct_of_bankroll(self) -> None:
        entries = [_entry(i, 0.95, 0.5, actual_home_win=1) for i in range(30)]
        trace = simulate_pnl(entries, strategy="quarter_kelly")
        # Verify each stake does not exceed 25% of the *prior* bankroll.
        bankroll = 100.0
        for m in trace:
            assert m.stake <= 0.26 * bankroll  # slight slack for rounding
            bankroll = m.cumulative_bankroll

    def test_flat_stakes_always_same(self) -> None:
        entries = _entries_all_home_bets(15)
        trace = simulate_pnl(entries, strategy="flat", flat_stake=2.0)
        for m in trace:
            assert m.stake == pytest.approx(2.0)

    def test_pnl_sign_matches_outcome(self) -> None:
        won = _entry(1, 0.6, 0.5, actual_home_win=1)
        lost = _entry(2, 0.6, 0.5, actual_home_win=0)
        trace = simulate_pnl([won, lost], strategy="flat")
        assert trace[0].pnl > 0.0
        assert trace[1].pnl < 0.0

    def test_reproducible_given_same_input(self) -> None:
        entries = _entries_all_home_bets(40)
        t1 = simulate_pnl(entries)
        t2 = simulate_pnl(entries)
        for a, b in zip(t1, t2):
            assert a.cumulative_bankroll == pytest.approx(b.cumulative_bankroll)


# ---------------------------------------------------------------------------
# compute_clv
# ---------------------------------------------------------------------------


class TestComputeClv:
    def test_positive_clv_when_model_beats_market(self) -> None:
        entries = _entries_all_model_beats_market()
        report = compute_clv(entries, "test")
        assert report.mean_clv > 0.0
        assert report.clv_positive_rate == pytest.approx(1.0)

    def test_negative_clv_when_model_trails_market(self) -> None:
        entries = _entries_model_trails_market()
        report = compute_clv(entries, "test")
        assert report.mean_clv < 0.0
        assert report.clv_positive_rate == pytest.approx(0.0)

    def test_empty_entries_returns_nan_report(self) -> None:
        report = compute_clv([], "empty", n_total=10)
        assert report.n_with_odds == 0
        assert report.n_total == 10
        import math

        assert math.isnan(report.mean_clv)

    def test_win_rate_correct(self) -> None:
        entries = [
            _entry(1, 0.6, 0.5, actual_home_win=1),  # win
            _entry(2, 0.6, 0.5, actual_home_win=0),  # loss
            _entry(3, 0.6, 0.5, actual_home_win=1),  # win
            _entry(4, 0.6, 0.5, actual_home_win=1),  # win
        ]
        report = compute_clv(entries, "test")
        assert report.win_rate == pytest.approx(0.75)

    def test_n_total_defaults_to_len_entries(self) -> None:
        entries = _entries_all_home_bets(30)
        report = compute_clv(entries, "test")
        assert report.n_total == 30
        assert report.n_with_odds == 30

    def test_n_total_override(self) -> None:
        entries = _entries_all_home_bets(30)
        report = compute_clv(entries, "test", n_total=50)
        assert report.n_total == 50
        assert report.n_with_odds == 30

    def test_clv_matches_manual_calculation(self) -> None:
        # Single home bet: p_model=0.65, p_close=0.55 → CLV = 0.10
        entries = [_entry(1, 0.65, 0.55, actual_home_win=1, predicted_home=True)]
        report = compute_clv(entries, "test")
        assert report.mean_clv == pytest.approx(0.10, abs=1e-6)

    def test_away_bet_clv_computed_correctly(self) -> None:
        # Away bet: p_model=0.4 (40% home), predicted_home=False.
        # p_model_away = 1 - 0.4 = 0.6, p_close_away = 1 - 0.5 = 0.5 → CLV = 0.1
        entries = [_entry(1, 0.4, 0.5, actual_home_win=0, predicted_home=False)]
        report = compute_clv(entries, "test")
        assert report.mean_clv == pytest.approx(0.10, abs=1e-6)

    def test_roi_sign_matches_pnl(self) -> None:
        # All wins → PnL positive → ROI positive.
        entries = _entries_all_home_bets(20)
        report = compute_clv(entries, "test")
        assert report.total_pnl_quarter_kelly > 0.0
        assert report.roi_quarter_kelly > 0.0

    def test_ending_bankroll_gt_starting_on_all_wins(self) -> None:
        entries = _entries_all_home_bets(30)
        report = compute_clv(entries, "test")
        assert report.ending_bankroll > report.starting_bankroll

    def test_match_clv_length_equals_entries(self) -> None:
        entries = _entries_all_home_bets(25)
        report = compute_clv(entries, "test")
        assert len(report.match_clv) == 25

    def test_report_fields_have_correct_types(self) -> None:
        entries = _entries_all_home_bets(15)
        report = compute_clv(entries, "test")
        assert isinstance(report, CLVReport)
        assert isinstance(report.predictor_name, str)
        assert isinstance(report.n_total, int)
        assert isinstance(report.n_with_odds, int)
        assert isinstance(report.mean_clv, float)
        assert isinstance(report.win_rate, float)
        assert isinstance(report.roi_quarter_kelly, float)


# ---------------------------------------------------------------------------
# render_profit_section
# ---------------------------------------------------------------------------


class TestRenderProfitSection:
    def test_returns_empty_on_no_reports(self) -> None:
        assert render_profit_section([]) == ""

    def test_returns_empty_when_all_no_odds(self) -> None:
        import math

        r = CLVReport(
            predictor_name="x",
            n_total=10,
            n_with_odds=0,
            mean_clv=math.nan,
            clv_positive_rate=math.nan,
            win_rate=math.nan,
            roi_quarter_kelly=math.nan,
            roi_flat=math.nan,
            total_pnl_quarter_kelly=math.nan,
            total_pnl_flat=math.nan,
            starting_bankroll=100.0,
            ending_bankroll=100.0,
        )
        assert render_profit_section([r]) == ""

    def test_contains_predictor_name(self) -> None:
        entries = _entries_all_home_bets(10)
        report = compute_clv(entries, "my_model")
        md = render_profit_section([report])
        assert "my_model" in md

    def test_contains_clv_header(self) -> None:
        entries = _entries_all_home_bets(10)
        report = compute_clv(entries, "m")
        md = render_profit_section([report])
        assert "CLV" in md
        assert "Bankroll" in md

    def test_valid_markdown_table_structure(self) -> None:
        entries = _entries_all_home_bets(25)
        report = compute_clv(entries, "xgboost")
        md = render_profit_section([report])
        # Every table row should have consistent pipe counts.
        rows = [l for l in md.splitlines() if l.startswith("|")]
        for row in rows:
            assert row.endswith("|"), f"Table row not closed: {row!r}"
