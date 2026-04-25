"""Closing-line value (CLV) and profit/loss simulation for model evaluation.

Closing-line value answers the economic question: does the model find value
the market misses? A prediction with positive CLV bought probability at a
price the market later corrected upward — a long-run profitable pattern even
on individual losses.

Usage:

    clv_report = compute_clv(eval_result, closing_probs)
    # closing_probs: dict[match_id -> de-vigged home-win probability]

The module is independent of specific predictor types — it works on any
``EvaluationResult`` from the walk-forward harness.

References:
  Lundberg & Lee (2017) — SHAP values (separate module; referenced here for
  the "market-efficient model" framing).
  Sharp Football Analysis — "Closing Line Value" methodology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_coach.evaluation.harness import EvaluationResult, Prediction

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchCLV:
    """Per-match CLV record, used for cumulative curves and per-match detail."""

    match_id: int
    p_model: float
    p_close: float
    actual_home_win: int
    clv: float  # p_model − p_close (positive = model liked home vs market)
    flat_pnl: float  # net PnL from a 1-unit flat stake on the model's pick
    kelly_stake: float  # quarter-Kelly stake in bankroll units
    kelly_pnl: float  # net PnL from the Kelly stake


@dataclass
class CLVReport:
    """Aggregate CLV and PnL statistics for one predictor."""

    model_name: str
    n_with_odds: int  # matches with closing-line coverage
    mean_clv: float  # average (p_model − p_close) across all covered matches
    cumulative_clv: list[float]  # running cumulative CLV (same length as records)
    # Flat-stake (1 unit per match on the model's pick):
    roi_flat_stake: float  # net profit per unit staked (= total_pnl / n)
    bankroll_flat_stake: list[float]  # running bankroll starting from 100 units
    # Quarter-Kelly stake on CLV-positive bets only:
    roi_quarter_kelly: float  # (final − 100) / 100 as a fraction
    bankroll_quarter_kelly: list[float]  # running bankroll starting from 100 units
    records: list[MatchCLV] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------


def kelly_fraction(p_model: float, decimal_odds: float, *, kelly_multiplier: float = 0.25) -> float:
    """Fractional Kelly stake for a single bet.

    ``decimal_odds`` is the price offered (e.g. 1.80). Returns 0 when the bet
    has no edge (p_model × decimal_odds ≤ 1). Capped at the multiplier to
    prevent over-betting on extreme estimates.
    """
    b = decimal_odds - 1.0  # net odds (profit per unit staked)
    if b <= 0:
        return 0.0
    f_full = (b * p_model - (1.0 - p_model)) / b
    return max(0.0, min(kelly_multiplier, f_full * kelly_multiplier))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_clv(
    result: "EvaluationResult",
    closing_probs: dict[int, float],
    *,
    kelly_multiplier: float = 0.25,
    starting_bankroll: float = 100.0,
) -> CLVReport:
    """Compute CLV and simulate P&L for a walk-forward result.

    Parameters
    ----------
    result:
        Walk-forward EvaluationResult from the harness.
    closing_probs:
        Mapping of ``match_id`` → de-vigged closing-line home-win probability.
        Matches not in this dict are excluded from CLV computation.
    kelly_multiplier:
        Fraction of full-Kelly to bet. 0.25 (quarter-Kelly) by default.
    starting_bankroll:
        Starting bankroll for the PnL simulation (100 units by default).
    """
    records: list[MatchCLV] = []
    bankroll_flat = starting_bankroll
    bankroll_kelly = starting_bankroll
    cum_clv = 0.0
    cumulative_clv: list[float] = []
    bankroll_flat_series: list[float] = []
    bankroll_kelly_series: list[float] = []

    for pred in result.predictions:
        p_close = closing_probs.get(pred.match_id)
        if p_close is None:
            continue
        p_close = max(_EPS, min(1.0 - _EPS, p_close))

        p_model = pred.p_home_win
        clv = p_model - p_close

        # --- Flat stake: 1 unit on the model's pick regardless of CLV sign ---
        home_odds = 1.0 / p_close
        away_odds = 1.0 / (1.0 - p_close)
        if p_model >= 0.5:
            # Bet home: win (home_odds − 1) on a win, lose −1 on a loss.
            flat_pnl = (home_odds - 1.0) * pred.actual_home_win - (1 - pred.actual_home_win)
        else:
            # Bet away.
            flat_pnl = (away_odds - 1.0) * (1 - pred.actual_home_win) - pred.actual_home_win
        bankroll_flat += flat_pnl

        # --- Quarter-Kelly: only bet when the model has positive edge ---
        if p_model > p_close:  # model likes home vs market
            k = kelly_fraction(p_model, home_odds, kelly_multiplier=kelly_multiplier)
            stake = k * bankroll_kelly
            kelly_pnl = stake * ((home_odds - 1.0) * pred.actual_home_win - (1 - pred.actual_home_win))
        elif p_model < p_close:  # model likes away vs market
            k = kelly_fraction(1.0 - p_model, away_odds, kelly_multiplier=kelly_multiplier)
            stake = k * bankroll_kelly
            kelly_pnl = stake * ((away_odds - 1.0) * (1 - pred.actual_home_win) - pred.actual_home_win)
        else:
            stake = 0.0
            kelly_pnl = 0.0
        bankroll_kelly += kelly_pnl

        cum_clv += clv
        cumulative_clv.append(round(cum_clv, 6))
        bankroll_flat_series.append(round(bankroll_flat, 4))
        bankroll_kelly_series.append(round(bankroll_kelly, 4))

        records.append(
            MatchCLV(
                match_id=pred.match_id,
                p_model=p_model,
                p_close=p_close,
                actual_home_win=pred.actual_home_win,
                clv=round(clv, 6),
                flat_pnl=round(flat_pnl, 4),
                kelly_stake=round(stake, 4),
                kelly_pnl=round(kelly_pnl, 4),
            )
        )

    n = len(records)
    if n == 0:
        return CLVReport(
            model_name=result.predictor_name,
            n_with_odds=0,
            mean_clv=float("nan"),
            cumulative_clv=[],
            roi_flat_stake=float("nan"),
            bankroll_flat_stake=[],
            roi_quarter_kelly=float("nan"),
            bankroll_quarter_kelly=[],
            records=[],
        )

    mean_clv = cum_clv / n
    roi_flat = (bankroll_flat - starting_bankroll) / n  # net profit per unit staked
    roi_kelly = (bankroll_kelly - starting_bankroll) / starting_bankroll

    return CLVReport(
        model_name=result.predictor_name,
        n_with_odds=n,
        mean_clv=round(mean_clv, 6),
        cumulative_clv=cumulative_clv,
        roi_flat_stake=round(roi_flat, 4),
        bankroll_flat_stake=bankroll_flat_series,
        roi_quarter_kelly=round(roi_kelly, 4),
        bankroll_quarter_kelly=bankroll_kelly_series,
        records=records,
    )


# ---------------------------------------------------------------------------
# Rendering helper (extends report.py output when --profit is passed)
# ---------------------------------------------------------------------------


def render_clv_section(reports: list[CLVReport]) -> str:
    """Return a markdown string summarising CLV and PnL across all models.

    Designed to be appended to the report written by ``evaluation/report.py``.
    """
    lines = [
        "",
        "## Market efficiency (Closing-Line Value)",
        "",
        "Positive CLV means the model found value the market corrected upward — "
        "a +EV pattern over many trials even on individual losses. "
        "Positive CLV on a small sample is not necessarily positive PnL.",
        "",
        "| Model | n w/ odds | Mean CLV | ROI flat | ROI ¼-Kelly | Final ¼-Kelly bankroll |",
        "|-------|-----------|----------|----------|-------------|------------------------|",
    ]
    for r in reports:
        if r.n_with_odds == 0:
            lines.append(f"| {r.model_name} | 0 | — | — | — | — |")
            continue
        final_kelly = r.bankroll_quarter_kelly[-1] if r.bankroll_quarter_kelly else 100.0
        lines.append(
            f"| {r.model_name} | {r.n_with_odds} "
            f"| {r.mean_clv:+.4f} "
            f"| {r.roi_flat_stake:+.4f} "
            f"| {r.roi_quarter_kelly:+.4f} "
            f"| {final_kelly:.1f} |"
        )
    lines.append("")
    lines.append(
        "> Note: positive CLV does not equal positive PnL on a small sample. "
        "Both metrics are reported for completeness. ¼-Kelly bets only CLV-positive matches."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wald-test for CLV significance
# ---------------------------------------------------------------------------


def clv_wald_pvalue(records: list[MatchCLV]) -> float:
    """Two-tailed Wald p-value for H0: mean CLV == 0.

    Uses a normal approximation: z = mean_clv / (std_clv / sqrt(n)).
    Returns ``float('nan')`` when n < 2.
    """
    n = len(records)
    if n < 2:
        return float("nan")
    clvs = [r.clv for r in records]
    mean = sum(clvs) / n
    var = sum((c - mean) ** 2 for c in clvs) / (n - 1)
    if var <= 0:
        return float("nan")
    z = mean / math.sqrt(var / n)
    # Two-tailed p-value via the complementary error function.
    p = math.erfc(abs(z) / math.sqrt(2))
    return round(p, 6)
