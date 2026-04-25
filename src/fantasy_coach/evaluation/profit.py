"""Closing-line value (CLV) and profit-simulation evaluation.

CLV measures the economic edge of a model as a *market participant*, not just
a forecaster.  A model that consistently finds value the market later corrects
(CLV-positive) is more valuable than one that is equally accurate but only
agrees with the market.

Definitions used here
---------------------
- **p_model**: model's probability for the *home team* winning.
- **p_close**: de-vigged closing-line probability for the home team.
- **CLV (per match)**: ``p_model_picked − p_close_picked`` where ``_picked``
  means the side the model backed.  Positive = model found value.
- **Quarter-Kelly stake**: ``f = 0.25 × (p × b − q) / b`` where ``b =
  decimal_odds − 1``, ``p`` = model prob for picked side, ``q = 1 − p``.
  Clipped to [0, 0.25 × bankroll] so a single bet never exceeds 25 % of funds.
- **PnL simulation**: sequential bankroll walk starting at 100 units.  When
  the bet wins, bankroll += stake × (odds − 1); when it loses, bankroll −= stake.

Neither CLV nor PnL over a season-length sample (≈ 200 matches) is
statistically significant on its own — the signal-to-noise ratio is low.
Report both but always note the sample size limitation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

_STRATEGY = Literal["quarter_kelly", "flat"]


@dataclass(frozen=True)
class CLVEntry:
    """One prediction matched to its closing-line counterpart.

    ``predicted_home``: True when the model backed the home side.
    ``actual_home_win``: 1 if home won, 0 if away won (draws excluded).
    ``home_decimal_odds`` / ``away_decimal_odds``: closing decimal odds
    (≥ 1.01) — used for Kelly sizing.  May be ``None`` when no closing-line
    data is available for the match; those entries are skipped in CLV metrics.
    ``p_close``: de-vigged closing-line probability for home win; derived from
    the closing odds via ``1/home − (1/home + 1/away )``.
    """

    match_id: int
    season: int
    round: int
    p_model: float  # model home-win probability (0–1)
    p_close: float  # de-vigged closing home-win probability (0–1)
    home_decimal_odds: float  # closing decimal odds for home team
    away_decimal_odds: float  # closing decimal odds for away team
    actual_home_win: int  # 1 or 0
    predicted_home: bool  # True when model picked home


@dataclass(frozen=True)
class MatchCLV:
    """Per-match CLV and bet result."""

    match_id: int
    season: int
    round: int
    clv: float  # p_model_picked − p_close_picked
    stake: float  # units bet (Kelly or flat)
    pnl: float  # profit/loss on this bet in units
    cumulative_bankroll: float  # running bankroll after this bet


@dataclass
class CLVReport:
    """Aggregated CLV and PnL metrics for one model over a sample."""

    predictor_name: str
    n_total: int  # predictions before closing-line filter
    n_with_odds: int  # predictions with usable closing-line data
    mean_clv: float  # average (p_model_picked − p_close_picked)
    clv_positive_rate: float  # fraction of bets with CLV > 0
    win_rate: float  # fraction of bets won (actual outcome agreed with pick)
    roi_quarter_kelly: float  # (total PnL) / (total staked) under quarter-Kelly
    roi_flat: float  # flat-1-unit-per-bet ROI
    total_pnl_quarter_kelly: float  # net profit/loss in units under quarter-Kelly
    total_pnl_flat: float  # net profit/loss in units under flat staking
    starting_bankroll: float  # always 100.0 (quarter-Kelly simulation)
    ending_bankroll: float  # bankroll at end of simulation
    match_clv: list[MatchCLV] = field(default_factory=list)  # per-match trace


def kelly_stake(
    p_model: float,
    decimal_odds: float,
    bankroll: float,
    *,
    kelly_fraction: float = 0.25,
) -> float:
    """Compute a fractional-Kelly stake in the same units as ``bankroll``.

    Returns 0 when the bet is negative-EV (Kelly fraction would be negative).
    Clips to ``kelly_fraction × bankroll`` so no single bet exceeds that cap.

    ``p_model``: model's probability for the bet side winning (0, 1).
    ``decimal_odds``: closing decimal odds for the bet side (> 1).
    ``kelly_fraction``: multiplier on the full-Kelly fraction (0.25 = quarter-Kelly).
    """
    if decimal_odds <= 1.0 or p_model <= 0.0 or p_model >= 1.0:
        return 0.0
    b = decimal_odds - 1.0  # net payout per unit stake if won
    q = 1.0 - p_model
    full_kelly = (p_model * b - q) / b
    if full_kelly <= 0.0:
        return 0.0
    stake = kelly_fraction * full_kelly * bankroll
    max_stake = kelly_fraction * bankroll
    return min(stake, max_stake)


def simulate_pnl(
    entries: list[CLVEntry],
    *,
    strategy: _STRATEGY = "quarter_kelly",
    kelly_fraction: float = 0.25,
    flat_stake: float = 1.0,
    starting_bankroll: float = 100.0,
) -> list[MatchCLV]:
    """Simulate sequential betting over ``entries``, returning a per-match trace.

    Bets are placed in order (entries should be sorted chronologically by the
    caller).  The bankroll evolves with each bet; Kelly sizing is computed
    against the current bankroll at the time of the bet.

    ``strategy="quarter_kelly"``: bet kelly_stake units based on current
    bankroll.  ``strategy="flat"``: bet ``flat_stake`` units regardless of
    bankroll.

    Each ``MatchCLV.pnl`` is the profit/loss on that bet; ``cumulative_bankroll``
    is the bankroll *after* the bet settles.
    """
    bankroll = starting_bankroll
    trace: list[MatchCLV] = []
    for e in entries:
        p_picked = e.p_model if e.predicted_home else (1.0 - e.p_model)
        decimal_odds = e.home_decimal_odds if e.predicted_home else e.away_decimal_odds
        p_close_picked = e.p_close if e.predicted_home else (1.0 - e.p_close)
        clv = p_picked - p_close_picked

        if strategy == "quarter_kelly":
            stake = kelly_stake(p_picked, decimal_odds, bankroll, kelly_fraction=kelly_fraction)
        else:
            stake = flat_stake

        won = (e.predicted_home and e.actual_home_win == 1) or (
            not e.predicted_home and e.actual_home_win == 0
        )
        if won:
            pnl = stake * (decimal_odds - 1.0)
        else:
            pnl = -stake
        bankroll += pnl

        trace.append(
            MatchCLV(
                match_id=e.match_id,
                season=e.season,
                round=e.round,
                clv=round(clv, 6),
                stake=round(stake, 4),
                pnl=round(pnl, 4),
                cumulative_bankroll=round(bankroll, 4),
            )
        )
    return trace


def compute_clv(
    entries: list[CLVEntry],
    predictor_name: str = "model",
    *,
    n_total: int | None = None,
    kelly_fraction: float = 0.25,
    flat_stake: float = 1.0,
    starting_bankroll: float = 100.0,
) -> CLVReport:
    """Compute CLV and PnL metrics from a list of matched prediction + closing-line pairs.

    ``entries`` should contain only predictions that have closing-line data
    (the caller filters out None-odds rows).  ``n_total`` is the total
    prediction count before filtering (for coverage reporting); defaults to
    ``len(entries)``.
    """
    if not entries:
        return CLVReport(
            predictor_name=predictor_name,
            n_total=n_total if n_total is not None else 0,
            n_with_odds=0,
            mean_clv=float("nan"),
            clv_positive_rate=float("nan"),
            win_rate=float("nan"),
            roi_quarter_kelly=float("nan"),
            roi_flat=float("nan"),
            total_pnl_quarter_kelly=float("nan"),
            total_pnl_flat=float("nan"),
            starting_bankroll=starting_bankroll,
            ending_bankroll=starting_bankroll,
        )

    clv_values: list[float] = []
    for e in entries:
        p_picked = e.p_model if e.predicted_home else (1.0 - e.p_model)
        p_close_picked = e.p_close if e.predicted_home else (1.0 - e.p_close)
        clv_values.append(p_picked - p_close_picked)

    wins = sum(
        1
        for e in entries
        if (e.predicted_home and e.actual_home_win == 1)
        or (not e.predicted_home and e.actual_home_win == 0)
    )

    qk_trace = simulate_pnl(
        entries,
        strategy="quarter_kelly",
        kelly_fraction=kelly_fraction,
        starting_bankroll=starting_bankroll,
    )
    flat_trace = simulate_pnl(entries, strategy="flat", flat_stake=flat_stake)

    total_staked_qk = sum(abs(m.stake) for m in qk_trace)
    total_pnl_qk = sum(m.pnl for m in qk_trace)
    roi_qk = total_pnl_qk / total_staked_qk if total_staked_qk > 0 else float("nan")

    total_staked_flat = flat_stake * len(entries)
    total_pnl_flat = sum(m.pnl for m in flat_trace)
    roi_flat = total_pnl_flat / total_staked_flat if total_staked_flat > 0 else float("nan")

    ending_bankroll = qk_trace[-1].cumulative_bankroll if qk_trace else starting_bankroll

    return CLVReport(
        predictor_name=predictor_name,
        n_total=n_total if n_total is not None else len(entries),
        n_with_odds=len(entries),
        mean_clv=sum(clv_values) / len(clv_values),
        clv_positive_rate=sum(1 for v in clv_values if v > 0) / len(clv_values),
        win_rate=wins / len(entries),
        roi_quarter_kelly=roi_qk,
        roi_flat=roi_flat,
        total_pnl_quarter_kelly=total_pnl_qk,
        total_pnl_flat=total_pnl_flat,
        starting_bankroll=starting_bankroll,
        ending_bankroll=ending_bankroll,
        match_clv=qk_trace,
    )


def render_profit_section(clv_reports: list[CLVReport]) -> str:
    """Render CLV + PnL summary as a markdown section.

    Designed to be appended to the output of ``evaluation/report.py``'s
    ``render_markdown``.  Returns an empty string when no CLV data is present.
    """
    if not clv_reports or all(r.n_with_odds == 0 for r in clv_reports):
        return ""

    lines = [
        "## Market efficiency (CLV + PnL)",
        "",
        "CLV = average(p_model_picked − p_close_picked).  "
        "Positive = model consistently finds value the closing line corrects.  "
        "PnL simulated on a 100-unit bankroll (quarter-Kelly) and flat 1-unit stakes.  "
        "**Caution**: n < 400 bets is insufficient for statistical significance.",
        "",
        "| Model | n (odds) | Mean CLV | CLV+ rate | Win rate | ROI (QK) | PnL (QK) | ROI (flat) |",
        "|-------|----------|----------|-----------|----------|----------|----------|------------|",
    ]

    def _fmt(v: float, fmt: str = ".4f") -> str:
        return f"{v:{fmt}}" if math.isfinite(v) else "n/a"

    for r in clv_reports:
        lines.append(
            f"| {r.predictor_name} | {r.n_with_odds}/{r.n_total} | "
            f"{_fmt(r.mean_clv)} | {_fmt(r.clv_positive_rate, '.1%')} | "
            f"{_fmt(r.win_rate, '.1%')} | {_fmt(r.roi_quarter_kelly, '.1%')} | "
            f"{_fmt(r.total_pnl_quarter_kelly, '.1f')} | {_fmt(r.roi_flat, '.1%')} |"
        )

    lines.append("")
    lines.append("### Cumulative bankroll (quarter-Kelly, 100 units start)")
    lines.append("")
    lines.append("Every 10th match shown; final row is the last bet.")
    lines.append("")

    for r in clv_reports:
        if not r.match_clv:
            continue
        lines.append(f"**{r.predictor_name}**")
        lines.append("")
        lines.append("| # | Season | Round | CLV | Bankroll |")
        lines.append("|---|--------|-------|-----|----------|")
        sample = [
            (i, m) for i, m in enumerate(r.match_clv, 1) if i % 10 == 0 or i == len(r.match_clv)
        ]
        for i, m in sample:
            lines.append(
                f"| {i} | {m.season} | {m.round} | "
                f"{'+' if m.clv >= 0 else ''}{m.clv:.4f} | {m.cumulative_bankroll:.1f} |"
            )
        lines.append("")

    return "\n".join(lines)
