"""Closing-line value (CLV) and profit-based evaluation.

CLV measures whether the model finds value the market misses:

    clv = p_model − p_close

A positive CLV means the model assigned a higher home-win probability than
the de-vigged closing line; the model was more bullish on home than the
market settled at. Over a large sample, consistently positive CLV is the
near-unfakeable indicator of a model with genuine edge — even if individual
bets lose, consistently betting above the closing line is a +EV pattern.

Usage:

    from fantasy_coach.evaluation.profit import compute_clv, simulate_pnl

    report = compute_clv(eval_result, match_rows)
    if report:
        bankroll_series = simulate_pnl(report.match_clvs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fantasy_coach.bookmaker.lines import devig_two_way
from fantasy_coach.evaluation.harness import EvaluationResult
from fantasy_coach.features import MatchRow


@dataclass(frozen=True)
class MatchCLV:
    match_id: int
    season: int
    round: int
    p_model: float  # model's home-win probability
    p_close: float  # de-vigged closing-line home-win probability
    actual: int  # 1 if home won, 0 if away won
    clv: float  # p_model − p_close (positive = found home value)
    home_decimal_odds: float
    away_decimal_odds: float


@dataclass
class CLVReport:
    predictor_name: str
    match_clvs: list[MatchCLV] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.match_clvs)

    @property
    def mean_clv(self) -> float:
        """Average CLV in probability units. Positive = model beats closing line."""
        if not self.match_clvs:
            return float("nan")
        return sum(c.clv for c in self.match_clvs) / len(self.match_clvs)

    @property
    def cumulative_clv(self) -> list[float]:
        total = 0.0
        result = []
        for c in self.match_clvs:
            total += c.clv
            result.append(round(total, 6))
        return result

    @property
    def roi_flat(self) -> float:
        """ROI under a flat £1 stake always on the model's pick."""
        if not self.match_clvs:
            return float("nan")
        total_stake = float(len(self.match_clvs))
        total_return = sum(_flat_return(c) for c in self.match_clvs)
        return (total_return - total_stake) / total_stake

    @property
    def win_rate(self) -> float:
        """Fraction of bets where the model's pick won."""
        if not self.match_clvs:
            return float("nan")
        wins = sum(1 for c in self.match_clvs if _model_pick_won(c))
        return wins / len(self.match_clvs)


def compute_clv(
    eval_result: EvaluationResult,
    matches: list[MatchRow],
) -> CLVReport | None:
    """Compute CLV for each prediction that has closing-line coverage.

    Joins ``eval_result`` predictions to ``matches`` by ``match_id``.  Only
    matches where both home and away decimal odds are available contribute to
    the report; the rest are silently dropped (no closing-line coverage).

    Returns ``None`` when fewer than 10 matches have coverage — too sparse to
    draw conclusions.
    """
    match_lookup: dict[int, MatchRow] = {m.match_id: m for m in matches}

    clvs: list[MatchCLV] = []
    for pred in eval_result.predictions:
        match = match_lookup.get(pred.match_id)
        if match is None:
            continue
        home_odds = match.home.odds
        away_odds = match.away.odds
        if home_odds is None or away_odds is None:
            continue
        if home_odds <= 1.0 or away_odds <= 1.0:
            continue
        try:
            p_close = devig_two_way(home_odds, away_odds)
        except ValueError:
            continue
        clvs.append(
            MatchCLV(
                match_id=pred.match_id,
                season=pred.season,
                round=pred.round,
                p_model=pred.p_home_win,
                p_close=p_close,
                actual=pred.actual_home_win,
                clv=round(pred.p_home_win - p_close, 6),
                home_decimal_odds=home_odds,
                away_decimal_odds=away_odds,
            )
        )

    if len(clvs) < 10:
        return None

    report = CLVReport(predictor_name=eval_result.predictor_name)
    report.match_clvs = sorted(clvs, key=lambda c: (c.season, c.round, c.match_id))
    return report


def kelly_stake(
    p_model: float,
    decimal_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Return the quarter-Kelly stake for a single bet.

    Kelly formula (full): ``f* = (p * b − (1 − p)) / b``
    where ``b = decimal_odds − 1`` (net odds).  Quarter-Kelly applies a
    fraction to reduce variance at the cost of a modest EV reduction.

    Returns 0 when the bet has no edge (``p * decimal_odds <= 1``), or when
    odds are degenerate (``decimal_odds <= 1``).
    """
    if decimal_odds <= 1.0 or bankroll <= 0.0:
        return 0.0
    b = decimal_odds - 1.0
    edge = p_model * decimal_odds - 1.0
    if edge <= 0.0:
        return 0.0
    full_kelly_frac = edge / b
    return kelly_fraction * full_kelly_frac * bankroll


Strategy = Literal["quarter_kelly", "flat"]


def simulate_pnl(
    match_clvs: list[MatchCLV],
    *,
    strategy: Strategy = "quarter_kelly",
    kelly_fraction: float = 0.25,
    starting_bankroll: float = 100.0,
) -> list[float]:
    """Simulate bankroll evolution across all covered matches.

    For each match, the model bets on its pick:
    - ``p_model > 0.5`` → bet on home at ``home_decimal_odds``
    - ``p_model <= 0.5`` → bet on away at ``away_decimal_odds``

    Under ``"flat"`` strategy, stakes exactly 1 unit regardless of bankroll.
    Under ``"quarter_kelly"`` strategy, stake is ``kelly_stake(p, odds, bankroll)``.

    Returns the bankroll value *after* each bet — length equals
    ``len(match_clvs)``.  Starting bankroll is prepended implicitly; the
    first element is the bankroll after the first bet.
    """
    bankroll = starting_bankroll
    series: list[float] = []
    for c in match_clvs:
        if c.p_model > 0.5:
            p_bet = c.p_model
            odds = c.home_decimal_odds
            won = c.actual == 1
        else:
            p_bet = 1.0 - c.p_model
            odds = c.away_decimal_odds
            won = c.actual == 0

        stake = 1.0 if strategy == "flat" else kelly_stake(p_bet, odds, bankroll, kelly_fraction)

        if won:
            bankroll += stake * (odds - 1.0)
        else:
            bankroll -= stake

        series.append(round(bankroll, 4))
    return series


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _model_pick_won(c: MatchCLV) -> bool:
    if c.p_model > 0.5:
        return c.actual == 1
    return c.actual == 0


def _flat_return(c: MatchCLV) -> float:
    """Gross return on a £1 flat stake on the model's pick."""
    if c.p_model > 0.5:
        return c.home_decimal_odds if c.actual == 1 else 0.0
    return c.away_decimal_odds if c.actual == 0 else 0.0
