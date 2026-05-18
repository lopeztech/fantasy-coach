"""Turn cached predictions + bookmaker odds into a weekly tips card.

Pure function — no I/O. The precompute job loads predictions from the
``predictions`` store, optionally fetches totals lines via ``OddsApiClient``,
and hands both to ``generate_tips``. The output is written to the
``betting_tips`` store and served read-only via the API.

Algorithm:

1. Build leg candidates per prediction:
   - H2H leg (always available when scraped nrl.com odds are populated).
   - Totals leg (only when both the prediction has ``predictedTotal`` and a
     ``TotalsLine`` is provided for the match).
2. Filter: positive de-vigged edge AND model probability strong enough that
   the leg is plausibly "almost guaranteed". Confidence band is folded into
   the score but is not a hard gate — keeping it soft lets legacy predictions
   (with ``confidenceBand=None``) still surface picks.
3. Rank by ``modelProb × (1 + edge) × confidenceMultiplier``.
4. Take top-2 singles, top-2 doubles, top-2 trebles, with each multi requiring
   match-disjoint legs (legs from the same match are perfectly correlated).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fantasy_coach.betting.models import (
    BettingTipsOut,
    Tip,
    TipLeg,
    TotalsLine,
)
from fantasy_coach.bookmaker.lines import devig_two_way

# Per-leg filters — calibrated by inspection: model_prob ≥ 0.55 means the
# model meaningfully favours the side; edge ≥ 0.02 strips out picks the market
# has already fully priced in. Both are tunable but should be raised together
# — relaxing one without the other tends to surface noise.
MIN_MODEL_PROB = 0.55
MIN_EDGE = 0.02

# Standard deviation of NRL match totals around their season mean. Used to
# convert ``predictedTotal − line`` into a normal-CDF probability of the
# selected side landing. 14 is a defensible NRL-specific value; tune from
# empirical totals residuals once we have a few months of multitask predictions
# to backtest. Quoted here as a constant so a single edit retunes all totals.
TOTALS_STDEV = 14.0
TOTALS_MIN_EDGE_PTS = 1.0  # require the model to disagree with the line by >1pt

# Score-weight ramp for the confidenceBand field. Untiered predictions
# (legacy / non-XGBoost artefacts) get a neutral 1.0.
_CONFIDENCE_WEIGHTS: dict[str, float] = {"high": 1.5, "medium": 1.0, "low": 0.6}


# ---------------------------------------------------------------------------
# Internal candidate type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LegCandidate:
    match_id: int
    home_name: str
    away_name: str
    kickoff: str
    market: str  # "h2h" | "totals"
    selection_code: str  # "home"|"away" | "over"|"under"
    selection_label: str  # display string
    line: float | None
    decimal_odds: float
    model_prob: float
    implied_prob: float
    bookmaker: str
    confidence_band: str | None

    @property
    def edge(self) -> float:
        return self.model_prob - self.implied_prob

    @property
    def score(self) -> float:
        cband = self.confidence_band or ""
        weight = _CONFIDENCE_WEIGHTS.get(cband, 1.0)
        return self.model_prob * (1.0 + self.edge) * weight

    def to_leg(self) -> TipLeg:
        return TipLeg(
            matchId=self.match_id,
            homeName=self.home_name,
            awayName=self.away_name,
            kickoff=self.kickoff,
            market=self.market,  # type: ignore[arg-type]
            selectionCode=self.selection_code,  # type: ignore[arg-type]
            selection=self.selection_label,
            line=self.line,
            decimalOdds=round(self.decimal_odds, 2),
            modelProb=round(self.model_prob, 4),
            impliedProb=round(self.implied_prob, 4),
            edge=round(self.edge, 4),
            bookmaker=self.bookmaker,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_tips(
    season: int,
    round_: int,
    predictions: list[Any],
    *,
    totals_lines: dict[tuple[str, str], TotalsLine] | None = None,
    odds_snapshot_at: str | None = None,
    now: datetime | None = None,
) -> BettingTipsOut:
    """Compute the 2-singles + 2-doubles + 2-trebles card for one round.

    ``predictions`` are ``PredictionOut`` instances (kept loose-typed here to
    avoid a circular import). ``totals_lines`` is keyed by
    ``(home_nickname, away_nickname)`` matching the prediction's team names —
    populated by ``OddsApiClient.fetch_totals`` when an API key is configured.

    ``now`` is injectable for deterministic tests.
    """
    timestamp = (now or datetime.now(UTC)).isoformat()

    candidates: list[_LegCandidate] = []
    for pred in predictions:
        h2h = _h2h_candidate(pred)
        if h2h is not None and _passes_filters(h2h):
            candidates.append(h2h)
        if totals_lines:
            tl = totals_lines.get((pred.home.name, pred.away.name))
            if tl is not None:
                tot = _totals_candidate(pred, tl)
                if tot is not None and _passes_filters(tot):
                    candidates.append(tot)

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

    singles = [_tip_from_legs([c], "single") for c in ranked[:2]]
    doubles = [_tip_from_legs(combo, "double") for combo in _best_multis(ranked, 2, 2)]
    trebles = [_tip_from_legs(combo, "treble") for combo in _best_multis(ranked, 3, 2)]

    return BettingTipsOut(
        season=season,
        round=round_,
        generatedAt=timestamp,
        oddsSnapshotAt=odds_snapshot_at,
        singles=singles,
        doubles=doubles,
        trebles=trebles,
    )


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _h2h_candidate(pred: Any) -> _LegCandidate | None:
    """Build the head-to-head leg candidate from a PredictionOut, if priced."""
    home_odds = getattr(pred.home, "odds", None)
    away_odds = getattr(pred.away, "odds", None)
    if home_odds is None or away_odds is None or home_odds <= 1.0 or away_odds <= 1.0:
        return None
    try:
        implied_home = devig_two_way(float(home_odds), float(away_odds))
    except ValueError:
        return None

    home_prob = float(pred.homeWinProbability)
    if home_prob >= 0.5:
        selection_code = "home"
        selection_label = pred.home.name
        model_prob = home_prob
        implied_prob = implied_home
        decimal_odds = float(home_odds)
    else:
        selection_code = "away"
        selection_label = pred.away.name
        model_prob = 1.0 - home_prob
        implied_prob = 1.0 - implied_home
        decimal_odds = float(away_odds)

    return _LegCandidate(
        match_id=int(pred.matchId),
        home_name=pred.home.name,
        away_name=pred.away.name,
        kickoff=pred.kickoff,
        market="h2h",
        selection_code=selection_code,
        selection_label=selection_label,
        line=None,
        decimal_odds=decimal_odds,
        model_prob=model_prob,
        implied_prob=implied_prob,
        bookmaker="nrl.com",
        confidence_band=getattr(pred, "confidenceBand", None),
    )


def _totals_candidate(pred: Any, line: TotalsLine) -> _LegCandidate | None:
    """Build the totals (over/under) leg, when the model has a totals prediction."""
    predicted_total = getattr(pred, "predictedTotal", None)
    if predicted_total is None:
        return None

    diff = float(predicted_total) - line.line
    if abs(diff) < TOTALS_MIN_EDGE_PTS:
        return None

    # De-vig the bookmaker's over/under prices.
    over_raw = 1.0 / line.overDecimalOdds
    under_raw = 1.0 / line.underDecimalOdds
    total_raw = over_raw + under_raw
    if total_raw <= 0:
        return None
    implied_over = over_raw / total_raw
    implied_under = under_raw / total_raw

    if diff > 0:
        model_prob = _phi(diff / TOTALS_STDEV)
        return _LegCandidate(
            match_id=int(pred.matchId),
            home_name=pred.home.name,
            away_name=pred.away.name,
            kickoff=pred.kickoff,
            market="totals",
            selection_code="over",
            selection_label=f"Over {line.line:g}",
            line=line.line,
            decimal_odds=line.overDecimalOdds,
            model_prob=model_prob,
            implied_prob=implied_over,
            bookmaker=line.bookmaker,
            confidence_band=getattr(pred, "confidenceBand", None),
        )
    model_prob = _phi(-diff / TOTALS_STDEV)
    return _LegCandidate(
        match_id=int(pred.matchId),
        home_name=pred.home.name,
        away_name=pred.away.name,
        kickoff=pred.kickoff,
        market="totals",
        selection_code="under",
        selection_label=f"Under {line.line:g}",
        line=line.line,
        decimal_odds=line.underDecimalOdds,
        model_prob=model_prob,
        implied_prob=implied_under,
        bookmaker=line.bookmaker,
        confidence_band=getattr(pred, "confidenceBand", None),
    )


def _passes_filters(cand: _LegCandidate) -> bool:
    return cand.model_prob >= MIN_MODEL_PROB and cand.edge >= MIN_EDGE


# ---------------------------------------------------------------------------
# Multi assembly
# ---------------------------------------------------------------------------


def _best_multis(
    candidates: list[_LegCandidate], n_legs: int, take: int
) -> list[tuple[_LegCandidate, ...]]:
    """Return up to ``take`` best n-leg multis with match-disjoint legs.

    Uses a greedy selection over the sorted combinations: highest scoring
    combo first, then the next-highest combo that isn't a permutation of an
    already-picked set. We allow leg reuse *across* multi rows (a leg in
    Double #1 may also appear in Treble #2) since each tip is independent
    consumer-facing — strict no-reuse drastically thins the card when the
    candidate pool is small.
    """
    if len(candidates) < n_legs:
        return []
    combos: list[tuple[tuple[_LegCandidate, ...], float]] = []
    for combo in itertools.combinations(candidates, n_legs):
        match_ids = {leg.match_id for leg in combo}
        if len(match_ids) < n_legs:
            continue  # two legs from the same match — perfectly correlated
        prob = 1.0
        edge_min = 1.0
        for leg in combo:
            prob *= leg.model_prob
            edge_min = min(edge_min, leg.edge)
        # Score: combined model prob × (1 + min edge). Picks "almost guaranteed"
        # multis first while still requiring every leg to have positive value.
        if edge_min < 0:
            continue
        combos.append((combo, prob * (1.0 + edge_min)))
    combos.sort(key=lambda x: x[1], reverse=True)

    chosen: list[tuple[_LegCandidate, ...]] = []
    seen: set[frozenset[tuple[int, str]]] = set()
    for combo, _ in combos:
        key = frozenset((leg.match_id, leg.selection_code) for leg in combo)
        if key in seen:
            continue
        chosen.append(combo)
        seen.add(key)
        if len(chosen) >= take:
            break
    return chosen


def _tip_from_legs(candidates: tuple[_LegCandidate, ...] | list[_LegCandidate], kind: str) -> Tip:
    legs = [c.to_leg() for c in candidates]
    combined_odds = 1.0
    combined_model = 1.0
    combined_implied = 1.0
    for c in candidates:
        combined_odds *= c.decimal_odds
        combined_model *= c.model_prob
        combined_implied *= c.implied_prob
    return Tip(
        kind=kind,  # type: ignore[arg-type]
        legs=legs,
        combinedOdds=round(combined_odds, 2),
        combinedModelProb=round(combined_model, 4),
        combinedImpliedProb=round(combined_implied, 4),
        edge=round(combined_model - combined_implied, 4),
        suggestedUnits=1.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phi(z: float) -> float:
    """Standard-normal CDF without a scipy dep (Abramowitz & Stegun 26.2.17)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
