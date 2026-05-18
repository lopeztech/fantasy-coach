"""Tests for ``betting.generator.generate_tips``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fantasy_coach.betting.generator import (
    MIN_MODEL_PROB,
    TOTALS_STDEV,
    generate_tips,
)
from fantasy_coach.betting.models import BettingTipsOut, TotalsLine
from fantasy_coach.predictions import PredictionOut, TeamInfo


def _pred(
    match_id: int,
    *,
    home_name: str,
    away_name: str,
    home_win_prob: float,
    home_odds: float | None,
    away_odds: float | None,
    confidence_band: str | None = "high",
    predicted_total: float | None = None,
    kickoff: str = "2026-05-01T08:00:00+00:00",
) -> PredictionOut:
    return PredictionOut(
        matchId=match_id,
        home=TeamInfo(id=match_id * 10, name=home_name, odds=home_odds),
        away=TeamInfo(id=match_id * 10 + 1, name=away_name, odds=away_odds),
        kickoff=kickoff,
        predictedWinner="home" if home_win_prob >= 0.5 else "away",
        homeWinProbability=home_win_prob,
        modelVersion="abc123",
        featureHash="def456",
        confidenceBand=confidence_band,
        predictedTotal=predicted_total,
    )


# ---------------------------------------------------------------------------
# Singles
# ---------------------------------------------------------------------------


def test_h2h_single_picked_when_model_beats_market() -> None:
    # Market implies ~50% (1.95 / 1.95 de-vigged ≈ 0.50); model says 0.70 → edge 0.20.
    preds = [
        _pred(
            1,
            home_name="Storm",
            away_name="Eels",
            home_win_prob=0.70,
            home_odds=1.95,
            away_odds=1.95,
        )
    ]
    out = generate_tips(2026, 7, preds, now=datetime(2026, 5, 1, tzinfo=UTC))
    assert isinstance(out, BettingTipsOut)
    assert len(out.singles) == 1
    leg = out.singles[0].legs[0]
    assert leg.market == "h2h"
    assert leg.selection == "Storm"
    assert leg.modelProb == pytest.approx(0.70)
    assert leg.edge > 0.15


def test_h2h_rejects_when_model_only_marginally_above_implied() -> None:
    # Model 0.55, market implies 0.54 → edge 0.01 < MIN_EDGE.
    preds = [
        _pred(
            1,
            home_name="Storm",
            away_name="Eels",
            home_win_prob=0.55,
            home_odds=1.85,
            away_odds=2.10,
        )
    ]
    out = generate_tips(2026, 7, preds)
    assert out.singles == []


def test_h2h_rejects_when_model_prob_below_threshold() -> None:
    # Model 0.54 < 0.55 floor — even with edge, skip.
    preds = [
        _pred(
            1,
            home_name="Storm",
            away_name="Eels",
            home_win_prob=0.54,
            home_odds=3.0,
            away_odds=1.40,
        )
    ]
    assert MIN_MODEL_PROB > 0.54  # invariant the test depends on
    out = generate_tips(2026, 7, preds)
    assert out.singles == []


def test_picks_away_side_when_model_favours_away() -> None:
    # home_win_prob 0.30 → away_prob 0.70; away decimal odds 1.95 (implied ~0.50).
    preds = [
        _pred(
            1,
            home_name="Wests Tigers",
            away_name="Panthers",
            home_win_prob=0.30,
            home_odds=2.20,
            away_odds=1.95,
        )
    ]
    out = generate_tips(2026, 7, preds)
    assert len(out.singles) == 1
    leg = out.singles[0].legs[0]
    assert leg.selectionCode == "away"
    assert leg.selection == "Panthers"


def test_skips_match_without_odds() -> None:
    preds = [
        _pred(
            1,
            home_name="Storm",
            away_name="Eels",
            home_win_prob=0.80,
            home_odds=None,
            away_odds=None,
        )
    ]
    assert generate_tips(2026, 7, preds).singles == []


# ---------------------------------------------------------------------------
# Multis
# ---------------------------------------------------------------------------


def _strong_h2h_pred(
    match_id: int, home_name: str, prob: float = 0.72, home_odds: float = 1.85
) -> PredictionOut:
    return _pred(
        match_id,
        home_name=home_name,
        away_name=f"Opp{match_id}",
        home_win_prob=prob,
        home_odds=home_odds,
        away_odds=2.05,
    )


def test_double_uses_two_different_matches() -> None:
    preds = [_strong_h2h_pred(i, name) for i, name in enumerate(["A", "B", "C"], start=1)]
    out = generate_tips(2026, 7, preds)
    assert len(out.doubles) >= 1
    d = out.doubles[0]
    assert d.kind == "double"
    assert len(d.legs) == 2
    match_ids = {leg.matchId for leg in d.legs}
    assert len(match_ids) == 2


def test_treble_uses_three_different_matches_and_correct_combined_odds() -> None:
    preds = [_strong_h2h_pred(i, name) for i, name in enumerate(["A", "B", "C", "D"], start=1)]
    out = generate_tips(2026, 7, preds)
    assert len(out.trebles) >= 1
    t = out.trebles[0]
    assert t.kind == "treble"
    assert len(t.legs) == 3
    assert len({leg.matchId for leg in t.legs}) == 3
    expected_odds = 1.0
    expected_model = 1.0
    for leg in t.legs:
        expected_odds *= leg.decimalOdds
        expected_model *= leg.modelProb
    assert t.combinedOdds == pytest.approx(round(expected_odds, 2), abs=0.02)
    assert t.combinedModelProb == pytest.approx(round(expected_model, 4), abs=0.001)


def test_no_trebles_when_only_two_qualifying_picks() -> None:
    preds = [_strong_h2h_pred(1, "A"), _strong_h2h_pred(2, "B")]
    out = generate_tips(2026, 7, preds)
    assert out.trebles == []


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------


def test_totals_leg_added_when_model_disagrees_with_line() -> None:
    pred = _pred(
        1,
        home_name="Storm",
        away_name="Eels",
        home_win_prob=0.55,
        home_odds=1.85,
        away_odds=2.10,
        predicted_total=48.0,
    )
    line = TotalsLine(
        line=40.0,
        overDecimalOdds=1.90,
        underDecimalOdds=1.90,
        bookmaker="TAB",
    )
    out = generate_tips(2026, 7, [pred], totals_lines={("Storm", "Eels"): line})
    totals_legs = [s.legs[0] for s in out.singles if s.legs[0].market == "totals"]
    assert len(totals_legs) == 1
    leg = totals_legs[0]
    assert leg.selectionCode == "over"
    assert leg.line == 40.0
    assert leg.bookmaker == "TAB"
    # Phi((48 - 40) / 14) ≈ 0.716
    from math import erf, sqrt

    expected = 0.5 * (1.0 + erf(8.0 / TOTALS_STDEV / sqrt(2.0)))
    assert leg.modelProb == pytest.approx(round(expected, 4), abs=1e-3)


def test_totals_skipped_when_predicted_total_close_to_line() -> None:
    pred = _pred(
        1,
        home_name="Storm",
        away_name="Eels",
        home_win_prob=0.40,  # low — no H2H pick to mask the test
        home_odds=2.50,
        away_odds=1.55,
        predicted_total=40.3,
    )
    line = TotalsLine(line=40.0, overDecimalOdds=1.90, underDecimalOdds=1.90, bookmaker="TAB")
    out = generate_tips(2026, 7, [pred], totals_lines={("Storm", "Eels"): line})
    totals_legs = [s.legs[0] for s in out.singles if s.legs[0].market == "totals"]
    assert totals_legs == []


def test_totals_skipped_when_prediction_lacks_predicted_total() -> None:
    pred = _pred(
        1,
        home_name="Storm",
        away_name="Eels",
        home_win_prob=0.40,
        home_odds=2.50,
        away_odds=1.55,
        predicted_total=None,
    )
    line = TotalsLine(line=40.0, overDecimalOdds=1.90, underDecimalOdds=1.90, bookmaker="TAB")
    out = generate_tips(2026, 7, [pred], totals_lines={("Storm", "Eels"): line})
    totals_legs = [s.legs[0] for s in out.singles if s.legs[0].market == "totals"]
    assert totals_legs == []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_empty_predictions_returns_empty_card() -> None:
    out = generate_tips(2026, 7, [])
    assert out.singles == []
    assert out.doubles == []
    assert out.trebles == []
    assert out.season == 2026
    assert out.round == 7


def test_caveat_string_present() -> None:
    out = generate_tips(2026, 7, [])
    assert "independent" in out.caveat.lower()
