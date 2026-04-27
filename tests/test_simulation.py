"""Tests for Monte Carlo season simulator (#217)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.simulation import FINALS_SPOTS, simulate_season

_T0 = datetime(2026, 5, 1, tzinfo=UTC)


def _match(  # type: ignore[return]
    mid: int,
    home_id: int,
    away_id: int,
    home_score: int | None = None,
    away_score: int | None = None,
    round_: int = 1,
) -> MatchRow:
    state = "FullTime" if home_score is not None else "Upcoming"
    return MatchRow(
        match_id=mid,
        season=2026,
        round=round_,
        start_time=_T0,
        match_state=state,
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name=f"T{home_id}",
            nick_name=f"T{home_id}",
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=away_id,
            name=f"T{away_id}",
            nick_name=f"T{away_id}",
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


def _eight_team_completed() -> list[MatchRow]:
    """8 teams with a known set of completed results; no upcoming matches."""
    return [
        # T0 wins all → clear 1st place
        _match(1, 0, 1, 24, 12),
        _match(2, 0, 2, 30, 10),
        _match(3, 0, 3, 28, 14),
        # T1-T3 beat T4-T7
        _match(4, 1, 4, 20, 14),
        _match(5, 2, 5, 18, 16),
        _match(6, 3, 6, 22, 10),
        # T4-T6 beat T7
        _match(7, 4, 7, 16, 10),
        _match(8, 5, 7, 14, 12),
    ]


def _eight_team_with_upcoming(n: int = 4) -> tuple[list[MatchRow], dict[int, float]]:
    completed = _eight_team_completed()
    upcoming = []
    for i in range(n):
        h, a = i % 8, (i + 4) % 8
        if h == a:
            a = (a + 1) % 8
        upcoming.append(_match(100 + i, h, a, round_=2))
    predictions = {m.match_id: 0.55 for m in upcoming}
    return completed + upcoming, predictions


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    matches, preds = _eight_team_with_upcoming(4)

    rng1 = np.random.default_rng(42)
    r1 = simulate_season(2026, matches, preds, n_simulations=200, rng=rng1)

    rng2 = np.random.default_rng(42)
    r2 = simulate_season(2026, matches, preds, n_simulations=200, rng=rng2)

    assert {t.team_id for t in r1.teams} == {t.team_id for t in r2.teams}
    by_id_1 = {t.team_id: t for t in r1.teams}
    by_id_2 = {t.team_id: t for t in r2.teams}
    for tid in by_id_1:
        assert by_id_1[tid].playoff_prob == pytest.approx(by_id_2[tid].playoff_prob)
        assert by_id_1[tid].premiership_prob == pytest.approx(by_id_2[tid].premiership_prob)
        assert by_id_1[tid].top_4_prob == pytest.approx(by_id_2[tid].top_4_prob)


# ---------------------------------------------------------------------------
# Probability conservation
# ---------------------------------------------------------------------------


def test_probability_sums() -> None:
    """Sum of probs across all teams equals expected count per mutually-exclusive outcome."""
    matches, preds = _eight_team_with_upcoming(4)
    result = simulate_season(
        2026, matches, preds, n_simulations=1_000, rng=np.random.default_rng(0)
    )
    assert len(result.teams) == 8

    tol = 0.02
    assert sum(t.playoff_prob for t in result.teams) == pytest.approx(FINALS_SPOTS, abs=tol)
    assert sum(t.top_4_prob for t in result.teams) == pytest.approx(4.0, abs=tol)
    assert sum(t.top_2_prob for t in result.teams) == pytest.approx(2.0, abs=tol)
    assert sum(t.minor_premiership_prob for t in result.teams) == pytest.approx(1.0, abs=tol)
    assert sum(t.grand_final_prob for t in result.teams) == pytest.approx(2.0, abs=tol)
    assert sum(t.premiership_prob for t in result.teams) == pytest.approx(1.0, abs=tol)


# ---------------------------------------------------------------------------
# Finals: exactly one champion
# ---------------------------------------------------------------------------


def test_finals_one_champion() -> None:
    """Every simulation produces exactly one champion — no double-counting."""
    matches, preds = _eight_team_with_upcoming(8)
    result = simulate_season(2026, matches, preds, n_simulations=500, rng=np.random.default_rng(7))
    total = sum(t.premiership_prob for t in result.teams)
    assert total == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Ladder tiebreaker: equal points, higher percentage wins
# ---------------------------------------------------------------------------


def test_percentage_tiebreaker() -> None:
    """When two teams have equal competition points, the higher-pct team ranks first."""
    # T0: beats T2 (50-10) and T3 (50-10) → 4 pts, pct = 500 %
    # T1: beats T4 (12-10) and T5 (12-10) → 4 pts, pct = 120 %
    # T2, T3, T4, T5: one win each (to bring 8-team fields up)
    # T6, T7: one loss each
    matches = [
        _match(1, 0, 2, 50, 10),
        _match(2, 0, 3, 50, 10),
        _match(3, 1, 4, 12, 10),
        _match(4, 1, 5, 12, 10),
        _match(5, 2, 6, 20, 10),
        _match(6, 3, 7, 20, 10),
    ]
    # No upcoming matches → runs 1 deterministic simulation.
    result = simulate_season(2026, matches, {}, n_simulations=1, rng=np.random.default_rng(0))
    by_tid = {t.team_id: t for t in result.teams}

    # T0 has better pct than T1 at equal competition points → T0 is 1st.
    assert by_tid[0].minor_premiership_prob == pytest.approx(1.0)
    assert by_tid[1].minor_premiership_prob == pytest.approx(0.0)
    assert by_tid[0].top_2_prob == pytest.approx(1.0)
    assert by_tid[1].top_2_prob == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# as_dict round-trip
# ---------------------------------------------------------------------------


def test_as_dict_structure() -> None:
    matches, preds = _eight_team_with_upcoming(2)
    result = simulate_season(2026, matches, preds, n_simulations=50, rng=np.random.default_rng(3))
    d = result.as_dict()

    assert d["season"] == 2026
    assert d["nSimulations"] == 50
    assert len(d["teams"]) == 8

    for entry in d["teams"]:
        for key in (
            "teamId",
            "teamName",
            "playoffProb",
            "top4Prob",
            "top2Prob",
            "minorPremiershipProb",
            "grandFinalProb",
            "premiershipProb",
        ):
            assert key in entry, f"Missing key {key}"
        # All probs in [0, 1]
        for key in (
            "playoffProb",
            "top4Prob",
            "top2Prob",
            "minorPremiershipProb",
            "grandFinalProb",
            "premiershipProb",
        ):
            assert 0.0 <= entry[key] <= 1.0, f"{key}={entry[key]} out of range"
