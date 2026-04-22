"""Tests for opponent-adjusted form features (#108).

Verifies:
1. form_diff_pf_adjusted and form_diff_pa_adjusted are present in FEATURE_NAMES.
2. No-leakage invariant: adjusted scores use only the opponent's baseline from
   matches with start_time strictly before the current match.
3. Correct arithmetic: adjusted_pf = actual_pf - opponent_rolling_pa_baseline.
4. Falls back to raw scores when the opponent has no prior history.
5. Home-minus-away convention matches the rest of FEATURE_NAMES.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fantasy_coach.feature_engineering import (
    ADJ_OPP_WINDOW,
    FEATURE_NAMES,
    FeatureBuilder,
    build_training_frame,
)
from fantasy_coach.features import MatchRow, TeamRow

_PF_ADJ_IDX = FEATURE_NAMES.index("form_diff_pf_adjusted")
_PA_ADJ_IDX = FEATURE_NAMES.index("form_diff_pa_adjusted")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _team(team_id: int, score: int) -> TeamRow:
    return TeamRow(
        team_id=team_id,
        name=f"Team{team_id}",
        nick_name=f"T{team_id}",
        score=score,
        players=[],
    )


def _match(
    match_id: int,
    *,
    home_id: int,
    away_id: int,
    home_score: int,
    away_score: int,
    when: datetime,
    season: int = 2024,
    round_: int = 1,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round_,
        start_time=when,
        match_state="FullTime",
        venue="Test Stadium",
        venue_city="Sydney",
        weather=None,
        home=_team(home_id, home_score),
        away=_team(away_id, away_score),
        team_stats=[],
    )


# ---------------------------------------------------------------------------
# Feature name presence
# ---------------------------------------------------------------------------


def test_adjusted_feature_names_present() -> None:
    assert "form_diff_pf_adjusted" in FEATURE_NAMES
    assert "form_diff_pa_adjusted" in FEATURE_NAMES
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "duplicate feature names"


def test_feature_names_length() -> None:
    # 19 original + 2 adjusted = 21
    assert len(FEATURE_NAMES) == 21


# ---------------------------------------------------------------------------
# No-leakage invariant
# ---------------------------------------------------------------------------


def test_no_leakage_feature_row_uses_pre_match_opponent_baseline() -> None:
    """At feature_row() time, adjusted form uses only pre-match opponent state."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    builder = FeatureBuilder()

    # Build team-2 history (10 matches as home: score 30, concede 10)
    for i in range(ADJ_OPP_WINDOW):
        m = _match(
            i + 1, home_id=2, away_id=9, home_score=30, away_score=10, when=base + timedelta(days=i)
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    # Team 1 vs team 2: team 1 scores 20, concedes 15
    m1 = _match(
        100,
        home_id=1,
        away_id=2,
        home_score=20,
        away_score=15,
        when=base + timedelta(days=ADJ_OPP_WINDOW + 1),
    )
    builder.advance_season_if_needed(m1)
    builder.record(m1)

    # At this point team 1 has 1 entry in _adj_points_for:
    # adj_pf_1 = 20 - opp_pa_2(baseline) = 20 - 10 = 10
    assert list(builder._adj_points_for[1]) == pytest.approx([10.0], abs=1e-9)
    # adj_pa_1 = 15 - opp_pf_2(baseline) = 15 - 30 = -15
    assert list(builder._adj_points_against[1]) == pytest.approx([-15.0], abs=1e-9)


def test_no_leakage_current_match_not_in_baseline() -> None:
    """The current match's own result must not appear in its adjusted feature."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    builder = FeatureBuilder()

    # Fill team-1 and team-2 adjusted form windows via 5 historical matches
    for i in range(5):
        m = _match(
            i + 1, home_id=1, away_id=2, home_score=20, away_score=10, when=base + timedelta(days=i)
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    row_before = builder.feature_row(
        _match(
            99, home_id=1, away_id=2, home_score=40, away_score=0, when=base + timedelta(days=10)
        )
    )
    # record the match (big win), then feature_row should have changed
    big_win = _match(
        99, home_id=1, away_id=2, home_score=40, away_score=0, when=base + timedelta(days=10)
    )
    builder.record(big_win)

    row_after = builder.feature_row(
        _match(
            100, home_id=1, away_id=2, home_score=20, away_score=10, when=base + timedelta(days=11)
        )
    )
    # The big-win's adjusted score (40 - opp_baseline) must appear in row_after
    # but NOT in row_before (leakage check).
    assert row_after[_PF_ADJ_IDX] != row_before[_PF_ADJ_IDX]


# ---------------------------------------------------------------------------
# Arithmetic correctness
# ---------------------------------------------------------------------------


def test_adjusted_pf_arithmetic() -> None:
    """adj_pf = actual_pf - opponent_rolling_pa_baseline."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    builder = FeatureBuilder()

    # Team 2 concedes exactly 12 in all their matches → rolling-10 PA baseline = 12
    for i in range(ADJ_OPP_WINDOW):
        m = _match(
            i + 1, home_id=2, away_id=9, home_score=24, away_score=12, when=base + timedelta(days=i)
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    # Team 1 scores 20 vs team 2
    m1 = _match(
        100,
        home_id=1,
        away_id=2,
        home_score=20,
        away_score=18,
        when=base + timedelta(days=ADJ_OPP_WINDOW + 1),
    )
    builder.advance_season_if_needed(m1)
    builder.record(m1)

    # Expected: adj_pf_h = 20 - 12 = 8
    assert list(builder._adj_points_for[1]) == pytest.approx([8.0], abs=1e-9)


def test_adjusted_pa_arithmetic() -> None:
    """adj_pa = actual_pa - opponent_rolling_pf_baseline."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    builder = FeatureBuilder()

    # Team 2 scores exactly 24 in all their matches → rolling-10 PF baseline = 24
    for i in range(ADJ_OPP_WINDOW):
        m = _match(
            i + 1, home_id=2, away_id=9, home_score=24, away_score=8, when=base + timedelta(days=i)
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    # Team 1 concedes 18 vs team 2
    m1 = _match(
        100,
        home_id=1,
        away_id=2,
        home_score=20,
        away_score=18,
        when=base + timedelta(days=ADJ_OPP_WINDOW + 1),
    )
    builder.advance_season_if_needed(m1)
    builder.record(m1)

    # Expected: adj_pa_h = 18 - 24 = -6 (better than expected: opponent usually scores 24)
    assert list(builder._adj_points_against[1]) == pytest.approx([-6.0], abs=1e-9)


# ---------------------------------------------------------------------------
# Fallback when opponent has no history
# ---------------------------------------------------------------------------


def test_no_opponent_history_baseline_is_zero() -> None:
    """When opponent has no prior matches, adjusted score equals actual score."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    builder = FeatureBuilder()

    m = _match(1, home_id=1, away_id=2, home_score=24, away_score=12, when=base)
    builder.advance_season_if_needed(m)
    builder.record(m)

    # Both teams first match: opponent baselines are 0.0
    assert list(builder._adj_points_for[1]) == pytest.approx([24.0])
    assert list(builder._adj_points_for[2]) == pytest.approx([12.0])


# ---------------------------------------------------------------------------
# build_training_frame integration
# ---------------------------------------------------------------------------


def test_build_training_frame_includes_adjusted_columns() -> None:
    """Training frame X has columns for all FEATURE_NAMES."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    matches = [
        _match(
            i + 1,
            home_id=1,
            away_id=2,
            home_score=24,
            away_score=12,
            when=base + timedelta(days=i * 7),
        )
        for i in range(10)
    ]
    frame = build_training_frame(matches, drop_draws=False)
    assert frame.X.shape[1] == len(FEATURE_NAMES)
    assert frame.feature_names == FEATURE_NAMES


# ---------------------------------------------------------------------------
# Adjusted differs from raw when opponent quality varies
# ---------------------------------------------------------------------------


def test_adjusted_differs_from_raw_with_quality_variation() -> None:
    """Adjusted form diverges from raw when opponents have different baselines."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    builder = FeatureBuilder()

    # Team 2 has a strong conceding record (high PA baseline = 8)
    for i in range(ADJ_OPP_WINDOW):
        m = _match(
            i + 1, home_id=2, away_id=9, home_score=30, away_score=8, when=base + timedelta(days=i)
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    # Team 1 scores 20 vs team 2 repeatedly (5 matches)
    for i in range(5):
        m = _match(
            20 + i,
            home_id=1,
            away_id=2,
            home_score=20,
            away_score=12,
            when=base + timedelta(days=10 + i),
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    upcoming = _match(
        99,
        home_id=1,
        away_id=3,
        home_score=0,
        away_score=0,
        when=base + timedelta(days=20),
        round_=3,
    )
    row = builder.feature_row(upcoming)

    raw_pf = row[FEATURE_NAMES.index("form_diff_pf")]
    adj_pf = row[_PF_ADJ_IDX]
    # raw_pf_h = avg([20]*5) = 20, adj_pf_h ≈ avg([20-8]*5) = 12 → differ by ~8
    assert abs(adj_pf - raw_pf) > 5.0, (
        f"expected adjusted ({adj_pf:.2f}) to differ noticeably from raw ({raw_pf:.2f})"
    )
