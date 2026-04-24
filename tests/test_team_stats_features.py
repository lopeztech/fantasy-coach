"""Tests for team-stat rolling features added in #160.

Covers the three requirements:
1. missing_team_stats flag fires when team_stats is empty.
2. Rolling kick metres deque accumulates correctly after completed matches.
3. FEATURE_NAMES length is correct (original 6 + 5 new = 11).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder
from fantasy_coach.features import MatchRow, TeamRow, TeamStat


def _team(team_id: int, score: int | None = None) -> TeamRow:
    return TeamRow(
        team_id=team_id,
        name=str(team_id),
        nick_name=str(team_id),
        score=score,
        players=[],
    )


def _match(
    *,
    match_id: int,
    home_id: int,
    away_id: int,
    home_score: int,
    away_score: int,
    when: datetime,
    team_stats: list[TeamStat] | None = None,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=when.year,
        round=1,
        start_time=when,
        match_state="FullTime",
        venue=None,
        venue_city=None,
        weather=None,
        home=_team(home_id, home_score),
        away=_team(away_id, away_score),
        team_stats=team_stats or [],
    )


def _stat(title: str, home_value: float, away_value: float) -> TeamStat:
    return TeamStat(
        title=title,
        type="stat",
        units=None,
        home_value=home_value,
        away_value=away_value,
    )


def test_missing_team_stats_flag() -> None:
    """missing_team_stats = 1.0 when team_stats is empty (no history yet)."""
    builder = FeatureBuilder()
    match = _match(
        match_id=1,
        home_id=10,
        away_id=20,
        home_score=24,
        away_score=18,
        when=datetime(2024, 3, 1, tzinfo=UTC),
        team_stats=[],  # empty — no stats scraped
    )
    row = dict(zip(FEATURE_NAMES, builder.feature_row(match), strict=True))
    assert row["missing_team_stats"] == 1.0


def test_missing_team_stats_clears_after_data_recorded() -> None:
    """missing_team_stats = 0.0 once kicking metres have been recorded."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    builder = FeatureBuilder()

    # First match — record with team stats
    m1 = _match(
        match_id=1,
        home_id=10,
        away_id=20,
        home_score=24,
        away_score=18,
        when=base,
        team_stats=[
            _stat("Kicking Metres", 400.0, 350.0),
            _stat("Kick Return Metres", 120.0, 90.0),
            _stat("Line Breaks", 3.0, 2.0),
            _stat("All Runs", 130.0, 115.0),
        ],
    )
    builder.record(m1)

    # Second match for same home team — now deques have entries
    m2 = _match(
        match_id=2,
        home_id=10,
        away_id=30,
        home_score=20,
        away_score=14,
        when=base + timedelta(days=7),
        team_stats=[],
    )
    row = dict(zip(FEATURE_NAMES, builder.feature_row(m2), strict=True))
    assert row["missing_team_stats"] == 0.0


def test_rolling_kick_metres_accumulates() -> None:
    """After two matches with known kicking metres, feature_row reflects the rolling average."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    builder = FeatureBuilder()

    # Match 1: team 10 home kicks 400m, team 20 away kicks 350m
    m1 = _match(
        match_id=1,
        home_id=10,
        away_id=20,
        home_score=24,
        away_score=18,
        when=base,
        team_stats=[_stat("Kicking Metres", 400.0, 350.0)],
    )
    builder.record(m1)

    # Match 2: team 10 home kicks 600m, team 30 away kicks 200m
    m2 = _match(
        match_id=2,
        home_id=10,
        away_id=30,
        home_score=20,
        away_score=14,
        when=base + timedelta(days=7),
        team_stats=[_stat("Kicking Metres", 600.0, 200.0)],
    )
    builder.record(m2)

    # Match 3: team 10 home vs team 20 away — check kick metres diff
    # team 10 has [400, 600] → avg = 500; team 20 has [350] → avg = 350
    m3 = _match(
        match_id=3,
        home_id=10,
        away_id=20,
        home_score=18,
        away_score=22,
        when=base + timedelta(days=14),
        team_stats=[],
    )
    row = dict(zip(FEATURE_NAMES, builder.feature_row(m3), strict=True))
    assert row["rolling_kick_metres_diff"] == 500.0 - 350.0


def test_new_feature_count() -> None:
    """FEATURE_NAMES should have exactly 11 entries (original 6 + 5 new team-stat features)."""
    assert len(FEATURE_NAMES) == 11
    # Verify the new names are present
    new_names = {
        "rolling_kick_metres_diff",
        "rolling_kick_return_metres_diff",
        "rolling_line_breaks_diff",
        "rolling_all_runs_diff",
        "missing_team_stats",
    }
    assert new_names.issubset(set(FEATURE_NAMES))
