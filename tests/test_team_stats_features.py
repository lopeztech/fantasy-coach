"""Unit tests for the rolling team-stat features added in #160."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder
from fantasy_coach.features import MatchRow, TeamRow, TeamStat


def _make_match(
    match_id: int,
    round_: int,
    h_score: int | None = None,
    a_score: int | None = None,
    state: str = "FullTime",
    stats: list[TeamStat] | None = None,
    home_id: int = 1,
    away_id: int = 2,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=2024,
        round=round_,
        start_time=datetime(2024, 3, 1, 9, 0, round_, tzinfo=UTC),
        match_state=state,
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(team_id=home_id, name="Home", nick_name="HOM", score=h_score, players=[]),
        away=TeamRow(team_id=away_id, name="Away", nick_name="AWY", score=a_score, players=[]),
        team_stats=stats or [],
    )


def _stat(title: str, home_val: float, away_val: float) -> TeamStat:
    return TeamStat(title=title, type="stat", units=None, home_value=home_val, away_value=away_val)


def _feature_dict(builder: FeatureBuilder, match: MatchRow) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, builder.feature_row(match), strict=True))


def test_missing_team_stats_is_1_when_no_history():
    """missing_team_stats = 1.0 before any stats have been recorded."""
    builder = FeatureBuilder()
    m = _make_match(1, 1, state="Upcoming")
    row = _feature_dict(builder, m)
    assert row["missing_team_stats"] == 1.0


def test_missing_team_stats_clears_after_recording():
    """missing_team_stats = 0.0 once both teams have the rolling stat inputs."""
    builder = FeatureBuilder()
    stats = [
        _stat("Kicking Metres", 400.0, 300.0),
        _stat("Kick Return Metres", 100.0, 80.0),
        _stat("Line Breaks", 3.0, 2.0),
        _stat("All Runs", 120.0, 100.0),
    ]
    m1 = _make_match(1, 1, h_score=20, a_score=10, stats=stats)
    builder.record(m1)
    m2 = _make_match(2, 2, state="Upcoming")
    row = _feature_dict(builder, m2)
    assert row["missing_team_stats"] == 0.0


def test_missing_team_stats_stays_1_when_away_team_has_no_history():
    """Zeros for a fresh opponent are missing data, not a real rolling average."""
    builder = FeatureBuilder()
    stats = [
        _stat("Kicking Metres", 400.0, 300.0),
        _stat("Kick Return Metres", 100.0, 80.0),
        _stat("Line Breaks", 3.0, 2.0),
        _stat("All Runs", 120.0, 100.0),
    ]
    m1 = _make_match(1, 1, h_score=20, a_score=10, stats=stats, home_id=1, away_id=2)
    builder.record(m1)
    m2 = _make_match(2, 2, state="Upcoming", home_id=1, away_id=3)
    row = _feature_dict(builder, m2)
    assert row["missing_team_stats"] == 1.0


def test_missing_team_stats_stays_1_when_a_stat_window_is_absent():
    """Partial NRL team-stat blocks should keep the missing flag raised."""
    builder = FeatureBuilder()
    partial_stats = [
        _stat("Kicking Metres", 400.0, 300.0),
        _stat("Kick Return Metres", 100.0, 80.0),
        _stat("Line Breaks", 3.0, 2.0),
    ]
    m1 = _make_match(1, 1, h_score=20, a_score=10, stats=partial_stats)
    builder.record(m1)
    m2 = _make_match(2, 2, state="Upcoming")
    row = _feature_dict(builder, m2)
    assert row["missing_team_stats"] == 1.0


def test_rolling_kick_metres_averages_correctly():
    """rolling_kick_metres_diff reflects the rolling-5 average of recorded values."""
    builder = FeatureBuilder()
    for i in range(1, 4):
        stats = [
            _stat("Kicking Metres", float(100 * i), float(50 * i)),
            _stat("Kick Return Metres", 80.0, 60.0),
            _stat("Line Breaks", 2.0, 1.0),
            _stat("All Runs", 100.0, 90.0),
        ]
        builder.record(_make_match(i, i, h_score=20, a_score=10, stats=stats))

    m_next = _make_match(10, 10, state="Upcoming")
    row = _feature_dict(builder, m_next)
    # Home avg kicking metres = (100+200+300)/3 = 200; away = (50+100+150)/3 = 100
    assert row["rolling_kick_metres_diff"] == pytest.approx(100.0, abs=0.5)


def test_feature_names_count():
    """FEATURE_NAMES should include the 5 new team-stat features."""
    assert "rolling_kick_metres_diff" in FEATURE_NAMES
    assert "rolling_kick_return_metres_diff" in FEATURE_NAMES
    assert "rolling_line_breaks_diff" in FEATURE_NAMES
    assert "rolling_all_runs_diff" in FEATURE_NAMES
    assert "missing_team_stats" in FEATURE_NAMES
