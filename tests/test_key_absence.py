"""Tests for the position-weighted "key absence" feature (#27)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fantasy_coach.feature_engineering import (
    FEATURE_NAMES,
    KEY_ABSENCE_REGULAR_MIN_STARTS,
    POSITION_WEIGHTS,
    FeatureBuilder,
)
from fantasy_coach.features import MatchRow, PlayerRow, TeamRow


def _player(
    player_id: int,
    *,
    jersey: int,
    position: str,
    is_on_field: bool = True,
) -> PlayerRow:
    return PlayerRow(
        player_id=player_id,
        jersey_number=jersey,
        position=position,
        first_name=f"F{player_id}",
        last_name=f"L{player_id}",
        is_on_field=is_on_field,
    )


# Canonical starting XIII: one player per starting position.
def _standard_xiii(base: int = 100) -> list[PlayerRow]:
    positions = [
        (1, "Fullback"),
        (2, "Winger"),
        (3, "Centre"),
        (4, "Centre"),
        (5, "Winger"),
        (6, "Five-Eighth"),
        (7, "Halfback"),
        (8, "Prop"),
        (9, "Hooker"),
        (10, "Prop"),
        (11, "2nd Row"),
        (12, "2nd Row"),
        (13, "Lock"),
    ]
    return [_player(base + j, jersey=j, position=pos) for j, pos in positions]


def _match(
    match_id: int,
    *,
    season: int = 2024,
    round_: int = 1,
    start_time: datetime,
    home_team_id: int,
    away_team_id: int,
    home_players: list[PlayerRow],
    away_players: list[PlayerRow],
    home_score: int = 0,
    away_score: int = 0,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round_,
        start_time=start_time,
        match_state="FullTime",
        venue="Eden Park",
        venue_city="Auckland",
        weather=None,
        home=TeamRow(
            team_id=home_team_id,
            name=f"Team{home_team_id}",
            nick_name=f"T{home_team_id}",
            score=home_score,
            players=home_players,
        ),
        away=TeamRow(
            team_id=away_team_id,
            name=f"Team{away_team_id}",
            nick_name=f"T{away_team_id}",
            score=away_score,
            players=away_players,
        ),
        team_stats=[],
    )


# ---------------------------------------------------------------------------
# FEATURE_NAMES wiring
# ---------------------------------------------------------------------------


def test_key_absence_diff_is_registered() -> None:
    assert "key_absence_diff" in FEATURE_NAMES


# ---------------------------------------------------------------------------
# _key_absence behaviour
# ---------------------------------------------------------------------------


def test_zero_absence_when_no_history() -> None:
    builder = FeatureBuilder()
    # No prior matches recorded; regular XIII is unknown → 0.
    absence = builder._key_absence(team_id=10, current_players=_standard_xiii())
    assert absence == 0.0


def test_zero_absence_when_no_is_on_field_today() -> None:
    builder = FeatureBuilder()
    # Prime history with a standard XIII.
    for r in range(3):
        match = _match(
            match_id=r,
            round_=r + 1,
            start_time=datetime(2024, 3, r + 1, tzinfo=UTC),
            home_team_id=10,
            away_team_id=20,
            home_players=_standard_xiii(base=100),
            away_players=_standard_xiii(base=200),
        )
        builder.record(match)

    # Today's scrape has no is_on_field flag anywhere — we can't tell who's
    # starting, so the signal should be zero rather than a false alarm.
    today_players = [
        _player(100 + j, jersey=j, position="Fullback", is_on_field=False) for j in range(1, 14)
    ]
    assert builder._key_absence(team_id=10, current_players=today_players) == 0.0


def test_halfback_missing_weighted_higher_than_prop_missing() -> None:
    builder = FeatureBuilder()
    # Fill the window with enough matches to make player 107 (halfback) and
    # player 108 (prop) count as regulars.
    for r in range(KEY_ABSENCE_REGULAR_MIN_STARTS):
        match = _match(
            match_id=r,
            round_=r + 1,
            start_time=datetime(2024, 3, r + 1, tzinfo=UTC),
            home_team_id=10,
            away_team_id=20,
            home_players=_standard_xiii(base=100),
            away_players=_standard_xiii(base=200),
        )
        builder.record(match)

    xiii = _standard_xiii(base=100)
    without_halfback = [p for p in xiii if p.player_id != 107]
    without_prop = [p for p in xiii if p.player_id != 108]

    abs_halfback = builder._key_absence(team_id=10, current_players=without_halfback)
    abs_prop = builder._key_absence(team_id=10, current_players=without_prop)

    assert abs_halfback == pytest.approx(POSITION_WEIGHTS["Halfback"])
    assert abs_prop == pytest.approx(POSITION_WEIGHTS["Prop"])
    # Sanity: the weight table puts halfback well above prop, so the feature
    # will distinguish the two.
    assert abs_halfback > abs_prop


def test_one_off_fill_in_not_counted_as_regular() -> None:
    builder = FeatureBuilder()
    # Three matches: two with the normal XIII, one with a fill-in halfback.
    for r, halfback_pid in enumerate([107, 107, 999]):
        xiii = _standard_xiii(base=100)
        xiii = [_player(halfback_pid, jersey=7, position="Halfback")] + [
            p for p in xiii if p.jersey_number != 7
        ]
        builder.record(
            _match(
                match_id=r,
                round_=r + 1,
                start_time=datetime(2024, 3, r + 1, tzinfo=UTC),
                home_team_id=10,
                away_team_id=20,
                home_players=xiii,
                away_players=_standard_xiii(base=200),
            )
        )

    # Today: regular halfback 107 is missing, fill-in 999 is also missing.
    # Only 107 should count — 999 started just once in the window.
    today = [p for p in _standard_xiii(base=100) if p.player_id != 107]
    absence = builder._key_absence(team_id=10, current_players=today)
    assert absence == pytest.approx(POSITION_WEIGHTS["Halfback"])


def test_key_absence_diff_shows_up_in_feature_row() -> None:
    builder = FeatureBuilder()
    # Prime both teams' history over enough matches.
    for r in range(KEY_ABSENCE_REGULAR_MIN_STARTS + 1):
        match = _match(
            match_id=r,
            round_=r + 1,
            start_time=datetime(2024, 3, r + 1, tzinfo=UTC),
            home_team_id=10,
            away_team_id=20,
            home_players=_standard_xiii(base=100),
            away_players=_standard_xiii(base=200),
        )
        builder.advance_season_if_needed(match)
        builder.record(match)

    # Today: home missing halfback, away intact. Expect diff = +W[Halfback].
    today_home = [p for p in _standard_xiii(base=100) if p.player_id != 107]
    today_away = _standard_xiii(base=200)
    current = _match(
        match_id=99,
        round_=99,
        start_time=datetime(2024, 4, 1, tzinfo=UTC),
        home_team_id=10,
        away_team_id=20,
        home_players=today_home,
        away_players=today_away,
    )
    row = builder.feature_row(current)
    idx = FEATURE_NAMES.index("key_absence_diff")
    assert row[idx] == pytest.approx(POSITION_WEIGHTS["Halfback"])
