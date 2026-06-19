from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from fantasy_coach.feature_engineering import (
    FEATURE_NAMES,
    FeatureBuilder,
    build_training_frame,
)
from fantasy_coach.features import MatchRow, PlayerMatchStat, PlayerRow, TeamRow


def _match(
    *,
    match_id: int,
    home_id: int,
    away_id: int,
    home_score: int,
    away_score: int,
    when: datetime,
    state: str = "FullTime",
    venue: str | None = None,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=when.year,
        round=1,
        start_time=when,
        match_state=state,
        venue=venue,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name=str(home_id),
            nick_name=str(home_id),
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=away_id,
            name=str(away_id),
            nick_name=str(away_id),
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


def test_first_match_uses_default_priors() -> None:
    when = datetime(2024, 3, 1, tzinfo=UTC)
    frame = build_training_frame(
        [
            _match(
                match_id=1,
                home_id=10,
                away_id=20,
                home_score=24,
                away_score=18,
                when=when,
            )
        ]
    )

    assert frame.X.shape == (1, len(FEATURE_NAMES))
    row = dict(zip(FEATURE_NAMES, frame.X[0], strict=True))
    # No prior matches for either team → form diffs are zero.
    assert row["form_diff_pf"] == 0.0
    assert row["form_diff_pa"] == 0.0
    # No prior h2h.
    assert row["h2h_recent_diff"] == 0.0
    # Both teams default to 14 days rest → diff is zero.
    assert row["days_rest_diff"] == 0.0
    # Constant home indicator.
    assert row["is_home_field"] == 1.0
    # Elo book starts symmetric → home advantage drives this.
    assert row["elo_diff"] > 0


def test_no_leakage_features_use_only_prior_data() -> None:
    # Two matches involving same team. The second match's features should
    # reflect only the first match's outcome — never its own.
    base = datetime(2024, 3, 1, tzinfo=UTC)
    matches = [
        _match(match_id=1, home_id=10, away_id=20, home_score=40, away_score=10, when=base),
        _match(
            match_id=2,
            home_id=10,
            away_id=30,
            home_score=12,
            away_score=24,
            when=base + timedelta(days=7),
        ),
    ]
    frame = build_training_frame(matches)

    second = dict(zip(FEATURE_NAMES, frame.X[1], strict=True))
    # After match 1, team 10's points-for avg is 40, points-against is 10.
    # Team 30 has no prior matches → 0/0. So:
    assert second["form_diff_pf"] == 40.0
    assert second["form_diff_pa"] == 10.0


def test_targets_match_outcomes_and_drop_draws() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    matches = [
        _match(match_id=1, home_id=10, away_id=20, home_score=24, away_score=12, when=base),
        _match(
            match_id=2,
            home_id=30,
            away_id=40,
            home_score=18,
            away_score=18,  # draw → dropped
            when=base + timedelta(days=1),
        ),
        _match(
            match_id=3,
            home_id=50,
            away_id=60,
            home_score=10,
            away_score=22,
            when=base + timedelta(days=2),
        ),
    ]
    frame = build_training_frame(matches)

    assert frame.match_ids.tolist() == [1, 3]
    assert frame.y.tolist() == [1, 0]


def test_h2h_recent_uses_home_perspective() -> None:
    # Lower-id team wins big when home; flip home/away in the next meeting and
    # the h2h_recent feature should be from the new home's perspective.
    base = datetime(2024, 3, 1, tzinfo=UTC)
    matches = [
        # First meeting: 10 home, beats 20 by 30.
        _match(match_id=1, home_id=10, away_id=20, home_score=40, away_score=10, when=base),
        # Second meeting: 20 is home now, plays 10. h2h should reflect that
        # 20 is on the wrong end of a recent +30 margin (i.e. h2h_recent_diff
        # from 20's perspective should be -30).
        _match(
            match_id=2,
            home_id=20,
            away_id=10,
            home_score=12,
            away_score=24,
            when=base + timedelta(days=7),
        ),
    ]
    frame = build_training_frame(matches)

    second = dict(zip(FEATURE_NAMES, frame.X[1], strict=True))
    assert second["h2h_recent_diff"] == -30.0


def test_chronology_independent_of_input_order() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    matches = [
        _match(match_id=1, home_id=10, away_id=20, home_score=24, away_score=12, when=base),
        _match(
            match_id=2,
            home_id=30,
            away_id=10,
            home_score=14,
            away_score=20,
            when=base + timedelta(days=7),
        ),
    ]
    rng = random.Random(0)
    shuffled = list(matches)
    rng.shuffle(shuffled)

    a = build_training_frame(matches)
    b = build_training_frame(shuffled)

    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.match_ids, b.match_ids)


def test_unfinished_matches_are_excluded() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    frame = build_training_frame(
        [
            _match(
                match_id=1,
                home_id=10,
                away_id=20,
                home_score=0,
                away_score=0,
                when=base,
                state="Upcoming",
            )
        ]
    )
    assert frame.X.shape == (0, len(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# H2H last-5 features (#168)
# ---------------------------------------------------------------------------


def test_h2h_last5_neutral_on_first_meeting() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    frame = build_training_frame(
        [_match(match_id=1, home_id=10, away_id=20, home_score=24, away_score=18, when=base)]
    )
    row = dict(zip(FEATURE_NAMES, frame.X[0], strict=True))
    assert row["h2h_last5_home_win_rate"] == 0.5
    assert row["h2h_last5_avg_margin"] == 0.0
    assert row["missing_h2h"] == 1.0


def test_h2h_last5_neutral_when_fewer_than_3_encounters() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    matches = [
        _match(match_id=1, home_id=10, away_id=20, home_score=24, away_score=18, when=base),
        _match(
            match_id=2,
            home_id=10,
            away_id=20,
            home_score=20,
            away_score=14,
            when=base + timedelta(days=7),
        ),
        # Third meeting — should still be missing since we need >= 3 PRIOR encounters.
        _match(
            match_id=3,
            home_id=10,
            away_id=20,
            home_score=30,
            away_score=10,
            when=base + timedelta(days=14),
        ),
    ]
    frame = build_training_frame(matches)
    rows = [dict(zip(FEATURE_NAMES, frame.X[i], strict=True)) for i in range(3)]
    # First two meetings: 0 and 1 prior encounters → missing.
    assert rows[0]["missing_h2h"] == 1.0
    assert rows[1]["missing_h2h"] == 1.0
    # Third meeting: 2 prior encounters → still below H2H_MIN_ENCOUNTERS=3 → missing.
    assert rows[2]["missing_h2h"] == 1.0


def test_h2h_last5_populated_after_3_encounters() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    # Build up 3 prior meetings, then check the 4th.
    matches = [
        _match(match_id=1, home_id=10, away_id=20, home_score=30, away_score=10, when=base),
        _match(
            match_id=2,
            home_id=10,
            away_id=20,
            home_score=20,
            away_score=20,  # draw → dropped from training frame, still recorded
            when=base + timedelta(days=7),
        ),
        _match(
            match_id=3,
            home_id=10,
            away_id=20,
            home_score=18,
            away_score=16,
            when=base + timedelta(days=14),
        ),
        _match(
            match_id=4,
            home_id=10,
            away_id=20,
            home_score=26,
            away_score=12,
            when=base + timedelta(days=21),
        ),
    ]
    frame = build_training_frame(matches)
    # frame has 3 rows (draw is dropped).  The last row corresponds to match 4.
    last = dict(zip(FEATURE_NAMES, frame.X[-1], strict=True))
    assert last["missing_h2h"] == 0.0
    # 3 prior encounters (margins from home's perspective: +20, 0, +2).
    # win_rate = 2/3 (drew counts as non-win), avg_margin = (20+0+2)/3 = 7.33
    assert last["h2h_last5_home_win_rate"] == pytest.approx(2 / 3)
    assert last["h2h_last5_avg_margin"] == pytest.approx(22 / 3, abs=0.01)


def test_h2h_last5_home_perspective_swaps_with_venue() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    # Build 5 meetings where team 10 always beats team 20 by 20 at home.
    prior = [
        _match(
            match_id=i,
            home_id=10,
            away_id=20,
            home_score=30,
            away_score=10,
            when=base + timedelta(days=(i - 1) * 7),
        )
        for i in range(1, 6)
    ]
    # 6th meeting: 20 is now home. All 5 prior meetings were wins for 10 (away).
    sixth = _match(
        match_id=6,
        home_id=20,
        away_id=10,
        home_score=16,
        away_score=24,
        when=base + timedelta(days=35),
    )
    frame = build_training_frame([*prior, sixth])
    last = dict(zip(FEATURE_NAMES, frame.X[-1], strict=True))
    assert last["missing_h2h"] == 0.0
    # From team 20's (home) perspective, every prior meeting was a -20 loss.
    assert last["h2h_last5_home_win_rate"] == pytest.approx(0.0)
    assert last["h2h_last5_avg_margin"] == pytest.approx(-20.0)


def test_h2h_last5_margin_clipped_to_30() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    # 5 blowout wins by home team: margin = 60.
    matches = [
        _match(
            match_id=i,
            home_id=10,
            away_id=20,
            home_score=70,
            away_score=10,
            when=base + timedelta(days=(i - 1) * 7),
        )
        for i in range(1, 7)
    ]
    frame = build_training_frame(matches)
    last = dict(zip(FEATURE_NAMES, frame.X[-1], strict=True))
    assert last["missing_h2h"] == 0.0
    assert last["h2h_last5_avg_margin"] == pytest.approx(30.0)  # clipped from 60


# ---------- line-move features (#169) ----------


def test_line_move_missing_when_no_opening_odds() -> None:
    """Without opening odds, line-move is zero and missing flag is 1."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    match = _match(match_id=1, home_id=10, away_id=20, home_score=24, away_score=18, when=base)
    # No odds at all → missing_odds=1 AND missing_line_move=1
    frame = build_training_frame([match])
    row = dict(zip(FEATURE_NAMES, frame.X[0], strict=True))
    assert row["missing_line_move"] == 1.0
    assert row["odds_line_move_home_prob"] == pytest.approx(0.0)
    assert row["odds_line_move_magnitude"] == pytest.approx(0.0)


def test_line_move_missing_when_only_closing_odds() -> None:
    """Closing odds present but no opening odds → still missing."""
    base = datetime(2024, 3, 1, tzinfo=UTC)
    match = MatchRow(
        match_id=1,
        season=2024,
        round=1,
        start_time=base,
        match_state="FullTime",
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(team_id=10, name="H", nick_name="H", score=24, players=[], odds=1.83),
        away=TeamRow(team_id=20, name="A", nick_name="A", score=18, players=[], odds=2.10),
        team_stats=[],
    )
    frame = build_training_frame([match])
    row = dict(zip(FEATURE_NAMES, frame.X[0], strict=True))
    assert row["missing_line_move"] == 1.0
    assert row["odds_line_move_home_prob"] == pytest.approx(0.0)


def test_line_move_computed_when_opening_odds_present() -> None:
    """With both open and close odds, line-move = close_prob − open_prob."""
    from fantasy_coach.bookmaker.lines import devig_two_way

    base = datetime(2024, 3, 1, tzinfo=UTC)
    match = MatchRow(
        match_id=1,
        season=2024,
        round=1,
        start_time=base,
        match_state="FullTime",
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=10, name="H", nick_name="H", score=24, players=[], odds=1.70, odds_open=1.80
        ),
        away=TeamRow(
            team_id=20, name="A", nick_name="A", score=18, players=[], odds=2.30, odds_open=2.25
        ),
        team_stats=[],
    )
    frame = build_training_frame([match])
    row = dict(zip(FEATURE_NAMES, frame.X[0], strict=True))

    p_open = devig_two_way(1.80, 2.25)
    p_close = devig_two_way(1.70, 2.30)
    expected_move = p_close - p_open

    assert row["missing_line_move"] == pytest.approx(0.0)
    assert row["odds_line_move_home_prob"] == pytest.approx(expected_move, rel=1e-6)
    assert row["odds_line_move_magnitude"] == pytest.approx(abs(expected_move), rel=1e-6)
    # Market moved toward home (shorter close price) → positive move.
    assert row["odds_line_move_home_prob"] > 0


# ---------------------------------------------------------------------------
# Position-pair matchup features (#210)
# ---------------------------------------------------------------------------


def _player(player_id: int, position: str, *, is_on_field: bool = True) -> PlayerRow:
    return PlayerRow(
        player_id=player_id,
        jersey_number=player_id,
        position=position,
        first_name="F",
        last_name="L",
        is_on_field=is_on_field,
    )


def _make_match_with_players(
    home_players: list[PlayerRow],
    away_players: list[PlayerRow],
    *,
    match_id: int = 1,
    home_id: int = 10,
    away_id: int = 20,
) -> MatchRow:
    when = datetime(2025, 3, 1, tzinfo=UTC)
    return MatchRow(
        match_id=match_id,
        season=2025,
        round=1,
        start_time=when,
        match_state="FullTime",
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name="H",
            nick_name="H",
            score=24,
            players=home_players,
        ),
        away=TeamRow(
            team_id=away_id,
            name="A",
            nick_name="A",
            score=18,
            players=away_players,
        ),
        team_stats=[],
    )


def test_position_pair_features_zero_without_roster_data() -> None:
    """When neither team has is_on_field data all group diffs are 0.0."""
    frame = build_training_frame([_make_match_with_players([], [])])
    row = dict(zip(FEATURE_NAMES, frame.X[0], strict=True))
    for feat in (
        "halves_strength_diff",
        "forwards_strength_diff",
        "hooker_strength_diff",
        "outside_backs_strength_diff",
        "halves_x_forwards_diff",
    ):
        assert row[feat] == pytest.approx(0.0), f"{feat} should be 0.0 with no roster data"


def test_position_pair_halves_diff_reflects_rating_gap() -> None:
    """Home has a halfback and away has none → positive halves_strength_diff."""
    builder = FeatureBuilder()
    home_players = [_player(101, "Halfback"), _player(102, "Five-Eighth")]
    away_players = [_player(201, "Prop"), _player(202, "Prop")]
    match = _make_match_with_players(home_players, away_players)

    row = dict(zip(FEATURE_NAMES, builder.feature_row(match), strict=True))

    # Home has 2 halves at default rating; away has 0 → positive diff.
    assert row["halves_strength_diff"] > 0
    assert row["forwards_strength_diff"] < 0  # away has 2 props
    assert row["hooker_strength_diff"] == pytest.approx(0.0)
    assert row["outside_backs_strength_diff"] == pytest.approx(0.0)


def test_halves_x_forwards_diff_fires_only_on_same_sign() -> None:
    """Interaction term is non-zero only when halves and forwards push the same way."""
    builder = FeatureBuilder()

    # Home dominant on halves AND forwards → interaction fires.
    home_both = [_player(1, "Halfback"), _player(2, "Prop")]
    away_neither = [_player(3, "Winger"), _player(4, "Winger")]
    match_same = _make_match_with_players(home_both, away_neither, match_id=1)
    row_same = dict(zip(FEATURE_NAMES, builder.feature_row(match_same), strict=True))
    assert row_same["halves_x_forwards_diff"] != pytest.approx(0.0)

    # Home strong halves, away strong forwards → opposing signs → interaction = 0.
    home_halves_only = [_player(5, "Halfback"), _player(6, "Five-Eighth")]
    away_forwards_only = [_player(7, "Prop"), _player(8, "Lock")]
    match_opp = _make_match_with_players(
        home_halves_only, away_forwards_only, match_id=2, home_id=11, away_id=21
    )
    row_opp = dict(zip(FEATURE_NAMES, builder.feature_row(match_opp), strict=True))
    assert row_opp["halves_x_forwards_diff"] == pytest.approx(0.0)


def test_position_pair_feature_count_matches_feature_names() -> None:
    """Smoke test: feature_row length matches FEATURE_NAMES."""
    builder = FeatureBuilder()
    match = _make_match_with_players([], [])
    row = builder.feature_row(match)
    assert len(row) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Ladder-position features (#255)
# ---------------------------------------------------------------------------


def _match_r(
    *,
    match_id: int,
    home_id: int,
    away_id: int,
    home_score: int,
    away_score: int,
    round_: int,
    season: int = 2025,
) -> MatchRow:
    """Helper like _match() but accepts explicit round number."""
    from datetime import UTC

    when = datetime(season, 3, 1, tzinfo=UTC) + timedelta(weeks=round_ - 1)
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round_,
        start_time=when,
        match_state="FullTime",
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name=str(home_id),
            nick_name=str(home_id),
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=away_id,
            name=str(away_id),
            nick_name=str(away_id),
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


def test_ladder_missing_flag_for_early_rounds() -> None:
    """missing_ladder=1.0 for rounds below LADDER_MIN_ROUND."""
    builder = FeatureBuilder()
    match = _match_r(match_id=1, home_id=10, away_id=20, home_score=24, away_score=12, round_=2)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(match), strict=True))
    assert row["missing_ladder"] == 1.0
    assert row["ladder_position_diff"] == 0.0
    assert row["must_win_intensity"] == 0.0
    assert row["dead_rubber_indicator"] == 0.0


def test_ladder_position_diff_after_several_rounds() -> None:
    """Home is ranked higher (lower position number) → negative ladder_position_diff."""
    builder = FeatureBuilder()
    teams = list(range(10, 26))  # 16 teams
    # Play 6 rounds: team 10 wins all, team 20 loses all, others split.
    match_id = 0
    for r in range(1, 7):
        for i in range(0, len(teams), 2):
            h, a = teams[i], teams[i + 1]
            # team 10 always wins at home, team 20 always loses as away
            match = _match_r(
                match_id=match_id,
                home_id=h,
                away_id=a,
                home_score=20,
                away_score=10,
                round_=r,
            )
            builder.advance_season_if_needed(match)
            builder.record(match)
            match_id += 1

    # After 6 rounds team 10 (home winner every round) should be high on table;
    # team 25 (always away loser) should be low.
    query = _match_r(
        match_id=match_id, home_id=10, away_id=25, home_score=0, away_score=0, round_=7
    )
    builder.advance_season_if_needed(query)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))
    assert row["missing_ladder"] == 0.0
    # home (10) should have better (lower) position than away (25)
    assert row["ladder_position_diff"] < 0


def test_ladder_must_win_intensity_for_outside_top8() -> None:
    """A team on the ladder bubble with few games remaining has high must_win_intensity."""
    from fantasy_coach.feature_engineering import NRL_SEASON_ROUNDS

    builder = FeatureBuilder()
    # Simulate late season: play NRL_SEASON_ROUNDS - 2 rounds for many teams.
    # team 10 sits just outside top 8 (8th has 2 more pts, 1 game remaining for both).
    teams_top8 = list(range(1, 9))
    teams_outside = list(range(9, 17))
    match_id = 0

    rounds_to_play = NRL_SEASON_ROUNDS - 2
    for r in range(1, rounds_to_play + 1):
        for h, a in zip(teams_top8, teams_outside, strict=True):
            m = _match_r(
                match_id=match_id,
                home_id=h,
                away_id=a,
                home_score=20,
                away_score=10,
                round_=r,
            )
            builder.advance_season_if_needed(m)
            builder.record(m)
            match_id += 1

    # Query: team 10 vs team 1 in final rounds — team 10 is outside top 8.
    query = _match_r(
        match_id=match_id,
        home_id=10,  # outside top 8
        away_id=1,  # inside top 8
        home_score=0,
        away_score=0,
        round_=NRL_SEASON_ROUNDS - 1,
    )
    builder.advance_season_if_needed(query)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))
    # Home is outside top 8 with few games left → must_win_intensity > 0
    assert row["missing_ladder"] == 0.0
    assert row["must_win_intensity"] > 0.0


def test_ladder_dead_rubber_when_top8_status_locked() -> None:
    """dead_rubber_indicator fires when both teams' top-8 status is mathematically decided."""
    from fantasy_coach.feature_engineering import NRL_SEASON_ROUNDS

    builder = FeatureBuilder()
    # team 1 wins all → comfortably in top 8
    # team 20 loses all → comfortably out of top 8
    # After 26 rounds both positions are locked.
    teams = list(range(1, 17))
    match_id = 0
    for r in range(1, NRL_SEASON_ROUNDS):
        for i in range(0, len(teams), 2):
            h, a = teams[i], teams[i + 1]
            m = _match_r(
                match_id=match_id,
                home_id=h,
                away_id=a,
                home_score=30,
                away_score=0,
                round_=r,
            )
            builder.advance_season_if_needed(m)
            builder.record(m)
            match_id += 1

    # Final round: top-ranked vs bottom-ranked — neither can change status.
    query = _match_r(
        match_id=match_id,
        home_id=teams[0],  # top of table
        away_id=teams[-1],  # bottom of table
        home_score=0,
        away_score=0,
        round_=NRL_SEASON_ROUNDS,
    )
    builder.advance_season_if_needed(query)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))
    assert row["missing_ladder"] == 0.0
    assert row["dead_rubber_indicator"] == 1.0


# ---------------------------------------------------------------------------
# Player form-trajectory features (#251)
# ---------------------------------------------------------------------------


def _make_player_stat(
    player_id: int,
    *,
    tackle_breaks: int = 0,
    line_breaks: int = 0,
    try_assists: int = 0,
    tackles_made: int = 20,
    errors: int = 0,
) -> PlayerMatchStat:
    return PlayerMatchStat(
        player_id=player_id,
        tackle_breaks=tackle_breaks,
        line_breaks=line_breaks,
        try_assists=try_assists,
        tackles_made=tackles_made,
        errors=errors,
    )


def test_trajectory_zero_with_no_player_data() -> None:
    """When no player stats are available both trajectory diffs are 0.0."""
    builder = FeatureBuilder()
    home_players = [_player(1, "Halfback")]
    away_players = [_player(2, "Halfback")]
    match = _make_match_with_players(home_players, away_players)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(match), strict=True))
    assert row["player_form_trajectory_diff"] == pytest.approx(0.0)
    assert row["key_player_trajectory_diff"] == pytest.approx(0.0)


def test_trajectory_missing_flag_fires_when_roster_lacks_history() -> None:
    """missing_player_trajectory=1.0 when ≥5 XIII players have < 3 recorded composites."""
    builder = FeatureBuilder()
    # 6 home starters, none with history → missing flag fires
    home_players = [_player(i, "Prop") for i in range(1, 7)]
    away_players = [_player(i, "Prop") for i in range(11, 17)]
    match = _make_match_with_players(home_players, away_players)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(match), strict=True))
    assert row["missing_player_trajectory"] == 1.0


def test_trajectory_strictly_walk_forward() -> None:
    """Trajectory must not include the current match's own stats."""
    builder = FeatureBuilder()
    home_players = [_player(1, "Halfback")]
    away_players = [_player(2, "Halfback")]

    # Play 5 prior matches so player 1 has history.
    for i in range(1, 6):
        m = MatchRow(
            match_id=i,
            season=2025,
            round=i,
            start_time=datetime(2025, 3, 1, tzinfo=UTC) + timedelta(weeks=i - 1),
            match_state="FullTime",
            venue=None,
            venue_city=None,
            weather=None,
            home=TeamRow(team_id=10, name="H", nick_name="H", score=20, players=home_players),
            away=TeamRow(team_id=20, name="A", nick_name="A", score=10, players=away_players),
            team_stats=[],
            home_player_stats=[_make_player_stat(1, line_breaks=i)],
            away_player_stats=[_make_player_stat(2, line_breaks=0)],
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    # In match 6, player 1 will have a huge line-break game; feature_row
    # must only see rounds 1–5 (not round 6's stats).
    query = MatchRow(
        match_id=6,
        season=2025,
        round=6,
        start_time=datetime(2025, 3, 1, tzinfo=UTC) + timedelta(weeks=5),
        match_state="FullTime",
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(team_id=10, name="H", nick_name="H", score=20, players=home_players),
        away=TeamRow(team_id=20, name="A", nick_name="A", score=10, players=away_players),
        team_stats=[],
        home_player_stats=[_make_player_stat(1, line_breaks=100)],  # huge game
        away_player_stats=[_make_player_stat(2, line_breaks=0)],
    )
    builder.advance_season_if_needed(query)
    row_before = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))
    # Trajectory is based on rounds 1–5 only; the 100-line-break round hasn't
    # been recorded yet → trajectory should be a small positive (rounds 3-5
    # avg > season avg of rounds 1-5).
    assert row_before["player_form_trajectory_diff"] < 50  # definitely not 100


def test_improving_spine_produces_positive_key_trajectory() -> None:
    """Team with improving spine (halves + hooker) has positive key_player_trajectory_diff."""
    builder = FeatureBuilder()
    halfback_home = _player(1, "Halfback")
    halfback_away = _player(2, "Halfback")

    # Play 6 matches: home halfback improves each round; away stays flat.
    for i in range(1, 7):
        m = MatchRow(
            match_id=i,
            season=2025,
            round=i,
            start_time=datetime(2025, 3, 1, tzinfo=UTC) + timedelta(weeks=i - 1),
            match_state="FullTime",
            venue=None,
            venue_city=None,
            weather=None,
            home=TeamRow(team_id=10, name="H", nick_name="H", score=20, players=[halfback_home]),
            away=TeamRow(team_id=20, name="A", nick_name="A", score=10, players=[halfback_away]),
            team_stats=[],
            home_player_stats=[_make_player_stat(1, line_breaks=i * 2)],  # trending up
            away_player_stats=[_make_player_stat(2, line_breaks=1)],  # flat
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    query = _make_match_with_players([halfback_home], [halfback_away], match_id=7)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))
    # Home halfback is trending up → positive key_player_trajectory_diff.
    assert row["key_player_trajectory_diff"] > 0


def test_winger_trajectory_does_not_count_as_key_spine_trajectory() -> None:
    """Key trajectory is halves + hooker + fullback, not all outside backs."""
    builder = FeatureBuilder()
    winger_home = _player(1, "Winger")
    winger_away = _player(2, "Winger")

    for i in range(1, 7):
        m = MatchRow(
            match_id=i,
            season=2025,
            round=i,
            start_time=datetime(2025, 3, 1, tzinfo=UTC) + timedelta(weeks=i - 1),
            match_state="FullTime",
            venue=None,
            venue_city=None,
            weather=None,
            home=TeamRow(team_id=10, name="H", nick_name="H", score=20, players=[winger_home]),
            away=TeamRow(team_id=20, name="A", nick_name="A", score=10, players=[winger_away]),
            team_stats=[],
            home_player_stats=[_make_player_stat(1, line_breaks=i * 2)],
            away_player_stats=[_make_player_stat(2, line_breaks=1)],
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    query = _make_match_with_players([winger_home], [winger_away], match_id=7)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))

    assert row["player_form_trajectory_diff"] > 0
    assert row["key_player_trajectory_diff"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cumulative fatigue index features (#252)
# ---------------------------------------------------------------------------


def test_fatigue_zero_with_no_prior_matches() -> None:
    """Both teams have no prior matches → fatigue indices are 0.0."""
    builder = FeatureBuilder()
    match = _match(
        match_id=1,
        home_id=10,
        away_id=20,
        home_score=0,
        away_score=0,
        when=datetime(2025, 3, 1, tzinfo=UTC),
    )
    row = dict(zip(FEATURE_NAMES, builder.feature_row(match), strict=True))
    assert row["team_fatigue_index_diff"] == pytest.approx(0.0)
    assert row["spine_fatigue_index_diff"] == pytest.approx(0.0)
    assert row["cumulative_origin_minutes_diff"] == pytest.approx(0.0)


def test_fatigue_accumulates_after_matches() -> None:
    """After recording several matches, at least one team has non-zero fatigue."""
    builder = FeatureBuilder()
    base = datetime(2025, 3, 1, tzinfo=UTC)

    # Play 5 matches for team 10 on short rest intervals.
    for i in range(5):
        m = _match(
            match_id=i + 1,
            home_id=10,
            away_id=20,
            home_score=20,
            away_score=10,
            when=base + timedelta(days=i * 5),  # 5-day turnarounds = short rest
        )
        builder.advance_season_if_needed(m)
        builder.record(m)

    # Team 10 has played 5 short-rest matches; team 30 is fresh.
    query = _match(
        match_id=10,
        home_id=10,
        away_id=30,
        home_score=0,
        away_score=0,
        when=base + timedelta(days=25),
    )
    builder.advance_season_if_needed(query)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))
    # Home (team 10) has accumulated fatigue; away (team 30) has none.
    # home fatigue > away → positive diff.
    assert row["team_fatigue_index_diff"] > 0.0


def test_fatigue_uses_previous_match_rest_and_venue() -> None:
    """Fatigue load should reflect the travel/rest before the recorded match."""
    base = datetime(2025, 3, 1, tzinfo=UTC)

    def _fatigue_after_second_match(days_between: int) -> float:
        builder = FeatureBuilder()
        first = _match(
            match_id=1,
            home_id=10,
            away_id=20,
            home_score=20,
            away_score=10,
            when=base,
            venue="Allianz Stadium",
        )
        second = _match(
            match_id=2,
            home_id=10,
            away_id=20,
            home_score=18,
            away_score=12,
            when=base + timedelta(days=days_between),
            venue="Suncorp Stadium",
        )
        query = _match(
            match_id=3,
            home_id=10,
            away_id=30,
            home_score=0,
            away_score=0,
            when=second.start_time + timedelta(days=5),
            venue="Suncorp Stadium",
        )
        for match in (first, second):
            builder.advance_season_if_needed(match)
            builder.record(match)
        row = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))
        return row["team_fatigue_index_diff"]

    normal_rest = _fatigue_after_second_match(7)
    short_rest = _fatigue_after_second_match(5)

    assert normal_rest > 0.0
    assert short_rest > normal_rest


def test_spine_fatigue_uses_four_spine_positions_only() -> None:
    """Centres and wingers should not inflate spine fatigue above team fatigue."""
    lineup = [
        _player(1, "Fullback"),
        _player(2, "Winger"),
        _player(3, "Centre"),
        _player(4, "Centre"),
        _player(5, "Winger"),
        _player(6, "Five-Eighth"),
        _player(7, "Halfback"),
        _player(8, "Prop"),
        _player(9, "Hooker"),
        _player(10, "Prop"),
        _player(11, "2nd Row"),
        _player(12, "2nd Row"),
        _player(13, "Lock"),
    ]
    builder = FeatureBuilder()
    base = datetime(2025, 3, 1, tzinfo=UTC)
    for i, venue in enumerate(("Allianz Stadium", "Suncorp Stadium"), start=1):
        match = MatchRow(
            match_id=i,
            season=2025,
            round=i,
            start_time=base + timedelta(days=(i - 1) * 5),
            match_state="FullTime",
            venue=venue,
            venue_city=None,
            weather=None,
            home=TeamRow(team_id=10, name="H", nick_name="H", score=20, players=lineup),
            away=TeamRow(team_id=20, name="A", nick_name="A", score=10, players=lineup),
            team_stats=[],
        )
        builder.advance_season_if_needed(match)
        builder.record(match)

    query = _match(
        match_id=3,
        home_id=10,
        away_id=30,
        home_score=0,
        away_score=0,
        when=base + timedelta(days=10),
        venue="Suncorp Stadium",
    )
    row = dict(zip(FEATURE_NAMES, builder.feature_row(query), strict=True))

    assert row["team_fatigue_index_diff"] > 0.0
    assert row["spine_fatigue_index_diff"] <= row["team_fatigue_index_diff"]


def test_fatigue_decays_over_time() -> None:
    """Fatigue from an old match has less impact than one from a recent match."""

    builder_recent = FeatureBuilder()
    builder_old = FeatureBuilder()
    base = datetime(2025, 3, 1, tzinfo=UTC)

    # Both builders: team 10 plays one normal match, then one travel match
    # whose completed load can decay before the query.
    for builder, when_offset in ((builder_recent, 7), (builder_old, 60)):
        first = _match(
            match_id=1,
            home_id=10,
            away_id=20,
            home_score=20,
            away_score=10,
            when=base,
            venue="Allianz Stadium",
        )
        second = _match(
            match_id=2,
            home_id=10,
            away_id=20,
            home_score=20,
            away_score=10,
            when=base + timedelta(days=5),
            venue="Suncorp Stadium",
        )
        for match in (first, second):
            builder.advance_season_if_needed(match)
            builder.record(match)

        # Query match is when_offset days after the travel match.
        query = _match(
            match_id=3,
            home_id=10,
            away_id=30,
            home_score=0,
            away_score=0,
            when=second.start_time + timedelta(days=when_offset),
        )
        builder.advance_season_if_needed(query)

    row_recent = dict(
        zip(
            FEATURE_NAMES,
            builder_recent.feature_row(
                _match(
                    match_id=2,
                    home_id=10,
                    away_id=30,
                    home_score=0,
                    away_score=0,
                    when=base + timedelta(days=12),
                )
            ),
            strict=True,
        )
    )
    row_old = dict(
        zip(
            FEATURE_NAMES,
            builder_old.feature_row(
                _match(
                    match_id=2,
                    home_id=10,
                    away_id=30,
                    home_score=0,
                    away_score=0,
                    when=base + timedelta(days=65),
                )
            ),
            strict=True,
        )
    )
    # Fatigue from 7 days ago > fatigue from 60 days ago.
    assert row_recent["team_fatigue_index_diff"] > row_old["team_fatigue_index_diff"]


# ---------------------------------------------------------------------------
# Injury severity feature (#269)
# ---------------------------------------------------------------------------


def _injury_row(builder, match):
    """Return the match's feature row as a {name: value} dict."""
    return dict(zip(FEATURE_NAMES, builder.feature_row(match), strict=True))


def _report(player_id, team_id, status, *, season=2025, round=1):
    from fantasy_coach.injury import InjuryReport

    return InjuryReport(
        player_id=player_id,
        season=season,
        round=round,
        team_id=team_id,
        status=status,
        weeks_out=None,
        body_part=None,
        source="test",
        scraped_at=datetime(2025, 2, 25, tzinfo=UTC),
    )


def test_injury_features_neutral_without_index() -> None:
    from fantasy_coach.feature_engineering import FeatureBuilder

    match = _make_match_with_players([_player(1, "Halfback")], [_player(2, "Prop")])
    row = _injury_row(FeatureBuilder(), match)
    assert row["injury_severity_diff"] == 0.0
    assert row["missing_injury_data"] == 1.0


def test_injury_severity_status_weighted_and_binary() -> None:
    from fantasy_coach.feature_engineering import FeatureBuilder, InjuryIndex
    from fantasy_coach.injury import InjuryStatus

    # Positions unknown (players never recorded) ⇒ weight 1.0 each. Home has one
    # OUT (1.0), away one TEST (0.5). Severity is a count — weeks_out ignored.
    idx = InjuryIndex.from_records(
        [_report(101, 10, InjuryStatus.OUT), _report(201, 20, InjuryStatus.TEST)]
    )
    match = _make_match_with_players([_player(1, "Halfback")], [_player(2, "Prop")])
    row = _injury_row(FeatureBuilder(injury_index=idx), match)
    assert row["injury_severity_diff"] == pytest.approx(0.5)
    assert row["missing_injury_data"] == 0.0


def test_injury_severity_ignores_weeks_out() -> None:
    from fantasy_coach.feature_engineering import FeatureBuilder, InjuryIndex
    from fantasy_coach.injury import InjuryReport, InjuryStatus

    def out(pid, team, weeks):
        return InjuryReport(
            player_id=pid,
            season=2025,
            round=1,
            team_id=team,
            status=InjuryStatus.OUT,
            weeks_out=weeks,
            body_part=None,
            source="t",
            scraped_at=datetime(2025, 2, 25, tzinfo=UTC),
        )

    # Same player count, wildly different weeks_out → identical severity.
    short = InjuryIndex.from_records([out(101, 10, 1)])
    long = InjuryIndex.from_records([out(101, 10, 26)])
    match = _make_match_with_players([_player(1, "Halfback")], [_player(2, "Prop")])
    s = _injury_row(FeatureBuilder(injury_index=short), match)["injury_severity_diff"]
    long_ = _injury_row(FeatureBuilder(injury_index=long), match)["injury_severity_diff"]
    assert s == pytest.approx(long_)


def test_injury_severity_position_weighted() -> None:
    from fantasy_coach.feature_engineering import (
        POSITION_WEIGHTS,
        FeatureBuilder,
        InjuryIndex,
    )
    from fantasy_coach.injury import InjuryStatus

    # Teach the builder that player 101 is a Halfback by recording a prior match
    # where they started there, then rule them OUT in a later round.
    prior = _make_match_with_players(
        [_player(101, "Halfback"), _player(102, "Prop")],
        [_player(201, "Hooker"), _player(202, "Lock")],
        match_id=1,
    )
    idx = InjuryIndex.from_records([_report(101, 10, InjuryStatus.OUT, round=2)])
    later = MatchRow(
        match_id=2,
        season=2025,
        round=2,
        start_time=datetime(2025, 3, 10, tzinfo=UTC),
        match_state="Upcoming",
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=10, name="H", nick_name="H", score=None, players=[_player(102, "Prop")]
        ),
        away=TeamRow(
            team_id=20, name="A", nick_name="A", score=None, players=[_player(201, "Hooker")]
        ),
        team_stats=[],
    )
    fb = FeatureBuilder(injury_index=idx)
    fb.record(prior)  # learns player 101's Halfback position walk-forward
    row = _injury_row(fb, later)
    # Halfback weight (3.0) >> the default 1.0 for an unknown player.
    assert row["injury_severity_diff"] == pytest.approx(POSITION_WEIGHTS["Halfback"])


def test_missing_injury_data_zero_for_scraped_quiet_week() -> None:
    from fantasy_coach.feature_engineering import FeatureBuilder, InjuryIndex
    from fantasy_coach.injury import InjuryStatus

    # The round was scraped (a report exists for some other team), but neither
    # of these two teams has an injury — that's data, not a gap.
    idx = InjuryIndex.from_records([_report(999, 30, InjuryStatus.OUT)])
    match = _make_match_with_players([_player(1, "Halfback")], [_player(2, "Prop")])
    row = _injury_row(FeatureBuilder(injury_index=idx), match)
    assert row["missing_injury_data"] == 0.0
    assert row["injury_severity_diff"] == 0.0


def test_injury_severity_diff_is_capped() -> None:
    from fantasy_coach.feature_engineering import (
        INJURY_SEVERITY_DIFF_CAP,
        FeatureBuilder,
        InjuryIndex,
    )
    from fantasy_coach.injury import InjuryStatus

    # 60 unknown-position OUT players on the home side ⇒ raw severity 60,
    # clamped to the cap.
    idx = InjuryIndex.from_records([_report(1000 + i, 10, InjuryStatus.OUT) for i in range(60)])
    match = _make_match_with_players([_player(1, "Halfback")], [_player(2, "Prop")])
    row = _injury_row(FeatureBuilder(injury_index=idx), match)
    assert row["injury_severity_diff"] == pytest.approx(INJURY_SEVERITY_DIFF_CAP)


def test_returning_status_carries_no_severity() -> None:
    from fantasy_coach.feature_engineering import FeatureBuilder, InjuryIndex
    from fantasy_coach.injury import InjuryStatus

    # RETURNING is excluded from severity (its rust hypothesis didn't hold in
    # the #269 ablation), but the round still counts as scraped.
    idx = InjuryIndex.from_records([_report(101, 10, InjuryStatus.RETURNING)])
    match = _make_match_with_players([_player(1, "Halfback")], [_player(2, "Prop")])
    row = _injury_row(FeatureBuilder(injury_index=idx), match)
    assert row["injury_severity_diff"] == 0.0
    assert row["missing_injury_data"] == 0.0


def test_active_injury_index_context_applies_to_bare_builder() -> None:
    from fantasy_coach.feature_engineering import (
        FeatureBuilder,
        InjuryIndex,
        active_injury_index,
    )
    from fantasy_coach.injury import InjuryStatus

    idx = InjuryIndex.from_records([_report(101, 10, InjuryStatus.OUT)])
    match = _make_match_with_players([_player(1, "Halfback")], [_player(2, "Prop")])
    # A FeatureBuilder() constructed with no explicit index picks up the active one.
    with active_injury_index(idx):
        row = _injury_row(FeatureBuilder(), match)
    assert row["injury_severity_diff"] == pytest.approx(1.0)
    # Outside the context the fallback is gone again.
    row_after = _injury_row(FeatureBuilder(), match)
    assert row_after["injury_severity_diff"] == 0.0
    assert row_after["missing_injury_data"] == 1.0
