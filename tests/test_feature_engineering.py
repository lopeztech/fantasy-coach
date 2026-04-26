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
from fantasy_coach.features import MatchRow, PlayerRow, TeamRow


def _match(
    *,
    match_id: int,
    home_id: int,
    away_id: int,
    home_score: int,
    away_score: int,
    when: datetime,
    state: str = "FullTime",
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=when.year,
        round=1,
        start_time=when,
        match_state=state,
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
