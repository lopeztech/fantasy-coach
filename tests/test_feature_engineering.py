from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np

from fantasy_coach.feature_engineering import (
    FEATURE_NAMES,
    build_training_frame,
)
from fantasy_coach.features import MatchRow, TeamRow


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
