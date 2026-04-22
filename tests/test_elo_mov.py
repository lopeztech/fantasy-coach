"""Tests for the MOV-weighted Elo rater (EloMOV)."""

from __future__ import annotations

import math

import pytest

from fantasy_coach.models.elo import DEFAULT_HOME_ADVANTAGE, DEFAULT_INITIAL_RATING, DEFAULT_K
from fantasy_coach.models.elo_mov import EloMOV

# ---------------------------------------------------------------------------
# Construction / interface
# ---------------------------------------------------------------------------


def test_elov_mov_inherits_defaults() -> None:
    elo = EloMOV()
    assert elo.k == DEFAULT_K
    assert elo.home_advantage == DEFAULT_HOME_ADVANTAGE
    assert elo.initial_rating == DEFAULT_INITIAL_RATING


def test_new_team_starts_at_initial_rating() -> None:
    elo = EloMOV()
    assert elo.rating(99) == DEFAULT_INITIAL_RATING


def test_predict_with_equal_ratings_reflects_home_advantage() -> None:
    elo = EloMOV()
    p_home = elo.predict(1, 2)
    assert p_home > 0.5
    expected = 1.0 / (1.0 + 10.0 ** (-DEFAULT_HOME_ADVANTAGE / 400.0))
    assert math.isclose(p_home, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# MOV-weighted update
# ---------------------------------------------------------------------------


def test_larger_margin_produces_larger_delta() -> None:
    """A blowout should move ratings more than a narrow win."""
    close = EloMOV()
    blowout = EloMOV()

    delta_close, _ = close.update(1, 2, 14, 12)      # 2-point win
    delta_blowout, _ = blowout.update(1, 2, 40, 0)   # 40-point win

    assert abs(delta_blowout) > abs(delta_close)


def test_update_is_zero_sum() -> None:
    elo = EloMOV()
    elo.update(1, 2, 24, 12)
    total = sum(elo.ratings().values())
    assert math.isclose(total, 2 * DEFAULT_INITIAL_RATING, rel_tol=1e-9)


def test_blowout_vs_weak_team_discounted_vs_equal_opponents() -> None:
    """Autocorrelation correction: same margin, weak opponent → smaller delta."""
    # Strong vs weak opponent
    mismatch = EloMOV()
    mismatch._ratings[1] = 1800.0
    mismatch._ratings[2] = 1200.0
    delta_mismatch, _ = mismatch.update(1, 2, 40, 0)

    # Equal opponents
    equal = EloMOV()
    delta_equal, _ = equal.update(1, 2, 40, 0)

    # Home advantage boosts both, but mismatch should be smaller because
    # autocorr discounts when winner was already heavily favoured.
    assert abs(delta_mismatch) < abs(delta_equal)


def test_draw_uses_plain_k_not_zero() -> None:
    """A draw (margin=0) must not collapse to a zero update."""
    elo = EloMOV()
    # Home and away both at 1500, so expected_home ≈ 0.58 (home adv).
    # A draw (actual = 0.5) should move ratings by a non-trivial amount.
    delta_home, delta_away = elo.update(1, 2, 12, 12)
    assert abs(delta_home) > 0.0
    assert math.isclose(delta_home, -delta_away, rel_tol=1e-9)


def test_draw_home_rating_falls() -> None:
    """Home team loses rating on a draw (was expected to win due to HA)."""
    elo = EloMOV()
    elo.update(1, 2, 12, 12)
    assert elo.rating(1) < DEFAULT_INITIAL_RATING
    assert elo.rating(2) > DEFAULT_INITIAL_RATING


def test_underdog_win_still_swings_more_than_favourite_win() -> None:
    """Upset win should produce a larger away-team delta than a favourite win."""
    fav_wins = EloMOV()
    fav_wins._ratings[1] = 1700.0
    fav_wins._ratings[2] = 1300.0
    _, delta_away_fav_wins = fav_wins.update(1, 2, 24, 12)

    upset = EloMOV()
    upset._ratings[1] = 1700.0
    upset._ratings[2] = 1300.0
    _, delta_away_upset = upset.update(1, 2, 0, 24)

    assert abs(delta_away_upset) > abs(delta_away_fav_wins)


# ---------------------------------------------------------------------------
# Regression / interface parity with plain Elo
# ---------------------------------------------------------------------------


def test_regress_to_mean_works_same_as_elo() -> None:
    elo = EloMOV(season_regression=0.5)
    elo._ratings[1] = 1800.0
    elo._ratings[2] = 1200.0
    elo.regress_to_mean()
    assert elo.rating(1) == 1650.0
    assert elo.rating(2) == 1350.0


def test_regress_weight_out_of_range_raises() -> None:
    elo = EloMOV()
    with pytest.raises(ValueError):
        elo.regress_to_mean(weight=2.0)


# ---------------------------------------------------------------------------
# Predictor adapter
# ---------------------------------------------------------------------------


def test_elo_mov_predictor_name() -> None:
    from fantasy_coach.evaluation.predictors import EloMOVPredictor

    p = EloMOVPredictor()
    assert p.name == "elo_mov"


def test_elo_mov_predictor_fit_and_predict() -> None:
    from pathlib import Path

    from fantasy_coach.evaluation.predictors import EloMOVPredictor
    from fantasy_coach.storage import SQLiteRepository

    db_path = Path(__file__).parent / "fixtures" / "baseline-nrl.db"
    if not db_path.exists():
        pytest.skip("baseline DB missing")

    repo = SQLiteRepository(db_path)
    try:
        matches = repo.list_matches(2024)
    finally:
        repo.close()

    predictor = EloMOVPredictor()
    predictor.fit(matches[:50])
    m = matches[50]
    prob = predictor.predict_home_win_prob(m)
    assert 0.0 < prob < 1.0


def test_elo_mov_predictor_fit_empty_history() -> None:
    from fantasy_coach.evaluation.predictors import EloMOVPredictor

    predictor = EloMOVPredictor()
    predictor.fit([])
    # Should not raise; predict returns the home-advantage prior.
    assert 0.5 < predictor.elo.predict(1, 2) < 1.0
