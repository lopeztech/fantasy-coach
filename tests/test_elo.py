from __future__ import annotations

import math

import pytest

from fantasy_coach.models.elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_INITIAL_RATING,
    DEFAULT_K,
    Elo,
)


def test_new_team_starts_at_initial_rating() -> None:
    elo = Elo()
    assert elo.rating(1) == DEFAULT_INITIAL_RATING


def test_predict_with_equal_ratings_reflects_home_advantage() -> None:
    elo = Elo()
    p_home = elo.predict(1, 2)
    # Pure 0 home advantage would give 0.5; any positive HA pushes >0.5.
    assert p_home > 0.5
    expected = 1.0 / (1.0 + 10.0 ** (-DEFAULT_HOME_ADVANTAGE / 400.0))
    assert math.isclose(p_home, expected, rel_tol=1e-9)


def test_update_arithmetic_for_equal_ratings_home_win() -> None:
    elo = Elo()
    expected_home = elo.predict(1, 2)
    delta_home, delta_away = elo.update(1, 2, 30, 10)

    expected_delta = DEFAULT_K * (1.0 - expected_home)
    assert math.isclose(delta_home, expected_delta, rel_tol=1e-9)
    assert math.isclose(delta_away, -expected_delta, rel_tol=1e-9)


def test_update_is_zero_sum() -> None:
    elo = Elo()
    elo.update(1, 2, 24, 18)
    total = sum(elo.ratings().values())
    assert math.isclose(total, 2 * DEFAULT_INITIAL_RATING, rel_tol=1e-9)


def test_draw_pulls_higher_rated_team_down() -> None:
    elo = Elo()
    elo._ratings[1] = 1700.0
    elo._ratings[2] = 1300.0

    elo.update(1, 2, 12, 12)

    assert elo.rating(1) < 1700.0
    assert elo.rating(2) > 1300.0


def test_underdog_win_swings_more_than_favorite_win() -> None:
    fav_wins = Elo()
    fav_wins._ratings[1] = 1700.0
    fav_wins._ratings[2] = 1300.0
    delta_fav, _ = fav_wins.update(1, 2, 30, 0)

    upset = Elo()
    upset._ratings[1] = 1700.0
    upset._ratings[2] = 1300.0
    _, delta_underdog = upset.update(1, 2, 0, 30)

    assert abs(delta_underdog) > abs(delta_fav)


def test_home_advantage_actually_affects_predictions() -> None:
    no_ha = Elo(home_advantage=0.0)
    with_ha = Elo(home_advantage=100.0)
    assert with_ha.predict(1, 2) > no_ha.predict(1, 2)


def test_repeated_home_wins_inflate_home_rating() -> None:
    elo = Elo()
    for _ in range(50):
        elo.update(1, 2, 24, 12)
    assert elo.rating(1) > elo.rating(2) + 200


def test_regress_to_mean_pulls_ratings_toward_initial() -> None:
    elo = Elo(season_regression=0.5)
    elo._ratings[1] = 1800.0
    elo._ratings[2] = 1200.0

    elo.regress_to_mean()

    assert elo.rating(1) == 1650.0  # 1800 - 0.5 * (1800 - 1500)
    assert elo.rating(2) == 1350.0


def test_regression_weight_zero_is_noop() -> None:
    elo = Elo()
    elo._ratings[1] = 1800.0
    elo.regress_to_mean(weight=0.0)
    assert elo.rating(1) == 1800.0


def test_regression_weight_one_resets() -> None:
    elo = Elo()
    elo._ratings[1] = 1800.0
    elo.regress_to_mean(weight=1.0)
    assert elo.rating(1) == DEFAULT_INITIAL_RATING


def test_regression_weight_out_of_range_raises() -> None:
    elo = Elo()
    with pytest.raises(ValueError):
        elo.regress_to_mean(weight=1.5)


def test_predict_probabilities_sum_to_one_when_swapped() -> None:
    elo = Elo(home_advantage=0.0)  # symmetric, so a swap is the complement
    elo._ratings[1] = 1600.0
    elo._ratings[2] = 1400.0
    p1 = elo.predict(1, 2)
    p2 = elo.predict(2, 1)
    assert math.isclose(p1 + p2, 1.0, rel_tol=1e-9)
