"""Tests for ``models.player_ratings.PlayerRatings``."""

from __future__ import annotations

import pytest

from fantasy_coach.models.player_ratings import PlayerRatings


def _xiii(base: int, position: str = "Halfback") -> list[tuple[int, str, bool]]:
    """Return a starting XIII of 13 players with ids starting at ``base``."""
    return [(base + i, position, True) for i in range(13)]


def _bench(base: int, position: str = "Interchange") -> list[tuple[int, str, bool]]:
    return [(base + i, position, False) for i in range(4)]


def test_new_player_starts_at_default_rating() -> None:
    book = PlayerRatings()
    assert book.rating(999) == PlayerRatings.initial_rating


def test_winning_starters_gain_rating_losers_lose() -> None:
    book = PlayerRatings()
    home = _xiii(100)
    away = _xiii(200)
    # Both teams start at default → expected_home = 0.5. Home wins.
    book.update(home, away, home_score=30, away_score=10)

    assert book.rating(100) > book.initial_rating
    assert book.rating(200) < book.initial_rating
    # Ratings are zero-sum within a match: the home delta equals the
    # negative away delta (same K, same (actual-expected) magnitude).
    home_delta = book.rating(100) - book.initial_rating
    away_delta = book.initial_rating - book.rating(200)
    assert home_delta == pytest.approx(away_delta)


def test_bench_update_is_half_weight_of_starter() -> None:
    book = PlayerRatings()
    home = _xiii(100) + [(113, "Interchange", False)]
    away = _xiii(200)
    book.update(home, away, home_score=30, away_score=10)

    starter_delta = book.rating(100) - book.initial_rating
    bench_delta = book.rating(113) - book.initial_rating
    # default K_starter=20, K_bench=10 → bench update is half the starter's.
    assert bench_delta == pytest.approx(starter_delta / 2)


def test_upset_moves_more_than_expected_win() -> None:
    """A lower-rated team winning produces a bigger delta than the
    same-rated team winning — standard Elo property."""
    strong = PlayerRatings()
    # Artificially seed home team well above the default.
    for pid in range(100, 113):
        strong._ratings[pid] = 1700.0  # noqa: SLF001 — test boundary ok

    # Case A: strong team wins as expected.
    baseline_before = strong.rating(100)
    strong.update(_xiii(100), _xiii(200), home_score=30, away_score=10)
    expected_delta = strong.rating(100) - baseline_before

    # Case B: fresh book where weak team wins the upset.
    upset = PlayerRatings()
    for pid in range(200, 213):
        upset._ratings[pid] = 1700.0  # noqa: SLF001
    upset_before = upset.rating(100)
    upset.update(_xiii(100), _xiii(200), home_score=30, away_score=10)  # home (weak) wins
    upset_delta = upset.rating(100) - upset_before

    assert upset_delta > expected_delta


def test_composite_sums_rating_times_position_weight() -> None:
    book = PlayerRatings()
    # Two custom ratings so the composite has asymmetry.
    book._ratings[7] = 1700.0  # star halfback  # noqa: SLF001
    book._ratings[1] = 1600.0  # solid fullback
    starters = [(7, "Halfback"), (1, "Fullback")]
    weights = {"Halfback": 3.0, "Fullback": 2.5}

    composite = book.composite(starters, position_weights=weights, bench_weight=0.0)
    # 3.0 * 1700 + 2.5 * 1600 = 5100 + 4000 = 9100
    assert composite == pytest.approx(9100.0)


def test_composite_includes_bench_at_fractional_weight() -> None:
    book = PlayerRatings()
    starters = [(7, "Halfback")]
    bench_xi = [(14, "Interchange")]
    weights = {"Halfback": 3.0, "Interchange": 0.5}

    composite = book.composite(starters, bench_xi, position_weights=weights, bench_weight=0.3)
    # 3.0 * 1500 (default) + 0.3 * 0.5 * 1500 = 4500 + 225 = 4725
    assert composite == pytest.approx(4725.0)


def test_position_returns_most_common_observed_slot() -> None:
    book = PlayerRatings()
    # Player plays halfback twice, five-eighth once.
    book.update([(7, "Halfback", True)], [(200, "Halfback", True)], 10, 20)
    book.update([(7, "Halfback", True)], [(200, "Halfback", True)], 20, 10)
    book.update([(7, "Five-Eighth", True)], [(200, "Halfback", True)], 10, 20)

    assert book.position(7) == "Halfback"
    assert book.position(999) is None  # unseen player


def test_update_skips_when_is_on_field_unknown() -> None:
    book = PlayerRatings()
    home = [(100, "Halfback", None)]  # is_on_field unknown
    away = [(200, "Halfback", True)]
    book.update(home, away, 30, 10)
    # Nothing written to the rating book — both are still at default.
    assert book.rating(100) == book.initial_rating
    assert book.rating(200) == book.initial_rating


def test_regress_to_mean_pulls_ratings_toward_default() -> None:
    book = PlayerRatings(season_regression=0.25)
    book._ratings[7] = 1700.0  # noqa: SLF001
    book._ratings[14] = 1300.0  # noqa: SLF001
    book.regress_to_mean()
    # 1700 + 0.25*(1500-1700) = 1650
    # 1300 + 0.25*(1500-1300) = 1350
    assert book.rating(7) == pytest.approx(1650.0)
    assert book.rating(14) == pytest.approx(1350.0)


def test_regress_to_mean_rejects_invalid_weight() -> None:
    book = PlayerRatings()
    with pytest.raises(ValueError):
        book.regress_to_mean(weight=1.5)
