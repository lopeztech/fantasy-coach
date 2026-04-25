"""Tests for representative.py — NRL calendar effects (Origin, Magic Round, Test windows).

Ground-truth fixture: 2024 Origin Round 2 is NRL Round 16.
"""

from __future__ import annotations

from datetime import date

from fantasy_coach.representative import (
    PlayerCallup,
    count_callups,
    is_magic_round,
    is_origin_round,
    is_test_window,
    origin_game_number,
)

# ---------------------------------------------------------------------------
# is_origin_round
# ---------------------------------------------------------------------------


def test_origin_round_2024_game1() -> None:
    assert is_origin_round(2024, 13) is True


def test_origin_round_2024_game2() -> None:
    assert is_origin_round(2024, 16) is True


def test_origin_round_2024_game3() -> None:
    assert is_origin_round(2024, 19) is True


def test_non_origin_round_returns_false() -> None:
    assert is_origin_round(2024, 1) is False
    assert is_origin_round(2024, 15) is False
    assert is_origin_round(2024, 25) is False


def test_origin_unknown_season_returns_false() -> None:
    assert is_origin_round(2023, 13) is False
    assert is_origin_round(2027, 13) is False


def test_origin_round_2025_all_three() -> None:
    assert is_origin_round(2025, 13) is True
    assert is_origin_round(2025, 16) is True
    assert is_origin_round(2025, 19) is True


def test_origin_round_2026_all_three() -> None:
    assert is_origin_round(2026, 13) is True
    assert is_origin_round(2026, 16) is True
    assert is_origin_round(2026, 19) is True


# ---------------------------------------------------------------------------
# origin_game_number
# ---------------------------------------------------------------------------


def test_origin_game_number_round_13_is_game1() -> None:
    assert origin_game_number(2024, 13) == 1


def test_origin_game_number_round_16_is_game2() -> None:
    assert origin_game_number(2024, 16) == 2


def test_origin_game_number_round_19_is_game3() -> None:
    assert origin_game_number(2024, 19) == 3


def test_origin_game_number_non_origin_round_returns_none() -> None:
    assert origin_game_number(2024, 12) is None
    assert origin_game_number(2024, 1) is None


def test_origin_game_number_unknown_season_returns_none() -> None:
    assert origin_game_number(2023, 13) is None


# ---------------------------------------------------------------------------
# is_magic_round
# ---------------------------------------------------------------------------


def test_magic_round_2024() -> None:
    assert is_magic_round(2024, 10) is True


def test_magic_round_2025() -> None:
    assert is_magic_round(2025, 11) is True


def test_magic_round_2026() -> None:
    assert is_magic_round(2026, 10) is True


def test_non_magic_round_returns_false() -> None:
    assert is_magic_round(2024, 11) is False
    assert is_magic_round(2024, 1) is False


def test_magic_round_unknown_season_returns_false() -> None:
    assert is_magic_round(2023, 10) is False
    assert is_magic_round(2027, 10) is False


# ---------------------------------------------------------------------------
# is_test_window
# ---------------------------------------------------------------------------


def test_test_window_pacific_champs_2024() -> None:
    assert is_test_window(date(2024, 10, 15)) is True
    assert is_test_window(date(2024, 10, 11)) is True  # boundary inclusive
    assert is_test_window(date(2024, 11, 3)) is True  # boundary inclusive


def test_test_window_outside_pacific_champs_2024() -> None:
    assert is_test_window(date(2024, 10, 10)) is False  # day before
    assert is_test_window(date(2024, 11, 4)) is False  # day after


def test_test_window_regular_nrl_rounds_not_flagged() -> None:
    assert is_test_window(date(2024, 3, 7)) is False
    assert is_test_window(date(2024, 8, 1)) is False


def test_test_window_unknown_season_returns_false() -> None:
    assert is_test_window(date(2023, 10, 15)) is False
    assert is_test_window(date(2027, 10, 15)) is False


def test_test_window_2025() -> None:
    assert is_test_window(date(2025, 10, 20)) is True


def test_test_window_2026() -> None:
    assert is_test_window(date(2026, 10, 20)) is True


# ---------------------------------------------------------------------------
# count_callups
# ---------------------------------------------------------------------------


def _make_callups(player_ids: list[int], fixture: str = "origin2") -> list[PlayerCallup]:
    return [
        PlayerCallup(player_id=pid, team_id=100 + pid, fixture=fixture, state="nsw")
        for pid in player_ids
    ]


def test_count_callups_exact_overlap() -> None:
    callups = _make_callups([1, 2, 3])
    assert count_callups([1, 2, 3], callups) == 3.0


def test_count_callups_partial_overlap() -> None:
    callups = _make_callups([1, 2, 10, 11])
    assert count_callups([1, 2, 3, 4], callups) == 2.0


def test_count_callups_no_overlap() -> None:
    callups = _make_callups([10, 11, 12])
    assert count_callups([1, 2, 3], callups) == 0.0


def test_count_callups_empty_player_list() -> None:
    callups = _make_callups([1, 2, 3])
    assert count_callups([], callups) == 0.0


def test_count_callups_empty_callups() -> None:
    assert count_callups([1, 2, 3], []) == 0.0


def test_count_callups_with_position_weights_same_as_unweighted_without_positions() -> None:
    # count_callups with position_weights still uses 1.0 per player (no positional data in callup)
    callups = _make_callups([1, 2, 3])
    unweighted = count_callups([1, 2, 3], callups)
    weighted = count_callups([1, 2, 3], callups, position_weights={"halves": 1.5})
    assert unweighted == weighted  # same, since callups don't carry position info


def test_count_callups_returns_float() -> None:
    callups = _make_callups([1, 2])
    result = count_callups([1, 2, 3], callups)
    assert isinstance(result, float)
