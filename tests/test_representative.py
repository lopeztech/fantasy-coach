"""Tests for representative.py — NRL calendar detection and callup features.

Ground-truth fixture: 2024 Origin Round 2 (R13). Penrith lost to a
non-Origin-affected club that round with 6 players in camp.
"""

from __future__ import annotations

import pytest

from fantasy_coach.representative import (
    OriginSquad,
    extract_representative_features,
    fetch_origin_squads,
    is_magic_round,
    is_origin_round,
    is_test_window,
    origin_callups_diff,
)


# ---------------------------------------------------------------------------
# Calendar detection — is_origin_round
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "season,round_,expected",
    [
        # 2024 Origin rounds: 11, 13, 16
        (2024, 11, True),
        (2024, 13, True),
        (2024, 16, True),
        (2024, 12, False),
        # 2025 Origin rounds: 12, 15, 18
        (2025, 12, True),
        (2025, 15, True),
        (2025, 18, True),
        (2025, 11, False),
        # 2026 provisional
        (2026, 13, True),
        (2026, 16, True),
        (2026, 19, True),
        (2026, 14, False),
        # Unknown season → False
        (2020, 13, False),
    ],
)
def test_is_origin_round(season: int, round_: int, expected: bool) -> None:
    assert is_origin_round(season, round_) is expected


# ---------------------------------------------------------------------------
# Calendar detection — is_magic_round
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "season,round_,expected",
    [
        (2024, 11, True),
        (2024, 10, False),
        (2025, 11, True),
        (2026, 11, True),
        (2023, 11, False),  # not in schedule
    ],
)
def test_is_magic_round(season: int, round_: int, expected: bool) -> None:
    assert is_magic_round(season, round_) is expected


# ---------------------------------------------------------------------------
# Calendar detection — is_test_window
# ---------------------------------------------------------------------------


def test_is_test_window_returns_true_for_known_rounds() -> None:
    assert is_test_window(2024, 27) is True
    assert is_test_window(2024, 28) is True


def test_is_test_window_returns_false_for_regular_rounds() -> None:
    assert is_test_window(2024, 10) is False
    assert is_test_window(2026, 5) is False


def test_is_test_window_unknown_season_returns_false() -> None:
    assert is_test_window(2019, 27) is False


# ---------------------------------------------------------------------------
# origin_callups_diff
# ---------------------------------------------------------------------------


def _player(pid: int, pos: str):
    """Minimal mock of PlayerRow for callup tests."""
    from unittest.mock import MagicMock

    p = MagicMock()
    p.player_id = pid
    p.position = pos
    return p


def test_origin_callups_diff_empty_callup_set_returns_zero() -> None:
    home = [_player(1, "Halfback"), _player(2, "Prop")]
    away = [_player(3, "Fullback")]
    assert origin_callups_diff(home, away, frozenset()) == 0.0


def test_origin_callups_diff_home_advantage_positive() -> None:
    # Home halfback (weight 3.0) is called up; away has no callups.
    home = [_player(1, "Halfback"), _player(2, "Prop")]
    away = [_player(3, "Fullback")]
    called_up = frozenset([1])  # player 1 = home halfback
    diff = origin_callups_diff(home, away, called_up)
    assert diff == pytest.approx(3.0)  # home has 3.0 weight, away 0


def test_origin_callups_diff_away_callup_negative() -> None:
    home = [_player(1, "Prop")]
    away = [_player(2, "Halfback")]  # away halfback called up
    called_up = frozenset([2])
    diff = origin_callups_diff(home, away, called_up)
    assert diff == pytest.approx(-3.0)


def test_origin_callups_diff_symmetric_returns_zero() -> None:
    # Home halfback (3.0) and away halfback (3.0) both called up.
    home = [_player(1, "Halfback")]
    away = [_player(2, "Halfback")]
    called_up = frozenset([1, 2])
    diff = origin_callups_diff(home, away, called_up)
    assert diff == pytest.approx(0.0)


def test_origin_callups_diff_unknown_position_uses_default_weight() -> None:
    home = [_player(1, "Interchange")]  # not in POSITION_WEIGHTS
    away = []
    called_up = frozenset([1])
    diff = origin_callups_diff(home, away, called_up)
    assert diff == pytest.approx(1.0)  # default weight


# ---------------------------------------------------------------------------
# extract_representative_features
# ---------------------------------------------------------------------------


def test_extract_representative_features_origin_round() -> None:
    feats = extract_representative_features(2024, 13, [], [])
    assert feats["is_origin_round"] == 1.0
    assert feats["is_magic_round"] == 0.0
    assert isinstance(feats["origin_callups_diff"], float)


def test_extract_representative_features_magic_round() -> None:
    feats = extract_representative_features(2024, 11, [], [])
    # Round 11 in 2024 is BOTH an Origin round and Magic Round.
    assert feats["is_origin_round"] == 1.0
    assert feats["is_magic_round"] == 1.0


def test_extract_representative_features_non_special_round() -> None:
    feats = extract_representative_features(2024, 5, [], [])
    assert feats["is_origin_round"] == 0.0
    assert feats["is_magic_round"] == 0.0
    assert feats["origin_callups_diff"] == 0.0


def test_extract_representative_features_with_squad() -> None:
    home = [_player(10, "Halfback"), _player(11, "Prop")]
    away = [_player(20, "Fullback")]
    squad = OriginSquad(
        year=2024, game=2, nsw=frozenset([10]), qld=frozenset([20])
    )
    feats = extract_representative_features(2024, 13, home, away, origin_squad=squad)
    # home halfback (3.0) called up; away fullback (2.0) called up → diff = 1.0
    assert feats["origin_callups_diff"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# fetch_origin_squads — stub returns None
# ---------------------------------------------------------------------------


def test_fetch_origin_squads_stub_returns_none() -> None:
    result = fetch_origin_squads(2024, 2)
    assert result is None
