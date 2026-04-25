"""Tests for representative.py — NRL calendar detection and callup counting.

Uses 2024 Origin Round 2 (round 17) as ground truth.  Penrith were missing
6 players that round according to published Origin squads; that historical
example anchors the callup-counting logic.
"""

from __future__ import annotations

from datetime import date, datetime, UTC
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fantasy_coach.representative import (
    Callup,
    count_callups_for_team,
    is_magic_round,
    is_origin_round,
    is_test_window,
    origin_fixture_label,
    store_callups,
)


# ---------------------------------------------------------------------------
# is_origin_round
# ---------------------------------------------------------------------------


class TestIsOriginRound:
    def test_2024_origin_rounds(self) -> None:
        assert is_origin_round(2024, 13)
        assert is_origin_round(2024, 17)
        assert is_origin_round(2024, 21)

    def test_2025_origin_rounds(self) -> None:
        assert is_origin_round(2025, 13)
        assert is_origin_round(2025, 17)
        assert is_origin_round(2025, 21)

    def test_2026_origin_rounds(self) -> None:
        assert is_origin_round(2026, 13)
        assert is_origin_round(2026, 17)
        assert is_origin_round(2026, 21)

    def test_non_origin_round(self) -> None:
        assert not is_origin_round(2024, 1)
        assert not is_origin_round(2024, 12)
        assert not is_origin_round(2024, 14)
        assert not is_origin_round(2024, 25)

    def test_unknown_season_returns_false(self) -> None:
        assert not is_origin_round(2019, 13)

    def test_ground_truth_2024_round17(self) -> None:
        """2024 Origin Round 2: Penrith had 6 representative callups."""
        assert is_origin_round(2024, 17)
        label = origin_fixture_label(2024, 17)
        assert label == "origin2"


# ---------------------------------------------------------------------------
# origin_fixture_label
# ---------------------------------------------------------------------------


class TestOriginFixtureLabel:
    def test_correct_labels(self) -> None:
        assert origin_fixture_label(2024, 13) == "origin1"
        assert origin_fixture_label(2024, 17) == "origin2"
        assert origin_fixture_label(2024, 21) == "origin3"

    def test_non_origin_returns_none(self) -> None:
        assert origin_fixture_label(2024, 5) is None
        assert origin_fixture_label(2024, 25) is None

    def test_unknown_season_returns_none(self) -> None:
        assert origin_fixture_label(2019, 13) is None


# ---------------------------------------------------------------------------
# is_magic_round
# ---------------------------------------------------------------------------


class TestIsMagicRound:
    def test_magic_rounds(self) -> None:
        assert is_magic_round(2024, 12)
        assert is_magic_round(2025, 11)
        assert is_magic_round(2026, 12)

    def test_non_magic_rounds(self) -> None:
        assert not is_magic_round(2024, 1)
        assert not is_magic_round(2024, 13)
        assert not is_magic_round(2026, 11)

    def test_unknown_season_returns_false(self) -> None:
        assert not is_magic_round(2019, 12)

    def test_consecutive_rounds_around_magic(self) -> None:
        assert not is_magic_round(2024, 11)
        assert is_magic_round(2024, 12)
        assert not is_magic_round(2024, 13)


# ---------------------------------------------------------------------------
# is_test_window
# ---------------------------------------------------------------------------


class TestIsTestWindow:
    def test_2024_pacific_championships(self) -> None:
        assert is_test_window(date(2024, 10, 25))
        assert is_test_window(date(2024, 11, 1))
        assert is_test_window(date(2024, 11, 10))

    def test_before_2024_window(self) -> None:
        assert not is_test_window(date(2024, 10, 24))

    def test_after_2024_window(self) -> None:
        assert not is_test_window(date(2024, 11, 11))

    def test_regular_season_not_a_test_window(self) -> None:
        assert not is_test_window(date(2024, 3, 1))
        assert not is_test_window(date(2024, 7, 15))

    def test_2025_window(self) -> None:
        assert is_test_window(date(2025, 10, 30))

    def test_2026_window_estimated(self) -> None:
        assert is_test_window(date(2026, 10, 30))


# ---------------------------------------------------------------------------
# count_callups_for_team
# ---------------------------------------------------------------------------


def _make_player(player_id: int, position: str):
    from fantasy_coach.features import PlayerRow

    return PlayerRow(
        player_id=player_id,
        jersey_number=player_id,
        position=position,
        first_name="Test",
        last_name="Player",
        is_on_field=True,
    )


def _make_callup(player_id: int, fixture: str = "origin2") -> Callup:
    return Callup(
        player_id=player_id,
        fixture=fixture,  # type: ignore[arg-type]
        fixture_date=date(2024, 6, 26),
        season=2024,
        round=17,
        state="NSW",
    )


class TestCountCallupsForTeam:
    def test_zero_when_no_callups(self) -> None:
        players = [_make_player(1, "Halfback"), _make_player(2, "Prop")]
        total = count_callups_for_team(10, [], players)
        assert total == 0.0

    def test_counts_matching_players(self) -> None:
        players = [_make_player(1, "Halfback"), _make_player(2, "Prop"), _make_player(3, "Centre")]
        callups = [_make_callup(1), _make_callup(2)]  # player 3 not called up
        total = count_callups_for_team(10, callups, players)
        assert total == pytest.approx(2.0)

    def test_position_weighted(self) -> None:
        weights = {"Halfback": 3.0, "Prop": 1.0, "Centre": 1.5}
        players = [_make_player(1, "Halfback"), _make_player(2, "Prop")]
        callups = [_make_callup(1), _make_callup(2)]
        total = count_callups_for_team(10, callups, players, position_weights=weights)
        assert total == pytest.approx(4.0)  # 3.0 + 1.0

    def test_player_not_in_match_roster_not_counted(self) -> None:
        # Callup for player 99, but they're not in the team's match_players list.
        players = [_make_player(1, "Halfback")]
        callups = [_make_callup(99)]
        total = count_callups_for_team(10, callups, players)
        assert total == 0.0

    def test_ground_truth_penrith_2024_origin2(self) -> None:
        """Penrith had 6 Origin callups in 2024 round 17 (Origin 2 week).
        Flat count (no position weights) should equal 6 when the callups and
        roster are fed in correctly.
        """
        # Ground truth: 6 Penrith players in the NSW squad for Origin 2 2024.
        penrith_callup_ids = {101, 102, 103, 104, 105, 106}
        players = [_make_player(pid, "Various") for pid in penrith_callup_ids]
        callups = [_make_callup(pid) for pid in penrith_callup_ids]
        total = count_callups_for_team(400045, callups, players)
        assert total == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# store_callups
# ---------------------------------------------------------------------------


class TestStoreCallups:
    def test_calls_record_callup_for_each(self) -> None:
        repo = MagicMock()
        callups = [_make_callup(1), _make_callup(2), _make_callup(3)]
        stored = store_callups(callups, repo)
        assert stored == 3
        assert repo.record_callup.call_count == 3

    def test_swallows_repo_errors_and_counts_successes(self) -> None:
        repo = MagicMock()
        # First call succeeds, second fails, third succeeds.
        repo.record_callup.side_effect = [None, RuntimeError("db error"), None]
        callups = [_make_callup(1), _make_callup(2), _make_callup(3)]
        stored = store_callups(callups, repo)
        assert stored == 2  # only successful inserts counted

    def test_empty_list_returns_zero(self) -> None:
        repo = MagicMock()
        assert store_callups([], repo) == 0
        repo.record_callup.assert_not_called()


# ---------------------------------------------------------------------------
# SQLite callup storage round-trip
# ---------------------------------------------------------------------------


def test_sqlite_repository_callup_round_trip(tmp_path: Path) -> None:
    from fantasy_coach.storage.sqlite import SQLiteRepository

    repo = SQLiteRepository(tmp_path / "test.db")

    # Insert three callups for two seasons/rounds.
    callups_r17 = [_make_callup(101), _make_callup(102)]
    callups_r21 = [_make_callup(103)]

    for c in callups_r17:
        callup = Callup(
            player_id=c.player_id,
            fixture="origin2",
            fixture_date=date(2024, 6, 26),
            season=2024,
            round=17,
            state="NSW",
        )
        repo.record_callup(callup)

    for c in callups_r21:
        callup = Callup(
            player_id=c.player_id,
            fixture="origin3",
            fixture_date=date(2024, 8, 7),
            season=2024,
            round=21,
            state="QLD",
        )
        repo.record_callup(callup)

    # Query round 17 — should return 2.
    result_17 = repo.list_callups(season=2024, round_=17)
    assert len(result_17) == 2
    assert {c.player_id for c in result_17} == {101, 102}

    # Query round 21 — should return 1.
    result_21 = repo.list_callups(season=2024, round_=21)
    assert len(result_21) == 1
    assert result_21[0].player_id == 103

    # Query by fixture filter.
    filtered = repo.list_callups(season=2024, round_=17, fixture="origin2")
    assert len(filtered) == 2


def test_sqlite_callup_duplicate_is_ignored(tmp_path: Path) -> None:
    from fantasy_coach.storage.sqlite import SQLiteRepository

    repo = SQLiteRepository(tmp_path / "test.db")
    callup = Callup(
        player_id=999,
        fixture="origin1",
        fixture_date=date(2024, 5, 22),
        season=2024,
        round=13,
        state="NSW",
    )
    repo.record_callup(callup)
    repo.record_callup(callup)  # idempotent — should not raise

    result = repo.list_callups(season=2024, round_=13)
    assert len(result) == 1
