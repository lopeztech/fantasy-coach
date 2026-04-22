"""Tests for the ``python -m fantasy_coach precompute`` subcommand.

The precompute Job (issue #65) runs this CLI twice a week. These tests
cover the two entry paths — explicit ``--season/--round`` and autodetect —
plus the ``--no-force`` opt-out. No real NRL scrapes, no real Firestore.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fantasy_coach.__main__ import _detect_upcoming_round, main


def test_detect_upcoming_round_returns_first_round_with_upcoming_fixture() -> None:
    """Iterates round 1..N and stops at the first round with Upcoming/Pre state."""
    payloads = {
        1: {"fixtures": [{"matchState": "FullTime"}, {"matchState": "FullTime"}]},
        2: {"fixtures": [{"matchState": "FullTime"}]},
        3: {"fixtures": [{"matchState": "Upcoming"}, {"matchState": "FullTime"}]},
        4: {"fixtures": [{"matchState": "Upcoming"}]},
    }
    calls: list[int] = []

    def _fake_fetch_round(year: int, round_: int, **_kw):
        calls.append(round_)
        return payloads.get(round_)

    with patch("fantasy_coach.scraper.fetch_round", side_effect=_fake_fetch_round):
        result = _detect_upcoming_round(2026)

    assert result == 3
    # Must have stopped at the first match — no extra round-4 request.
    assert calls == [1, 2, 3]


def test_detect_upcoming_round_accepts_pre_state_too() -> None:
    """``Pre`` means 'about to kick off' — treat it as upcoming."""
    payloads = {1: {"fixtures": [{"matchState": "Pre"}]}}
    with patch(
        "fantasy_coach.scraper.fetch_round",
        side_effect=lambda y, r, **kw: payloads.get(r),
    ):
        assert _detect_upcoming_round(2026) == 1


def test_detect_upcoming_round_returns_none_when_season_is_over() -> None:
    """Every round fully played → no upcoming round to precompute."""
    payloads = {r: {"fixtures": [{"matchState": "FullTime"}]} for r in range(1, 31)}
    with patch(
        "fantasy_coach.scraper.fetch_round",
        side_effect=lambda y, r, **kw: payloads.get(r),
    ):
        assert _detect_upcoming_round(2026) is None


def test_detect_upcoming_round_skips_missing_rounds() -> None:
    """``fetch_round`` returning None (404) means that round doesn't exist;
    iteration should keep going, not bail."""
    payloads = {
        1: None,
        2: None,
        5: {"fixtures": [{"matchState": "Upcoming"}]},
    }
    with patch(
        "fantasy_coach.scraper.fetch_round",
        side_effect=lambda y, r, **kw: payloads.get(r),
    ):
        assert _detect_upcoming_round(2026) == 5


@pytest.fixture
def _mocked_environment(monkeypatch):
    """Stub out the three IO sinks precompute touches: repo, store, and
    ``compute_predictions`` itself. Returns the mocks for per-test assertions."""
    fake_repo = MagicMock()
    fake_store = MagicMock()
    fake_compute = MagicMock(return_value=[MagicMock(), MagicMock()])  # 2 predictions
    monkeypatch.setattr("fantasy_coach.config.get_repository", lambda: fake_repo)
    monkeypatch.setattr("fantasy_coach.predictions.get_prediction_store", lambda: fake_store)
    monkeypatch.setattr("fantasy_coach.predictions.compute_predictions", fake_compute)
    return fake_repo, fake_store, fake_compute


def test_precompute_with_explicit_season_and_round(_mocked_environment) -> None:
    _, _, fake_compute = _mocked_environment
    rc = main(["precompute", "--season", "2026", "--round", "8"])
    assert rc == 0
    fake_compute.assert_called_once()
    # Positional args: (season, round, repo, store)
    args, kwargs = fake_compute.call_args
    assert args[0] == 2026
    assert args[1] == 8
    assert kwargs.get("force") is True  # default


def test_precompute_no_force_passes_force_false(_mocked_environment) -> None:
    _, _, fake_compute = _mocked_environment
    rc = main(["precompute", "--season", "2026", "--round", "8", "--no-force"])
    assert rc == 0
    _, kwargs = fake_compute.call_args
    assert kwargs.get("force") is False


def test_precompute_autodetects_round_when_omitted(_mocked_environment) -> None:
    _, _, fake_compute = _mocked_environment
    with patch("fantasy_coach.__main__._detect_upcoming_round", return_value=12):
        rc = main(["precompute", "--season", "2026"])
    assert rc == 0
    args, _ = fake_compute.call_args
    assert args[0] == 2026
    assert args[1] == 12


def test_precompute_noop_when_no_upcoming_round(_mocked_environment) -> None:
    _, _, fake_compute = _mocked_environment
    with patch("fantasy_coach.__main__._detect_upcoming_round", return_value=None):
        rc = main(["precompute", "--season", "2026"])
    assert rc == 0
    fake_compute.assert_not_called()
