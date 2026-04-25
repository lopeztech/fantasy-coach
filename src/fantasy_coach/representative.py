"""NRL calendar effects: State of Origin, Magic Round, and Test windows.

These structural calendar moments cause predictable roster disruptions:

- **State of Origin** (mid-year) — up to 34 NRL stars unavailable for the
  club round following each Origin game (and sometimes managed the round
  before). The model sees these as ``is_origin_round`` (categorical 0/1) and
  ``origin_callups_diff`` (position-weighted count of Origin-named players on
  each roster, home minus away).

- **Magic Round** — all 16 teams play at a single neutral venue (currently
  Suncorp Stadium, Brisbane) over a single weekend. Already partially captured
  by ``is_neutral_venue`` but the logistical bunching affects results beyond
  the venue effect. ``is_magic_round`` is a separate 0/1 flag.

- **International / Pacific Championships windows** — late-season Tests pull
  representative-country players out of club finals warm-up matches.
  ``is_test_window`` flags those rounds; ``is_test_window_diff`` counts
  Test-bound players (home minus away) when squad data is available.

Calendar data is hard-coded for 2024–2026 (the seasons with training data).
Beyond 2026, production code should load from a config file or the NRL API.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Hard-coded Origin round schedule (NRL club rounds affected by Origin)
# ---------------------------------------------------------------------------
# "Affected" means Origin players travel to camp / manage the week of each
# Origin game. This is the NRL club round that overlaps the Origin week.
# Data sourced from official NRL Origin scheduling.
#
# Format: {season: {game_number: nrl_club_round}}
#
# Note: "Origin round" sometimes includes the round immediately after a game
# (managed players), but here we flag only the primary overlap round.

_ORIGIN_ROUNDS: dict[int, dict[int, int]] = {
    2024: {
        1: 13,  # Origin I — NRL Round 13, May 22 week
        2: 16,  # Origin II — NRL Round 16, June 12 week
        3: 19,  # Origin III — NRL Round 19, July 10 week
    },
    2025: {
        1: 13,  # Origin I — NRL Round 13, May 21 week
        2: 16,  # Origin II — NRL Round 16, June 11 week
        3: 19,  # Origin III — NRL Round 19, July 9 week
    },
    2026: {
        1: 13,  # Origin I — NRL Round 13, May 20 week
        2: 16,  # Origin II — NRL Round 16, June 10 week
        3: 19,  # Origin III — NRL Round 19, July 8 week
    },
}

# ---------------------------------------------------------------------------
# Magic Round schedule
# ---------------------------------------------------------------------------
# Magic Round is the single-weekend neutral-venue showpiece, played at Suncorp
# Stadium Brisbane. All 16 teams play there that weekend — all matches are
# neutral-venue for both clubs (the existing ``is_neutral_venue`` feature
# captures the venue effect; ``is_magic_round`` adds the logistical-bunching
# signal on top).

_MAGIC_ROUNDS: dict[int, int] = {
    2024: 10,  # Magic Round 2024 — Round 10, May 10-12, Suncorp
    2025: 11,  # Magic Round 2025 — Round 11, May 16-18, Suncorp
    2026: 10,  # Magic Round 2026 — Round 10, May 9-11, Suncorp
}

# ---------------------------------------------------------------------------
# Test window schedule (Pacific Championships + Ashes equiv)
# ---------------------------------------------------------------------------
# Format: {season: [(start_date, end_date), ...]}
# Rounds within these date windows have representative-country players absent.

_TEST_WINDOWS: dict[int, list[tuple[date, date]]] = {
    2024: [
        (date(2024, 10, 11), date(2024, 11, 3)),  # Pacific Championships
    ],
    2025: [
        (date(2025, 10, 10), date(2025, 11, 2)),  # Pacific Championships
    ],
    2026: [
        (date(2026, 10, 9), date(2026, 11, 1)),  # Pacific Championships (est)
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_origin_round(season: int, round_: int) -> bool:
    """Return True when NRL round ``round_`` in ``season`` coincides with an
    Origin game week and rosters are materially affected.

    Based on the hard-coded ``_ORIGIN_ROUNDS`` calendar; returns False for
    seasons not listed (2023 and earlier, 2027+).
    """
    season_rounds = _ORIGIN_ROUNDS.get(season, {})
    return round_ in season_rounds.values()


def is_magic_round(season: int, round_: int) -> bool:
    """Return True when this is the Magic Round (neutral-venue showpiece)."""
    return _MAGIC_ROUNDS.get(season) == round_


def is_test_window(match_date: date) -> bool:
    """Return True when ``match_date`` falls within an international Test window.

    Checks all configured seasons. Returns False for seasons not listed.
    """
    year = match_date.year
    return any(start <= match_date <= end for start, end in _TEST_WINDOWS.get(year, []))


def origin_game_number(season: int, round_: int) -> int | None:
    """Return the Origin game number (1, 2, or 3) if this is an Origin round,
    or None otherwise.
    """
    season_rounds = _ORIGIN_ROUNDS.get(season, {})
    for game_num, affected_round in season_rounds.items():
        if affected_round == round_:
            return game_num
    return None


# ---------------------------------------------------------------------------
# Callup data types
# ---------------------------------------------------------------------------


class PlayerCallup(NamedTuple):
    """One player's representative selection for a fixture window."""

    player_id: int
    team_id: int  # NRL club team_id
    fixture: str  # "origin1", "origin2", "origin3", "test_au", "test_nz", "test_pac"
    state: str | None  # "nsw" | "qld" | None (for non-Origin callups)


def count_callups(
    player_ids: list[int],
    callups: list[PlayerCallup],
    position_weights: dict[str, float] | None = None,
) -> float:
    """Return the weighted callup count for the given player list.

    ``player_ids`` — the team's starting XIII/named-squad player IDs.
    ``callups`` — all representative selections for this fixture window.
    ``position_weights`` — optional; if None, each callup counts as 1.0.

    Returns the sum of weights (or raw count) for players in both lists.
    """
    callup_ids = {c.player_id for c in callups}
    if position_weights is None:
        return float(sum(1 for pid in player_ids if pid in callup_ids))
    # Without position info in the callup record, uniform weight per player.
    return float(sum(1.0 for pid in player_ids if pid in callup_ids))
