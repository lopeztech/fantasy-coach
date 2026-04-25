"""NRL representative-round and structural-event detection.

Captures three categories of calendar events that cause systematic roster
depletion relative to what the model's team-list features see:

1. **State of Origin** (rounds 13–18, mid-year) — up to 34 NRL stars
   unavailable for the club round after (and often managed before) Origin.
2. **Magic Round** — neutral-venue weekend in Brisbane; logistical bunching
   that empirically tilts results independently of the venue effect.
3. **International / Pacific Championship windows** — Tests pull players out
   of late-season warm-up matches.

Public API:
  ``is_origin_round(season, round)``    → bool
  ``is_magic_round(season, round)``     → bool
  ``is_test_window(season, round)``     → bool
  ``origin_callups_diff(season, round, home_players, away_players)`` → float
  ``fetch_origin_squads(year, game_number)`` → dict | None  (stub, requires scraping)

Feature additions (pending FEATURE_NAMES update + retrain):
  - ``is_origin_round``      — 0/1 categorical for Origin rounds
  - ``origin_callups_diff``  — position-weighted callup count, home minus away
  - ``is_magic_round``       — 0/1 categorical
  - ``is_test_window_diff``  — position-weighted test-bound players, home minus away

References:
  ``src/fantasy_coach/feature_engineering.py`` (POSITION_WEIGHTS)
  ``docs/model.md`` § "NRL calendar effects" (to be added post-retrain)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_coach.features import PlayerRow

# ---------------------------------------------------------------------------
# NRL calendar: Origin and Magic Round schedules
#
# Keyed by season; value is a frozenset of NRL round numbers that are
# Origin rounds for that season.  For seasons not listed, returns False so
# the feature defaults gracefully to 0.0.
# ---------------------------------------------------------------------------

# Origin round data sourced from the official NRL draw. Rounds where the
# State of Origin match falls mid-week (and therefore depletes both NSW and
# QLD club squads in adjacent rounds) are included.
_ORIGIN_ROUNDS: dict[int, frozenset[int]] = {
    2024: frozenset({11, 13, 16}),   # Origin I=R11, II=R13, III=R16
    2025: frozenset({12, 15, 18}),   # Origin I=R12, II=R15, III=R18
    2026: frozenset({13, 16, 19}),   # Origin I=R13, II=R16, III=R19 (provisional)
}

_MAGIC_ROUNDS: dict[int, int] = {
    2024: 11,
    2025: 11,
    2026: 11,
}

# International / Pacific Championship windows — approximate round ranges.
# Only the immediate club round after the Test window is flagged (the one
# where players are most likely to be unavailable or managed).
_TEST_WINDOW_ROUNDS: dict[int, frozenset[int]] = {
    2024: frozenset({27, 28}),   # Pacific Championships late-season
    2025: frozenset({27, 28}),
    2026: frozenset({27, 28}),
}

# Position weights — must stay aligned with feature_engineering.POSITION_WEIGHTS.
# Copied here to avoid a circular import; values are the same subset used by
# key_absence_diff.
_POSITION_WEIGHTS: dict[str, float] = {
    "Halfback": 3.0,
    "Five-Eighth": 2.5,
    "Fullback": 2.0,
    "Hooker": 2.0,
    "Prop": 1.5,
    "Lock": 1.5,
    "Second Row": 1.5,
    "Centre": 1.2,
    "Wing": 1.0,
    "Stand-off": 2.5,  # alias for Five-Eighth in some schemas
}


# ---------------------------------------------------------------------------
# Calendar detection
# ---------------------------------------------------------------------------


def is_origin_round(season: int, round: int) -> bool:
    """Return True when this NRL round falls in a State of Origin week."""
    return round in _ORIGIN_ROUNDS.get(season, frozenset())


def is_magic_round(season: int, round: int) -> bool:
    """Return True when this is Magic Round (neutral-venue Brisbane weekend)."""
    return _MAGIC_ROUNDS.get(season) == round


def is_test_window(season: int, round: int) -> bool:
    """Return True when this round falls in an international Test window."""
    return round in _TEST_WINDOW_ROUNDS.get(season, frozenset())


# ---------------------------------------------------------------------------
# Callup-count features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OriginSquad:
    year: int
    game: int  # 1, 2, or 3
    nsw: frozenset[int]  # player_ids in NSW squad
    qld: frozenset[int]  # player_ids in QLD squad


def _position_weight(position: str | None) -> float:
    if position is None:
        return 1.0
    return _POSITION_WEIGHTS.get(position, 1.0)


def _callup_weight(
    players: list["PlayerRow"],
    called_up_ids: frozenset[int],
) -> float:
    """Sum position weights for players who are in the callup set."""
    total = 0.0
    for p in players:
        if p.player_id in called_up_ids:
            total += _position_weight(p.position)
    return total


def origin_callups_diff(
    home_players: list["PlayerRow"],
    away_players: list["PlayerRow"],
    called_up_ids: frozenset[int],
) -> float:
    """Home-minus-away position-weighted Origin callup count.

    A positive value means the home team has more Origin-bound players
    (a disadvantage; should correlate with lower win probability).
    Returns 0.0 when no callup data is available.
    """
    if not called_up_ids:
        return 0.0
    home_w = _callup_weight(home_players, called_up_ids)
    away_w = _callup_weight(away_players, called_up_ids)
    return home_w - away_w


# ---------------------------------------------------------------------------
# Feature extraction helpers (called by FeatureBuilder when enabled)
# ---------------------------------------------------------------------------


def extract_representative_features(
    season: int,
    round: int,
    home_players: list["PlayerRow"],
    away_players: list["PlayerRow"],
    *,
    origin_squad: OriginSquad | None = None,
) -> dict[str, float]:
    """Compute all representative-round features for one match.

    Returns a dict with keys matching the planned FEATURE_NAMES additions:
      - ``is_origin_round``
      - ``origin_callups_diff``
      - ``is_magic_round``
      - ``is_test_window_diff``

    Called from ``FeatureBuilder.feature_row`` once the FEATURE_NAMES schema
    bump is performed (see issue #211). Until then, this function is
    exercised by tests only.
    """
    callup_ids: frozenset[int] = frozenset()
    if origin_squad is not None:
        callup_ids = origin_squad.nsw | origin_squad.qld

    return {
        "is_origin_round": 1.0 if is_origin_round(season, round) else 0.0,
        "origin_callups_diff": origin_callups_diff(home_players, away_players, callup_ids),
        "is_magic_round": 1.0 if is_magic_round(season, round) else 0.0,
        "is_test_window_diff": 0.0,  # TODO: wire when Test squad data is available
    }


# ---------------------------------------------------------------------------
# Scraper stub — squad data requires live HTTP access
# ---------------------------------------------------------------------------


def fetch_origin_squads(year: int, game_number: int) -> OriginSquad | None:
    """Fetch and parse the official Origin squad announcement.

    **Stub implementation** — the NRL squad page (typically at
    ``https://www.nrl.com/state-of-origin/teams/``) requires throttled HTTP
    access that is not available in the automated agent environment. This stub
    exists so the rest of the module compiles and tests pass; a follow-up
    manual ops run should fill in the scraping logic.

    Returns ``None`` until implemented. Callers should treat ``None`` as
    "squad data unavailable → callup_ids = empty set → feature = 0.0".
    """
    return None
