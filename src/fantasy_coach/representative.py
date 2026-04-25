"""NRL calendar effects: State of Origin, Magic Round, and Test windows.

This module answers three structural questions before kickoff:
1. Is this round an Origin round (named squads pull up to 17 players/state)?
2. Is this Magic Round (all games at a single venue — currently Brisbane)?
3. Is this a Test window (internationals pull Kangaroos / Pacific nations)?

For each round that falls into category 1 or 3, ``fetch_origin_squads`` can
scrape the official squad announcement and write it to the
``representative_callups`` table so ``FeatureBuilder`` can compute an
availability-adjusted callup differential per team.

Calendar data is hard-coded for 2024–2026 and falls back to a published
JSON config at ``data/representative_schedule.json`` for 2027+. The
caller (``precompute``) is responsible for calling ``fetch_origin_squads``
once per Origin week; re-running is idempotent.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

State = Literal["NSW", "QLD"]
Fixture = Literal["origin1", "origin2", "origin3", "test_au", "test_nz", "test_pac"]

# ---------------------------------------------------------------------------
# Hard-coded NRL representative-round calendar
# ---------------------------------------------------------------------------

# Mapping: season → list of (round_number, fixture_label) pairs.
# Origin rounds are when a State of Origin game is played that week.
# The *club round* affected is the round that immediately follows the
# Origin match (players are typically managed the round before as well,
# but the formal squad-publication trigger is the named 17).
#
# NRL rounds are approximate; confirm from the official draw each year.
_ORIGIN_ROUNDS: dict[int, list[tuple[int, Fixture]]] = {
    2024: [
        (13, "origin1"),
        (17, "origin2"),
        (21, "origin3"),
    ],
    2025: [
        (13, "origin1"),
        (17, "origin2"),
        (21, "origin3"),
    ],
    2026: [
        (13, "origin1"),
        (17, "origin2"),
        (21, "origin3"),
    ],
}

# Magic Round: all games played at a single neutral venue (Brisbane Stadium).
# Currently held at approximately round 12 each year; varies slightly by draw.
_MAGIC_ROUNDS: dict[int, int] = {
    2024: 12,
    2025: 11,
    2026: 12,
}

# International Test windows: date ranges (inclusive) when Kangaroos /
# Pacific nations tournaments pull NRL players out of finals warm-ups.
_TEST_WINDOWS: list[tuple[date, date]] = [
    (date(2024, 10, 25), date(2024, 11, 10)),  # Pacific Championships 2024
    (date(2025, 10, 24), date(2025, 11, 9)),   # Pacific Championships 2025
    (date(2026, 10, 23), date(2026, 11, 8)),   # Pacific Championships 2026 (est.)
]

# Path to the JSON override for years not hard-coded above.
_SCHEDULE_JSON = Path("data/representative_schedule.json")


def _load_schedule_json() -> dict[str, Any]:
    if _SCHEDULE_JSON.exists():
        try:
            return json.loads(_SCHEDULE_JSON.read_text())
        except Exception:
            logger.debug("Could not parse %s — using built-in calendar only", _SCHEDULE_JSON)
    return {}


# ---------------------------------------------------------------------------
# Calendar detection functions
# ---------------------------------------------------------------------------


def is_origin_round(season: int, round_: int) -> bool:
    """Return True when ``round_`` is an Origin round for ``season``.

    Covers rounds in ``_ORIGIN_ROUNDS`` plus any overrides in
    ``data/representative_schedule.json`` (key ``"origin_rounds"``).
    """
    hard_coded = {r for r, _ in _ORIGIN_ROUNDS.get(season, [])}
    if round_ in hard_coded:
        return True
    schedule = _load_schedule_json()
    extra = schedule.get("origin_rounds", {}).get(str(season), [])
    return round_ in extra


def origin_fixture_label(season: int, round_: int) -> Fixture | None:
    """Return the fixture label (``"origin1"`` etc.) for an Origin round.

    Returns ``None`` when the round is not an Origin round.
    """
    for r, label in _ORIGIN_ROUNDS.get(season, []):
        if r == round_:
            return label
    return None


def is_magic_round(season: int, round_: int) -> bool:
    """Return True when all games are played at the neutral Magic Round venue."""
    hard_coded = _MAGIC_ROUNDS.get(season)
    if hard_coded == round_:
        return True
    schedule = _load_schedule_json()
    return schedule.get("magic_rounds", {}).get(str(season)) == round_


def is_test_window(match_date: date) -> bool:
    """Return True when ``match_date`` falls inside a Test tournament window."""
    for start, end in _TEST_WINDOWS:
        if start <= match_date <= end:
            return True
    extra = _load_schedule_json().get("test_windows", [])
    for entry in extra:
        try:
            s = date.fromisoformat(entry["start"])
            e = date.fromisoformat(entry["end"])
            if s <= match_date <= e:
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Callup data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Callup:
    """One player selected to a representative squad."""

    player_id: int
    fixture: Fixture
    fixture_date: date
    season: int
    round: int
    state: State | None  # None for Test fixtures (not state-based)


@dataclass
class SquadAnnouncement:
    """All callups for one representative fixture."""

    fixture: Fixture
    fixture_date: date
    season: int
    round: int
    callups: list[Callup] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Squad-fetching infrastructure
# ---------------------------------------------------------------------------

_NRL_ORIGIN_BASE = "https://www.nrl.com/state-of-origin/teams/"
_SCRAPE_INTERVAL = float(os.getenv("FANTASY_COACH_SCRAPE_INTERVAL_SECONDS", "1.0"))


def fetch_origin_squads(
    season: int,
    game_number: int,
    *,
    session: Any = None,
) -> SquadAnnouncement | None:
    """Scrape the official Origin squad announcement for a given game.

    ``game_number`` is 1, 2, or 3 (corresponding to ``origin1`` / ``origin2`` /
    ``origin3`` fixtures).

    Returns ``None`` when:
    - The URL yields no parseable squad data (pre-announcement).
    - Any network/parsing error occurs (non-fatal — logged and swallowed).
    - The ``season`` is not yet in ``_ORIGIN_ROUNDS``.

    The response is idempotent: re-running for the same season+game writes
    the same callups.  The caller (``precompute``) is responsible for de-
    duplicating storage — see ``representative_callups`` table in schema.sql.

    NOTE: The NRL endpoint contract for Origin team pages is not yet
    documented in ``docs/nrl-endpoints.md`` (marked TBD in #211).  This
    function is a stub that will be completed once the endpoint is
    confirmed; it currently returns ``None`` to avoid network traffic in
    CI.  The storage interface (``SquadAnnouncement`` + ``Callup``) and
    schema are production-ready.
    """
    if season not in _ORIGIN_ROUNDS:
        logger.debug("No Origin schedule for season %d", season)
        return None

    round_map = {label: r for r, label in _ORIGIN_ROUNDS[season]}
    fixture: Fixture = f"origin{game_number}"  # type: ignore[assignment]
    round_ = round_map.get(fixture)
    if round_ is None:
        return None

    # Stub: actual HTTP scrape to be implemented when endpoint is confirmed.
    logger.info(
        "fetch_origin_squads: stub — no HTTP request made (endpoint TBD, #211). "
        "season=%d game=%d would target %s",
        season,
        game_number,
        _NRL_ORIGIN_BASE,
    )
    return None


def store_callups(callups: list[Callup], repo: Any) -> int:
    """Persist callups into the repository's ``representative_callups`` table.

    Returns the number of rows inserted (duplicates silently ignored via
    INSERT OR IGNORE).  ``repo`` must expose a ``record_callup(callup)`` method
    (implemented in ``SQLiteRepository`` — see schema.sql).
    """
    stored = 0
    for callup in callups:
        try:
            repo.record_callup(callup)
            stored += 1
        except Exception:
            logger.exception("Failed to store callup for player %d", callup.player_id)
    return stored


def get_callups_for_round(
    repo: Any,
    season: int,
    round_: int,
    *,
    fixture: Fixture | None = None,
) -> list[Callup]:
    """Return all callups active during a given season/round.

    When ``fixture`` is given, restricts to that specific representative event.
    Returns an empty list when no callup data is stored for the round (the
    feature will silently be zero — a safe default that the model handles
    via its intercept for Origin weeks).
    """
    try:
        return repo.list_callups(season=season, round_=round_, fixture=fixture)
    except Exception:
        logger.debug("Could not load callups for %d r%d", season, round_)
        return []


# ---------------------------------------------------------------------------
# Feature-level helpers (used by FeatureBuilder once data is backfilled)
# ---------------------------------------------------------------------------


def count_callups_for_team(
    team_id: int,
    callups: list[Callup],
    match_players: list[Any],
    *,
    position_weights: dict[str, float] | None = None,
) -> float:
    """Count position-weighted representative callups for players on one team.

    ``match_players`` is a list of ``PlayerRow`` objects for the team's
    current named XIII.  Only players who appear in *both* the team's roster
    AND the callup list are counted — this avoids counting a called-up player
    who was replaced (injured etc.) before the match.

    Returns the sum of ``position_weights[position]`` for each called-up
    player in the XIII, or a flat count of 1.0 per player when
    ``position_weights`` is ``None``.
    """
    called_up_ids = {c.player_id for c in callups}
    total = 0.0
    for p in match_players:
        if p.player_id not in called_up_ids:
            continue
        if position_weights is not None:
            weight = position_weights.get(p.position or "", 1.0)
        else:
            weight = 1.0
        total += weight
    return total
