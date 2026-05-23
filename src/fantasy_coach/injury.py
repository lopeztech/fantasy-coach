"""Injury reports + late-withdrawal data model (#208).

`InjuryReport` captures one player's weekly injury-list status — produced
by a parser over NRL.com's weekly injury list (or a secondary source like
NRL Physio). One report per (player, season, round, source).

`LateTeamChange` captures a starting-XIII change observed inside the
kickoff hour. Produced by the `watch-team-lists` job that polls match
team lists every 15 minutes on match days and diffs against the prior
snapshot.

Neither dataclass touches storage — see `storage/injury.py` for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class InjuryStatus(StrEnum):
    """Coarse injury-list buckets that map across sources.

    NRL.com, NRL Physio, and Fox Sports all use slightly different
    vocabularies; these are the buckets every parser must normalise into.
    """

    OUT = "out"
    """Confirmed unavailable for the round."""

    TEST = "test"
    """Listed for a fitness test late in the week — availability uncertain."""

    SQUAD_21 = "21_man_squad"
    """In the 21-man extended squad but not the 17-man matchday team."""

    RETURNING = "returning"
    """First match back from injury (proxy for rust)."""

    SUSPENDED = "suspended"
    """Judiciary suspension. Knowable a week in advance."""


_VALID_STATUSES = frozenset(s.value for s in InjuryStatus)


@dataclass(frozen=True)
class InjuryReport:
    """One player's injury-list status for one round.

    `weeks_out` is the source's best estimate of how many further rounds
    the player will miss after this one. ``None`` means unspecified
    (common for "test" status).
    """

    player_id: int
    season: int
    round: int
    team_id: int
    status: InjuryStatus
    weeks_out: int | None
    body_part: str | None
    source: str
    scraped_at: datetime

    def __post_init__(self) -> None:
        if self.scraped_at.tzinfo is None:
            raise ValueError("scraped_at must be timezone-aware (UTC)")
        if self.status.value not in _VALID_STATUSES:
            raise ValueError(f"unknown injury status: {self.status!r}")
        if self.weeks_out is not None and self.weeks_out < 0:
            raise ValueError("weeks_out must be non-negative")


@dataclass(frozen=True)
class LateTeamChange:
    """A starting-XIII add/drop observed inside the kickoff hour.

    Distinct from `team_lists.TeamListChange` — that's the diff between
    two arbitrary scrapes. This is specifically the diff between the last
    pre-kickoff-window snapshot and a snapshot taken inside the kickoff
    window. `detected_at` is the timestamp of the *later* snapshot, so
    `start_time - detected_at` gives "how late was the change".
    """

    match_id: int
    team_id: int
    player_id: int
    change_type: str  # "withdrawn" | "added" | "position_change"
    position: str | None
    was_starting: bool
    is_starting: bool
    detected_at: datetime

    def __post_init__(self) -> None:
        if self.detected_at.tzinfo is None:
            raise ValueError("detected_at must be timezone-aware (UTC)")
        if self.change_type not in {"withdrawn", "added", "position_change"}:
            raise ValueError(f"unknown change_type: {self.change_type!r}")
