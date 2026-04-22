"""Team-list snapshots: capture how starting XIIIs change between scrapes.

The NRL match endpoint's ``players`` array is a structured view of each
team's team list — jersey, position, and an ``isOnField`` flag for every
player named, which distinguishes the starting XIII from the bench.

The precompute Job scrapes this twice a week (Tue 09:00 AEST after team
lists drop, Thu 06:00 AEST after late changes). Persisting each scrape as
a snapshot lets the model upgrade in #27 see which players moved between
the two — e.g. a halfback in Tuesday's starting XIII who ends up on the
bench by Thursday kickoff. No LLM / announcement parsing required: the
structured data is already in the scrape payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from fantasy_coach.features import PlayerRow


@dataclass(frozen=True)
class TeamListSnapshot:
    """One team's team list as observed at a specific scrape time."""

    season: int
    round: int
    match_id: int
    team_id: int
    scraped_at: datetime  # UTC; naive datetimes are rejected at construction
    players: tuple[PlayerRow, ...]

    def __post_init__(self) -> None:
        if self.scraped_at.tzinfo is None:
            raise ValueError("scraped_at must be timezone-aware (UTC)")


@dataclass(frozen=True)
class TeamListChange:
    """Starting-XIII adds/drops between two snapshots of the same team list.

    ``dropped`` and ``added`` are the raw ``PlayerRow`` objects from the
    source snapshot (so callers see jersey/position/name without another
    lookup). A "drop" here means the player was starting in ``earlier`` but
    is not starting in ``later``: could have moved to bench, been withdrawn,
    or replaced. The reverse for "add".
    """

    dropped: tuple[PlayerRow, ...]
    added: tuple[PlayerRow, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.dropped or self.added)


def compute_team_list_changes(
    earlier: TeamListSnapshot,
    later: TeamListSnapshot,
) -> TeamListChange:
    """Diff starting-XIII membership between two snapshots of the same team.

    Raises ``ValueError`` when the snapshots are for different ``(match_id,
    team_id)`` pairs (diffing across teams is almost always a bug). Players
    whose ``is_on_field`` is ``None`` are ignored — completed matches and
    pre-team-list-drop scrapes don't carry the flag, so a sensible "starter"
    set isn't derivable.
    """
    if earlier.match_id != later.match_id or earlier.team_id != later.team_id:
        raise ValueError(
            f"Snapshots must share (match_id, team_id); got "
            f"({earlier.match_id}, {earlier.team_id}) vs ({later.match_id}, {later.team_id})"
        )

    earlier_starters = {p.player_id for p in earlier.players if p.is_on_field}
    later_starters = {p.player_id for p in later.players if p.is_on_field}
    dropped_ids = earlier_starters - later_starters
    added_ids = later_starters - earlier_starters
    return TeamListChange(
        dropped=tuple(p for p in earlier.players if p.player_id in dropped_ids),
        added=tuple(p for p in later.players if p.player_id in added_ids),
    )
