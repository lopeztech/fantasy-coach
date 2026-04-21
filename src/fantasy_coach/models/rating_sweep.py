"""Walk a Repository chronologically and apply Elo updates.

Rebuilds the rating book from scratch — call after each backfill or to test a
new K / home-advantage / regression combo. Cheap (one rating update per
match), so re-running is preferred over incremental persistence.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from fantasy_coach.features import MatchRow
from fantasy_coach.models.elo import Elo
from fantasy_coach.storage.repository import Repository

logger = logging.getLogger(__name__)

# Final state we treat as a result-bearing match. Anything else (Upcoming,
# Pre, Live mid-game) is skipped — its scores are missing or provisional.
RATEABLE_STATES = frozenset({"FullTime", "FullTimeED"})


def sweep_repository(
    repo: Repository,
    seasons: Iterable[int],
    *,
    elo: Elo | None = None,
) -> Elo:
    """Apply Elo updates for every completed match across `seasons`, in order.

    Between seasons the rating book is regressed toward the mean per
    `elo.season_regression`. Returns the (mutated) `Elo` instance.
    """

    elo = elo or Elo()
    seasons = sorted(seasons)
    for index, season in enumerate(seasons):
        if index > 0:
            elo.regress_to_mean()
        applied = _apply_season(repo, season, elo)
        logger.info("Season %d: applied %d rated matches", season, applied)
    return elo


def _apply_season(repo: Repository, season: int, elo: Elo) -> int:
    matches = sorted(repo.list_matches(season), key=_match_order_key)
    applied = 0
    for match in matches:
        if not _is_rateable(match):
            continue
        elo.update(
            match.home.team_id,
            match.away.team_id,
            int(match.home.score or 0),
            int(match.away.score or 0),
        )
        applied += 1
    return applied


def _is_rateable(match: MatchRow) -> bool:
    if match.match_state not in RATEABLE_STATES:
        return False
    return match.home.score is not None and match.away.score is not None


def _match_order_key(match: MatchRow) -> tuple:
    # list_matches already orders by start_time, but be defensive — Elo is
    # path-dependent and an out-of-order match would silently bias ratings.
    return (match.start_time, match.match_id)
