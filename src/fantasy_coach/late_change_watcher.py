"""Late team-list watcher (#208).

Polls upcoming matches whose kickoff is inside a configurable lead-time
window and snapshots their team list, then diffs against the most recent
prior snapshot to detect late starting-XIII changes — the kind of move
that bookmaker closing lines react to in 0.5–1.5 point swings but the
Tue/Thu precompute Job misses entirely.

The watcher is meant to run on a third Cloud Run schedule (every 15 min
on match days). Each invocation:

1. Lists upcoming matches with kickoff ≤ ``lead_minutes`` away.
2. For each, re-fetches the per-match endpoint to get a fresh team list.
3. Records the snapshot in ``team_list_snapshots`` (so the Tue/Thu diff
   downstream picks it up too).
4. Computes a starting-XIII diff vs the latest prior snapshot whose
   ``scraped_at`` is at least ``cutoff_minutes_before_kickoff`` before
   kickoff (so the "early" reference is the named team list, not another
   intra-window scrape).
5. Writes one ``LateTeamChange`` per add/drop/position-change to
   ``late_team_changes``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from fantasy_coach.features import PlayerRow
from fantasy_coach.injury import LateTeamChange
from fantasy_coach.storage.injury import LateTeamChangeRepository
from fantasy_coach.storage.team_list import TeamListRepository
from fantasy_coach.team_lists import TeamListSnapshot

logger = logging.getLogger(__name__)

# Default kickoff-window settings. Tunable via the CLI subcommand.
DEFAULT_LEAD_MINUTES = 90
DEFAULT_CUTOFF_MINUTES = 60


def compute_late_changes(
    *,
    earlier: TeamListSnapshot,
    later: TeamListSnapshot,
) -> list[LateTeamChange]:
    """Diff two snapshots into a flat list of ``LateTeamChange`` rows.

    Mirrors the (match_id, team_id) guard from
    ``compute_team_list_changes`` but emits one row per *player* rather
    than two grouped tuples — that lines up with how ``late_team_changes``
    stores them and how the feature builder reads them.

    Players whose ``is_on_field`` is ``None`` in *either* snapshot are
    skipped. A pre-team-list-drop scrape doesn't carry ``is_on_field``,
    so a sensible "starter" set isn't derivable.
    """
    if earlier.match_id != later.match_id or earlier.team_id != later.team_id:
        raise ValueError(
            f"Snapshots must share (match_id, team_id); got "
            f"({earlier.match_id}, {earlier.team_id}) vs ({later.match_id}, {later.team_id})"
        )

    earlier_by_id = {p.player_id: p for p in earlier.players}
    later_by_id = {p.player_id: p for p in later.players}

    # Skip players whose is_on_field is None in either snapshot — we
    # can't tell whether they're starting, so we can't claim a change.
    # A pre-team-list-drop scrape carries None for everyone, so this
    # is the right guard against false-positive "added" rows on the
    # first scrape after the team list drops.
    rows: list[LateTeamChange] = []

    union_pids = set(earlier_by_id) | set(later_by_id)
    for pid in union_pids:
        before = earlier_by_id.get(pid)
        after = later_by_id.get(pid)
        before_known = before is not None and before.is_on_field is not None
        after_known = after is not None and after.is_on_field is not None
        if not (before_known and after_known):
            continue
        was = bool(before.is_on_field)
        now_starting = bool(after.is_on_field)
        if was and not now_starting:
            rows.append(
                _row(
                    later,
                    pid,
                    before.position,
                    was_starting=True,
                    is_starting=False,
                    kind="withdrawn",
                )
            )
        elif not was and now_starting:
            rows.append(
                _row(
                    later,
                    pid,
                    after.position,
                    was_starting=False,
                    is_starting=True,
                    kind="added",
                )
            )
        elif was and now_starting and before.position != after.position:
            # Position changes inside the starting XIII (e.g. halfback
            # → five-eighth). Bench/reserve position swaps don't move
            # closing lines so we don't bother recording them.
            rows.append(
                _row(
                    later,
                    pid,
                    after.position,
                    was_starting=True,
                    is_starting=True,
                    kind="position_change",
                )
            )

    rows.sort(key=lambda r: (r.player_id, r.change_type))
    return rows


def _row(
    later: TeamListSnapshot,
    player_id: int,
    position: str | None,
    *,
    was_starting: bool,
    is_starting: bool,
    kind: str,
) -> LateTeamChange:
    return LateTeamChange(
        match_id=later.match_id,
        team_id=later.team_id,
        player_id=player_id,
        change_type=kind,
        position=position,
        was_starting=was_starting,
        is_starting=is_starting,
        detected_at=later.scraped_at,
    )


def select_reference_snapshot(
    snapshots: Iterable[TeamListSnapshot],
    *,
    kickoff: datetime,
    cutoff_minutes_before_kickoff: int = DEFAULT_CUTOFF_MINUTES,
) -> TeamListSnapshot | None:
    """Pick the latest snapshot at least ``cutoff_minutes`` before kickoff.

    Filters out snapshots from inside the kickoff window so multiple
    watcher invocations don't compare against each other (we want each
    intra-window scrape to compare against the named team list, not
    against a 30-minute-old version of itself).
    """
    cutoff = kickoff - timedelta(minutes=cutoff_minutes_before_kickoff)
    eligible = [s for s in snapshots if s.scraped_at < cutoff]
    if not eligible:
        return None
    return max(eligible, key=lambda s: s.scraped_at)


def watch_match(
    *,
    match: Any,
    fetch_match: Any,
    team_list_repo: TeamListRepository,
    late_repo: LateTeamChangeRepository,
    now: datetime | None = None,
    cutoff_minutes_before_kickoff: int = DEFAULT_CUTOFF_MINUTES,
) -> list[LateTeamChange]:
    """Snapshot a match's team lists + emit any late changes vs the
    pre-window reference snapshot.

    ``match`` is a ``MatchRow``-shaped object: requires ``match_id``,
    ``season``, ``round``, ``start_time``, ``home.team_id``,
    ``away.team_id``. ``fetch_match`` is a ``(matchCentreUrl,) -> dict``
    callable (or ``None`` on 304/404) so callers can plug in the live
    scraper or a fake. The watcher only writes when there's actual
    fresh data — a 304 cache hit short-circuits.

    Returns the rows written for this match (empty if no late changes).
    """
    now = now or datetime.now(UTC)
    payload = fetch_match(match)
    if payload is None:
        # 304 or 404 — nothing to record.
        logger.info("watch_match: no fresh payload for match_id=%s", match.match_id)
        return []

    fresh_home = _snapshot_from_payload(payload, match=match, side="home", scraped_at=now)
    fresh_away = _snapshot_from_payload(payload, match=match, side="away", scraped_at=now)

    written: list[LateTeamChange] = []
    for fresh in (fresh_home, fresh_away):
        # Always store the fresh snapshot so the next invocation has a
        # current baseline for the broader Tue/Thu diff path too.
        team_list_repo.record_snapshot(fresh)

        prior_snapshots = team_list_repo.list_snapshots(
            match_id=match.match_id, team_id=fresh.team_id
        )
        # Filter out the snapshot we just inserted — listing is
        # repository-implementation dependent on whether the read sees
        # an uncommitted insert in the same transaction. Ordering by
        # scraped_at lets us drop anything at or after `now`.
        prior_snapshots = [s for s in prior_snapshots if s.scraped_at < now]

        reference = select_reference_snapshot(
            prior_snapshots,
            kickoff=match.start_time,
            cutoff_minutes_before_kickoff=cutoff_minutes_before_kickoff,
        )
        if reference is None:
            logger.info(
                "watch_match: no reference snapshot for match=%s team=%s; recording snapshot only",
                match.match_id,
                fresh.team_id,
            )
            continue

        for change in compute_late_changes(earlier=reference, later=fresh):
            late_repo.record_change(change)
            written.append(change)

    return written


def _snapshot_from_payload(
    payload: dict[str, Any], *, match: Any, side: str, scraped_at: datetime
) -> TeamListSnapshot:
    team_key = "homeTeam" if side == "home" else "awayTeam"
    team_payload = payload.get(team_key) or {}
    raw_players = team_payload.get("players") or []
    fallback_team = match.home if side == "home" else match.away
    team_id = int(team_payload.get("teamId") or fallback_team.team_id)
    return TeamListSnapshot(
        season=match.season,
        round=match.round,
        match_id=match.match_id,
        team_id=team_id,
        scraped_at=scraped_at,
        players=tuple(_player_from_payload(p) for p in raw_players),
    )


def _player_from_payload(p: dict[str, Any]) -> PlayerRow:
    raw_on_field = p.get("isOnField")
    return PlayerRow(
        player_id=int(p["playerId"]),
        jersey_number=_optional_int(p.get("number")),
        position=p.get("position"),
        first_name=p.get("firstName"),
        last_name=p.get("lastName"),
        is_on_field=bool(raw_on_field) if raw_on_field is not None else None,
    )


def _optional_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def find_matches_in_window(
    *,
    matches: Iterable[Any],
    now: datetime,
    lead_minutes: int = DEFAULT_LEAD_MINUTES,
) -> list[Any]:
    """Filter to upcoming matches whose kickoff is inside ``[now, now + lead]``.

    Already-kicked-off matches are excluded — we only care about late
    changes, not in-game lineups.
    """
    horizon = now + timedelta(minutes=lead_minutes)
    return [m for m in matches if now <= m.start_time <= horizon]
