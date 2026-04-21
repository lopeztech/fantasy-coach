"""Feature extraction from raw NRL `/data` payloads.

Architectural decision (see issue #7): we don't persist the raw JSON. We
extract a narrow row of fields the model needs and discard the rest, so
storage stays small and schema drift surfaces as failed extractions
rather than as silent rot inside opaque blobs.

Schema drift strategy: any top-level or per-team key the extractor
doesn't recognize is logged at WARNING level (once per call), but is not
fatal. Missing fields default to None — most match payloads in
`Upcoming` state lack scores, players, and stats, and that's expected.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


_KNOWN_TOP_LEVEL_KEYS = {
    "animateMatchClock",
    "attendance",
    "awayTeam",
    "backgroundImageLarge",
    "backgroundImageSmall",
    "broadcastChannels",
    "competition",
    "gameSeconds",
    "groundConditions",
    "hasExtraTime",
    "hasOnFieldTracking",
    "homeTeam",
    "imageUrl",
    "matchId",
    "matchMode",
    "matchState",
    "officials",
    "positionGroups",
    "roundNumber",
    "roundTitle",
    "segmentCount",
    "segmentDuration",
    "showPlayerPositions",
    "showTeamPositions",
    "startTime",
    "stats",
    "timeline",
    "updated",
    "url",
    "venue",
    "venueCity",
    "videoProviders",
    "weather",
}

_KNOWN_TEAM_KEYS = {
    "captainPlayerId",
    "coaches",
    "discipline",
    "name",
    "nextOpponent",
    "nickName",
    "odds",
    "players",
    "recentForm",
    "score",
    "scoring",
    "teamId",
    "teamPosition",
    "theme",
    "url",
}


class PlayerRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    player_id: int
    jersey_number: int | None
    position: str | None
    first_name: str | None
    last_name: str | None


class TeamRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    team_id: int
    name: str
    nick_name: str
    score: int | None
    players: list[PlayerRow]


class TeamStat(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    type: str
    units: str | None
    home_value: float | None
    away_value: float | None


class MatchRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_id: int
    season: int
    round: int
    start_time: datetime
    match_state: str
    venue: str | None
    venue_city: str | None
    weather: str | None
    home: TeamRow
    away: TeamRow
    team_stats: list[TeamStat]
    referee_id: int | None = None
    video_referee_id: int | None = None


def extract_match_features(raw: dict[str, Any]) -> MatchRow:
    """Project raw `/data` JSON into a narrow `MatchRow` for storage."""

    _log_unknown_keys("match", raw, _KNOWN_TOP_LEVEL_KEYS)

    start_time = _parse_iso_utc(raw["startTime"])
    referee_id, video_referee_id = _extract_referee_ids(raw.get("officials") or [])

    return MatchRow(
        match_id=int(raw["matchId"]),
        season=start_time.year,
        round=int(raw["roundNumber"]),
        start_time=start_time,
        match_state=str(raw["matchState"]),
        venue=raw.get("venue"),
        venue_city=raw.get("venueCity"),
        weather=raw.get("weather"),
        home=_extract_team(raw["homeTeam"]),
        away=_extract_team(raw["awayTeam"]),
        team_stats=_extract_team_stats(raw.get("stats") or {}),
        referee_id=referee_id,
        video_referee_id=video_referee_id,
    )


def _extract_team(raw: dict[str, Any]) -> TeamRow:
    _log_unknown_keys("team", raw, _KNOWN_TEAM_KEYS)
    return TeamRow(
        team_id=int(raw["teamId"]),
        name=str(raw["name"]),
        nick_name=str(raw["nickName"]),
        score=_optional_int(raw.get("score")),
        players=[_extract_player(p) for p in raw.get("players") or []],
    )


def _extract_player(raw: dict[str, Any]) -> PlayerRow:
    return PlayerRow(
        player_id=int(raw["playerId"]),
        jersey_number=_optional_int(raw.get("number")),
        position=raw.get("position"),
        first_name=raw.get("firstName"),
        last_name=raw.get("lastName"),
    )


def _extract_team_stats(stats: dict[str, Any]) -> list[TeamStat]:
    rows: list[TeamStat] = []
    for group in stats.get("groups") or []:
        for stat in group.get("stats") or []:
            rows.append(
                TeamStat(
                    title=str(stat["title"]),
                    type=str(stat["type"]),
                    units=stat.get("units"),
                    home_value=_stat_value(stat.get("homeValue")),
                    away_value=_stat_value(stat.get("awayValue")),
                )
            )
    return rows


def _stat_value(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        v = raw.get("value")
        return float(v) if v is not None else None
    return float(raw)


def _optional_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def _parse_iso_utc(raw: str) -> datetime:
    # `2024-03-03T02:30:00Z` — fromisoformat handles `Z` from Python 3.11+.
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _extract_referee_ids(officials: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Return (referee_id, video_referee_id) from the officials array.

    NRL positions: "Referee" = main on-field referee, "Senior Review Official" = video ref.
    Both default to None when the officials block is absent (upcoming fixtures).
    """
    referee_id: int | None = None
    video_referee_id: int | None = None
    for official in officials:
        pid = official.get("profileId")
        if pid is None:
            continue
        position = (official.get("position") or "").strip()
        if position == "Referee" and referee_id is None:
            referee_id = int(pid)
        elif position == "Senior Review Official" and video_referee_id is None:
            video_referee_id = int(pid)
    return referee_id, video_referee_id


def _log_unknown_keys(scope: str, raw: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(raw.keys()) - known)
    if unknown:
        logger.warning("Unknown %s keys in payload: %s", scope, unknown)
