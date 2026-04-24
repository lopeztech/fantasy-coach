"""Firestore implementation of `Repository`.

Uses Google Cloud Firestore as the production storage backend. Each match
is stored as a document with ID ``str(match_id)`` inside the ``matches``
collection so point-lookups are O(1).

Queries that filter by season or season+round use a composite index on
``(season, start_time, match_id)``. The emulator creates indexes automatically;
production requires a one-time ``gcloud firestore indexes create`` or the
equivalent Terraform resource in ``lopeztech/platform-infra``.

Pass ``client`` in the constructor for testing (any object that satisfies the
Firestore client duck-type). Production code omits ``client``; the real
``google.cloud.firestore.Client`` is imported lazily so the package remains
optional during local SQLite-only development.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fantasy_coach.features import MatchRow, PlayerRow, TeamRow, TeamStat

_COLLECTION = "matches"


class FirestoreRepository:
    """Firestore-backed implementation of the Repository protocol."""

    def __init__(
        self,
        client: Any = None,
        project: str | None = None,
        database: str = "(default)",
    ) -> None:
        if client is not None:
            self._db = client
        else:
            from google.cloud import firestore  # noqa: PLC0415

            self._db = firestore.Client(project=project, database=database)

    # ---- Repository protocol ----

    def upsert_match(self, row: MatchRow) -> None:
        self._col.document(str(row.match_id)).set(_to_doc(row))

    def get_match(self, match_id: int) -> MatchRow | None:
        snap = self._col.document(str(match_id)).get()
        return _from_doc(snap.to_dict()) if snap.exists else None

    def list_matches(self, season: int, round: int | None = None) -> list[MatchRow]:
        query = self._col.where("season", "==", season)
        if round is not None:
            query = query.where("round", "==", round)
        return [
            _from_doc(d.to_dict())
            for d in query.order_by("start_time").order_by("match_id").stream()
        ]

    @property
    def _col(self) -> Any:
        return self._db.collection(_COLLECTION)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _to_doc(row: MatchRow) -> dict[str, Any]:
    return {
        "match_id": row.match_id,
        "season": row.season,
        "round": row.round,
        "start_time": row.start_time.isoformat(),
        "match_state": row.match_state,
        "venue": row.venue,
        "venue_city": row.venue_city,
        "weather": row.weather,
        "home": _team_to_doc(row.home),
        "away": _team_to_doc(row.away),
        "team_stats": [_stat_to_doc(s) for s in row.team_stats],
        "referee_id": row.referee_id,
        "video_referee_id": row.video_referee_id,
    }


def _team_to_doc(t: TeamRow) -> dict[str, Any]:
    return {
        "team_id": t.team_id,
        "name": t.name,
        "nick_name": t.nick_name,
        "score": t.score,
        "players": [_player_to_doc(p) for p in t.players],
        "odds": t.odds,
        "odds_open": t.odds_open,
    }


def _player_to_doc(p: PlayerRow) -> dict[str, Any]:
    return {
        "player_id": p.player_id,
        "jersey_number": p.jersey_number,
        "position": p.position,
        "first_name": p.first_name,
        "last_name": p.last_name,
        "is_on_field": p.is_on_field,
    }


def _stat_to_doc(s: TeamStat) -> dict[str, Any]:
    return {
        "title": s.title,
        "type": s.type,
        "units": s.units,
        "home_value": s.home_value,
        "away_value": s.away_value,
    }


def _from_doc(d: dict[str, Any]) -> MatchRow:
    return MatchRow(
        match_id=d["match_id"],
        season=d["season"],
        round=d["round"],
        start_time=datetime.fromisoformat(d["start_time"]),
        match_state=d["match_state"],
        venue=d.get("venue"),
        venue_city=d.get("venue_city"),
        weather=d.get("weather"),
        home=_team_from_doc(d["home"]),
        away=_team_from_doc(d["away"]),
        team_stats=[_stat_from_doc(s) for s in d.get("team_stats", [])],
        referee_id=d.get("referee_id"),
        video_referee_id=d.get("video_referee_id"),
    )


def _team_from_doc(d: dict[str, Any]) -> TeamRow:
    return TeamRow(
        team_id=d["team_id"],
        name=d["name"],
        nick_name=d["nick_name"],
        score=d.get("score"),
        players=[_player_from_doc(p) for p in d.get("players", [])],
        odds=d.get("odds"),
        odds_open=d.get("odds_open"),
    )


def _player_from_doc(d: dict[str, Any]) -> PlayerRow:
    return PlayerRow(
        player_id=d["player_id"],
        jersey_number=d.get("jersey_number"),
        position=d.get("position"),
        first_name=d.get("first_name"),
        last_name=d.get("last_name"),
        is_on_field=d.get("is_on_field"),
    )


def _stat_from_doc(d: dict[str, Any]) -> TeamStat:
    return TeamStat(
        title=d["title"],
        type=d["type"],
        units=d.get("units"),
        home_value=d.get("home_value"),
        away_value=d.get("away_value"),
    )
