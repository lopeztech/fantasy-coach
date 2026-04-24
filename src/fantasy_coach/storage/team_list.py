"""Storage for team-list snapshots.

Parallel to ``Repository`` but narrower — the snapshot collection is
append-only and queried by ``(match_id, team_id)`` for change detection
or by ``season`` for training-time analytics. Firestore is the
production backend; SQLite exists for local dev + test parity.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Protocol

from fantasy_coach.features import PlayerRow
from fantasy_coach.team_lists import TeamListSnapshot

_COLLECTION = "team_list_snapshots"

# Retain team-list snapshots for 10 rounds ≈ 80 days.
_TEAM_LIST_TTL_DAYS = 80


class TeamListRepository(Protocol):
    def record_snapshot(self, snapshot: TeamListSnapshot) -> None: ...

    def list_snapshots(
        self, match_id: int, team_id: int | None = None
    ) -> list[TeamListSnapshot]: ...


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


class SQLiteTeamListRepository:
    """SQLite-backed team-list snapshot store.

    Takes a live ``sqlite3.Connection`` rather than a path so callers can
    share one file with ``SQLiteRepository`` — the snapshot table lives
    alongside ``matches`` and is created by the same ``schema.sql``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    def record_snapshot(self, snapshot: TeamListSnapshot) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO team_list_snapshots
                    (season, round, match_id, team_id, scraped_at, players_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.season,
                    snapshot.round,
                    snapshot.match_id,
                    snapshot.team_id,
                    snapshot.scraped_at.isoformat(),
                    json.dumps([_player_to_dict(p) for p in snapshot.players]),
                ),
            )

    def list_snapshots(self, match_id: int, team_id: int | None = None) -> list[TeamListSnapshot]:
        if team_id is None:
            rows = self._conn.execute(
                """
                SELECT * FROM team_list_snapshots
                WHERE match_id = ?
                ORDER BY scraped_at ASC, snapshot_id ASC
                """,
                (match_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM team_list_snapshots
                WHERE match_id = ? AND team_id = ?
                ORDER BY scraped_at ASC, snapshot_id ASC
                """,
                (match_id, team_id),
            ).fetchall()
        return [_row_to_snapshot(r) for r in rows]


def _row_to_snapshot(r: sqlite3.Row) -> TeamListSnapshot:
    return TeamListSnapshot(
        season=r["season"],
        round=r["round"],
        match_id=r["match_id"],
        team_id=r["team_id"],
        scraped_at=datetime.fromisoformat(r["scraped_at"]),
        players=tuple(_dict_to_player(p) for p in json.loads(r["players_json"])),
    )


# ---------------------------------------------------------------------------
# Firestore
# ---------------------------------------------------------------------------


class FirestoreTeamListRepository:
    """Firestore-backed team-list snapshot store.

    Collection ``team_list_snapshots`` with auto-generated doc IDs. Each
    doc carries ``(season, round, match_id, team_id, scraped_at,
    players)`` so listing by ``(match_id, team_id)`` or by ``season`` is
    cheap (both queries are backed by composite indexes declared in
    ``lopeztech/platform-infra``).
    """

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

    def record_snapshot(self, snapshot: TeamListSnapshot) -> None:
        ttl_timestamp = snapshot.scraped_at + timedelta(days=_TEAM_LIST_TTL_DAYS)
        self._col.add(
            {
                "season": snapshot.season,
                "round": snapshot.round,
                "match_id": snapshot.match_id,
                "team_id": snapshot.team_id,
                "scraped_at": snapshot.scraped_at.isoformat(),
                "players": [_player_to_dict(p) for p in snapshot.players],
                "ttl_timestamp": ttl_timestamp,
            }
        )

    def list_snapshots(self, match_id: int, team_id: int | None = None) -> list[TeamListSnapshot]:
        query = self._col.where("match_id", "==", match_id)
        if team_id is not None:
            query = query.where("team_id", "==", team_id)
        docs = query.order_by("scraped_at").stream()
        return [_firestore_doc_to_snapshot(d.to_dict()) for d in docs]

    @property
    def _col(self) -> Any:
        return self._db.collection(_COLLECTION)


def _firestore_doc_to_snapshot(d: dict[str, Any]) -> TeamListSnapshot:
    return TeamListSnapshot(
        season=d["season"],
        round=d["round"],
        match_id=d["match_id"],
        team_id=d["team_id"],
        scraped_at=datetime.fromisoformat(d["scraped_at"]),
        players=tuple(_dict_to_player(p) for p in d.get("players", [])),
    )


# ---------------------------------------------------------------------------
# Shared serialisation helpers
# ---------------------------------------------------------------------------


def _player_to_dict(p: PlayerRow) -> dict[str, Any]:
    return {
        "player_id": p.player_id,
        "jersey_number": p.jersey_number,
        "position": p.position,
        "first_name": p.first_name,
        "last_name": p.last_name,
        "is_on_field": p.is_on_field,
    }


def _dict_to_player(d: dict[str, Any]) -> PlayerRow:
    return PlayerRow(
        player_id=d["player_id"],
        jersey_number=d.get("jersey_number"),
        position=d.get("position"),
        first_name=d.get("first_name"),
        last_name=d.get("last_name"),
        is_on_field=d.get("is_on_field"),
    )


def get_team_list_repository(
    *, sqlite_conn: sqlite3.Connection | None = None
) -> TeamListRepository:
    """Factory mirroring ``config.get_repository``'s STORAGE_BACKEND switch.

    Pass ``sqlite_conn`` when running against SQLite so the snapshot table
    lives in the same file as the main matches store. In production
    (``STORAGE_BACKEND=firestore``) the arg is ignored and a fresh
    Firestore client is created.
    """
    import os  # noqa: PLC0415

    backend = os.getenv("STORAGE_BACKEND", "sqlite").lower()
    if backend == "firestore":
        return FirestoreTeamListRepository()
    if sqlite_conn is None:
        raise RuntimeError(
            "SQLite team-list repository requires a shared sqlite3.Connection — "
            "pass `sqlite_conn=...` or set STORAGE_BACKEND=firestore"
        )
    return SQLiteTeamListRepository(sqlite_conn)
