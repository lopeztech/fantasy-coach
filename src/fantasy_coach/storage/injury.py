"""Storage for injury reports + late team-list changes (#208).

Two parallel append-only stores, each with a `Repository` Protocol plus
SQLite and Firestore implementations. Mirrors the shape of
``storage/team_list.py`` so the code patterns are familiar.

Lookup patterns the feature builder needs:
- Injury reports: by ``(season, round)`` to bulk-load all known injuries
  for a slate of matches; by ``(player_id, season, round)`` to check a
  specific player.
- Late team changes: by ``(match_id)`` to read everything that happened
  inside a kickoff window for one match.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Protocol

from fantasy_coach.injury import InjuryReport, InjuryStatus, LateTeamChange

_INJURY_COLLECTION = "injury_reports"
_LATE_CHANGE_COLLECTION = "late_team_changes"

# Retain injury reports for ~1 NRL season so we can re-train walk-forward.
_INJURY_TTL_DAYS = 365
# Late changes are cheap to keep; same retention as team-list snapshots.
_LATE_CHANGE_TTL_DAYS = 80


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class InjuryReportRepository(Protocol):
    def record_report(self, report: InjuryReport) -> None: ...

    def list_reports(
        self,
        *,
        season: int,
        round: int | None = None,
        team_id: int | None = None,
        player_id: int | None = None,
    ) -> list[InjuryReport]: ...


class LateTeamChangeRepository(Protocol):
    def record_change(self, change: LateTeamChange) -> None: ...

    def list_changes(
        self, *, match_id: int, team_id: int | None = None
    ) -> list[LateTeamChange]: ...


# ---------------------------------------------------------------------------
# SQLite — injury reports
# ---------------------------------------------------------------------------


class SQLiteInjuryReportRepository:
    """SQLite-backed injury-report store.

    Takes a live ``sqlite3.Connection`` rather than a path so callers can
    share one file with ``SQLiteRepository`` — the table lives alongside
    ``matches`` and is created by the same ``schema.sql``.

    Writes upsert via ``INSERT OR REPLACE`` keyed by
    ``(player_id, season, round, source)`` so a re-scrape of the same
    weekly list overwrites stale rows in place.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    def record_report(self, report: InjuryReport) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO injury_reports
                    (player_id, season, round, team_id, status,
                     weeks_out, body_part, source, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.player_id,
                    report.season,
                    report.round,
                    report.team_id,
                    report.status.value,
                    report.weeks_out,
                    report.body_part,
                    report.source,
                    report.scraped_at.isoformat(),
                ),
            )

    def list_reports(
        self,
        *,
        season: int,
        round: int | None = None,
        team_id: int | None = None,
        player_id: int | None = None,
    ) -> list[InjuryReport]:
        clauses = ["season = ?"]
        params: list[object] = [season]
        if round is not None:
            clauses.append("round = ?")
            params.append(round)
        if team_id is not None:
            clauses.append("team_id = ?")
            params.append(team_id)
        if player_id is not None:
            clauses.append("player_id = ?")
            params.append(player_id)
        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"""
            SELECT player_id, season, round, team_id, status,
                   weeks_out, body_part, source, scraped_at
            FROM injury_reports
            WHERE {where}
            ORDER BY round ASC, team_id ASC, player_id ASC, source ASC
            """,  # noqa: S608 — clauses are static identifiers, params are bound
            params,
        ).fetchall()
        return [_row_to_injury_report(r) for r in rows]


# ---------------------------------------------------------------------------
# SQLite — late team changes
# ---------------------------------------------------------------------------


class SQLiteLateTeamChangeRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    def record_change(self, change: LateTeamChange) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO late_team_changes
                    (match_id, team_id, player_id, change_type,
                     position, was_starting, is_starting, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    change.match_id,
                    change.team_id,
                    change.player_id,
                    change.change_type,
                    change.position,
                    int(change.was_starting),
                    int(change.is_starting),
                    change.detected_at.isoformat(),
                ),
            )

    def list_changes(self, *, match_id: int, team_id: int | None = None) -> list[LateTeamChange]:
        if team_id is None:
            rows = self._conn.execute(
                """
                SELECT match_id, team_id, player_id, change_type,
                       position, was_starting, is_starting, detected_at
                FROM late_team_changes
                WHERE match_id = ?
                ORDER BY detected_at ASC, change_id ASC
                """,
                (match_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT match_id, team_id, player_id, change_type,
                       position, was_starting, is_starting, detected_at
                FROM late_team_changes
                WHERE match_id = ? AND team_id = ?
                ORDER BY detected_at ASC, change_id ASC
                """,
                (match_id, team_id),
            ).fetchall()
        return [_row_to_late_change(r) for r in rows]


# ---------------------------------------------------------------------------
# Firestore — injury reports
# ---------------------------------------------------------------------------


class FirestoreInjuryReportRepository:
    """Firestore-backed injury-report store.

    Doc IDs are deterministic (``{season}-{round}-{player_id}-{source}``)
    so re-scrapes overwrite in place, mirroring SQLite's ``INSERT OR
    REPLACE`` semantics. Composite indexes on ``(season, round)`` and
    ``(team_id, season, round)`` are declared in
    ``lopeztech/platform-infra``.
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

    def record_report(self, report: InjuryReport) -> None:
        ttl_timestamp = report.scraped_at + timedelta(days=_INJURY_TTL_DAYS)
        doc_id = f"{report.season}-{report.round}-{report.player_id}-{report.source}"
        self._col.document(doc_id).set(
            {
                "player_id": report.player_id,
                "season": report.season,
                "round": report.round,
                "team_id": report.team_id,
                "status": report.status.value,
                "weeks_out": report.weeks_out,
                "body_part": report.body_part,
                "source": report.source,
                "scraped_at": report.scraped_at.isoformat(),
                "ttl_timestamp": ttl_timestamp,
            }
        )

    def list_reports(
        self,
        *,
        season: int,
        round: int | None = None,
        team_id: int | None = None,
        player_id: int | None = None,
    ) -> list[InjuryReport]:
        query = self._col.where("season", "==", season)
        if round is not None:
            query = query.where("round", "==", round)
        if team_id is not None:
            query = query.where("team_id", "==", team_id)
        if player_id is not None:
            query = query.where("player_id", "==", player_id)
        docs = query.order_by("round").stream()
        return [_firestore_doc_to_injury_report(d.to_dict()) for d in docs]

    @property
    def _col(self) -> Any:
        return self._db.collection(_INJURY_COLLECTION)


# ---------------------------------------------------------------------------
# Firestore — late team changes
# ---------------------------------------------------------------------------


class FirestoreLateTeamChangeRepository:
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

    def record_change(self, change: LateTeamChange) -> None:
        ttl_timestamp = change.detected_at + timedelta(days=_LATE_CHANGE_TTL_DAYS)
        self._col.add(
            {
                "match_id": change.match_id,
                "team_id": change.team_id,
                "player_id": change.player_id,
                "change_type": change.change_type,
                "position": change.position,
                "was_starting": change.was_starting,
                "is_starting": change.is_starting,
                "detected_at": change.detected_at.isoformat(),
                "ttl_timestamp": ttl_timestamp,
            }
        )

    def list_changes(self, *, match_id: int, team_id: int | None = None) -> list[LateTeamChange]:
        query = self._col.where("match_id", "==", match_id)
        if team_id is not None:
            query = query.where("team_id", "==", team_id)
        docs = query.order_by("detected_at").stream()
        return [_firestore_doc_to_late_change(d.to_dict()) for d in docs]

    @property
    def _col(self) -> Any:
        return self._db.collection(_LATE_CHANGE_COLLECTION)


# ---------------------------------------------------------------------------
# Row → dataclass helpers
# ---------------------------------------------------------------------------


def _row_to_injury_report(r: sqlite3.Row) -> InjuryReport:
    return InjuryReport(
        player_id=r["player_id"],
        season=r["season"],
        round=r["round"],
        team_id=r["team_id"],
        status=InjuryStatus(r["status"]),
        weeks_out=r["weeks_out"],
        body_part=r["body_part"],
        source=r["source"],
        scraped_at=datetime.fromisoformat(r["scraped_at"]),
    )


def _row_to_late_change(r: sqlite3.Row) -> LateTeamChange:
    return LateTeamChange(
        match_id=r["match_id"],
        team_id=r["team_id"],
        player_id=r["player_id"],
        change_type=r["change_type"],
        position=r["position"],
        was_starting=bool(r["was_starting"]),
        is_starting=bool(r["is_starting"]),
        detected_at=datetime.fromisoformat(r["detected_at"]),
    )


def _firestore_doc_to_injury_report(d: dict[str, Any]) -> InjuryReport:
    return InjuryReport(
        player_id=d["player_id"],
        season=d["season"],
        round=d["round"],
        team_id=d["team_id"],
        status=InjuryStatus(d["status"]),
        weeks_out=d.get("weeks_out"),
        body_part=d.get("body_part"),
        source=d["source"],
        scraped_at=datetime.fromisoformat(d["scraped_at"]),
    )


def _firestore_doc_to_late_change(d: dict[str, Any]) -> LateTeamChange:
    return LateTeamChange(
        match_id=d["match_id"],
        team_id=d["team_id"],
        player_id=d["player_id"],
        change_type=d["change_type"],
        position=d.get("position"),
        was_starting=bool(d["was_starting"]),
        is_starting=bool(d["is_starting"]),
        detected_at=datetime.fromisoformat(d["detected_at"]),
    )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def get_injury_report_repository(
    *, sqlite_conn: sqlite3.Connection | None = None
) -> InjuryReportRepository:
    """Mirrors ``config.get_repository`` / team-list factory.

    Pass ``sqlite_conn`` for SQLite (shared with the matches store);
    in production (``STORAGE_BACKEND=firestore``) the arg is ignored.
    """
    import os  # noqa: PLC0415

    backend = os.getenv("STORAGE_BACKEND", "sqlite").lower()
    if backend == "firestore":
        return FirestoreInjuryReportRepository()
    if sqlite_conn is None:
        raise RuntimeError(
            "SQLite injury-report repository requires a shared sqlite3.Connection — "
            "pass `sqlite_conn=...` or set STORAGE_BACKEND=firestore"
        )
    return SQLiteInjuryReportRepository(sqlite_conn)


def get_late_team_change_repository(
    *, sqlite_conn: sqlite3.Connection | None = None
) -> LateTeamChangeRepository:
    import os  # noqa: PLC0415

    backend = os.getenv("STORAGE_BACKEND", "sqlite").lower()
    if backend == "firestore":
        return FirestoreLateTeamChangeRepository()
    if sqlite_conn is None:
        raise RuntimeError(
            "SQLite late-team-change repository requires a shared sqlite3.Connection — "
            "pass `sqlite_conn=...` or set STORAGE_BACKEND=firestore"
        )
    return SQLiteLateTeamChangeRepository(sqlite_conn)
