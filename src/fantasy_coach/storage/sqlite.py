"""SQLite implementation of `Repository`.

Uses `sqlite3` stdlib — no ORM needed at this size. Upserts are done as
`DELETE + INSERT` inside a transaction so child rows (players, stats)
stay consistent when a match transitions Upcoming → Live → FullTime.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime
from importlib import resources
from pathlib import Path

from fantasy_coach.features import MatchRow, PlayerRow, TeamRow, TeamStat

SCHEMA_VERSION = 3


class SQLiteRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upsert_match(self, row: MatchRow) -> None:
        with self._conn:  # commit on success, rollback on exception
            self._conn.execute("DELETE FROM matches WHERE match_id = ?", (row.match_id,))
            self._conn.execute(
                """
                INSERT INTO matches (
                    match_id, season, round, start_time, match_state,
                    venue, venue_city, weather,
                    home_team_id, home_name, home_nick, home_score,
                    away_team_id, away_name, away_nick, away_score,
                    referee_id, video_referee_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.match_id,
                    row.season,
                    row.round,
                    row.start_time.isoformat(),
                    row.match_state,
                    row.venue,
                    row.venue_city,
                    row.weather,
                    row.home.team_id,
                    row.home.name,
                    row.home.nick_name,
                    row.home.score,
                    row.away.team_id,
                    row.away.name,
                    row.away.nick_name,
                    row.away.score,
                    row.referee_id,
                    row.video_referee_id,
                ),
            )
            self._insert_players(row.match_id, "home", row.home.players)
            self._insert_players(row.match_id, "away", row.away.players)
            self._insert_stats(row.match_id, row.team_stats)

    def get_match(self, match_id: int) -> MatchRow | None:
        match = self._conn.execute(
            "SELECT * FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()
        if match is None:
            return None
        return self._hydrate(match)

    def list_matches(self, season: int, round: int | None = None) -> list[MatchRow]:
        if round is None:
            rows = self._conn.execute(
                "SELECT * FROM matches WHERE season = ? ORDER BY start_time, match_id",
                (season,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM matches
                WHERE season = ? AND round = ?
                ORDER BY start_time, match_id
                """,
                (season, round),
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    # ----- internals -----

    def _migrate(self) -> None:
        schema_sql = (
            resources.files("fantasy_coach.storage")
            .joinpath("schema.sql")
            .read_text(encoding="utf-8")
        )
        self._conn.executescript(schema_sql)
        current = self._conn.execute("SELECT version FROM schema_version").fetchone()
        if current is None:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
        elif current["version"] < SCHEMA_VERSION:
            self._apply_migrations(current["version"])
        elif current["version"] > SCHEMA_VERSION:
            raise RuntimeError(
                f"SQLite DB at {self._path} is schema v{current['version']}, "
                f"code expects v{SCHEMA_VERSION}. Downgrade not supported."
            )

    def _apply_migrations(self, from_version: int) -> None:
        if from_version < 2:
            with self._conn:
                # v1 → v2: add referee columns (ALTER TABLE is idempotent-ish via try/except)
                for col in ("referee_id INTEGER", "video_referee_id INTEGER"):
                    with contextlib.suppress(Exception):
                        self._conn.execute(f"ALTER TABLE matches ADD COLUMN {col}")
                self._conn.execute("UPDATE schema_version SET version = 2")
        if from_version < 3:
            with self._conn:
                # v2 → v3: add is_on_field to match_players. The
                # team_list_snapshots table is created by executescript(schema)
                # above via CREATE TABLE IF NOT EXISTS, so no work needed here.
                with contextlib.suppress(Exception):
                    self._conn.execute("ALTER TABLE match_players ADD COLUMN is_on_field INTEGER")
                self._conn.execute("UPDATE schema_version SET version = 3")

    def _insert_players(self, match_id: int, side: str, players: list[PlayerRow]) -> None:
        if not players:
            return
        self._conn.executemany(
            """
            INSERT INTO match_players
                (match_id, side, player_id, jersey_number, position,
                 first_name, last_name, is_on_field)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    match_id,
                    side,
                    p.player_id,
                    p.jersey_number,
                    p.position,
                    p.first_name,
                    p.last_name,
                    None if p.is_on_field is None else int(p.is_on_field),
                )
                for p in players
            ],
        )

    def _insert_stats(self, match_id: int, stats: list[TeamStat]) -> None:
        if not stats:
            return
        self._conn.executemany(
            """
            INSERT INTO match_team_stats
                (match_id, ordinal, title, type, units, home_value, away_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    match_id,
                    ordinal,
                    s.title,
                    s.type,
                    s.units,
                    s.home_value,
                    s.away_value,
                )
                for ordinal, s in enumerate(stats)
            ],
        )

    def _hydrate(self, match: sqlite3.Row) -> MatchRow:
        match_id = match["match_id"]
        players = self._conn.execute(
            """
            SELECT side, player_id, jersey_number, position, first_name, last_name, is_on_field
            FROM match_players
            WHERE match_id = ?
            ORDER BY side, jersey_number, player_id
            """,
            (match_id,),
        ).fetchall()
        stats = self._conn.execute(
            """
            SELECT title, type, units, home_value, away_value
            FROM match_team_stats
            WHERE match_id = ?
            ORDER BY ordinal
            """,
            (match_id,),
        ).fetchall()

        home_players = [self._row_to_player(p) for p in players if p["side"] == "home"]
        away_players = [self._row_to_player(p) for p in players if p["side"] == "away"]

        return MatchRow(
            match_id=match_id,
            season=match["season"],
            round=match["round"],
            start_time=datetime.fromisoformat(match["start_time"]),
            match_state=match["match_state"],
            venue=match["venue"],
            venue_city=match["venue_city"],
            weather=match["weather"],
            home=TeamRow(
                team_id=match["home_team_id"],
                name=match["home_name"],
                nick_name=match["home_nick"],
                score=match["home_score"],
                players=home_players,
            ),
            away=TeamRow(
                team_id=match["away_team_id"],
                name=match["away_name"],
                nick_name=match["away_nick"],
                score=match["away_score"],
                players=away_players,
            ),
            team_stats=[
                TeamStat(
                    title=s["title"],
                    type=s["type"],
                    units=s["units"],
                    home_value=s["home_value"],
                    away_value=s["away_value"],
                )
                for s in stats
            ],
            referee_id=match["referee_id"],
            video_referee_id=match["video_referee_id"],
        )

    @staticmethod
    def _row_to_player(p: sqlite3.Row) -> PlayerRow:
        raw_on_field = p["is_on_field"]
        return PlayerRow(
            player_id=p["player_id"],
            jersey_number=p["jersey_number"],
            position=p["position"],
            first_name=p["first_name"],
            last_name=p["last_name"],
            is_on_field=None if raw_on_field is None else bool(raw_on_field),
        )
