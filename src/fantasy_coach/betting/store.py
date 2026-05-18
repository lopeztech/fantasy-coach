"""Storage backends for the betting-tips cache.

Mirrors the prediction-cache layout: one document per ``(season, round)``,
written by the precompute job and read by the ``/betting-tips`` endpoint.
Picks SQLite vs Firestore via the same ``STORAGE_BACKEND`` env switch as
``predictions.get_prediction_store`` so the API and Job share one knob.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fantasy_coach.betting.models import BettingTipsOut

logger = logging.getLogger(__name__)

BETTING_TIPS_DB_ENV = "FANTASY_COACH_BETTING_TIPS_DB_PATH"
DEFAULT_BETTING_TIPS_DB = "data/betting_tips.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS betting_tips (
    season     INTEGER NOT NULL,
    round      INTEGER NOT NULL,
    payload    TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    PRIMARY KEY (season, round)
);
"""

# Tips are immutable between scheduled precompute runs (Tue/Thu) so a long
# warm-cache TTL is safe. A forced refresh requires a Cloud Run instance
# bounce — same constraint as the prediction cache.
_TIPS_CACHE_TTL: float = 6 * 3600.0


class BettingTipsStore:
    """SQLite-backed cache. Used in local dev and tests."""

    def __init__(self, path: str | Path | None = None) -> None:
        db_path = Path(path or os.getenv(BETTING_TIPS_DB_ENV, DEFAULT_BETTING_TIPS_DB))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False mirrors PredictionStore — the FastAPI
        # threadpool reads from a different thread than module import.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE)

    def close(self) -> None:
        self._conn.close()

    def get(self, season: int, round_: int) -> BettingTipsOut | None:
        row = self._conn.execute(
            "SELECT payload FROM betting_tips WHERE season=? AND round=?",
            (season, round_),
        ).fetchone()
        if not row:
            return None
        return BettingTipsOut(**json.loads(row["payload"]))

    def put(self, tips: BettingTipsOut) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO betting_tips (season, round, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    tips.season,
                    tips.round,
                    tips.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )


class FirestoreBettingTipsStore:
    """Firestore-backed cache. Production.

    Doc ID: ``"{season}-{round}"`` inside the ``betting_tips`` collection.
    Shape mirrors the predictions collection: one doc per round with the full
    tips payload + a ``createdAt`` timestamp.
    """

    _COLLECTION = "betting_tips"

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
        self._cache: dict[tuple[int, int], tuple[float, BettingTipsOut]] = {}

    def close(self) -> None:
        return  # symmetry with BettingTipsStore — Firestore has no conn to close

    def get(self, season: int, round_: int) -> BettingTipsOut | None:
        key = (season, round_)
        cached = self._cache.get(key)
        if cached is not None:
            ts, tips = cached
            if time.monotonic() - ts < _TIPS_CACHE_TTL:
                return tips
            del self._cache[key]

        snap = self._db.collection(self._COLLECTION).document(_doc_id(season, round_)).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        payload = data.get("payload")
        if not payload:
            return None
        tips = BettingTipsOut(**payload)
        self._cache[key] = (time.monotonic(), tips)
        return tips

    def put(self, tips: BettingTipsOut) -> None:
        self._db.collection(self._COLLECTION).document(_doc_id(tips.season, tips.round)).set(
            {
                "season": tips.season,
                "round": tips.round,
                "payload": tips.model_dump(),
                "createdAt": datetime.now(UTC).isoformat(),
            }
        )
        self._cache[(tips.season, tips.round)] = (time.monotonic(), tips)


def _doc_id(season: int, round_: int) -> str:
    return f"{season}-{round_}"


def get_betting_tips_store() -> BettingTipsStore | FirestoreBettingTipsStore:
    """Factory: pick the betting-tips store from ``STORAGE_BACKEND``.

    Mirrors ``predictions.get_prediction_store`` so the API + Job share a
    single switch. Defaults to SQLite for local dev.
    """
    backend = os.getenv("STORAGE_BACKEND", "sqlite").lower()
    if backend == "firestore":
        return FirestoreBettingTipsStore()
    return BettingTipsStore()
