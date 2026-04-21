"""Runtime configuration via environment variables.

Storage backend selection:
    STORAGE_BACKEND=sqlite    (default) — local SQLite file
    STORAGE_BACKEND=firestore — Google Cloud Firestore (production)

SQLite-specific:
    FANTASY_COACH_DB_PATH     path to the SQLite file (default: data/nrl.db)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_coach.storage.repository import Repository

STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "sqlite")
_DEFAULT_DB_PATH: str = os.getenv("FANTASY_COACH_DB_PATH", "data/nrl.db")


def get_repository(db_path: str | None = None) -> Repository:
    """Return the configured storage backend.

    Reads ``STORAGE_BACKEND`` (``sqlite`` or ``firestore``). For SQLite, the
    path falls back to ``FANTASY_COACH_DB_PATH`` then ``data/nrl.db``.
    """
    backend = STORAGE_BACKEND.lower()
    if backend == "firestore":
        from fantasy_coach.storage.firestore import FirestoreRepository  # noqa: PLC0415

        return FirestoreRepository()

    from fantasy_coach.storage.sqlite import SQLiteRepository  # noqa: PLC0415

    return SQLiteRepository(db_path or _DEFAULT_DB_PATH)
