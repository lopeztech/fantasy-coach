#!/usr/bin/env python3
"""One-shot migration: read all matches from SQLite and write to Firestore.

Reads every match from a local SQLite database and upserts it into Firestore.
Idempotent — safe to run multiple times; Firestore ``set()`` is an atomic
full-document replace.

Usage:
    python scripts/migrate_sqlite_to_firestore.py --db data/nrl.db
    python scripts/migrate_sqlite_to_firestore.py --db data/nrl.db --seasons 2024,2025

Environment:
    GOOGLE_CLOUD_PROJECT   GCP project ID (required unless already set by ADC)
    FIRESTORE_DATABASE     Firestore database name (default: "(default)")
    STORAGE_BACKEND        Must NOT be set to "sqlite" for this script to make
                           sense, but the script doesn't enforce this — it
                           always writes to Firestore regardless.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fantasy_coach.storage.firestore import FirestoreRepository
from fantasy_coach.storage.sqlite import SQLiteRepository

logger = logging.getLogger(__name__)


def _all_seasons(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT season FROM matches ORDER BY season").fetchall()
    return [r[0] for r in rows]


def migrate(
    db_path: Path,
    seasons: list[int] | None,
    project: str | None,
    database: str,
) -> int:
    src = SQLiteRepository(db_path)
    dst = FirestoreRepository(project=project, database=database)

    season_list = seasons or _all_seasons(db_path)
    if not season_list:
        logger.warning("No seasons found in %s — nothing to migrate.", db_path)
        return 0

    total = errors = 0
    for season in season_list:
        matches = src.list_matches(season)
        logger.info("Season %d: migrating %d matches …", season, len(matches))
        for match in matches:
            try:
                dst.upsert_match(match)
                total += 1
            except Exception:
                logger.exception("Failed to migrate match_id=%d", match.match_id)
                errors += 1

    src.close()
    logger.info("Done — migrated %d, errors %d.", total, errors)
    return 0 if errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db", type=Path, default=Path("data/nrl.db"), help="Source SQLite DB path"
    )
    parser.add_argument(
        "--seasons",
        help="Comma-separated list of seasons to migrate (default: all)",
    )
    parser.add_argument("--project", help="GCP project ID (defaults to ADC project)")
    parser.add_argument(
        "--database",
        default="(default)",
        help="Firestore database name (default: '(default)')",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s")

    seasons = [int(s.strip()) for s in args.seasons.split(",")] if args.seasons else None
    return migrate(args.db, seasons, args.project, args.database)


if __name__ == "__main__":
    sys.exit(main())
