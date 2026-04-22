"""Tests for the ``copy-matches-to-firestore`` CLI subcommand (#123).

One-off bootstrap that fills a prod Firestore with history from a local
SQLite backfill. Real Firestore writes are mocked; we just verify the
CLI wiring + per-season counting.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fantasy_coach.__main__ import main
from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.storage.sqlite import SQLiteRepository


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """Pre-seed a SQLite DB with a handful of matches across two seasons."""
    path = tmp_path / "nrl.db"
    repo = SQLiteRepository(path)
    try:
        for i, (season, round_) in enumerate([(2024, 1), (2024, 2), (2025, 1)]):
            repo.upsert_match(
                MatchRow(
                    match_id=1000 + i,
                    season=season,
                    round=round_,
                    start_time=datetime(season, 3, round_, 19, 30, tzinfo=UTC),
                    match_state="FullTime",
                    venue="Eden Park",
                    venue_city="Auckland",
                    weather=None,
                    home=TeamRow(
                        team_id=10, name="Tigers", nick_name="Tigers", score=20, players=[]
                    ),
                    away=TeamRow(
                        team_id=20, name="Raiders", nick_name="Raiders", score=14, players=[]
                    ),
                    team_stats=[],
                )
            )
    finally:
        repo.close()
    return path


def test_copy_all_seasons_writes_every_match(seeded_db: Path) -> None:
    fake_repo = MagicMock()
    with patch(
        "fantasy_coach.storage.firestore.FirestoreRepository",
        return_value=fake_repo,
    ) as factory:
        code = main(["copy-matches-to-firestore", "--db", str(seeded_db), "--log-level", "WARNING"])

    assert code == 0
    factory.assert_called_once()  # single Firestore client
    assert fake_repo.upsert_match.call_count == 3
    # All three matches should reach Firestore in season order.
    match_ids = [c.args[0].match_id for c in fake_repo.upsert_match.call_args_list]
    assert sorted(match_ids) == [1000, 1001, 1002]


def test_copy_filtered_by_season(seeded_db: Path) -> None:
    fake_repo = MagicMock()
    with patch(
        "fantasy_coach.storage.firestore.FirestoreRepository",
        return_value=fake_repo,
    ):
        code = main(
            [
                "copy-matches-to-firestore",
                "--db",
                str(seeded_db),
                "--season",
                "2024",
                "--log-level",
                "WARNING",
            ]
        )

    assert code == 0
    # Only the two 2024 matches should be copied.
    assert fake_repo.upsert_match.call_count == 2


def test_copy_dry_run_does_not_instantiate_firestore(seeded_db: Path) -> None:
    with patch("fantasy_coach.storage.firestore.FirestoreRepository") as factory:
        code = main(
            [
                "copy-matches-to-firestore",
                "--db",
                str(seeded_db),
                "--dry-run",
                "--log-level",
                "WARNING",
            ]
        )

    assert code == 0
    factory.assert_not_called()


def test_copy_missing_db_is_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    code = main(["copy-matches-to-firestore", "--db", str(missing), "--log-level", "WARNING"])
    assert code == 2
