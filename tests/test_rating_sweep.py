from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.models.elo import DEFAULT_INITIAL_RATING, Elo
from fantasy_coach.models.rating_sweep import sweep_repository
from fantasy_coach.storage import SQLiteRepository


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SQLiteRepository]:
    db = SQLiteRepository(tmp_path / "ratings.db")
    yield db
    db.close()


def _make_match(
    *,
    match_id: int,
    season: int,
    round: int,
    start: datetime,
    home_id: int,
    away_id: int,
    home_score: int | None,
    away_score: int | None,
    state: str = "FullTime",
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round,
        start_time=start,
        match_state=state,
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name=f"Team {home_id}",
            nick_name=str(home_id),
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=away_id,
            name=f"Team {away_id}",
            nick_name=str(away_id),
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


def test_sweep_applies_completed_matches_in_order(repo: SQLiteRepository) -> None:
    repo.upsert_match(
        _make_match(
            match_id=1,
            season=2024,
            round=1,
            start=datetime(2024, 3, 1, tzinfo=UTC),
            home_id=10,
            away_id=20,
            home_score=30,
            away_score=12,
        )
    )
    repo.upsert_match(
        _make_match(
            match_id=2,
            season=2024,
            round=2,
            start=datetime(2024, 3, 8, tzinfo=UTC),
            home_id=20,
            away_id=10,
            home_score=6,
            away_score=24,
        )
    )

    elo = sweep_repository(repo, [2024])

    # Team 10 won both matches → should be highest-rated.
    ratings = elo.ratings()
    assert ratings[10] > DEFAULT_INITIAL_RATING
    assert ratings[20] < DEFAULT_INITIAL_RATING


def test_sweep_skips_unfinished_matches(repo: SQLiteRepository) -> None:
    repo.upsert_match(
        _make_match(
            match_id=1,
            season=2024,
            round=1,
            start=datetime(2024, 3, 1, tzinfo=UTC),
            home_id=10,
            away_id=20,
            home_score=None,
            away_score=None,
            state="Upcoming",
        )
    )

    elo = sweep_repository(repo, [2024])
    assert elo.ratings() == {}


def test_sweep_regresses_between_seasons(repo: SQLiteRepository) -> None:
    # Team 10 dominates in 2023; new season should pull it back toward 1500.
    repo.upsert_match(
        _make_match(
            match_id=1,
            season=2023,
            round=1,
            start=datetime(2023, 3, 1, tzinfo=UTC),
            home_id=10,
            away_id=20,
            home_score=40,
            away_score=0,
        )
    )
    repo.upsert_match(
        _make_match(
            match_id=2,
            season=2024,
            round=1,
            start=datetime(2024, 3, 1, tzinfo=UTC),
            home_id=10,
            away_id=20,
            home_score=24,
            away_score=18,
        )
    )

    no_regress = sweep_repository(repo, [2023, 2024], elo=Elo(season_regression=0.0))
    with_regress = sweep_repository(repo, [2023, 2024], elo=Elo(season_regression=0.5))

    # Team 10's lead over Team 20 should be smaller when regression is on.
    spread_no = no_regress.rating(10) - no_regress.rating(20)
    spread_yes = with_regress.rating(10) - with_regress.rating(20)
    assert spread_yes < spread_no
