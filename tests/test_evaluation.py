from __future__ import annotations

import math
import random
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fantasy_coach.evaluation import (
    EloPredictor,
    HomePickPredictor,
    LogisticPredictor,
    accuracy,
    brier_score,
    log_loss,
    walk_forward,
)
from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.evaluation.report import render_markdown
from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.storage import SQLiteRepository


def _match(
    *,
    match_id: int,
    season: int,
    round: int,
    home_id: int,
    away_id: int,
    home_score: int,
    away_score: int,
    when: datetime,
    state: str = "FullTime",
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round,
        start_time=when,
        match_state=state,
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name=str(home_id),
            nick_name=str(home_id),
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=away_id,
            name=str(away_id),
            nick_name=str(away_id),
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


# ---------- metrics ----------


def test_accuracy_threshold_at_half() -> None:
    assert accuracy([0.6, 0.4, 0.51], [1, 0, 1]) == 1.0
    assert accuracy([0.6, 0.4, 0.49], [1, 0, 1]) == 2 / 3


def test_log_loss_perfect_is_near_zero_and_wrong_is_huge() -> None:
    assert log_loss([0.99], [1]) < 0.02
    assert log_loss([0.01], [1]) > 4.0


def test_brier_known_value() -> None:
    # ((0.7-1)^2 + (0.2-0)^2) / 2 = (0.09 + 0.04) / 2 = 0.065
    assert math.isclose(brier_score([0.7, 0.2], [1, 0]), 0.065, rel_tol=1e-9)


def test_metrics_handle_empty() -> None:
    assert math.isnan(accuracy([], []))
    assert math.isnan(log_loss([], []))
    assert math.isnan(brier_score([], []))


# ---------- walk-forward harness ----------


def _two_round_season() -> list[tuple[int, int, list[MatchRow]]]:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    round1 = [
        _match(
            match_id=i,
            season=2024,
            round=1,
            home_id=10 + i * 2,
            away_id=11 + i * 2,
            home_score=24,
            away_score=12,
            when=base + timedelta(hours=i),
        )
        for i in range(4)
    ]
    round2 = [
        _match(
            match_id=100 + i,
            season=2024,
            round=2,
            home_id=10 + i * 2,
            away_id=11 + i * 2,
            home_score=20,
            away_score=14,
            when=base + timedelta(days=7, hours=i),
        )
        for i in range(4)
    ]
    return [(2024, 1, round1), (2024, 2, round2)]


def test_walk_forward_records_one_prediction_per_match() -> None:
    rounds = _two_round_season()
    result = walk_forward(rounds, HomePickPredictor)
    assert result.n == 8
    assert result.predictor_name == "home"
    # All predictions should be the constant 0.55.
    assert all(p.p_home_win == 0.55 for p in result.predictions)


def test_walk_forward_drops_draws_from_scoring() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    rounds = [
        (
            2024,
            1,
            [
                _match(
                    match_id=1,
                    season=2024,
                    round=1,
                    home_id=10,
                    away_id=20,
                    home_score=12,
                    away_score=12,
                    when=base,
                ),
                _match(
                    match_id=2,
                    season=2024,
                    round=1,
                    home_id=30,
                    away_id=40,
                    home_score=24,
                    away_score=18,
                    when=base + timedelta(hours=1),
                ),
            ],
        )
    ]
    result = walk_forward(rounds, HomePickPredictor)
    assert [p.match_id for p in result.predictions] == [2]


def test_walk_forward_uses_only_prior_rounds_for_training() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    # Round 1: team 10 beats team 20 by a lot.
    # Round 2: same matchup. EloPredictor should pick team 10 strongly.
    rounds = [
        (
            2024,
            1,
            [
                _match(
                    match_id=1,
                    season=2024,
                    round=1,
                    home_id=10,
                    away_id=20,
                    home_score=40,
                    away_score=10,
                    when=base,
                )
            ],
        ),
        (
            2024,
            2,
            [
                _match(
                    match_id=2,
                    season=2024,
                    round=2,
                    home_id=10,
                    away_id=20,
                    home_score=24,
                    away_score=18,
                    when=base + timedelta(days=7),
                )
            ],
        ),
    ]
    result = walk_forward(rounds, EloPredictor)
    # Round 1 prediction is from a fresh book → ~0.5 + home-advantage.
    # Round 2 prediction should be higher because team 10 just won by 30.
    p1, p2 = result.predictions
    assert p2.p_home_win > p1.p_home_win


def test_logistic_predictor_falls_back_when_history_too_small() -> None:
    base = datetime(2024, 3, 1, tzinfo=UTC)
    rounds = [
        (
            2024,
            1,
            [
                _match(
                    match_id=1,
                    season=2024,
                    round=1,
                    home_id=10,
                    away_id=20,
                    home_score=24,
                    away_score=12,
                    when=base,
                )
            ],
        )
    ]
    result = walk_forward(rounds, LogisticPredictor)
    assert result.predictions[0].p_home_win == 0.55


# ---------- end-to-end via SQLite repository ----------


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SQLiteRepository]:
    db = SQLiteRepository(tmp_path / "eval.db")
    yield db
    db.close()


def _stack_season(repo: SQLiteRepository) -> None:
    rng = random.Random(0)
    base = datetime(2024, 3, 1, tzinfo=UTC)
    for i in range(40):
        # Team 10 always wins; alternates home/away to get a real signal.
        if i % 2 == 0:
            home, away, hs, as_ = 10, 20, rng.randint(20, 36), rng.randint(6, 18)
        else:
            home, away, hs, as_ = 20, 10, rng.randint(6, 18), rng.randint(20, 36)
        repo.upsert_match(
            _match(
                match_id=i + 1,
                season=2024,
                round=(i // 8) + 1,
                home_id=home,
                away_id=away,
                home_score=hs,
                away_score=as_,
                when=base + timedelta(days=i),
            )
        )


def test_walk_forward_from_repo_orders_seasons_and_rounds(
    repo: SQLiteRepository,
) -> None:
    _stack_season(repo)
    result = walk_forward_from_repo(repo, [2024], EloPredictor)
    assert result.n > 0
    rounds_seen = [p.round for p in result.predictions]
    assert rounds_seen == sorted(rounds_seen)


def test_render_markdown_contains_metrics_for_each_model() -> None:
    rounds = _two_round_season()
    home = walk_forward(rounds, HomePickPredictor)
    elo = walk_forward(rounds, EloPredictor)

    md = render_markdown([home, elo], seasons=[2024])

    assert "| home |" in md
    assert "| elo |" in md
    assert "Accuracy" in md
    assert "Brier" in md
