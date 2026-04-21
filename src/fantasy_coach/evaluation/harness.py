"""Walk-forward evaluation: refit per round, predict that round, score.

For each (season, round) in evaluation order:
1. Build the history = every completed match strictly before this round.
2. `predictor.fit(history)`.
3. For every match in this round, call `predictor.predict_home_win_prob(m)`
   and store (match_id, p_home_win, actual).

Yields one `EvaluationResult` per (predictor, run) so the report writer can
diff metrics across models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from fantasy_coach.evaluation.metrics import accuracy, brier_score, log_loss
from fantasy_coach.evaluation.predictors import Predictor
from fantasy_coach.features import MatchRow
from fantasy_coach.storage.repository import Repository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    season: int
    round: int
    match_id: int
    home_id: int
    away_id: int
    p_home_win: float
    actual_home_win: int


@dataclass
class EvaluationResult:
    predictor_name: str
    predictions: list[Prediction] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.predictions)

    @property
    def probs(self) -> list[float]:
        return [p.p_home_win for p in self.predictions]

    @property
    def actuals(self) -> list[int]:
        return [p.actual_home_win for p in self.predictions]

    def metrics(self) -> dict[str, float]:
        return {
            "accuracy": accuracy(self.probs, self.actuals),
            "log_loss": log_loss(self.probs, self.actuals),
            "brier": brier_score(self.probs, self.actuals),
        }


def walk_forward(
    matches_by_round: Sequence[tuple[int, int, list[MatchRow]]],
    predictor_factory: Callable[[], Predictor],
) -> EvaluationResult:
    """Score `predictor_factory()` over (season, round, matches) in order.

    Caller is responsible for ordering and grouping matches into rounds —
    `walk_forward_from_repo` is the convenience that does this from a
    `Repository`.
    """
    predictor = predictor_factory()
    history: list[MatchRow] = []
    result = EvaluationResult(predictor_name=predictor.name)

    for season, round_, round_matches in matches_by_round:
        rateable = [m for m in round_matches if _has_outcome(m)]
        if not rateable:
            logger.info("Skip %d/round %d: no completed matches", season, round_)
            continue

        predictor.fit(history)
        for match in rateable:
            if match.home.score == match.away.score:
                # Drop draws from scoring — binary metrics expect {0, 1}.
                continue
            p = predictor.predict_home_win_prob(match)
            result.predictions.append(
                Prediction(
                    season=season,
                    round=round_,
                    match_id=match.match_id,
                    home_id=match.home.team_id,
                    away_id=match.away.team_id,
                    p_home_win=p,
                    actual_home_win=1 if (match.home.score or 0) > (match.away.score or 0) else 0,
                )
            )
        history.extend(rateable)

    return result


def walk_forward_from_repo(
    repo: Repository,
    seasons: Iterable[int],
    predictor_factory: Callable[[], Predictor],
) -> EvaluationResult:
    grouped: list[tuple[int, int, list[MatchRow]]] = []
    for season in sorted(seasons):
        matches = sorted(repo.list_matches(season), key=lambda m: (m.start_time, m.match_id))
        rounds: dict[int, list[MatchRow]] = {}
        for match in matches:
            rounds.setdefault(match.round, []).append(match)
        for round_ in sorted(rounds.keys()):
            grouped.append((season, round_, rounds[round_]))
    return walk_forward(grouped, predictor_factory)


def _has_outcome(match: MatchRow) -> bool:
    return (
        match.match_state in {"FullTime", "FullTimeED"}
        and match.home.score is not None
        and match.away.score is not None
    )
