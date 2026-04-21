"""Adapters that wrap each model behind a common `Predictor` interface.

The walk-forward harness only knows about `Predictor.fit(matches)` and
`Predictor.predict_home_win_prob(match)`. New models slot in by adding an
adapter here — no harness changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from fantasy_coach.feature_engineering import (
    FeatureBuilder,
    build_training_frame,
)
from fantasy_coach.features import MatchRow
from fantasy_coach.models.elo import Elo
from fantasy_coach.models.logistic import train_logistic


class Predictor(Protocol):
    name: str

    def fit(self, history: Sequence[MatchRow]) -> None: ...

    def predict_home_win_prob(self, match: MatchRow) -> float: ...


class HomePickPredictor:
    """Trivial baseline — every prediction is `p_home_win = 0.5 + epsilon`.

    Useful as a sanity floor: any real model that does worse than this is
    actively miscalibrated, not just unlucky.
    """

    name = "home"

    def fit(self, history: Sequence[MatchRow]) -> None:  # noqa: ARG002
        return

    def predict_home_win_prob(self, match: MatchRow) -> float:  # noqa: ARG002
        return 0.55  # NRL home-win rate ≈ 55–58 % historically


class EloPredictor:
    name = "elo"

    def __init__(
        self,
        *,
        k: float | None = None,
        home_advantage: float | None = None,
        season_regression: float | None = None,
    ) -> None:
        kwargs: dict[str, float] = {}
        if k is not None:
            kwargs["k"] = k
        if home_advantage is not None:
            kwargs["home_advantage"] = home_advantage
        if season_regression is not None:
            kwargs["season_regression"] = season_regression
        self._kwargs = kwargs
        self._elo = Elo(**kwargs)

    def fit(self, history: Sequence[MatchRow]) -> None:
        # Rebuild from scratch so the harness can call fit() repeatedly with
        # an extending history without leaking later updates into earlier
        # predictions.
        self._elo = Elo(**self._kwargs)
        # `sweep_repository` consumes a Repository, but it just calls
        # `list_matches(season)`; for a clean in-memory rebuild, walk the
        # provided history directly.
        seasons = sorted({m.season for m in history})
        history_by_season = {s: [m for m in history if m.season == s] for s in seasons}
        for index, season in enumerate(seasons):
            if index > 0:
                self._elo.regress_to_mean()
            for match in sorted(
                history_by_season[season], key=lambda m: (m.start_time, m.match_id)
            ):
                if match.home.score is None or match.away.score is None:
                    continue
                self._elo.update(
                    match.home.team_id,
                    match.away.team_id,
                    int(match.home.score),
                    int(match.away.score),
                )

    def predict_home_win_prob(self, match: MatchRow) -> float:
        return self._elo.predict(match.home.team_id, match.away.team_id)

    @property
    def elo(self) -> Elo:
        return self._elo


class LogisticPredictor:
    name = "logistic"

    def __init__(self) -> None:
        self._train_result = None
        # Inference-time builder lets us score one match in O(1) instead of
        # rebuilding the entire training frame per prediction.
        self._inference_builder = FeatureBuilder()

    def fit(self, history: Sequence[MatchRow]) -> None:
        frame = build_training_frame(history)
        if frame.X.shape[0] < 10:
            self._train_result = None
        else:
            # No internal holdout — the walk-forward harness owns the split.
            self._train_result = train_logistic(frame, test_fraction=0.0)

        # Re-derive the inference-time feature state from history. We have
        # to walk it ourselves (rather than reuse the training builder)
        # because draws are dropped from the training frame but their
        # outcomes still belong in the rolling state.
        self._inference_builder = FeatureBuilder()
        for match in sorted(history, key=lambda m: (m.start_time, m.match_id)):
            if match.home.score is None or match.away.score is None:
                continue
            self._inference_builder.advance_season_if_needed(match)
            self._inference_builder.record(match)

    def predict_home_win_prob(self, match: MatchRow) -> float:
        if self._train_result is None:
            return 0.55  # too little history; fall back to home prior
        # advance_season_if_needed is a no-op here — `match` hasn't been
        # recorded yet, so the season transition is purely Elo regression
        # and would over-pull ratings if applied speculatively. Skip it
        # at inference time; the harness re-fits between rounds anyway.
        x = np.asarray([self._inference_builder.feature_row(match)], dtype=float)
        proba = self._train_result.pipeline.predict_proba(x)[0, 1]
        return float(proba)
