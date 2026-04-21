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
from fantasy_coach.models.calibration import CalibrationMethod, CalibrationWrapper
from fantasy_coach.models.elo import Elo
from fantasy_coach.models.ensemble import (
    EnsembleMode,
    EnsembleResult,
    fit_ensemble,
    predict_ensemble,
)
from fantasy_coach.models.logistic import TrainResult, train_logistic
from fantasy_coach.models.xgboost_model import train_xgboost


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


class CalibratedLogisticPredictor:
    """LogReg predictor with Platt-scaling calibration on a held-out fold.

    Splits each round's available history into:
    - first 80% (chronological) → base model training
    - last 20% → calibration fitting

    Falls back to the uncalibrated prediction when there are fewer than 20
    rows of history (not enough for a meaningful calibration split).
    """

    name = "logistic+cal"

    def __init__(self, method: CalibrationMethod = "platt") -> None:
        self._method = method
        self._train_result: TrainResult | None = None
        self._calibration_wrapper: CalibrationWrapper | None = None
        self._inference_builder = FeatureBuilder()

    def fit(self, history: Sequence[MatchRow]) -> None:
        frame = build_training_frame(history)
        if frame.X.shape[0] < 20:
            self._train_result = None
            self._calibration_wrapper = None
        else:
            n = frame.X.shape[0]
            order = np.argsort(frame.start_times)
            X = frame.X[order]
            y = frame.y[order]

            split = int(n * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_cal, y_cal = X[split:], y[split:]

            # Train base model on the 80% training partition.
            self._train_result = train_logistic(
                frame.__class__(
                    X=X_train,
                    y=y_train,
                    match_ids=frame.match_ids[order][:split],
                    start_times=frame.start_times[order][:split],
                    feature_names=frame.feature_names,
                ),
                test_fraction=0.0,
            )

            # Fit calibrator on the held-out 20%.
            self._calibration_wrapper = CalibrationWrapper(
                self._train_result.pipeline, method=self._method
            )
            self._calibration_wrapper.fit(X_cal, y_cal)

        self._inference_builder = FeatureBuilder()
        for match in sorted(history, key=lambda m: (m.start_time, m.match_id)):
            if match.home.score is None or match.away.score is None:
                continue
            self._inference_builder.advance_season_if_needed(match)
            self._inference_builder.record(match)

    def predict_home_win_prob(self, match: MatchRow) -> float:
        if self._train_result is None:
            return 0.55
        x = np.asarray([self._inference_builder.feature_row(match)], dtype=float)
        if self._calibration_wrapper is not None and self._calibration_wrapper.is_fitted:
            return float(self._calibration_wrapper.predict_home_win_prob(x)[0])
        return float(self._train_result.pipeline.predict_proba(x)[0, 1])


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


class XGBoostPredictor:
    name = "xgboost"

    def __init__(self) -> None:
        self._train_result = None
        self._inference_builder = FeatureBuilder()

    def fit(self, history: Sequence[MatchRow]) -> None:
        frame = build_training_frame(history)
        if frame.X.shape[0] < 10:
            self._train_result = None
        else:
            self._train_result = train_xgboost(frame, test_fraction=0.0)

        self._inference_builder = FeatureBuilder()
        for match in sorted(history, key=lambda m: (m.start_time, m.match_id)):
            if match.home.score is None or match.away.score is None:
                continue
            self._inference_builder.advance_season_if_needed(match)
            self._inference_builder.record(match)

    def predict_home_win_prob(self, match: MatchRow) -> float:
        if self._train_result is None:
            return 0.55
        x = np.asarray([self._inference_builder.feature_row(match)], dtype=float)
        proba = self._train_result.estimator.predict_proba(x)[0, 1]
        return float(proba)


_MIN_ENSEMBLE_ROWS = 30  # minimum completed matches needed to fit an ensemble layer
_ENSEMBLE_SPLIT = 0.75  # fraction of history used to train the base models


class EnsemblePredictor:
    """Blend of Elo, LogReg, and XGBoost via a stacked meta-learner.

    Training protocol (chronological, no future-leak):
    1. Fit all three base models on the first ``_ENSEMBLE_SPLIT`` (75 %) of
       completed history.
    2. Generate out-of-sample probabilities for the remaining 25 % using
       those base models.
    3. Fit the ensemble layer on those probabilities.
    4. Re-fit all base models on 100 % of completed history for inference.

    Kill switch: if the ensemble's cross-validated log-loss does not beat the
    best single base model by 0.5 pp, ``predict_home_win_prob`` falls back to
    that best base model.
    """

    name = "ensemble"

    def __init__(self, mode: EnsembleMode = "stacked") -> None:
        self._mode = mode
        self._elo = EloPredictor()
        self._logistic = LogisticPredictor()
        self._xgboost = XGBoostPredictor()
        self._ensemble_result: EnsembleResult | None = None

    def fit(self, history: Sequence[MatchRow]) -> None:
        completed = sorted(
            [m for m in history if m.home.score is not None and m.away.score is not None],
            key=lambda m: (m.start_time, m.match_id),
        )
        n = len(completed)

        if n < _MIN_ENSEMBLE_ROWS:
            # Not enough data — fit base models only; ensemble falls back to Elo.
            for base in (self._elo, self._logistic, self._xgboost):
                base.fit(history)
            self._ensemble_result = None
            return

        split = int(n * _ENSEMBLE_SPLIT)
        base_history = completed[:split]
        ens_matches = completed[split:]

        # Fit base models on the first 75 % of history.
        for base in (self._elo, self._logistic, self._xgboost):
            base.fit(base_history)

        # Collect held-out probabilities for the remaining 25 %.
        probs = np.column_stack(
            [
                [self._elo.predict_home_win_prob(m) for m in ens_matches],
                [self._logistic.predict_home_win_prob(m) for m in ens_matches],
                [self._xgboost.predict_home_win_prob(m) for m in ens_matches],
            ]
        )
        y = np.array(
            [int(m.home.score > m.away.score) for m in ens_matches],  # type: ignore[operator]
            dtype=int,
        )

        if len(np.unique(y)) >= 2:
            self._ensemble_result = fit_ensemble(
                probs, y, ["elo", "logistic", "xgboost"], mode=self._mode
            )
        else:
            self._ensemble_result = None

        # Re-fit base models on the full history so inference is up-to-date.
        for base in (self._elo, self._logistic, self._xgboost):
            base.fit(history)

    def predict_home_win_prob(self, match: MatchRow) -> float:
        if self._ensemble_result is None:
            # Fall back to Elo (best single model on this dataset).
            return self._elo.predict_home_win_prob(match)

        probs = np.array(
            [
                [
                    self._elo.predict_home_win_prob(match),
                    self._logistic.predict_home_win_prob(match),
                    self._xgboost.predict_home_win_prob(match),
                ]
            ]
        )
        return float(predict_ensemble(probs, self._ensemble_result)[0])
