"""Adapters that wrap each model behind a common `Predictor` interface.

The walk-forward harness only knows about `Predictor.fit(matches)` and
`Predictor.predict_home_win_prob(match)`. New models slot in by adding an
adapter here — no harness changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from fantasy_coach.feature_engineering import (
    FeatureBuilder,
    build_training_frame,
)
from fantasy_coach.features import MatchRow
from fantasy_coach.models.calibration import CalibrationMethod, CalibrationWrapper
from fantasy_coach.models.elo import Elo
from fantasy_coach.models.elo_mov import EloMOV
from fantasy_coach.models.ensemble import EnsembleMode, EnsembleModel, fit_ensemble
from fantasy_coach.models.logistic import TrainResult, train_logistic

# Import ``train_xgboost`` lazily inside the XGBoost predictors — loading
# xgboost eagerly pulls in libxgboost.dylib, which can't load on macOS
# without libomp installed. Lazy import keeps the rest of the module
# (Elo, logistic, ensemble adapter) importable in any environment.
# Skellam import is also deferred — it pulls in scipy which is heavier.


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


class EloMOVPredictor:
    """Walk-forward adapter for the MOV-weighted Elo rater.

    Drop-in replacement for ``EloPredictor`` — identical constructor kwargs
    and ``fit``/``predict_home_win_prob`` interface; uses ``EloMOV`` instead
    of plain ``Elo`` so the walk-forward harness can A/B the two directly.
    """

    name = "elo_mov"

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
        self._elo = EloMOV(**kwargs)

    def fit(self, history: Sequence[MatchRow]) -> None:
        self._elo = EloMOV(**self._kwargs)
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
    def elo(self) -> EloMOV:
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
        from fantasy_coach.models.xgboost_model import train_xgboost  # noqa: PLC0415

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


class CalibratedXGBoostPredictor:
    """XGBoost predictor with isotonic calibration on a held-out fold.

    Same 80/20 chronological split as ``CalibratedLogisticPredictor``:
    - first 80 % → base XGBoost training
    - last 20 % → isotonic calibrator fitting

    Isotonic (rather than Platt) because tree models tend to push
    probabilities toward 0/1 and need a non-linear correction; see
    ``fantasy_coach.models.calibration`` for the rationale.
    """

    name = "xgboost+cal"

    def __init__(self, method: CalibrationMethod = "isotonic") -> None:
        self._method = method
        self._train_result = None
        self._calibration_wrapper: CalibrationWrapper | None = None
        self._inference_builder = FeatureBuilder()

    def fit(self, history: Sequence[MatchRow]) -> None:
        frame = build_training_frame(history)
        if frame.X.shape[0] < 20:
            self._train_result = None
            self._calibration_wrapper = None
        else:
            order = np.argsort(frame.start_times)
            X = frame.X[order]
            y = frame.y[order]
            split = int(X.shape[0] * 0.8)

            from fantasy_coach.models.xgboost_model import train_xgboost as _train

            self._train_result = _train(
                frame.__class__(
                    X=X[:split],
                    y=y[:split],
                    match_ids=frame.match_ids[order][:split],
                    start_times=frame.start_times[order][:split],
                    feature_names=frame.feature_names,
                ),
                test_fraction=0.0,
            )

            # Wrap the fitted XGB estimator in a sklearn-compatible Pipeline
            # shim so CalibrationWrapper's ``predict_proba`` contract holds
            # without duplicating calibration logic.
            from sklearn.pipeline import Pipeline as _SkPipeline

            pipeline = _SkPipeline([("xgb", self._train_result.estimator)])
            self._calibration_wrapper = CalibrationWrapper(pipeline, method=self._method)
            self._calibration_wrapper.fit(X[split:], y[split:])

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
        return float(self._train_result.estimator.predict_proba(x)[0, 1])


class EnsemblePredictor:
    """Combine N base predictors via a ``weighted`` or ``stacked`` meta-layer.

    Per round, history is split 80/20 chronologically. Each base predictor
    is fitted on the first 80 %; their out-of-fold probabilities on the
    last 20 % feed ``fit_ensemble``, which learns either convex weights or
    a LogReg meta-learner. At inference the same base predictors (still
    trained on the 80 % slice) produce the input row for the ensemble.

    If fewer than ``min_meta_rows`` history rows are available, the
    predictor falls back to the first base predictor unchanged — the
    ensemble fit would be too noisy to trust with 5–10 samples.

    The kill switch from ``fit_ensemble`` is honoured: when the fitted
    ensemble can't beat the best base by ``min_improvement`` log-loss
    points, we route all predictions through that base predictor (and
    ``last_fit_info['fallback_to_base']`` records which one).
    """

    def __init__(
        self,
        base_factories: Sequence[Callable[[], Predictor]],
        *,
        mode: EnsembleMode = "weighted",
        name: str | None = None,
        min_meta_rows: int = 30,
    ) -> None:
        if not base_factories:
            raise ValueError("EnsemblePredictor needs at least one base predictor")
        self._base_factories = list(base_factories)
        self._mode: EnsembleMode = mode
        self._min_meta_rows = min_meta_rows
        self._bases: list[Predictor] = []
        self._ensemble: EnsembleModel | None = None
        self._disabled = False
        self.name = name or f"ensemble/{mode}"
        self.last_fit_info: dict[str, object] = {}

    def fit(self, history: Sequence[MatchRow]) -> None:
        rateable = [
            m
            for m in sorted(history, key=lambda m: (m.start_time, m.match_id))
            if m.home.score is not None and m.away.score is not None
        ]

        # Re-create base predictors from scratch every fit — they accumulate
        # Elo / feature-builder state internally and the harness calls fit
        # repeatedly with an extending history.
        self._bases = [factory() for factory in self._base_factories]

        if len(rateable) < self._min_meta_rows:
            # Not enough data to fit a meaningful meta-learner; degrade to
            # the first base predictor, fitted on everything we have.
            for base in self._bases:
                base.fit(rateable)
            self._ensemble = None
            self._disabled = True
            self.last_fit_info = {"disabled": True, "reason": "insufficient_history"}
            return

        split = int(len(rateable) * 0.8)
        base_train, meta_train = rateable[:split], rateable[split:]

        for base in self._bases:
            base.fit(base_train)

        # Collect OOF base probabilities on the held-out 20 %.
        base_probs = np.empty((len(meta_train), len(self._bases)), dtype=float)
        for j, base in enumerate(self._bases):
            for i, match in enumerate(meta_train):
                base_probs[i, j] = base.predict_home_win_prob(match)
        y = np.array([1 if (m.home.score or 0) > (m.away.score or 0) else 0 for m in meta_train])
        # Drop draws (binary metric contract); they'd skew weight fitting.
        draw_mask = np.array(
            [
                (m.home.score or 0) == (m.away.score or 0)
                and m.home.score is not None
                and m.away.score is not None
                for m in meta_train
            ]
        )
        if draw_mask.any():
            keep = ~draw_mask
            base_probs = base_probs[keep]
            y = y[keep]

        if base_probs.shape[0] < 5:
            # Post-draw-filter slice is too small — degrade as above.
            self._ensemble = None
            self._disabled = True
            self.last_fit_info = {"disabled": True, "reason": "insufficient_meta_rows"}
            return

        names = tuple(b.name for b in self._bases)
        self._ensemble = fit_ensemble(base_probs, y, mode=self._mode, base_model_names=names)
        self._disabled = False
        self.last_fit_info = {
            "disabled": False,
            "base_log_losses": dict(self._ensemble.base_log_losses),
            "ensemble_log_loss": self._ensemble.ensemble_log_loss,
            "fallback_to_base": self._ensemble.fallback_to_base,
            "mode": self._mode,
        }

    def predict_home_win_prob(self, match: MatchRow) -> float:
        if not self._bases:
            return 0.55
        if self._disabled or self._ensemble is None:
            return self._bases[0].predict_home_win_prob(match)
        probs = np.array([[base.predict_home_win_prob(match) for base in self._bases]], dtype=float)
        return float(self._ensemble.predict_home_win_prob(probs)[0])


class SkellamPredictor:
    """Walk-forward adapter for the two-Poisson Skellam margin model.

    Win probability is derived from the Skellam distribution so it is
    coherent with the predicted margin — the same λ_home / λ_away
    parameters drive both outputs.
    """

    name = "skellam"

    def __init__(self) -> None:
        self._train_result = None
        self._inference_builder = FeatureBuilder()

    def fit(self, history: Sequence[MatchRow]) -> None:
        from fantasy_coach.models.skellam import build_skellam_frame, train_skellam  # noqa: PLC0415

        frame = build_skellam_frame(history)
        if frame.X.shape[0] < 10:
            self._train_result = None
        else:
            self._train_result = train_skellam(frame)

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
        dist = self._train_result.model.predict_margin_distribution(x)
        return dist.home_win_prob
