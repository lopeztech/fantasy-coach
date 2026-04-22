"""Skellam/Poisson margin model for NRL score prediction.

Models home and away scores as independent Poisson processes (λ_home, λ_away),
each fit via a Poisson GLM with log-link using the existing feature set.
The score difference follows a Skellam(λ_home, λ_away) distribution, which
gives:
  - P(home win)  = P(margin > 0) = sum of Skellam PMF over margins 1..∞
  - Predicted margin = E[home_score - away_score] = λ_home − λ_away
  - 95% CI from the symmetric 2.5/97.5 percentiles of the distribution

Feature convention: all features are (home − away) differences, so the
same feature matrix is used for both Poisson sub-models; the away model
sees the negated matrix so positive feature values consistently correspond
to "better away team" from its perspective.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market", Applied Statistics 46(2).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import skellam as skellam_dist
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder
from fantasy_coach.features import MatchRow
from fantasy_coach.models.elo import Elo

_MARGIN_LO = -40
_MARGIN_HI = 40
_MARGINS = np.arange(_MARGIN_LO, _MARGIN_HI + 1)  # shape (81,)


# ---------------------------------------------------------------------------
# Training frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkellamTrainingFrame:
    """Feature matrix paired with home + away score targets."""

    X: np.ndarray
    y_home: np.ndarray  # float, shape (n,)
    y_away: np.ndarray  # float, shape (n,)
    match_ids: np.ndarray
    start_times: np.ndarray  # dtype datetime64[s]
    feature_names: tuple[str, ...] = field(default_factory=lambda: FEATURE_NAMES)


def build_skellam_frame(
    matches: Iterable[MatchRow],
    *,
    elo: Elo | None = None,
) -> SkellamTrainingFrame:
    """Compute feature rows and score targets for all completed matches.

    Unlike ``build_training_frame``, draws are *included* — they're valid
    scoring observations for the Poisson sub-models even though logistic
    regression excludes them (binary target).
    """
    completed = sorted(
        (
            m
            for m in matches
            if m.match_state in {"FullTime", "FullTimeED"}
            and m.home.score is not None
            and m.away.score is not None
        ),
        key=lambda m: (m.start_time, m.match_id),
    )

    builder = FeatureBuilder(elo=elo)
    rows: list[list[float]] = []
    y_home: list[float] = []
    y_away: list[float] = []
    match_ids: list[int] = []
    start_times: list[np.datetime64] = []

    for match in completed:
        builder.advance_season_if_needed(match)
        rows.append(builder.feature_row(match))
        y_home.append(float(match.home.score))  # type: ignore[arg-type]
        y_away.append(float(match.away.score))  # type: ignore[arg-type]
        match_ids.append(match.match_id)
        start_times.append(np.datetime64(match.start_time.replace(tzinfo=None), "s"))
        builder.record(match)

    if not rows:
        return SkellamTrainingFrame(
            X=np.zeros((0, len(FEATURE_NAMES))),
            y_home=np.zeros((0,), dtype=float),
            y_away=np.zeros((0,), dtype=float),
            match_ids=np.zeros((0,), dtype=int),
            start_times=np.zeros((0,), dtype="datetime64[s]"),
        )

    return SkellamTrainingFrame(
        X=np.asarray(rows, dtype=float),
        y_home=np.asarray(y_home, dtype=float),
        y_away=np.asarray(y_away, dtype=float),
        match_ids=np.asarray(match_ids, dtype=int),
        start_times=np.asarray(start_times, dtype="datetime64[s]"),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarginDistribution:
    """Prediction output for a single match."""

    lambda_home: float
    lambda_away: float
    pmf: np.ndarray  # shape (81,), margins _MARGIN_LO .. _MARGIN_HI
    mean: float  # E[home_score - away_score]
    ci_95: tuple[int, int]  # (lo, hi) inclusive at the 2.5/97.5 percentiles
    home_win_prob: float


class SkellamModel:
    """Two-Poisson score model."""

    def __init__(
        self,
        home_pipeline: Pipeline,
        away_pipeline: Pipeline,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        self.home_pipeline = home_pipeline
        self.away_pipeline = away_pipeline
        self.feature_names = feature_names

    def predict_margin_distribution(self, x: np.ndarray) -> MarginDistribution:
        """Return the full Skellam margin distribution for a single match.

        x: (1, n_features) pre-kickoff feature row (home − away convention).
        """
        lh = float(self.home_pipeline.predict(x)[0])
        la = float(self.away_pipeline.predict(-x)[0])
        # Clamp to avoid numerical issues with near-zero rates.
        lh = max(lh, 0.1)
        la = max(la, 0.1)

        pmf = skellam_dist.pmf(_MARGINS, lh, la).astype(float)
        total = pmf.sum()
        if total > 0:
            pmf /= total

        home_win_prob = float(pmf[_MARGINS > 0].sum())
        mean_margin = float((pmf * _MARGINS).sum())

        cdf = np.cumsum(pmf)
        lo_idx = min(int(np.searchsorted(cdf, 0.025)), len(_MARGINS) - 1)
        hi_idx = min(int(np.searchsorted(cdf, 0.975)), len(_MARGINS) - 1)

        return MarginDistribution(
            lambda_home=lh,
            lambda_away=la,
            pmf=pmf,
            mean=mean_margin,
            ci_95=(int(_MARGINS[lo_idx]), int(_MARGINS[hi_idx])),
            home_win_prob=home_win_prob,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkellamTrainResult:
    model: SkellamModel
    feature_names: tuple[str, ...]
    n_train: int


def train_skellam(
    frame: SkellamTrainingFrame,
    *,
    alpha: float = 200.0,
) -> SkellamTrainResult:
    """Fit two Poisson GLMs on the training frame.

    The home model uses features as-is; the away model sees negated features
    so that positive (home − away) features consistently imply *lower* away
    scoring rate, matching human intuition and aiding convergence.
    """
    if frame.X.shape[0] < 10:
        raise ValueError(f"Need at least 10 rows to train; got {frame.X.shape[0]}")

    home_pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("glm", PoissonRegressor(alpha=alpha, max_iter=500)),
        ]
    )
    away_pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("glm", PoissonRegressor(alpha=alpha, max_iter=500)),
        ]
    )

    home_pipeline.fit(frame.X, frame.y_home)
    away_pipeline.fit(-frame.X, frame.y_away)

    model = SkellamModel(
        home_pipeline=home_pipeline,
        away_pipeline=away_pipeline,
        feature_names=frame.feature_names,
    )
    return SkellamTrainResult(model=model, feature_names=frame.feature_names, n_train=len(frame.X))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_skellam(path: Path | str, result: SkellamTrainResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "skellam",
            "home_pipeline": result.model.home_pipeline,
            "away_pipeline": result.model.away_pipeline,
            "feature_names": result.feature_names,
        },
        path,
    )


def load_skellam(path: Path | str) -> SkellamModel:
    blob = joblib.load(Path(path))
    if blob.get("model_type") != "skellam":
        raise ValueError(f"Expected model_type='skellam', got {blob.get('model_type')!r}")
    return SkellamModel(
        home_pipeline=blob["home_pipeline"],
        away_pipeline=blob["away_pipeline"],
        feature_names=tuple(blob["feature_names"]),
    )
