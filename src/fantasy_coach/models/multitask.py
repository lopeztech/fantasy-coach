"""Multi-task XGBoost model: joint winner / margin / total prediction (#215).

Trains two XGBoost models that share the same input feature set:

1. **Winner classifier** (``XGBClassifier``) — binary home-win probability.
2. **Joint regressor** (``XGBRegressor`` with ``multi_strategy='multi_output_tree'``)
   — outputs [margin, total_points] simultaneously, using one set of trees that
   splits on all features and is optimised jointly for both regression targets.

The joint tree structure captures the correlation between margin and total
(a dominant home team scores more AND concedes less → both targets move
together).  Predictions are post-hoc forced to be *coherent*: the
``predicted_winner`` always agrees with the sign of ``predicted_margin``
(ties within a configurable dead-band go to the winner classifier).

Satisfies the ``Model`` protocol via ``predict_home_win_prob`` so it can be
loaded from a joblib artefact by the standard ``loader.load_model`` dispatch.

References
----------
Caruana, R. (1997). Multitask Learning. *Machine Learning*, 28(1), 41–75.
XGBoost multi-output regression:
    https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder
from fantasy_coach.features import MatchRow
from fantasy_coach.models.elo import Elo

# Margin dead-band: when |predicted_margin| < this value the winner head
# takes precedence over the sign of the margin (avoid flipping near-draw
# predictions on tiny margin estimates that may be noise).
_MARGIN_DEADBAND = 2.0


# ---------------------------------------------------------------------------
# Training frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiTaskFrame:
    """Feature matrix paired with three supervised targets."""

    X: np.ndarray  # shape (n, n_features)
    y_winner: np.ndarray  # binary float: 1.0 = home won, 0.0 = away won
    y_margin: np.ndarray  # float: home_score − away_score
    y_total: np.ndarray  # float: home_score + away_score
    match_ids: np.ndarray
    start_times: np.ndarray  # dtype datetime64[s]
    feature_names: tuple[str, ...] = field(default_factory=lambda: FEATURE_NAMES)


def build_multitask_frame(
    matches: Iterable[MatchRow],
    *,
    elo: Elo | None = None,
) -> MultiTaskFrame:
    """Build a multi-task training frame from completed, non-draw matches.

    Draws are excluded (as in the binary winner training frame) so the winner
    head is trained on unambiguous outcomes only.
    """
    completed_non_draw = sorted(
        (
            m
            for m in matches
            if m.match_state in {"FullTime", "FullTimeED"}
            and m.home.score is not None
            and m.away.score is not None
            and m.home.score != m.away.score
        ),
        key=lambda m: (m.start_time, m.match_id),
    )

    builder = FeatureBuilder(elo=elo)
    rows: list[list[float]] = []
    y_win: list[float] = []
    y_mar: list[float] = []
    y_tot: list[float] = []
    match_ids: list[int] = []
    start_times: list[np.datetime64] = []

    for match in completed_non_draw:
        builder.advance_season_if_needed(match)
        rows.append(builder.feature_row(match))
        hs = float(match.home.score)  # type: ignore[arg-type]
        as_ = float(match.away.score)  # type: ignore[arg-type]
        y_win.append(1.0 if hs > as_ else 0.0)
        y_mar.append(hs - as_)
        y_tot.append(hs + as_)
        match_ids.append(match.match_id)
        start_times.append(np.datetime64(match.start_time.replace(tzinfo=None), "s"))
        builder.record(match)

    n = len(rows)
    if n == 0:
        empty = np.zeros((0,), dtype=float)
        return MultiTaskFrame(
            X=np.zeros((0, len(FEATURE_NAMES))),
            y_winner=empty,
            y_margin=empty,
            y_total=empty,
            match_ids=np.zeros((0,), dtype=int),
            start_times=np.zeros((0,), dtype="datetime64[s]"),
        )

    return MultiTaskFrame(
        X=np.asarray(rows, dtype=float),
        y_winner=np.asarray(y_win, dtype=float),
        y_margin=np.asarray(y_mar, dtype=float),
        y_total=np.asarray(y_tot, dtype=float),
        match_ids=np.asarray(match_ids, dtype=int),
        start_times=np.asarray(start_times, dtype="datetime64[s]"),
    )


# ---------------------------------------------------------------------------
# Multi-task prediction output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiTaskPrediction:
    """Coherent four-field prediction from the multi-task model."""

    home_win_prob: float  # P(home wins) — coherent with margin sign
    predicted_margin: float  # E[home_score − away_score]
    predicted_total: float  # E[home_score + away_score]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MultiTaskXGBoostModel:
    """Multi-task XGBoost model satisfying the standard Model protocol.

    Maintains a winner classifier and a joint margin+total regressor.
    Predictions are post-hoc coherenced so ``predicted_winner`` always
    agrees with ``sign(predicted_margin)`` outside the dead-band.
    """

    def __init__(
        self,
        winner_clf: XGBClassifier,
        joint_reg: XGBRegressor,
        margin_deadband: float = _MARGIN_DEADBAND,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        self.winner_clf = winner_clf
        self.joint_reg = joint_reg
        self.margin_deadband = margin_deadband
        self.feature_names = feature_names

    def predict(self, X: np.ndarray) -> list[MultiTaskPrediction]:
        """Return multi-task predictions for each row in X."""
        raw_win_prob = self.winner_clf.predict_proba(X)[:, 1]
        joint = self.joint_reg.predict(X)  # shape (n, 2): [margin, total]
        pred_margin = joint[:, 0]
        pred_total = joint[:, 1]

        results: list[MultiTaskPrediction] = []
        for i in range(len(X)):
            prob = float(raw_win_prob[i])
            mar = float(pred_margin[i])
            tot = float(pred_total[i])

            # Coherence: outside the dead-band, winner must agree with margin sign.
            if abs(mar) >= self.margin_deadband:
                if mar > 0 and prob < 0.5:
                    prob = 0.5 + (1.0 - prob) * 0.1  # nudge to home-favoured side
                elif mar < 0 and prob >= 0.5:
                    prob = 0.5 - prob * 0.1

            # Clip to valid probability range
            prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
            results.append(
                MultiTaskPrediction(
                    home_win_prob=prob,
                    predicted_margin=mar,
                    predicted_total=max(tot, 0.0),
                )
            )
        return results

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray:
        """Return P(home wins) per row. Satisfies Model protocol."""
        return np.array([p.home_win_prob for p in self.predict(X)])

    def coherence_fraction(self, X: np.ndarray) -> float:
        """Fraction of predictions where winner and margin sign agree."""
        preds = self.predict(X)
        coherent = sum((p.home_win_prob >= 0.5) == (p.predicted_margin >= 0) for p in preds)
        return coherent / len(preds) if preds else 1.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiTaskTrainResult:
    model: MultiTaskXGBoostModel
    feature_names: tuple[str, ...]
    n_train: int


def train_multitask(
    frame: MultiTaskFrame,
    *,
    winner_params: dict | None = None,
    regressor_params: dict | None = None,
) -> MultiTaskTrainResult:
    """Fit the multi-task XGBoost model.

    Parameters
    ----------
    frame:
        Training data from ``build_multitask_frame``.
    winner_params:
        Optional XGBClassifier kwargs (override defaults).
    regressor_params:
        Optional XGBRegressor kwargs (override defaults).
    """
    if frame.X.shape[0] < 10:
        raise ValueError(f"Need at least 10 rows to train; got {frame.X.shape[0]}")

    clf_defaults: dict = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": 1,
        "use_label_encoder": False,
    }
    reg_defaults: dict = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "multi_strategy": "multi_output_tree",
        "verbosity": 0,
        "n_jobs": 1,
    }

    clf_params = {**clf_defaults, **(winner_params or {})}
    reg_params = {**reg_defaults, **(regressor_params or {})}

    winner_clf = XGBClassifier(**clf_params)
    joint_reg = XGBRegressor(**reg_params)

    winner_clf.fit(frame.X, frame.y_winner)
    # Joint regressor targets: margin and total stacked as columns
    Y_reg = np.stack([frame.y_margin, frame.y_total], axis=1)
    joint_reg.fit(frame.X, Y_reg)

    model = MultiTaskXGBoostModel(
        winner_clf=winner_clf,
        joint_reg=joint_reg,
        feature_names=frame.feature_names,
    )
    return MultiTaskTrainResult(
        model=model,
        feature_names=frame.feature_names,
        n_train=frame.X.shape[0],
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_multitask(path: Path | str, result: MultiTaskTrainResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "multitask",
            "winner_clf": result.model.winner_clf,
            "joint_reg": result.model.joint_reg,
            "margin_deadband": result.model.margin_deadband,
            "feature_names": list(result.feature_names),
        },
        path,
    )


def load_multitask(path: Path | str) -> MultiTaskXGBoostModel:
    blob = joblib.load(Path(path))
    if blob.get("model_type") != "multitask":
        raise ValueError(f"Expected model_type='multitask', got {blob.get('model_type')!r}")
    return _from_blob(blob)


def _from_blob(blob: dict) -> MultiTaskXGBoostModel:
    return MultiTaskXGBoostModel(
        winner_clf=blob["winner_clf"],
        joint_reg=blob["joint_reg"],
        margin_deadband=float(blob.get("margin_deadband", _MARGIN_DEADBAND)),
        feature_names=tuple(blob["feature_names"]),
    )
