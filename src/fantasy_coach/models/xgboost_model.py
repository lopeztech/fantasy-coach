"""XGBoost model for NRL match prediction.

Trains an XGBClassifier on the same feature set as the logistic baseline.
Hyperparameters are selected via time-series-aware CV (TimeSeriesSplit) to
avoid future-leak. Uses the same joblib artefact format as logistic.py so the
prediction API can swap models via config without code changes.

Comparison rule (see AC for #25): if XGBoost log_loss is not at least 1 point
better than logistic on the walk-forward baseline, keep logistic as the default
(simpler = fewer things to break). See docs/model.md for recorded comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from fantasy_coach.feature_engineering import FEATURE_NAMES, TrainingFrame

# Fixed hyperparameters not tuned (sensible defaults for small datasets).
_FIXED_PARAMS: dict[str, object] = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "verbosity": 0,
    "use_label_encoder": False,
}

# Grid searched via TimeSeriesSplit.
_PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
}

_CV_SPLITS = 3  # TimeSeriesSplit — small dataset needs small n_splits
_MIN_CV_ROWS = 50  # fall back to default params when training set is too small for CV


@dataclass(frozen=True)
class TrainResult:
    estimator: XGBClassifier
    feature_names: tuple[str, ...]
    best_params: dict[str, object]
    train_accuracy: float
    test_accuracy: float
    n_train: int
    n_test: int


def train_xgboost(
    frame: TrainingFrame,
    *,
    test_fraction: float = 0.2,
    random_state: int = 0,
) -> TrainResult:
    """Train on the chronologically earliest (1 − test_fraction) of ``frame``.

    Hyperparameters are tuned on the training split with TimeSeriesSplit CV,
    then the best configuration is refit on the full training split.
    """
    if frame.X.shape[0] < 10:
        raise ValueError(f"Need at least 10 rows to train; got {frame.X.shape[0]}")

    order = np.argsort(frame.start_times)
    X = frame.X[order]
    y = frame.y[order]

    split = int(len(X) * (1.0 - test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_train) >= _MIN_CV_ROWS:
        tscv = TimeSeriesSplit(n_splits=_CV_SPLITS)
        search = GridSearchCV(
            XGBClassifier(**_FIXED_PARAMS, random_state=random_state),
            _PARAM_GRID,
            scoring="neg_log_loss",
            cv=tscv,
            refit=True,
            n_jobs=-1,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)
        best: XGBClassifier = search.best_estimator_
        best_params = dict(search.best_params_)
    else:
        # Too few rows for reliable CV — use conservative fixed defaults.
        best_params = {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1}
        best = XGBClassifier(**_FIXED_PARAMS, **best_params, random_state=random_state)
        best.fit(X_train, y_train)

    train_acc = float(best.score(X_train, y_train))
    test_acc = float(best.score(X_test, y_test)) if len(X_test) else float("nan")

    return TrainResult(
        estimator=best,
        feature_names=frame.feature_names,
        best_params=best_params,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
    )


def save_model(result: TrainResult, path: Path | str) -> None:
    """Persist the trained estimator + feature-name ordering to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": result.estimator,
            "feature_names": result.feature_names,
            "model_type": "xgboost",
            "best_params": result.best_params,
        },
        path,
    )


@dataclass(frozen=True)
class LoadedModel:
    estimator: XGBClassifier
    feature_names: tuple[str, ...]

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features "
                f"({self.feature_names}), got {X.shape[1]}"
            )
        return self.estimator.predict_proba(X)[:, 1]


def _from_blob(blob: dict) -> LoadedModel:
    if blob.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError(
            f"Model trained with features {blob.get('feature_names')}, "
            f"current code expects {FEATURE_NAMES}. Retrain before loading."
        )
    return LoadedModel(
        estimator=blob["estimator"],
        feature_names=tuple(blob["feature_names"]),
    )


def load_model(path: Path | str) -> LoadedModel:
    return _from_blob(joblib.load(Path(path)))
