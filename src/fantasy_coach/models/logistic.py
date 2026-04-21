"""Logistic regression baseline.

Wraps `sklearn.LogisticRegression` plus a `StandardScaler` in a single
`Pipeline` so the same preprocessing applies at training and inference.
Saved with joblib; the artefact is self-describing (carries the feature-name
ordering used at fit time so callers can't pass features in the wrong order).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fantasy_coach.feature_engineering import FEATURE_NAMES, TrainingFrame


@dataclass(frozen=True)
class TrainResult:
    pipeline: Pipeline
    feature_names: tuple[str, ...]
    train_accuracy: float
    test_accuracy: float
    n_train: int
    n_test: int


def train_logistic(
    frame: TrainingFrame,
    *,
    test_fraction: float = 0.2,
    C: float = 1.0,
    random_state: int = 0,
) -> TrainResult:
    """Train on the chronologically earliest (1 - test_fraction) of `frame`.

    The train/test split is *time-ordered*, not random — the test set is
    always the most recent matches, mimicking how the model would be used
    in production (predict the upcoming round given everything before).
    """

    if frame.X.shape[0] < 10:
        raise ValueError(f"Need at least 10 rows to train; got {frame.X.shape[0]}")

    order = np.argsort(frame.start_times)
    X = frame.X[order]
    y = frame.y[order]

    split = int(len(X) * (1.0 - test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    return TrainResult(
        pipeline=pipeline,
        feature_names=frame.feature_names,
        train_accuracy=float(pipeline.score(X_train, y_train)),
        test_accuracy=float(pipeline.score(X_test, y_test)),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
    )


def save_model(result: TrainResult, path: Path | str) -> None:
    """Persist the trained pipeline + feature-name ordering to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": result.pipeline, "feature_names": result.feature_names},
        path,
    )


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    feature_names: tuple[str, ...]

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features "
                f"({self.feature_names}), got {X.shape[1]}"
            )
        # Class labels in y are {0, 1}; column 1 is "home win".
        return self.pipeline.predict_proba(X)[:, 1]


def load_model(path: Path | str) -> LoadedModel:
    blob = joblib.load(Path(path))
    if blob.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError(
            f"Model trained with features {blob.get('feature_names')}, "
            f"current code expects {FEATURE_NAMES}. Retrain before loading."
        )
    return LoadedModel(
        pipeline=blob["pipeline"],
        feature_names=tuple(blob["feature_names"]),
    )
