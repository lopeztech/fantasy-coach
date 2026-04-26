"""XGBoost model for NRL match prediction.

Trains an XGBClassifier on the same feature set as the logistic baseline.
Hyperparameters are selected via time-series-aware CV (TimeSeriesSplit) to
avoid future-leak. Uses the same joblib artefact format as logistic.py so the
prediction API can swap models via config without code changes.

Comparison rule (see AC for #25): if XGBoost log_loss is not at least 1 point
better than logistic on the walk-forward baseline, keep logistic as the default
(simpler = fewer things to break). See docs/model.md for recorded comparison.

Hyperparameter tuning (#167): ``optuna_search`` runs a TPE-sampled search
over a wider hyperparameter space than the original hand-picked grid.
Resumable via a storage URL. Output goes to ``artifacts/best_params.json``;
``train_xgboost`` prefers that file when present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from fantasy_coach.feature_engineering import FEATURE_NAMES, TrainingFrame

# Monotonic constraints for features whose direction is a physics-of-the-game
# guarantee, not a thing the model should rediscover on 424 matches. Signs
# are in the home-perspective feature frame (home − away unless noted).
#
# Example of why we do this: on 2026 R8 Wests Tigers v Raiders, XGBoost
# assigned a negative home-push to ``odds_home_win_prob = 0.6135`` — a tree
# split that directly contradicts the feature's definition. Constraints
# prevent that class of split from being learned at all. Features not
# listed stay unconstrained (0) because their relationship to home win is
# genuinely ambiguous or depends on interactions.
#
# The #107 retrain gate catches any log-loss / brier regression, so this
# is a bounded-downside change.
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "elo_diff": 1,
    "form_diff_pf": 1,
    "form_diff_pa": -1,
    "h2h_recent_diff": 1,
    "venue_home_win_rate": 1,
    "key_absence_diff": -1,
    "form_diff_pf_adjusted": 1,
    "form_diff_pa_adjusted": -1,
    "player_strength_diff": 1,
    "odds_home_win_prob": 1,
    # Position-group matchup differentials (#210): positive diff = home stronger.
    "halves_strength_diff": 1,
    "forwards_strength_diff": 1,
    "hooker_strength_diff": 1,
    "outside_backs_strength_diff": 1,
    "halves_x_forwards_diff": 1,
}


def _monotone_tuple() -> tuple[int, ...]:
    """Build the ``monotone_constraints`` tuple aligned with ``FEATURE_NAMES``.

    Unknown features default to ``0`` (unconstrained). Raises if a
    ``MONOTONE_CONSTRAINTS`` key doesn't exist in ``FEATURE_NAMES`` — that
    would be a typo and silently having no effect is worse than failing.
    """
    unknown = set(MONOTONE_CONSTRAINTS) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"MONOTONE_CONSTRAINTS has unknown features: {sorted(unknown)}")
    return tuple(MONOTONE_CONSTRAINTS.get(name, 0) for name in FEATURE_NAMES)


# Structural params — never tuned, always applied. Monotone constraints
# (#165) + the classifier-level config XGBoost needs.
#
# ``n_jobs=1`` forces single-threaded OMP reductions inside each tree
# split. Without it, XGBoost's parallel gradient accumulation is
# non-deterministic across platforms — a handful of floating-point ties
# land on different sides depending on OS scheduler + thread count, a
# few predictions flip across the 0.5 boundary, and walk-forward
# metrics drift 1–3 pp between macOS dev and Ubuntu CI. That drift made
# the XGBoost baseline tolerance grow 0.008 → 0.015 → 0.020 → 0.035
# across four PRs (#27, #109, #165, #167) — each "widen tol to
# pass CI" commit eroded the test's ability to catch real regressions.
# Single-threaded kills the drift at source. GridSearchCV + Optuna still
# parallelise at the trial level (different processes), so the
# throughput hit is modest.
_FIXED_PARAMS: dict[str, object] = {
    "eval_metric": "logloss",
    "verbosity": 0,
    "use_label_encoder": False,
    "monotone_constraints": _monotone_tuple(),
    # ``n_jobs=1`` makes XGBoost's OMP reductions deterministic *within*
    # a platform. It does NOT eliminate cross-platform drift — we
    # verified this empirically on #187: macOS (Apple Silicon NEON) and
    # Ubuntu CI (x86 AVX) give different splits on the same data because
    # the underlying FP operations round differently. That's why the
    # xgboost tolerance in test_baseline_metrics stays at 3.5e-2 rather
    # than 5e-3.
    "n_jobs": 1,
}

# Defaults for the grid-search / small-dataset fallback paths only. Not
# applied when best_params (from HPO #167) is passed in, because HPO
# already tuned these across the full search space.
_GRID_DEFAULTS: dict[str, object] = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# Grid searched via TimeSeriesSplit.
_PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
}

_CV_SPLITS = 3  # TimeSeriesSplit — small dataset needs small n_splits
_MIN_CV_ROWS = 50  # fall back to default params when training set is too small for CV

# Per-season sample-weight multipliers passed to ``XGBClassifier.fit``.
# Recent seasons represent the current rosters + coaching + any rule
# tweaks, so matches from those weigh more heavily in training than
# matches from prior seasons when team composition was different.
# Default (for seasons not in this dict) is 1.0.
#
# 2025 at 1.5× and 2026 at 2.5× is a moderate down-weight of the older
# half of the training set — aggressive enough to let the model adapt
# to current-season form, conservative enough that we still benefit from
# the ~400 rows of 2024 history. Tuned empirically once the HPO ablation
# lands; revisit if a walk-forward rerun shows the weights hurt log-loss.
SEASON_WEIGHTS: dict[int, float] = {
    2024: 1.0,
    2025: 1.5,
    2026: 2.5,
}


def recency_weights(
    start_times: np.ndarray,
    *,
    season_weights: dict[int, float] | None = None,
) -> np.ndarray:
    """Return per-row sample weights keyed by season of ``start_time``.

    ``start_times`` is a ``datetime64[s]`` array (``TrainingFrame.start_times``
    convention). Seasons not in ``season_weights`` use 1.0 — so a fresh
    historical row from 2022 gets the default weight, no manual fallback
    needed. Output shape matches input; values always > 0.
    """
    weights_by_season = season_weights if season_weights is not None else SEASON_WEIGHTS
    years = start_times.astype("datetime64[Y]").astype(int) + 1970
    return np.array([float(weights_by_season.get(int(y), 1.0)) for y in years])


# Location of the persisted tuned-hyperparams blob (#167). If present,
# ``train_xgboost`` uses its contents instead of running the coarse grid
# search. ``optuna_search`` writes this path.
BEST_PARAMS_PATH = Path("artifacts/best_params.json")


def load_best_params(path: Path | str | None = None) -> dict[str, object] | None:
    """Return tuned hyperparams from disk, or ``None`` if the file is missing.

    Falsy/missing file is non-fatal — ``train_xgboost`` falls back to the
    coarse grid in that case. This keeps HPO optional + gracefully skipped
    on e.g. a fresh clone without ``artifacts/``.
    """
    p = Path(path) if path is not None else BEST_PARAMS_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text())


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
    best_params: dict[str, object] | None = None,
    use_hpo: bool = True,
) -> TrainResult:
    """Train on the chronologically earliest (1 − test_fraction) of ``frame``.

    Three hyperparameter sources, in order:
    1. Explicit ``best_params`` kwarg — used as-is.
    2. ``artifacts/best_params.json`` via ``load_best_params()`` — auto
       when ``use_hpo=True`` (default) and the file exists. This is how
       HPO results (#167) flow into training without plumbing through
       every callsite (retrain Job, walk-forward predictor, CLI).
    3. Legacy grid search with ``_PARAM_GRID`` via ``GridSearchCV`` +
       ``TimeSeriesSplit`` — used when the first two are absent.

    Set ``use_hpo=False`` to force the legacy path (tests that exercise
    the grid-search fallback in isolation).
    """
    if frame.X.shape[0] < 10:
        raise ValueError(f"Need at least 10 rows to train; got {frame.X.shape[0]}")

    if best_params is None and use_hpo:
        best_params = load_best_params()

    order = np.argsort(frame.start_times)
    X = frame.X[order]
    y = frame.y[order]
    start_times = frame.start_times[order]

    split = int(len(X) * (1.0 - test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Recency-weighted sample weights: rows from recent seasons carry more
    # signal about current team composition. Shape matches X_train.
    weights_train = recency_weights(start_times[:split])

    if best_params is not None:
        # HPO path — Optuna or a committed best_params.json already picked
        # the hyperparameters. Skip the internal grid search.
        #
        # HPO is tuned on the full-dataset size; ``n_estimators`` values it
        # picks (e.g. 439) massively overfit on walk-forward rounds where
        # the training set is much smaller. Early-stopping on a held-out
        # tail slice trims ``n_estimators`` to what the data actually
        # supports, making the HPO params robust across dataset sizes.
        params = dict(best_params)
        if len(X_train) >= 80:
            cut = int(len(X_train) * 0.85)
            X_fit, X_val = X_train[:cut], X_train[cut:]
            y_fit, y_val = y_train[:cut], y_train[cut:]
            w_fit = weights_train[:cut]
            best = XGBClassifier(
                **_FIXED_PARAMS,
                **params,
                early_stopping_rounds=30,
                random_state=random_state,
            )
            best.fit(
                X_fit,
                y_fit,
                sample_weight=w_fit,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            # Too few rows for a stable val split — train with the full
            # tuned n_estimators and let monotone constraints + reg_alpha
            # carry the overfitting load.
            best = XGBClassifier(**_FIXED_PARAMS, **params, random_state=random_state)
            best.fit(X_train, y_train, sample_weight=weights_train)
    elif len(X_train) >= _MIN_CV_ROWS:
        tscv = TimeSeriesSplit(n_splits=_CV_SPLITS)
        search = GridSearchCV(
            XGBClassifier(**_FIXED_PARAMS, **_GRID_DEFAULTS, random_state=random_state),
            _PARAM_GRID,
            scoring="neg_log_loss",
            cv=tscv,
            refit=True,
            n_jobs=-1,
            error_score=np.nan,
        )
        search.fit(X_train, y_train, sample_weight=weights_train)
        best = search.best_estimator_
        params = {**_GRID_DEFAULTS, **dict(search.best_params_)}
    else:
        # Too few rows for reliable CV — use conservative fixed defaults.
        params = {**_GRID_DEFAULTS, "max_depth": 3, "n_estimators": 100, "learning_rate": 0.1}
        best = XGBClassifier(**_FIXED_PARAMS, **params, random_state=random_state)
        best.fit(X_train, y_train, sample_weight=weights_train)

    train_acc = float(best.score(X_train, y_train))
    test_acc = float(best.score(X_test, y_test)) if len(X_test) else float("nan")

    return TrainResult(
        estimator=best,
        feature_names=frame.feature_names,
        best_params=params,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
    )


# ---------------------------------------------------------------------------
# Optuna hyperparameter search (#167)
# ---------------------------------------------------------------------------


_OPTUNA_SEARCH_SPACE: dict[str, Any] = {
    # Conservative upper bounds for a 500-row dataset — Optuna explores the
    # whole space, but with MedianPruner the unpromising trials die early.
    # See #167 AC for the reasoning behind each range.
    "max_depth": ("int", 3, 9),
    "learning_rate": ("loguniform", 0.005, 0.2),
    "n_estimators": ("int", 100, 1500),
    "min_child_weight": ("int", 1, 20),
    "gamma": ("uniform", 0.0, 5.0),
    "subsample": ("uniform", 0.5, 1.0),
    "colsample_bytree": ("uniform", 0.5, 1.0),
    "reg_alpha": ("loguniform", 1e-4, 10.0),
    "reg_lambda": ("loguniform", 1e-4, 10.0),
}


def _sample_params(trial: Any) -> dict[str, object]:
    params: dict[str, object] = {}
    for name, spec in _OPTUNA_SEARCH_SPACE.items():
        kind, lo, hi = spec
        if kind == "int":
            params[name] = trial.suggest_int(name, int(lo), int(hi))
        elif kind == "loguniform":
            params[name] = trial.suggest_float(name, float(lo), float(hi), log=True)
        elif kind == "uniform":
            params[name] = trial.suggest_float(name, float(lo), float(hi))
        else:  # pragma: no cover — defensive
            raise ValueError(f"unknown search-space kind {kind!r}")
    return params


def optuna_search(
    frame: TrainingFrame,
    *,
    n_trials: int = 200,
    storage: str | None = None,
    study_name: str = "xgboost-hpo",
    random_state: int = 0,
    show_progress: bool = False,
) -> dict[str, object]:
    """Run Optuna TPE search over XGBoost hyperparameters.

    Objective: mean log-loss across ``TimeSeriesSplit(n_splits=_CV_SPLITS)``
    folds, minimized. Early-stopping on validation log-loss trims wasted
    trees; ``MedianPruner`` stops unpromising trials after the first couple
    of folds. ``MONOTONE_CONSTRAINTS`` stays fixed across trials — we're
    tuning around them, not against them.

    ``storage`` is an Optuna storage URL (``sqlite:///artifacts/optuna.db``
    in the CLI wrapper) for resumability. When absent, the study lives
    in-memory and evaporates on return.

    Returns the best-trial parameters as a plain dict, ready to pass to
    ``train_xgboost(best_params=...)`` or ``json.dump`` to disk.
    """
    import optuna  # noqa: PLC0415 — heavy import, keep lazy
    from optuna.pruners import MedianPruner  # noqa: PLC0415
    from optuna.samplers import TPESampler  # noqa: PLC0415

    if frame.X.shape[0] < _MIN_CV_ROWS:
        raise ValueError(
            f"optuna_search requires at least {_MIN_CV_ROWS} rows; got {frame.X.shape[0]}"
        )

    order = np.argsort(frame.start_times)
    X = frame.X[order]
    y = frame.y[order]
    start_times = frame.start_times[order]
    weights = recency_weights(start_times)

    tscv = TimeSeriesSplit(n_splits=_CV_SPLITS)

    def objective(trial: Any) -> float:
        params = _sample_params(trial)
        fold_losses: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            w_tr = weights[train_idx]
            est = XGBClassifier(
                **_FIXED_PARAMS,
                early_stopping_rounds=50,
                random_state=random_state,
                **params,
            )
            est.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            probs = est.predict_proba(X_val)[:, 1]
            ll = float(sklearn_log_loss(y_val, probs, labels=[0, 1]))
            fold_losses.append(ll)

            # Report intermediate → enables MedianPruner.
            trial.report(float(np.mean(fold_losses)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    return dict(study.best_params)


def save_best_params(params: dict[str, object], path: Path | str | None = None) -> Path:
    """Persist tuned hyperparameters as JSON. Returns the write target."""
    target = Path(path) if path is not None else BEST_PARAMS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    return target


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
