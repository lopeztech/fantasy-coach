"""Pin walk-forward metrics for each baseline against a snapshot DB.

If you change feature engineering, the Elo defaults, or the logistic
pipeline in a way that shifts these numbers, this test fails — by design.
Update the expected dict in the same PR so the new numbers are reviewed
deliberately, not silently.

To regenerate after a deliberate change:
  uv run python -m fantasy_coach backfill --season 2024 --db data/nrl.db
  uv run python -m fantasy_coach backfill --season 2025 --db data/nrl.db
  cp data/nrl.db tests/fixtures/baseline-nrl.db
  # then run this test, copy the printed metrics into EXPECTED, commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fantasy_coach.evaluation import (
    EloPredictor,
    EnsemblePredictor,
    HomePickPredictor,
    LogisticPredictor,
    Predictor,
    XGBoostPredictor,
)
from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.storage import SQLiteRepository

BASELINE_DB = Path(__file__).parent / "fixtures" / "baseline-nrl.db"
SEASONS = (2024, 2025)

# Snapshot from a 2024+2025 backfill on 2026-04-21 (213 matches/season,
# draws dropped → 424 scored predictions).
# Logistic updated in #55 (travel), #54 (weather/venue), and #57 (referee). Extra
# features increase variance at this sample size; expected to help tree models with
# more data. Referee features show negligible signal on the 2024-2025 baseline (all
# referee_id values NULL after v1→v2 migration) — see docs/model.md for ablation notes.
#
# XGBoost (#25): log_loss 0.7599 vs logistic 0.7640 (Δ=+0.41pp) — below the 1-point
# threshold; logistic remains the default model. See docs/model.md for comparison.
#
# Ensemble (#56): accuracy 0.6014 beats Elo (+0.71pp) but log_loss 0.6782 is worse
# than Elo 0.6570 (+2.12pp). Kill switch fires for most rounds (ensemble degrades
# Elo's log-loss by mixing in less-calibrated logistic/XGBoost signals). Recommend
# calibrating ensemble output before using as default. See docs/model.md.
EXPECTED = {
    "ensemble": {"n": 424, "accuracy": 0.6014, "log_loss": 0.6782, "brier": 0.2390},
    "home": {"n": 424, "accuracy": 0.5731, "log_loss": 0.6835, "brier": 0.2452},
    "elo": {"n": 424, "accuracy": 0.5943, "log_loss": 0.6570, "brier": 0.2325},
    "logistic": {"n": 424, "accuracy": 0.5660, "log_loss": 0.7640, "brier": 0.2654},
    "xgboost": {"n": 424, "accuracy": 0.5448, "log_loss": 0.7599, "brier": 0.2721},
}

PREDICTORS: dict[str, type[Predictor]] = {
    "ensemble": EnsemblePredictor,
    "home": HomePickPredictor,
    "elo": EloPredictor,
    "logistic": LogisticPredictor,
    "xgboost": XGBoostPredictor,
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_walk_forward_metrics_match_baseline(name: str) -> None:
    if not BASELINE_DB.exists():
        pytest.skip(f"baseline DB missing at {BASELINE_DB}; see module docstring")

    repo = SQLiteRepository(BASELINE_DB)
    try:
        result = walk_forward_from_repo(repo, SEASONS, PREDICTORS[name])
    finally:
        repo.close()

    expected = EXPECTED[name]
    metrics = result.metrics()

    assert result.n == expected["n"], (
        f"{name}: prediction count drift, got n={result.n}, want {expected['n']}"
    )
    # 1e-3 tolerance covers cross-platform sklearn / BLAS jitter without
    # masking real regressions (a 0.5pp accuracy drop is meaningful here).
    for key in ("accuracy", "log_loss", "brier"):
        assert metrics[key] == pytest.approx(expected[key], abs=1e-3), (
            f"{name}: {key} drifted from baseline. got={metrics[key]:.4f} want={expected[key]:.4f}"
        )
