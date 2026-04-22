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
    HomePickPredictor,
    LogisticPredictor,
    Predictor,
    XGBoostPredictor,
)
from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.storage import SQLiteRepository

BASELINE_DB = Path(__file__).parent / "fixtures" / "baseline-nrl.db"
SEASONS = (2024, 2025)

# Snapshot from a 2024+2025 backfill on 2026-04-22 (213 matches/season,
# draws dropped → 424 scored predictions). Baseline DB refreshed in #27 so
# the ``is_on_field`` flag (from #24) is populated for all historical rows.
#
# Logistic updated in #55 (travel), #54 (weather/venue), #57 (referee), and
# #27 (key-absence). Referee features show negligible signal on this window
# (all referee_id NULL after v1→v2 migration). The #27 absence feature trades
# accuracy (+1.2pp) for a small log-loss regression (+0.013) — see
# docs/model.md's "Ablation notes — key-absence feature" for the full table.
#
# XGBoost picks up the biggest absolute improvement from #27's new feature:
# accuracy 0.5448 → 0.5708 (+2.6pp), log_loss 0.7599 → 0.7708 (+0.011).
# Tree models can split on position-specific thresholds the logistic can't.
# Elo stays the top model on log_loss; xgboost closes the accuracy gap.
EXPECTED = {
    "home": {"n": 424, "accuracy": 0.5731, "log_loss": 0.6835, "brier": 0.2452},
    "elo": {"n": 424, "accuracy": 0.5943, "log_loss": 0.6570, "brier": 0.2325},
    "logistic": {"n": 424, "accuracy": 0.5519, "log_loss": 0.7965, "brier": 0.2740},
    "xgboost": {"n": 424, "accuracy": 0.5708, "log_loss": 0.7708, "brier": 0.2717},
}

PREDICTORS: dict[str, type[Predictor]] = {
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
