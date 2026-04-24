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
    EloMOVPredictor,
    EloPredictor,
    HomePickPredictor,
    LogisticPredictor,
    Predictor,
    SkellamPredictor,
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
# (all referee_id NULL after v1→v2 migration).
#
# #106 promoted EloMOV as the default rater used by FeatureBuilder for the
# ``elo_diff`` feature (+2.36pp accuracy over plain Elo on this baseline).
# Logistic and XGBoost numbers updated to reflect EloMOV elo_diff inputs.
# Plain Elo (EloPredictor) uses its own rater and is unchanged.
#
# #108 adds form_diff_pf_adjusted + form_diff_pa_adjusted (opponent-adjusted
# rolling form). Logistic shows a small regression on this 2-season window —
# features are kept (alongside raw form) per the issue decision; signal is
# expected to improve with more opponent-history data.
#
# #110 adds the Skellam two-Poisson margin model. alpha=200 (strong L2)
# eliminates extreme probabilities and gives better log_loss + Brier than
# logistic (0.7110 vs 0.7978 log_loss; 0.2534 vs 0.2744 Brier) with similar
# accuracy. Does not beat EloMOV on any metric.
# #109 adds `player_strength_diff` + `missing_player_strength` — per-player
# Elo-style ratings rolled up as an availability-aware composite. Logistic
# roughly flat, XGBoost gains across all three metrics.
#
# #26 adds `odds_home_win_prob` + `missing_odds` — de-vigged bookmaker-implied
# home win probability, populated from the scrape for upcoming matches and
# merged from the aussportsbetting xlsx for historical training rows.
# `merge-closing-lines` CLI joined 373 of 630 matches (2024+2025 ~77% coverage,
# 2026 rounds 1-5 ~21% — pre-season + finals tend to be unpriced).
# Both logistic AND XGBoost improve across all three metrics — the first
# feature this release to cleanly lift both models.
EXPECTED = {
    "home": {"n": 424, "accuracy": 0.5731, "log_loss": 0.6835, "brier": 0.2452},
    "elo": {"n": 424, "accuracy": 0.5943, "log_loss": 0.6570, "brier": 0.2325},
    "elo_mov": {"n": 424, "accuracy": 0.6179, "log_loss": 0.6578, "brier": 0.2323},
    "logistic": {"n": 424, "accuracy": 0.5566, "log_loss": 0.8017, "brier": 0.2735},
    # Pins refreshed in #165 — monotonic constraints on 10 direction-certain
    # features (see MONOTONE_CONSTRAINTS) moved XGBoost from 0.5755 → 0.6132
    # accuracy, −1.7 % log_loss, −2.5 % brier. See docs/model.md "Monotone
    # constraints (#165)".
    "xgboost": {"n": 424, "accuracy": 0.6132, "log_loss": 0.7364, "brier": 0.2559},
    "skellam": {"n": 424, "accuracy": 0.5778, "log_loss": 0.7051, "brier": 0.2508},
}

PREDICTORS: dict[str, type[Predictor]] = {
    "home": HomePickPredictor,
    "elo": EloPredictor,
    "elo_mov": EloMOVPredictor,
    "logistic": LogisticPredictor,
    "xgboost": XGBoostPredictor,
    "skellam": SkellamPredictor,
}

# Per-predictor tolerance. sklearn-based predictors are bit-stable across
# Linux + macOS, so 1e-3 catches real regressions. xgboost's
# OMP-parallelised tree splits are *not* bit-stable across platforms —
# a handful of close predictions flip between macOS and Ubuntu CI.
# #27 measured ~0.005 drift, #109 measured ~0.011, and #165 (monotone
# constraints) widened it to ~0.0165 — the constraints push more
# predictions into "just over the decision boundary" territory where
# OMP-ordering flips get amplified. 2.0e-2 swallows observed drift;
# still tight enough to catch a 2pp regression, well above noise floor.
_TOL: dict[str, float] = {"xgboost": 2.0e-2, "skellam": 5e-3}
_DEFAULT_TOL = 1e-3


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
    tol = _TOL.get(name, _DEFAULT_TOL)
    for key in ("accuracy", "log_loss", "brier"):
        assert metrics[key] == pytest.approx(expected[key], abs=tol), (
            f"{name}: {key} drifted from baseline. got={metrics[key]:.4f} "
            f"want={expected[key]:.4f} (tol={tol})"
        )
