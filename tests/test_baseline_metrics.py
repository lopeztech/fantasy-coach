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
    StackedEnsemblePredictor,
    XGBoostPredictor,
)
from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.storage import SQLiteRepository

BASELINE_DB = Path(__file__).parent / "fixtures" / "baseline-nrl.db"
# #167 expands to include 2026 R1–7 (56 completed matches, 54 non-draws).
# 2026 rows represent the current rosters / coaching / any rule tweaks;
# XGBoost additionally weights them 2.5× via ``SEASON_WEIGHTS`` so they
# dominate fit decisions proportionally.
SEASONS = (2024, 2025, 2026)

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
# Pins refreshed in #167 — SEASONS extended to include 2026 R1–7 (56 new
# matches, 480 total predictions). Pooled metrics move because the 2026
# in-season rounds are harder to predict (thinner rolling history at
# early rounds) — that shows up as lower pooled accuracy for every
# model. XGBoost additionally picks up Optuna-tuned hyperparameters +
# recency weights: log_loss drops 0.7364 → 0.7045 (−4.3 %) and brier
# 0.2559 → 0.2496 (−2.5 %) — both proper scoring rules improve on the
# larger eval pool, which is what the #107 retrain gate checks.
EXPECTED = {
    "home": {"n": 480, "accuracy": 0.5646, "log_loss": 0.6852, "brier": 0.2460},
    "elo": {"n": 480, "accuracy": 0.5833, "log_loss": 0.6628, "brier": 0.2353},
    "elo_mov": {"n": 480, "accuracy": 0.6125, "log_loss": 0.6668, "brier": 0.2366},
    # #168 adds h2h_last5_home_win_rate + h2h_last5_avg_margin + missing_h2h.
    # Logistic regresses slightly — sparse H2H history on the 2024–2026 window
    # adds noise for logistic regression (less robust to sparse features than
    # tree-based models). Expected to improve once the 2023 backfill (#158) lands.
    # XGBoost marginally regresses on this narrow window for the same reason;
    # feature signal is expected to compound with deeper H2H history.
    #
    # #170 adds home_days_rest + away_days_rest + short_turnaround_diff (granular
    # scheduling features). Logistic accuracy improves +0.83 pp (0.5604 → 0.5687)
    # on the 2024–2026 baseline; XGBoost and Skellam gain marginally. Logistic
    # log_loss regresses slightly — extra features add estimation variance on this
    # narrow training window; expected to improve once the 2023 backfill (#158)
    # provides more scheduling history.
    "logistic": {"n": 480, "accuracy": 0.5687, "log_loss": 0.8505, "brier": 0.2780},
    "xgboost": {"n": 480, "accuracy": 0.5792, "log_loss": 0.7104, "brier": 0.2532},
    "skellam": {"n": 480, "accuracy": 0.5750, "log_loss": 0.7120, "brier": 0.2538},
    # #171 stacks XGBoost + Skellam + EloMOV behind a logistic-regression
    # meta-learner fit on out-of-fold base probabilities. Beats XGBoost on
    # all three metrics (+1.25 pp accuracy, −3.8 % log_loss, −3.7 % brier)
    # on this baseline. Still trails EloMOV on accuracy; meta-learner
    # regularisation on the 20 % val slice dilutes EloMOV's lead.
    # #170: stacked gains +0.42 pp accuracy (0.5854 → 0.5896).
    "stacked": {"n": 480, "accuracy": 0.5896, "log_loss": 0.6767, "brier": 0.2407},
}

PREDICTORS: dict[str, type[Predictor]] = {
    "home": HomePickPredictor,
    "elo": EloPredictor,
    "elo_mov": EloMOVPredictor,
    "logistic": LogisticPredictor,
    "xgboost": XGBoostPredictor,
    "skellam": SkellamPredictor,
    "stacked": StackedEnsemblePredictor,
}

# Per-predictor tolerance. sklearn-based predictors are bit-stable across
# Linux + macOS (1e-3 catches real regressions); xgboost is NOT.
#
# Cross-platform drift history: #27 ~0.005, #109 ~0.011, #165 ~0.0165,
# #167 ~0.029, re-measured on #187 as 0.025. Root cause is architectural
# — macOS Apple Silicon (NEON) vs Ubuntu CI x86 (AVX) round FP ops
# differently, a handful of split tiebreaks go different ways, several
# predictions land on different sides of 0.5. Not fixable at the model
# level without Dockerising the CI runner:
# - ``n_jobs=1`` fixed within-platform determinism (no more thread-order
#   flakiness) but left the cross-platform gap unchanged.
# - ``tree_method="exact"`` shrank the gap marginally (0.029 → 0.025)
#   at a meaningful model-quality cost — reverted.
#
# 3.5e-2 swallows the observed cross-platform drift. Real regressions
# of that magnitude are rare and would show up simultaneously on log_loss
# and brier, so the test still serves its purpose as a signal — it's
# just calibrated to hardware reality rather than to an unachievable
# ideal. The within-platform tightening from n_jobs=1 means that on a
# developer's own machine, drift between runs should be < 1e-3, so an
# individual's test-suite runs stay tight even when cross-platform does
# not.
_TOL: dict[str, float] = {
    "xgboost": 3.5e-2,
    # Stacked wraps XGBoost, so inherits the same cross-platform FP drift.
    "stacked": 3.5e-2,
    "skellam": 5e-3,
}
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
