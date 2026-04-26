"""Pin walk-forward metrics for each baseline against a snapshot DB.

If you change feature engineering, the Elo defaults, or the logistic
pipeline in a way that shifts these numbers, this test fails — by design.
Update the expected dict in the same PR so the new numbers are reviewed
deliberately, not silently.

To regenerate after a deliberate change:
  uv run python -m fantasy_coach backfill --season 2023 --db data/nrl.db
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
# #158 prepends 2023 (213 matches), bringing the walk-forward sample to
# 692 non-draw predictions. 2023 is included as scored, not warmup-only —
# the harness has no warmup mode — so 2023's cold-start predictions
# (early rounds with no rolling history) sit in the pooled metrics.
SEASONS = (2023, 2024, 2025, 2026)

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
#
# Pins refreshed again in #158 — SEASONS prepends 2023 (213 matches),
# pool grows 480 → 692 non-draw predictions. Effects on pooled metrics:
#
#   - Plain Elo +3.5pp accuracy / brier −0.0038. Cold-start ratings move
#     through 2023 first, so the 2024+ portion runs on warmer ratings;
#     the larger pool also dilutes the 2026-R1 thin-history rounds.
#   - EloMOV +1.5pp accuracy / brier −0.0051 — same effect, smaller
#     because EloMOV converges faster than plain Elo.
#   - Logistic accuracy −0.9pp BUT log_loss −0.0423 / brier −0.0152.
#     The accuracy dip is the cold-2023 portion (sparse warmup features
#     misclassify near 0.5); the calibration improvements are the larger
#     training pool letting the regulariser settle on saner coefficients.
#   - XGBoost −3.9pp accuracy / brier flat. Cold-start 2023 predictions
#     drag accuracy because XGBoost without rolling features is essentially
#     guessing; the rest of the pool is roughly unchanged. Held-out 2026
#     metrics (the production-relevant slice) are not regressed — see the
#     follow-up audit script in scripts/ for the per-season split.
#   - Skellam +2.5pp accuracy / brier −0.0154. Strong-L2 prior dominates
#     early-2023 predictions toward 0.55; the rest tightens with more data.
#   - Stacked +1.0pp accuracy / brier −0.0039. Inherits the XGBoost-component
#     drag and the Elo-component lift; net positive.
EXPECTED = {
    "home": {"n": 692, "accuracy": 0.5650, "log_loss": 0.6851, "brier": 0.2460},
    "elo": {"n": 692, "accuracy": 0.6185, "log_loss": 0.6549, "brier": 0.2315},
    "elo_mov": {"n": 692, "accuracy": 0.6272, "log_loss": 0.6566, "brier": 0.2315},
    "logistic": {"n": 692, "accuracy": 0.5679, "log_loss": 0.8898, "brier": 0.2831},
    "xgboost": {"n": 692, "accuracy": 0.5650, "log_loss": 0.6899, "brier": 0.2465},
    "skellam": {"n": 692, "accuracy": 0.5939, "log_loss": 0.6757, "brier": 0.2407},
    "stacked": {"n": 692, "accuracy": 0.5954, "log_loss": 0.6745, "brier": 0.2404},
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
