"""Pin walk-forward metrics for each baseline against a snapshot DB.

If you change feature engineering, the Elo defaults, or the logistic
pipeline in a way that shifts these numbers, this test fails — by design.
Update the expected dict in the same PR so the new numbers are reviewed
deliberately, not silently.

To regenerate after a deliberate change:
  uv run python -m fantasy_coach backfill --season 2023 --db data/nrl.db
  uv run python -m fantasy_coach backfill --season 2024 --db data/nrl.db
  uv run python -m fantasy_coach backfill --season 2025 --db data/nrl.db
  # IMPORTANT: re-merge closing lines for every season — a stale merge
  # silently degrades the odds feature (see "Closing-line coverage fix"
  # in docs/model.md; it cost 3.3pp of XGBoost accuracy once).
  uv run python -m fantasy_coach merge-closing-lines --db data/nrl.db \
      --xlsx data/odds/nrl.xlsx
  cp data/nrl.db tests/fixtures/baseline-nrl.db
  # then run this test, copy the printed metrics into EXPECTED, commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fantasy_coach.evaluation import (
    BlendedPredictor,
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
# Updated when `elo_mov_home_win_prob` was added to FEATURE_NAMES — XGBoost
# loses 0.7pp on the all-seasons pool but GAINS 5.3pp on the 2026 R1-R7
# slice (the recent season the user actually cares about). Skellam improves
# universally (+1.15pp accuracy, log_loss/brier better). The new feature is
# the calibrated EloMOV home-win probability with a +1 monotone constraint;
# tree models needed direct sigmoid access rather than reconstructing it
# from `elo_diff`.
#
# #255 adds 5 ladder-position / finals-race features (ladder_position_diff,
# points_to_top8_diff, must_win_intensity, dead_rubber_indicator,
# missing_ladder). home / elo / elo_mov are unchanged (don't read FEATURE_NAMES).
# Effect on the pooled walk-forward window:
#   - logistic: −1.44pp accuracy (0.5708 → 0.5564), log_loss / brier worsen.
#     Linear model picks up noise from the new ladder columns at low rounds;
#     features are kept because they're targeted at tree models.
#   - xgboost: −0.57pp accuracy (0.5939 → 0.5882), log_loss/brier ~flat
#     (within the 3.5e-2 cross-platform tolerance).
#   - skellam: +0.58pp accuracy (0.5997 → 0.6055), log_loss / brier improve.
#   - stacked: +0.14pp accuracy, log_loss / brier improve marginally.
#
# #251 adds 3 per-player rolling-form trajectory features
# (player_form_trajectory_diff, key_player_trajectory_diff,
# missing_player_trajectory). home / elo / elo_mov unchanged.
#   - xgboost: +1.30pp accuracy (0.5882 → 0.6012), log_loss / brier improve —
#     this is the cleanest tree-model gain in the last few PRs.
#   - logistic: accuracy flat (0.5564), log_loss / brier worsen modestly
#     (linear pipeline keeps absorbing tree-model-shaped features).
#   - skellam: −0.87pp accuracy (0.6055 → 0.5968), log_loss / brier ~flat —
#     skellam doesn't read FEATURE_NAMES but the new `record()` state
#     ordering perturbs a handful of edge-case predictions near 0.5.
#   - stacked: −0.72pp accuracy (0.5968 → 0.5896), log_loss / brier worsen
#     slightly — inherits the skellam shift through the ensemble weights.
#
# #252 adds 3 cumulative fatigue features (team_fatigue_index_diff,
# spine_fatigue_index_diff, cumulative_origin_minutes_diff). The last
# defaults to 0.0 until representative_callups is wired into FeatureBuilder.
#   - xgboost: −2.32pp accuracy (0.6012 → 0.5780), log_loss / brier ~flat.
#     Inside the 3.5e-2 cross-platform tolerance but a real pooled
#     regression — the constant-zero origin-minutes column likely adds a
#     useless dimension the trees split on spuriously. Worth revisiting
#     once callup data wires through, or dropping the column for now.
#   - logistic: −0.73pp accuracy (0.5564 → 0.5491), log_loss / brier worsen.
#   - skellam: −0.14pp accuracy (0.5968 → 0.5954), log_loss / brier ~flat.
#   - stacked: +0.43pp accuracy (0.5896 → 0.5939), log_loss/brier ~flat.
# Fatigue state-ordering fix: `record()` now snapshots previous kickoff/venue
# before updating `_last_played` / `_last_venue`, so cumulative fatigue measures
# the rest/travel load before the completed match instead of the just-recorded
# match. Production XGBoost improves across all three metrics:
#   - xgboost: +0.43pp accuracy (0.5780 → 0.5824), log_loss −0.0045,
#     brier −0.0021, ECE −0.0126.
#   - logistic: −0.14pp accuracy, log_loss / brier worsen (linear model keeps
#     struggling with tree-shaped roster/travel features).
#   - skellam: −0.87pp accuracy, log_loss / brier essentially flat.
#   - stacked: −0.43pp accuracy, log_loss / brier essentially flat; inherits
#     the Skellam shift more than the XGBoost gain.
# Spine-position fix: `key_player_trajectory_diff` and `spine_fatigue_index_diff`
# now treat spine as halves + hooker + fullback, rather than all outside backs
# (fullback + centres + wingers). Winner-pick impact after both fixes:
#   - xgboost: accuracy unchanged at 0.5824; log_loss / brier worsen versus
#     fatigue-only, but production winner accuracy still stays above the #252
#     baseline.
#   - logistic: +0.29pp accuracy versus fatigue-only, log_loss / brier ~flat.
#   - skellam: +0.43pp accuracy versus fatigue-only, log_loss / brier ~flat.
#   - stacked: +0.29pp accuracy and log_loss −0.0057 versus fatigue-only.
# XGBoost rest/fatigue monotone constraints: add hard directionality for
# days-rest, short-turnaround, and cumulative fatigue columns. Higher rest for
# home / lower rest for away should not hurt home; higher home fatigue should
# not help home. Production XGBoost gains another +0.58pp accuracy
# (0.5824 → 0.5882), brier/ECE improve, with a tiny log_loss tradeoff.
# Stacked shifts down on accuracy versus the spine-only point but keeps
# better log_loss / brier than the #252 baseline.
#
# XGBoost no-signal column sampling weights: keep placeholder/constant columns
# in FEATURE_NAMES for train/serve schema compatibility, but give them near-zero
# XGBoost column-sampling weight until their backing data is wired/backfilled.
# This stops all-zero line-movement / representative / forecast columns from
# consuming colsample_bytree slots in tiny walk-forward fits. Production
# XGBoost gains +0.87pp accuracy (0.5882 → 0.5968) while log_loss and brier
# both improve. ECE worsens slightly (0.0422 → 0.0487), still acceptable given
# the accuracy and proper-scoring-rule lift; the 2026 slice improves
# materially (0.5357 → 0.6071 accuracy). Stacked accuracy is unchanged, with
# a small log_loss / brier regression from the XGBoost base shift.
#
# Closing-line coverage fix + stacked rework + production logit blend
# (2026-06, see docs/model.md). Three changes landed together:
#   1. Baseline DB re-merged with the aussportsbetting xlsx — 2023 odds had
#      never been merged (0 → 213 rows) and 2024/2025 were stale (~77 % →
#      100 %). The odds feature is XGBoost's strongest, so xgboost jumps
#      +3.3pp accuracy (0.5968 → 0.6301) with log_loss/brier improving;
#      skellam and logistic also improve on all three. home / elo / elo_mov
#      don't read odds and are unchanged.
#   2. StackedEnsemblePredictor reworked: the per-round 20 %-tail LogReg
#      meta is replaced by convex logit-space weights fit on OOF rows
#      accumulated across walk-forward rounds. Stacked improves on all
#      three metrics (0.5997 → 0.6113 accuracy on the enriched DB).
#   3. New "blended" pin = production parity (XGBoost output blended with
#      the elo_mov_home_win_prob + odds_home_win_prob features in logit
#      space, models/blend.py). Best pinned row on every metric — this is
#      what /predictions actually serves.
EXPECTED = {
    "home": {"n": 692, "accuracy": 0.5650, "log_loss": 0.6851, "brier": 0.2460},
    "elo": {"n": 692, "accuracy": 0.6185, "log_loss": 0.6549, "brier": 0.2315},
    "elo_mov": {"n": 692, "accuracy": 0.6272, "log_loss": 0.6566, "brier": 0.2315},
    "logistic": {"n": 692, "accuracy": 0.5650, "log_loss": 0.9267, "brier": 0.2876},
    "xgboost": {"n": 692, "accuracy": 0.6301, "log_loss": 0.6802, "brier": 0.2404},
    "skellam": {"n": 692, "accuracy": 0.6098, "log_loss": 0.6625, "brier": 0.2341},
    "stacked": {"n": 692, "accuracy": 0.6113, "log_loss": 0.6641, "brier": 0.2349},
    "blended": {"n": 692, "accuracy": 0.6445, "log_loss": 0.6380, "brier": 0.2232},
}

PREDICTORS: dict[str, type[Predictor]] = {
    "home": HomePickPredictor,
    "elo": EloPredictor,
    "elo_mov": EloMOVPredictor,
    "logistic": LogisticPredictor,
    "xgboost": XGBoostPredictor,
    "skellam": SkellamPredictor,
    "stacked": StackedEnsemblePredictor,
    "blended": BlendedPredictor,
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
    # Stacked and blended wrap XGBoost, so inherit the same cross-platform
    # FP drift.
    "stacked": 3.5e-2,
    "blended": 3.5e-2,
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
