"""Position-weight sweep for key_absence_diff (#159).

Runs walk-forward evaluation for three POSITION_WEIGHTS schemes:
1. Expert prior (current weights in feature_engineering.py)
2. Flat (all positions = 1.0)
3. Data-driven (linear regression of point margin on per-position absence counts)

Reports accuracy / log_loss / brier for each scheme × model.

Usage:
    uv run python scripts/sweep_position_weights.py
    uv run python scripts/sweep_position_weights.py --db tests/fixtures/baseline-nrl.db
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Ensure src/ is importable when running the script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.evaluation.predictors import LogisticPredictor, XGBoostPredictor
from fantasy_coach.feature_engineering import (
    KEY_ABSENCE_REGULAR_MIN_STARTS,
    POSITION_WEIGHTS,
)
from fantasy_coach.storage import SQLiteRepository

SEASONS = (2024, 2025, 2026)

# ---------------------------------------------------------------------------
# Data-driven weight computation
# ---------------------------------------------------------------------------


def compute_data_driven_weights(
    repo: SQLiteRepository, seasons: tuple[int, ...]
) -> dict[str, float]:
    """Fit per-position weights by regressing point margin on absence counts.

    For each completed match:
    - Compute per-position absence deltas (home_absences - away_absences) for each position.
    - Target: (home_score - away_score)
    - Fit OLS with no intercept (we only want relative weights, not an absolute margin offset).
    - Normalize weights to sum to same total as the expert prior.
    """
    from fantasy_coach.feature_engineering import FeatureBuilder
    from fantasy_coach.features import MatchRow

    positions = list(POSITION_WEIGHTS.keys())

    # Collect features and targets — uppercase X matches sklearn convention (N803 ignored)
    x_rows: list[list[float]] = []
    y_rows: list[float] = []

    all_matches: list[MatchRow] = []
    for season in seasons:
        all_matches.extend(repo.list_matches(season))
    all_matches.sort(key=lambda m: (m.start_time, m.match_id))

    # Walk through matches building state for the key absence feature
    builder = FeatureBuilder()
    current_season = None

    for match in all_matches:
        if match.home.score is None or match.away.score is None:
            continue
        if match.home.score == match.away.score:
            # skip draws
            builder.record(match)
            continue

        if current_season is None:
            current_season = match.season
        elif match.season != current_season:
            builder.elo.regress_to_mean()
            builder.player_ratings.regress_to_mean()
            current_season = match.season

        # Compute per-position absence delta for this match
        h_id, a_id = match.home.team_id, match.away.team_id
        absence_delta = _per_position_absence_delta(builder, h_id, a_id, match)

        if absence_delta is not None:
            row = [absence_delta.get(pos, 0.0) for pos in positions]
            x_rows.append(row)
            margin = float(match.home.score) - float(match.away.score)
            y_rows.append(margin)

        builder.record(match)

    if len(x_rows) < 20:
        print(
            f"WARNING: Only {len(x_rows)} rows with absence data; data-driven weights may be noisy"
        )
        # Fall back to expert prior
        return POSITION_WEIGHTS.copy()

    # Uppercase X, XtX, Xty match sklearn/linear-algebra convention  # noqa: N806
    X = np.array(x_rows, dtype=float)  # noqa: N806
    y = np.array(y_rows, dtype=float)

    # OLS: weights = (X'X)^{-1} X'y
    # Use ridge regularization for stability (alpha=1.0)
    ridge_alpha = 1.0
    XtX = X.T @ X + ridge_alpha * np.eye(len(positions))  # noqa: N806
    Xty = X.T @ y  # noqa: N806
    weights_raw = np.linalg.solve(XtX, Xty)

    # Take absolute value (absence should always hurt) and clip negative coefficients
    weights_abs = np.abs(weights_raw)

    # Normalize to same total as expert prior
    expert_total = sum(POSITION_WEIGHTS.values())
    raw_total = weights_abs.sum()
    if raw_total < 1e-9:
        print("WARNING: Data-driven weights near zero; falling back to expert prior")
        return POSITION_WEIGHTS.copy()
    scale = expert_total / raw_total
    weights_normalized = weights_abs * scale

    data_driven = {pos: float(w) for pos, w in zip(positions, weights_normalized, strict=True)}
    return data_driven


def _per_position_absence_delta(  # type: ignore[no-untyped-def]
    builder,  # noqa: ANN001 — FeatureBuilder imported lazily inside compute_data_driven_weights
    home_id: int,
    away_id: int,
    match,  # noqa: ANN001 — MatchRow imported lazily
) -> dict[str, float] | None:
    """Return per-position (home_absences - away_absences) dict, or None if no data."""
    home_history = builder._team_starters.get(home_id)
    away_history = builder._team_starters.get(away_id)

    if not home_history or not away_history:
        return None

    home_starters = {p.player_id for p in match.home.players if p.is_on_field}
    away_starters = {p.player_id for p in match.away.players if p.is_on_field}

    if not home_starters or not away_starters:
        return None

    delta: dict[str, float] = defaultdict(float)

    for history, current_starters, sign in [
        (home_history, home_starters, 1.0),
        (away_history, away_starters, -1.0),
    ]:
        from collections import Counter

        starts: Counter[int] = Counter()
        position_counts: dict[int, Counter[str]] = defaultdict(Counter)

        for match_starters in history:
            for pid, (pos, _name) in match_starters.items():
                starts[pid] += 1
                position_counts[pid][pos] += 1

        for pid, count in starts.items():
            if count < KEY_ABSENCE_REGULAR_MIN_STARTS:
                continue
            if pid in current_starters:
                continue
            reg_pos = position_counts[pid].most_common(1)[0][0]
            delta[reg_pos] += sign  # home absent = +1, away absent = -1

    return dict(delta)


# ---------------------------------------------------------------------------
# Walk-forward with custom POSITION_WEIGHTS
# ---------------------------------------------------------------------------


def run_sweep_with_weights(
    repo: SQLiteRepository,
    weight_scheme: dict[str, float],
    scheme_name: str,
) -> dict[str, dict[str, float]]:
    """Monkey-patch POSITION_WEIGHTS, run walk-forward for both models, restore."""
    import fantasy_coach.feature_engineering as fe

    original = fe.POSITION_WEIGHTS.copy()
    fe.POSITION_WEIGHTS.clear()
    fe.POSITION_WEIGHTS.update(weight_scheme)

    results: dict[str, dict[str, float]] = {}

    try:
        for predictor_cls, model_name in [
            (LogisticPredictor, "logistic"),
            (XGBoostPredictor, "xgboost"),
        ]:
            print(f"  Running {model_name} ({scheme_name})...", flush=True)
            result = walk_forward_from_repo(repo, SEASONS, predictor_cls)
            metrics = result.metrics()
            results[model_name] = {
                "n": result.n,
                **metrics,
            }
    finally:
        fe.POSITION_WEIGHTS.clear()
        fe.POSITION_WEIGHTS.update(original)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep position weights for key_absence_diff")
    parser.add_argument(
        "--db",
        default="tests/fixtures/baseline-nrl.db",
        help="Path to SQLite database (default: tests/fixtures/baseline-nrl.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    repo = SQLiteRepository(db_path)

    print("=" * 70)
    print("Position-weight sweep for key_absence_diff (#159)")
    print("=" * 70)
    print()

    # Scheme 1: Expert prior (current)
    expert_weights = POSITION_WEIGHTS.copy()
    print("Scheme 1 — Expert prior (current POSITION_WEIGHTS):")
    for pos, w in sorted(expert_weights.items(), key=lambda x: -x[1]):
        print(f"  {pos:<20} {w:.2f}")
    print()

    # Scheme 2: Flat weights
    flat_weights = {pos: 1.0 for pos in POSITION_WEIGHTS}
    print("Scheme 2 — Flat (all positions = 1.0):")
    print()

    # Scheme 3: Data-driven
    print("Computing data-driven weights from match history...", flush=True)
    data_weights = compute_data_driven_weights(repo, SEASONS)
    print("Scheme 3 — Data-driven (OLS regression, normalized):")
    for pos, w in sorted(data_weights.items(), key=lambda x: -x[1]):
        print(f"  {pos:<20} {w:.3f}")
    print()

    schemes = {
        "expert_prior": expert_weights,
        "flat": flat_weights,
        "data_driven": data_weights,
    }

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for scheme_name, weights in schemes.items():
        print(f"Running walk-forward: {scheme_name}...")
        scheme_results = run_sweep_with_weights(repo, weights, scheme_name)
        all_results[scheme_name] = scheme_results

    repo.close()

    # Print results table
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    for model_name in ["logistic", "xgboost"]:
        print(f"Model: {model_name.upper()}")
        print(f"{'Scheme':<20} {'n':>5} {'accuracy':>10} {'log_loss':>10} {'brier':>10}")
        print("-" * 60)
        for scheme_name in schemes:
            r = all_results[scheme_name][model_name]
            print(
                f"{scheme_name:<20} {r['n']:>5} "
                f"{r['accuracy']:>10.4f} {r['log_loss']:>10.4f} {r['brier']:>10.4f}"
            )
        print()

    # Determine winner and print recommendation
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    for model_name in ["logistic", "xgboost"]:
        print(f"\n{model_name.upper()}:")
        best_acc_scheme = min(
            schemes.keys(),
            key=lambda s: -all_results[s][model_name]["accuracy"],
        )
        best_ll_scheme = min(
            schemes.keys(),
            key=lambda s: all_results[s][model_name]["log_loss"],
        )
        best_brier_scheme = min(
            schemes.keys(),
            key=lambda s: all_results[s][model_name]["brier"],
        )
        acc_val = all_results[best_acc_scheme][model_name]["accuracy"]
        ll_val = all_results[best_ll_scheme][model_name]["log_loss"]
        brier_val = all_results[best_brier_scheme][model_name]["brier"]
        print(f"  Best accuracy:  {best_acc_scheme} ({acc_val:.4f})")
        print(f"  Best log_loss:  {best_ll_scheme} ({ll_val:.4f})")
        print(f"  Best brier:     {best_brier_scheme} ({brier_val:.4f})")

        # Expert vs flat comparison
        exp_acc = all_results["expert_prior"][model_name]["accuracy"]
        flat_acc = all_results["flat"][model_name]["accuracy"]
        dd_acc = all_results["data_driven"][model_name]["accuracy"]
        print(f"  expert_prior vs flat:       Δacc = {exp_acc - flat_acc:+.4f}")
        print(f"  expert_prior vs data_driven: Δacc = {exp_acc - dd_acc:+.4f}")


if __name__ == "__main__":
    main()
