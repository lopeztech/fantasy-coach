"""Walk-forward eval across all predictors for the seasons given on stdin.

Run with:
    uv run python scripts/eval_all_predictors.py \
        --db tests/fixtures/baseline-nrl.db \
        --seasons 2023,2024,2025,2026

Prints a markdown table of log_loss / brier / accuracy / ECE per predictor.
Bookmaker baseline is skipped if --closing-lines is not provided.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.evaluation.predictors import (
    CalibratedXGBoostPredictor,
    EloMOVPredictor,
    EloPredictor,
    HomePickPredictor,
    LogisticPredictor,
    SkellamPredictor,
    StackedEnsemblePredictor,
    XGBoostPredictor,
)
from fantasy_coach.storage.sqlite import SQLiteRepository


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--seasons", required=True, help="comma-separated seasons")
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(",")]
    repo = SQLiteRepository(args.db)

    factories = [
        ("home", HomePickPredictor),
        ("elo", EloPredictor),
        ("elo_mov", EloMOVPredictor),
        ("logistic", LogisticPredictor),
        ("xgboost", XGBoostPredictor),
        ("xgboost+cal", CalibratedXGBoostPredictor),
        ("skellam", SkellamPredictor),
        ("stacked", StackedEnsemblePredictor),
    ]

    rows = []
    for name, cls in factories:
        t0 = time.time()
        result = walk_forward_from_repo(repo, seasons, cls)
        dt = time.time() - t0
        m = result.metrics()
        rows.append(
            (name, result.n, m["log_loss"], m["brier"], m["accuracy"], m["ece"], dt)
        )
        print(
            f"{name:14s}  n={result.n:4d}  log_loss={m['log_loss']:.4f}  "
            f"brier={m['brier']:.4f}  acc={m['accuracy']:.4f}  ece={m['ece']:.4f}  "
            f"({dt:.1f}s)",
            flush=True,
        )

    print()
    print("| model | n | log_loss | brier | accuracy | ECE |")
    print("|-------|---|----------|-------|----------|-----|")
    for name, n, ll, br, acc, ece, _ in rows:
        print(f"| {name} | {n} | {ll:.4f} | {br:.4f} | {acc:.4f} | {ece:.4f} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
