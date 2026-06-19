"""A/B walk-forward backtest for the #269 injury features.

Runs the XGBoost predictor over the same seasons on the same DB twice — once
with the scraped injury index active, once without — and prints the metric
delta. This is the acceptance gate for #269: the injury features only ship if
the WITH arm beats the WITHOUT arm on log-loss / Brier.

    uv run python scripts/ab_injury_backtest.py --db data/nrl.db --seasons 2024,2025
"""

from __future__ import annotations

import argparse

from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.evaluation.predictors import XGBoostPredictor
from fantasy_coach.feature_engineering import InjuryIndex, active_injury_index
from fantasy_coach.storage import SQLiteRepository
from fantasy_coach.storage.injury import SQLiteInjuryReportRepository


def _load_index(conn, seasons: list[int]) -> InjuryIndex:
    repo = SQLiteInjuryReportRepository(conn)
    reports = []
    for s in seasons:
        reports.extend(repo.list_reports(season=s))
    return InjuryIndex.from_records(reports)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/nrl.db")
    ap.add_argument("--seasons", default="2024,2025")
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]

    repo = SQLiteRepository(args.db)
    try:
        idx = _load_index(repo._conn, seasons)
        print(f"injury index: {len(idx.rounds_with_data)} (season,round) keys with reports")

        # WITHOUT injury features (active index = None).
        with active_injury_index(None):
            without = walk_forward_from_repo(repo, seasons, XGBoostPredictor)

        # WITH injury features.
        with active_injury_index(idx):
            with_inj = walk_forward_from_repo(repo, seasons, XGBoostPredictor)
    finally:
        repo.close()

    mw, mi = without.metrics(), with_inj.metrics()
    print(f"\nseasons={seasons}  n={without.n}")
    print(f"{'metric':<10}{'without':>12}{'with':>12}{'delta':>12}")
    for k in ("accuracy", "log_loss", "brier"):
        d = mi[k] - mw[k]
        print(f"{k:<10}{mw[k]:>12.4f}{mi[k]:>12.4f}{d:>+12.4f}")
    print("\nlog_loss/brier: lower is better (negative delta = injury helps)")
    print("accuracy: higher is better (positive delta = injury helps)")


if __name__ == "__main__":
    main()
