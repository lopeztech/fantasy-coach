"""`python -m fantasy_coach` entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fantasy_coach.backfill import (
    BackfillState,
    RetryLog,
    backfill_season,
)
from fantasy_coach.bookmaker import BookmakerPredictor, load_closing_lines
from fantasy_coach.evaluation import (
    EloPredictor,
    HomePickPredictor,
    LogisticPredictor,
)
from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.evaluation.report import write_markdown
from fantasy_coach.feature_engineering import build_training_frame
from fantasy_coach.models.logistic import save_model, train_logistic
from fantasy_coach.storage import SQLiteRepository

_BUILTIN_PREDICTORS = {
    "home": HomePickPredictor,
    "elo": EloPredictor,
    "logistic": LogisticPredictor,
}
# `bookmaker` is appended at parse time so users see it in --help even
# though it requires a separate --closing-lines argument.
_PREDICTOR_CHOICES = sorted([*_BUILTIN_PREDICTORS, "bookmaker"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fantasy_coach")
    sub = parser.add_subparsers(dest="command", required=True)

    bf = sub.add_parser("backfill", help="Backfill a season's matches into a SQLite DB.")
    bf.add_argument("--season", type=int, required=True)
    bf.add_argument("--db", type=Path, default=Path("data/nrl.db"))
    bf.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Sidecar JSON tracking processed match URLs. Defaults to <db>.backfill.json",
    )
    bf.add_argument(
        "--retry-file",
        type=Path,
        default=None,
        help="Append failed matches here for a second pass. Defaults to <db>.retry.tsv",
    )
    bf.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    tl = sub.add_parser(
        "train-logistic",
        help="Train the logistic-regression baseline against backfilled matches.",
    )
    tl.add_argument("--season", type=int, action="append", required=True)
    tl.add_argument("--db", type=Path, default=Path("data/nrl.db"))
    tl.add_argument("--out", type=Path, default=Path("artifacts/logistic.joblib"))
    tl.add_argument("--test-fraction", type=float, default=0.2)
    tl.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    ev = sub.add_parser(
        "evaluate",
        help="Walk-forward evaluation across one or more models.",
    )
    ev.add_argument(
        "--model",
        action="append",
        choices=_PREDICTOR_CHOICES,
        required=True,
        help="May be specified multiple times to compare models.",
    )
    ev.add_argument(
        "--seasons",
        required=True,
        help="Comma-separated season list, e.g. 2024,2025",
    )
    ev.add_argument("--db", type=Path, default=Path("data/nrl.db"))
    ev.add_argument(
        "--report",
        type=Path,
        default=Path("reports/evaluation.md"),
    )
    ev.add_argument(
        "--closing-lines",
        type=Path,
        default=None,
        help="Path to the aussportsbetting NRL xlsx. Required when --model bookmaker is used.",
    )
    ev.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "backfill":
        return _run_backfill(args)
    if args.command == "train-logistic":
        return _run_train_logistic(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    parser.error(f"unknown command {args.command!r}")
    return 2  # unreachable


def _run_backfill(args: argparse.Namespace) -> int:
    db_path: Path = args.db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    state_path: Path = args.state or db_path.with_suffix(db_path.suffix + ".backfill.json")
    retry_path: Path = args.retry_file or db_path.with_suffix(db_path.suffix + ".retry.tsv")

    repo = SQLiteRepository(db_path)
    state = BackfillState.load(state_path)
    retry_log = RetryLog(retry_path)
    try:
        summaries = backfill_season(args.season, repo, state, retry_log)
    finally:
        repo.close()

    totals = {
        "fetched": sum(s.fetched for s in summaries),
        "skipped": sum(s.skipped for s in summaries),
        "failed": sum(s.failed for s in summaries),
    }
    print(
        f"Season {args.season}: rounds={len(summaries)} "
        f"fetched={totals['fetched']} skipped={totals['skipped']} failed={totals['failed']}"
    )
    return 0 if totals["failed"] == 0 else 1


def _run_train_logistic(args: argparse.Namespace) -> int:
    repo = SQLiteRepository(args.db)
    try:
        matches = []
        for season in args.season:
            matches.extend(repo.list_matches(season))
    finally:
        repo.close()

    frame = build_training_frame(matches)
    result = train_logistic(frame, test_fraction=args.test_fraction)
    save_model(result, args.out)

    print(
        f"Trained on {result.n_train} matches, tested on {result.n_test}. "
        f"Train acc={result.train_accuracy:.3f} "
        f"test acc={result.test_accuracy:.3f}. "
        f"Saved to {args.out}"
    )
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]

    factories = {}
    for name in args.model:
        if name == "bookmaker":
            if args.closing_lines is None:
                print(
                    "error: --model bookmaker requires --closing-lines PATH",
                    file=sys.stderr,
                )
                return 2
            lines = load_closing_lines(args.closing_lines)
            factories[name] = lambda lines=lines: BookmakerPredictor(lines)
        else:
            factories[name] = _BUILTIN_PREDICTORS[name]

    repo = SQLiteRepository(args.db)
    try:
        results = []
        for model_name in args.model:
            results.append(walk_forward_from_repo(repo, seasons, factories[model_name]))
    finally:
        repo.close()

    write_markdown(args.report, results, seasons=seasons)
    for r in results:
        m = r.metrics()
        print(
            f"{r.predictor_name}: n={r.n} "
            f"acc={m['accuracy']:.3f} log_loss={m['log_loss']:.3f} brier={m['brier']:.3f}"
        )
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
