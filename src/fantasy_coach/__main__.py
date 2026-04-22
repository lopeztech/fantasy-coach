"""`python -m fantasy_coach` entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

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

    pc = sub.add_parser(
        "precompute",
        help=(
            "Scrape + compute predictions and write them to the configured store. "
            "Called by the Cloud Run Job twice a week so the API never scrapes "
            "on the hot path."
        ),
    )
    pc.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to precompute. Omit to autodetect the current year.",
    )
    pc.add_argument(
        "--round",
        type=int,
        default=None,
        help=(
            "Round to precompute. Omit to autodetect the next upcoming round "
            "(first round containing a fixture with matchState Upcoming/Pre)."
        ),
    )
    pc.add_argument(
        "--no-force",
        action="store_true",
        help=(
            "Respect existing cached predictions instead of re-scraping. "
            "Default is --force so scheduled runs pick up team-list changes."
        ),
    )
    pc.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    cp = sub.add_parser(
        "copy-matches-to-firestore",
        help=(
            "Copy completed matches from a local SQLite DB to Firestore. "
            "One-off bootstrap so the precompute Job's FeatureBuilder sees "
            "the same history the model was trained against — otherwise every "
            "historical feature (Elo, form, h2h, venue, referee, key-absence) "
            "defaults to zero at inference."
        ),
    )
    cp.add_argument("--db", type=Path, default=Path("data/nrl.db"))
    cp.add_argument(
        "--season",
        type=int,
        action="append",
        default=None,
        help="Season to copy. May be repeated. Omit to copy every season in the DB.",
    )
    cp.add_argument(
        "--project",
        type=str,
        default=None,
        help=(
            "GCP project hosting Firestore. Defaults to GOOGLE_CLOUD_PROJECT / "
            "FIREBASE_PROJECT_ID / ADC project."
        ),
    )
    cp.add_argument(
        "--database",
        type=str,
        default="(default)",
        help='Firestore database name (Python client default: "(default)")',
    )
    cp.add_argument(
        "--dry-run",
        action="store_true",
        help="Count matches per season but do not write.",
    )
    cp.add_argument(
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
    if args.command == "precompute":
        return _run_precompute(args)
    if args.command == "copy-matches-to-firestore":
        return _run_copy_matches_to_firestore(args)
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


def _detect_upcoming_round(year: int) -> int | None:
    """Return the first round in ``year`` with an unstarted fixture, or None.

    Iterates ``fetch_round`` starting at round 1. ``matchState`` values
    ``Upcoming`` and ``Pre`` both mean "hasn't kicked off yet". Capped at 30
    rounds to cover regular season + finals; beyond that the season is done.
    """
    from fantasy_coach.scraper import fetch_round  # noqa: PLC0415

    for round_ in range(1, 31):
        payload = fetch_round(year, round_)
        if payload is None:
            continue
        fixtures = payload.get("fixtures") or []
        if any(f.get("matchState") in ("Upcoming", "Pre") for f in fixtures):
            return round_
    return None


def _run_precompute(args: argparse.Namespace) -> int:
    import os  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    from fantasy_coach.config import get_repository  # noqa: PLC0415
    from fantasy_coach.predictions import compute_predictions, get_prediction_store  # noqa: PLC0415
    from fantasy_coach.storage.team_list import (  # noqa: PLC0415
        FirestoreTeamListRepository,
        SQLiteTeamListRepository,
    )

    season = args.season or datetime.now(UTC).year
    round_ = args.round
    if round_ is None:
        round_ = _detect_upcoming_round(season)
        if round_ is None:
            print(f"No upcoming round found in season {season} — nothing to precompute.")
            return 0
        print(f"Autodetected upcoming round: season={season} round={round_}")

    repo = get_repository()
    store = get_prediction_store()
    # Shadow the STORAGE_BACKEND switch on the team-list side so snapshots
    # land in the same backend as matches. SQLite dev reuses the repo's
    # connection to keep everything in one file.
    if os.getenv("STORAGE_BACKEND", "sqlite").lower() == "firestore":
        team_list_repo: Any = FirestoreTeamListRepository()
    else:
        # Access the underlying sqlite3.Connection. SQLiteRepository exposes
        # it as ``_conn``; tolerate absence for fake repos in tests.
        conn = getattr(repo, "_conn", None)
        team_list_repo = SQLiteTeamListRepository(conn) if conn is not None else None

    try:
        predictions = compute_predictions(
            season,
            round_,
            repo,
            store,
            force=not args.no_force,
            team_list_repo=team_list_repo,
        )
    finally:
        if hasattr(repo, "close"):
            repo.close()
        if hasattr(store, "close"):
            store.close()

    print(f"Precomputed {len(predictions)} predictions for season={season} round={round_}")
    return 0


def _run_copy_matches_to_firestore(args: argparse.Namespace) -> int:
    """Copy completed matches from SQLite → Firestore one by one.

    Written for the #123 one-off bootstrap — Firestore's ``matches`` collection
    was empty in prod, which meant every rolling / historical feature defaulted
    to zero at inference. Idempotent: ``FirestoreRepository.upsert_match`` uses
    the match id as the doc id, so re-running this command just overwrites.
    """
    from fantasy_coach.storage.firestore import FirestoreRepository  # noqa: PLC0415

    if not args.db.exists():
        print(f"error: SQLite DB not found at {args.db}", file=sys.stderr)
        return 2

    src = SQLiteRepository(args.db)
    try:
        if args.season:
            seasons = sorted(set(args.season))
        else:
            rows = src._conn.execute(  # noqa: SLF001
                "SELECT DISTINCT season FROM matches ORDER BY season"
            ).fetchall()
            seasons = [r[0] for r in rows]

        if not seasons:
            print(f"No seasons present in {args.db}", file=sys.stderr)
            return 1

        dst: FirestoreRepository | None = None
        if not args.dry_run:
            dst = FirestoreRepository(project=args.project, database=args.database)

        grand_total = 0
        for season in seasons:
            matches = src.list_matches(season)
            print(f"Season {season}: {len(matches)} matches{' (dry-run)' if args.dry_run else ''}")
            if not args.dry_run and dst is not None:
                for row in matches:
                    dst.upsert_match(row)
            grand_total += len(matches)

        verb = "Would copy" if args.dry_run else "Copied"
        print(f"{verb} {grand_total} matches across {len(seasons)} season(s)")
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
