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

    txg = sub.add_parser(
        "train-xgboost",
        help=(
            "Train the XGBoost model against backfilled matches. Produces a "
            "joblib artefact at the same shape as train-logistic; the "
            "inference loader (``models.loader.load_model``) dispatches by "
            "``model_type`` so either artefact can serve at the same GCS path."
        ),
    )
    txg.add_argument("--season", type=int, action="append", required=True)
    txg.add_argument("--db", type=Path, default=Path("data/nrl.db"))
    txg.add_argument("--out", type=Path, default=Path("artifacts/xgboost.joblib"))
    txg.add_argument("--test-fraction", type=float, default=0.2)
    txg.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    tu = sub.add_parser(
        "tune-xgboost",
        help=(
            "Run Optuna (TPE) hyperparameter search over XGBoost (#167). "
            "Writes the best parameters to artifacts/best_params.json which "
            "train_xgboost + the walk-forward predictor both pick up "
            "automatically on the next fit. --storage gives a SQLite URL "
            "for resumable studies."
        ),
    )
    tu.add_argument(
        "--season",
        type=int,
        action="append",
        required=True,
        help="Season(s) to include in training. Repeatable.",
    )
    tu.add_argument("--db", type=Path, default=Path("data/nrl.db"))
    tu.add_argument("--n-trials", type=int, default=200)
    tu.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///artifacts/optuna.db). In-memory if omitted.",
    )
    tu.add_argument(
        "--study-name",
        type=str,
        default="xgboost-hpo",
    )
    tu.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/best_params.json"),
        help="Where to write the tuned best-params JSON blob.",
    )
    tu.add_argument(
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

    rt = sub.add_parser(
        "retrain",
        help=(
            "Weekly retrain + drift-report pipeline (#107). Trains a fresh "
            "XGBoost candidate, shadow-evaluates incumbent vs candidate on "
            "the last N rounds, promotes (uploads to GCS) or blocks + opens "
            "a GitHub issue, and always writes a drift report to Firestore."
        ),
    )
    rt.add_argument(
        "--incumbent-path",
        type=Path,
        default=Path("artifacts/incumbent.joblib"),
        help=(
            "Local path to the current production artefact. If missing and "
            "FANTASY_COACH_MODEL_GCS_URI is set, the file is downloaded on first use."
        ),
    )
    rt.add_argument(
        "--candidate-path",
        type=Path,
        default=Path("artifacts/candidate.joblib"),
        help="Local path to write the freshly-trained candidate artefact.",
    )
    rt.add_argument(
        "--gcs-uri",
        type=str,
        default=None,
        help=(
            "gs://bucket/blob path to upload the candidate to on promote. "
            "Defaults to FANTASY_COACH_MODEL_GCS_URI if set, else no upload."
        ),
    )
    rt.add_argument(
        "--season",
        type=int,
        action="append",
        default=None,
        help="Season(s) to include. Repeatable. Defaults to last 3 calendar years.",
    )
    rt.add_argument("--holdout-rounds", type=int, default=4)
    rt.add_argument("--max-regression-pct", type=float, default=2.0)
    rt.add_argument(
        "--github-repo",
        type=str,
        default="lopeztech/fantasy-coach",
        help="owner/name for the model-drift issue when the gate blocks.",
    )
    rt.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't upload to GCS, don't write Firestore, don't open issues.",
    )
    rt.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    mcl = sub.add_parser(
        "merge-closing-lines",
        help=(
            "Merge bookmaker closing lines from the aussportsbetting xlsx "
            "into a local SQLite DB's match rows (#26). Historical matches "
            "lose their odds in the NRL API post-kickoff, so the logistic "
            "feature pipeline needs a secondary source to see odds during "
            "training. Idempotent; re-running updates existing rows."
        ),
    )
    mcl.add_argument("--db", type=Path, default=Path("data/nrl.db"))
    mcl.add_argument(
        "--xlsx",
        type=Path,
        required=True,
        help="aussportsbetting NRL xlsx — download from "
        "https://www.aussportsbetting.com/historical_data/nrl.xlsx",
    )
    mcl.add_argument(
        "--season",
        type=int,
        action="append",
        default=None,
        help="Season to merge. Repeatable. Omit to merge every season in the DB.",
    )
    mcl.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    bttl = sub.add_parser(
        "backfill-ttl",
        help=(
            "One-shot: set ttl_timestamp on existing Firestore documents that "
            "were written before TTL fields were added (#153). Idempotent — "
            "docs already carrying ttl_timestamp are skipped. "
            "Requires STORAGE_BACKEND=firestore."
        ),
    )
    bttl.add_argument(
        "--collection",
        choices=["team_list_snapshots", "model_drift_reports", "all"],
        default="all",
        help="Which collection(s) to backfill. Default: all eligible collections.",
    )
    bttl.add_argument(
        "--dry-run",
        action="store_true",
        help="Count affected documents but do not write.",
    )
    bttl.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    ccc = sub.add_parser(
        "clear-commentary-cache",
        help=(
            "Manually evict in-memory Gemini commentary cache entries. "
            "Use --version-mismatch to purge entries from a previous CACHE_KEY_VERSION "
            "(e.g. after rolling back a bad template). "
            "Use --before-days N to purge entries older than N days."
        ),
    )
    ccc.add_argument(
        "--version-mismatch",
        action="store_true",
        help="Evict all entries whose cache_key_version != the current CACHE_KEY_VERSION.",
    )
    ccc.add_argument(
        "--before-days",
        type=float,
        default=None,
        metavar="DAYS",
        help="Evict all entries older than DAYS days (wall-clock age at write time).",
    )
    ccc.add_argument(
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
    if args.command == "train-xgboost":
        return _run_train_xgboost(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "precompute":
        return _run_precompute(args)
    if args.command == "copy-matches-to-firestore":
        return _run_copy_matches_to_firestore(args)
    if args.command == "merge-closing-lines":
        return _run_merge_closing_lines(args)
    if args.command == "retrain":
        return _run_retrain(args)
    if args.command == "tune-xgboost":
        return _run_tune_xgboost(args)
    if args.command == "backfill-ttl":
        return _run_backfill_ttl(args)
    if args.command == "clear-commentary-cache":
        return _run_clear_commentary_cache(args)
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


def _run_train_xgboost(args: argparse.Namespace) -> int:
    """Train XGBoost + write artefact at the same joblib shape as logistic.

    ``models.loader.load_model`` dispatches by ``model_type``, so swapping
    artefacts at the GCS path the Job pulls from is the only step needed
    to promote XGBoost to production (#136).
    """
    from fantasy_coach.models.xgboost_model import (  # noqa: PLC0415
        save_model as save_xgb,
    )
    from fantasy_coach.models.xgboost_model import (
        train_xgboost,
    )

    repo = SQLiteRepository(args.db)
    try:
        matches = []
        for season in args.season:
            matches.extend(repo.list_matches(season))
    finally:
        repo.close()

    frame = build_training_frame(matches)
    result = train_xgboost(frame, test_fraction=args.test_fraction)
    save_xgb(result, args.out)

    print(
        f"Trained on {result.n_train} matches, tested on {result.n_test}. "
        f"Train acc={result.train_accuracy:.3f} "
        f"test acc={result.test_accuracy:.3f}. "
        f"best_params={result.best_params}. "
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
        # Refresh any previously-scraped matches whose outcomes should be
        # known by now but weren't captured — the precompute flow only
        # processes the *upcoming* round, so without this step rounds that
        # kicked off and finished between Thu and the next Tue would stay
        # stuck at ``Upcoming`` indefinitely, hiding them from /accuracy
        # and dropping them from the retrain loop's holdout. See #107
        # post-mortem on the round-8 Tigers/Raiders pick.
        from fantasy_coach.match_sync import refresh_stale_matches  # noqa: PLC0415

        refreshed = refresh_stale_matches(repo, season)
        if refreshed:
            print(f"Refreshed {refreshed} stale past-start-time matches in season {season}")
    finally:
        if hasattr(repo, "close"):
            repo.close()
        if hasattr(store, "close"):
            store.close()

    print(f"Precomputed {len(predictions)} predictions for season={season} round={round_}")
    # Commentary summary — emitted even when commentary is not wired in so
    # the log line is always present for monitoring (zero-requests is valid).
    try:
        from fantasy_coach.commentary.cache import ResponseCache  # noqa: PLC0415

        _rc = ResponseCache.__new__(ResponseCache)
        print(_rc.summary() if hasattr(_rc, "_hits") else "commentary summary: not wired")
    except Exception:  # noqa: BLE001
        pass
    return 0


def _run_clear_commentary_cache(args: argparse.Namespace) -> int:
    """Manually evict Gemini commentary cache entries.

    Because the in-memory cache is per-process, this command is most useful
    as a Cloud Run Job invocation or as a dev-time reset during template
    iteration.  Running it against a fresh process (no warm cache) reports
    0 evictions, which is the correct no-op behaviour.
    """
    from fantasy_coach.commentary.cache import CACHE_KEY_VERSION, ResponseCache  # noqa: PLC0415

    cache = ResponseCache()
    total_evicted = 0

    if args.version_mismatch:
        n = cache.clear_version_mismatch()
        print(f"Evicted {n} entries with cache_key_version != {CACHE_KEY_VERSION}")
        total_evicted += n

    if args.before_days is not None:
        max_age_secs = args.before_days * 86_400
        n = cache.clear_stale(max_age_secs)
        print(f"Evicted {n} entries older than {args.before_days:.1f} days")
        total_evicted += n

    if not args.version_mismatch and args.before_days is None:
        print(
            "No eviction criteria specified. "
            "Use --version-mismatch or --before-days DAYS.",
            file=sys.stderr,
        )
        return 1

    print(f"Total evicted: {total_evicted}")
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


def _run_merge_closing_lines(args: argparse.Namespace) -> int:
    """Merge historical closing lines from the aussportsbetting xlsx into SQLite.

    Written for #26 — bookmaker odds as a live feature. The live NRL scrape
    carries odds pre-match but wipes them after kickoff, so historical
    matches in the training set don't have any odds signal. This CLI reads
    the xlsx, canonicalises team names via the existing
    ``fantasy_coach.bookmaker.team_names.canonicalize`` helper, matches each
    line to a SQLite match row by (kickoff-local date, canonical home,
    canonical away), and updates ``home.odds`` / ``away.odds`` on that row.

    Idempotent — ``SQLiteRepository.upsert_match`` deletes + inserts, so
    re-running the command overwrites any previously merged odds.
    """
    from datetime import timedelta, timezone  # noqa: PLC0415

    from fantasy_coach.bookmaker.lines import load_closing_lines  # noqa: PLC0415
    from fantasy_coach.bookmaker.team_names import canonicalize  # noqa: PLC0415

    if not args.db.exists():
        print(f"error: SQLite DB not found at {args.db}", file=sys.stderr)
        return 2
    if not args.xlsx.exists():
        print(f"error: xlsx not found at {args.xlsx}", file=sys.stderr)
        return 2

    lines = load_closing_lines(args.xlsx)
    aest = timezone(timedelta(hours=10))  # AEST fallback — same as BookmakerPredictor

    repo = SQLiteRepository(args.db)
    try:
        if args.season:
            seasons = sorted(set(args.season))
        else:
            rows = repo._conn.execute(  # noqa: SLF001
                "SELECT DISTINCT season FROM matches ORDER BY season"
            ).fetchall()
            seasons = [r[0] for r in rows]

        total_updated = 0
        total_missed = 0
        for season in seasons:
            updated, missed = 0, 0
            for match in repo.list_matches(season):
                home_canon = canonicalize(match.home.nick_name) or canonicalize(match.home.name)
                away_canon = canonicalize(match.away.nick_name) or canonicalize(match.away.name)
                if not home_canon or not away_canon:
                    missed += 1
                    continue
                center = match.start_time.astimezone(aest).date()
                line = None
                for delta in (-1, 0, 1):  # ±1 day to absorb DST/scheduling slop
                    line = lines.get((center + timedelta(days=delta), home_canon, away_canon))
                    if line is not None:
                        break
                if line is None:
                    missed += 1
                    continue
                updated_row = match.model_copy(
                    update={
                        "home": match.home.model_copy(update={"odds": line.home_odds_close}),
                        "away": match.away.model_copy(update={"odds": line.away_odds_close}),
                    }
                )
                repo.upsert_match(updated_row)
                updated += 1
            print(f"Season {season}: merged {updated} odds rows, {missed} unmatched")
            total_updated += updated
            total_missed += missed
        print(f"Total: {total_updated} merged, {total_missed} unmatched")
    finally:
        repo.close()
    return 0


def _run_retrain(args: argparse.Namespace) -> int:
    """Drive the weekly retrain + drift-report pipeline (#107).

    Production wiring: the Cloud Run Job invokes this once a week. On a
    cold container the incumbent artefact is downloaded from
    ``FANTASY_COACH_MODEL_GCS_URI`` before the pipeline starts, mirroring
    the API's cold-start pattern in ``predictions._ensure_model``.
    """
    import os  # noqa: PLC0415

    from fantasy_coach.config import get_repository  # noqa: PLC0415
    from fantasy_coach.github_issue import open_github_model_drift_issue  # noqa: PLC0415
    from fantasy_coach.predictions import _ensure_model  # noqa: PLC0415
    from fantasy_coach.retrain import (  # noqa: PLC0415
        default_drift_writer,
        default_gcs_uploader,
        run_retrain,
    )

    _ensure_model(args.incumbent_path)

    gcs_uri = args.gcs_uri or os.getenv("FANTASY_COACH_MODEL_GCS_URI")

    drift_writer = None if args.dry_run else default_drift_writer
    gcs_uploader = None if args.dry_run else default_gcs_uploader

    def _issue_opener(decision, report):
        if args.dry_run:
            return None
        return open_github_model_drift_issue(decision, report, repo=args.github_repo)

    repo = get_repository()
    try:
        # Pre-flight — refresh any stale match states so Sunday's matches
        # reach the holdout with their real outcomes. Without this, the
        # precompute Job's "current round only" scrape means matches that
        # kicked off + finished since the last Tue scrape are still
        # ``Upcoming`` in Firestore, and ``split_training_holdout`` drops
        # them silently. Runs per season that the retrain actually covers.
        from datetime import UTC as _UTC  # noqa: PLC0415
        from datetime import datetime as _dt  # noqa: PLC0415

        from fantasy_coach.match_sync import refresh_stale_matches  # noqa: PLC0415

        seasons = args.season or (
            _dt.now(_UTC).year - 2,
            _dt.now(_UTC).year - 1,
            _dt.now(_UTC).year,
        )
        for s in seasons:
            try:
                refreshed = refresh_stale_matches(repo, s)
                if refreshed:
                    print(f"Pre-flight: refreshed {refreshed} stale matches in season {s}")
            except Exception:
                logging.getLogger(__name__).exception(
                    "pre-flight refresh_stale_matches failed for season %d — continuing", s
                )

        result = run_retrain(
            repo,
            incumbent_path=args.incumbent_path,
            candidate_out_path=args.candidate_path,
            gcs_uri=gcs_uri,
            seasons=args.season,
            holdout_rounds=args.holdout_rounds,
            max_regression_pct=args.max_regression_pct,
            drift_writer=drift_writer,
            gcs_uploader=gcs_uploader,
            issue_opener=_issue_opener,
        )
    finally:
        if hasattr(repo, "close"):
            repo.close()

    print(
        f"promoted={result.promoted} "
        f"gcs_uploaded={result.gcs_uploaded} "
        f"issue={result.issue_number} "
        f"reason={result.decision.reason!r}"
    )
    return 0 if result.promoted else 1


def _run_tune_xgboost(args: argparse.Namespace) -> int:
    """Walk-forward Optuna search over XGBoost hyperparameters (#167).

    Training set = union of every ``--season`` arg from the provided DB.
    Writes ``best_params.json`` so ``train_xgboost`` and ``XGBoostPredictor``
    pick up the tuned config on the next fit — no other code changes
    required for the retrain Job to benefit on its next Monday run.
    """
    from fantasy_coach.feature_engineering import build_training_frame  # noqa: PLC0415
    from fantasy_coach.models.xgboost_model import (  # noqa: PLC0415
        optuna_search,
        save_best_params,
    )

    repo = SQLiteRepository(args.db)
    try:
        matches = []
        for season in args.season:
            matches.extend(repo.list_matches(season))
    finally:
        repo.close()

    frame = build_training_frame(matches)
    if frame.X.shape[0] == 0:
        print(
            f"error: no completed matches found in {args.db} for seasons {args.season}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Tuning XGBoost on {frame.X.shape[0]} matches across seasons "
        f"{sorted(args.season)} ({args.n_trials} trials, "
        f"storage={args.storage or 'in-memory'})"
    )
    best = optuna_search(
        frame,
        n_trials=args.n_trials,
        storage=args.storage,
        study_name=args.study_name,
    )
    path = save_best_params(best, args.out)
    print(f"Saved tuned parameters to {path}")
    for key, value in sorted(best.items()):
        if isinstance(value, float):
            print(f"  {key:18s} = {value:.6g}")
        else:
            print(f"  {key:18s} = {value}")
    return 0


def _run_backfill_ttl(args: argparse.Namespace) -> int:
    """Backfill ttl_timestamp on existing Firestore docs that predate #153."""
    import os  # noqa: PLC0415
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    if os.getenv("STORAGE_BACKEND", "sqlite").lower() != "firestore":
        print(
            "error: backfill-ttl requires STORAGE_BACKEND=firestore",
            file=sys.stderr,
        )
        return 2

    from google.cloud import firestore  # noqa: PLC0415

    db = firestore.Client()
    dry = args.dry_run
    collections = (
        ["team_list_snapshots", "model_drift_reports"]
        if args.collection == "all"
        else [args.collection]
    )

    # TTL config per collection: (field_with_base_date, delta_days).
    ttl_config: dict[str, tuple[str, int]] = {
        "team_list_snapshots": ("scraped_at", 80),
        "model_drift_reports": ("", 548),  # "" → use now() as base
    }

    total_updated = 0
    for col_name in collections:
        field, delta_days = ttl_config[col_name]
        col = db.collection(col_name)
        updated = 0
        skipped = 0
        for doc in col.stream():
            data = doc.to_dict()
            if "ttl_timestamp" in data:
                skipped += 1
                continue
            if field and field in data:
                try:
                    base = datetime.fromisoformat(data[field])
                    if base.tzinfo is None:
                        base = base.replace(tzinfo=UTC)
                except (ValueError, TypeError):
                    base = datetime.now(UTC)
            else:
                base = datetime.now(UTC)
            ttl = base + timedelta(days=delta_days)
            if not dry:
                doc.reference.update({"ttl_timestamp": ttl})
            updated += 1
        total_updated += updated
        print(
            f"{col_name}: {'would update' if dry else 'updated'} {updated} docs, "
            f"skipped {skipped} (already had ttl_timestamp)"
        )

    print(
        f"{'Dry run — ' if dry else ''}Total: "
        f"{'would update' if dry else 'updated'} {total_updated} documents."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
