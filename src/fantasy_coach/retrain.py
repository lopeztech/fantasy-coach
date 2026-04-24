"""Weekly retrain + drift-report orchestration (#107).

Run once per week (Monday post-round — Cloud Scheduler + Cloud Run Job in
platform-infra). Flow:

1. Load completed matches from the repo (Firestore in prod).
2. Split into training (everything before) + holdout (last N rounds).
3. Train a fresh XGBoost candidate on training matches.
4. Shadow-evaluate incumbent + candidate on holdout.
5. Gate: promote iff candidate does not regress log-loss OR brier by more
   than ``max_regression_pct`` (2 % per AC). Accuracy informational only.
6. On promote: optionally upload candidate to GCS ``gcs_uri``.
   On block: optionally open a GitHub issue tagged ``model-drift``.
7. Always: write a ``DriftReport`` describing the incumbent's behaviour
   on holdout + per-feature PSI vs training distribution.

All side effects (drift-report writer, GCS uploader, GitHub issue opener)
are injected so the pipeline is testable without GCP or GitHub access.
The defaults (``default_drift_writer``, ``default_gcs_uploader``,
``open_github_model_drift_issue``) wire to the production backends.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fantasy_coach.drift import (
    DriftReport,
    RoundLogLoss,
    per_feature_psi,
    psi_warnings,
)
from fantasy_coach.feature_engineering import FEATURE_NAMES, build_training_frame
from fantasy_coach.features import MatchRow
from fantasy_coach.models.loader import Model, load_model
from fantasy_coach.models.promotion import (
    DEFAULT_HOLDOUT_ROUNDS,
    DEFAULT_MAX_REGRESSION_PCT,
    GateDecision,
    ShadowMetrics,
    gate_decision,
    shadow_evaluate,
    split_training_holdout,
)
from fantasy_coach.models.xgboost_model import save_model as save_xgboost_model
from fantasy_coach.models.xgboost_model import train_xgboost
from fantasy_coach.storage.repository import Repository

logger = logging.getLogger(__name__)

DriftWriter = Callable[[DriftReport], None]
GcsUploader = Callable[[Path, str], None]
IssueOpener = Callable[[GateDecision, DriftReport], int | None]


@dataclass(frozen=True)
class RetrainResult:
    """Summary of one retrain run."""

    decision: GateDecision
    drift_report: DriftReport
    candidate_path: Path
    promoted: bool
    gcs_uploaded: bool
    issue_number: int | None


def run_retrain(
    repo: Repository,
    *,
    incumbent_path: Path,
    candidate_out_path: Path,
    gcs_uri: str | None = None,
    seasons: Sequence[int] | None = None,
    holdout_rounds: int = DEFAULT_HOLDOUT_ROUNDS,
    max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT,
    drift_writer: DriftWriter | None = None,
    issue_opener: IssueOpener | None = None,
    gcs_uploader: GcsUploader | None = None,
) -> RetrainResult:
    """End-to-end weekly retrain. See module docstring."""
    all_matches = _load_all_matches(repo, seasons)
    training, holdout = split_training_holdout(all_matches, holdout_rounds=holdout_rounds)
    if not holdout:
        raise RuntimeError(f"Not enough completed rounds to form a {holdout_rounds}-round holdout")
    logger.info(
        "loaded %d matches (training=%d, holdout=%d)",
        len(all_matches),
        len(training),
        len(holdout),
    )

    frame = build_training_frame(training)
    train_result = train_xgboost(frame)
    save_xgboost_model(train_result, candidate_out_path)
    logger.info(
        "trained candidate on %d rows (test acc=%.3f) saved to %s",
        train_result.n_train,
        train_result.test_accuracy,
        candidate_out_path,
    )

    candidate_model = load_model(candidate_out_path)
    incumbent_model = load_model(incumbent_path)

    incumbent_metrics = shadow_evaluate(incumbent_model, training, holdout)
    candidate_metrics = shadow_evaluate(candidate_model, training, holdout)
    decision = gate_decision(
        incumbent_metrics,
        candidate_metrics,
        max_regression_pct=max_regression_pct,
    )
    logger.info("gate decision: promote=%s reason=%s", decision.promote, decision.reason)

    report = _build_drift_report(
        incumbent_model=incumbent_model,
        incumbent_path=incumbent_path,
        training=training,
        holdout=holdout,
    )
    if drift_writer is not None:
        try:
            drift_writer(report)
        except Exception:
            logger.exception("failed to write drift report")

    gcs_uploaded = False
    issue_number: int | None = None
    if decision.promote:
        if gcs_uri and gcs_uploader is not None:
            try:
                gcs_uploader(candidate_out_path, gcs_uri)
                gcs_uploaded = True
                logger.info("uploaded candidate to %s", gcs_uri)
            except Exception:
                logger.exception("failed to upload candidate to GCS")
    else:
        if issue_opener is not None:
            try:
                issue_number = issue_opener(decision, report)
                if issue_number is not None:
                    logger.info("opened model-drift issue #%d", issue_number)
            except Exception:
                logger.exception("failed to open drift issue")

    return RetrainResult(
        decision=decision,
        drift_report=report,
        candidate_path=candidate_out_path,
        promoted=decision.promote,
        gcs_uploaded=gcs_uploaded,
        issue_number=issue_number,
    )


def _load_all_matches(repo: Repository, seasons: Sequence[int] | None) -> list[MatchRow]:
    """Return all matches from the repo across ``seasons``.

    ``Repository.list_matches`` requires a season. With no explicit
    seasons given we pull the current year plus the two prior — matches
    what the training set is built from and covers the dataset the
    retrain pipeline cares about.
    """
    if seasons is None:
        current_year = datetime.now(UTC).year
        seasons = (current_year - 2, current_year - 1, current_year)
    rows: list[MatchRow] = []
    for season in seasons:
        rows.extend(repo.list_matches(season))
    return rows


def _build_drift_report(
    *,
    incumbent_model: Model,
    incumbent_path: Path,
    training: Sequence[MatchRow],
    holdout: Sequence[MatchRow],
) -> DriftReport:
    """Produce a drift report for the incumbent on the holdout window.

    Three signals per the AC:
    - past-round metrics (last holdout round)
    - rolling log-loss trend (per holdout round)
    - per-feature PSI comparing training vs holdout feature matrices.
    """
    training_frame = build_training_frame(training)
    # Build a combined frame so holdout rows see warm rolling state; then
    # slice out the holdout-only rows. build_training_frame filters
    # incomplete matches itself, so this is safe.
    combined_frame = build_training_frame(list(training) + list(holdout))
    holdout_ids = {m.match_id for m in holdout}
    mask = np.isin(combined_frame.match_ids, np.asarray(list(holdout_ids)))
    recent_X = combined_frame.X[mask]

    feature_psi = per_feature_psi(training_frame.X, recent_X, FEATURE_NAMES)
    warnings = psi_warnings(feature_psi)

    by_round: dict[tuple[int, int], list[MatchRow]] = {}
    for match in holdout:
        by_round.setdefault((match.season, match.round), []).append(match)

    all_matches = list(training) + list(holdout)
    rolling: list[RoundLogLoss] = []
    last_metrics: ShadowMetrics | None = None
    for key in sorted(by_round):
        pre_round = [m for m in all_matches if (m.season, m.round) < key]
        metrics = shadow_evaluate(incumbent_model, pre_round, by_round[key])
        rolling.append(
            RoundLogLoss(
                season=key[0],
                round=key[1],
                n=metrics.n,
                log_loss=metrics.log_loss,
                accuracy=metrics.accuracy,
            )
        )
        last_metrics = metrics

    model_version = (
        hashlib.sha256(incumbent_path.read_bytes()).hexdigest()[:12]
        if incumbent_path.exists()
        else "unknown"
    )

    season_now, round_now = (rolling[-1].season, rolling[-1].round) if rolling else (0, 0)
    return DriftReport(
        season=season_now,
        round=round_now,
        generated_at=datetime.now(UTC).isoformat(),
        model_version=model_version,
        past_round_accuracy=last_metrics.accuracy if last_metrics else None,
        past_round_log_loss=last_metrics.log_loss if last_metrics else None,
        past_round_brier=last_metrics.brier if last_metrics else None,
        rolling_log_loss=rolling,
        feature_psi=feature_psi,
        psi_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Default side-effect implementations (production wiring)
# ---------------------------------------------------------------------------


def default_drift_writer(report: DriftReport) -> None:
    """Write a drift report document to Firestore (production path).

    Collection: ``model_drift_reports``. Doc id: ``{season}-{round:02d}``
    so a week's report replaces itself on re-run (idempotent).
    """
    from google.cloud import firestore  # noqa: PLC0415

    client = firestore.Client()
    doc_id = f"{report.season}-{report.round:02d}"
    client.collection("model_drift_reports").document(doc_id).set(report.to_dict())


def default_gcs_uploader(local_path: Path, gcs_uri: str) -> None:
    """Upload ``local_path`` to the object at ``gcs_uri`` (gs://bucket/object)."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"gcs_uri must start with gs:// (got {gcs_uri!r})")
    bucket_name, _, blob_name = gcs_uri.removeprefix("gs://").partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"gcs_uri must be gs://<bucket>/<blob> (got {gcs_uri!r})")

    from google.cloud import storage  # noqa: PLC0415

    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
