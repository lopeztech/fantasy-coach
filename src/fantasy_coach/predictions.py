"""Prediction caching and computation for the /predictions API endpoint.

Cache hit:  stored predictions returned, no scraping.
Cache miss: fixtures scraped, features computed from the saved model,
            predictions stored and returned.

Predictions are persisted in a SQLite file (``FANTASY_COACH_PREDICTIONS_DB_PATH``,
default ``data/predictions.db``) for the lifetime of the process. On Cloud Run
this survives within a container instance; a future issue can move to Firestore
for true cross-restart persistence.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fantasy_coach.features import MatchRow, extract_match_features
from fantasy_coach.scraper import fetch_match_from_url, fetch_round

logger = logging.getLogger(__name__)

PREDICTIONS_DB_ENV = "FANTASY_COACH_PREDICTIONS_DB_PATH"
MODEL_PATH_ENV = "FANTASY_COACH_MODEL_PATH"
MODEL_GCS_URI_ENV = "FANTASY_COACH_MODEL_GCS_URI"
LOGISTIC_PATH_ENV = "FANTASY_COACH_LOGISTIC_PATH"
LOGISTIC_GCS_URI_ENV = "FANTASY_COACH_LOGISTIC_GCS_URI"
DEFAULT_MODEL_PATH = "artifacts/logistic.joblib"
DEFAULT_LOGISTIC_PATH = "artifacts/logistic.joblib"
DEFAULT_PREDICTIONS_DB = "data/predictions.db"

# Market-anchored shrinkage applied to the final home-win probability when the
# bookmaker's de-vigged line is available for the match. Justification: the
# #166 audit (docs/audits/player_strength_diff.md) measured 152 cases where
# `player_strength_diff` and `odds_home_win_prob` disagreed on direction —
# market won 56.6% vs PSD 43.4%. Anchoring the model 30% toward the market on
# every priced match preserves direction on agreement (most common case) and
# pulls disagreement cases toward the more-accurate signal. Tunable; raise if
# we see persistent model-overrules-market regressions in production.
MARKET_SHRINKAGE_WEIGHT = 0.3

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    season           INTEGER NOT NULL,
    round            INTEGER NOT NULL,
    match_id         INTEGER NOT NULL,
    home_id          INTEGER NOT NULL,
    home_name        TEXT    NOT NULL,
    away_id          INTEGER NOT NULL,
    away_name        TEXT    NOT NULL,
    kickoff          TEXT    NOT NULL,
    predicted_winner TEXT    NOT NULL,
    home_win_prob    REAL    NOT NULL,
    model_version    TEXT    NOT NULL,
    feature_hash     TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    contributions    TEXT,
    alternatives     TEXT,
    PRIMARY KEY (season, round, match_id)
);
"""


# ---------------------------------------------------------------------------
# Response schema (also used for OpenAPI docs via FastAPI)
# ---------------------------------------------------------------------------


class TeamInfo(BaseModel):
    id: int
    name: str


class ScorelineProb(BaseModel):
    """One (home_score, away_score) outcome with its predicted probability."""

    homeScore: int  # noqa: N815
    awayScore: int  # noqa: N815
    probability: float


class FeatureContribution(BaseModel):
    """One feature's signed push on the model's log-odds for a match.

    ``contribution`` is in log-odds units: positive ⇒ pushes home-win
    probability up. The UI picks top-K by ``abs(contribution)`` and renders
    plain-English labels per ``feature``. Only emitted for logistic models
    today (see ``_compute_contributions``); other model types get
    ``contributions`` omitted so the UI degrades gracefully.

    ``detail`` carries optional per-feature structured data the label
    renderer can use for narrative (e.g. the list of missing regular
    starters behind a ``key_absence_diff`` row). Shape is feature-specific;
    the backend adds keys only when the detail is materially more
    informative than the raw ``value``.
    """

    feature: str  # matches a FEATURE_NAMES entry
    value: float  # raw (pre-scaling) feature value
    contribution: float  # signed log-odds contribution
    detail: dict[str, Any] | None = None


class PickSummary(BaseModel):
    """Compact pick + probability for one source in the three-way consensus panel."""

    predictedWinner: str  # "home" | "away"
    homeWinProbability: float


class AlternativeModels(BaseModel):
    """Secondary picks shown alongside the primary XGBoost prediction.

    Both fields are optional: ``logistic`` is absent when the logistic
    artefact path isn't configured (``FANTASY_COACH_LOGISTIC_PATH`` unset);
    ``bookmaker`` is absent when odds data wasn't available for the match
    (``missing_odds`` feature = 1.0).
    """

    logistic: PickSummary | None = None
    bookmaker: PickSummary | None = None


class PredictionOut(BaseModel):
    matchId: int
    home: TeamInfo
    away: TeamInfo
    kickoff: str  # ISO 8601 UTC
    predictedWinner: str  # "home" | "away"
    homeWinProbability: float
    modelVersion: str  # first 12 hex chars of SHA-256 of the model file
    featureHash: str  # first 12 hex chars of SHA-256 of feature names
    # Optional so predictions written before #58 shipped still deserialise.
    contributions: list[FeatureContribution] | None = None
    # Populated at serve-time when match_state == "FullTime" (not cached).
    actualWinner: str | None = None  # "home" | "away"
    # Populated when a Skellam model is used (optional — additive, does not
    # break existing SPA clients that ignore unknown fields).
    predictedMargin: float | None = None  # E[home_score - away_score]
    marginCi95: tuple[int, int] | None = None  # (lo, hi) at 2.5/97.5 pct
    # Three-way consensus (#140): logistic + bookmaker picks alongside the
    # primary XGBoost pick. Absent on predictions cached before #140 shipped.
    alternatives: AlternativeModels | None = None
    # Prediction uncertainty fields (#146). All optional so predictions
    # written before this feature shipped still deserialise correctly.
    # `baseModelSpread` is max − min probability across available base models
    # (XGBoost + logistic + bookmaker). `winProbability80ci` is a proxy 80%
    # interval derived from the spread (clipped to [0, 1]).
    # `trainingDataSimilarity` is cosine similarity between this match's
    # feature vector and the training-set centroid — low values flag
    # out-of-distribution matchups. `confidenceBand` is a 3-level label
    # derived from spread + OOD similarity; thresholds are heuristic
    # pending held-out calibration (see docs/model.md).
    baseModelSpread: float | None = None
    winProbability80ci: tuple[float, float] | None = None
    trainingDataSimilarity: float | None = None
    confidenceBand: str | None = None  # "low" | "medium" | "high"
    # Joint score-distribution / multi-task fields (#209, #215). Optional so
    # predictions from other model types still deserialise correctly.
    predictedTotal: float | None = None  # noqa: N815
    drawProbability: float | None = None  # noqa: N815
    topScorelines: list[ScorelineProb] | None = None  # noqa: N815
    predictedLineBreakDiff: float | None = None  # noqa: N815  multi-task #215
    # OOD detection fields (#216). Optional; absent on predictions from models
    # without a fitted OOD detector.
    oodScore: float | None = None  # noqa: N815  0–100 percentile
    oodFlag: str | None = None  # noqa: N815  "in_distribution" | "edge" | "out_of_distribution"
    # Conformal prediction intervals (#214). Guaranteed-coverage intervals
    # derived from calibration-set residuals; absence means conformalizer was
    # not fitted or the model type doesn't support it.
    homeWinProbabilityCi80: tuple[float, float] | None = None  # noqa: N815
    marginCi80: tuple[float, float] | None = None  # noqa: N815


# ---------------------------------------------------------------------------
# Prediction store (SQLite-backed cache)
# ---------------------------------------------------------------------------


class PredictionStore:
    """SQLite-backed cache for computed predictions."""

    def __init__(self, path: str | Path | None = None) -> None:
        db_path = Path(path or os.getenv(PREDICTIONS_DB_ENV, DEFAULT_PREDICTIONS_DB))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because FastAPI serves the endpoint in a
        # worker thread (run_in_threadpool) distinct from the one that
        # instantiated the module-level store. Reads are serialised by the
        # GIL and writes only come from the precompute Job (different
        # process), so cross-thread sharing is safe here.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE)
        # Forward-compat migrations — each ALTER TABLE is a no-op on a fresh
        # DB (CREATE TABLE above already defines the column).
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("ALTER TABLE predictions ADD COLUMN contributions TEXT")
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("ALTER TABLE predictions ADD COLUMN alternatives TEXT")
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("ALTER TABLE predictions ADD COLUMN uncertainty TEXT")

    def close(self) -> None:
        self._conn.close()

    def get(self, season: int, round_: int) -> list[PredictionOut]:
        rows = self._conn.execute(
            "SELECT * FROM predictions WHERE season=? AND round=? ORDER BY match_id",
            (season, round_),
        ).fetchall()
        return [_row_to_out(r) for r in rows]

    def put(self, season: int, round_: int, predictions: list[PredictionOut]) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn:
            for p in predictions:
                contribs_json = (
                    json.dumps([c.model_dump() for c in p.contributions])
                    if p.contributions is not None
                    else None
                )
                alts_json = (
                    json.dumps(p.alternatives.model_dump()) if p.alternatives is not None else None
                )
                uncertainty_json = _uncertainty_json(p)
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO predictions
                        (season, round, match_id,
                         home_id, home_name, away_id, away_name,
                         kickoff, predicted_winner, home_win_prob,
                         model_version, feature_hash, created_at,
                         contributions, alternatives, uncertainty)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        season,
                        round_,
                        p.matchId,
                        p.home.id,
                        p.home.name,
                        p.away.id,
                        p.away.name,
                        p.kickoff,
                        p.predictedWinner,
                        p.homeWinProbability,
                        p.modelVersion,
                        p.featureHash,
                        now,
                        contribs_json,
                        alts_json,
                        uncertainty_json,
                    ),
                )


def _row_to_out(r: sqlite3.Row) -> PredictionOut:
    # sqlite3.Row exposes column names via .keys(); plain `in r` iterates values.
    cols = tuple(r.keys())
    contribs_raw = r["contributions"] if "contributions" in cols else None
    contribs = (
        [FeatureContribution(**c) for c in json.loads(contribs_raw)] if contribs_raw else None
    )
    alts_raw = r["alternatives"] if "alternatives" in cols else None
    alternatives = AlternativeModels(**json.loads(alts_raw)) if alts_raw else None
    unc_raw = r["uncertainty"] if "uncertainty" in cols else None
    unc = json.loads(unc_raw) if unc_raw else {}
    ci = unc.get("winProbability80ci")
    return PredictionOut(
        matchId=r["match_id"],
        home=TeamInfo(id=r["home_id"], name=r["home_name"]),
        away=TeamInfo(id=r["away_id"], name=r["away_name"]),
        kickoff=r["kickoff"],
        predictedWinner=r["predicted_winner"],
        homeWinProbability=r["home_win_prob"],
        modelVersion=r["model_version"],
        featureHash=r["feature_hash"],
        contributions=contribs,
        alternatives=alternatives,
        baseModelSpread=unc.get("baseModelSpread"),
        winProbability80ci=tuple(ci) if ci else None,
        trainingDataSimilarity=unc.get("trainingDataSimilarity"),
        confidenceBand=unc.get("confidenceBand"),
    )


# ---------------------------------------------------------------------------
# Firestore-backed prediction store (production)
# ---------------------------------------------------------------------------


class FirestorePredictionStore:
    """Firestore-backed prediction cache.

    Document layout: one doc per ``(season, round)``, ID ``"{season}-{round}"``,
    inside the ``predictions`` collection. The doc contains the full list of
    ``PredictionOut`` entries for that round plus a ``createdAt`` timestamp.

    This is the shape the precompute Job writes (twice a week) and the
    ``/predictions`` endpoint reads (on every request). Using one doc per
    round keeps both sides to a single Firestore RPC — cheaper than
    per-match docs, and Firestore's 1 MiB doc limit is ~orders of magnitude
    more than a round's prediction payload.
    """

    _COLLECTION = "predictions"

    def __init__(
        self,
        client: Any = None,
        project: str | None = None,
        database: str = "(default)",
    ) -> None:
        if client is not None:
            self._db = client
        else:
            from google.cloud import firestore  # noqa: PLC0415

            self._db = firestore.Client(project=project, database=database)

    def close(self) -> None:  # symmetry with PredictionStore; Firestore has no conn to close
        return

    def get(self, season: int, round_: int) -> list[PredictionOut]:
        snap = self._db.collection(self._COLLECTION).document(_doc_id(season, round_)).get()
        if not snap.exists:
            return []
        data = snap.to_dict() or {}
        return [PredictionOut(**p) for p in data.get("predictions", [])]

    def put(self, season: int, round_: int, predictions: list[PredictionOut]) -> None:
        self._db.collection(self._COLLECTION).document(_doc_id(season, round_)).set(
            {
                "season": season,
                "round": round_,
                "predictions": [p.model_dump() for p in predictions],
                "createdAt": datetime.now(UTC).isoformat(),
            }
        )


def _doc_id(season: int, round_: int) -> str:
    return f"{season}-{round_}"


def get_prediction_store() -> PredictionStore | FirestorePredictionStore:
    """Factory: pick the prediction store backend from ``STORAGE_BACKEND``.

    Mirrors ``fantasy_coach.config.get_repository`` so API + Job share one
    switch. Defaults to SQLite for local dev; Cloud Run deploy sets
    ``STORAGE_BACKEND=firestore``.
    """
    backend = os.getenv("STORAGE_BACKEND", "sqlite").lower()
    if backend == "firestore":
        return FirestorePredictionStore()
    return PredictionStore()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_model(path: Path) -> None:
    """Ensure a model artefact exists at ``path``, downloading from GCS if not.

    The container image ships without a ``.joblib`` (training requires
    historical data the image doesn't carry). In production, the precompute
    Job sets ``FANTASY_COACH_MODEL_GCS_URI=gs://.../latest.joblib`` and this
    function streams the blob to ``path`` on first miss. Subsequent calls
    inside the same container short-circuit on the local file.

    Raises ``FileNotFoundError`` if the file is missing and no GCS URI is
    configured — the pre-#93 behaviour local dev relies on.
    """
    if path.exists():
        return

    gcs_uri = os.getenv(MODEL_GCS_URI_ENV)
    if not gcs_uri:
        raise FileNotFoundError(path)

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"{MODEL_GCS_URI_ENV} must start with gs:// (got {gcs_uri!r})")
    bucket_name, _, blob_name = gcs_uri.removeprefix("gs://").partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"{MODEL_GCS_URI_ENV} must be gs://<bucket>/<object> (got {gcs_uri!r})")

    from google.cloud import storage  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model from %s to %s", gcs_uri, path)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(path))


def _model_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _feature_hash() -> str:
    from fantasy_coach.feature_engineering import FEATURE_NAMES  # noqa: PLC0415

    return hashlib.sha256(",".join(sorted(FEATURE_NAMES)).encode()).hexdigest()[:12]


def _record_team_list_snapshots(team_list_repo: Any, row: MatchRow) -> None:
    """Persist one snapshot per team for this match, if team lists are present.

    Skips teams whose players array is empty or missing the ``isOnField``
    flag entirely — those scrapes pre-date the team-list drop and carry no
    signal about the starting XIII. Failures are logged and swallowed so
    the precompute loop keeps processing other matches.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from fantasy_coach.team_lists import TeamListSnapshot  # noqa: PLC0415

    now = datetime.now(UTC)
    for side in (row.home, row.away):
        players = side.players
        if not players:
            continue
        if not any(p.is_on_field is not None for p in players):
            continue
        snapshot = TeamListSnapshot(
            season=row.season,
            round=row.round,
            match_id=row.match_id,
            team_id=side.team_id,
            scraped_at=now,
            players=tuple(players),
        )
        try:
            team_list_repo.record_snapshot(snapshot)
        except Exception:
            logger.exception(
                "Failed to record team-list snapshot for match %s team %s",
                row.match_id,
                side.team_id,
            )


# Features whose default / flag value carries no narrative signal. Entries
# map feature name → predicate on the raw value; when the predicate returns
# True the contribution is dropped from the user-facing list. Background:
# #124 — several rows surfaced with nonsensical labels ("Venue typically
# sees 0 combined points per match" for a one-off neutral venue, "Weather
# data available" for every modern match, etc.) because the underlying
# feature value is a sentinel or a binary era flag, not a measurement.
_SENTINEL_PREDICATES: dict[str, Callable[[float], bool]] = {
    "is_home_field": lambda _: True,  # constant 1.0, never a discriminator
    "missing_weather": lambda v: v < 0.5,  # hide when we *do* have weather
    "missing_referee": lambda v: v < 0.5,  # hide when we *do* have a ref
    "venue_avg_total_points": lambda v: v == 0.0,  # no venue history yet
    "venue_home_win_rate": lambda v: v == 0.5,  # neutral prior
    "ref_avg_total_points": lambda v: v == 0.0,  # no ref data + empty league_total
    "ref_home_penalty_diff": lambda v: v == 0.0,  # no ref penalty data
    "key_absence_diff": lambda v: v == 0.0,  # no missing regulars today
    "h2h_recent_diff": lambda v: v == 0.0,  # no recent head-to-head
    "missing_player_strength": lambda v: v < 0.5,  # hide when we *do* have player data
    "missing_odds": lambda v: v < 0.5,  # hide when odds data is present
    "team_venue_hga_estimate": lambda v: v == 0.0,  # no team-venue history yet
    "is_neutral_venue": lambda v: v < 0.5,  # hide when venue is not neutral
}


def _compute_contributions(
    loaded: Any,
    x: Any,
    *,
    top_k: int = 5,
    builder: Any = None,
    match: Any = None,
) -> list[FeatureContribution] | None:
    """Return top-K signed log-odds contributions for the loaded model.

    Both logistic and XGBoost produce per-feature contributions in log-odds
    units — the dispatcher picks whichever path the artefact supports:

    - **Logistic** (``loaded.pipeline`` is a sklearn ``Pipeline`` with
      ``scale`` + ``lr`` steps): ``contribution_i = coef_i × (x_i − mean_i)
      / scale_i``. Sum of contributions + intercept = logit(home_win_prob)
      for the uncalibrated pipeline.

    - **XGBoost** (``loaded.estimator`` is an ``XGBClassifier``): uses the
      booster's built-in ``pred_contribs=True`` mode, which returns per-
      feature contributions to the raw margin (log-odds-equivalent for
      binary classification). Drops the bias column.

    Ensemble artefacts currently return ``None`` — attributing a weighted
    average of two correlated bases back to features doesn't have a clean
    interpretation and nothing in the UI consumes it yet.

    ``builder`` + ``match`` are optional; when both are supplied we enrich
    specific rows with structured ``detail`` (today: the missing-regulars
    list behind ``key_absence_diff``).
    """
    feature_names = getattr(loaded, "feature_names", None)
    if not feature_names:
        return None

    import numpy as np  # noqa: PLC0415

    raw = np.asarray(x, dtype=float).reshape(-1)
    if len(raw) != len(feature_names):
        return None

    contributions = _logistic_raw_contribs(loaded, raw)
    is_logistic = contributions is not None
    if contributions is None:
        contributions = _xgboost_raw_contribs(loaded, x)
    if contributions is None:
        return None

    # For XGBoost, pre-compute the dominant interaction partner per feature
    # so the UI can surface "× partner" sub-rows on the contribution panel.
    interactions: dict[str, tuple[str, float]] | None = None
    if not is_logistic:
        from fantasy_coach.models.explainability import shap_interactions  # noqa: PLC0415

        interactions = shap_interactions(loaded, x)

    # Rank all features by |contribution|, then filter out sentinel/flag
    # rows before slicing to top_k. Filtering *after* the rank keeps us
    # deterministic even if a filtered feature outranks a real one.
    order = np.argsort(-np.abs(contributions))
    picked: list[FeatureContribution] = []
    for idx in order:
        name = feature_names[idx]
        value = float(raw[idx])
        predicate = _SENTINEL_PREDICATES.get(name)
        if predicate is not None and predicate(value):
            continue
        detail = _contribution_detail(name, builder, match)
        if interactions and name in interactions:
            partner, magnitude = interactions[name]
            if detail is None:
                detail = {}
            detail["interaction"] = {"partner": partner, "magnitude": round(magnitude, 4)}
        picked.append(
            FeatureContribution(
                feature=name,
                value=round(value, 6),
                contribution=round(float(contributions[idx]), 4),
                detail=detail,
            )
        )
        if len(picked) >= top_k:
            break
    return picked


def _logistic_raw_contribs(loaded: Any, raw: Any) -> Any | None:
    """Compute per-feature log-odds contributions for a logistic pipeline.

    Returns ``None`` when ``loaded`` isn't a logistic pipeline — caller
    dispatches to the XGBoost path in that case.
    """
    pipeline = getattr(loaded, "pipeline", None)
    if pipeline is None:
        return None
    try:
        scaler = pipeline.named_steps["scale"]
        lr = pipeline.named_steps["lr"]
    except (KeyError, AttributeError):
        return None

    import numpy as np  # noqa: PLC0415

    mean = np.asarray(getattr(scaler, "mean_", None))
    scale = np.asarray(getattr(scaler, "scale_", None))
    coef = np.asarray(getattr(lr, "coef_", None)).reshape(-1)
    if mean.shape != raw.shape or scale.shape != raw.shape or coef.shape != raw.shape:
        return None

    # Guard against a zero-variance column that would have shipped as
    # scale_=0. sklearn normally replaces those with 1.0 but be defensive.
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    return coef * (raw - mean) / safe_scale


def _xgboost_raw_contribs(loaded: Any, x: Any) -> Any | None:
    """Compute per-feature SHAP contributions for an XGBoost estimator.

    Delegates to ``models.explainability.shap_contributions`` which uses the
    booster's ``pred_contribs=True`` mode (TreeSHAP). Returns per-feature
    contributions in raw-margin (log-odds for binary classification) with the
    bias column dropped. Returns ``None`` when the artefact isn't XGBoost or
    xgboost is unavailable.
    """
    from fantasy_coach.models.explainability import shap_contributions  # noqa: PLC0415

    return shap_contributions(loaded, x)


def _contribution_detail(feature: str, builder: Any, match: Any) -> dict[str, Any] | None:
    """Return per-feature structured narrative detail, or None."""
    if builder is None or match is None:
        return None
    if feature == "key_absence_diff":
        home = builder.key_absence_detail(match.home.team_id, match.home.players)
        away = builder.key_absence_detail(match.away.team_id, match.away.players)
        if not home and not away:
            return None
        return {"home_missing": home, "away_missing": away}
    return None


def _apply_market_shrinkage(
    prob: float, x: Any, feature_names: tuple[str, ...]
) -> tuple[float, float | None]:
    """Blend the model's home-win probability with the bookmaker's implied prob.

    Returns ``(final_prob, market_prob)`` where ``market_prob`` is None when
    odds are unavailable for this match (in which case ``final_prob == prob``
    and no shrinkage is applied). When odds are present, the blend is

        final_prob = (1 - w) * prob + w * odds_home_win_prob

    with ``w = MARKET_SHRINKAGE_WEIGHT``. See the constant's docstring for
    why this exists.
    """
    try:
        odds_idx = feature_names.index("odds_home_win_prob")
        missing_idx = feature_names.index("missing_odds")
    except ValueError:
        return prob, None
    import numpy as np  # noqa: PLC0415

    raw = np.asarray(x, dtype=float).reshape(-1)
    if len(raw) <= max(odds_idx, missing_idx):
        return prob, None
    if raw[missing_idx] > 0.5:
        return prob, None
    market = float(raw[odds_idx])
    blended = (1.0 - MARKET_SHRINKAGE_WEIGHT) * prob + MARKET_SHRINKAGE_WEIGHT * market
    return round(blended, 4), market


def _bookmaker_pick_summary(x: Any, feature_names: tuple[str, ...]) -> PickSummary | None:
    """Derive a bookmaker-implied pick from the odds_home_win_prob feature.

    Returns None when odds data is absent (``missing_odds`` = 1.0) or when
    the feature names tuple doesn't contain the expected columns.
    """
    try:
        odds_idx = feature_names.index("odds_home_win_prob")
        missing_idx = feature_names.index("missing_odds")
    except ValueError:
        return None
    import numpy as np  # noqa: PLC0415

    raw = np.asarray(x, dtype=float).reshape(-1)
    if len(raw) <= max(odds_idx, missing_idx):
        return None
    if raw[missing_idx] > 0.5:  # odds not available for this match
        return None
    prob = round(float(raw[odds_idx]), 4)
    return PickSummary(
        predictedWinner="home" if prob >= 0.5 else "away",
        homeWinProbability=prob,
    )


def _try_load_secondary_model(path: Path, gcs_uri_env: str) -> Any | None:
    """Load an optional secondary model artefact; return None on any failure.

    Used for the alternatives panel (#140): loads the logistic artefact
    alongside the primary XGBoost model so both picks can be shown. Returns
    None silently when the file is missing and no GCS URI is configured, so
    the absence of a secondary model never blocks prediction computation.
    """
    try:
        if not path.exists():
            gcs_uri = os.getenv(gcs_uri_env)
            if not gcs_uri:
                return None
            if not gcs_uri.startswith("gs://"):
                logger.warning("Secondary model GCS URI malformed: %s", gcs_uri)
                return None
            bucket_name, _, blob_name = gcs_uri.removeprefix("gs://").partition("/")
            if not bucket_name or not blob_name:
                logger.warning("Secondary model GCS URI malformed: %s", gcs_uri)
                return None
            from google.cloud import storage as gcs  # noqa: PLC0415

            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading secondary model from %s to %s", gcs_uri, path)
            client = gcs.Client()
            client.bucket(bucket_name).blob(blob_name).download_to_filename(str(path))
        from fantasy_coach.models.loader import load_model  # noqa: PLC0415

        return load_model(path)
    except Exception:
        logger.debug(
            "Secondary model not available at %s — alternatives panel will be partial", path
        )
        return None


def _uncertainty_json(p: PredictionOut) -> str | None:
    """Serialise uncertainty fields to a JSON string for SQLite storage."""
    if all(
        v is None
        for v in (
            p.baseModelSpread,
            p.winProbability80ci,
            p.trainingDataSimilarity,
            p.confidenceBand,
        )
    ):
        return None
    return json.dumps(
        {
            "baseModelSpread": p.baseModelSpread,
            "winProbability80ci": list(p.winProbability80ci) if p.winProbability80ci else None,
            "trainingDataSimilarity": p.trainingDataSimilarity,
            "confidenceBand": p.confidenceBand,
        }
    )


def _compute_training_centroid(history: list[MatchRow]) -> Any:
    """Return the mean feature vector across completed historical matches.

    Used as the reference point for training-data similarity (OOD detection).
    Returns None when history is empty. Re-processes history via
    ``build_training_frame`` — acceptable because history is already in memory
    and the walk is O(n) with no I/O.
    """
    import numpy as np  # noqa: PLC0415

    from fantasy_coach.feature_engineering import build_training_frame  # noqa: PLC0415

    frame = build_training_frame(history)
    if frame.X.shape[0] == 0:
        return None
    return np.mean(frame.X, axis=0)


def _cosine_similarity(x: Any, centroid: Any) -> float:
    """Cosine similarity between a 1-d feature vector and the training centroid.

    Returns 1.0 when vectors are identical, 0.0 when orthogonal. Clipped to
    [0, 1] since feature vectors are mostly non-negative and near-orthogonal
    vectors are rare in practice.
    """
    import numpy as np  # noqa: PLC0415

    v = np.asarray(x, dtype=float).flatten()
    c = np.asarray(centroid, dtype=float).flatten()
    norm_v = np.linalg.norm(v)
    norm_c = np.linalg.norm(c)
    if norm_v == 0 or norm_c == 0:
        return 0.0
    return float(np.clip(np.dot(v, c) / (norm_v * norm_c), 0.0, 1.0))


# Uncertainty thresholds — heuristic pending held-out calibration.
# At `high`: spread ≤ 0.10 and OOD similarity ≥ 0.80 → model is confident
#            and the matchup is in-distribution.
# At `low`: spread > 0.20 or OOD similarity < 0.50 → either base models
#           strongly disagree or the matchup is unusual.
_SPREAD_HIGH = 0.10
_SPREAD_MEDIUM = 0.20
_OOD_HIGH = 0.80
_OOD_MEDIUM = 0.50


def _confidence_band(spread: float, ood_similarity: float) -> str:
    if spread <= _SPREAD_HIGH and ood_similarity >= _OOD_HIGH:
        return "high"
    if spread <= _SPREAD_MEDIUM and ood_similarity >= _OOD_MEDIUM:
        return "medium"
    return "low"


def _build_inference_state(history: list[MatchRow]) -> Any:
    from fantasy_coach.feature_engineering import FeatureBuilder  # noqa: PLC0415

    builder = FeatureBuilder()
    for match in sorted(history, key=lambda m: (m.start_time, m.match_id)):
        if match.home.score is None or match.away.score is None:
            continue
        builder.advance_season_if_needed(match)
        builder.record(match)
    return builder


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_predictions(
    season: int,
    round_: int,
    repo: Any,
    store: PredictionStore,
    *,
    model_path: Path | None = None,
    logistic_path: Path | None = None,
    fetch_round_fn: Any = fetch_round,
    fetch_match_fn: Any = fetch_match_from_url,
    force: bool = False,
    team_list_repo: Any = None,
    weather_forecasts: dict[int, Any] | None = None,
) -> list[PredictionOut]:
    """Return predictions for season/round; compute and cache them on miss.

    ``force=True`` bypasses the cache-hit short-circuit so the precompute
    Job can refresh stored predictions when team lists change between
    scheduled runs. On cache miss ``force`` is a no-op.

    ``team_list_repo`` is optional. When provided (production path via
    ``_run_precompute``), each scraped match with populated team-list data
    appends a snapshot per side so downstream model features (#27) can
    diff named-squad vs kickoff lineup across scrapes. Silently skipped
    when ``None`` — tests that don't exercise the snapshot path pass None.

    ``weather_forecasts`` is an optional ``{match_id: WeatherForecast}`` dict
    populated by the precompute Job (#207). When provided, upcoming matches
    use the forecast for ``is_wet``, ``wind_kph``, ``temperature_c``, and the
    new ``rain_intensity`` / ``weather_source`` features instead of the
    post-match NRL payload (which is absent before kickoff).

    Raises ``FileNotFoundError`` if the model artefact does not exist (caller
    should surface this as HTTP 503).
    """
    if not force:
        cached = store.get(season, round_)
        if cached:
            logger.info(
                "Cache hit: returning %d stored predictions for %d r%d",
                len(cached),
                season,
                round_,
            )
            return cached

    # Resolve and load the primary model — fetching from GCS on first miss if
    # FANTASY_COACH_MODEL_GCS_URI is set (production path).
    from fantasy_coach.models.loader import load_model  # noqa: PLC0415

    path = model_path or Path(os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))
    _ensure_model(path)

    loaded = load_model(path)
    mv = _model_version(path)
    fh = _feature_hash()

    # Load the secondary logistic model for the three-way consensus panel (#140).
    # Returns None when unconfigured — alternatives.logistic will be absent.
    log_path = logistic_path or Path(os.getenv(LOGISTIC_PATH_ENV, DEFAULT_LOGISTIC_PATH))
    logistic_loaded = None
    if log_path != path:  # skip if operator pointed both models at the same artefact
        logistic_loaded = _try_load_secondary_model(log_path, LOGISTIC_GCS_URI_ENV)

    # Scrape fixtures for the requested round
    round_payload = fetch_round_fn(season, round_)
    fixtures = (round_payload or {}).get("fixtures") or []
    logger.info("Scraping %d fixtures for %d r%d", len(fixtures), season, round_)

    round_matches: list[MatchRow] = []
    for fixture in fixtures:
        url = fixture.get("matchCentreUrl")
        if not url:
            continue
        try:
            raw = fetch_match_fn(url)
        except Exception:
            logger.exception("Failed to fetch match from %s", url)
            continue
        if raw is None:
            continue
        try:
            row = extract_match_features(raw)
            repo.upsert_match(row)
            round_matches.append(row)
            if team_list_repo is not None:
                _record_team_list_snapshots(team_list_repo, row)
        except Exception:
            logger.exception("Failed to extract/store match from %s", url)

    if not round_matches:
        return []

    # Build feature state from prior completed matches (current season + up to 3 prior seasons)
    history: list[MatchRow] = []
    for s in range(max(2020, season - 3), season + 1):
        try:
            season_matches = repo.list_matches(s)
        except Exception:
            continue
        for m in season_matches:
            if m.season < season or (m.season == season and m.round < round_):
                history.append(m)

    builder = _build_inference_state(history)

    import numpy as np  # noqa: PLC0415

    from fantasy_coach.feature_engineering import FEATURE_NAMES  # noqa: PLC0415

    _fn = getattr(loaded, "feature_names", None)
    feature_names: tuple[str, ...] = _fn if isinstance(_fn, tuple) else FEATURE_NAMES

    # Training centroid for OOD / uncertainty estimation (#146).
    # Computed from the same history used for the inference builder so there
    # is no leakage — the centroid only sees completed pre-round matches.
    training_centroid = _compute_training_centroid(history)

    predictions: list[PredictionOut] = []
    _wx_forecasts = weather_forecasts or {}
    for match in sorted(round_matches, key=lambda m: (m.start_time, m.match_id)):
        wx_forecast = _wx_forecasts.get(match.match_id)
        x = np.asarray([builder.feature_row(match, weather_forecast=wx_forecast)], dtype=float)
        raw_prob = round(float(loaded.predict_home_win_prob(x)[0]), 4)
        prob, _ = _apply_market_shrinkage(raw_prob, x, feature_names)

        # Build the three-way consensus alternatives (#140).
        logistic_pick: PickSummary | None = None
        if logistic_loaded is not None:
            lprob = round(float(logistic_loaded.predict_home_win_prob(x)[0]), 4)
            logistic_pick = PickSummary(
                predictedWinner="home" if lprob >= 0.5 else "away",
                homeWinProbability=lprob,
            )
        bm_pick = _bookmaker_pick_summary(x, feature_names)
        alternatives = (
            AlternativeModels(logistic=logistic_pick, bookmaker=bm_pick)
            if (logistic_pick is not None or bm_pick is not None)
            else None
        )

        # Uncertainty estimation (#146).
        all_probs = [prob]
        if logistic_pick is not None:
            all_probs.append(logistic_pick.homeWinProbability)
        if bm_pick is not None:
            all_probs.append(bm_pick.homeWinProbability)
        spread = round(float(max(all_probs) - min(all_probs)), 4)
        ci_lo = round(max(0.0, prob - spread / 2), 4)
        ci_hi = round(min(1.0, prob + spread / 2), 4)
        ood_sim = (
            round(_cosine_similarity(x[0], training_centroid), 4)
            if training_centroid is not None
            else 1.0
        )
        band = _confidence_band(spread, ood_sim)

        predictions.append(
            PredictionOut(
                matchId=match.match_id,
                home=TeamInfo(id=match.home.team_id, name=match.home.name),
                away=TeamInfo(id=match.away.team_id, name=match.away.name),
                kickoff=match.start_time.isoformat(),
                predictedWinner="home" if prob >= 0.5 else "away",
                homeWinProbability=prob,
                modelVersion=mv,
                featureHash=fh,
                contributions=_compute_contributions(loaded, x, builder=builder, match=match),
                alternatives=alternatives,
                baseModelSpread=spread,
                winProbability80ci=(ci_lo, ci_hi),
                trainingDataSimilarity=ood_sim,
                confidenceBand=band,
            )
        )

    store.put(season, round_, predictions)
    logger.info("Computed and cached %d predictions for %d r%d", len(predictions), season, round_)
    return predictions
