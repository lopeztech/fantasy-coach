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

import hashlib
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder
from fantasy_coach.features import MatchRow, extract_match_features
from fantasy_coach.models.logistic import load_model
from fantasy_coach.scraper import fetch_match_from_url, fetch_round

logger = logging.getLogger(__name__)

PREDICTIONS_DB_ENV = "FANTASY_COACH_PREDICTIONS_DB_PATH"
MODEL_PATH_ENV = "FANTASY_COACH_MODEL_PATH"
DEFAULT_MODEL_PATH = "artifacts/logistic.joblib"
DEFAULT_PREDICTIONS_DB = "data/predictions.db"

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
    PRIMARY KEY (season, round, match_id)
);
"""


# ---------------------------------------------------------------------------
# Response schema (also used for OpenAPI docs via FastAPI)
# ---------------------------------------------------------------------------


class TeamInfo(BaseModel):
    id: int
    name: str


class PredictionOut(BaseModel):
    matchId: int
    home: TeamInfo
    away: TeamInfo
    kickoff: str  # ISO 8601 UTC
    predictedWinner: str  # "home" | "away"
    homeWinProbability: float
    modelVersion: str  # first 12 hex chars of SHA-256 of the model file
    featureHash: str  # first 12 hex chars of SHA-256 of feature names


# ---------------------------------------------------------------------------
# Prediction store (SQLite-backed cache)
# ---------------------------------------------------------------------------


class PredictionStore:
    """SQLite-backed cache for computed predictions."""

    def __init__(self, path: str | Path | None = None) -> None:
        db_path = Path(path or os.getenv(PREDICTIONS_DB_ENV, DEFAULT_PREDICTIONS_DB))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE)

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
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO predictions
                        (season, round, match_id,
                         home_id, home_name, away_id, away_name,
                         kickoff, predicted_winner, home_win_prob,
                         model_version, feature_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )


def _row_to_out(r: sqlite3.Row) -> PredictionOut:
    return PredictionOut(
        matchId=r["match_id"],
        home=TeamInfo(id=r["home_id"], name=r["home_name"]),
        away=TeamInfo(id=r["away_id"], name=r["away_name"]),
        kickoff=r["kickoff"],
        predictedWinner=r["predicted_winner"],
        homeWinProbability=r["home_win_prob"],
        modelVersion=r["model_version"],
        featureHash=r["feature_hash"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _feature_hash() -> str:
    return hashlib.sha256(",".join(sorted(FEATURE_NAMES)).encode()).hexdigest()[:12]


def _build_inference_state(history: list[MatchRow]) -> FeatureBuilder:
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
    fetch_round_fn: Any = fetch_round,
    fetch_match_fn: Any = fetch_match_from_url,
) -> list[PredictionOut]:
    """Return predictions for season/round; compute and cache them on miss.

    Raises ``FileNotFoundError`` if the model artefact does not exist (caller
    should surface this as HTTP 503).
    """
    cached = store.get(season, round_)
    if cached:
        logger.info(
            "Cache hit: returning %d stored predictions for %d r%d", len(cached), season, round_
        )
        return cached

    # Resolve and load the model
    path = model_path or Path(os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))
    if not path.exists():
        raise FileNotFoundError(path)

    loaded = load_model(path)
    mv = _model_version(path)
    fh = _feature_hash()

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

    predictions: list[PredictionOut] = []
    for match in sorted(round_matches, key=lambda m: (m.start_time, m.match_id)):
        x = np.asarray([builder.feature_row(match)], dtype=float)
        prob = round(float(loaded.predict_home_win_prob(x)[0]), 4)
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
            )
        )

    store.put(season, round_, predictions)
    logger.info("Computed and cached %d predictions for %d r%d", len(predictions), season, round_)
    return predictions
