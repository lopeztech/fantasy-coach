import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from fantasy_coach import __version__
from fantasy_coach.auth import FirebaseAuthMiddleware
from fantasy_coach.config import get_repository
from fantasy_coach.predictions import (
    FirestorePredictionStore,
    PredictionOut,
    PredictionStore,
    get_prediction_store,
)
from fantasy_coach.storage.repository import Repository

_ACCURACY_THRESHOLD = 0.55


class RoundAccuracy(BaseModel):
    season: int
    round: int
    modelVersion: str
    total: int
    correct: int
    accuracy: float


class ModelVersionAccuracy(BaseModel):
    modelVersion: str
    total: int
    correct: int
    accuracy: float


class AccuracyOut(BaseModel):
    rounds: list[RoundAccuracy]
    byModelVersion: list[ModelVersionAccuracy]
    overallAccuracy: float | None
    belowThreshold: bool
    threshold: float
    scoredMatches: int

ALLOWED_ORIGINS_ENV = "FANTASY_COACH_ALLOWED_ORIGINS"
DEFAULT_ALLOWED_ORIGINS = (
    "https://fantasy.lopezcloud.dev,http://localhost:5173,http://localhost:4173"
)


def _allowed_origins() -> list[str]:
    raw = os.getenv(ALLOWED_ORIGINS_ENV, DEFAULT_ALLOWED_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Fantasy Coach",
    version=__version__,
    description="NRL match prediction API. All non-health endpoints require a Firebase ID token.",
)

# Enable Firebase token verification when a project ID is configured.
# Omitting FIREBASE_PROJECT_ID disables the check (useful for local SQLite dev).
#
# Middleware ordering: the last-added middleware runs first on inbound requests,
# so CORSMiddleware must be added after FirebaseAuthMiddleware. This lets the
# browser's preflight OPTIONS request short-circuit inside CORS before auth
# sees it — auth middleware only understands Bearer tokens and would 401 the
# preflight otherwise.
if os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"):
    app.add_middleware(FirebaseAuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

# Module-level singletons — created lazily on first use.
_store: PredictionStore | FirestorePredictionStore | None = None
_repo: Repository | None = None


def _get_store() -> PredictionStore | FirestorePredictionStore:
    global _store
    if _store is None:
        _store = get_prediction_store()
    return _store


def _get_repo() -> Repository:
    global _repo
    if _repo is None:
        _repo = get_repository()
    return _repo


def _annotate_results(
    predictions: list[PredictionOut], season: int, round_: int
) -> list[PredictionOut]:
    """Attach actualWinner from the match repo (FullTime matches only)."""
    try:
        matches = {m.match_id: m for m in _get_repo().list_matches(season, round_)}
    except Exception:
        return predictions  # repo unavailable — return predictions as-is
    result = []
    for p in predictions:
        m = matches.get(p.matchId)
        if (
            m
            and m.match_state == "FullTime"
            and m.home.score is not None
            and m.away.score is not None
        ):
            winner = "home" if m.home.score > m.away.score else "away"
            p = p.model_copy(update={"actualWinner": winner})
        result.append(p)
    return result



@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get(
    "/predictions",
    response_model=list[PredictionOut],
    summary="Get predictions for a season/round",
    description=(
        "Returns predicted winner and home-win probability for every match in "
        "the requested round. Predictions are precomputed twice a week by a "
        "Cloud Run Job (Tue 09:00 and Thu 06:00 AEST); this endpoint is a "
        "cache read only. Returns 503 with a retry hint if the cache is empty."
    ),
)
def get_predictions(
    season: int = Query(..., description="NRL season year, e.g. 2026"),
    round: int = Query(..., description="Round number, e.g. 7", alias="round"),
) -> list[PredictionOut]:
    cached = _get_store().get(season, round)
    if not cached:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No cached predictions for season {season} round {round}. "
                "The precompute job runs Tue 09:00 AEST and Thu 06:00 AEST. "
                "Retry in a few minutes or trigger it manually with "
                "`gcloud run jobs execute fantasy-coach-precompute`."
            ),
        )
    return _annotate_results(cached, season, round)


@app.get(
    "/accuracy",
    response_model=AccuracyOut,
    summary="Rolling model accuracy over recent rounds",
    description=(
        "Returns per-round and per-model-version accuracy for the last N completed "
        "rounds of the given season. A round is considered complete when all its "
        "matches have match_state=FullTime in the match store."
    ),
)
def get_accuracy(
    season: int = Query(..., description="NRL season year, e.g. 2026"),
    last_n_rounds: int = Query(
        default=10, ge=1, le=27, description="How many recent completed rounds to include"
    ),
) -> AccuracyOut:
    try:
        matches = _get_repo().list_matches(season)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Match store unavailable: {exc}") from exc

    # Actual winner per match_id (FullTime only)
    results: dict[int, str] = {}
    completed_match_ids_by_round: dict[int, list[int]] = {}
    for m in matches:
        if (
            m.match_state == "FullTime"
            and m.home.score is not None
            and m.away.score is not None
        ):
            winner = "home" if m.home.score > m.away.score else "away"
            results[m.match_id] = winner
            completed_match_ids_by_round.setdefault(m.round, []).append(m.match_id)

    # Take the most-recent N rounds that have at least one FullTime result
    completed_rounds = sorted(completed_match_ids_by_round.keys(), reverse=True)[:last_n_rounds]
    completed_rounds.reverse()  # oldest-first for charting

    round_accuracy_list: list[RoundAccuracy] = []
    mv_stats: dict[str, dict[str, int]] = {}
    total_correct = 0
    total_scored = 0

    for round_ in completed_rounds:
        preds = _get_store().get(season, round_)
        if not preds:
            continue

        # Dominant model version for this round (by plurality of predictions)
        mv_counts: dict[str, int] = {}
        for p in preds:
            mv_counts[p.modelVersion] = mv_counts.get(p.modelVersion, 0) + 1
        model_version = max(mv_counts, key=lambda k: mv_counts[k])

        scored = [(p, results[p.matchId]) for p in preds if p.matchId in results]
        n_total = len(scored)
        n_correct = sum(1 for p, actual in scored if p.predictedWinner == actual)
        accuracy = n_correct / n_total if n_total > 0 else 0.0

        round_accuracy_list.append(
            RoundAccuracy(
                season=season,
                round=round_,
                modelVersion=model_version,
                total=n_total,
                correct=n_correct,
                accuracy=accuracy,
            )
        )

        for p, actual in scored:
            mv = p.modelVersion
            if mv not in mv_stats:
                mv_stats[mv] = {"total": 0, "correct": 0}
            mv_stats[mv]["total"] += 1
            if p.predictedWinner == actual:
                mv_stats[mv]["correct"] += 1

        total_correct += n_correct
        total_scored += n_total

    overall_accuracy = total_correct / total_scored if total_scored > 0 else None

    by_model_version = [
        ModelVersionAccuracy(
            modelVersion=mv,
            total=s["total"],
            correct=s["correct"],
            accuracy=s["correct"] / s["total"] if s["total"] > 0 else 0.0,
        )
        for mv, s in mv_stats.items()
    ]

    return AccuracyOut(
        rounds=round_accuracy_list,
        byModelVersion=by_model_version,
        overallAccuracy=overall_accuracy,
        belowThreshold=(
            overall_accuracy is not None and overall_accuracy < _ACCURACY_THRESHOLD
        ),
        threshold=_ACCURACY_THRESHOLD,
        scoredMatches=total_scored,
    )
