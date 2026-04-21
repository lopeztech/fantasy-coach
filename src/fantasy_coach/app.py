import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from fantasy_coach import __version__
from fantasy_coach.auth import FirebaseAuthMiddleware
from fantasy_coach.config import get_repository
from fantasy_coach.predictions import PredictionOut, PredictionStore, compute_predictions

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

# Module-level prediction store — created lazily on first use.
_store: PredictionStore | None = None


def _get_store() -> PredictionStore:
    global _store
    if _store is None:
        _store = PredictionStore()
    return _store


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get(
    "/predictions",
    response_model=list[PredictionOut],
    summary="Get predictions for a season/round",
    description=(
        "Returns predicted winner and home-win probability for every match in "
        "the requested round. Results are cached on first call; subsequent calls "
        "return stored predictions without re-scraping."
    ),
)
def get_predictions(
    season: int = Query(..., description="NRL season year, e.g. 2026"),
    round: int = Query(..., description="Round number, e.g. 7", alias="round"),
) -> list[PredictionOut]:
    repo = get_repository()
    try:
        return compute_predictions(season, round, repo, _get_store())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not available ({exc}). "
                "Train one with: python -m fantasy_coach train-logistic"
            ),
        ) from exc
    finally:
        if hasattr(repo, "close"):
            repo.close()
