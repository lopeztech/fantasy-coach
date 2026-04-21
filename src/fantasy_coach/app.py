import os

from fastapi import FastAPI, HTTPException, Query

from fantasy_coach import __version__
from fantasy_coach.auth import FirebaseAuthMiddleware
from fantasy_coach.config import get_repository
from fantasy_coach.predictions import PredictionOut, PredictionStore, compute_predictions

app = FastAPI(
    title="Fantasy Coach",
    version=__version__,
    description="NRL match prediction API. All non-health endpoints require a Firebase ID token.",
)

# Enable Firebase token verification when a project ID is configured.
# Omitting FIREBASE_PROJECT_ID disables the check (useful for local SQLite dev).
if os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"):
    app.add_middleware(FirebaseAuthMiddleware)

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
