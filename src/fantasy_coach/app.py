import os

from fastapi import FastAPI

from fantasy_coach import __version__
from fantasy_coach.auth import FirebaseAuthMiddleware

app = FastAPI(title="Fantasy Coach", version=__version__)

# Enable Firebase token verification when a project ID is configured.
# Omitting FIREBASE_PROJECT_ID disables the check (useful for local SQLite dev),
# but the middleware is always installed — a missing project ID will reject all
# protected requests rather than silently pass them.
if os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"):
    app.add_middleware(FirebaseAuthMiddleware)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
