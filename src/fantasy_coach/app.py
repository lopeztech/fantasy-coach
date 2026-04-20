from fastapi import FastAPI

from fantasy_coach import __version__

app = FastAPI(title="Fantasy Coach", version=__version__)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
