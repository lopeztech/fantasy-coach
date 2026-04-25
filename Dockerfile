# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy the lockfile first so the dependency-install layer is cached across
# source-only changes; install deps without the project source.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

COPY src/ ./src/
COPY deploy/ ./deploy/
COPY assets/ ./assets/
# Tuned XGBoost hyperparameters (#167). train_xgboost +
# XGBoostPredictor.fit both read this path via load_best_params();
# without it, they fall back to the hand-picked grid (pre-HPO behaviour).
# The .joblib blobs in artifacts/ are left out deliberately — model
# artefacts are downloaded from GCS at runtime, not baked into images.
COPY artifacts/best_params.json ./artifacts/best_params.json
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

FROM python:3.12-slim AS runtime

# Liberation Sans fonts for OG image card rendering (og_image.py). Adds ~1.5 MB.
RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn fantasy_coach.app:app --host 0.0.0.0 --port ${PORT} --log-config deploy/log-config.json"]
