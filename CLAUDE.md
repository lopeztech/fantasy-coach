# Instructions for Claude Code

This file is loaded into every Claude session (including Routines-triggered
runs) so agents don't have to re-scan the repo. Keep it dense and current —
point at the source of truth instead of duplicating it.

## What this repo is

**Fantasy Coach** — NRL (rugby league) match prediction service.
Public URL: <https://fantasy.lopezcloud.dev>.

End-to-end pipeline:

```
nrl.com JSON  →  scraper  →  SQLite/Firestore (matches)  →  feature engineering
                                                                    │
                                     ┌──────────────────────────────┘
                                     ▼
                      logistic / XGBoost / Elo / Skellam models
                                     │
                                     ▼
                     Cloud Run Job (precompute, Tue+Thu)
                                     │
                                     ▼
                   Firestore (predictions cache)  ──►  FastAPI /predictions
                                                              │
                                                              ▼
                                           Vite+React SPA on Firebase Hosting
```

The API **never scrapes on the hot path** (#65). A Cloud Run Job
(`fantasy-coach-precompute`) runs the scrape+feature+predict pipeline Tue
09:00 AEST and Thu 06:00 AEST and writes results to Firestore; the API is a
cache read. On miss, `/predictions` returns `503` with a retry hint.

## Stack

- **Python 3.12+** managed by [uv](https://docs.astral.sh/uv/). Single `pyproject.toml`.
- **FastAPI + uvicorn** for the API.
- **scikit-learn + xgboost** for models; production artefact is XGBoost as of #136.
- **Storage**: SQLite locally, **Firestore in production**. `STORAGE_BACKEND={sqlite,firestore}` picks one.
- **SPA**: Vite + React + TypeScript in `web/`, deploys to Firebase Hosting.
- **Hosting**: Cloud Run (API + Job) in `australia-southeast1` (Sydney). Scale-to-zero.
- **Auth**: Firebase ID tokens verified in `FirebaseAuthMiddleware`; `/healthz` is the only open path.
- **Infra**: Terraform lives in [`lopeztech/platform-infra`](https://github.com/lopeztech/platform-infra) under `projects/fantasy-coach/` — **not in this repo**. DNS for `*.lopezcloud.dev` is in Cloudflare (not Cloud DNS).

## Repo map

| Path | What's there |
|------|--------------|
| `src/fantasy_coach/app.py` | FastAPI app. Routes: `/healthz`, `/predictions`, `/accuracy`, `/teams/{id}/form`. CORS + auth middleware wired here. |
| `src/fantasy_coach/__main__.py` | `python -m fantasy_coach` CLI: `backfill`, `train-logistic`, `train-xgboost`, `evaluate`, `precompute`, `copy-matches-to-firestore`, `merge-closing-lines`. |
| `src/fantasy_coach/scraper.py` | Throttled nrl.com scrapers (`fetch_round`, `fetch_match_from_url`). See `docs/nrl-endpoints.md`. |
| `src/fantasy_coach/features.py` | Extracts `MatchRow` from raw match JSON (no raw JSON is persisted). |
| `src/fantasy_coach/feature_engineering.py` | `FeatureBuilder` + `build_training_frame` + `FEATURE_NAMES`. All features are home-minus-away, walk-forward, no leakage. |
| `src/fantasy_coach/models/` | `elo.py`, `elo_mov.py` (default rater), `logistic.py`, `xgboost_model.py` (**production**), `skellam.py` (Poisson margin), `ensemble.py`, `calibration.py`, `player_ratings.py`, `rating_sweep.py`, `loader.py` (dispatches by `model_type`). |
| `src/fantasy_coach/predictions.py` | `PredictionStore` (SQLite), `FirestorePredictionStore`, `compute_predictions`, `PredictionOut`, `_compute_contributions` (per-feature log-odds attribution). |
| `src/fantasy_coach/storage/` | `Repository` interface, `SQLiteRepository`, `FirestoreRepository`, `schema.sql`, `team_list.py`. |
| `src/fantasy_coach/commentary/` | Gemini-powered match-preview text + cache. |
| `src/fantasy_coach/evaluation/` | Walk-forward harness, predictors (home/elo/logistic/bookmaker), markdown report writer. |
| `src/fantasy_coach/bookmaker/` | Closing-line parser (aussportsbetting xlsx), team-name canonicalisation, `BookmakerPredictor`. |
| `src/fantasy_coach/auth.py` | `FirebaseAuthMiddleware`. |
| `src/fantasy_coach/config.py` | `get_repository()` — picks SQLite vs Firestore from env. |
| `src/fantasy_coach/backfill.py` | Season-bulk scraper with idempotent state (`*.backfill.json`) + retry log. |
| `src/fantasy_coach/travel.py`, `weather.py`, `team_lists.py` | Per-feature helpers. |
| `tests/` | Pytest. One file per module; fixtures in `tests/fixtures/`. `test_baseline_metrics.py` pins walk-forward numbers — regressions fail CI. |
| `web/src/` | SPA: `App.tsx`, `routes/{Home,Round,MatchDetail,Accuracy,Scoreboard}.tsx`, `components/*`, `api.ts`, `auth.tsx`, `firebase.ts`. |
| `data/` | Local SQLite (`nrl.db`) + backfill sidecar + `venues.csv` + `odds/` xlsx. Gitignored where appropriate. |
| `artifacts/` | Local trained model joblibs. Prod reads from GCS. |
| `docs/` | Authoritative long-form docs (see next section). |
| `scripts/migrate_sqlite_to_firestore.py` | One-off migration. |
| `.github/workflows/` | `ci.yml` (lint+test), `deploy.yml` (Cloud Run), `web-ci.yml`, `web-deploy.yml`. |

## Authoritative docs (read before touching these areas)

| Concern | Doc |
|---------|-----|
| Model architecture, features, ablations, production choice | [`docs/model.md`](docs/model.md) |
| Firestore + SQLite schema | [`docs/data-model.md`](docs/data-model.md) |
| Cloud Run deploy, runtime sizing, auth model, rollback | [`docs/deploy.md`](docs/deploy.md) |
| SPA dev/build/deploy, CORS, routes | [`docs/spa.md`](docs/spa.md) |
| Secret Manager + env var naming | [`docs/secrets.md`](docs/secrets.md) |
| Cost projections, budgets, CI caching | [`docs/cost.md`](docs/cost.md) |
| nrl.com endpoint contract + quirks | [`docs/nrl-endpoints.md`](docs/nrl-endpoints.md) |

These docs are kept up to date alongside code changes — prefer linking to them
from PR descriptions over duplicating their content.

## Commands

```bash
make install         # uv sync + pre-commit install
make test            # pytest
make lint            # ruff format + check --fix
make run             # uvicorn on :8080
make docker-build    # Cloud Run image
make docker-run      # run the image locally

python -m fantasy_coach backfill --season 2024
python -m fantasy_coach train-xgboost --season 2024 --season 2025
python -m fantasy_coach evaluate --model elo --model logistic --model xgboost --seasons 2024,2025
python -m fantasy_coach precompute   # what the Cloud Run Job runs
python -m fantasy_coach merge-closing-lines --xlsx data/odds/nrl.xlsx
```

SPA: `cd web && npm install && npm run dev` (Vite on :5173).

## Environment

`.env.example` is the canonical list. Key vars:

- `STORAGE_BACKEND` — `sqlite` (default) or `firestore`.
- `FANTASY_COACH_DB_PATH` — SQLite file path.
- `FIREBASE_PROJECT_ID` — activates auth middleware + Firestore project selection.
- `FANTASY_COACH_MODEL_PATH`, `FANTASY_COACH_MODEL_GCS_URI` — model load path; Job downloads from GCS to `/tmp` on cold start.
- `FANTASY_COACH_PREDICTIONS_DB_PATH` — SQLite predictions cache.
- `FANTASY_COACH_ALLOWED_ORIGINS` — CORS override.
- `FANTASY_COACH_SCRAPE_INTERVAL_SECONDS` — scrape throttle (default 1.0).

Cloud Run Job env is managed by `deploy.yml` (image + env) and Terraform
(fixed config with `ignore_changes` on env). **Both must stay aligned** —
see the memory note "Terraform can wipe Cloud Run runtime config".

## Conventions

- **No leakage in features.** Every feature is computed from state strictly
  before `start_time`. `FeatureBuilder` is the state machine that guarantees
  this — don't bypass it.
- **`FEATURE_NAMES` is the schema.** `load_model` refuses artefacts whose
  saved feature list doesn't match the current `FEATURE_NAMES`; adding a
  feature means retraining. Test assertions use `len(FEATURE_NAMES)`, not
  literal counts (see memory note on stale-branch test breakage).
- **Response models use camelCase** (`N815` ignored in `predictions.py` +
  `app.py`). sklearn models use capital `X`/`C` (`N803`/`N806` ignored).
- **Model artefacts are joblib blobs** with `{"pipeline": ..., "feature_names": ..., "model_type": ...}`. `models.loader.load_model` dispatches by `model_type`, so swapping the file at the GCS path is the only promotion step.
- **Walk-forward metrics are pinned** in `tests/test_baseline_metrics.py`. If you change features or models, expect to update the `EXPECTED` dict, and include the before/after in the PR.
- **Ruff, pytest, pre-commit** all run in CI. Don't bypass hooks.
- **Lint line length = 100.**

## Deploy model (quick ref)

- Push to `main` touching `src/`, `Dockerfile`, `pyproject.toml`, `uv.lock`, or
  `deploy.yml` → GitHub Actions builds image, pushes to Artifact Registry,
  rolls a new Cloud Run revision, updates the precompute Job image.
- SPA: `web-deploy.yml` handles Firebase Hosting.
- Secrets (`FANTASY_COACH_WIF_PROVIDER`, `FANTASY_COACH_DEPLOYER_SA`,
  `FANTASY_COACH_PROJECT_ID`) are GitHub repo secrets, sourced from
  platform-infra outputs.
- Manual deploy escape hatch + rollback command in `docs/deploy.md`.
- **After every push/PR/merge**: enumerate workflows, pull failed logs before
  declaring done. Don't rely on the happy path (memory note).

## GCP resources (platform-infra is source of truth)

- Project: `fantasy-coach-lcd`
- Region: `australia-southeast1`
- Cloud Run service: `fantasy-coach-api`
- Cloud Run Job: `fantasy-coach-precompute`
- Artifact Registry repo: `fantasy-coach` (`australia-southeast1-docker.pkg.dev/fantasy-coach-lcd/fantasy-coach/api:<sha>`)
- Model bucket: `gs://fantasy-coach-lcd-models/logistic/latest.joblib` (name kept `logistic` for historical reasons; currently serves the XGBoost artefact)
- Firestore: `(default)` DB, collections `matches`, `predictions`, `team_lists`
- Runtime SA: `fantasy-coach-runtime@fantasy-coach-lcd.iam.gserviceaccount.com`
- Deployer SA: `fantasy-coach-deployer@fantasy-coach-lcd.iam.gserviceaccount.com`
- Budget: $20/mo, alerts to `admin@lopezcloud.dev`.

## Working style preferences (from memory, so new sessions don't re-learn them)

- **Stack PRs freely** by dependency; mention the choice in PR body, don't ask.
- **Don't pause between stacked PRs** during a backlog sweep.
- **Re-target stacked PR bases to `main` before merging the bottom**, or downstream PRs auto-close irretrievably.
- **Merge and deploy without asking** once CI is green on a Claude-opened PR — then verify.
- **Label closed-by-Claude issues `claude-vs`** when moving them to Done.
- **Check all workflows after pushes**, including paired platform-infra runs. `platform-infra`'s plan-on-PR check is pre-broken (WIF auth); apply-on-merge still works.

## Kanban status sync

When you start work on a GitHub issue (either because the user named it, or
because you picked the next backlog item autonomously), move that issue to
**In Progress** on the `Fantasy Coach — Backlog` project *before* opening the
first PR. When the PR that closes the issue merges to `main`, move the issue
to **Done**. Don't wait for the user to ask — this is how they see what you're
working on.

Commands — IDs are pre-discovered, don't re-look-them-up each turn:

```bash
# Project:       PVT_kwHOAIfoRM4BVIQ8    (lopeztech/projects/10)
# Status field:  PVTSSF_lAHOAIfoRM4BVIQ8zhQmJ_s
# Options:       Todo=f75ad846, In Progress=47fc9ee4, Done=98236657

# Find the project item ID for an issue number:
ITEM_ID=$(gh project item-list 10 --owner lopeztech --format json --limit 200 \
  | jq -r --argjson n "$ISSUE_NUMBER" \
      '.items[] | select(.content.number == $n) | .id')

# Move it. Swap --single-select-option-id for the target column.
gh project item-edit \
  --id "$ITEM_ID" \
  --project-id PVT_kwHOAIfoRM4BVIQ8 \
  --field-id PVTSSF_lAHOAIfoRM4BVIQ8zhQmJ_s \
  --single-select-option-id 47fc9ee4   # In Progress (use 98236657 for Done)
```

If an issue isn't on the board yet (rare — most new issues get auto-added):
`gh project item-add 10 --owner lopeztech --url <issue-url>`.
