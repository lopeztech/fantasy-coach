# Fantasy Coach

NRL match predictions: scraper → feature extraction → baseline model → web API.

Backlog: <https://github.com/users/lopeztech/projects/10>

## Stack

- **Language**: Python 3.12+
- **Dependency manager**: [uv](https://docs.astral.sh/uv/)
- **Web**: FastAPI + uvicorn (prediction API)
- **Storage**: SQLite locally, Firestore in production
- **Deployment**: Cloud Run (GCP). Terraform lives in [`lopeztech/platform-infra`](https://github.com/lopeztech/platform-infra).

## Quick start

```bash
# One-time: install uv if you don't have it
brew install uv

make install    # create .venv, install deps
make test       # run pytest
make run        # start the API at http://localhost:8080
```

First `make install` will produce a `uv.lock`; commit it so subsequent installs are reproducible.

## Commands

| Command             | Purpose                           |
|---------------------|-----------------------------------|
| `make install`      | Install deps (runtime + dev)      |
| `make test`         | Run pytest                        |
| `make lint`         | Ruff format + check (auto-fix)    |
| `make run`          | Start API on localhost:8080       |
| `make docker-build` | Build the Cloud Run container     |
| `make docker-run`   | Run the container locally         |

## Layout

```
src/fantasy_coach/   application code
tests/               pytest tests
Dockerfile           Cloud Run image
```

## Infrastructure

All GCP infrastructure (Cloud Run service, Firestore, Secret Manager, Firebase, Vertex AI, IAM) is provisioned via Terraform in [`lopeztech/platform-infra`](https://github.com/lopeztech/platform-infra) — not in this repo.

See [`docs/deploy.md`](docs/deploy.md) for the Cloud Run deploy command and the GitHub Actions workflow that ships every push to `main`.
