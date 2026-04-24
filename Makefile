.PHONY: install test lint ci run docker-build docker-run check-uv

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
	  echo "uv is not installed. Install with: brew install uv"; \
	  exit 1; \
	}

install: check-uv
	uv sync --all-groups
	uv run pre-commit install

test: check-uv
	uv run pytest

lint: check-uv
	uv run ruff format .
	uv run ruff check --fix .

# Mirror of what CI runs — check-only (no auto-fix), plus tests.
# Run before pushing to catch formatting / lint / metric-drift issues
# locally rather than discovering them on the Ubuntu CI runner.
ci: check-uv
	uv run ruff format --check
	uv run ruff check
	uv run pytest

run: check-uv
	uv run uvicorn fantasy_coach.app:app --reload --host 0.0.0.0 --port 8080

docker-build:
	docker build -t fantasy-coach:latest .

docker-run:
	docker run --rm -p 8080:8080 -e PORT=8080 fantasy-coach:latest
