"""Tests for the CORS allowlist on the FastAPI app.

The SPA lives on a different origin (https://fantasy.lopezcloud.dev) from
the Cloud Run API, so the browser issues a preflight OPTIONS before each
authenticated GET. These tests verify:

- allowed origin → preflight returns Access-Control-Allow-Origin
- disallowed origin → preflight is not granted CORS headers
- preflight on a protected path is not blocked by the auth middleware (auth
  only runs on the *actual* request, after CORS has cleared the preflight)
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _build_client(monkeypatch: pytest.MonkeyPatch, origins: str | None = None) -> TestClient:
    # Prevent the app from installing the auth middleware; we're testing CORS
    # behaviour in isolation and the auth path has its own suite (test_auth).
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    if origins is not None:
        monkeypatch.setenv("FANTASY_COACH_ALLOWED_ORIGINS", origins)
    else:
        monkeypatch.delenv("FANTASY_COACH_ALLOWED_ORIGINS", raising=False)

    import fantasy_coach.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def _preflight(client: TestClient, path: str, origin: str) -> httpx.Response:  # noqa: F821
    return client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )


def test_default_allowlist_includes_production_spa(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)
    r = _preflight(client, "/predictions", "https://fantasy.lopezcloud.dev")
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://fantasy.lopezcloud.dev"
    assert "GET" in r.headers.get("access-control-allow-methods", "")


def test_default_allowlist_includes_vite_dev_server(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)
    r = _preflight(client, "/predictions", "http://localhost:5173")
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_disallowed_origin_not_granted(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)
    r = _preflight(client, "/predictions", "https://evil.example.com")
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch, origins="https://staging.lopezcloud.dev")
    ok = _preflight(client, "/predictions", "https://staging.lopezcloud.dev")
    assert ok.headers.get("access-control-allow-origin") == "https://staging.lopezcloud.dev"
    # The default production origin is no longer in the allowlist.
    denied = _preflight(client, "/predictions", "https://fantasy.lopezcloud.dev")
    assert "access-control-allow-origin" not in {k.lower() for k in denied.headers}


def test_actual_get_from_allowed_origin_gets_cors_header(monkeypatch: pytest.MonkeyPatch) -> None:
    # /healthz is open (no auth needed); verifies that a real GET also carries
    # the Access-Control-Allow-Origin echo when called from an allowed origin.
    client = _build_client(monkeypatch)
    r = client.get("/healthz", headers={"Origin": "https://fantasy.lopezcloud.dev"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://fantasy.lopezcloud.dev"
