"""Tests for FirebaseAuthMiddleware.

All Firebase Admin SDK calls are patched so no real GCP credentials are
needed in CI. Tests cover the five required cases from issue #17:
valid token, expired token, missing token, wrong project audience,
and the health endpoint being open without auth.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import fantasy_coach.auth as auth_module
from fantasy_coach.auth import FirebaseAuthMiddleware

# ---------------------------------------------------------------------------
# Minimal FastAPI app with the middleware wired in
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.add_middleware(FirebaseAuthMiddleware)


@_app.get("/healthz")
def healthz():
    return {"status": "ok"}


@_app.get("/protected")
def protected():
    return {"status": "protected"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_firebase_singleton():
    """Reset the cached Firebase app between tests to avoid cross-test leakage."""
    auth_module._app = None
    yield
    auth_module._app = None


@pytest.fixture
def client():
    return TestClient(_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _firebase_patched(*, verify_result=None, verify_side_effect=None):
    """Context manager: mocks initialize_app + verify_id_token together."""

    class _Ctx:
        def __enter__(self):
            self._init_patcher = patch(
                "fantasy_coach.auth.firebase_admin.initialize_app",
                return_value=MagicMock(),
            )
            self._verify_patcher = patch(
                "fantasy_coach.auth.firebase_auth.verify_id_token",
                return_value=verify_result,
                side_effect=verify_side_effect,
            )
            self._init_patcher.start()
            self._verify_patcher.start()
            return self

        def __exit__(self, *_):
            self._verify_patcher.stop()
            self._init_patcher.stop()

    return _Ctx()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_healthz_requires_no_auth(client):
    """Health endpoint must be open — Cloud Run health checks have no token."""
    response = client.get("/healthz")
    assert response.status_code == 200


def test_missing_token_returns_401(client):
    with _firebase_patched(verify_result={"uid": "u1"}):
        response = client.get("/protected")
    assert response.status_code == 401
    assert "Missing" in response.json()["detail"]


def test_malformed_token_returns_401(client):
    """'Token abc' (not 'Bearer abc') is malformed and must be rejected."""
    with _firebase_patched(verify_result={"uid": "u1"}):
        response = client.get("/protected", headers={"Authorization": "Token abc"})
    assert response.status_code == 401


def test_valid_token_grants_access(client):
    with _firebase_patched(verify_result={"uid": "user-123"}):
        response = client.get("/protected", headers={"Authorization": "Bearer valid-token"})
    assert response.status_code == 200


def test_expired_token_returns_401(client):
    import firebase_admin.auth as fba

    exc = fba.ExpiredIdTokenError("Token expired", cause=None)
    with _firebase_patched(verify_side_effect=exc):
        response = client.get("/protected", headers={"Authorization": "Bearer expired-token"})
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_invalid_token_returns_401(client):
    import firebase_admin.auth as fba

    exc = fba.InvalidIdTokenError("Invalid token", cause=None)
    with _firebase_patched(verify_side_effect=exc):
        response = client.get("/protected", headers={"Authorization": "Bearer bad-token"})
    assert response.status_code == 401
    assert "Invalid" in response.json()["detail"]


def test_wrong_audience_returns_401(client):
    """Wrong Firebase project audience is reported as an InvalidIdTokenError."""
    import firebase_admin.auth as fba

    exc = fba.InvalidIdTokenError("Firebase ID token has incorrect 'aud' claim", cause=None)
    with _firebase_patched(verify_side_effect=exc):
        response = client.get("/protected", headers={"Authorization": "Bearer wrong-aud-token"})
    assert response.status_code == 401


def test_unexpected_verification_error_returns_401(client):
    """Any unexpected exception during verification also yields 401 (not 500)."""
    with _firebase_patched(verify_side_effect=RuntimeError("unexpected")):
        response = client.get("/protected", headers={"Authorization": "Bearer any-token"})
    assert response.status_code == 401
