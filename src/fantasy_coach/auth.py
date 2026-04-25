"""Firebase ID-token verification middleware for FastAPI.

Every non-health request must carry a valid Firebase ID token in the
``Authorization: Bearer <token>`` header. Verification uses the Firebase
Admin SDK (not hand-rolled JWT parsing) against the configured project.

The middleware is enabled when ``FIREBASE_PROJECT_ID`` (or the fallback
``GOOGLE_CLOUD_PROJECT``) is set. When neither is set the middleware is
installed but will reject all protected requests — this surfaces
misconfiguration early rather than silently passing unauthenticated traffic.

Rate-limiting decision (deferred): Cloud Run's ``--no-allow-unauthenticated``
flag combined with Firebase token cost (one round-trip to Google's JWKS
endpoint + local verification) is sufficient bot deterrent at this scale.
Per-user quotas and rate-limits will be revisited if the service is opened
more broadly.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import firebase_admin
import firebase_admin.auth as firebase_auth
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that bypass auth — kept as a frozenset for O(1) exact lookup.
_OPEN_PATHS: frozenset[str] = frozenset({"/healthz"})
# Path prefixes that bypass auth (social-embed endpoints — crawlers have no tokens).
_OPEN_PREFIXES: tuple[str, ...] = ("/og/", "/share/")

# Module-level singleton so we initialise Firebase once per process.
_app: firebase_admin.App | None = None


def _get_app() -> firebase_admin.App:
    global _app
    if _app is None:
        project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        options: dict[str, str] = {}
        if project_id:
            options["projectId"] = project_id
        try:
            _app = firebase_admin.get_app()
        except ValueError:
            _app = firebase_admin.initialize_app(options=options)
    return _app


class FirebaseAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid Firebase ID token.

    Open paths (``/healthz``) bypass verification so Cloud Run's health
    checks and uptime monitors never require credentials.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _OPEN_PATHS or any(
            request.url.path.startswith(p) for p in _OPEN_PREFIXES
        ):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse({"detail": "Missing or malformed bearer token"}, status_code=401)

        token = auth_header[len("Bearer ") :]
        try:
            decoded = firebase_auth.verify_id_token(token, app=_get_app(), check_revoked=False)
            request.state.uid = decoded["uid"]
        except firebase_auth.ExpiredIdTokenError:
            return JSONResponse({"detail": "Token expired"}, status_code=401)
        except firebase_auth.InvalidIdTokenError:
            return JSONResponse({"detail": "Invalid token"}, status_code=401)
        except Exception:
            logger.exception("Token verification failed for request to %s", request.url.path)
            return JSONResponse({"detail": "Token verification failed"}, status_code=401)

        return await call_next(request)
