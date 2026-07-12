"""Lambda entrypoint tests (#292 Phase 2).

Invokes the Mangum handler in-process with a synthesized API Gateway HTTP API
(v2) event — no Docker / Lambda runtime needed. Proves the event → ASGI → app
→ response path works end-to-end for the storage-free health endpoint. The
container image's live boot is verified at deploy time (Phase 2b).
"""

from __future__ import annotations

import json
from typing import Any


def _apigw_v2_event(method: str, path: str) -> dict[str, Any]:
    """Minimal API Gateway HTTP API (payload format 2.0) event."""
    return {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"host": "example.execute-api.ap-southeast-2.amazonaws.com"},
        "requestContext": {
            "http": {
                "method": method,
                "path": path,
                "sourceIp": "127.0.0.1",
            },
            "stage": "$default",
        },
        "isBase64Encoded": False,
    }


class _Context:
    """Stand-in for the Lambda context object (Mangum reads a few attrs)."""

    function_name = "fantasy-coach-api"
    memory_limit_in_mb = 512
    invoked_function_arn = "arn:aws:lambda:ap-southeast-2:000000000000:function:fantasy-coach-api"
    aws_request_id = "test-request-id"


def test_handler_serves_healthz() -> None:
    from fantasy_coach.lambda_app import handler

    resp = handler(_apigw_v2_event("GET", "/healthz"), _Context())

    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["status"] == "ok"
    assert "version" in body


def test_handler_unknown_route_404() -> None:
    from fantasy_coach.lambda_app import handler

    resp = handler(_apigw_v2_event("GET", "/does-not-exist"), _Context())

    assert resp["statusCode"] == 404
