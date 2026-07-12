"""AWS Lambda entrypoint — wraps the FastAPI app in Mangum (#292 Phase 2).

API Gateway (HTTP API v2) proxies every request to this handler; Mangum
translates the Lambda event into an ASGI scope, runs it through the same
``fantasy_coach.app:app`` that uvicorn serves on Cloud Run, and marshals the
response back. Nothing about the routes/middleware changes — this is purely
the AWS invocation shim.
"""

from __future__ import annotations

from mangum import Mangum

from fantasy_coach.app import app

# lifespan="off": the app's only lifespan work is a best-effort Firestore
# cache-warm (_prefetch_current_rounds), which isn't wanted per-cold-start on
# Lambda (it would add cross-cloud latency and needs GCP creds). The regular
# per-request path is unaffected, so skipping it is safe.
handler = Mangum(app, lifespan="off")
