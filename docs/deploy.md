# Deployment

The fantasy-coach API runs on Cloud Run, scaled to zero so idle cost is $0.
Image lives in Artifact Registry. Region: **`australia-southeast1`** (Sydney)
to keep `nrl.com` scrape latency low and host data near the predicted matches.

## One-time setup (paired with `lopeztech/platform-infra`)

The GCP project, Cloud Run service, Artifact Registry repo, service account,
and Workload Identity Federation pool are provisioned via Terraform in
[`lopeztech/platform-infra`](https://github.com/lopeztech/platform-infra) under
`projects/fantasy-coach/`. Don't `gcloud iam create` anything by hand — that
state will get overwritten on the next `terraform apply`.

After the platform-infra PR lands and `terraform apply` runs, three GitHub
secrets must be added to **this** repo:

| Secret | Source | What it's for |
|--------|--------|----------------|
| `FANTASY_COACH_WIF_PROVIDER` | platform-infra `output "wif_provider"` | OIDC issuer for keyless `gcloud auth` |
| `FANTASY_COACH_DEPLOYER_SA` | platform-infra `output "deployer_sa_email"` | Service account the workflow impersonates |
| `FANTASY_COACH_PROJECT_ID` | platform-infra `output "project_id"` | GCP project the deploy targets |

## Automatic deploy (preferred)

`.github/workflows/deploy.yml` runs on every push to `main` that touches
`src/`, `Dockerfile`, `pyproject.toml`, `uv.lock`, or the workflow itself.
It builds the image, pushes to Artifact Registry, and rolls a new Cloud Run
revision over the previous one. No manual step required after the secrets
are in place.

To trigger a deploy without a code change:

```bash
gh workflow run deploy.yml --repo lopeztech/fantasy-coach
```

## Manual deploy (escape hatch)

If CI is wedged or you need to ship a one-off image, this is the equivalent
locally. It assumes you've authenticated with `gcloud auth login` and have
`roles/run.developer` + `roles/artifactregistry.writer` on the project.

```bash
PROJECT_ID=fantasy-coach-lcd
REGION=australia-southeast1
SERVICE=fantasy-coach-api
REPO=fantasy-coach
TAG=$(git rev-parse --short HEAD)
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/api:$TAG"

# 1. Build a linux/amd64 image (Cloud Run doesn't accept arm64).
docker buildx build --platform linux/amd64 -t "$IMAGE" --push .

# 2. Roll a new revision.
gcloud run deploy "$SERVICE" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --image "$IMAGE" \
    --platform managed \
    --allow-unauthenticated \
    --min-instances 0 \
    --max-instances 2 \
    --cpu 1 \
    --memory 512Mi \
    --concurrency 80 \
    --timeout 120 \
    --cpu-throttling \
    --port 8080 \
    --execution-environment gen2 \
    --service-account "fantasy-coach-runtime@$PROJECT_ID.iam.gserviceaccount.com" \
    --set-env-vars "FIREBASE_PROJECT_ID=$PROJECT_ID"
```

`--cpu-throttling` is the flag that achieves "CPU allocated only during
requests" — without it, the container is billed for CPU even when idle.

## Runtime sizing

Flags above and in `.github/workflows/deploy.yml` must stay in sync with
the Terraform resource (`google_cloud_run_v2_service.api` in
platform-infra). Cloud Run stores whatever was last deployed, so the
workflow is the effective source of truth — TF matches so fresh applies
don't drift.

| Flag | Value | Rationale |
|------|-------|-----------|
| `--memory` | `512Mi` | Observed p99 RSS < 100 MiB, so 256Mi would fit — but gen2 execution environment has a 512Mi minimum, enforced by `gcloud run deploy`. Kept at 512Mi rather than downgrade to gen1. |
| `--cpu` | `1` | One request at a time fits well under one vCPU; the logistic predict is µs-scale. |
| `--concurrency` | `80` | Cloud Run's default; pinned explicitly so TF/deploy can't silently drift. Revisit after we have real traffic; 200 may be viable for a mostly-I/O endpoint. |
| `--timeout` | `120` | Down from the 300s default. First request of a round does a live scrape of nrl.com (~1s/fixture × 8 fixtures + overhead), so we can't safely cut to the AC's 60s yet — drop once #65 lands and scrape is off-path. |
| `--cpu-throttling` | on | Billable CPU only during requests; cold idle is free. |
| `--min-instances` | `0` | Scale to zero when idle. |
| `--max-instances` | `2` | Keeps the blast radius bounded while traffic is low. |

These values were chosen against 13 hours of post-launch metrics, not a
30-day window. Revisit once we have real traffic (tracked as a follow-up
on #64).

## Auth model

The service is `--allow-unauthenticated` at the Cloud Run IAM layer
because browser clients can't mint Google-signed OAuth2 ID tokens. Real
auth lives in `FirebaseAuthMiddleware` (`src/fantasy_coach/auth.py`):

- `/healthz` is open (for Cloud Run's own liveness probes + uptime checks).
- Every other path must carry `Authorization: Bearer <firebase-id-token>`;
  the middleware rejects missing / expired / wrong-project tokens with 401.

`FIREBASE_PROJECT_ID` activates the middleware. Plain `--set-env-vars`
(not Secret Manager) is deliberate: the project ID isn't sensitive, and
Secret Manager wiring is deferred to #16 when the rest of the secrets
land together.

## Smoke test

Cloud Run blocks external GETs on `/healthz` because Google's Frontend
reserves that path for its own health routing. The container's own
liveness probes still hit it at 200 OK (visible in Cloud Run logs). For
an external smoke test, use `/predictions` with a Firebase ID token
instead — that reaches FastAPI and returns a real response. The
`gcloud run services describe` still surfaces the current URL:

```bash
URL=$(gcloud run services describe $SERVICE --region $REGION --format 'value(status.url)')
curl "$URL/predictions?season=2026&round=1"
# → {"detail":"Missing or malformed bearer token"}  (401 from FirebaseAuthMiddleware)
```

The 401 without an `Authorization` header *is* the green signal — it
proves the middleware is active. To actually fetch predictions, let the
SPA sign in and let the browser send its Firebase ID token.

## Service account least-privilege

The runtime SA (`fantasy-coach-runtime`) holds only:

- `roles/datastore.user` (Firestore read/write — for #15)
- `roles/secretmanager.secretAccessor` (config secrets — for #16)
- `roles/aiplatform.user` (Vertex Gemini — for #22)

The deployer SA (`fantasy-coach-deployer`) holds only the roles needed to
build, push, and roll a revision — `roles/run.developer`,
`roles/artifactregistry.writer`, `roles/iam.serviceAccountUser` (to
impersonate the runtime SA at deploy time).

Both SA definitions live in platform-infra and are the source of truth.

## Rolling back

Cloud Run keeps every revision. To roll back:

```bash
gcloud run services update-traffic $SERVICE \
    --project $PROJECT_ID --region $REGION \
    --to-revisions $SERVICE-00042-abc=100  # the revision name from the console
```
