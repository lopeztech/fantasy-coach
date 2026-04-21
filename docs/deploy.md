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
    --no-allow-unauthenticated \
    --min-instances 0 \
    --max-instances 2 \
    --cpu 1 \
    --memory 512Mi \
    --cpu-throttling \
    --port 8080 \
    --execution-environment gen2 \
    --service-account "fantasy-coach-runtime@$PROJECT_ID.iam.gserviceaccount.com"
```

`--cpu-throttling` is the flag that achieves "CPU allocated only during
requests" — without it, the container is billed for CPU even when idle.

## Smoke test

The Cloud Run URL is in the deploy output (`gcloud run services describe
$SERVICE --region $REGION --format 'value(status.url)'`). With auth still
required (it is — see #17), use an identity token:

```bash
URL=$(gcloud run services describe $SERVICE --region $REGION --format 'value(status.url)')
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" "$URL/healthz"
# → {"status":"ok","version":"0.1.0"}
```

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
