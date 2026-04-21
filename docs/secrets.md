# Secrets

All production secrets are stored in GCP Secret Manager. Cloud Run injects
them as environment variables at deploy time via `--set-secrets`. Local
development reads the same variable names from a `.env` file (gitignored).

## Required secrets

| Secret Manager name | Env var | Purpose | Who can rotate |
|---|---|---|---|
| `firebase-project-id` | `FIREBASE_PROJECT_ID` | Firebase project for ID-token verification (issue #17) | Project owner |

## Naming convention

Secrets follow the pattern `<resource>-<purpose>` in lowercase with hyphens,
e.g. `firebase-project-id`. All secrets live in the same GCP project as the
Cloud Run service (`fantasy-coach-lcd`).

Full resource path: `projects/<PROJECT_ID>/secrets/<name>`

## Creating a secret

```bash
# Create the secret (empty, no version yet)
gcloud secrets create firebase-project-id \
  --project=$PROJECT_ID

# Add the first version
echo -n "my-firebase-project" | \
  gcloud secrets versions add firebase-project-id \
    --data-file=- \
    --project=$PROJECT_ID
```

## Rotating a secret

```bash
# Add a new version (previous version stays active until pinned)
echo -n "new-value" | \
  gcloud secrets versions add firebase-project-id \
    --data-file=- \
    --project=$PROJECT_ID

# Destroy old version once Cloud Run has picked up the new one
gcloud secrets versions destroy <OLD_VERSION_NUMBER> \
  --secret=firebase-project-id \
  --project=$PROJECT_ID
```

Cloud Run uses `:latest` aliases in `--set-secrets`, so a new version is
picked up on the next revision deployment automatically.

## IAM

The runtime service account (`fantasy-coach-runtime`) holds
`roles/secretmanager.secretAccessor` scoped to each secret individually —
not project-wide. Bindings are managed in `lopeztech/platform-infra`:

```hcl
resource "google_secret_manager_secret_iam_member" "firebase_project_id" {
  secret_id = google_secret_manager_secret.firebase_project_id.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}
```

## Local development

Copy `.env.example` to `.env` and fill in values before running locally:

```bash
cp .env.example .env
# edit .env — never commit this file
```

The application reads env vars directly (FastAPI/uvicorn picks them up from
the process environment). For a local dev helper you can source the file:

```bash
set -a; source .env; set +a
make run
```

## Future secrets

If a paid bookmaker odds API is added (issue #26), its API key goes here
under `bookmaker-odds-api-key` / `BOOKMAKER_ODDS_API_KEY`.
