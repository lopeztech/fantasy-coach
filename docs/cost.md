# Cost

Running costs for the fantasy-coach stack, how we keep them low, and what
to do when a number moves in the wrong direction.

> Most of the infrastructure scales to zero and GCP credits cover the
> rest today, so real-dollar costs are near-zero. This doc is the
> early-warning plumbing for when credits run out or usage moves.

## Baseline (April 2026)

The stack is currently billed in three components. All numbers are
projections — real dollars will come from the BigQuery billing export
once it's manually enabled (see below).

| Component | Driver | Expected monthly cost at current traffic |
|-----------|--------|-------------------------------------------|
| Cloud Run (API) | 0 → low; scale-to-zero; 256Mi × concurrency 80 | < $1 |
| Firestore | Free tier (1 GiB storage, 50K reads/day) | $0 |
| Artifact Registry | One image tag per deploy, ~1 GiB total | < $0.10 |
| Cloud Logging | Default retention; low request volume | $0 (free tier) |
| Vertex AI (Gemini) | Commentary previews; ~2K tokens in + 400 out per match | ~$0.05/month at current volume |
| Egress | SPA bundle from Firebase Hosting; low traffic | $0 (free tier) |

Refresh these numbers by querying the BigQuery billing export dataset
(`billing_export.gcp_billing_export_v1_<billing-account-id>`) once
enabled — see the manual bootstrap step below.

## Budget alerts

A single project-wide budget is defined in
[`lopeztech/platform-infra:projects/fantasy-coach/billing.tf`](https://github.com/lopeztech/platform-infra/blob/master/projects/fantasy-coach/billing.tf)
via the shared `modules/budget`. Configuration lives in
`projects/fantasy-coach/environments/prod/terraform.tfvars`:

- **Monthly cap:** `$20` USD (generous alarm threshold given current
  burn is < $1/mo — tighten later if real spend rises).
- **Thresholds:** 50 %, 80 %, 100 % of current spend, plus
  forecasted-100 %.
- **Recipient:** `admin@lopezcloud.dev` (`notification_email`
  variable). Routed via a `google_monitoring_notification_channel`
  email channel, not the default IAM recipients.
- **Rotation:** bump `monthly_budget_usd` in tfvars + re-apply. The
  email address comes from the same tfvars, single source of truth.

Slack webhook routing is an explicit follow-up — add a second
notification channel to the `budget` module when we need it.

## Per-SKU dashboard (manual bootstrap)

Terraform manages the budget, but the per-SKU breakdown dashboard
needs Billing Export → BigQuery turned on *at the billing-account
level*. Enabling it requires `roles/billing.admin` on the billing
account, which is outside `projects/fantasy-coach/`'s Terraform scope.
One-time manual steps:

1. **Enable the export** — Billing console → Billing export → BigQuery
   export → pick project `fantasy-coach-lcd` and dataset name
   `billing_export` (create on first run). Choose *Standard usage
   cost* at minimum; *Detailed usage cost* if we need per-SKU
   pricing later.
2. **Label propagation** — Cloud Run already carries
   `app=fantasy-coach`, `environment=prod`, `managed_by=terraform`
   (see `cloudrun.tf`). Both the billing-account-level export and
   BigQuery preserve these, so a Looker Studio panel can segment on
   `labels.app` without us adding new tags.
3. **Dashboard** — clone a Looker Studio billing template, point it at
   the new dataset, save the public URL here once built.

Until step 3 lands, per-SKU visibility lives in the Billing console's
Reports view (filter by label `app=fantasy-coach`).

## Cloud Run right-sizing (April 2026)

The prediction API was deployed with gcloud defaults for its first 13
hours, then tightened once we had enough observability to make informed
cuts. Nothing was cut below observed p99 + meaningful headroom.

| Flag | Before (gcloud defaults / original deploy) | After (pinned in `deploy.yml` + TF) | Why |
|------|---------------------------------------------|--------------------------------------|-----|
| `--memory` | `512Mi` | `512Mi` unchanged | p99 container RSS < 20% of 512Mi across 20 revisions — 256Mi would fit, but **gen2 execution environment has a 512Mi minimum enforced by `gcloud run deploy`**. We'd need to drop to gen1 to cut further, and the scrape-heavy workload values gen2's networking + startup CPU boost more than the pennies/month of memory savings. |
| `--concurrency` | Cloud Run default 80 (not pinned) | `80` pinned | Explicit so TF and deploy stay aligned; no behavioural change. |
| `--timeout` | Cloud Run default `300s` (not pinned) | `120s` pinned | Cold-round scrape does ~8 HTTPS round-trips to nrl.com, so 60s is unsafe until #65 moves scrape off-path. 120s is still a 60% reduction from the default. |
| `--cpu` | `1` | `1` unchanged | Predict is µs-scale; scraping is I/O-bound. |
| `--min-instances` | `0` | `0` unchanged | Scale-to-zero stays. |
| `--max-instances` | `2` | `2` unchanged | Blast-radius cap. |

**Expected impact:** Memory is unchanged; the concurrency + timeout
pins prevent the flags from silently drifting apart between the deploy
workflow and Terraform, and the 300s → 120s timeout cut reduces the
maximum billable CPU time per stuck request by 60 %. Real dollars
today are pennies — this is drift insurance, not a major cost cut.

The measurements above came from a ~13-hour window dominated by CI
churn, not real user traffic. Revisit concurrency and memory once we
have 30 days of real requests — concurrency especially may have room to
go from 80 → 200 on a mostly-I/O endpoint. Tracked as a follow-up on #64.

## Precompute Job vs on-request scrape (#65)

The `/predictions` endpoint no longer scrapes on demand. A Cloud Run
Job (`fantasy-coach-precompute`) runs the scrape + feature + predict
pipeline twice a week — Tue 09:00 and Thu 06:00 AEST — and writes the
round's predictions to Firestore. The API reads from the cache and
returns 503 on miss.

| Path | Before (on-request) | After (scheduled Job) |
|------|---------------------|-----------------------|
| Cloud Run billable time per round | ~30s per first-of-round request on the API instance (while it scrapes ~8 fixtures sequentially) | ~30s on the Job (one-shot container, CPU not throttled while running) — times 2 runs per round |
| Tail latency to user | p99 dominated by the scrape on cache miss; first user of every round pays 30s+ | p99 is a Firestore doc-get (<100ms) |
| Max-billable-CPU per stuck request | 120s timeout (deploy flag) | N/A — API path never scrapes |
| Blast radius on scrape failure | Request returns 500 | Job fails, API still serves last-known predictions; Cloud Scheduler retries (2 retries, 30–300s backoff) |

Net effect: real-dollar cost is a wash (Job instance time ≈ former
on-request time, just moved), but p99 latency for users is an order of
magnitude better and we gain a retry safety net. The scheduled Job also
picks up team-list changes between Tue and Thu — the `--force` default
re-scrapes even when a round already has cached predictions.

## Container image (April 2026)

Two-stage `python:3.12-slim` build. Key optimisations:

| Technique | Effect |
|-----------|--------|
| Multi-stage build | Dev tools (uv, pip) stripped from the runtime image |
| `uv.lock` copied before `src/` | Dependency layer cached across source-only changes |
| `uv sync --no-install-project` first pass | Heavy deps installed separately; re-used when only app code changes |
| `--mount=type=cache,target=/root/.cache/uv` | Wheel cache persisted across local builds; CI uses a fresh layer |
| `UV_COMPILE_BYTECODE=1` | `.pyc` files pre-compiled at build time, not on first import |

**Estimated image size (uncompressed, linux/amd64):** ~340–380 MB. The two
largest contributors are `xgboost` (~80 MB, includes OpenMP shared libs) and
the `grpc` native extension pulled by `google-cloud-firestore` (~40 MB). The
`python:3.12-slim` base accounts for ~45 MB.

**Why `python:3.12-slim` and not distroless?** `gcr.io/distroless/python3-debian12`
was evaluated as the final-stage base. It is incompatible with two runtime
dependencies: `google-cloud-firestore` (pulls `grpc` which requires
`libssl`/`libcrypto`/`libstdc++`) and `xgboost` (links `libgomp`). Neither
library is present in the distroless image. Switching would require a custom
base, which outweighs the ~10 MB saving. Revisit if `xgboost` moves to a
training-only dependency and `grpc` is replaced by the HTTP/1.1 transport.

**Artifact Registry cleanup policy** is Terraform-managed in the
`lopeztech/platform-infra` repository (`projects/fantasy-coach/`). The desired
policy — keep the last 10 tagged images plus any image tagged `latest` or
referenced by a live Cloud Run revision; delete untagged and older images — is
tracked there. At the current deploy cadence (~2/week), storage is < $0.10/month
even without a cleanup policy, but it should be added before traffic scales.

## CI costs (#120)

GitHub Actions minutes are free for public repos but every workflow run costs
real developer wall-clock time. The four workflows are optimised as follows:

### Path filters

All four workflows have `paths:` filters on both `push` and `pull_request`
events so unrelated changes don't trigger unrelated CI:

| Workflow | Triggers on |
|----------|-------------|
| `ci.yml` | `src/`, `tests/`, `pyproject.toml`, `uv.lock`, `ci.yml` |
| `web-ci.yml` | `web/`, `firebase.json`, `.firebaserc`, `web-ci.yml` |
| `deploy.yml` | `src/`, `Dockerfile`, `pyproject.toml`, `uv.lock`, `deploy.yml` |
| `web-deploy.yml` | `web/`, `firebase.json`, `.firebaserc`, `web-deploy.yml` |

A pure-frontend change (e.g. `web/src/styles.css`) skips `ci.yml` and
`deploy.yml` entirely; a Python-only change skips `web-ci.yml` and
`web-deploy.yml`.

### uv dependency cache

`ci.yml` uses `astral-sh/setup-uv@v5` with `enable-cache: true`. The action
stores the resolved wheel cache (keyed on `uv.lock`) in the GHA cache;
subsequent runs that don't change `uv.lock` restore the entire virtual-env in
seconds instead of re-downloading wheels.

**Breaking the cache intentionally:** bump any dependency in `pyproject.toml`
and commit. The new `uv.lock` SHA invalidates the old cache entry and forces a
fresh wheel download on the next run.

### npm dependency cache

Both `web-ci.yml` and `web-deploy.yml` use `actions/setup-node@v4` with
`cache: 'npm'` and `cache-dependency-path: web/package-lock.json`. The node
modules are restored from the GHA cache unless `package-lock.json` changes.

### Docker buildx layer cache

`deploy.yml` passes `--cache-from type=gha` and `--cache-to type=gha,mode=max`
to `docker buildx build`. On a warm run, unchanged layers (especially the
`uv sync` layer which pulls `xgboost`, `grpc`, etc.) are restored from the GHA
cache rather than re-downloaded and rebuilt. A source-only change typically
skips all dependency layers and rebuilds only the final `COPY src/ .` layer.

The GHA cache backend uses the repo's 10 GB free-tier cache. The buildx cache
entry for this image is ~300–400 MB compressed; it's invalidated any time
`uv.lock` or `Dockerfile` changes.

**Note:** `type=gha` is preferred over `type=registry` here because it carries
no extra AR storage cost and the 10 GB free tier is ample for a single-image
project. Switch to `type=registry` if the project ever moves off GitHub Actions.

## Vertex AI Gemini (commentary previews)

Gemini Flash on Vertex AI generates per-match preview text in
`src/fantasy_coach/commentary/`. The cost model:

| Metric | Value |
|--------|-------|
| Input cost | $0.00025 / 1K tokens |
| Output cost | $0.0005 / 1K tokens |
| Tokens per preview | ~2K in + ~400 out |
| Cost per preview | ~$0.0007 |
| Matches per year | ~8 × 2 runs × 27 rounds = ~432 (with caching) |
| Worst-case annual cost | ~$0.30/year (0% cache hit rate) |
| Expected annual cost at >90% hit rate | ~$0.03/year |

**Cache hit rate target:** > 90%. The `ResponseCache` in
`src/fantasy_coach/commentary/cache.py` keys on
`CACHE_KEY_VERSION + model_version + combined_prompt`, so
any prompt-template or feature-schema change busts the cache
via a `CACHE_KEY_VERSION` bump. Each cache entry stores
`feature_snapshot_hash` (SHA-8 of the MatchContext inputs)
for auditing which input state generated a cached response.

**Invalidation triggers:**
1. Bump `CACHE_KEY_VERSION` in `cache.py` when the prompt template changes.
2. Run `python -m fantasy_coach clear-commentary-cache --version-mismatch`
   for eager eviction after a bad-template rollback.
3. Run `python -m fantasy_coach clear-commentary-cache --before-days 7`
   to evict stale past-match entries.

**Monitoring:** At the end of every precompute run a one-line summary
is emitted: `commentary summary: N requests, M cache hits (X% hit rate),
Y tokens in, Z tokens out, est. cost $0.NN`. Set a Cloud Monitoring
log-metric alert if hit rate drops below 70% in a single run.

**Note:** Vertex AI bills per successful call only; retried-and-failed
calls are not billed. The token budget circuit-breaker in `TokenBudget`
(100K tokens/day, 512 tokens/request) prevents runaway spend if the
cache layer is bypassed.

## Cloud Logging

Cloud Logging's pricing: first 50 GiB/month of ingestion is free;
above that costs **$0.50/GiB ingested** + $0.01/GiB/month retained past
30 days.

Current estimated volume: **~2 GiB/month** (well within the free tier).
The concern is unbounded growth as traffic scales.

### Exclusion filters (managed via Terraform in platform-infra)

Three exclusion filters are applied in `logging.tf`:

| Filter | What it drops | Why |
|--------|--------------|-----|
| `/healthz` requests (status < 400) | Cloud Run liveness probe hits | 30 req/min × 8KB each; ~350 MB/month of noise with zero diagnostic value |
| `severity < INFO` app logs | DEBUG-level uvicorn / app output | Only useful locally; emitted when `LOG_LEVEL=DEBUG` is accidentally set |
| Precompute `"feature: "` lines | Per-match feature-computation progress | ~30 MB/run × 2 runs/week; replay-able locally |

Exclusions operate at the log router before ingestion so excluded bytes
don't count against the 50 GiB free tier and don't retain.

**Audit-critical logs that are never excluded:**
- Any log with `severity >= WARNING`
- `FirebaseAuthMiddleware` authentication events
- Precompute success/failure summary lines

### Retention policy

Log retention is pinned to **30 days** in Terraform. Logs older than 30
days are automatically deleted. This prevents silent drift from the
platform default.

### Expected savings

At current volume the exclusions reduce ingestion by an estimated
40–60% (health check + debug noise dominate the raw stream). At 2×
traffic the project would still be within the 50 GiB free tier with
exclusions in place.

## Firestore retention (TTL policies)

Firestore's free tier covers 1 GiB storage/day. TTL policies auto-delete
ephemeral documents at no cost (TTL deletion is not billed as write operations).

| Collection | Retention | Rationale |
|------------|-----------|-----------|
| `team_list_snapshots` | 80 days (~10 rounds) | Only the most-recent snapshot per team per round is used at prediction time |
| `model_drift_reports` | 18 months | Long enough for season-over-season comparisons |
| `matches` | Permanent | Audit data + feature-engineering history |
| `predictions` | Permanent | Used by the Accuracy page indefinitely |

TTL is implemented via a `ttl_timestamp` field on each eligible document.
The Firestore TTL policy (configured in platform-infra) reads this field and
schedules deletion. Because TTL is asynchronous and best-effort, downstream
code must never depend on a document being gone by a specific time.

To retrofit existing documents: `python -m fantasy_coach backfill-ttl --dry-run`
(preview) then `python -m fantasy_coach backfill-ttl` (apply).
