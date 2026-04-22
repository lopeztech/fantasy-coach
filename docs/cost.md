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
| Vertex AI (Gemini) | Not yet wired (tracked by #22) | $0 |
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
