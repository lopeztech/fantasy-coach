# Cost

Running costs for the fantasy-coach stack, how we keep them low, and what
to do when a number moves in the wrong direction.

> This doc is thin today — most of the infrastructure scales to zero and
> GCP credits cover the rest. Budget alerts, per-SKU dashboards, and a
> real baseline come with #63 (`projects/fantasy-coach/billing.tf` and
> Billing Export → BigQuery). Anything not filled in here is a known gap,
> not forgotten.

## Cloud Run right-sizing (April 2026)

The prediction API was deployed with gcloud defaults for its first 13
hours, then tightened once we had enough observability to make informed
cuts. Nothing was cut below observed p99 + meaningful headroom.

| Flag | Before (gcloud defaults / original deploy) | After (pinned in `deploy.yml` + TF) | Why |
|------|---------------------------------------------|--------------------------------------|-----|
| `--memory` | `512Mi` | `256Mi` | p99 container RSS < 20% of 512Mi across 20 revisions. 256Mi is still > 2.5× the working set. |
| `--concurrency` | Cloud Run default 80 (not pinned) | `80` pinned | Explicit so TF and deploy stay aligned; no behavioural change. |
| `--timeout` | Cloud Run default `300s` (not pinned) | `120s` pinned | Cold-round scrape does ~8 HTTPS round-trips to nrl.com, so 60s is unsafe until #65 moves scrape off-path. 120s is still a 60% reduction from the default. |
| `--cpu` | `1` | `1` unchanged | Predict is µs-scale; scraping is I/O-bound. |
| `--min-instances` | `0` | `0` unchanged | Scale-to-zero stays. |
| `--max-instances` | `2` | `2` unchanged | Blast-radius cap. |

**Expected impact:** Cloud Run bills per GiB-second while an instance is
running. Halving memory halves that component's cost for every billable
second. With scale-to-zero and low traffic the absolute saving is small
today, but the ratio persists as traffic grows.

The measurements above came from a ~13-hour window dominated by CI
churn, not real user traffic. Revisit concurrency and memory once we
have 30 days of real requests — concurrency especially may have room to
go from 80 → 200 on a mostly-I/O endpoint. Tracked as a follow-up on #64.
