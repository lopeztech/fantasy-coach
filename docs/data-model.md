# Data Model

## Firestore

Production storage uses Cloud Firestore (Native mode). All documents live in a
single database — the default `(default)` database unless overridden by the
`FIRESTORE_DATABASE` env var.

### Collection: `matches`

One document per NRL match. Document ID is `str(match_id)`.

```
matches/{match_id}
├── match_id        int          NRL internal ID
├── season          int          e.g. 2024
├── round           int          1-indexed round number
├── start_time      string       ISO 8601 UTC, e.g. "2024-03-03T02:30:00+00:00"
├── match_state     string       "Upcoming" | "Live" | "FullTime"
├── venue           string|null
├── venue_city      string|null
├── weather         string|null
├── home            map
│   ├── team_id     int
│   ├── name        string       e.g. "Manly Warringah Sea Eagles"
│   ├── nick_name   string       e.g. "Sea Eagles"
│   ├── score       int|null     null until FullTime
│   └── players     list[map]
│       ├── player_id      int
│       ├── jersey_number  int|null
│       ├── position       string|null
│       ├── first_name     string|null
│       └── last_name      string|null
├── away            map          same shape as home
└── team_stats      list[map]
    ├── title       string       e.g. "Possession"
    ├── type        string       e.g. "percentage"
    ├── units       string|null  e.g. "%"
    ├── home_value  float|null
    └── away_value  float|null
```

### Indexes

Firestore requires a composite index for queries that combine a filter with an
`order_by` on a different field. The `list_matches` query uses:

```
Collection: matches
Fields: season ASC, start_time ASC, match_id ASC
```

In the emulator this index is created automatically. For production, add the
index via Terraform (`google_firestore_index` resource) or the GCP console
before running the first query.

### Upsert behaviour

`upsert_match` calls `document.set(data)` which is an atomic full-document
replace. This ensures child arrays (players, stats) are always consistent with
the parent document — the same semantic as the SQLite `DELETE + INSERT` pattern.

---

## SQLite (local development)

Local development uses `SQLiteRepository` backed by a file at
`data/nrl.db` (configurable via `FANTASY_COACH_DB_PATH`).

Schema is defined in `src/fantasy_coach/storage/schema.sql`. The table layout
uses normalised child tables for players and stats instead of JSON arrays,
keeping queries simple and the schema version-tracked.

### Tables

| Table | Purpose |
|---|---|
| `schema_version` | Single-row version guard — raises on mismatch |
| `matches` | One row per match (IDs, times, scores, venue, weather) |
| `match_players` | Player rosters (FK to `matches`, cascade delete) |
| `match_team_stats` | Per-team stats (FK to `matches`, cascade delete) |

---

## Selecting the backend

Set the `STORAGE_BACKEND` environment variable:

```bash
STORAGE_BACKEND=sqlite    # default — uses data/nrl.db
STORAGE_BACKEND=firestore # Cloud Run / production
```

The factory `fantasy_coach.config.get_repository()` reads this variable and
returns the appropriate `Repository` implementation.

---

## Firestore TTL Policies

Firestore native TTL policies are configured in `lopeztech/platform-infra`
→ `projects/fantasy-coach/firestore.tf`. Each eligible collection has a
`ttl_timestamp` field (type: `Timestamp`) written by the application at doc
creation time. Firestore auto-deletes expired docs asynchronously (no cost,
best-effort, not guaranteed to be exact-to-the-second).

| Collection | Retention | `ttl_timestamp` base | Writer |
|------------|-----------|----------------------|--------|
| `team_list_snapshots` | 80 days | `scraped_at` | `FirestoreTeamListRepository.record_snapshot()` |
| `model_drift_reports` | 18 months (~548 days) | write time (`now()`) | `retrain.default_drift_writer()` |
| `matches` | **Never expires** — permanent audit data | n/a | — |
| `predictions` | **Never expires** — used by Accuracy page indefinitely | n/a | — |

### Backfilling existing documents

Documents written before #153 (the PR that added `ttl_timestamp`) have no TTL
field and will not be auto-deleted by Firestore. Run the one-shot backfill
command to retrofit them:

```bash
STORAGE_BACKEND=firestore python -m fantasy_coach backfill-ttl [--dry-run] [--collection team_list_snapshots|model_drift_reports|all]
```

This command:
- Iterates all documents in the eligible collections.
- Skips documents that already carry a `ttl_timestamp` (idempotent).
- Computes `ttl_timestamp` from the doc's `scraped_at` / `created_at` field,
  falling back to `now()` when the base field is absent.
- `--dry-run` reports counts without writing.

### TTL index requirements

Firestore TTL requires a single-field index on the TTL field. This is
created automatically when you enable TTL in the Firestore console or via
Terraform. Ensure no composite index conflicts with the TTL field in
existing indexes.
