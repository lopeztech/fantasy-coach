# SPA (web/)

Lightweight frontend for triggering predictions and browsing results. Lives
under `web/` and deploys to Firebase Hosting at
[`fantasy.lopezcloud.dev`](https://fantasy.lopezcloud.dev). Everything else
(the Cloud Run API, the model, the scraper) is intentionally the main
attraction — this frontend is deliberately thin.

## Framework

**Vite + React + TypeScript.** Chosen over SvelteKit because:

- The prediction API contract is typed (Pydantic response models), and TypeScript
  lets us mirror that shape without a second source of truth.
- React Router 6's loader + typed params cover everything the UI needs;
  nothing here warrants a meta-framework.
- Vite build output is small (<100 kB gzipped for this app) and Firebase
  Hosting's default gzip/brotli + long-lived asset caching handle delivery.

Bundler: Vite 5. Router: react-router-dom 6. No CSS framework — plain CSS in
`src/styles.css` is enough for a few pages.

## Local dev

```bash
cd web
npm install          # one-time
npm run dev          # Vite dev server on http://localhost:5173
npm run build        # type-check + production build → web/dist
npm run preview      # serve the production build locally
```

Pointing the dev server at a local backend is a follow-up (needed once
authenticated fetches arrive in #20).

## Routes

| Path                          | Purpose                                         |
|-------------------------------|-------------------------------------------------|
| `/`                           | Landing — links into the current round          |
| `/round/:season/:round`       | Match-card view for a given season/round (#21)  |

## Deploy

Firebase Hosting is configured via `firebase.json` + `.firebaserc` at the
repo root (not `web/`) so `firebase deploy` can be run from anywhere without
a `--config` flag.

```bash
# One-time: install the CLI and sign in
npm install -g firebase-tools
firebase login

# Build + deploy
cd web && npm run build && cd ..
firebase deploy --only hosting
```

The `default` project alias is `fantasy-coach-lcd` (the same GCP project
the Cloud Run API runs in — see `docs/deploy.md`). The `fantasy.lopezcloud.dev`
custom domain, Firebase Hosting site, and Cloudflare DNS records are all
provisioned via Terraform in [`lopeztech/platform-infra`](https://github.com/lopeztech/platform-infra)
under `projects/fantasy-coach/`.

CI (`.github/workflows/web-ci.yml`) builds the SPA on every PR that touches
`web/`, the root Firebase config, or the workflow itself. Automatic deploys
from `main` are a follow-up — wiring them up requires a Firebase Hosting
deploy service account exported from platform-infra.

## Configuration

The SPA needs the Firebase web-app config (API key, auth domain, project ID,
app ID) to initialise the client SDK, plus the Cloud Run API base URL.
See [`web/.env.example`](../web/.env.example) for the full list; copy it to
`web/.env.local` and fill in the values.

| Variable                       | Source                                                      |
|--------------------------------|-------------------------------------------------------------|
| `VITE_FIREBASE_API_KEY`        | Firebase console → Project settings → Your apps → Web       |
| `VITE_FIREBASE_AUTH_DOMAIN`    | `<project-id>.firebaseapp.com`                              |
| `VITE_FIREBASE_PROJECT_ID`     | GCP project ID (e.g. `fantasy-coach-lcd`)                   |
| `VITE_FIREBASE_APP_ID`         | Firebase console → Project settings → Your apps → Web       |
| `VITE_API_BASE_URL`            | Cloud Run service URL, no trailing slash                    |

The Firebase web config values are safe to ship in the client bundle — they
identify *which* project the SDK talks to, not *whether* a caller is
allowed to act. Authorisation is enforced server-side via the Firebase ID
token (see `src/fantasy_coach/auth.py`).

## Auth flow

1. User clicks **Sign in with Google** (`AuthButton`).
2. Firebase Auth handles the popup + token issuance.
3. `auth.tsx` stores the `User` in React context via `onAuthStateChanged`.
4. `api.ts`'s `apiFetch` calls `user.getIdToken()` before every request and
   attaches `Authorization: Bearer <token>`. The Firebase SDK refreshes the
   token silently when it expires — we never touch refresh logic by hand.
5. Unauthenticated users see a `<SignInRequired>` CTA on protected routes;
   there is no code path that issues an unauthenticated fetch, so we don't
   race the auth state on first load.
6. Sign-out calls `firebase/auth.signOut()`; `onAuthStateChanged` then
   clears `user` in context, which flips the UI back to the signed-out state.

## CORS (SPA origin ≠ API origin)

The SPA is served from `https://fantasy.lopezcloud.dev`; the API runs on a
different origin (Cloud Run `*.run.app`). Every authenticated fetch is
therefore a cross-origin request and triggers a preflight `OPTIONS`.

`CORSMiddleware` sits in front of `FirebaseAuthMiddleware` so preflight
requests short-circuit inside CORS before auth sees them (auth only
understands `Bearer` tokens and would 401 an empty preflight otherwise).
The default allowlist is:

| Origin                                 | Purpose                   |
|----------------------------------------|---------------------------|
| `https://fantasy.lopezcloud.dev`       | Production SPA            |
| `http://localhost:5173`                | Vite dev server           |
| `http://localhost:4173`                | `npm run preview`         |

Set `FANTASY_COACH_ALLOWED_ORIGINS` (comma-separated) to override — useful
for staging origins or review apps.


## Firebase Hosting cache strategy (#154)

`firebase.json` sets the following `Cache-Control` headers — chosen to balance
deploy immediacy against repeat-visit load time:

| Source glob | Cache-Control | Rationale |
|------------|---------------|-----------|
| `/assets/**` | `public, max-age=31536000, immutable` | Vite content-hashes all JS/CSS filenames — any change produces a new URL, so year-long immutable caching is safe. |
| `/index.html` | `no-cache, no-store, must-revalidate` | Shell must be revalidated on every visit so new deploys go live instantly; the file is tiny (< 2 KB). |
| `**/*.webmanifest` | `public, max-age=3600` | PWA manifest is generated by vite-plugin-pwa with versioned precache URLs; 1-hour freshness is enough. |
| `/icons/**` | `public, max-age=604800` | PWA icons are stable but not content-hashed; 1-week TTL avoids stale branding on rebrand. |

## Global search

The command bar (⌘K / Ctrl+K) is a purely client-side search over a small
in-memory index built at startup.

**Keyboard shortcut**: `⌘K` (Mac) / `Ctrl+K` (Windows/Linux) opens the bar
from anywhere. `Escape` closes it. Arrow keys navigate results; `Enter` selects.

**Indexed entity types**:

| Category | Source | Navigation target |
|----------|--------|-------------------|
| Teams | `GET /teams?season=...` (once per session) | `/team/:id` |
| Rounds | 1–27 generated client-side | `/round/:season/:round` |
| Venues | `GET /venues` (once per session) | `/accuracy?venue=...` |
| Pages | Static list (Home, Scoreboard, Accuracy) | direct URL |

**Recent visits**: When the query is empty the bar shows the last 5 visited
items from `localStorage` key `fc:recent_visits`. Call `recordVisit()` from
`search.ts` at navigation time to update the list.

**Matching**: case-insensitive substring. Prefix matches rank above interior
matches. No fuzzy matching — dataset is under 100 entries so plain substring
beats fuzzy for short NRL team name queries.

**API endpoints** (both require Firebase ID token):
- `GET /teams?season=<year>` — `[{id, name}]`, in-memory cached until restart.
- `GET /venues` — `[{name, city}]`, read from `data/venues.csv`.

**Mobile**: on viewports ≤ 480 px the dialog slides in from the bottom.

### Service-worker layering

The Workbox service worker (registered by `vite-plugin-pwa`) precaches all
built assets during install and serves them from the cache-first. This means
**the Firebase Hosting headers and the service worker cache operate at
different layers**:

- Firebase Hosting headers control the **network response** from the CDN to
  the browser's HTTP cache.
- The service worker's precache operates **on top** of the HTTP cache —
  requests to precached URLs bypass the network entirely after the first load.

The two layers don't conflict: the service worker updates atomically on
version bump (via `registerType: "autoUpdate"`), and `index.html` carries
`no-cache` so the browser always fetches a fresh shell and can discover the
new service worker registration.
