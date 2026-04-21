# SPA (web/)

Lightweight frontend for triggering predictions and browsing results. Lives
under `web/` and deploys to Firebase Hosting. Everything else (the Cloud Run
API, the model, the scraper) is intentionally the main attraction — this
frontend is deliberately thin.

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
the Cloud Run API runs in — see `docs/deploy.md`). Enabling the Firebase
Hosting product on the project is handled in
[`lopeztech/platform-infra`](https://github.com/lopeztech/platform-infra).

CI (`.github/workflows/web-ci.yml`) builds the SPA on every PR that touches
`web/`, the root Firebase config, or the workflow itself. Automatic deploys
from `main` are a follow-up — wiring them up requires a Firebase Hosting
deploy service account exported from platform-infra.

## Configuration

The SPA needs the Firebase web-app config (API key, auth domain, project ID)
to initialise the client SDK. These land in #20 via `VITE_FIREBASE_*` env
vars; for now the scaffold builds without them.
