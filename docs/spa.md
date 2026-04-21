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
