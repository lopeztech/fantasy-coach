import { initializeApp, type FirebaseApp } from "firebase/app";
import { getAuth, type Auth } from "firebase/auth";
import { getFirestore, type Firestore } from "firebase/firestore";

type FirebaseConfig = {
  apiKey: string;
  authDomain: string;
  projectId: string;
  appId: string;
};

function readConfig(): FirebaseConfig | null {
  const cfg = {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
    appId: import.meta.env.VITE_FIREBASE_APP_ID,
  };
  if (!cfg.apiKey || !cfg.authDomain || !cfg.projectId || !cfg.appId) {
    return null;
  }
  return cfg as FirebaseConfig;
}

let app: FirebaseApp | null = null;
let auth: Auth | null = null;
let firestore: Firestore | null = null;

function _ensureApp(): FirebaseApp | null {
  if (app) return app;
  const cfg = readConfig();
  if (!cfg) return null;
  app = initializeApp(cfg);
  return app;
}

export function getFirebaseAuth(): Auth | null {
  if (auth) return auth;
  const a = _ensureApp();
  if (!a) return null;
  auth = getAuth(a);
  return auth;
}

export function getFirebaseFirestore(): Firestore | null {
  if (firestore) return firestore;
  const a = _ensureApp();
  if (!a) return null;
  firestore = getFirestore(a);
  return firestore;
}

export function isFirebaseConfigured(): boolean {
  return readConfig() !== null;
}

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "";
