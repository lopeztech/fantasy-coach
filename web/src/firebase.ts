import { initializeApp, type FirebaseApp } from "firebase/app";
import { getAuth, type Auth } from "firebase/auth";

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

export function getFirebaseAuth(): Auth | null {
  if (auth) return auth;
  const cfg = readConfig();
  if (!cfg) return null;
  app = initializeApp(cfg);
  auth = getAuth(app);
  return auth;
}

export function isFirebaseConfigured(): boolean {
  return readConfig() !== null;
}

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "";
