import { getFirebaseAuth, API_BASE_URL } from "./firebase";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class NotSignedInError extends Error {
  constructor() {
    super("Sign in required");
    this.name = "NotSignedInError";
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const auth = getFirebaseAuth();
  const user = auth?.currentUser;
  if (!user) {
    throw new NotSignedInError();
  }

  // Firebase SDK refreshes the ID token transparently when it expires.
  const token = await user.getIdToken();
  const url = `${API_BASE_URL}${path}`;
  const headers = new Headers(init?.headers);
  headers.set("Authorization", `Bearer ${token}`);
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, body || res.statusText);
  }
  return (await res.json()) as T;
}
