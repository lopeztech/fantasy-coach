import { getFirebaseAuth, API_BASE_URL } from "./firebase";
import type { DashboardOut, SeasonSimulation, TeamOption } from "./types";

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

export type VenueOption = { name: string; city: string };

export async function getTeams(season: number): Promise<TeamOption[]> {
  return apiFetch<TeamOption[]>(`/teams?season=${season}`);
}

export async function getVenues(): Promise<VenueOption[]> {
  return apiFetch<VenueOption[]>("/venues");
}

export async function getDashboard(
  season: number,
  favouriteTeamId?: number | null,
): Promise<DashboardOut> {
  const params = new URLSearchParams({ season: String(season) });
  if (favouriteTeamId != null) {
    params.set("favourite_team_id", String(favouriteTeamId));
  }
  return apiFetch<DashboardOut>(`/me/dashboard?${params.toString()}`);
}

export async function getSimulation(season: number): Promise<SeasonSimulation> {
  return apiFetch<SeasonSimulation>(`/season/${season}/simulation`);
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
