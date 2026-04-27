const FAV_TEAMS_KEY = "fc:fav_teams";
const ONBOARDING_KEY = "fc:onboarding_v1";

export function getFavouriteTeams(): Set<string> {
  try {
    const raw = localStorage.getItem(FAV_TEAMS_KEY);
    return raw ? new Set(JSON.parse(raw) as string[]) : new Set();
  } catch {
    return new Set();
  }
}

export function setFavouriteTeams(teams: Set<string>): void {
  try {
    localStorage.setItem(FAV_TEAMS_KEY, JSON.stringify([...teams]));
  } catch {
    // storage full — ignore
  }
}

export function getOnboardingCompleted(): boolean {
  try {
    return localStorage.getItem(ONBOARDING_KEY) === "1";
  } catch {
    return false;
  }
}

export function setOnboardingCompleted(): void {
  try {
    localStorage.setItem(ONBOARDING_KEY, "1");
  } catch {
    // storage full — ignore
  }
}
