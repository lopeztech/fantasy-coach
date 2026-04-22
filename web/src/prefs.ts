const FAV_TEAMS_KEY = "fc:fav_teams";

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
