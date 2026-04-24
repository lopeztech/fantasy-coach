/** Client-side search index for the global command bar. */

export type SearchResultKind = "team" | "match" | "round" | "venue" | "page";

export type SearchResult = {
  kind: SearchResultKind;
  label: string;
  sublabel?: string;
  href: string;
  /** Lower = higher priority within the same query */
  score: number;
};

export type SearchIndex = {
  teams: Array<{ id: number; name: string }>;
  venues: Array<{ name: string; city: string }>;
  /** Current season rounds known client-side (1–27). */
  season: number;
  maxRound: number;
};

const STATIC_PAGES: Array<{ label: string; href: string }> = [
  { label: "Home", href: "/" },
  { label: "Scoreboard", href: "/scoreboard" },
  { label: "Accuracy", href: "/accuracy" },
];

/** Normalise a string for substring comparison. */
function norm(s: string): string {
  return s.toLowerCase();
}

/** Score: 0 = prefix match (best), 1 = contained, 2 = multi-word contained. */
function matchScore(query: string, target: string): number | null {
  const q = norm(query);
  const t = norm(target);
  if (t.startsWith(q)) return 0;
  if (t.includes(q)) return 1;
  // multi-word: every word in the query appears somewhere in target
  const words = q.split(/\s+/).filter(Boolean);
  if (words.length > 1 && words.every((w) => t.includes(w))) return 2;
  return null;
}

export function search(query: string, index: SearchIndex): SearchResult[] {
  const q = query.trim();
  if (!q) return getRecentVisits(index);

  const results: SearchResult[] = [];

  // Teams
  for (const team of index.teams) {
    const s = matchScore(q, team.name);
    if (s !== null) {
      results.push({
        kind: "team",
        label: team.name,
        sublabel: "Team",
        href: `/team/${team.id}`,
        score: s,
      });
    }
  }

  // Rounds (season/round combos)
  for (let r = 1; r <= index.maxRound; r++) {
    const label = `Round ${r}`;
    const s = matchScore(q, label);
    if (s !== null) {
      results.push({
        kind: "round",
        label,
        sublabel: String(index.season),
        href: `/round/${index.season}/${r}`,
        score: s + 0.5, // slight deprioritisation vs teams
      });
    }
  }

  // Venues
  for (const venue of index.venues) {
    const s = matchScore(q, venue.name) ?? matchScore(q, venue.city);
    if (s !== null) {
      results.push({
        kind: "venue",
        label: venue.name,
        sublabel: venue.city,
        href: `/accuracy?venue=${encodeURIComponent(venue.name)}`,
        score: s + 0.5,
      });
    }
  }

  // Static pages
  for (const page of STATIC_PAGES) {
    const s = matchScore(q, page.label);
    if (s !== null) {
      results.push({
        kind: "page",
        label: page.label,
        href: page.href,
        score: s + 0.5,
      });
    }
  }

  // Sort: score asc, then kind priority (team < round < venue < page), then label asc.
  const kindPriority: Record<SearchResultKind, number> = {
    team: 0,
    match: 1,
    round: 2,
    venue: 3,
    page: 4,
  };
  results.sort((a, b) => {
    if (a.score !== b.score) return a.score - b.score;
    if (a.kind !== b.kind) return kindPriority[a.kind] - kindPriority[b.kind];
    return a.label.localeCompare(b.label);
  });

  return results.slice(0, 12);
}

// ---------------------------------------------------------------------------
// Recent visits (shown when query is empty)
// ---------------------------------------------------------------------------

const RECENT_KEY = "fc:recent_visits";
const RECENT_MAX = 5;

export type RecentVisit = {
  label: string;
  href: string;
  kind: SearchResultKind;
};

export function recordVisit(visit: RecentVisit): void {
  try {
    const existing = getStoredVisits();
    const filtered = existing.filter((v) => v.href !== visit.href);
    const updated = [visit, ...filtered].slice(0, RECENT_MAX);
    localStorage.setItem(RECENT_KEY, JSON.stringify(updated));
  } catch {
    // storage full — ignore
  }
}

function getStoredVisits(): RecentVisit[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    return raw ? (JSON.parse(raw) as RecentVisit[]) : [];
  } catch {
    return [];
  }
}

function getRecentVisits(_index: SearchIndex): SearchResult[] {
  return getStoredVisits().map((v, i) => ({
    kind: v.kind,
    label: v.label,
    href: v.href,
    sublabel: "Recent",
    score: i,
  }));
}

export function buildIndex(
  teams: Array<{ id: number; name: string }>,
  venues: Array<{ name: string; city: string }>,
  season: number,
  maxRound: number,
): SearchIndex {
  return { teams, venues, season, maxRound };
}
