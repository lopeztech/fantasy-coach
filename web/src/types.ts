export type Team = {
  id: number;
  name: string;
};

// Present for logistic-model predictions (see backend #58). Older cached
// predictions and non-logistic artefacts omit this — consumers must treat
// it as optional and hide their reasons panel if absent.
export type FeatureContribution = {
  feature: string;
  value: number;
  contribution: number; // signed log-odds push
  // Optional structured data for richer rendering (#124).
  // For key_absence_diff: { home_missing: MissingPlayer[], away_missing: MissingPlayer[] }
  detail?: Record<string, unknown> | null;
};

export type Prediction = {
  matchId: number;
  home: Team;
  away: Team;
  kickoff: string; // ISO 8601 UTC
  predictedWinner: "home" | "away";
  homeWinProbability: number; // 0..1
  modelVersion: string;
  featureHash: string;
  contributions?: FeatureContribution[] | null;
  actualWinner?: "home" | "away" | null;
};

export type RoundAccuracy = {
  season: number;
  round: number;
  modelVersion: string;
  total: number;
  correct: number;
  accuracy: number;
};

export type TeamFormEntry = {
  round: number;
  matchId: number;
  opponentId: number;
  opponentName: string;
  isHome: boolean;
  result: "win" | "loss" | "draw";
  score: number;
  opponentScore: number;
  eloAfter: number;
  eloDelta: number;
  kickoff: string;
};

export type TeamFormHistory = {
  teamId: number;
  teamName: string;
  season: number;
  matches: TeamFormEntry[];
};

export type AccuracyOut = {
  rounds: RoundAccuracy[];
  byModelVersion: Array<{ modelVersion: string; total: number; correct: number; accuracy: number }>;
  overallAccuracy: number | null;
  belowThreshold: boolean;
  threshold: number;
  scoredMatches: number;
};
