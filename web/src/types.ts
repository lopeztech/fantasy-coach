export type Team = {
  id: number;
  name: string;
};

// One entry in a starting-XIII "missing regulars" list — carried as
// structured detail on the `key_absence_diff` FeatureContribution row.
export type MissingPlayer = {
  player_id: number;
  name: string;
  position: string;
  weight: number;
};

// Present for logistic-model predictions (see backend #58). Older cached
// predictions and non-logistic artefacts omit this — consumers must treat
// it as optional and hide their reasons panel if absent.
export type FeatureContribution = {
  feature: string;
  value: number;
  contribution: number; // signed log-odds push
  // Optional per-feature structured narrative detail (#124). Backend populates
  // specific keys when a plain text label would under-sell the reason —
  // e.g. `{home_missing, away_missing}` behind `key_absence_diff`.
  detail?: {
    home_missing?: MissingPlayer[];
    away_missing?: MissingPlayer[];
  } | null;
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

export type AccuracyOut = {
  rounds: RoundAccuracy[];
  byModelVersion: Array<{ modelVersion: string; total: number; correct: number; accuracy: number }>;
  overallAccuracy: number | null;
  belowThreshold: boolean;
  threshold: number;
  scoredMatches: number;
};
