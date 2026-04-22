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
