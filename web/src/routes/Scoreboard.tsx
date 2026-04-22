import { useEffect, useState } from "react";

import { apiFetch, ApiError } from "../api";
import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import { getAllTips, type TipChoice } from "../tips";
import type { Prediction } from "../types";

type TipRecord = {
  matchId: number;
  season: number;
  round: number;
  tip: TipChoice;
};

type MatchResult = {
  matchId: number;
  home: string;
  away: string;
  kickoff: string;
  tip: TipChoice;
  predictedWinner: "home" | "away";
  homeWinProbability: number;
  actualWinner: "home" | "away" | null;
};

type RoundGroup = {
  season: number;
  round: number;
  matches: MatchResult[];
};

type Status =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; rounds: RoundGroup[] };

function logLoss(prob: number, actual: "home" | "away"): number {
  const p = actual === "home" ? prob : 1 - prob;
  return -Math.log(Math.max(p, 1e-9));
}

function formatPct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

export default function Scoreboard() {
  const { user, loading: authLoading } = useAuth();
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  useEffect(() => {
    if (authLoading) return;
    if (!user) return;

    let cancelled = false;
    setStatus({ kind: "loading" });

    getAllTips(user.uid)
      .then(async (tipRecords: TipRecord[]) => {
        if (cancelled) return;
        if (tipRecords.length === 0) {
          setStatus({ kind: "ok", rounds: [] });
          return;
        }

        // Group by season+round, fetch predictions for each distinct round
        const roundKeys = [
          ...new Set(tipRecords.map((t) => `${t.season}:${t.round}`)),
        ];
        const predsByRound = new Map<string, Prediction[]>();
        await Promise.all(
          roundKeys.map(async (key) => {
            const [season, round] = key.split(":").map(Number);
            try {
              const preds = await apiFetch<Prediction[]>(
                `/predictions?season=${season}&round=${round}`,
              );
              predsByRound.set(key, preds);
            } catch {
              predsByRound.set(key, []);
            }
          }),
        );

        if (cancelled) return;

        // Build round groups
        const grouped = new Map<string, RoundGroup>();
        for (const tip of tipRecords) {
          const key = `${tip.season}:${tip.round}`;
          const preds = predsByRound.get(key) ?? [];
          const pred = preds.find((p) => p.matchId === tip.matchId);
          if (!pred) continue;
          if (!grouped.has(key)) {
            grouped.set(key, { season: tip.season, round: tip.round, matches: [] });
          }
          grouped.get(key)!.matches.push({
            matchId: tip.matchId,
            home: pred.home.name,
            away: pred.away.name,
            kickoff: pred.kickoff,
            tip: tip.tip,
            predictedWinner: pred.predictedWinner,
            homeWinProbability: pred.homeWinProbability,
            actualWinner: pred.actualWinner ?? null,
          });
        }

        const rounds = [...grouped.values()].sort(
          (a, b) => b.season - a.season || b.round - a.round,
        );
        setStatus({ kind: "ok", rounds });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg = err instanceof ApiError ? err.message : "Couldn't load scoreboard.";
        setStatus({ kind: "error", message: msg });
      });

    return () => {
      cancelled = true;
    };
  }, [authLoading, user]);

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to see your scoreboard." />;

  if (status.kind === "loading") return <p role="status">Loading scoreboard…</p>;
  if (status.kind === "error") {
    return (
      <div className="error-box" role="alert">
        {status.message}
      </div>
    );
  }

  if (status.rounds.length === 0) {
    return (
      <div className="sign-in-required">
        <p>No tips yet. Head to a round and start tipping!</p>
      </div>
    );
  }

  // Overall stats across all scored matches
  const allScored = status.rounds
    .flatMap((r) => r.matches)
    .filter((m) => m.actualWinner !== null);

  const userCorrect = allScored.filter((m) => m.tip === m.actualWinner).length;
  const modelCorrect = allScored.filter((m) => m.predictedWinner === m.actualWinner).length;
  const modelLoss =
    allScored.length > 0
      ? allScored.reduce(
          (sum, m) => sum + logLoss(m.homeWinProbability, m.actualWinner!),
          0,
        ) / allScored.length
      : null;

  return (
    <section className="scoreboard">
      <h1>You vs the Model</h1>

      {allScored.length > 0 && (
        <div className="scoreboard-summary">
          <div className="scoreboard-stat">
            <span className="scoreboard-stat-label">Your accuracy</span>
            <span className="scoreboard-stat-value">
              {formatPct(userCorrect / allScored.length)}{" "}
              <span className="muted">
                ({userCorrect}/{allScored.length})
              </span>
            </span>
          </div>
          <div className="scoreboard-stat">
            <span className="scoreboard-stat-label">Model accuracy</span>
            <span className="scoreboard-stat-value">
              {formatPct(modelCorrect / allScored.length)}{" "}
              <span className="muted">
                ({modelCorrect}/{allScored.length})
              </span>
            </span>
          </div>
          {modelLoss !== null && (
            <div className="scoreboard-stat">
              <span className="scoreboard-stat-label">Model log loss</span>
              <span className="scoreboard-stat-value">{modelLoss.toFixed(3)}</span>
            </div>
          )}
        </div>
      )}

      {status.rounds.map((rg) => {
        const scored = rg.matches.filter((m) => m.actualWinner !== null);
        return (
          <details key={`${rg.season}-${rg.round}`} className="scoreboard-round" open>
            <summary className="scoreboard-round-header">
              Round {rg.round}, {rg.season}
              {scored.length > 0 && (
                <span className="muted scoreboard-round-score">
                  {" "}
                  — you {scored.filter((m) => m.tip === m.actualWinner).length}/
                  {scored.length}, model {scored.filter((m) => m.predictedWinner === m.actualWinner).length}/
                  {scored.length}
                </span>
              )}
            </summary>
            <ul className="scoreboard-list">
              {rg.matches.map((m) => {
                const pending = m.actualWinner === null;
                const userRight = !pending && m.tip === m.actualWinner;
                const modelRight = !pending && m.predictedWinner === m.actualWinner;
                const tipName = m.tip === "home" ? m.home : m.away;
                const predName = m.predictedWinner === "home" ? m.home : m.away;
                return (
                  <li
                    key={m.matchId}
                    className={`scoreboard-match${pending ? " scoreboard-match--pending" : userRight ? " scoreboard-match--correct" : " scoreboard-match--wrong"}`}
                  >
                    <span className="scoreboard-teams">
                      {m.home} vs {m.away}
                    </span>
                    <span className="scoreboard-tip">
                      Your tip: <strong>{tipName}</strong>
                      {!pending && (
                        <span className={`tip-result ${userRight ? "tip-result--correct" : "tip-result--wrong"}`}>
                          {userRight ? "✓" : "✗"}
                        </span>
                      )}
                    </span>
                    <span className="scoreboard-model">
                      Model: <strong>{predName}</strong>
                      {!pending && (
                        <span className={`tip-result ${modelRight ? "tip-result--correct" : "tip-result--wrong"}`}>
                          {modelRight ? "✓" : "✗"}
                        </span>
                      )}
                    </span>
                    {pending && <span className="scoreboard-pending muted">Pending</span>}
                  </li>
                );
              })}
            </ul>
          </details>
        );
      })}
    </section>
  );
}
