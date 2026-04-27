import { useEffect, useState } from "react";

import { getSimulation, ApiError } from "../api";
import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import type { LadderEntry, SeasonSimulation, SeasonTeamOutcome } from "../types";

const CURRENT_SEASON = 2026;

type Status =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; sim: SeasonSimulation };

function probBar(value: number) {
  const pctVal = Math.round(value * 100);
  const color =
    pctVal >= 70
      ? "var(--color-win, #22c55e)"
      : pctVal >= 40
        ? "var(--color-text-muted)"
        : "var(--color-loss, #ef4444)";
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
      <span
        style={{
          display: "inline-block",
          width: `${pctVal}%`,
          maxWidth: "4rem",
          minWidth: "2px",
          height: "0.55rem",
          background: color,
          borderRadius: "2px",
          opacity: 0.7,
        }}
      />
      <span style={{ minWidth: "2.5rem", textAlign: "right" }}>{pctVal}%</span>
    </span>
  );
}

function LadderTable({ standings }: { standings: LadderEntry[] }) {
  return (
    <div className="ladder-scroll">
      <table className="ladder-table">
        <thead>
          <tr>
            <th>#</th>
            <th className="ladder-team-col">Team</th>
            <th>W</th>
            <th>L</th>
            <th>D</th>
            <th>Pts</th>
            <th>%</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((row) => (
            <tr
              key={row.teamId}
              className={
                row.position <= 2
                  ? "ladder-row ladder-row--top2"
                  : row.position <= 4
                    ? "ladder-row ladder-row--top4"
                    : row.position <= 8
                      ? "ladder-row ladder-row--top8"
                      : "ladder-row"
              }
            >
              <td className="ladder-pos">{row.position}</td>
              <td className="ladder-team-col">{row.teamName}</td>
              <td>{row.wins}</td>
              <td>{row.losses}</td>
              <td>{row.draws}</td>
              <td>
                <strong>{row.points}</strong>
              </td>
              <td>{row.percentage.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SimTable({
  teams,
  standings,
}: {
  teams: SeasonTeamOutcome[];
  standings: LadderEntry[];
}) {
  const posMap = new Map(standings.map((s) => [s.teamId, s.position]));
  const sorted = [...teams].sort(
    (a, b) => (posMap.get(a.teamId) ?? 99) - (posMap.get(b.teamId) ?? 99),
  );

  return (
    <div className="ladder-scroll">
      <table className="ladder-table ladder-sim-table">
        <thead>
          <tr>
            <th className="ladder-team-col">Team</th>
            <th>Top 8</th>
            <th>Top 4</th>
            <th>GF</th>
            <th>Premier</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((t) => (
            <tr key={t.teamId} className="ladder-row">
              <td className="ladder-team-col">{t.teamName}</td>
              <td>{probBar(t.playoffProb)}</td>
              <td>{probBar(t.top4Prob)}</td>
              <td>{probBar(t.grandFinalProb)}</td>
              <td>{probBar(t.premiershipProb)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Ladder() {
  const { user, loading: authLoading } = useAuth();
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  useEffect(() => {
    if (authLoading) return;
    if (!user) return;

    let cancelled = false;
    setStatus({ kind: "loading" });

    getSimulation(CURRENT_SEASON)
      .then((sim) => {
        if (!cancelled) setStatus({ kind: "ok", sim });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg =
          err instanceof ApiError
            ? err.message
            : "Couldn't load season simulation.";
        setStatus({ kind: "error", message: msg });
      });

    return () => {
      cancelled = true;
    };
  }, [authLoading, user]);

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to see the ladder and finals odds." />;
  if (status.kind === "loading") return <p role="status">Running simulation…</p>;
  if (status.kind === "error") {
    return (
      <div className="error-box" role="alert">
        {status.message}
      </div>
    );
  }

  const { sim } = status;
  const computedAt = new Date(sim.computedAt).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "numeric",
    minute: "2-digit",
  });

  return (
    <section className="ladder-page">
      <h1>
        {sim.season} Ladder &amp; Finals Odds
      </h1>
      <p className="ladder-meta muted">
        Based on {sim.nSimulations.toLocaleString()} simulations · updated {computedAt}
      </p>

      <div className="ladder-grid">
        <div>
          <h2 className="ladder-section-title">Current Ladder</h2>
          <LadderTable standings={sim.standings} />
          <p className="ladder-legend muted" style={{ fontSize: "0.75rem", marginTop: "0.5rem" }}>
            <span className="ladder-badge ladder-badge--top2">■</span> Top 2 &nbsp;
            <span className="ladder-badge ladder-badge--top4">■</span> Top 4 &nbsp;
            <span className="ladder-badge ladder-badge--top8">■</span> Top 8
          </p>
        </div>

        <div>
          <h2 className="ladder-section-title">Finals Probabilities</h2>
          <SimTable teams={sim.teams} standings={sim.standings} />
          <p className="ladder-legend muted" style={{ fontSize: "0.75rem", marginTop: "0.5rem" }}>
            GF = Grand Final appearance &nbsp;·&nbsp; Premier = premiership winner
          </p>
        </div>
      </div>
    </section>
  );
}
