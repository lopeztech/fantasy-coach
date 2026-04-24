import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import type { TeamProfile, TeamScheduleEntry, TeamOption } from "../types";

const CURRENT_SEASON = new Date().getFullYear();

type Status =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; data: TeamProfile };

function ResultPill({ result }: { result: "win" | "loss" | "draw" }) {
  const cls =
    result === "win"
      ? "result-pill result-pill--win"
      : result === "loss"
        ? "result-pill result-pill--loss"
        : "result-pill result-pill--draw";
  const label = result === "win" ? "W" : result === "loss" ? "L" : "D";
  return (
    <span className={cls} aria-label={result}>
      {label}
    </span>
  );
}

function EloTrendArrow({ trend }: { trend: number | null }) {
  if (trend === null) return null;
  if (trend > 5) return <span className="elo-trend elo-trend--up" aria-label="Rising">↑</span>;
  if (trend < -5) return <span className="elo-trend elo-trend--down" aria-label="Falling">↓</span>;
  return <span className="elo-trend elo-trend--flat" aria-label="Stable">→</span>;
}

function ModelBadge({ correct }: { correct: boolean | null }) {
  if (correct === null) return <span className="muted">—</span>;
  return (
    <span
      className={`model-badge ${correct ? "model-badge--correct" : "model-badge--wrong"}`}
      aria-label={correct ? "Model correct" : "Model wrong"}
    >
      {correct ? "✓" : "✗"}
    </span>
  );
}

function formatKickoff(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "numeric",
    minute: "2-digit",
  });
}

function RecentFormStrip({ schedule }: { schedule: TeamScheduleEntry[] }) {
  const completed = schedule.filter((e) => e.result !== null);
  const last10 = completed.slice(-10);
  if (last10.length === 0) return <p className="muted">No completed matches yet.</p>;
  return (
    <div className="form-strip" aria-label="Recent form (last 10 completed matches)">
      {last10.map((e) => (
        <ResultPill key={e.matchId} result={e.result!} />
      ))}
    </div>
  );
}

function H2HSection({
  schedule,
  rivals,
}: {
  schedule: TeamScheduleEntry[];
  rivals: TeamOption[];
}) {
  const [rivalId, setRivalId] = useState<number | null>(
    rivals.length > 0 ? rivals[0].id : null,
  );

  const h2h = schedule
    .filter((e) => e.opponentId === rivalId && e.result !== null)
    .slice(-10);

  return (
    <section className="team-section" aria-label="Head to head">
      <h2 className="accuracy-section-title">Head to Head</h2>
      <div className="accuracy-filters">
        <label className="accuracy-filter-label">
          Opponent
          <select
            value={rivalId ?? ""}
            onChange={(e) => setRivalId(e.target.value ? Number(e.target.value) : null)}
            aria-label="Select opponent"
          >
            {rivals.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {h2h.length === 0 ? (
        <p className="muted">No completed matches vs this opponent yet this season.</p>
      ) : (
        <div className="accuracy-table-wrap">
          <table className="accuracy-table">
            <thead>
              <tr>
                <th>Round</th>
                <th>Venue</th>
                <th>Score</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {[...h2h].reverse().map((e) => (
                <tr key={e.matchId}>
                  <td>
                    <Link to={`/round/${e.kickoff.slice(0, 4)}/${e.round}/${e.matchId}`}>
                      Rd {e.round}
                    </Link>
                  </td>
                  <td>{e.isHome ? "Home" : "Away"}</td>
                  <td>
                    {e.score}–{e.opponentScore}
                  </td>
                  <td>
                    <ResultPill result={e.result!} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function TeamProfileRoute() {
  const { teamId } = useParams<{ teamId: string }>();
  const { user, loading: authLoading } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();

  const [season, setSeason] = useState(() => {
    const s = searchParams.get("season");
    return s ? Number(s) : CURRENT_SEASON;
  });

  const [status, setStatus] = useState<Status>({ kind: "loading" });

  useEffect(() => {
    if (authLoading || !user || !teamId) return;

    setSearchParams({ season: String(season) }, { replace: true });

    let cancelled = false;
    setStatus({ kind: "loading" });
    apiFetch<TeamProfile>(`/teams/${teamId}?season=${season}`)
      .then((data) => {
        if (!cancelled) setStatus({ kind: "ok", data });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          setStatus({ kind: "error", message: "Sign in required." });
        } else if (err instanceof ApiError) {
          setStatus({ kind: "error", message: err.message });
        } else {
          setStatus({ kind: "error", message: "Couldn't load team profile." });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [authLoading, user, teamId, season]); // eslint-disable-line react-hooks/exhaustive-deps

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to view team profiles." />;

  return (
    <section className="team-profile-page">
      <div className="round-selector">
        <label>
          Season
          <input
            type="number"
            value={season}
            min={2024}
            max={CURRENT_SEASON}
            onChange={(e) => setSeason(Number(e.target.value))}
          />
        </label>
      </div>

      {status.kind === "loading" && <p role="status">Loading team profile…</p>}

      {status.kind === "error" && (
        <div className="error-box" role="alert">
          {status.message}
        </div>
      )}

      {status.kind === "ok" && (
        <>
          {/* Header */}
          <header className="team-profile-header">
            <h1 className="team-profile-name">{status.data.teamName}</h1>
            <div className="team-profile-meta">
              <span className="team-record" aria-label="Season record">
                {status.data.wins}W–{status.data.losses}L
                {status.data.draws > 0 ? `–${status.data.draws}D` : ""}
              </span>
              {status.data.currentElo !== null && (
                <span className="team-elo" aria-label="Current Elo rating">
                  Elo {status.data.currentElo}
                  <EloTrendArrow trend={status.data.eloTrend} />
                </span>
              )}
            </div>
          </header>

          {/* Recent form */}
          <section className="team-section" aria-label="Recent form">
            <h2 className="accuracy-section-title">Recent Form</h2>
            <RecentFormStrip schedule={status.data.schedule} />
          </section>

          {/* Full season schedule */}
          <section className="team-section" aria-label="Season schedule">
            <h2 className="accuracy-section-title">Season Schedule</h2>
            <div className="accuracy-table-wrap">
              <table className="accuracy-table">
                <thead>
                  <tr>
                    <th>Round</th>
                    <th>Opponent</th>
                    <th>Venue</th>
                    <th>Kickoff</th>
                    <th>Score</th>
                    <th>Result</th>
                    <th>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {status.data.schedule.map((e) => (
                    <tr
                      key={e.matchId}
                      className={e.matchState === "FullTime" ? "" : "schedule-upcoming"}
                    >
                      <td>
                        <Link
                          to={`/round/${e.kickoff.slice(0, 4)}/${e.round}/${e.matchId}`}
                          aria-label={`Round ${e.round} match detail`}
                        >
                          Rd {e.round}
                        </Link>
                      </td>
                      <td>
                        <Link
                          to={`/team/${e.opponentId}?season=${season}`}
                          aria-label={`View ${e.opponentName} profile`}
                        >
                          {e.opponentName}
                        </Link>
                      </td>
                      <td>{e.isHome ? "Home" : "Away"}</td>
                      <td>
                        <time dateTime={e.kickoff}>{formatKickoff(e.kickoff)}</time>
                      </td>
                      <td>
                        {e.score !== null && e.opponentScore !== null
                          ? `${e.score}–${e.opponentScore}`
                          : "—"}
                      </td>
                      <td>
                        {e.result ? <ResultPill result={e.result} /> : <span className="muted">—</span>}
                      </td>
                      <td>
                        <ModelBadge correct={e.modelCorrect} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Head-to-head */}
          {status.data.rivals.length > 0 && (
            <H2HSection schedule={status.data.schedule} rivals={status.data.rivals} />
          )}
        </>
      )}
    </section>
  );
}
