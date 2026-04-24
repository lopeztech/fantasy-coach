import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import type { TeamProfile } from "../types";

const CURRENT_SEASON = 2026;

type Status =
  | { kind: "loading" }
  | { kind: "notfound" }
  | { kind: "error"; message: string }
  | { kind: "ok"; profile: TeamProfile };

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

function EloTrendArrow({ trend }: { trend: "up" | "down" | "flat" }) {
  if (trend === "up") return <span aria-label="trending up" style={{ color: "var(--color-win, #22c55e)" }}>↑</span>;
  if (trend === "down") return <span aria-label="trending down" style={{ color: "var(--color-loss, #ef4444)" }}>↓</span>;
  return <span aria-label="flat" style={{ color: "var(--color-text-muted)" }}>→</span>;
}

function FormPill({ result }: { result: string }) {
  let bg = "var(--color-text-muted)";
  let color = "#fff";
  if (result === "W") { bg = "var(--color-win, #22c55e)"; }
  else if (result === "L") { bg = "var(--color-loss, #ef4444)"; }
  return (
    <span
      aria-label={result === "W" ? "Win" : result === "L" ? "Loss" : "Draw"}
      style={{
        display: "inline-block",
        width: "1.5rem",
        height: "1.5rem",
        lineHeight: "1.5rem",
        textAlign: "center",
        borderRadius: "0.25rem",
        fontSize: "0.75rem",
        fontWeight: 700,
        background: bg,
        color,
        marginRight: "0.25rem",
      }}
    >
      {result}
    </span>
  );
}

export default function Team() {
  const { teamId: teamIdParam } = useParams<{ teamId: string }>();
  const { user, loading: authLoading } = useAuth();
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  const teamId = Number(teamIdParam);
  const validId = Number.isFinite(teamId) && teamId > 0;

  useEffect(() => {
    if (authLoading) return;
    if (!user) return;
    if (!validId) {
      setStatus({ kind: "error", message: "Invalid team ID." });
      return;
    }

    let cancelled = false;
    setStatus({ kind: "loading" });

    apiFetch<TeamProfile>(`/teams/${teamId}/profile?season=${CURRENT_SEASON}`)
      .then((profile) => {
        if (!cancelled) setStatus({ kind: "ok", profile });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          setStatus({ kind: "error", message: "Sign in required." });
        } else if (err instanceof ApiError) {
          if (err.status === 404) {
            setStatus({ kind: "notfound" });
          } else {
            setStatus({ kind: "error", message: `API error (${err.status}): ${err.message}` });
          }
        } else {
          setStatus({ kind: "error", message: "Could not load team profile." });
        }
      });

    return () => { cancelled = true; };
  }, [authLoading, user, teamId, validId]);

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to view team profiles." />;

  if (status.kind === "loading") {
    return <p role="status">Loading team profile…</p>;
  }

  if (status.kind === "notfound") {
    return (
      <div className="error-box" role="alert">
        Team not found in {CURRENT_SEASON}. The team may not have played any matches yet.
      </div>
    );
  }

  if (status.kind === "error") {
    return (
      <div className="error-box" role="alert">
        {status.message}
      </div>
    );
  }

  const { profile } = status;
  const { wins, losses, draws } = profile.currentRecord;

  return (
    <section className="team-profile">
      {/* Header */}
      <header className="team-profile-header">
        <h1 className="team-profile-name">{profile.teamName}</h1>
        <p className="team-profile-record muted">
          {wins}W – {losses}L{draws > 0 ? ` – ${draws}D` : ""} &nbsp;|&nbsp; {CURRENT_SEASON}
        </p>
      </header>

      {/* Elo */}
      <div className="team-profile-elo">
        <span className="team-profile-elo-value">Elo: <strong>{profile.currentElo}</strong></span>
        {" "}
        <EloTrendArrow trend={profile.eloTrend} />
      </div>

      {/* Recent form */}
      {profile.recentForm.length > 0 && (
        <div className="team-profile-form">
          <h2>Recent Form</h2>
          <div className="team-profile-form-pills" aria-label="Last 10 results">
            {profile.recentForm.map((r, i) => (
              <FormPill key={i} result={r} />
            ))}
          </div>
        </div>
      )}

      {/* Next fixture */}
      {profile.nextFixture && (
        <div className="team-profile-next">
          <h2>Next Fixture</h2>
          <div className="team-profile-next-card">
            <p>
              <strong>Round {profile.nextFixture.round}</strong>
              {" — "}
              {profile.nextFixture.isHome ? "Home vs " : "Away @ "}
              <Link to={`/team/${profile.nextFixture.opponentId}`}>
                {profile.nextFixture.opponent}
              </Link>
            </p>
            <p className="muted">
              <time dateTime={profile.nextFixture.kickoff}>
                {formatKickoff(profile.nextFixture.kickoff)}
              </time>
            </p>
            {profile.nextFixture.predWinner != null && (
              <p className="team-profile-prediction">
                Prediction:{" "}
                <strong>
                  {profile.nextFixture.predWinner === "home"
                    ? (profile.nextFixture.isHome ? profile.teamName : profile.nextFixture.opponent)
                    : (profile.nextFixture.isHome ? profile.nextFixture.opponent : profile.teamName)}
                </strong>
                {profile.nextFixture.predProb != null && (
                  <span className="muted">
                    {" "}({Math.round(
                      (profile.nextFixture.predWinner === "home"
                        ? profile.nextFixture.predProb
                        : 1 - profile.nextFixture.predProb) * 100
                    )}%)
                  </span>
                )}
              </p>
            )}
          </div>
        </div>
      )}

      {/* All fixtures */}
      {profile.allFixtures.length > 0 && (
        <div className="team-profile-fixtures">
          <h2>Season Fixtures</h2>
          <table className="team-profile-fixture-table">
            <thead>
              <tr>
                <th>Rd</th>
                <th>Opponent</th>
                <th>H/A</th>
                <th>Kickoff</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {profile.allFixtures.map((f) => (
                <tr key={f.matchId} className={f.result ? `fixture-row--${f.result === "W" ? "win" : f.result === "L" ? "loss" : "draw"}` : ""}>
                  <td>{f.round}</td>
                  <td>
                    <Link to={`/team/${f.opponentId}`}>{f.opponent}</Link>
                  </td>
                  <td>{f.isHome ? "H" : "A"}</td>
                  <td className="muted">
                    <time dateTime={f.kickoff}>{formatKickoff(f.kickoff)}</time>
                  </td>
                  <td>
                    {f.result ? (
                      <span>
                        <FormPill result={f.result} />
                        {f.score}–{f.opponentScore}
                      </span>
                    ) : f.predWinner != null ? (
                      <span className="muted">
                        Pred: {f.predWinner === "home"
                          ? (f.isHome ? profile.teamName : f.opponent)
                          : (f.isHome ? f.opponent : profile.teamName)}
                        {f.predProb != null && (
                          <> ({Math.round((f.predWinner === "home" ? f.predProb : 1 - f.predProb) * 100)}%)</>
                        )}
                      </span>
                    ) : (
                      <span className="muted">—</span>
                    )}
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
