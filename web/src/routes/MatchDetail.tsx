import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { labelFor } from "../features";
import { SignInRequired } from "../components/SignInRequired";
import type { Prediction } from "../types";

type Status =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "not_found" }
  | { kind: "ok"; prediction: Prediction };

function formatKickoff(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    weekday: "long",
    day: "numeric",
    month: "short",
    hour: "numeric",
    minute: "2-digit",
  });
}

export default function MatchDetail() {
  const { season: seasonParam, round: roundParam, matchId: matchIdParam } = useParams();
  const { user, loading: authLoading } = useAuth();
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  const season = Number(seasonParam);
  const round = Number(roundParam);
  const matchId = Number(matchIdParam);
  const paramsValid =
    Number.isFinite(season) && Number.isFinite(round) && Number.isFinite(matchId);

  useEffect(() => {
    if (authLoading) return;
    if (!user) return;
    if (!paramsValid) {
      setStatus({ kind: "error", message: "Season, round, and match id must be numbers." });
      return;
    }

    let cancelled = false;
    setStatus({ kind: "loading" });
    // Reuse the round endpoint — the API returns the full list for the
    // round in one RPC and the cache is already warm from Round.tsx, so
    // this avoids a second Firestore read for the common navigate-in case.
    apiFetch<Prediction[]>(`/predictions?season=${season}&round=${round}`)
      .then((predictions) => {
        if (cancelled) return;
        const match = predictions.find((p) => p.matchId === matchId);
        if (!match) {
          setStatus({ kind: "not_found" });
        } else {
          setStatus({ kind: "ok", prediction: match });
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          setStatus({ kind: "error", message: "Sign in required." });
        } else if (err instanceof ApiError) {
          setStatus({
            kind: "error",
            message: `Couldn't load prediction (HTTP ${err.status}). ${err.message}`,
          });
        } else {
          setStatus({ kind: "error", message: "Couldn't reach the prediction API." });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [authLoading, user, paramsValid, season, round, matchId]);

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to see this match." />;

  return (
    <section>
      <p className="back-link">
        <Link to={`/round/${season}/${round}`}>← Back to Round {round}</Link>
      </p>

      {status.kind === "loading" && <p role="status">Loading match…</p>}

      {status.kind === "error" && (
        <div className="error-box" role="alert">
          {status.message}
        </div>
      )}

      {status.kind === "not_found" && (
        <div className="error-box" role="alert">
          No prediction found for match {matchId} in round {round}.
        </div>
      )}

      {status.kind === "ok" && <MatchDetailBody prediction={status.prediction} />}
    </section>
  );
}

const MOBILE_TOP = 3;

function MatchDetailBody({ prediction }: { prediction: Prediction }) {
  const homePct = Math.round(prediction.homeWinProbability * 100);
  const awayPct = 100 - homePct;
  const winnerName =
    prediction.predictedWinner === "home" ? prediction.home.name : prediction.away.name;
  const winnerPct = prediction.predictedWinner === "home" ? homePct : awayPct;

  const contributions = prediction.contributions ?? [];
  const [expanded, setExpanded] = useState(false);
  const hasExtra = contributions.length > MOBILE_TOP;

  return (
    <article className="match-detail">
      <header className="match-detail-header">
        <h1>
          {prediction.home.name} <span className="muted">vs</span> {prediction.away.name}
        </h1>
        <time className="kickoff muted" dateTime={prediction.kickoff}>
          {formatKickoff(prediction.kickoff)}
        </time>
      </header>

      <div className="prob-dial" role="img" aria-label={`Home win probability ${homePct}%`}>
        <div className="prob-dial-bar">
          <span className="prob-bar-home" style={{ width: `${homePct}%` }} aria-hidden="true" />
        </div>
        <div className="prob-dial-labels">
          <span>
            <strong>{prediction.home.name}</strong> {homePct}%
          </span>
          <span>
            {awayPct}% <strong>{prediction.away.name}</strong>
          </span>
        </div>
        <p className="pick">
          Pick: <strong>{winnerName}</strong> ({winnerPct}%)
        </p>
      </div>

      {contributions.length > 0 ? (
        <section
          className={`contributions${expanded ? " contributions--expanded" : ""}`}
          aria-labelledby="why-heading"
        >
          <h2 id="why-heading">Why this pick</h2>
          <ol className="contribution-list">
            {contributions.map((c, idx) => {
              const label = labelFor(c, prediction.home.name, prediction.away.name);
              return (
                <li
                  key={c.feature}
                  className={`contribution favours-${label.favours}${idx >= MOBILE_TOP ? " contribution--extra" : ""}`}
                >
                  <span className="contribution-text">{label.text}</span>
                  <span
                    className="contribution-magnitude muted"
                    aria-label={`log-odds push ${c.contribution.toFixed(2)}`}
                  >
                    {c.contribution >= 0 ? "+" : ""}
                    {c.contribution.toFixed(2)}
                  </span>
                </li>
              );
            })}
          </ol>
          {hasExtra && (
            <button
              className="contributions-toggle"
              onClick={() => setExpanded((e) => !e)}
              aria-expanded={expanded}
            >
              {expanded
                ? "Show fewer"
                : `Show all ${contributions.length} factors`}
            </button>
          )}
          <p className="muted fine-print">
            Contributions are in log-odds units — higher magnitude means the feature pushed the
            probability harder in the direction shown.
          </p>
        </section>
      ) : (
        <p className="muted">
          This prediction was generated before per-feature explanations were available.
        </p>
      )}
    </article>
  );
}
