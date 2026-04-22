import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { MatchCard } from "../components/MatchCard";
import { RoundSelector } from "../components/RoundSelector";
import { SignInRequired } from "../components/SignInRequired";
import type { Prediction } from "../types";

type Status =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; predictions: Prediction[] };

export default function Round() {
  const { season: seasonParam, round: roundParam } = useParams();
  const { user, loading: authLoading } = useAuth();
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  const season = Number(seasonParam);
  const round = Number(roundParam);
  const paramsValid = Number.isFinite(season) && Number.isFinite(round);

  useEffect(() => {
    if (authLoading) return;
    if (!user) return;
    if (!paramsValid) {
      setStatus({ kind: "error", message: "Season and round must be numbers." });
      return;
    }

    let cancelled = false;
    setStatus({ kind: "loading" });
    apiFetch<Prediction[]>(`/predictions?season=${season}&round=${round}`)
      .then((predictions) => {
        if (!cancelled) setStatus({ kind: "ok", predictions });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          setStatus({ kind: "error", message: "Sign in required." });
        } else if (err instanceof ApiError) {
          setStatus({
            kind: "error",
            message: `Couldn't load predictions (HTTP ${err.status}). ${err.message}`,
          });
        } else {
          setStatus({ kind: "error", message: "Couldn't reach the prediction API." });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [authLoading, user, paramsValid, season, round]);

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to see predictions for this round." />;

  return (
    <section>
      <header className="round-header">
        <h1>
          Round {round}, {season}
        </h1>
        <RoundSelector initialSeason={season} initialRound={round} />
      </header>

      {status.kind === "loading" && <p>Loading predictions…</p>}

      {status.kind === "error" && (
        <div className="error-box" role="alert">
          {status.message}
        </div>
      )}

      {status.kind === "ok" && status.predictions.length === 0 && (
        <p className="muted">No predictions for this round yet.</p>
      )}

      {status.kind === "ok" && status.predictions.length > 0 && (
        <div className="match-grid">
          {status.predictions.map((p) => (
            <MatchCard key={p.matchId} prediction={p} season={season} round={round} />
          ))}
        </div>
      )}
    </section>
  );
}
