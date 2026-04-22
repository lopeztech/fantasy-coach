import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { MatchCard } from "../components/MatchCard";
import { RoundSelector } from "../components/RoundSelector";
import { SignInRequired } from "../components/SignInRequired";
import { getTipsByRound, saveTip, type TipChoice } from "../tips";
import type { Prediction, TeamFormHistory } from "../types";

const FORM_CACHE_TTL = 6 * 60 * 60 * 1000;

function formCacheGet(teamId: number, season: number): TeamFormHistory | null {
  try {
    const raw = localStorage.getItem(`form:${season}:${teamId}`);
    if (!raw) return null;
    const { ts, data } = JSON.parse(raw) as { ts: number; data: TeamFormHistory };
    if (Date.now() - ts > FORM_CACHE_TTL) return null;
    return data;
  } catch {
    return null;
  }
}

function formCacheSet(teamId: number, season: number, data: TeamFormHistory): void {
  try {
    localStorage.setItem(`form:${season}:${teamId}`, JSON.stringify({ ts: Date.now(), data }));
  } catch {
    // storage full — ignore
  }
}

async function fetchTeamForm(
  apiFetchFn: typeof apiFetch,
  teamId: number,
  season: number,
): Promise<TeamFormHistory | null> {
  const cached = formCacheGet(teamId, season);
  if (cached) return cached;
  try {
    const data = await apiFetchFn<TeamFormHistory>(
      `/teams/${teamId}/form?season=${season}&last=20`,
    );
    formCacheSet(teamId, season, data);
    return data;
  } catch {
    return null;
  }
}

type Status =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; predictions: Prediction[]; teamForm: Map<number, TeamFormHistory | null> };

export default function Round() {
  const { season: seasonParam, round: roundParam } = useParams();
  const { user, loading: authLoading } = useAuth();
  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const [tips, setTips] = useState<Map<number, TipChoice>>(new Map());
  const [savingTip, setSavingTip] = useState<number | null>(null);

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

    Promise.all([
      apiFetch<Prediction[]>(`/predictions?season=${season}&round=${round}`),
      getTipsByRound(user.uid, season, round).catch(() => new Map<number, TipChoice>()),
    ])
      .then(async ([predictions, loadedTips]) => {
        if (cancelled) return;
        setTips(loadedTips);

        const teamIds = [...new Set(predictions.flatMap((p) => [p.home.id, p.away.id]))];
        const formResults = await Promise.all(
          teamIds.map((id) => fetchTeamForm(apiFetch, id, season)),
        );
        if (cancelled) return;

        const teamForm = new Map(teamIds.map((id, i) => [id, formResults[i]]));
        setStatus({ kind: "ok", predictions, teamForm });
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

  const handleTip = useCallback(
    async (matchId: number, kickoff: string, choice: TipChoice) => {
      if (!user) return;
      setSavingTip(matchId);
      try {
        await saveTip(user.uid, matchId, choice, kickoff, season, round);
        setTips((prev) => new Map(prev).set(matchId, choice));
      } catch {
        // silently fail — tip just won't persist
      } finally {
        setSavingTip(null);
      }
    },
    [user, season, round],
  );

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

      {status.kind === "loading" && <p role="status">Loading predictions…</p>}

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
            <MatchCard
              key={p.matchId}
              prediction={p}
              season={season}
              round={round}
              tip={tips.get(p.matchId) ?? null}
              savingTip={savingTip === p.matchId}
              onTip={(choice) => void handleTip(p.matchId, p.kickoff, choice)}
              homeForm={status.teamForm.get(p.home.id)}
              awayForm={status.teamForm.get(p.away.id)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
