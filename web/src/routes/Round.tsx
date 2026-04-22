import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { FilterBar, type SortKey } from "../components/FilterBar";
import { MatchCard } from "../components/MatchCard";
import { RoundSelector } from "../components/RoundSelector";
import { SignInRequired } from "../components/SignInRequired";
import { getFavouriteTeams, setFavouriteTeams } from "../prefs";
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

const SKELETON_COUNT = 8;

export default function Round() {
  const { season: seasonParam, round: roundParam } = useParams();
  const { user, loading: authLoading } = useAuth();
  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const [tips, setTips] = useState<Map<number, TipChoice>>(new Map());
  const [savingTip, setSavingTip] = useState<number | null>(null);
  const [favTeams, setFavTeams] = useState<Set<string>>(() => getFavouriteTeams());

  const [searchParams, setSearchParams] = useSearchParams();
  const q = searchParams.get("q") ?? "";
  const tossup = searchParams.get("tossup") === "1";
  const myTeams = searchParams.get("myteams") === "1";
  const sort = (searchParams.get("sort") ?? "kickoff") as SortKey;

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

  function updateParam(key: string, value: string | null) {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (!value) next.delete(key);
        else next.set(key, value);
        return next;
      },
      { replace: true },
    );
  }

  function handleToggleFav(name: string) {
    setFavTeams((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      setFavouriteTeams(next);
      return next;
    });
  }

  const allTeams = useMemo(() => {
    if (status.kind !== "ok") return [];
    const names = new Set<string>();
    for (const p of status.predictions) {
      names.add(p.home.name);
      names.add(p.away.name);
    }
    return [...names].sort();
  }, [status]);

  const visiblePredictions = useMemo(() => {
    if (status.kind !== "ok") return [];
    let result = status.predictions;
    if (q) {
      const ql = q.toLowerCase();
      result = result.filter(
        (p) =>
          p.home.name.toLowerCase().includes(ql) || p.away.name.toLowerCase().includes(ql),
      );
    }
    if (tossup) {
      result = result.filter((p) => Math.abs(p.homeWinProbability - 0.5) < 0.1);
    }
    if (myTeams) {
      result = result.filter((p) => favTeams.has(p.home.name) || favTeams.has(p.away.name));
    }
    if (sort === "confidence") {
      result = [...result].sort(
        (a, b) =>
          Math.abs(b.homeWinProbability - 0.5) - Math.abs(a.homeWinProbability - 0.5),
      );
    } else if (sort === "alpha") {
      result = [...result].sort((a, b) => a.home.name.localeCompare(b.home.name));
    }
    return result;
  }, [status, q, tossup, myTeams, favTeams, sort]);

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to see predictions for this round." />;

  const hasActiveFilter = q || tossup || myTeams || sort !== "kickoff";

  return (
    <section>
      <header className="round-header">
        <h1>
          Round {round}, {season}
        </h1>
        <RoundSelector initialSeason={season} initialRound={round} />
      </header>

      <FilterBar
        q={q}
        onQ={(v) => updateParam("q", v)}
        tossup={tossup}
        onTossup={(v) => updateParam("tossup", v ? "1" : null)}
        myTeams={myTeams}
        onMyTeams={(v) => updateParam("myteams", v ? "1" : null)}
        sort={sort}
        onSort={(v) => updateParam("sort", v === "kickoff" ? null : v)}
        allTeams={allTeams}
        favTeams={favTeams}
        onToggleFav={handleToggleFav}
      />

      {status.kind === "loading" && (
        <div className="match-grid" aria-busy="true">
          {Array.from({ length: SKELETON_COUNT }, (_, i) => (
            <div key={i} className="match-card-skeleton" aria-hidden="true" />
          ))}
        </div>
      )}

      {status.kind === "error" && (
        <div className="error-box" role="alert">
          {status.message}
        </div>
      )}

      {status.kind === "ok" && visiblePredictions.length === 0 && (
        <p className="muted">
          {hasActiveFilter
            ? "No matches match your filter."
            : "No predictions for this round yet."}
        </p>
      )}

      {status.kind === "ok" && visiblePredictions.length > 0 && (
        <div className="match-grid">
          {visiblePredictions.map((p) => (
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
