import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { ShareButton } from "../components/ShareButton";
import { TeamFormChart } from "../components/TeamFormChart";
import { labelFor } from "../features";
import { SignInRequired } from "../components/SignInRequired";
import type {
  AlternativeModels,
  FeatureContribution,
  PickSummary,
  Prediction,
  TeamFormHistory,
  WhatIfOut,
} from "../types";

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
  const [homeForm, setHomeForm] = useState<TeamFormHistory | null>(null);
  const [awayForm, setAwayForm] = useState<TeamFormHistory | null>(null);

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
      .then(async (predictions) => {
        if (cancelled) return;
        const match = predictions.find((p) => p.matchId === matchId);
        if (!match) {
          setStatus({ kind: "not_found" });
          return;
        }
        setStatus({ kind: "ok", prediction: match });

        // Fetch Elo trend charts for both teams (best-effort; errors silently ignored).
        const fetchForm = async (teamId: number): Promise<TeamFormHistory | null> => {
          try {
            return await apiFetch<TeamFormHistory>(
              `/teams/${teamId}/form?season=${season}&last=20`,
            );
          } catch {
            return null;
          }
        };
        const [hf, af] = await Promise.all([
          fetchForm(match.home.id),
          fetchForm(match.away.id),
        ]);
        if (!cancelled) {
          setHomeForm(hf);
          setAwayForm(af);
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

      {status.kind === "ok" && (
        <MatchDetailBody
          prediction={status.prediction}
          season={season}
          round={round}
          homeForm={homeForm}
          awayForm={awayForm}
        />
      )}
    </section>
  );
}

const MOBILE_TOP = 3;

type ConsensusRow = {
  label: string;
  pick: PickSummary;
  isPrimary?: boolean;
};

function ConsensusPanel({
  primary,
  alternatives,
  home,
  away,
}: {
  primary: PickSummary;
  alternatives: AlternativeModels;
  home: string;
  away: string;
}) {
  const rows: ConsensusRow[] = [
    { label: "XGBoost", pick: primary, isPrimary: true },
    ...(alternatives.logistic ? [{ label: "Logistic", pick: alternatives.logistic }] : []),
    ...(alternatives.bookmaker ? [{ label: "Market", pick: alternatives.bookmaker }] : []),
  ];

  const allAgree = rows.every((r) => r.pick.predictedWinner === primary.predictedWinner);
  const winnerName = primary.predictedWinner === "home" ? home : away;

  return (
    <section className="consensus-panel" aria-labelledby="consensus-heading">
      <h2 id="consensus-heading">Three-way consensus</h2>
      {allAgree && rows.length === 3 ? (
        <div className="consensus-unanimous" role="status">
          <span className="consensus-badge consensus-badge--agree">Unanimous</span>
          <p>
            All three sources agree: <strong>{winnerName}</strong> wins.
          </p>
        </div>
      ) : null}
      <ol className="consensus-list">
        {rows.map((row) => {
          const agrees = row.pick.predictedWinner === primary.predictedWinner;
          const pct = Math.round(row.pick.homeWinProbability * 100);
          const pickName = row.pick.predictedWinner === "home" ? home : away;
          return (
            <li key={row.label} className="consensus-row">
              <span className="consensus-source">{row.label}</span>
              <span className="consensus-pick">
                <strong>{pickName}</strong> ({row.pick.predictedWinner === "home" ? pct : 100 - pct}%)
              </span>
              {!row.isPrimary && (
                <span
                  className={`consensus-badge consensus-badge--${agrees ? "agree" : "disagree"}`}
                  aria-label={agrees ? "Agrees with primary" : "Disagrees with primary"}
                >
                  {agrees ? "agree" : "disagree"}
                </span>
              )}
            </li>
          );
        })}
      </ol>
      <p className="muted fine-print">
        XGBoost is the primary pick. Logistic and Market are shown for cross-reference only.
      </p>
    </section>
  );
}

function getContribValue(contributions: FeatureContribution[], name: string, fallback: number) {
  return contributions.find((c) => c.feature === name)?.value ?? fallback;
}

function WhatIfPanel({
  matchId,
  season,
  round,
  homeTeam,
  awayTeam,
  baseProb,
  contributions,
}: {
  matchId: number;
  season: number;
  round: number;
  homeTeam: string;
  awayTeam: string;
  baseProb: number;
  contributions: FeatureContribution[];
}) {
  const [open, setOpen] = useState(false);

  const initWet = getContribValue(contributions, "is_wet", 0) > 0.5;
  const initWind = Math.round(getContribValue(contributions, "wind_kph", 0));
  const initTemp = Math.round(getContribValue(contributions, "temperature_c", 20));
  const baseRestDiff = getContribValue(contributions, "days_rest_diff", 0);

  const [isWet, setIsWet] = useState(initWet);
  const [wind, setWind] = useState(initWind);
  const [temp, setTemp] = useState(initTemp);
  const [restDelta, setRestDelta] = useState(0);

  const [result, setResult] = useState<WhatIfOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasOverrides =
    isWet !== initWet || wind !== initWind || temp !== initTemp || restDelta !== 0;

  useEffect(() => {
    if (!open) return;

    const overrides: Record<string, number> = {
      is_wet: isWet ? 1 : 0,
      wind_kph: wind,
      temperature_c: temp,
      rain_intensity: isWet ? 2 : 0,
      days_rest_diff: baseRestDiff + restDelta,
    };

    setLoading(true);
    setError(null);

    const timer = setTimeout(() => {
      apiFetch<WhatIfOut>(`/predictions/${matchId}/whatif`, {
        method: "POST",
        body: JSON.stringify({ season, round, overrides }),
      })
        .then((r) => {
          setResult(r);
          setLoading(false);
        })
        .catch((e: unknown) => {
          setError(e instanceof Error ? e.message : "Request failed");
          setLoading(false);
        });
    }, 300);

    return () => clearTimeout(timer);
  }, [open, isWet, wind, temp, restDelta, matchId, season, round, baseRestDiff]);

  const handleReset = () => {
    setIsWet(initWet);
    setWind(initWind);
    setTemp(initTemp);
    setRestDelta(0);
    setResult(null);
  };

  const displayResult = result;
  const finalProb = displayResult?.homeWinProbability ?? baseProb;
  const basePct = Math.round(baseProb * 100);
  const finalPct = Math.round(finalProb * 100);
  const delta = finalPct - basePct;

  return (
    <section className="whatif-panel" aria-labelledby="whatif-heading">
      <button
        className="whatif-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span id="whatif-heading">Try it yourself</span>
        <span aria-hidden="true">{open ? " ▲" : " ▼"}</span>
      </button>

      {open && (
        <div className="whatif-body">
          <div className="whatif-controls">
            <fieldset className="whatif-fieldset">
              <legend>Conditions</legend>
              <label className="whatif-label">
                <span>Wet weather</span>
                <input
                  type="checkbox"
                  checked={isWet}
                  onChange={(e) => setIsWet(e.target.checked)}
                  aria-label={`Wet weather: currently ${isWet ? "on" : "off"}`}
                />
              </label>
              <label className="whatif-label">
                <span>Wind: {wind} km/h</span>
                <input
                  type="range"
                  min={0}
                  max={60}
                  step={5}
                  value={wind}
                  onChange={(e) => setWind(Number(e.target.value))}
                  aria-label={`Wind speed: ${wind} km/h`}
                />
              </label>
              <label className="whatif-label">
                <span>Temperature: {temp}°C</span>
                <input
                  type="range"
                  min={5}
                  max={35}
                  step={1}
                  value={temp}
                  onChange={(e) => setTemp(Number(e.target.value))}
                  aria-label={`Temperature: ${temp}°C`}
                />
              </label>
            </fieldset>

            <fieldset className="whatif-fieldset">
              <legend>Scheduling</legend>
              <label className="whatif-label">
                <span>
                  Home rest edge:{" "}
                  {restDelta > 0 ? `+${restDelta}` : String(restDelta)} days
                </span>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={1}
                  value={restDelta}
                  onChange={(e) => setRestDelta(Number(e.target.value))}
                  aria-label={`Home rest advantage relative to actual: ${restDelta} days`}
                />
              </label>
            </fieldset>
          </div>

          {loading && (
            <p className="whatif-status muted" role="status">
              Recalculating…
            </p>
          )}
          {error && (
            <p className="whatif-error" role="alert">
              {error}
            </p>
          )}

          <div className="whatif-result" aria-live="polite">
            <div className="whatif-row">
              <span className="whatif-label-sm muted">Baseline</span>
              <div className="prob-dial-bar">
                <span
                  className="prob-bar-home"
                  style={{ width: `${basePct}%` }}
                  aria-hidden="true"
                />
              </div>
              <div className="prob-dial-labels">
                <span>
                  <strong>{homeTeam}</strong> {basePct}%
                </span>
                <span>
                  {100 - basePct}% <strong>{awayTeam}</strong>
                </span>
              </div>
            </div>

            {displayResult && (
              <div className="whatif-row">
                <span className="whatif-label-sm muted">What-if</span>
                <div className="prob-dial-bar">
                  <span
                    className="prob-bar-home"
                    style={{ width: `${finalPct}%` }}
                    aria-hidden="true"
                  />
                </div>
                <div className="prob-dial-labels">
                  <span>
                    <strong>{homeTeam}</strong> {finalPct}%
                    {delta !== 0 && (
                      <span
                        className={delta > 0 ? "whatif-delta-pos" : "whatif-delta-neg"}
                        aria-label={`change of ${delta > 0 ? "+" : ""}${delta} percentage points`}
                      >
                        {" "}
                        ({delta > 0 ? "+" : ""}
                        {delta}pp)
                      </span>
                    )}
                  </span>
                  <span>
                    {100 - finalPct}% <strong>{awayTeam}</strong>
                  </span>
                </div>
              </div>
            )}
          </div>

          {hasOverrides && (
            <button className="whatif-reset" onClick={handleReset}>
              Reset to actual
            </button>
          )}

          <p className="muted fine-print">
            Adjusts model inputs and reruns the prediction. Other factors may dominate —
            a pick flip is not guaranteed.
          </p>
        </div>
      )}
    </section>
  );
}

function MatchDetailBody({
  prediction,
  season,
  round,
  homeForm,
  awayForm,
}: {
  prediction: Prediction;
  season: number;
  round: number;
  homeForm: TeamFormHistory | null;
  awayForm: TeamFormHistory | null;
}) {
  const homePct = Math.round(prediction.homeWinProbability * 100);
  const awayPct = 100 - homePct;
  const winnerName =
    prediction.predictedWinner === "home" ? prediction.home.name : prediction.away.name;
  const winnerPct = prediction.predictedWinner === "home" ? homePct : awayPct;

  const contributions = prediction.contributions ?? [];
  const [expanded, setExpanded] = useState(false);
  const hasExtra = contributions.length > MOBILE_TOP;

  // Inject per-match OG meta tags so JS-capable crawlers (e.g. Facebookbot) get
  // rich previews. Static crawlers (Twitterbot, Slackbot) should use the
  // server-rendered /share/match/{id} URL instead (see ShareButton).
  useEffect(() => {
    const title = `${prediction.home.name} vs ${prediction.away.name} — ${winnerName} ${winnerPct}% — Fantasy Coach`;
    document.title = title;
    const setMeta = (prop: string, content: string) => {
      let el = document.querySelector<HTMLMetaElement>(`meta[property="${prop}"]`);
      if (!el) {
        el = document.createElement("meta");
        el.setAttribute("property", prop);
        document.head.appendChild(el);
      }
      el.setAttribute("content", content);
    };
    setMeta("og:title", title);
    setMeta(
      "og:description",
      `${prediction.home.name} ${homePct}% vs ${prediction.away.name} ${awayPct}%`,
    );
    return () => {
      document.title = "Fantasy Coach";
    };
  }, [prediction, winnerName, winnerPct, homePct, awayPct]);

  return (
    <article className="match-detail">
      <header className="match-detail-header">
        <h1>
          {prediction.home.name} <span className="muted">vs</span> {prediction.away.name}
        </h1>
        <div className="match-detail-meta">
          <time className="kickoff muted" dateTime={prediction.kickoff}>
            {formatKickoff(prediction.kickoff)}
          </time>
          <ShareButton
            matchId={prediction.matchId}
            season={season}
            round={round}
            homeTeam={prediction.home.name}
            awayTeam={prediction.away.name}
          />
        </div>
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

      {(homeForm || awayForm) && (
        <section className="form-charts" aria-label="Team Elo trends">
          {homeForm && <TeamFormChart matches={homeForm.matches} teamName={prediction.home.name} />}
          {awayForm && <TeamFormChart matches={awayForm.matches} teamName={prediction.away.name} />}
        </section>
      )}

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
                  {c.detail?.interaction && (
                    <span className="contribution-interaction muted">
                      × {c.detail.interaction.partner.replace(/_/g, " ")}
                    </span>
                  )}
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

      {prediction.alternatives &&
        (prediction.alternatives.logistic != null ||
          prediction.alternatives.bookmaker != null) && (
          <ConsensusPanel
            primary={{
              predictedWinner: prediction.predictedWinner,
              homeWinProbability: prediction.homeWinProbability,
            }}
            alternatives={prediction.alternatives}
            home={prediction.home.name}
            away={prediction.away.name}
          />
        )}

      <WhatIfPanel
        matchId={prediction.matchId}
        season={season}
        round={round}
        homeTeam={prediction.home.name}
        awayTeam={prediction.away.name}
        baseProb={prediction.homeWinProbability}
        contributions={prediction.contributions ?? []}
      />
    </article>
  );
}
