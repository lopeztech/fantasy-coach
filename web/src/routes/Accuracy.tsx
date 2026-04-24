import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { AccuracyFilterBar } from "../components/AccuracyFilters";
import { AccuracyTrendChart } from "../components/AccuracyTrendChart";
import { SignInRequired } from "../components/SignInRequired";
import type { AccuracyFilters, AccuracyOut } from "../types";

const CURRENT_SEASON = new Date().getFullYear();
const DEBOUNCE_MS = 200;

type Status =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; data: AccuracyOut };

function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

function AccuracyBadge({ accuracy, threshold }: { accuracy: number; threshold: number }) {
  const ok = accuracy >= threshold;
  return (
    <span className={`accuracy-badge ${ok ? "accuracy-badge--ok" : "accuracy-badge--warn"}`}>
      {pct(accuracy)}
    </span>
  );
}

function buildUrl(
  season: number,
  lastN: number,
  filters: AccuracyFilters,
): string {
  const p = new URLSearchParams({ season: String(season), last_n_rounds: String(lastN) });
  if (filters.teamId !== null) p.set("team_id", String(filters.teamId));
  if (filters.venue !== null) p.set("venue", filters.venue);
  if (filters.modelVersion !== null) p.set("model_version", filters.modelVersion);
  return `/accuracy?${p}`;
}

export default function Accuracy() {
  const { user, loading: authLoading } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();

  // Initialise from URL search params so links are shareable.
  const [season, setSeason] = useState(() => {
    const s = searchParams.get("season");
    return s ? Number(s) : CURRENT_SEASON;
  });
  const [lastN, setLastN] = useState(() => {
    const n = searchParams.get("last_n_rounds");
    return n ? Math.max(1, Math.min(27, Number(n))) : 10;
  });
  const [filters, setFilters] = useState<AccuracyFilters>(() => ({
    teamId: searchParams.get("team_id") ? Number(searchParams.get("team_id")) : null,
    venue: searchParams.get("venue") ?? null,
    modelVersion: searchParams.get("model_version") ?? null,
  }));

  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (authLoading || !user) return;

    // Sync filter state back into the URL so links are shareable.
    const p: Record<string, string> = { season: String(season), last_n_rounds: String(lastN) };
    if (filters.teamId !== null) p.team_id = String(filters.teamId);
    if (filters.venue !== null) p.venue = filters.venue;
    if (filters.modelVersion !== null) p.model_version = filters.modelVersion;
    setSearchParams(p, { replace: true });

    // Debounce fetches to avoid stampeding the API on rapid filter changes.
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      let cancelled = false;
      setStatus({ kind: "loading" });
      apiFetch<AccuracyOut>(buildUrl(season, lastN, filters))
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
            setStatus({ kind: "error", message: "Couldn't load accuracy data." });
          }
        });
      return () => {
        cancelled = true;
      };
    }, DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [authLoading, user, season, lastN, filters]); // eslint-disable-line react-hooks/exhaustive-deps

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to view model accuracy." />;

  const catalogueTeams = status.kind === "ok" ? status.data.teams : [];
  const catalogueVenues = status.kind === "ok" ? status.data.venues : [];
  const catalogueMvs = status.kind === "ok" ? status.data.modelVersions : [];

  return (
    <section className="accuracy-page">
      <h1>Model Accuracy</h1>

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
        <label>
          Last N rounds
          <input
            type="number"
            value={lastN}
            min={1}
            max={27}
            onChange={(e) => setLastN(Math.max(1, Math.min(27, Number(e.target.value))))}
          />
        </label>
      </div>

      <AccuracyFilterBar
        filters={filters}
        teams={catalogueTeams}
        venues={catalogueVenues}
        modelVersions={catalogueMvs}
        onChange={setFilters}
      />

      {status.kind === "loading" && <p role="status">Loading accuracy data…</p>}

      {status.kind === "error" && (
        <div className="error-box" role="alert">
          {status.message}
        </div>
      )}

      {status.kind === "ok" && status.data.scoredMatches === 0 && (
        <p className="muted">No scored matches found for this selection.</p>
      )}

      {status.kind === "ok" && status.data.scoredMatches > 0 && (
        <>
          {/* Overall summary */}
          <div className="accuracy-summary">
            <div className="accuracy-stat">
              <span className="accuracy-stat-label">Overall accuracy</span>
              {status.data.overallAccuracy !== null ? (
                <AccuracyBadge
                  accuracy={status.data.overallAccuracy}
                  threshold={status.data.threshold}
                />
              ) : (
                <span className="muted">—</span>
              )}
            </div>
            <div className="accuracy-stat">
              <span className="accuracy-stat-label">Scored matches</span>
              <span className="accuracy-stat-value">{status.data.scoredMatches}</span>
            </div>
            <div className="accuracy-stat">
              <span className="accuracy-stat-label">Threshold</span>
              <span className="accuracy-stat-value">{pct(status.data.threshold)}</span>
            </div>
          </div>

          {status.data.belowThreshold && (
            <div className="accuracy-alert" role="alert">
              ⚠ Model accuracy is below the {pct(status.data.threshold)} threshold — consider
              retraining.
            </div>
          )}

          {/* Rolling accuracy trend chart */}
          {status.data.rounds.length >= 2 && (
            <>
              <h2 className="accuracy-section-title">Accuracy trend</h2>
              <AccuracyTrendChart rounds={status.data.rounds} />
            </>
          )}

          {/* Round-by-round table */}
          <h2 className="accuracy-section-title">By round</h2>
          <div className="accuracy-table-wrap">
            <table className="accuracy-table">
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Model</th>
                  <th>Correct</th>
                  <th>Total</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {[...status.data.rounds].reverse().map((r) => (
                  <tr key={`${r.season}-${r.round}`}>
                    <td>Rd {r.round}</td>
                    <td>
                      <code className="model-version">{r.modelVersion.slice(0, 8)}</code>
                    </td>
                    <td>{r.correct}</td>
                    <td>{r.total}</td>
                    <td>
                      <AccuracyBadge
                        accuracy={r.accuracy}
                        threshold={status.data.threshold}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Model-version breakdown */}
          {status.data.byModelVersion.length > 1 && (
            <>
              <h2 className="accuracy-section-title">By model version</h2>
              <div className="accuracy-table-wrap">
                <table className="accuracy-table">
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Correct</th>
                      <th>Total</th>
                      <th>Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {status.data.byModelVersion.map((mv) => (
                      <tr key={mv.modelVersion}>
                        <td>
                          <code className="model-version">{mv.modelVersion.slice(0, 8)}</code>
                        </td>
                        <td>{mv.correct}</td>
                        <td>{mv.total}</td>
                        <td>
                          <AccuracyBadge
                            accuracy={mv.accuracy}
                            threshold={status.data.threshold}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
