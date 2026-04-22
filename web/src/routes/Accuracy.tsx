import { useEffect, useState } from "react";

import { apiFetch, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import type { AccuracyOut } from "../types";

const CURRENT_SEASON = new Date().getFullYear();

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

export default function Accuracy() {
  const { user, loading: authLoading } = useAuth();
  const [season, setSeason] = useState(CURRENT_SEASON);
  const [lastN, setLastN] = useState(10);
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  useEffect(() => {
    if (authLoading || !user) return;
    let cancelled = false;
    setStatus({ kind: "loading" });
    apiFetch<AccuracyOut>(`/accuracy?season=${season}&last_n_rounds=${lastN}`)
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
  }, [authLoading, user, season, lastN]);

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to view model accuracy." />;

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

      {status.kind === "loading" && <p role="status">Loading accuracy data…</p>}

      {status.kind === "error" && (
        <div className="error-box" role="alert">
          {status.message}
        </div>
      )}

      {status.kind === "ok" && status.data.scoredMatches === 0 && (
        <p className="muted">No scored matches found for this season and range yet.</p>
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
                    <td>
                      Rd {r.round}
                    </td>
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
