import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError, getJobRun, getJobRuns, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import type { JobRunChange, JobRunDetail, JobRunListItem } from "../types";

type ListStatus =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; data: JobRunListItem[] };

type DetailStatus =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; data: JobRunDetail };

function fmtDate(iso: string): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function fmtPct(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return `${Math.round(n * 100)}%`;
}

function StatusPill({ status }: { status: string }) {
  const cls =
    status === "success"
      ? "job-status-pill job-status-pill--ok"
      : status === "failed"
        ? "job-status-pill job-status-pill--err"
        : "job-status-pill job-status-pill--warn";
  return <span className={cls}>{status}</span>;
}

function SignalBadge({ signal }: { signal: JobRunChange["triggerSignal"] }) {
  if (!signal) return null;
  const label =
    signal === "team_list"
      ? "team list"
      : signal === "weather"
        ? "weather"
        : "model retrained";
  return <span className="job-signal-badge">{label}</span>;
}

export function Jobs() {
  const { user, loading } = useAuth();
  const [status, setStatus] = useState<ListStatus>({ kind: "loading" });

  useEffect(() => {
    if (loading || !user) return;
    let cancelled = false;
    setStatus({ kind: "loading" });
    getJobRuns(50)
      .then((data) => {
        if (!cancelled) setStatus({ kind: "ok", data });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          setStatus({ kind: "error", message: "Sign in required" });
          return;
        }
        const msg =
          err instanceof ApiError
            ? `API ${err.status}: ${err.message}`
            : err instanceof Error
              ? err.message
              : String(err);
        setStatus({ kind: "error", message: msg });
      });
    return () => {
      cancelled = true;
    };
  }, [user, loading]);

  if (loading) return <p>Loading…</p>;
  if (!user) return <SignInRequired />;

  return (
    <div className="jobs-page">
      <h1>Precompute runs</h1>
      <p className="muted">
        The precompute job runs Tue 09:00 and Thu 06:00 AEST. Each row shows how predictions
        moved relative to the prior run — useful for spotting late team-list or weather
        changes that might shift your tips.
      </p>

      {status.kind === "loading" && <p>Loading runs…</p>}
      {status.kind === "error" && (
        <div className="error-box">Failed to load runs: {status.message}</div>
      )}
      {status.kind === "ok" && status.data.length === 0 && (
        <p className="muted">No runs recorded yet.</p>
      )}
      {status.kind === "ok" && status.data.length > 0 && (
        <ul className="job-list">
          {status.data.map((run) => (
            <li key={run.id} className="job-list-item">
              <Link to={`/jobs/${run.id}`} className="job-list-link">
                <div className="job-list-row">
                  <span className="job-list-time">{fmtDate(run.startedAt)}</span>
                  <StatusPill status={run.status} />
                  <span className="job-list-meta">
                    Round {run.round} · {run.matchesProcessed} matches
                  </span>
                </div>
                <div className="job-list-summary">
                  <span>
                    <strong>{run.summary.flips}</strong> flip
                    {run.summary.flips === 1 ? "" : "s"}
                  </span>
                  <span>
                    max Δ <strong>{Math.round(run.summary.maxAbsDelta * 100)}pt</strong>
                  </span>
                  <span>
                    avg Δ <strong>{Math.round(run.summary.meanAbsDelta * 100)}pt</strong>
                  </span>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function JobDetail() {
  const { runId } = useParams<{ runId: string }>();
  const { user, loading } = useAuth();
  const [status, setStatus] = useState<DetailStatus>({ kind: "loading" });

  useEffect(() => {
    if (loading || !user || !runId) return;
    let cancelled = false;
    setStatus({ kind: "loading" });
    getJobRun(runId)
      .then((data) => {
        if (!cancelled) setStatus({ kind: "ok", data });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          setStatus({ kind: "error", message: "Sign in required" });
          return;
        }
        const msg =
          err instanceof ApiError
            ? err.status === 404
              ? "Run not found."
              : `API ${err.status}: ${err.message}`
            : err instanceof Error
              ? err.message
              : String(err);
        setStatus({ kind: "error", message: msg });
      });
    return () => {
      cancelled = true;
    };
  }, [user, loading, runId]);

  if (loading) return <p>Loading…</p>;
  if (!user) return <SignInRequired />;

  return (
    <div className="jobs-page">
      <p>
        <Link to="/jobs" className="back-link">
          ← All runs
        </Link>
      </p>
      {status.kind === "loading" && <p>Loading run…</p>}
      {status.kind === "error" && (
        <div className="error-box">Failed to load run: {status.message}</div>
      )}
      {status.kind === "ok" && (
        <>
          <h1>Run on {fmtDate(status.data.startedAt)}</h1>
          <div className="job-detail-meta">
            <StatusPill status={status.data.status} />
            <span>Round {status.data.round}</span>
            <span>{status.data.matchesProcessed} matches</span>
            <span className="muted">model {status.data.modelVersion}</span>
          </div>
          <div className="job-detail-summary">
            <span>
              <strong>{status.data.summary.flips}</strong> flip
              {status.data.summary.flips === 1 ? "" : "s"}
            </span>
            <span>
              max Δ{" "}
              <strong>{Math.round(status.data.summary.maxAbsDelta * 100)}pt</strong>
            </span>
            <span>
              avg Δ{" "}
              <strong>{Math.round(status.data.summary.meanAbsDelta * 100)}pt</strong>
            </span>
          </div>

          {status.data.error && (
            <div className="error-box">Error: {status.data.error}</div>
          )}

          <h2>Per-match changes</h2>
          {status.data.changes.length === 0 ? (
            <p className="muted">No matches in this run.</p>
          ) : (
            <ul className="job-change-list">
              {status.data.changes.map((c) => (
                <li
                  key={c.matchId}
                  className={`job-change-card${c.flipped ? " job-change-card--flipped" : ""}`}
                >
                  <div className="job-change-teams">
                    <strong>{c.homeTeam}</strong> v <strong>{c.awayTeam}</strong>
                    {c.flipped && <span className="job-flip-badge">flipped</span>}
                    <SignalBadge signal={c.triggerSignal} />
                  </div>
                  <div className="job-change-probs">
                    <span>{fmtPct(c.prevHomeProb)}</span>
                    <span aria-hidden="true">→</span>
                    <span>
                      <strong>{fmtPct(c.newHomeProb)}</strong> home
                    </span>
                  </div>
                  <p className="job-change-summary">{c.summary}</p>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

export default Jobs;
