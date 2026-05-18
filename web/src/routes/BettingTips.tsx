import { useEffect, useState } from "react";

import { ApiError, NotSignedInError, getBettingTips, getDashboard } from "../api";
import { useAuth } from "../auth";
import { ResponsibleGamblingBanner } from "../components/ResponsibleGamblingBanner";
import { SignInRequired } from "../components/SignInRequired";
import { TipCard } from "../components/TipCard";
import type { BettingTipsResponse, Tip } from "../types";

type Status =
  | { kind: "loading" }
  | { kind: "ok"; data: BettingTipsResponse }
  | { kind: "unavailable" }
  | { kind: "error"; message: string };

// Detect "season + round" we should fetch. The /me/dashboard endpoint owns the
// authoritative current-round logic; we hit that first, then fall back to a
// best-guess for unauthenticated edge cases (which shouldn't happen since the
// page is auth-gated, but keeps the type story tidy).
async function pickCurrentRound(
  fetchDashboard: () => Promise<{ season: number; currentRound: number | null }>,
): Promise<{ season: number; round: number } | null> {
  try {
    const d = await fetchDashboard();
    if (d.currentRound) return { season: d.season, round: d.currentRound };
  } catch {
    // fall through
  }
  return null;
}

function Section({ title, tips }: { title: string; tips: Tip[] }) {
  return (
    <section className="tips-section">
      <h2>{title}</h2>
      {tips.length === 0 ? (
        <p className="muted">
          Not enough confident picks this round — check back after the Thursday refresh.
        </p>
      ) : (
        <div className="tips-grid">
          {tips.map((t, i) => (
            <TipCard key={`${title}-${i}`} tip={t} index={i} />
          ))}
        </div>
      )}
    </section>
  );
}

const CURRENT_SEASON = new Date().getUTCFullYear();

export default function BettingTips() {
  const { user, loading: authLoading } = useAuth();
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  useEffect(() => {
    document.title = "Betting Tips — Fantasy Coach";
  }, []);

  useEffect(() => {
    if (authLoading) return;
    if (!user) return;

    let cancelled = false;
    setStatus({ kind: "loading" });

    (async () => {
      // /me/dashboard is the authoritative source for "what's the current round".
      const target = await pickCurrentRound(() =>
        getDashboard(CURRENT_SEASON).then((d) => ({
          season: d.season,
          currentRound: d.currentRound,
        })),
      );

      if (cancelled) return;
      if (!target) {
        setStatus({ kind: "unavailable" });
        return;
      }

      try {
        const data = await getBettingTips(target.season, target.round);
        if (!cancelled) setStatus({ kind: "ok", data });
      } catch (err) {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          setStatus({ kind: "error", message: "Sign in required." });
        } else if (err instanceof ApiError && err.status === 503) {
          setStatus({ kind: "unavailable" });
        } else if (err instanceof ApiError) {
          setStatus({
            kind: "error",
            message: `Couldn't load betting tips (HTTP ${err.status}).`,
          });
        } else {
          setStatus({ kind: "error", message: "Couldn't reach the API." });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [authLoading, user]);

  if (authLoading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to see this week's betting tips." />;

  return (
    <section className="betting-tips-page">
      <ResponsibleGamblingBanner />

      <header className="round-header">
        <h1>Betting Tips</h1>
      </header>

      {status.kind === "loading" && (
        <div className="tips-grid" aria-busy="true">
          {Array.from({ length: 3 }, (_, i) => (
            <div key={i} className="tip-card-skeleton" aria-hidden="true" />
          ))}
        </div>
      )}

      {status.kind === "unavailable" && (
        <div className="sign-in-required" role="status">
          <p>
            Tips for this week aren&apos;t ready yet. The model refreshes them every
            Tuesday morning and Thursday morning (Sydney time) — please check back
            shortly.
          </p>
        </div>
      )}

      {status.kind === "error" && (
        <div className="error-box" role="alert">
          {status.message}
        </div>
      )}

      {status.kind === "ok" && (
        <>
          <p className="tips-meta muted">
            Round {status.data.round}, {status.data.season} · Generated{" "}
            {new Date(status.data.generatedAt).toLocaleString()}
            {status.data.oddsSnapshotAt
              ? ` · Totals odds refreshed ${new Date(status.data.oddsSnapshotAt).toLocaleString()}`
              : ""}
          </p>
          <p className="tips-caveat muted">{status.data.caveat}</p>

          <Section title="Singles" tips={status.data.singles} />
          <Section title="Doubles" tips={status.data.doubles} />
          <Section title="Trebles" tips={status.data.trebles} />
        </>
      )}
    </section>
  );
}
