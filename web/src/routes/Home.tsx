import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { getDashboard, ApiError, NotSignedInError } from "../api";
import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import type { DashboardOut } from "../types";

const CURRENT_SEASON = 2026;
const FAV_TEAM_KEY = "fc:fav_team_id";

function getFavouriteTeamId(): number | null {
  try {
    const raw = localStorage.getItem(FAV_TEAM_KEY);
    if (raw === null) return null;
    const n = parseInt(raw, 10);
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
}

function setFavouriteTeamId(id: number | null): void {
  try {
    if (id === null) {
      localStorage.removeItem(FAV_TEAM_KEY);
    } else {
      localStorage.setItem(FAV_TEAM_KEY, String(id));
    }
  } catch {
    // storage full — ignore
  }
}

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

// ---------------------------------------------------------------------------
// Team picker — minimal form for entering a team ID
// ---------------------------------------------------------------------------

function TeamPicker({ onPick }: { onPick: (id: number | null) => void }) {
  const [input, setInput] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const n = parseInt(input, 10);
    if (Number.isFinite(n) && n > 0) {
      onPick(n);
    }
  }

  return (
    <form className="dashboard-team-picker-form" onSubmit={handleSubmit}>
      <label>
        Team ID
        <input
          type="number"
          value={input}
          min={1}
          placeholder="e.g. 7"
          onChange={(e) => setInput(e.target.value)}
        />
      </label>
      <button type="submit" disabled={input === ""}>
        Set favourite
      </button>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Dashboard cards
// ---------------------------------------------------------------------------

function YourTeamCard({
  dashboard,
  onTeamChange,
}: {
  dashboard: DashboardOut;
  onTeamChange: (id: number | null) => void;
}) {
  const fixture = dashboard.nextFixture;

  return (
    <div className="dashboard-card">
      <h2 className="dashboard-card-title">Your Team</h2>
      {dashboard.favouriteTeamId == null ? (
        <div className="dashboard-team-picker">
          <p className="muted">No favourite team set.</p>
          <TeamPicker onPick={onTeamChange} />
        </div>
      ) : fixture != null ? (
        <div>
          <p className="muted dashboard-card-subtitle">Next fixture</p>
          <p className="dashboard-fixture-line">
            <strong>Round {fixture.round}</strong>
            {" — "}
            {fixture.isHome ? "Home vs " : "Away @ "}
            <Link to={`/team/${fixture.opponentId}`}>{fixture.opponent}</Link>
          </p>
          <p className="muted">
            <time dateTime={fixture.kickoff}>{formatKickoff(fixture.kickoff)}</time>
          </p>
          {fixture.predWinner != null && (
            <p className="dashboard-prediction muted">
              Model pick:{" "}
              <strong>
                {fixture.predWinner === "home"
                  ? fixture.isHome
                    ? "your team"
                    : fixture.opponent
                  : fixture.isHome
                    ? fixture.opponent
                    : "your team"}
              </strong>
              {fixture.predProb != null && (
                <span>
                  {" "}
                  (
                  {Math.round(
                    (fixture.predWinner === "home"
                      ? fixture.predProb
                      : 1 - fixture.predProb) * 100,
                  )}
                  %)
                </span>
              )}
            </p>
          )}
          <button
            type="button"
            className="dashboard-change-team"
            onClick={() => onTeamChange(null)}
          >
            Change team
          </button>
        </div>
      ) : (
        <div>
          <p className="muted">No upcoming fixtures for your team this season.</p>
          <button
            type="button"
            className="dashboard-change-team"
            onClick={() => onTeamChange(null)}
          >
            Change team
          </button>
        </div>
      )}
    </div>
  );
}

function TipRoundCard({ dashboard }: { dashboard: DashboardOut }) {
  const round = dashboard.currentRound;
  const matchIds = dashboard.untippedMatchIds;

  if (round == null) {
    return (
      <div className="dashboard-card">
        <h2 className="dashboard-card-title">Tip Round</h2>
        <p className="muted">No active round found.</p>
      </div>
    );
  }

  if (matchIds.length === 0) {
    return (
      <div className="dashboard-card">
        <h2 className="dashboard-card-title">Round {round}</h2>
        <p className="dashboard-round-complete">Round {round} complete ✓</p>
      </div>
    );
  }

  return (
    <div className="dashboard-card">
      <h2 className="dashboard-card-title">Tip Round {round}</h2>
      <p className="muted dashboard-card-subtitle">
        {matchIds.length} match{matchIds.length !== 1 ? "es" : ""} this round
      </p>
      <Link
        to={`/round/${dashboard.season}/${round}`}
        className="dashboard-view-round-link"
      >
        View predictions →
      </Link>
    </div>
  );
}

function AccuracyCard({ dashboard }: { dashboard: DashboardOut }) {
  const { seasonAccuracy, totalTips, correctTips } = dashboard;

  return (
    <div className="dashboard-card">
      <h2 className="dashboard-card-title">Your Accuracy</h2>
      {totalTips === 0 ? (
        <p className="muted">No tips yet this season.</p>
      ) : (
        <div>
          <p className="dashboard-accuracy-value">
            {seasonAccuracy != null
              ? `${Math.round(seasonAccuracy * 100)}%`
              : "—"}
          </p>
          <p className="muted">
            {correctTips} correct from {totalTips} tip{totalTips !== 1 ? "s" : ""}
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Signed-in personalised dashboard
// ---------------------------------------------------------------------------

type DashStatus =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ok"; data: DashboardOut };

function PersonalisedDashboard() {
  const { user } = useAuth();
  const [favTeamId, setFavTeamId] = useState<number | null>(() =>
    getFavouriteTeamId(),
  );
  const [status, setStatus] = useState<DashStatus>({ kind: "loading" });

  useEffect(() => {
    if (!user) return;
    let cancelled = false;

    setStatus({ kind: "loading" });
    getDashboard(CURRENT_SEASON, favTeamId)
      .then((data) => {
        if (!cancelled) setStatus({ kind: "ok", data });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          setStatus({ kind: "error", message: "Sign in required." });
        } else if (err instanceof ApiError) {
          setStatus({
            kind: "error",
            message: `Could not load dashboard (${err.status}).`,
          });
        } else {
          setStatus({ kind: "error", message: "Could not load dashboard." });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [user, favTeamId]);

  function handleTeamChange(id: number | null) {
    setFavouriteTeamId(id);
    setFavTeamId(id);
  }

  if (status.kind === "loading") {
    return <p role="status">Loading dashboard…</p>;
  }

  if (status.kind === "error") {
    return (
      <div className="error-box" role="alert">
        {status.message}
      </div>
    );
  }

  const { data } = status;

  return (
    <section className="dashboard">
      <h1>Dashboard</h1>
      <div className="dashboard-grid">
        <YourTeamCard dashboard={data} onTeamChange={handleTeamChange} />
        <TipRoundCard dashboard={data} />
        <AccuracyCard dashboard={data} />
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Landing page for signed-out visitors
// ---------------------------------------------------------------------------

function LandingPage() {
  return (
    <section>
      <h1>Fantasy Coach</h1>
      <p>
        NRL match predictions powered by machine learning. Pick your round to see
        predicted winners and home-win probabilities for every match.
      </p>
      <div className="landing-features">
        <div className="landing-feature">
          <strong>XGBoost predictions</strong>
          <p>Machine-learning model trained on historical NRL data.</p>
        </div>
        <div className="landing-feature">
          <strong>Feature insights</strong>
          <p>See which factors drove each prediction.</p>
        </div>
        <div className="landing-feature">
          <strong>Accuracy tracking</strong>
          <p>Walk-forward model accuracy updated each round.</p>
        </div>
      </div>
      <SignInRequired message="Sign in to view predictions and track your tips." />
    </section>
  );
}

// ---------------------------------------------------------------------------
// Root Home route
// ---------------------------------------------------------------------------

export default function Home() {
  const { user, loading } = useAuth();

  if (loading) return <p role="status">Loading…</p>;

  if (!user) return <LandingPage />;

  return <PersonalisedDashboard />;
}
