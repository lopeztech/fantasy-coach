import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import { getDashboard } from "../api";
import type { DashboardOut } from "../types";

// ── Local storage helpers ────────────────────────────────────────────────────

const FAV_TEAM_KEY = "fc:fav_team_id";

function getFavouriteTeamId(): number | null {
  const raw = localStorage.getItem(FAV_TEAM_KEY);
  if (raw === null) return null;
  const n = parseInt(raw, 10);
  return isNaN(n) ? null : n;
}

function setFavouriteTeamId(id: number | null): void {
  if (id === null) {
    localStorage.removeItem(FAV_TEAM_KEY);
  } else {
    localStorage.setItem(FAV_TEAM_KEY, String(id));
  }
}

// ── Current NRL season ───────────────────────────────────────────────────────
// Update each year or derive dynamically if needed.
const CURRENT_SEASON = new Date().getFullYear();

// ── Sub-components ───────────────────────────────────────────────────────────

function TeamPicker({ onPick }: { onPick: (id: number) => void }) {
  const [value, setValue] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const n = parseInt(value, 10);
    if (!isNaN(n) && n > 0) {
      onPick(n);
    }
  }

  return (
    <form className="dashboard-team-picker-form" onSubmit={handleSubmit}>
      <label htmlFor="team-id-input">Enter your team ID</label>
      <input
        id="team-id-input"
        type="number"
        min={1}
        placeholder="e.g. 500019"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button type="submit">Save</button>
    </form>
  );
}

function YourTeamCard({
  data,
  onChangeFav,
  onPickTeam,
}: {
  data: DashboardOut;
  onChangeFav: () => void;
  onPickTeam: (id: number) => void;
}) {
  const { nextFixture, favouriteTeamId } = data;

  if (favouriteTeamId === null) {
    return (
      <div className="dashboard-card">
        <h2 className="dashboard-card-title">Your Team</h2>
        <p className="dashboard-card-subtitle">Pick a team to track their next fixture.</p>
        <TeamPicker onPick={onPickTeam} />
      </div>
    );
  }

  return (
    <div className="dashboard-card">
      <h2 className="dashboard-card-title">Your Team</h2>
      {nextFixture ? (
        <div>
          <p className="dashboard-card-subtitle">
            Round {nextFixture.round} — {nextFixture.isHome ? "vs" : "@"}{" "}
            <strong>{nextFixture.opponent}</strong>
          </p>
          <p>
            {new Date(nextFixture.kickoff).toLocaleDateString("en-AU", {
              weekday: "short",
              day: "numeric",
              month: "short",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </p>
          {nextFixture.predWinner !== null && nextFixture.predProb !== null && (
            <p>
              Tip:{" "}
              <strong>
                {nextFixture.predWinner === "home"
                  ? nextFixture.isHome
                    ? "Win"
                    : "Loss"
                  : nextFixture.isHome
                    ? "Loss"
                    : "Win"}
              </strong>{" "}
              ({Math.round((nextFixture.isHome ? nextFixture.predProb : 1 - nextFixture.predProb) * 100)}%)
            </p>
          )}
          <Link
            className="dashboard-view-round-link"
            to={`/round?season=${nextFixture.season}&round=${nextFixture.round}`}
          >
            View round {nextFixture.round} predictions
          </Link>
        </div>
      ) : (
        <p className="dashboard-card-subtitle">No upcoming fixtures found.</p>
      )}
      <button className="dashboard-change-team" onClick={onChangeFav}>
        Change team
      </button>
    </div>
  );
}

function TipRoundCard({ data }: { data: DashboardOut }) {
  const { currentRound, untippedMatchIds, season } = data;

  if (currentRound === null) {
    return (
      <div className="dashboard-card">
        <h2 className="dashboard-card-title">Tip Round</h2>
        <p className="dashboard-card-subtitle">Season complete.</p>
      </div>
    );
  }

  const hasUntipped = untippedMatchIds.length > 0;

  return (
    <div className="dashboard-card">
      <h2 className="dashboard-card-title">Tip Round</h2>
      {hasUntipped ? (
        <div>
          <p className="dashboard-card-subtitle">
            Round {currentRound} — {untippedMatchIds.length} match
            {untippedMatchIds.length !== 1 ? "es" : ""} to view
          </p>
          <Link
            className="dashboard-view-round-link"
            to={`/round?season=${season}&round=${currentRound}`}
          >
            View round {currentRound}
          </Link>
        </div>
      ) : (
        <p className="dashboard-round-complete">Round {currentRound} complete</p>
      )}
    </div>
  );
}

function AccuracyCard({ data }: { data: DashboardOut }) {
  const { seasonAccuracy, totalTips, correctTips } = data;

  return (
    <div className="dashboard-card">
      <h2 className="dashboard-card-title">Your Accuracy</h2>
      {totalTips === 0 ? (
        <p className="dashboard-card-subtitle">No tips yet this season.</p>
      ) : (
        <div>
          <p className="dashboard-accuracy-value">
            {seasonAccuracy !== null ? `${Math.round(seasonAccuracy * 100)}%` : "—"}
          </p>
          <p className="dashboard-card-subtitle">
            {correctTips} correct from {totalTips} tips
          </p>
        </div>
      )}
    </div>
  );
}

// ── Personalised dashboard ───────────────────────────────────────────────────

function PersonalisedDashboard() {
  const [favTeamId, setFavTeamIdState] = useState<number | null>(getFavouriteTeamId);
  const [data, setData] = useState<DashboardOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  function reload() {
    setLoading(true);
    setError(null);
    getDashboard(CURRENT_SEASON, favTeamId)
      .then((d: DashboardOut) => {
        setData(d);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load dashboard.");
        setLoading(false);
      });
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(reload, [favTeamId]);

  function handleChangeFav() {
    setFavouriteTeamId(null);
    setFavTeamIdState(null);
  }

  function handlePickTeam(id: number) {
    setFavouriteTeamId(id);
    setFavTeamIdState(id);
  }

  if (loading) return <p role="status">Loading dashboard…</p>;
  if (error) return <p role="alert" style={{ color: "var(--color-error, red)" }}>{error}</p>;
  if (!data) return null;

  // Merge locally chosen fav into dashboard data so YourTeamCard sees it
  // even before the cache refreshes with the server-side value.
  const merged: DashboardOut = { ...data, favouriteTeamId: favTeamId };

  return (
    <section>
      <h1>Dashboard</h1>
      <div className="dashboard-grid">
        <YourTeamCard
          data={merged}
          onChangeFav={handleChangeFav}
          onPickTeam={handlePickTeam}
        />
        <TipRoundCard data={data} />
        <AccuracyCard data={data} />
      </div>
    </section>
  );
}

// ── Landing page (signed-out) ────────────────────────────────────────────────

function LandingPage() {
  return (
    <section>
      <h1>Fantasy Coach</h1>
      <p>NRL match predictions powered by machine learning.</p>
      <ul className="landing-features">
        <li className="landing-feature">Round-by-round win predictions</li>
        <li className="landing-feature">Per-team form and Elo progression</li>
        <li className="landing-feature">Model accuracy tracking</li>
        <li className="landing-feature">Key absence impact analysis</li>
      </ul>
      <SignInRequired message="Sign in to view predictions and your personalised dashboard." />
    </section>
  );
}

// ── Route component ──────────────────────────────────────────────────────────

export default function Home() {
  const { user, loading } = useAuth();

  if (loading) return <p role="status">Loading…</p>;

  return user ? <PersonalisedDashboard /> : <LandingPage />;
}
