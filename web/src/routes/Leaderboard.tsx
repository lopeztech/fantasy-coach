import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import { getLeaderboard, getUserGroups } from "../groups";
import type { LeaderboardEntry, LeaderboardOut } from "../groups";

const CURRENT_SEASON = 2026;

type View = "all" | "group";

export default function Leaderboard() {
  const { user, loading } = useAuth();
  const [searchParams] = useSearchParams();

  const [view, setView] = useState<View>("all");
  const [groupId, setGroupId] = useState<string | null>(searchParams.get("group_id"));
  const [data, setData] = useState<LeaderboardOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fetching, setFetching] = useState(false);
  const [userGroups, setUserGroups] = useState<Array<{ gid: string; name: string }>>([]);

  // Load user's groups for the group filter dropdown.
  useEffect(() => {
    if (!user) return;
    getUserGroups(user.uid)
      .then(setUserGroups)
      .catch(() => {});
  }, [user]);

  // Load leaderboard whenever view/groupId changes.
  useEffect(() => {
    if (!user) return;
    setFetching(true);
    setError(null);
    const gid = view === "group" ? groupId ?? undefined : undefined;
    getLeaderboard(CURRENT_SEASON, gid)
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Failed to load leaderboard"))
      .finally(() => setFetching(false));
  }, [user, view, groupId]);

  if (loading) return <p role="status">Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to see the leaderboard." />;

  return (
    <section>
      <h1>Leaderboard</h1>

      {/* View toggle */}
      <div className="lb-controls">
        <div className="lb-toggle">
          <button
            type="button"
            className={`lb-toggle-btn${view === "all" ? " active" : ""}`}
            onClick={() => setView("all")}
          >
            All users
          </button>
          <button
            type="button"
            className={`lb-toggle-btn${view === "group" ? " active" : ""}`}
            onClick={() => setView("group")}
            disabled={userGroups.length === 0}
            title={userGroups.length === 0 ? "Join a group first" : undefined}
          >
            My groups
          </button>
        </div>

        {view === "group" && userGroups.length > 0 && (
          <select
            className="lb-group-select"
            value={groupId ?? ""}
            onChange={(e) => setGroupId(e.target.value || null)}
          >
            <option value="">— Select group —</option>
            {userGroups.map((g) => (
              <option key={g.gid} value={g.gid}>
                {g.name}
              </option>
            ))}
          </select>
        )}

        <Link to="/groups" className="lb-groups-link">
          Manage groups →
        </Link>
      </div>

      {error && <div className="error-box">{error}</div>}

      {fetching && <p role="status">Loading leaderboard…</p>}

      {!fetching && data && (
        <LeaderboardTable entries={data.entries} currentUid={user.uid} />
      )}

      {!fetching && data && data.entries.length === 0 && (
        <p className="muted">No tippers found yet. Be the first!</p>
      )}
    </section>
  );
}

function LeaderboardTable({
  entries,
  currentUid,
}: {
  entries: LeaderboardEntry[];
  currentUid: string;
}) {
  return (
    <div className="lb-table-wrapper">
      <table className="lb-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Tipper</th>
            <th className="lb-col-num">Wins</th>
            <th className="lb-col-num">Tips</th>
            <th className="lb-col-num">Accuracy</th>
            <th className="lb-col-num lb-col-hide-sm">Streak</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => (
            <tr key={e.uid} className={e.uid === currentUid ? "lb-row--me" : undefined}>
              <td className="lb-col-rank">{e.rank}</td>
              <td>{e.displayName || `Tipster-${e.uid.slice(-4)}`}</td>
              <td className="lb-col-num">{e.wins}</td>
              <td className="lb-col-num">{e.totalTips}</td>
              <td className="lb-col-num">
                {e.totalTips > 0 ? `${Math.round(e.accuracy * 100)}%` : "—"}
              </td>
              <td className="lb-col-num lb-col-hide-sm">{e.currentStreak}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
