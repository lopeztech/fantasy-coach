import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";

import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import { getGroupInfo, getLeaderboard } from "../groups";
import type { Group, LeaderboardEntry } from "../groups";

const CURRENT_SEASON = 2026;

export default function GroupDetail() {
  const { gid } = useParams<{ gid: string }>();
  const { user, loading } = useAuth();
  const [group, setGroup] = useState<Group | null>(null);
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [fetching, setFetching] = useState(false);

  useEffect(() => {
    if (!user || !gid) return;
    setFetching(true);
    setError(null);
    Promise.all([getGroupInfo(gid), getLeaderboard(CURRENT_SEASON, gid)])
      .then(([g, lb]) => {
        setGroup(g);
        setEntries(lb.entries);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setFetching(false));
  }, [user, gid]);

  if (loading) return <p role="status">Loading…</p>;
  if (!user) return <SignInRequired />;

  return (
    <section>
      <Link to="/groups" className="back-link">
        ← Groups
      </Link>

      {error && <div className="error-box">{error}</div>}

      {fetching && <p role="status">Loading group…</p>}

      {group && (
        <>
          <h1>{group.name}</h1>
          <p className="muted">
            {group.memberCount} member{group.memberCount !== 1 ? "s" : ""} · Invite code:{" "}
            <strong className="group-invite-code">{group.inviteCode}</strong>
          </p>
        </>
      )}

      {!fetching && entries.length > 0 && (
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
                <tr key={e.uid} className={e.uid === user.uid ? "lb-row--me" : undefined}>
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
      )}

      {!fetching && entries.length === 0 && (
        <p className="muted">No tips scored yet this season.</p>
      )}
    </section>
  );
}
