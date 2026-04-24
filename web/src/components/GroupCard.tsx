import { Link } from "react-router-dom";

import type { Group } from "../groups";

export function GroupCard({
  group,
  rank,
  onLeave,
}: {
  group: Group & { userRank?: number };
  rank?: number;
  onLeave: (gid: string) => void;
}) {
  return (
    <div className="group-card">
      <div className="group-card-header">
        <h3 className="group-card-name">{group.name}</h3>
        <span className="group-card-members">{group.memberCount} member{group.memberCount !== 1 ? "s" : ""}</span>
      </div>
      <p className="group-card-code muted">
        Invite code: <strong>{group.inviteCode}</strong>
      </p>
      {rank != null && (
        <p className="group-card-rank muted">You are #{rank} in this group</p>
      )}
      <div className="group-card-actions">
        <Link to={`/groups/${group.gid}`} className="group-card-view-btn">
          View leaderboard →
        </Link>
        <button
          type="button"
          className="group-card-leave-btn"
          onClick={() => onLeave(group.gid)}
        >
          Leave
        </button>
      </div>
    </div>
  );
}
