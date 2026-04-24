import { useEffect, useState } from "react";

import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";
import { GroupCard } from "../components/GroupCard";
import { createGroup, joinGroup, leaveGroup, getUserGroups, getGroupInfo } from "../groups";
import type { Group } from "../groups";

export default function Groups() {
  const { user, loading } = useAuth();
  const [groups, setGroups] = useState<Group[]>([]);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Create group form
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  // Join group form
  const [joinGid, setJoinGid] = useState("");
  const [joinCode, setJoinCode] = useState("");
  const [joining, setJoining] = useState(false);

  async function loadGroups() {
    if (!user) return;
    setFetching(true);
    try {
      const refs = await getUserGroups(user.uid);
      const full = await Promise.all(refs.map((r) => getGroupInfo(r.gid)));
      setGroups(full.filter((g): g is Group => g !== null));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load groups");
    } finally {
      setFetching(false);
    }
  }

  useEffect(() => {
    loadGroups();
  }, [user]);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const g = await createGroup(newName.trim());
      setGroups((prev) => [...prev, g]);
      setNewName("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create group");
    } finally {
      setCreating(false);
    }
  }

  async function handleJoin(e: React.FormEvent) {
    e.preventDefault();
    if (!joinGid.trim() || !joinCode.trim()) return;
    setJoining(true);
    setError(null);
    try {
      const g = await joinGroup(joinGid.trim(), joinCode.trim().toUpperCase());
      if (!groups.some((x) => x.gid === g.gid)) {
        setGroups((prev) => [...prev, g]);
      }
      setJoinGid("");
      setJoinCode("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to join group");
    } finally {
      setJoining(false);
    }
  }

  async function handleLeave(gid: string) {
    if (!confirm("Leave this group?")) return;
    try {
      await leaveGroup(gid);
      setGroups((prev) => prev.filter((g) => g.gid !== gid));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to leave group");
    }
  }

  if (loading) return <p role="status">Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to manage groups." />;

  return (
    <section>
      <h1>Groups</h1>

      {error && <div className="error-box">{error}</div>}

      {/* Existing groups */}
      {fetching && <p role="status">Loading groups…</p>}

      {!fetching && groups.length > 0 && (
        <div className="groups-list">
          {groups.map((g) => (
            <GroupCard key={g.gid} group={g} onLeave={handleLeave} />
          ))}
        </div>
      )}

      {!fetching && groups.length === 0 && (
        <p className="muted">You're not in any groups yet. Create one or join with a code.</p>
      )}

      <div className="groups-forms">
        {/* Create */}
        <div className="groups-form-card">
          <h2>Create a group</h2>
          <form onSubmit={handleCreate} className="groups-form">
            <label>
              Group name
              <input
                type="text"
                value={newName}
                placeholder="e.g. Work footy tipping"
                maxLength={64}
                onChange={(e) => setNewName(e.target.value)}
                required
              />
            </label>
            <button type="submit" disabled={creating || !newName.trim()}>
              {creating ? "Creating…" : "Create"}
            </button>
          </form>
        </div>

        {/* Join */}
        <div className="groups-form-card">
          <h2>Join a group</h2>
          <form onSubmit={handleJoin} className="groups-form">
            <label>
              Group ID
              <input
                type="text"
                value={joinGid}
                placeholder="Group ID from the invite link"
                onChange={(e) => setJoinGid(e.target.value)}
                required
              />
            </label>
            <label>
              Invite code
              <input
                type="text"
                value={joinCode}
                placeholder="6-char code, e.g. AB12CD"
                maxLength={6}
                style={{ textTransform: "uppercase" }}
                onChange={(e) => setJoinCode(e.target.value)}
                required
              />
            </label>
            <button type="submit" disabled={joining || !joinGid.trim() || !joinCode.trim()}>
              {joining ? "Joining…" : "Join"}
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}
