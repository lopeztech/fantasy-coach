/**
 * Firestore operations for tipping groups (#173).
 * Higher-level helpers that go beyond the REST API (e.g. listing user's groups).
 */

import {
  collection,
  doc,
  getDoc,
  getDocs,
} from "firebase/firestore";

import { getFirebaseFirestore } from "./firebase";
import { apiFetch } from "./api";

export type Group = {
  gid: string;
  name: string;
  inviteCode: string;
  ownerUid: string;
  memberCount: number;
};

export type LeaderboardEntry = {
  rank: number;
  uid: string;
  displayName: string;
  wins: number;
  losses: number;
  totalTips: number;
  accuracy: number;
  marginPoints: number;
  currentStreak: number;
  longestStreak: number;
};

export type LeaderboardOut = {
  season: number;
  groupId: string | null;
  entries: LeaderboardEntry[];
};

/** Create a new group. Returns the server response. */
export async function createGroup(name: string): Promise<Group> {
  return apiFetch<Group>("/groups", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

/** Join a group by invite code. */
export async function joinGroup(gid: string, inviteCode: string): Promise<Group> {
  return apiFetch<Group>(`/groups/${gid}/join`, {
    method: "POST",
    body: JSON.stringify({ inviteCode }),
  });
}

/** Leave a group. */
export async function leaveGroup(gid: string): Promise<void> {
  await apiFetch<void>(`/groups/${gid}/leave`, { method: "POST" });
}

/** Fetch leaderboard (global or group-filtered). */
export async function getLeaderboard(season: number, groupId?: string): Promise<LeaderboardOut> {
  const params = new URLSearchParams({ season: String(season) });
  if (groupId) params.set("group_id", groupId);
  return apiFetch<LeaderboardOut>(`/leaderboard?${params.toString()}`);
}

/** List groups the authenticated user belongs to (Firestore direct read). */
export async function getUserGroups(uid: string): Promise<Array<{ gid: string; name: string }>> {
  const db = getFirebaseFirestore();
  if (!db) return [];
  const col = collection(db, "users", uid, "groups");
  const snap = await getDocs(col);
  return snap.docs.map((d) => ({ gid: d.id, name: (d.data() as { name: string }).name }));
}

/** Get a single group's public info directly from Firestore. */
export async function getGroupInfo(gid: string): Promise<Group | null> {
  const db = getFirebaseFirestore();
  if (!db) return null;
  const ref = doc(db, "groups", gid);
  const snap = await getDoc(ref);
  if (!snap.exists()) return null;
  const d = snap.data() as {
    name: string;
    invite_code: string;
    owner_uid: string;
    member_count: number;
  };
  return {
    gid,
    name: d.name,
    inviteCode: d.invite_code,
    ownerUid: d.owner_uid,
    memberCount: d.member_count ?? 0,
  };
}
