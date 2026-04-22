import {
  collection,
  doc,
  getDoc,
  getDocs,
  query,
  serverTimestamp,
  setDoc,
  Timestamp,
  where,
} from "firebase/firestore";

import { getFirebaseFirestore } from "./firebase";

export type TipChoice = "home" | "away";

export type TipDoc = {
  tip: TipChoice;
  kickoff: Timestamp;
  season: number;
  round: number;
  updatedAt: Timestamp;
};

/** Collection path: users/{uid}/tips/{matchId} */
function tipsCollection(uid: string) {
  const db = getFirebaseFirestore();
  if (!db) throw new Error("Firestore not configured");
  return collection(db, "users", uid, "tips");
}

export async function saveTip(
  uid: string,
  matchId: number,
  tip: TipChoice,
  kickoffIso: string,
  season: number,
  round: number,
): Promise<void> {
  const db = getFirebaseFirestore();
  if (!db) throw new Error("Firestore not configured");
  const ref = doc(db, "users", uid, "tips", String(matchId));
  await setDoc(ref, {
    tip,
    kickoff: Timestamp.fromDate(new Date(kickoffIso)),
    season,
    round,
    updatedAt: serverTimestamp(),
  } satisfies Omit<TipDoc, "updatedAt"> & { updatedAt: ReturnType<typeof serverTimestamp> });
}

export async function getTip(uid: string, matchId: number): Promise<TipChoice | null> {
  const db = getFirebaseFirestore();
  if (!db) return null;
  const ref = doc(db, "users", uid, "tips", String(matchId));
  const snap = await getDoc(ref);
  return snap.exists() ? (snap.data() as TipDoc).tip : null;
}

export async function getTipsByRound(
  uid: string,
  season: number,
  round: number,
): Promise<Map<number, TipChoice>> {
  const col = tipsCollection(uid);
  const q = query(col, where("season", "==", season), where("round", "==", round));
  const snap = await getDocs(q);
  const result = new Map<number, TipChoice>();
  snap.forEach((d) => {
    result.set(Number(d.id), (d.data() as TipDoc).tip);
  });
  return result;
}

export async function getAllTips(uid: string): Promise<
  Array<{ matchId: number; season: number; round: number; tip: TipChoice }>
> {
  const col = tipsCollection(uid);
  const snap = await getDocs(col);
  return snap.docs.map((d) => ({
    matchId: Number(d.id),
    ...(d.data() as Pick<TipDoc, "season" | "round" | "tip">),
  }));
}
