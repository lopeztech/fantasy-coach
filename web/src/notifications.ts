/**
 * FCM token lifecycle for push notifications (#172).
 *
 * Flow:
 * 1. Call requestNotificationPermission() after a meaningful user action.
 * 2. On grant, call acquireAndRegisterToken(uid) to get the FCM token and
 *    persist it to the backend.
 * 3. Background messages are handled by firebase-messaging-sw.js.
 * 4. Foreground messages fire the onMessage callback registered in App.tsx.
 */

import { getToken, onMessage, type Messaging } from "firebase/messaging";
import { getFirebaseMessaging, VAPID_KEY } from "./firebase";
import { apiFetch } from "./api";

const DISMISSED_KEY = "fc:notif_dismissed";
const REGISTERED_KEY = "fc:notif_registered";

export type NotificationPermissionState = "default" | "granted" | "denied" | "unsupported";

export function getPermissionState(): NotificationPermissionState {
  if (typeof Notification === "undefined") return "unsupported";
  return Notification.permission as NotificationPermissionState;
}

export function isDismissed(): boolean {
  try {
    return localStorage.getItem(DISMISSED_KEY) === "1";
  } catch {
    return false;
  }
}

export function setDismissed(): void {
  try {
    localStorage.setItem(DISMISSED_KEY, "1");
  } catch {
    // storage full
  }
}

export function isRegistered(): boolean {
  try {
    return localStorage.getItem(REGISTERED_KEY) === "1";
  } catch {
    return false;
  }
}

function setRegistered(): void {
  try {
    localStorage.setItem(REGISTERED_KEY, "1");
  } catch {
    // storage full
  }
}

/** Request browser notification permission. Returns final permission state. */
export async function requestNotificationPermission(): Promise<NotificationPermissionState> {
  if (typeof Notification === "undefined") return "unsupported";
  if (Notification.permission !== "default") return Notification.permission as NotificationPermissionState;
  const result = await Notification.requestPermission();
  return result as NotificationPermissionState;
}

/**
 * Get the FCM token and register it with the backend.
 * Safe to call on every page load after permission is granted — the backend is
 * idempotent on token.
 */
export async function acquireAndRegisterToken(uid: string): Promise<void> {
  if (!VAPID_KEY) return; // no VAPID key configured
  const m = getFirebaseMessaging();
  if (!m) return;

  let token: string;
  try {
    token = await getToken(m, { vapidKey: VAPID_KEY });
  } catch {
    return; // permission denied or SW not ready
  }

  if (!token) return;

  try {
    await apiFetch("/notifications/subscribe", {
      method: "POST",
      body: JSON.stringify({ token, platform: "web", uid }),
    });
    setRegistered();
  } catch {
    // Non-fatal — retry on next load
  }
}

/** Register a foreground message handler. Returns an unsubscribe function. */
export function onForegroundMessage(
  handler: (payload: { notification?: { title?: string; body?: string }; data?: Record<string, string> }) => void,
): (() => void) | null {
  const m: Messaging | null = getFirebaseMessaging();
  if (!m) return null;
  // `onMessage` returns an unsubscribe function
  return onMessage(m, handler);
}
