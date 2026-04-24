import { useState } from "react";

import { useAuth } from "../auth";
import {
  getPermissionState,
  isDismissed,
  setDismissed,
  requestNotificationPermission,
  acquireAndRegisterToken,
  isRegistered,
} from "../notifications";

/**
 * Soft opt-in banner for push notifications.
 *
 * Shown once after the user has made at least one tip (indicated by
 * `tipsCount > 0` prop). Persists dismissal in localStorage so we never
 * show it twice.
 *
 * Per retention research: deferred ask (after first meaningful action) yields
 * 3× the opt-in rate compared to showing it on first load.
 */
export function NotificationPrompt({ tipsCount }: { tipsCount: number }) {
  const { user } = useAuth();
  const [state, setState] = useState<"idle" | "requesting" | "done" | "denied">("idle");

  // Only show if: user is signed in, has made at least one tip, hasn't
  // dismissed or already registered, and permission is not yet granted/denied.
  if (!user) return null;
  if (tipsCount === 0) return null;
  if (isDismissed()) return null;
  if (isRegistered()) return null;
  const perm = getPermissionState();
  if (perm === "unsupported" || perm === "denied") return null;
  if (perm === "granted" && state === "idle") return null;
  if (state === "done") return null;

  async function handleEnable() {
    setState("requesting");
    const result = await requestNotificationPermission();
    if (result === "granted") {
      if (user) await acquireAndRegisterToken(user.uid);
      setState("done");
    } else {
      setState("denied");
    }
  }

  function handleDismiss() {
    setDismissed();
    setState("done");
  }

  if (state === "requesting") {
    return (
      <div className="notif-prompt">
        <span className="notif-prompt-text">Requesting permission…</span>
      </div>
    );
  }

  if (state === "denied") {
    return (
      <div className="notif-prompt notif-prompt--error">
        <span className="notif-prompt-text">
          Notifications blocked. Enable them in browser settings to get round-published alerts.
        </span>
        <button type="button" className="notif-prompt-dismiss" onClick={handleDismiss}>
          ✕
        </button>
      </div>
    );
  }

  return (
    <div className="notif-prompt" role="banner">
      <span className="notif-prompt-text">
        Get notified when this week's predictions are ready
      </span>
      <div className="notif-prompt-actions">
        <button type="button" onClick={handleEnable}>
          Enable
        </button>
        <button
          type="button"
          className="notif-prompt-dismiss"
          onClick={handleDismiss}
          aria-label="Dismiss notification prompt"
        >
          Not now
        </button>
      </div>
    </div>
  );
}
