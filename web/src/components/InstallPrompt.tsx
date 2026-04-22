import { useEffect, useState } from "react";

// BeforeInstallPromptEvent is not in the standard TypeScript lib yet.
interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  readonly userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

const VISITS_KEY = "fc-visits";
const DISMISSED_KEY = "fc-install-dismissed";
const SHOW_AFTER_VISITS = 3;

export function InstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    // Track visit count; only show the CTA after SHOW_AFTER_VISITS sessions.
    try {
      const prev = parseInt(localStorage.getItem(VISITS_KEY) ?? "0", 10);
      localStorage.setItem(VISITS_KEY, String(prev + 1));
      if (localStorage.getItem(DISMISSED_KEY)) setDismissed(true);
    } catch {
      // localStorage unavailable; skip prompt entirely
      setDismissed(true);
    }

    const handler = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
    };
    window.addEventListener("beforeinstallprompt", handler);
    return () => window.removeEventListener("beforeinstallprompt", handler);
  }, []);

  const visitCount = (() => {
    try {
      return parseInt(localStorage.getItem(VISITS_KEY) ?? "0", 10);
    } catch {
      return 0;
    }
  })();

  if (dismissed || !deferredPrompt || visitCount < SHOW_AFTER_VISITS) return null;

  async function handleInstall() {
    if (!deferredPrompt) return;
    await deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === "accepted" || outcome === "dismissed") {
      handleDismiss();
    }
  }

  function handleDismiss() {
    setDismissed(true);
    setDeferredPrompt(null);
    try {
      localStorage.setItem(DISMISSED_KEY, "1");
    } catch {
      // ignore
    }
  }

  return (
    <aside className="install-prompt" role="complementary" aria-label="Install app">
      <p className="install-prompt-text">
        Add Fantasy Coach to your home screen for offline access.
      </p>
      <div className="install-prompt-actions">
        <button type="button" onClick={() => void handleInstall()}>
          Install
        </button>
        <button type="button" className="install-prompt-dismiss" onClick={handleDismiss}>
          Not now
        </button>
      </div>
    </aside>
  );
}
