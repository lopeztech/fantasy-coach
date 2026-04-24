import { useEffect, useState } from "react";
import { Link, Outlet } from "react-router-dom";

import { AuthProvider, useAuth } from "./auth";
import { AuthButton } from "./components/AuthButton";
import { InstallPrompt } from "./components/InstallPrompt";
import { OfflineBanner } from "./components/OfflineBanner";
import { SearchBar } from "./components/SearchBar";
import { ThemeToggle } from "./components/ThemeToggle";
import { onForegroundMessage, acquireAndRegisterToken, getPermissionState, isRegistered } from "./notifications";

type Toast = { title: string; body: string; href?: string };

function ForegroundToast({ toast, onClose }: { toast: Toast; onClose: () => void }) {
  useEffect(() => {
    const t = setTimeout(onClose, 6000);
    return () => clearTimeout(t);
  }, [onClose]);

  return (
    <div className="fcm-toast" role="alert">
      <div className="fcm-toast-text">
        <strong>{toast.title}</strong>
        {toast.body && <span>{toast.body}</span>}
      </div>
      <button type="button" className="fcm-toast-close" onClick={onClose} aria-label="Dismiss">
        ✕
      </button>
    </div>
  );
}

function NotificationManager() {
  const { user } = useAuth();
  const [toast, setToast] = useState<Toast | null>(null);

  // Re-register token on sign-in if permission already granted.
  useEffect(() => {
    if (!user) return;
    if (getPermissionState() === "granted" && !isRegistered()) {
      acquireAndRegisterToken(user.uid).catch(() => {});
    }
  }, [user]);

  // Listen for foreground messages.
  useEffect(() => {
    const unsub = onForegroundMessage((payload) => {
      const { notification, data } = payload;
      setToast({
        title: notification?.title ?? "Fantasy Coach",
        body: notification?.body ?? "",
        href: data?.action_url,
      });
    });
    return () => { if (unsub) unsub(); };
  }, []);

  if (!toast) return null;
  return <ForegroundToast toast={toast} onClose={() => setToast(null)} />;
}

export default function App() {
  return (
    <AuthProvider>
      <div className="app">
        <OfflineBanner />
        <header className="app-header">
          <Link to="/" className="brand">
            Fantasy Coach
          </Link>
          <div className="header-controls">
            <SearchBar />
            <Link to="/scoreboard" className="nav-link">
              Scoreboard
            </Link>
            <Link to="/accuracy" className="nav-link">
              Accuracy
            </Link>
            <ThemeToggle />
            <AuthButton />
          </div>
        </header>
        <main className="app-main">
          <Outlet />
        </main>
        <InstallPrompt />
        <NotificationManager />
      </div>
    </AuthProvider>
  );
}
