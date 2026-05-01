import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";

import { AuthProvider, useAuth } from "./auth";
import { AuthButton } from "./components/AuthButton";
import { InstallPrompt } from "./components/InstallPrompt";
import { OfflineBanner } from "./components/OfflineBanner";
import { SearchBar } from "./components/SearchBar";
import { ThemeToggle } from "./components/ThemeToggle";
import { onForegroundMessage, acquireAndRegisterToken, getPermissionState, isRegistered } from "./notifications";

type Toast = { title: string; body: string; href?: string };

const NAV_ITEMS: { to: string; label: string }[] = [
  { to: "/predictions", label: "Predictions" },
  { to: "/ladder", label: "Ladder" },
  { to: "/leaderboard", label: "Leaderboard" },
  { to: "/groups", label: "Groups" },
  { to: "/scoreboard", label: "Scoreboard" },
  { to: "/accuracy", label: "Accuracy" },
  { to: "/jobs", label: "Job runs" },
];

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
  const [navOpen, setNavOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  return (
    <AuthProvider>
      <div className="app">
        <aside
          className={`app-sidebar${navOpen ? " is-open" : ""}`}
          aria-label="Primary navigation"
        >
          <Link to="/" className="brand" onClick={() => setNavOpen(false)}>
            Fantasy Coach
          </Link>
          <nav className="sidebar-nav">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `nav-link${isActive ? " is-active" : ""}`}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </aside>
        <button
          type="button"
          className={`sidebar-backdrop${navOpen ? " is-open" : ""}`}
          aria-label="Close menu"
          tabIndex={navOpen ? 0 : -1}
          onClick={() => setNavOpen(false)}
        />
        <div className="app-body">
          <OfflineBanner />
          <header className="app-header">
            <button
              type="button"
              className="nav-toggle"
              aria-label={navOpen ? "Close menu" : "Open menu"}
              aria-expanded={navOpen}
              onClick={() => setNavOpen((o) => !o)}
            >
              ☰
            </button>
            <div className="header-controls">
              <SearchBar />
              <ThemeToggle />
              <AuthButton />
            </div>
          </header>
          <main className="app-main">
            <Outlet />
          </main>
        </div>
        <InstallPrompt />
        <NotificationManager />
      </div>
    </AuthProvider>
  );
}
