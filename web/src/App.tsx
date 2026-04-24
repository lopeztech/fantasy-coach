import { Link, Outlet } from "react-router-dom";

import { AuthProvider } from "./auth";
import { AuthButton } from "./components/AuthButton";
import { InstallPrompt } from "./components/InstallPrompt";
import { OfflineBanner } from "./components/OfflineBanner";
import { SearchBar } from "./components/SearchBar";
import { ThemeToggle } from "./components/ThemeToggle";

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
      </div>
    </AuthProvider>
  );
}
