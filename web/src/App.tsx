import { Link, Outlet } from "react-router-dom";

import { AuthProvider } from "./auth";
import { AuthButton } from "./components/AuthButton";
import { InstallPrompt } from "./components/InstallPrompt";
import { OfflineBanner } from "./components/OfflineBanner";

export default function App() {
  return (
    <AuthProvider>
      <div className="app">
        <OfflineBanner />
        <header className="app-header">
          <Link to="/" className="brand">
            Fantasy Coach
          </Link>
          <AuthButton />
        </header>
        <main className="app-main">
          <Outlet />
        </main>
        <InstallPrompt />
      </div>
    </AuthProvider>
  );
}
