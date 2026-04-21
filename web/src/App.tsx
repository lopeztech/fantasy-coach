import { Link, Outlet } from "react-router-dom";

import { AuthProvider } from "./auth";
import { AuthButton } from "./components/AuthButton";

export default function App() {
  return (
    <AuthProvider>
      <div className="app">
        <header className="app-header">
          <Link to="/" className="brand">
            Fantasy Coach
          </Link>
          <AuthButton />
        </header>
        <main className="app-main">
          <Outlet />
        </main>
      </div>
    </AuthProvider>
  );
}
