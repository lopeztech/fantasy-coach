import { useAuth } from "../auth";

export function AuthButton() {
  const { user, loading, configured, signIn, signOut } = useAuth();

  if (!configured) {
    return <span className="auth-status muted">Auth not configured</span>;
  }
  if (loading) {
    return <span className="auth-status muted">…</span>;
  }
  if (user) {
    return (
      <div className="auth-status">
        <span className="muted">{user.email ?? user.displayName ?? "Signed in"}</span>
        <button type="button" onClick={() => void signOut()}>
          Sign out
        </button>
      </div>
    );
  }
  return (
    <button type="button" onClick={() => void signIn()}>
      Sign in with Google
    </button>
  );
}
