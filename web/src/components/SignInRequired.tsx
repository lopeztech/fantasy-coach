import { useAuth } from "../auth";

export function SignInRequired({ message }: { message?: string }) {
  const { signIn, configured } = useAuth();
  if (!configured) {
    return (
      <div className="sign-in-required">
        <p>
          The app is not wired to a Firebase project. Set{" "}
          <code>VITE_FIREBASE_*</code> environment variables and reload.
        </p>
      </div>
    );
  }
  return (
    <div className="sign-in-required">
      <p>{message ?? "Sign in to see predictions."}</p>
      <button type="button" onClick={() => void signIn()}>
        Sign in with Google
      </button>
    </div>
  );
}
