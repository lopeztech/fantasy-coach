import { useAuth } from "../auth";
import { RoundSelector } from "../components/RoundSelector";
import { SignInRequired } from "../components/SignInRequired";

export default function Home() {
  const { user, loading } = useAuth();

  if (loading) return <p>Loading…</p>;

  return (
    <section>
      <h1>Predictions</h1>
      <p>
        Pick a round to see predicted winners and home-win probabilities for every match.
      </p>
      {user ? (
        <RoundSelector />
      ) : (
        <SignInRequired message="Sign in to pick a round and view predictions." />
      )}
    </section>
  );
}
