import { useParams } from "react-router-dom";

import { useAuth } from "../auth";
import { SignInRequired } from "../components/SignInRequired";

export default function Round() {
  const { season, round } = useParams();
  const { user, loading } = useAuth();

  if (loading) return <p>Loading…</p>;
  if (!user) return <SignInRequired message="Sign in to see predictions for this round." />;

  return (
    <section>
      <h1>
        Round {round}, {season}
      </h1>
      <p>Predictions UI lands in #21.</p>
    </section>
  );
}
