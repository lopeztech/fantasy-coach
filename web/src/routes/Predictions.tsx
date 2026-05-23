import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { getDashboard, NotSignedInError } from "../api";
import { useAuth } from "../auth";

const CURRENT_SEASON = 2026;
const FALLBACK_ROUND = 1;

/**
 * Index route for /predictions — there's no single "predictions" page, so this
 * resolves the active round and redirects to /round/{season}/{round}.
 *
 * - Signed-in users: hits /me/dashboard to read currentRound (lowest round
 *   with at least one non-FullTime match, or highest round otherwise).
 * - Signed-out / dashboard error: falls back to round 1 so the page still
 *   renders (the Round component handles 503/empty cache itself).
 */
export default function Predictions() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (loading) return;

    const goTo = (round: number) => {
      navigate(`/round/${CURRENT_SEASON}/${round}`, { replace: true });
    };

    if (!user) {
      goTo(FALLBACK_ROUND);
      return;
    }

    let cancelled = false;
    getDashboard(CURRENT_SEASON, null)
      .then((d) => {
        if (cancelled) return;
        goTo(d.currentRound ?? FALLBACK_ROUND);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof NotSignedInError) {
          goTo(FALLBACK_ROUND);
          return;
        }
        // Soft-fail to a default round so the user still gets somewhere.
        setError("Couldn't detect the current round; showing Round 1.");
        goTo(FALLBACK_ROUND);
      });

    return () => {
      cancelled = true;
    };
  }, [user, loading, navigate]);

  return (
    <section>
      <p role="status">Loading predictions…</p>
      {error && <p className="muted">{error}</p>}
    </section>
  );
}
