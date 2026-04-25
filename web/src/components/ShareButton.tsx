import { useState } from "react";

import { API_BASE_URL } from "../firebase";

interface Props {
  matchId: number;
  season: number;
  round: number;
  homeTeam: string;
  awayTeam: string;
}

export function ShareButton({ matchId, season, round, homeTeam, awayTeam }: Props) {
  const [state, setState] = useState<"idle" | "copied" | "shared">("idle");

  // The share URL points at the server-side HTML shell which carries og:image /
  // og:title meta tags that social crawlers (Twitterbot, Slackbot) can read.
  const shareUrl = `${API_BASE_URL}/share/match/${matchId}?season=${season}&round=${round}`;
  const shareTitle = `${homeTeam} vs ${awayTeam} — Fantasy Coach`;

  const handleShare = async () => {
    if (typeof navigator !== "undefined" && "share" in navigator) {
      try {
        await navigator.share({ title: shareTitle, url: shareUrl });
        setState("shared");
        setTimeout(() => setState("idle"), 2000);
        return;
      } catch {
        // User cancelled share sheet or it failed — fall through to clipboard.
      }
    }
    try {
      await navigator.clipboard.writeText(shareUrl);
      setState("copied");
      setTimeout(() => setState("idle"), 2000);
    } catch {
      // Clipboard API unavailable (e.g. non-secure context) — ignore.
    }
  };

  return (
    <button
      className="share-button"
      onClick={handleShare}
      aria-label={`Share ${homeTeam} vs ${awayTeam} prediction`}
    >
      {state === "copied" ? "Link copied!" : state === "shared" ? "Shared!" : "Share"}
    </button>
  );
}
