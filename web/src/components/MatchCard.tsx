import { Link } from "react-router-dom";

import { TipEntry } from "./TipEntry";
import type { TipChoice } from "../tips";
import type { Prediction } from "../types";

function formatKickoff(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function MatchCard({
  prediction,
  season,
  round,
  preview,
  tip,
  savingTip,
  onTip,
}: {
  prediction: Prediction;
  season: number;
  round: number;
  preview?: string;
  tip?: TipChoice | null;
  savingTip?: boolean;
  onTip?: (choice: TipChoice) => void;
}) {
  const p = prediction;
  const homePct = Math.round(p.homeWinProbability * 100);
  const awayPct = 100 - homePct;
  const winnerName = p.predictedWinner === "home" ? p.home.name : p.away.name;
  const winnerPct = p.predictedWinner === "home" ? homePct : awayPct;
  const isKickedOff = new Date(p.kickoff) <= new Date();

  return (
    <Link
      to={`/round/${season}/${round}/${p.matchId}`}
      className="match-card-link"
      aria-label={`${p.home.name} vs ${p.away.name} — view why`}
    >
      <article className="match-card">
        <header className="match-card-head">
          <div className="teams">
            <span className="team home">{p.home.name}</span>
            <span className="muted"> vs </span>
            <span className="team away">{p.away.name}</span>
          </div>
          <time className="kickoff muted" dateTime={p.kickoff}>
            {formatKickoff(p.kickoff)}
          </time>
        </header>

        <div className="prob-bar" role="img" aria-label={`Home win probability ${homePct}%`}>
          <span
            className="prob-bar-home"
            style={{ width: `${homePct}%` }}
            aria-hidden="true"
          />
        </div>
        <div className="prob-labels">
          <span>
            <strong>{p.home.name}</strong> {homePct}%
          </span>
          <span>
            {awayPct}% <strong>{p.away.name}</strong>
          </span>
        </div>

        <p className="pick">
          Pick: <strong>{winnerName}</strong> ({winnerPct}%)
        </p>

        {preview ? <p className="preview">{preview}</p> : null}

        {onTip != null && (
          <div
            onClick={(e) => e.preventDefault()}
            onKeyDown={(e) => e.preventDefault()}
            role="none"
          >
            <TipEntry
              homeTeam={p.home.name}
              awayTeam={p.away.name}
              value={tip ?? null}
              locked={isKickedOff}
              saving={savingTip ?? false}
              onTip={onTip}
            />
          </div>
        )}
      </article>
    </Link>
  );
}
