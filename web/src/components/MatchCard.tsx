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

export function MatchCard({ prediction, preview }: { prediction: Prediction; preview?: string }) {
  const p = prediction;
  const homePct = Math.round(p.homeWinProbability * 100);
  const awayPct = 100 - homePct;
  const winnerName = p.predictedWinner === "home" ? p.home.name : p.away.name;
  const winnerPct = p.predictedWinner === "home" ? homePct : awayPct;

  return (
    <article className="match-card" aria-label={`${p.home.name} vs ${p.away.name}`}>
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
    </article>
  );
}
