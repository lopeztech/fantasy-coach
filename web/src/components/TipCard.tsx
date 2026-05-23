import type { Tip, TipLeg } from "../types";

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

function formatOdds(odds: number): string {
  return `$${odds.toFixed(2)}`;
}

function formatPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function legSubtitle(leg: TipLeg): string {
  if (leg.market === "h2h") {
    return `${leg.homeName} v ${leg.awayName} — Head-to-head`;
  }
  return `${leg.homeName} v ${leg.awayName} — Total points`;
}

export function TipCard({ tip, index }: { tip: Tip; index: number }) {
  const kindLabel = tip.kind.charAt(0).toUpperCase() + tip.kind.slice(1);
  const edgeClass = tip.edge >= 0.15 ? "tip-edge tip-edge--strong" : "tip-edge";

  return (
    <article className="tip-card" aria-label={`${kindLabel} pick #${index + 1}`}>
      <header className="tip-card-head">
        <span className="tip-kind">
          {kindLabel} #{index + 1}
        </span>
        <span className={edgeClass}>+{(tip.edge * 100).toFixed(1)}% edge</span>
      </header>

      <ol className="tip-legs">
        {tip.legs.map((leg, i) => (
          <li key={`${leg.matchId}-${leg.market}-${i}`} className="tip-leg">
            <div className="tip-leg-meta">
              <span className="tip-leg-subtitle">{legSubtitle(leg)}</span>
              <time className="tip-leg-kickoff" dateTime={leg.kickoff}>
                {formatKickoff(leg.kickoff)}
              </time>
            </div>
            <div className="tip-leg-pick">
              <strong className="tip-leg-selection">{leg.selection}</strong>
              <span className="tip-leg-price">{formatOdds(leg.decimalOdds)}</span>
            </div>
            <div className="tip-leg-stats">
              <span>Model: {formatPct(leg.modelProb)}</span>
              <span>Market: {formatPct(leg.impliedProb)}</span>
              <span className="muted">via {leg.bookmaker}</span>
            </div>
          </li>
        ))}
      </ol>

      <footer className="tip-card-foot">
        <div className="tip-totals">
          <span>
            Combined odds <strong>{formatOdds(tip.combinedOdds)}</strong>
          </span>
          <span>
            Model joint prob <strong>{formatPct(tip.combinedModelProb)}</strong>
          </span>
        </div>
        <span className="tip-stake">
          Suggested stake: <strong>{tip.suggestedUnits.toFixed(1)}u</strong>
        </span>
      </footer>
    </article>
  );
}
