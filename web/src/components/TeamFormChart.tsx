import type { TeamFormEntry } from "../types";

const W = 240;
const H = 80;
const PAD = 8;

export function TeamFormChart({
  matches,
  teamName,
}: {
  matches: TeamFormEntry[];
  teamName: string;
}) {
  const last20 = matches.slice(-20);
  if (last20.length < 2) return null;

  const elos = last20.map((m) => m.eloAfter);
  const minElo = Math.min(...elos) - 20;
  const maxElo = Math.max(...elos) + 20;
  const range = maxElo - minElo;

  const toX = (i: number) => PAD + (i / (last20.length - 1)) * (W - 2 * PAD);
  const toY = (elo: number) => H - PAD - ((elo - minElo) / range) * (H - 2 * PAD);

  const points = last20.map((m, i) => `${toX(i).toFixed(1)},${toY(m.eloAfter).toFixed(1)}`).join(" ");
  const latestElo = elos[elos.length - 1];

  return (
    <figure className="form-chart-figure">
      <svg
        width={W}
        height={H}
        role="img"
        aria-label={`${teamName} Elo trend over last ${last20.length} matches`}
        className="form-chart"
      >
        <polyline
          points={points}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {/* Terminal dot */}
        <circle
          cx={toX(last20.length - 1)}
          cy={toY(latestElo)}
          r={3}
          fill="currentColor"
        />
      </svg>
      <figcaption className="muted fine-print">
        {teamName} — Elo {latestElo}
      </figcaption>
    </figure>
  );
}
