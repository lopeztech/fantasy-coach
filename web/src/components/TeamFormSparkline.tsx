import type { TeamFormEntry } from "../types";

const CELL_W = 5;
const CELL_H = 16;
const CELL_GAP = 1;
const SVG_W = 60;
const SVG_H = 20;
const COLORS: Record<string, string> = {
  win: "#27ae60",
  loss: "#e74c3c",
  draw: "#bdbdbd",
};

export function TeamFormSparkline({
  matches,
  teamName,
}: {
  matches: TeamFormEntry[];
  teamName: string;
}) {
  const last10 = matches.slice(-10);
  if (last10.length === 0) return null;

  const labels = last10.map((m) => (m.result === "win" ? "W" : m.result === "loss" ? "L" : "D"));
  const ariaLabel = `${teamName} last ${last10.length}: ${labels.join(" ")}`;

  return (
    <svg
      width={SVG_W}
      height={SVG_H}
      role="img"
      aria-label={ariaLabel}
      className="form-sparkline"
    >
      {last10.map((m, i) => (
        <rect
          key={m.matchId}
          x={i * (CELL_W + CELL_GAP)}
          y={2}
          width={CELL_W}
          height={CELL_H}
          fill={COLORS[m.result] ?? COLORS.draw}
          rx={1}
        />
      ))}
    </svg>
  );
}
