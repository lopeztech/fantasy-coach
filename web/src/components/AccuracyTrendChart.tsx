import type { RoundAccuracy } from "../types";

const SVG_H = 80;
const SVG_W = 400;
const PAD_X = 8;
const PAD_Y = 8;
const THRESHOLD = 0.55;

function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

export function AccuracyTrendChart({ rounds }: { rounds: RoundAccuracy[] }) {
  if (rounds.length < 2) return null;

  const innerW = SVG_W - PAD_X * 2;
  const innerH = SVG_H - PAD_Y * 2;

  const minAcc = Math.min(...rounds.map((r) => r.accuracy), THRESHOLD) - 0.05;
  const maxAcc = Math.max(...rounds.map((r) => r.accuracy), THRESHOLD) + 0.05;
  const accRange = maxAcc - minAcc || 0.1;

  const toX = (i: number) => PAD_X + (i / (rounds.length - 1)) * innerW;
  const toY = (acc: number) => PAD_Y + (1 - (acc - minAcc) / accRange) * innerH;

  const points = rounds.map((r, i) => `${toX(i).toFixed(1)},${toY(r.accuracy).toFixed(1)}`);
  const polyline = points.join(" ");

  const thresholdY = toY(THRESHOLD).toFixed(1);

  return (
    <div className="accuracy-trend-wrap" aria-label="Accuracy trend chart">
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        role="img"
        aria-label={`Rolling accuracy: ${rounds.map((r) => `Rd ${r.round} ${pct(r.accuracy)}`).join(", ")}`}
        className="accuracy-trend-svg"
        preserveAspectRatio="none"
      >
        {/* Threshold line */}
        <line
          x1={PAD_X}
          y1={thresholdY}
          x2={SVG_W - PAD_X}
          y2={thresholdY}
          stroke="#f39c12"
          strokeWidth="1"
          strokeDasharray="4 3"
          aria-hidden="true"
        />

        {/* Accuracy polyline */}
        <polyline
          points={polyline}
          fill="none"
          stroke="#2980b9"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Data points */}
        {rounds.map((r, i) => (
          <circle
            key={`${r.season}-${r.round}`}
            cx={toX(i).toFixed(1)}
            cy={toY(r.accuracy).toFixed(1)}
            r="3"
            fill={r.accuracy >= THRESHOLD ? "#27ae60" : "#e74c3c"}
            aria-label={`Round ${r.round}: ${pct(r.accuracy)}`}
          />
        ))}
      </svg>

      <div className="accuracy-trend-labels" aria-hidden="true">
        <span>Rd {rounds[0].round}</span>
        <span>Rd {rounds[rounds.length - 1].round}</span>
      </div>
    </div>
  );
}
