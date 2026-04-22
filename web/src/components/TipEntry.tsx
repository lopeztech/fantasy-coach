import type { TipChoice } from "../tips";

type Props = {
  homeTeam: string;
  awayTeam: string;
  value: TipChoice | null;
  locked: boolean;
  saving: boolean;
  onTip: (choice: TipChoice) => void;
};

export function TipEntry({ homeTeam, awayTeam, value, locked, saving, onTip }: Props) {
  return (
    <div className="tip-entry" aria-label="Your tip">
      <span className="tip-label">Your tip</span>
      <div className="tip-buttons" role="group" aria-label={`Tip for ${homeTeam} vs ${awayTeam}`}>
        {(["home", "away"] as const).map((side) => {
          const name = side === "home" ? homeTeam : awayTeam;
          const selected = value === side;
          return (
            <button
              key={side}
              type="button"
              className={`tip-btn${selected ? " tip-btn--selected" : ""}`}
              aria-pressed={selected}
              disabled={locked || saving}
              onClick={(e) => {
                e.preventDefault();
                onTip(side);
              }}
            >
              {name}
            </button>
          );
        })}
      </div>
      {locked && <span className="tip-locked">Locked</span>}
    </div>
  );
}
