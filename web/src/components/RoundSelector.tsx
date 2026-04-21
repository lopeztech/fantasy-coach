import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";

type Props = {
  initialSeason?: number;
  initialRound?: number;
};

export function RoundSelector({ initialSeason, initialRound }: Props) {
  const now = new Date();
  const [season, setSeason] = useState<number>(initialSeason ?? now.getFullYear());
  const [round, setRound] = useState<number>(initialRound ?? 1);
  const navigate = useNavigate();

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    navigate(`/round/${season}/${round}`);
  }

  return (
    <form className="round-selector" onSubmit={handleSubmit}>
      <label>
        Season
        <input
          type="number"
          value={season}
          min={2020}
          max={now.getFullYear() + 1}
          onChange={(e) => setSeason(Number(e.target.value))}
        />
      </label>
      <label>
        Round
        <input
          type="number"
          value={round}
          min={1}
          max={30}
          onChange={(e) => setRound(Number(e.target.value))}
        />
      </label>
      <button type="submit">View predictions</button>
    </form>
  );
}
