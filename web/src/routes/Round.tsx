import { useParams } from "react-router-dom";

export default function Round() {
  const { season, round } = useParams();
  return (
    <section>
      <h1>
        Round {round}, {season}
      </h1>
      <p>Predictions UI lands in #21.</p>
    </section>
  );
}
