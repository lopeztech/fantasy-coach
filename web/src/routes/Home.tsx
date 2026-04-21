import { Link } from "react-router-dom";

export default function Home() {
  const now = new Date();
  const season = now.getFullYear();
  return (
    <section>
      <h1>Predictions</h1>
      <p>
        Pick a round to see predicted winners and home-win probabilities for every match.
      </p>
      <p>
        Example:{" "}
        <Link to={`/round/${season}/1`}>
          /round/{season}/1
        </Link>
      </p>
    </section>
  );
}
