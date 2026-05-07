/** Confidence-tier pill badge for MatchCard (#257).
 *
 * Tier is computed server-side from Bayesian posterior spread + OOD score.
 * The badge maps the 3-level band to a colour and a human-readable tooltip.
 */

type Band = "low" | "medium" | "high";

function tierLabel(band: Band): string {
  if (band === "high") return "High confidence";
  if (band === "low") return "Low confidence";
  return "Medium confidence";
}

function tierReason(band: Band, oodFlag: string | null | undefined): string {
  if (band === "high") return "Model is decisive and the matchup is within training distribution.";
  if (band === "low") {
    if (oodFlag === "out_of_distribution") return "Outside training distribution — treat with caution.";
    if (oodFlag === "edge") return "Edge case — model is less certain than usual.";
    return "Coin-flip — model is evenly split on this matchup.";
  }
  return "Moderate confidence — take as a guide, not a certainty.";
}

export function ConfidenceBadge({
  band,
  oodFlag,
}: {
  band: Band;
  oodFlag?: string | null;
}) {
  const label = tierLabel(band);
  const reason = tierReason(band, oodFlag);

  return (
    <span
      className={`confidence-badge confidence-badge--${band}`}
      title={reason}
      aria-label={`${label}: ${reason}`}
    >
      {label}
    </span>
  );
}
