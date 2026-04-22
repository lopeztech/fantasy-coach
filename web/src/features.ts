// Plain-English labels for each model feature. The backend emits raw
// feature names (matching fantasy_coach.feature_engineering.FEATURE_NAMES);
// this module turns them into a one-line sentence the UI can render with
// the concrete home/away team names.
//
// All numeric feature values are home-minus-away by convention (see the
// backend's feature docstring), so positive values favour the home team.

import type { FeatureContribution } from "./types";

export type ContributionLabel = {
  /** One-line sentence suitable for a list row. */
  text: string;
  /** "home" when the sign of the contribution favours the home team, else "away". */
  favours: "home" | "away" | "neutral";
};

function favouredBy(contribution: number): "home" | "away" | "neutral" {
  if (contribution > 0) return "home";
  if (contribution < 0) return "away";
  return "neutral";
}

function teamFavoured(contribution: number, home: string, away: string): string {
  return contribution >= 0 ? home : away;
}

export function labelFor(
  c: FeatureContribution,
  home: string,
  away: string,
): ContributionLabel {
  const { feature, value, contribution } = c;
  const favours = favouredBy(contribution);
  const text = describe(feature, value, contribution, home, away, c.detail);
  return { text, favours };
}

type MissingPlayer = { name: string | null; position: string };

function describe(
  feature: string,
  value: number,
  contribution: number,
  home: string,
  away: string,
  detail?: Record<string, unknown> | null,
): string {
  const mag = Math.abs(value);
  const rounded = (n: number, d = 1) => n.toFixed(d).replace(/\.0$/, "");

  switch (feature) {
    case "elo_diff": {
      const who = teamFavoured(contribution, home, away);
      return `${who} rated ${rounded(mag, 0)} Elo points higher`;
    }
    case "form_diff_pf": {
      const who = teamFavoured(value, home, away);
      return `${who} scoring ${rounded(mag)} more points per game (recent form)`;
    }
    case "form_diff_pa": {
      // form_diff_pa: home_pa - away_pa. Lower pa = tighter defence, so a
      // negative value favours the home team. Invert for a natural sentence.
      const who = teamFavoured(-value, home, away);
      return `${who} conceding ${rounded(mag)} fewer points per game (recent defence)`;
    }
    case "days_rest_diff": {
      const who = teamFavoured(value, home, away);
      return `${who} had ${rounded(mag)} more days' rest`;
    }
    case "h2h_recent_diff": {
      const who = teamFavoured(value, home, away);
      return `${who} leads recent head-to-head by ${rounded(mag)} points`;
    }
    case "is_home_field":
      return `Home-field advantage for ${home}`;
    case "travel_km_diff": {
      // Positive = home travelled further, which hurts home.
      const disadvantaged = teamFavoured(-value, home, away);
      return `${disadvantaged} travelled ${rounded(mag, 0)} km less`;
    }
    case "timezone_delta_diff": {
      const disadvantaged = teamFavoured(-value, home, away);
      return `${disadvantaged} had the smaller timezone shift (${rounded(mag)} h)`;
    }
    case "back_to_back_short_week_diff":
      // +1/−1/0 flag for short-week + travel combo.
      if (value === 0) return "No short-week travel penalty on either side";
      return value > 0
        ? `${away} avoided a short-week long-travel penalty`
        : `${home} avoided a short-week long-travel penalty`;
    case "is_wet":
      return value > 0.5 ? "Wet-weather forecast" : "Dry-weather forecast";
    case "wind_kph":
      return `Wind ${rounded(value, 0)} kph`;
    case "temperature_c":
      return `Temperature ${rounded(value, 0)} °C`;
    case "missing_weather":
      return value > 0.5 ? "Weather data unavailable for venue" : "Weather data available";
    case "venue_avg_total_points":
      return `Venue typically sees ${rounded(value, 0)} combined points per match`;
    case "venue_home_win_rate": {
      const pct = Math.round(value * 100);
      return `Home teams win ${pct}% of matches at this venue`;
    }
    case "ref_avg_total_points":
      return `Referee averages ${rounded(value, 0)} combined points per match`;
    case "ref_home_penalty_diff": {
      const who = teamFavoured(-value, home, away);
      return `Referee tends to penalise ${who === home ? away : home} more (${rounded(mag)} fewer home penalties)`;
    }
    case "missing_referee":
      return value > 0.5 ? "Referee not yet confirmed" : "Referee confirmed";
    case "form_diff_pf_adjusted": {
      const who = teamFavoured(value, home, away);
      return `${who} scoring ${rounded(Math.abs(value))} more points above opponent's defensive baseline`;
    }
    case "form_diff_pa_adjusted": {
      const who = teamFavoured(-value, home, away);
      return `${who} conceding ${rounded(Math.abs(value))} fewer points relative to opponent's scoring baseline`;
    }
    case "key_absence_diff": {
      if (value === 0) return "No key player absences for either side";
      // value > 0 → home missing more (hurts home); value < 0 → away missing more.
      const affectedTeam = value > 0 ? home : away;
      const missingKey = value > 0 ? "home_missing" : "away_missing";
      const missing = detail?.[missingKey] as MissingPlayer[] | undefined;
      if (missing && missing.length > 0) {
        const first = missing[0];
        const playerStr = first.name
          ? `${first.name} (${first.position})`
          : first.position;
        if (missing.length === 1) return `${affectedTeam} missing ${playerStr}`;
        return `${affectedTeam} missing ${playerStr} and ${missing.length - 1} other${missing.length > 2 ? "s" : ""}`;
      }
      return `${affectedTeam} missing key player${Math.abs(value) > 1 ? "s" : ""}`;
    }
    default:
      // Fallback for unknown feature names (e.g. future feature additions).
      return `${feature} = ${rounded(value, 2)}`;
  }
}
