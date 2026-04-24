import type { AccuracyFilters, TeamOption } from "../types";

interface Props {
  filters: AccuracyFilters;
  teams: TeamOption[];
  venues: string[];
  modelVersions: string[];
  onChange: (next: AccuracyFilters) => void;
}

export function AccuracyFilterBar({ filters, teams, venues, modelVersions, onChange }: Props) {
  return (
    <div className="accuracy-filters" role="search" aria-label="Filter accuracy data">
      <label className="accuracy-filter-label">
        Team
        <select
          value={filters.teamId ?? ""}
          onChange={(e) =>
            onChange({ ...filters, teamId: e.target.value ? Number(e.target.value) : null })
          }
          aria-label="Filter by team"
        >
          <option value="">All teams</option>
          {teams.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </label>

      <label className="accuracy-filter-label">
        Venue
        <select
          value={filters.venue ?? ""}
          onChange={(e) => onChange({ ...filters, venue: e.target.value || null })}
          aria-label="Filter by venue"
        >
          <option value="">All venues</option>
          {venues.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>

      {modelVersions.length > 1 && (
        <label className="accuracy-filter-label">
          Model version
          <select
            value={filters.modelVersion ?? ""}
            onChange={(e) => onChange({ ...filters, modelVersion: e.target.value || null })}
            aria-label="Filter by model version"
          >
            <option value="">All versions</option>
            {modelVersions.map((mv) => (
              <option key={mv} value={mv}>
                {mv.slice(0, 8)}
              </option>
            ))}
          </select>
        </label>
      )}

      {(filters.teamId !== null || filters.venue !== null || filters.modelVersion !== null) && (
        <button
          className="accuracy-filter-clear"
          onClick={() => onChange({ teamId: null, venue: null, modelVersion: null })}
          type="button"
        >
          Clear filters
        </button>
      )}
    </div>
  );
}
