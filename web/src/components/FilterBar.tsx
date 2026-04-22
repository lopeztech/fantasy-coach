import { useEffect, useRef, useState } from "react";

export type SortKey = "kickoff" | "confidence" | "alpha";

interface Props {
  q: string;
  onQ: (v: string) => void;
  tossup: boolean;
  onTossup: (v: boolean) => void;
  myTeams: boolean;
  onMyTeams: (v: boolean) => void;
  sort: SortKey;
  onSort: (v: SortKey) => void;
  allTeams: string[];
  favTeams: Set<string>;
  onToggleFav: (name: string) => void;
}

export function FilterBar({
  q,
  onQ,
  tossup,
  onTossup,
  myTeams,
  onMyTeams,
  sort,
  onSort,
  allTeams,
  favTeams,
  onToggleFav,
}: Props) {
  const searchRef = useRef<HTMLInputElement>(null);
  const [prefsOpen, setPrefsOpen] = useState(false);

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const active = document.activeElement;
      const tag = (active as HTMLElement | null)?.tagName;
      if (e.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") {
        e.preventDefault();
        searchRef.current?.focus();
      }
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  return (
    <div className="filter-bar">
      <div className="filter-bar-row">
        <label className="sr-only" htmlFor="round-search">
          Search teams
        </label>
        <input
          id="round-search"
          ref={searchRef}
          type="search"
          className="filter-search"
          value={q}
          placeholder="Search teams… ( / )"
          onChange={(e) => onQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              onQ("");
              e.currentTarget.blur();
            }
          }}
        />

        <label className="filter-toggle">
          <input type="checkbox" checked={tossup} onChange={(e) => onTossup(e.target.checked)} />
          Toss-ups
        </label>

        <label className="filter-toggle">
          <input type="checkbox" checked={myTeams} onChange={(e) => onMyTeams(e.target.checked)} />
          My teams
        </label>

        <label className="filter-sort">
          Sort
          <select value={sort} onChange={(e) => onSort(e.target.value as SortKey)}>
            <option value="kickoff">Kickoff</option>
            <option value="confidence">Confidence</option>
            <option value="alpha">A–Z</option>
          </select>
        </label>

        {allTeams.length > 0 && (
          <button
            className="prefs-btn"
            onClick={() => setPrefsOpen((o) => !o)}
            aria-label="Manage favourite teams"
            aria-expanded={prefsOpen}
            title="Favourite teams"
          >
            ⚙
          </button>
        )}
      </div>

      {prefsOpen && allTeams.length > 0 && (
        <div className="prefs-panel" role="group" aria-label="Favourite teams">
          <p className="prefs-heading">Favourite teams</p>
          <ul className="prefs-team-list">
            {allTeams.map((name) => (
              <li key={name}>
                <label className="filter-toggle">
                  <input
                    type="checkbox"
                    checked={favTeams.has(name)}
                    onChange={() => onToggleFav(name)}
                  />
                  {name}
                </label>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
