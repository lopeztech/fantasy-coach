import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useAuth } from "../auth";
import { getTeams, getVenues } from "../api";
import { buildIndex, search, recordVisit } from "../search";
import type { SearchIndex, SearchResult, SearchResultKind } from "../search";

const CURRENT_SEASON = 2026;
const MAX_ROUND = 27;

const KIND_LABEL: Record<SearchResultKind, string> = {
  team: "Teams",
  match: "Matches",
  round: "Rounds",
  venue: "Venues",
  page: "Pages",
};

export function SearchBar() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [index, setIndex] = useState<SearchIndex | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Build the search index once the user is authenticated.
  useEffect(() => {
    if (!user || index) return;
    Promise.all([
      getTeams(CURRENT_SEASON).catch(() => [] as Array<{ id: number; name: string }>),
      getVenues().catch(() => [] as Array<{ name: string; city: string }>),
    ]).then(([teams, venues]) => {
      setIndex(buildIndex(teams, venues, CURRENT_SEASON, MAX_ROUND));
    });
  }, [user, index]);

  // Update results whenever query or index changes.
  useEffect(() => {
    if (!index) return;
    const r = search(query, index);
    setResults(r);
    setActiveIndex(0);
  }, [query, index]);

  const openBar = useCallback(() => {
    setOpen(true);
    setQuery("");
    setTimeout(() => inputRef.current?.focus(), 0);
  }, []);

  const closeBar = useCallback(() => {
    setOpen(false);
    setQuery("");
  }, []);

  const selectResult = useCallback(
    (result: SearchResult) => {
      recordVisit({ label: result.label, href: result.href, kind: result.kind });
      closeBar();
      navigate(result.href);
    },
    [closeBar, navigate],
  );

  // Global keyboard shortcut: ⌘K / Ctrl+K
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (open) closeBar();
        else openBar();
      }
      if (e.key === "Escape" && open) closeBar();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, openBar, closeBar]);

  // Click-outside to close
  useEffect(() => {
    if (!open) return;
    function onClick(e: MouseEvent) {
      if (overlayRef.current && !overlayRef.current.contains(e.target as Node)) {
        closeBar();
      }
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open, closeBar]);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && results[activeIndex]) {
      selectResult(results[activeIndex]);
    }
  }

  if (!user) return null;

  // Group results by kind for display headers.
  const grouped: Array<{ header: string; items: SearchResult[] }> = [];
  let currentKind: SearchResultKind | null = null;
  for (const r of results) {
    if (r.kind !== currentKind) {
      currentKind = r.kind;
      grouped.push({ header: KIND_LABEL[r.kind as SearchResultKind], items: [] });
    }
    grouped[grouped.length - 1].items.push(r);
  }

  // Flat index mapping for keyboard navigation.
  const flatResults = results;

  return (
    <>
      <button
        className="search-trigger"
        onClick={openBar}
        aria-label="Open global search (⌘K)"
        title="Search (⌘K)"
      >
        <span className="search-trigger-icon" aria-hidden>🔍</span>
        <span className="search-trigger-hint">⌘K</span>
      </button>

      {open && (
        <div className="search-overlay" role="presentation">
          <div
            className="search-dialog"
            ref={overlayRef}
            role="dialog"
            aria-label="Global search"
            aria-modal="true"
          >
            <div className="search-input-row">
              <span className="search-input-icon" aria-hidden>🔍</span>
              <input
                ref={inputRef}
                className="search-input"
                role="combobox"
                aria-expanded={results.length > 0}
                aria-autocomplete="list"
                aria-controls="search-results"
                aria-activedescendant={
                  results[activeIndex] ? `sr-${activeIndex}` : undefined
                }
                type="text"
                placeholder="Search teams, rounds, venues…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                autoComplete="off"
                spellCheck={false}
              />
              <kbd className="search-esc-hint" onClick={closeBar}>
                Esc
              </kbd>
            </div>

            {results.length > 0 && (
              <ul
                id="search-results"
                className="search-results"
                role="listbox"
                aria-label="Search results"
              >
                {grouped.map((group) =>
                  group.items.map((result) => {
                    const idx = flatResults.indexOf(result);
                    return (
                      <li
                        key={result.href}
                        id={`sr-${idx}`}
                        role="option"
                        aria-selected={idx === activeIndex}
                        className={`search-result${idx === activeIndex ? " search-result--active" : ""}`}
                        onMouseEnter={() => setActiveIndex(idx)}
                        onClick={() => selectResult(result)}
                      >
                        <span className="search-result-label">{result.label}</span>
                        {result.sublabel && (
                          <span className="search-result-sub">{result.sublabel}</span>
                        )}
                      </li>
                    );
                  }),
                )}
              </ul>
            )}

            {results.length === 0 && query.trim() && (
              <p className="search-empty">No results for &ldquo;{query}&rdquo;</p>
            )}

            {results.length === 0 && !query.trim() && (
              <p className="search-empty search-empty--hint">Type to search teams, rounds, and venues</p>
            )}
          </div>
        </div>
      )}
    </>
  );
}
