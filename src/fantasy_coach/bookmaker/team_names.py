"""Map between NRL nicknames and aussportsbetting full team names.

The bookmaker dataset uses full club names ("Parramatta Eels"); the NRL
endpoints use short nicknames ("Eels"). Both must collapse to a single
canonical token before joining on (home, away).

Maintained by hand — only ~17 active teams, and a name change is a once-
every-few-years event we'd want to notice anyway.
"""

from __future__ import annotations

# Canonical key → set of accepted aliases (lowercased, hyphenated where useful).
CANONICAL_TEAMS: dict[str, frozenset[str]] = {
    "broncos": frozenset({"broncos", "brisbane broncos", "brisbane"}),
    "raiders": frozenset({"raiders", "canberra raiders", "canberra"}),
    "bulldogs": frozenset(
        {
            "bulldogs",
            "canterbury bulldogs",
            "canterbury-bankstown bulldogs",
            "canterbury bankstown bulldogs",
            "canterbury",
        }
    ),
    "sharks": frozenset(
        {
            "sharks",
            "cronulla sharks",
            "cronulla-sutherland sharks",
            "cronulla sutherland sharks",
            "cronulla",
        }
    ),
    "dolphins": frozenset({"dolphins", "redcliffe dolphins", "redcliffe"}),
    "titans": frozenset({"titans", "gold coast titans", "gold coast"}),
    "sea-eagles": frozenset(
        {"sea eagles", "sea-eagles", "manly sea eagles", "manly-warringah sea eagles", "manly"}
    ),
    "storm": frozenset({"storm", "melbourne storm", "melbourne"}),
    "knights": frozenset({"knights", "newcastle knights", "newcastle"}),
    "cowboys": frozenset(
        {
            "cowboys",
            "north queensland cowboys",
            "north qld cowboys",
            "nq cowboys",
            "north queensland",
            "north qld",
        }
    ),
    "eels": frozenset({"eels", "parramatta eels", "parramatta"}),
    "panthers": frozenset({"panthers", "penrith panthers", "penrith"}),
    "rabbitohs": frozenset({"rabbitohs", "south sydney rabbitohs", "south sydney"}),
    "dragons": frozenset(
        {
            "dragons",
            "st george illawarra dragons",
            "st. george illawarra dragons",
            "st george illawarra",
            "st. george illawarra",
            "st george dragons",
            "st. george dragons",
        }
    ),
    "roosters": frozenset({"roosters", "sydney roosters", "sydney"}),
    "warriors": frozenset({"warriors", "new zealand warriors", "nz warriors", "new zealand"}),
    "wests-tigers": frozenset({"wests tigers", "wests-tigers", "west tigers", "western suburbs"}),
}


def canonicalize(name: str) -> str | None:
    """Return the canonical team key for a free-form team name, or None."""
    needle = name.strip().lower()
    for canonical, aliases in CANONICAL_TEAMS.items():
        if needle in aliases:
            return canonical
    return None
