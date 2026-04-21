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
    "broncos": frozenset({"broncos", "brisbane broncos"}),
    "raiders": frozenset({"raiders", "canberra raiders"}),
    "bulldogs": frozenset({"bulldogs", "canterbury bulldogs", "canterbury-bankstown bulldogs"}),
    "sharks": frozenset({"sharks", "cronulla sharks", "cronulla-sutherland sharks"}),
    "dolphins": frozenset({"dolphins", "redcliffe dolphins"}),
    "titans": frozenset({"titans", "gold coast titans"}),
    "sea-eagles": frozenset(
        {"sea eagles", "sea-eagles", "manly sea eagles", "manly-warringah sea eagles"}
    ),
    "storm": frozenset({"storm", "melbourne storm"}),
    "knights": frozenset({"knights", "newcastle knights"}),
    "cowboys": frozenset({"cowboys", "north queensland cowboys"}),
    "eels": frozenset({"eels", "parramatta eels"}),
    "panthers": frozenset({"panthers", "penrith panthers"}),
    "rabbitohs": frozenset({"rabbitohs", "south sydney rabbitohs"}),
    "dragons": frozenset(
        {"dragons", "st george illawarra dragons", "st. george illawarra dragons"}
    ),
    "roosters": frozenset({"roosters", "sydney roosters"}),
    "warriors": frozenset({"warriors", "new zealand warriors", "nz warriors"}),
    "wests-tigers": frozenset({"wests tigers", "wests-tigers", "west tigers"}),
}


def canonicalize(name: str) -> str | None:
    """Return the canonical team key for a free-form team name, or None."""
    needle = name.strip().lower()
    for canonical, aliases in CANONICAL_TEAMS.items():
        if needle in aliases:
            return canonical
    return None
