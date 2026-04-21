"""Load historical NRL closing lines and de-vig them.

Source: aussportsbetting.com NRL Excel sheet
(`https://www.aussportsbetting.com/historical_data/nrl.xlsx`). The columns
we read are `Date`, `Home Team`, `Away Team`, and the home/away closing
odds — preferring the explicit `Home Odds Close` / `Away Odds Close`
columns and falling back to the unsuffixed `Home Odds` / `Away Odds`
columns when they're empty (BlueBet sourcing from April 2024 onward
populates the unsuffixed columns reliably and the Close columns sparsely).

Decimal odds → implied probabilities via `1 / odds`. The two implied
probabilities sum to slightly more than 1 (the bookmaker's overround /
"vig"). De-vigging here uses the simplest fair approach — proportional
normalisation — which is biased on long shots but accurate enough for a
benchmark that we're trying to *match*, not exploit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import openpyxl

from fantasy_coach.bookmaker.team_names import canonicalize


@dataclass(frozen=True)
class ClosingLine:
    match_date: date
    home_canonical: str
    away_canonical: str
    home_odds_close: float
    away_odds_close: float
    p_home_devigged: float

    @property
    def key(self) -> tuple[date, str, str]:
        return (self.match_date, self.home_canonical, self.away_canonical)


_REQUIRED_HEADERS = (
    "Date",
    "Home Team",
    "Away Team",
    "Home Odds Close",
    "Away Odds Close",
    "Home Odds",
    "Away Odds",
)


def devig_two_way(home_odds: float, away_odds: float) -> float:
    """Return the de-vigged home win probability.

    Implied probs = 1/odds; normalise so they sum to 1.
    """
    if home_odds <= 1.0 or away_odds <= 1.0:
        raise ValueError(f"odds must be >1, got home={home_odds}, away={away_odds}")
    p_home_raw = 1.0 / home_odds
    p_away_raw = 1.0 / away_odds
    total = p_home_raw + p_away_raw
    return p_home_raw / total


def load_closing_lines(path: Path | str) -> dict[tuple[date, str, str], ClosingLine]:
    """Parse the aussportsbetting NRL xlsx into a date+team-keyed dict.

    Rows missing required columns or with un-mappable team names are skipped
    silently — the dataset includes pre-NRL rep games and historical teams
    that aren't relevant to the modern fixture list.
    """
    wb = openpyxl.load_workbook(Path(path), read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)

    # Header row is the second physical row (the first is a freeform note).
    next(rows, None)  # banner
    header = next(rows, None)
    if not header or not all(h in header for h in _REQUIRED_HEADERS):
        raise ValueError(f"xlsx is missing required columns; saw header={header!r}")
    cols = {name: header.index(name) for name in _REQUIRED_HEADERS}

    out: dict[tuple[date, str, str], ClosingLine] = {}
    for row in rows:
        try:
            line = _row_to_line(row, cols)
        except _SkipRowError:
            continue
        out[line.key] = line
    return out


class _SkipRowError(Exception):
    pass


def _row_to_line(row: tuple, cols: dict[str, int]) -> ClosingLine:
    raw_date = row[cols["Date"]]
    home_raw = row[cols["Home Team"]]
    away_raw = row[cols["Away Team"]]
    home_odds = row[cols["Home Odds Close"]] or row[cols["Home Odds"]]
    away_odds = row[cols["Away Odds Close"]] or row[cols["Away Odds"]]

    if raw_date is None or home_raw is None or away_raw is None:
        raise _SkipRowError
    if home_odds is None or away_odds is None:
        raise _SkipRowError

    match_date = raw_date.date() if hasattr(raw_date, "date") else raw_date
    home_canonical = canonicalize(home_raw)
    away_canonical = canonicalize(away_raw)
    if home_canonical is None or away_canonical is None:
        raise _SkipRowError

    try:
        p_home = devig_two_way(float(home_odds), float(away_odds))
    except ValueError as exc:
        raise _SkipRowError from exc

    return ClosingLine(
        match_date=match_date,
        home_canonical=home_canonical,
        away_canonical=away_canonical,
        home_odds_close=float(home_odds),
        away_odds_close=float(away_odds),
        p_home_devigged=p_home,
    )
