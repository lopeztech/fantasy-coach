"""Travel & scheduling features for the NRL prediction model.

Three features, all expressed as home-minus-away differentials:

- ``travel_km_diff``: great-circle kilometres each team travelled from their
  previous match venue to this one.  Teams with no prior match this season
  get 0 km (neutral — no advantage modelled for a team starting from home).

- ``timezone_delta_diff``: absolute hours of timezone shift since the previous
  match.  The Warriors flying Auckland→Brisbane is +3 h; Brisbane→Sydney is 0.
  Zero for first match of season.

- ``back_to_back_short_week_diff``: +1 / −1 / 0 flag capturing the interaction
  of short rest (< 6 days) AND long travel (> 1 000 km).  Both conditions must
  hold for a team to score ±1; partial matches are 0.
"""

from __future__ import annotations

import csv
import math
from importlib import resources
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Venue database
# ---------------------------------------------------------------------------

_VENUES_FILE = Path(__file__).parent.parent.parent / "data" / "venues.csv"


class _VenueInfo(NamedTuple):
    name: str
    city: str
    lat: float
    lon: float
    tz_offset: int  # hours offset from UTC


_VENUE_DB: dict[str, _VenueInfo] | None = None


def _load_venues() -> dict[str, _VenueInfo]:
    global _VENUE_DB
    if _VENUE_DB is not None:
        return _VENUE_DB

    path = _VENUES_FILE
    db: dict[str, _VenueInfo] = {}
    try:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                info = _VenueInfo(
                    name=row["name"],
                    city=row["city"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    tz_offset=int(row["timezone_offset"]),
                )
                db[row["name"].lower()] = info
    except FileNotFoundError:
        pass  # No venues file — features will degrade to zero.
    _VENUE_DB = db
    return db


def lookup_venue(venue_name: str | None) -> _VenueInfo | None:
    """Return venue info for *venue_name* using a case-insensitive lookup."""
    if not venue_name:
        return None
    return _load_venues().get(venue_name.lower())


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two coordinates."""
    r = 6_371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(d_lon / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def travel_features(
    home_prev_venue: str | None,
    away_prev_venue: str | None,
    current_venue: str | None,
    home_days_rest: float,
    away_days_rest: float,
) -> tuple[float, float, float]:
    """Return ``(travel_km_diff, timezone_delta_diff, bb_short_week_diff)``.

    All three values are expressed as home − away.

    Parameters
    ----------
    home_prev_venue:
        Venue name where the home team last played (None if no prior match).
    away_prev_venue:
        Venue name where the away team last played (None if no prior match).
    current_venue:
        Venue name of the upcoming match.
    home_days_rest, away_days_rest:
        Days since each team last played (from `FeatureBuilder`).
    """
    cur_info = lookup_venue(current_venue)

    home_km = _travel_km(home_prev_venue, cur_info)
    away_km = _travel_km(away_prev_venue, cur_info)
    travel_km_diff = home_km - away_km

    home_tz = _tz_delta(home_prev_venue, cur_info)
    away_tz = _tz_delta(away_prev_venue, cur_info)
    timezone_delta_diff = float(home_tz - away_tz)

    home_bb = _is_brutal(home_days_rest, home_km)
    away_bb = _is_brutal(away_days_rest, away_km)
    bb_diff = float(home_bb - away_bb)

    return travel_km_diff, timezone_delta_diff, bb_diff


def _travel_km(prev_venue: str | None, cur_info: _VenueInfo | None) -> float:
    if cur_info is None or prev_venue is None:
        return 0.0
    prev_info = lookup_venue(prev_venue)
    if prev_info is None:
        return 0.0
    return haversine_km(prev_info.lat, prev_info.lon, cur_info.lat, cur_info.lon)


def _tz_delta(prev_venue: str | None, cur_info: _VenueInfo | None) -> int:
    if cur_info is None or prev_venue is None:
        return 0
    prev_info = lookup_venue(prev_venue)
    if prev_info is None:
        return 0
    return abs(cur_info.tz_offset - prev_info.tz_offset)


def _is_brutal(days_rest: float, travel_km: float) -> int:
    """Return 1 if the team has short rest AND long travel, else 0."""
    return int(days_rest < 6.0 and travel_km > 1_000.0)
