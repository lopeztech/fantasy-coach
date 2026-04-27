"""Pre-kickoff weather forecast fetcher using Open-Meteo (free, no auth).

Closes the train/serve skew documented in issue #207: the existing weather
features (`is_wet`, `wind_kph`, `temperature_c`) are derived from the
*post-match* NRL payload, so at precompute time (Tue/Thu, 2-4 days before
kickoff) they default to ``missing_weather=1.0``.  This module fetches a
3-hour-window forecast from Open-Meteo for each upcoming match so the feature
pipeline can use real weather signal at inference time.

Cache: in-process dict keyed by (lat, lon, kickoff_date); 6-hour TTL.
Throttle: 1 request / second (independent of the NRL scraper throttle so
the two don't interfere, but gentle enough not to hit Open-Meteo rate limits).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_CACHE_TTL = timedelta(hours=6)
_MIN_INTERVAL_SECONDS = 1.0
_REQUEST_TIMEOUT = 10.0


@dataclass
class WeatherForecast:
    """Forecast for a single match venue / kickoff window."""

    rain_mm_3h: float  # total precipitation (mm) in the 3h window around kickoff
    wind_kph: float  # mean 10 m wind speed (km/h) in the same window
    temperature_c: float  # mean air temperature (°C) in the same window
    source: str  # always "open-meteo"
    fetched_at: datetime  # UTC timestamp when this forecast was retrieved


# --- module-level state (thread-safe) ----------------------------------------

_cache: dict[tuple[float, float, str], tuple[datetime, WeatherForecast | None]] = {}
_cache_lock = threading.Lock()

_last_request_at: float = 0.0
_throttle_lock = threading.Lock()


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL_SECONDS - (now - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


# --- public API --------------------------------------------------------------


def fetch_forecast(
    venue_lat: float,
    venue_lon: float,
    kickoff: datetime,
) -> WeatherForecast | None:
    """Return a weather forecast for *venue* at *kickoff* time, or ``None`` on failure.

    The caller should treat ``None`` the same as ``missing_weather=1.0`` —
    fall back to the ``missing`` sentinel and let the model use its intercept.

    *kickoff* may be timezone-aware or naive; only the date and hour are used
    for the Open-Meteo hourly query so timezone differences within ±12 h of
    kickoff don't affect the result.

    Results are cached for 6 hours per (lat, lon, kickoff_date).
    """
    kickoff_date_str = kickoff.date().isoformat()
    # Round coordinates to 4 dp (~11 m) to collapse near-identical venues.
    cache_key = (round(venue_lat, 4), round(venue_lon, 4), kickoff_date_str)
    now_utc = datetime.now(UTC)

    with _cache_lock:
        if cache_key in _cache:
            cached_at, result = _cache[cache_key]
            if now_utc - cached_at < _CACHE_TTL:
                return result

    _throttle()

    params = {
        "latitude": venue_lat,
        "longitude": venue_lon,
        "hourly": "rain,wind_speed_10m,temperature_2m",
        "timezone": "auto",
        "start_date": kickoff_date_str,
        "end_date": kickoff_date_str,
    }

    try:
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            resp = client.get(_OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.warning(
            "Open-Meteo fetch failed lat=%.4f lon=%.4f date=%s",
            venue_lat,
            venue_lon,
            kickoff_date_str,
            exc_info=True,
        )
        with _cache_lock:
            _cache[cache_key] = (now_utc, None)
        return None

    result = _parse_response(data, kickoff)

    with _cache_lock:
        _cache[cache_key] = (now_utc, result)

    return result


def bin_rain_mm_3h(mm: float) -> float:
    """Bin total 3-hour precipitation into a 4-level ordinal (0–3).

    Levels: 0 = dry (<0.5 mm), 1 = light (0.5–5 mm),
            2 = moderate (5–15 mm), 3 = heavy (>15 mm).

    Exported so ``feature_engineering`` can apply the same binning to
    forecast values without re-importing the full module.
    """
    if mm < 0.5:
        return 0.0
    if mm < 5.0:
        return 1.0
    if mm < 15.0:
        return 2.0
    return 3.0


# --- private helpers ---------------------------------------------------------


def _parse_response(data: dict, kickoff: datetime) -> WeatherForecast:
    """Extract values for the 3-hour window centred on kickoff."""
    hourly = data.get("hourly", {})
    times: list[str] = hourly.get("time", [])
    rain_vals: list[float | None] = hourly.get("rain", [])
    wind_vals: list[float | None] = hourly.get("wind_speed_10m", [])
    temp_vals: list[float | None] = hourly.get("temperature_2m", [])

    # Compare against a naive kickoff (Open-Meteo returns local naive datetimes).
    kickoff_naive = kickoff.replace(tzinfo=None) if kickoff.tzinfo else kickoff

    rain_window: list[float] = []
    wind_window: list[float] = []
    temp_window: list[float] = []

    for i, t_str in enumerate(times):
        t = datetime.fromisoformat(t_str)
        delta_h = (t - kickoff_naive).total_seconds() / 3600.0
        # 3-hour window: [kickoff-1h, kickoff+2h)
        if -1.0 <= delta_h < 2.0:
            if i < len(rain_vals) and rain_vals[i] is not None:
                rain_window.append(float(rain_vals[i]))  # type: ignore[arg-type]
            if i < len(wind_vals) and wind_vals[i] is not None:
                wind_window.append(float(wind_vals[i]))  # type: ignore[arg-type]
            if i < len(temp_vals) and temp_vals[i] is not None:
                temp_window.append(float(temp_vals[i]))  # type: ignore[arg-type]

    return WeatherForecast(
        rain_mm_3h=sum(rain_window),
        wind_kph=sum(wind_window) / len(wind_window) if wind_window else 0.0,
        temperature_c=sum(temp_window) / len(temp_window) if temp_window else 0.0,
        source="open-meteo",
        fetched_at=datetime.now(UTC),
    )
