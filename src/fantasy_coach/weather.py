"""Weather & venue-condition feature extraction.

Weather data on NRL.com:
- 2026+: a structured JSON object with keys `description`, `tempInCelsius`,
  `windSpeedKm`, and `isWet`.
- 2024–2025: field absent (null) — gracefully degrade to zeros with a
  `missing_weather` flag so the model can learn a separate intercept.

The `parse_weather` function handles both representations and plain-text
fall-backs (e.g. "Rain, 15°C") for robustness.

Decision (null strategy): use a dedicated `missing_weather` flag rather than
imputing from venue/seasonal averages.  Imputation would silently introduce
synthetic correlations; an explicit flag lets the model learn "no weather info
available" as a distinct case.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)

_WET_KEYWORDS = re.compile(
    r"\b(rain|wet|shower|storm|thunder|thunderstorm|drizzle|hail|sleet)\b", re.IGNORECASE
)


class WeatherInfo(NamedTuple):
    """Parsed, null-safe weather values for a single match."""

    is_wet: float  # 1.0 if wet/rainy conditions, else 0.0
    wind_kph: float  # wind speed (0.0 if unknown)
    temperature_c: float  # temperature in Celsius (0.0 if unknown)
    missing: float  # 1.0 if source data was absent


_MISSING = WeatherInfo(is_wet=0.0, wind_kph=0.0, temperature_c=0.0, missing=1.0)
_ZERO = WeatherInfo(is_wet=0.0, wind_kph=0.0, temperature_c=0.0, missing=0.0)


def parse_weather(raw: str | dict | None) -> WeatherInfo:
    """Parse a raw weather value into numeric features.

    Handles:
    - ``None`` / empty: returns the ``missing`` sentinel (all zeros + flag=1).
    - ``dict``: structured NRL 2026+ format.
    - ``str`` that is valid JSON: parsed as dict.
    - ``str`` plain-text: keyword + numeric regex extraction (best-effort).
    """
    if raw is None:
        return _MISSING

    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return _MISSING
        # Try to decode as JSON first.
        if raw.startswith("{"):
            with contextlib.suppress(json.JSONDecodeError):
                raw = json.loads(raw)

    if isinstance(raw, dict):
        return _from_dict(raw)

    # Plain text fall-back.
    return _from_text(str(raw))


def _from_dict(d: dict) -> WeatherInfo:
    is_wet = float(bool(d.get("isWet") or _WET_KEYWORDS.search(str(d.get("description", "")))))
    wind_kph = float(d.get("windSpeedKm") or d.get("wind_kph") or 0.0)
    temperature_c = float(d.get("tempInCelsius") or d.get("temperature_c") or 0.0)
    return WeatherInfo(is_wet=is_wet, wind_kph=wind_kph, temperature_c=temperature_c, missing=0.0)


def _from_text(text: str) -> WeatherInfo:
    is_wet = 1.0 if _WET_KEYWORDS.search(text) else 0.0

    # Temperature: looks for "22°C" or "22 C" or "22 degrees"
    temp_match = re.search(r"(-?\d+(?:\.\d+)?)\s*[°º]?\s*[Cc](?:\b|elsius|\s|$)", text)
    temperature_c = float(temp_match.group(1)) if temp_match else 0.0

    # Wind: looks for "20 km/h", "20 kph", "20kmh", "20 KPH"
    wind_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km/?h|kph|kmh)\b", text, re.IGNORECASE)
    wind_kph = float(wind_match.group(1)) if wind_match else 0.0

    return WeatherInfo(is_wet=is_wet, wind_kph=wind_kph, temperature_c=temperature_c, missing=0.0)
