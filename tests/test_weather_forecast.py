"""Tests for weather_forecast module (#207).

All HTTP calls are mocked so tests are hermetic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from fantasy_coach.weather_forecast import (
    WeatherForecast,
    _parse_response,
    bin_rain_mm_3h,
    fetch_forecast,
)

# ---------------------------------------------------------------------------
# bin_rain_mm_3h
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mm", "expected"),
    [
        (0.0, 0.0),
        (0.4, 0.0),
        (0.5, 1.0),
        (4.9, 1.0),
        (5.0, 2.0),
        (14.9, 2.0),
        (15.0, 3.0),
        (50.0, 3.0),
    ],
)
def test_bin_rain_mm_3h(mm: float, expected: float) -> None:
    assert bin_rain_mm_3h(mm) == expected


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

_HOURLY_PAYLOAD = {
    "hourly": {
        "time": [
            "2026-05-01T18:00",
            "2026-05-01T19:00",
            "2026-05-01T20:00",
            "2026-05-01T21:00",
        ],
        "rain": [0.0, 3.0, 5.0, 0.0],
        "wind_speed_10m": [10.0, 20.0, 25.0, 15.0],
        "temperature_2m": [20.0, 18.0, 17.0, 16.0],
    }
}


def test_parse_response_window() -> None:
    # kickoff at 19:50 → window is [18:50, 21:50) → hours 19, 20, 21
    kickoff = datetime(2026, 5, 1, 19, 50)
    result = _parse_response(_HOURLY_PAYLOAD, kickoff)
    # hours in window: 19 (delta=-0.83h), 20 (delta=0.17h), 21 (delta=1.17h)
    assert result.rain_mm_3h == pytest.approx(3.0 + 5.0 + 0.0)
    assert result.wind_kph == pytest.approx((20.0 + 25.0 + 15.0) / 3)
    assert result.temperature_c == pytest.approx((18.0 + 17.0 + 16.0) / 3)
    assert result.source == "open-meteo"
    assert isinstance(result.fetched_at, datetime)


def test_parse_response_no_matching_hours() -> None:
    # kickoff at 08:00 — none of the 18-21 hours fall in window
    kickoff = datetime(2026, 5, 1, 8, 0)
    result = _parse_response(_HOURLY_PAYLOAD, kickoff)
    assert result.rain_mm_3h == 0.0
    assert result.wind_kph == 0.0
    assert result.temperature_c == 0.0


def test_parse_response_handles_none_values() -> None:
    payload = {
        "hourly": {
            "time": ["2026-05-01T19:00"],
            "rain": [None],
            "wind_speed_10m": [None],
            "temperature_2m": [None],
        }
    }
    kickoff = datetime(2026, 5, 1, 19, 30)
    result = _parse_response(payload, kickoff)
    assert result.rain_mm_3h == 0.0
    assert result.wind_kph == 0.0
    assert result.temperature_c == 0.0


# ---------------------------------------------------------------------------
# fetch_forecast — mocked HTTP
# ---------------------------------------------------------------------------


def _make_mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_forecast_success() -> None:
    kickoff = datetime(2026, 5, 1, 19, 50, tzinfo=UTC)
    with (
        patch("fantasy_coach.weather_forecast._cache", {}),
        patch("fantasy_coach.weather_forecast._last_request_at", 0.0),
        patch("httpx.Client") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.get.return_value = _make_mock_response(_HOURLY_PAYLOAD)

        result = fetch_forecast(-33.85, 151.06, kickoff)

    assert isinstance(result, WeatherForecast)
    assert result.source == "open-meteo"
    assert result.rain_mm_3h >= 0.0


def test_fetch_forecast_http_error_returns_none() -> None:
    kickoff = datetime(2026, 5, 1, 19, 50, tzinfo=UTC)
    with (
        patch("fantasy_coach.weather_forecast._cache", {}),
        patch("fantasy_coach.weather_forecast._last_request_at", 0.0),
        patch("httpx.Client") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client
        mock_client.get.side_effect = Exception("network error")

        result = fetch_forecast(-33.85, 151.06, kickoff)

    assert result is None


def test_fetch_forecast_cache_hit() -> None:
    kickoff = datetime(2026, 5, 2, 19, 50, tzinfo=UTC)
    cached_forecast = WeatherForecast(
        rain_mm_3h=5.0,
        wind_kph=20.0,
        temperature_c=18.0,
        source="open-meteo",
        fetched_at=datetime.now(UTC),
    )
    cache_key = (round(-33.85, 4), round(151.06, 4), "2026-05-02")

    with (
        patch(
            "fantasy_coach.weather_forecast._cache",
            {cache_key: (datetime.now(UTC), cached_forecast)},
        ),
        patch("httpx.Client") as mock_client_cls,
    ):
        result = fetch_forecast(-33.85, 151.06, kickoff)
        mock_client_cls.assert_not_called()

    assert result is cached_forecast


# ---------------------------------------------------------------------------
# Two-source merge in feature_engineering
# ---------------------------------------------------------------------------


def _make_match(weather: str | None):  # type: ignore[return]
    """Create a minimal MatchRow for feature_row testing."""
    from fantasy_coach.features import MatchRow, TeamRow  # noqa: PLC0415

    when = datetime(2026, 5, 1, 19, 50, tzinfo=UTC)
    return MatchRow(
        match_id=99,
        season=2026,
        round=9,
        start_time=when,
        match_state="Upcoming",
        venue=None,
        venue_city=None,
        weather=weather,
        home=TeamRow(team_id=1, name="Home", nick_name="H", score=None, players=[]),
        away=TeamRow(team_id=2, name="Away", nick_name="A", score=None, players=[]),
        team_stats=[],
    )


def test_feature_row_uses_forecast_when_weather_missing() -> None:
    """When match.weather is None, forecast values appear in the feature row."""
    from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder  # noqa: PLC0415

    builder = FeatureBuilder()
    match = _make_match(weather=None)
    forecast = WeatherForecast(
        rain_mm_3h=8.0,  # moderate → intensity 2
        wind_kph=30.0,
        temperature_c=15.0,
        source="open-meteo",
        fetched_at=datetime.now(UTC),
    )
    row = dict(
        zip(FEATURE_NAMES, builder.feature_row(match, weather_forecast=forecast), strict=True)
    )

    assert row["is_wet"] == 1.0
    assert row["wind_kph"] == pytest.approx(30.0)
    assert row["temperature_c"] == pytest.approx(15.0)
    assert row["missing_weather"] == 0.0
    assert row["rain_intensity"] == 2.0  # moderate bin
    assert row["weather_source"] == 1.0  # forecast


def test_feature_row_actual_weather_takes_priority() -> None:
    """When match.weather has NRL data, it takes priority over any forecast."""
    from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder  # noqa: PLC0415

    builder = FeatureBuilder()
    match = _make_match(weather='{"isWet": false, "windSpeedKm": 10, "tempInCelsius": 22}')
    forecast = WeatherForecast(
        rain_mm_3h=20.0,
        wind_kph=50.0,
        temperature_c=5.0,
        source="open-meteo",
        fetched_at=datetime.now(UTC),
    )
    row = dict(
        zip(FEATURE_NAMES, builder.feature_row(match, weather_forecast=forecast), strict=True)
    )

    assert row["is_wet"] == 0.0  # from NRL actual
    assert row["wind_kph"] == pytest.approx(10.0)
    assert row["temperature_c"] == pytest.approx(22.0)
    assert row["weather_source"] == 0.0  # actual, not forecast
    assert row["rain_intensity"] == 0.0  # dry from is_wet=False


def test_feature_row_no_forecast_no_weather_is_missing() -> None:
    """When neither actual nor forecast is available, missing_weather=1.0."""
    from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder  # noqa: PLC0415

    builder = FeatureBuilder()
    match = _make_match(weather=None)
    row = dict(zip(FEATURE_NAMES, builder.feature_row(match, weather_forecast=None), strict=True))

    assert row["missing_weather"] == 1.0
    assert row["rain_intensity"] == 0.0
    assert row["weather_source"] == 0.0
