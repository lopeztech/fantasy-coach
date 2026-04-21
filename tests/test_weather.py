"""Tests for weather parsing and venue rolling stats integration."""

from __future__ import annotations

from datetime import UTC

import pytest

from fantasy_coach.weather import _MISSING, WeatherInfo, parse_weather

# ---------------------------------------------------------------------------
# parse_weather — null / missing inputs
# ---------------------------------------------------------------------------


def test_parse_none_returns_missing() -> None:
    result = parse_weather(None)
    assert result == _MISSING
    assert result.missing == 1.0


def test_parse_empty_string_returns_missing() -> None:
    assert parse_weather("") == _MISSING
    assert parse_weather("   ") == _MISSING


# ---------------------------------------------------------------------------
# parse_weather — structured dict (NRL 2026+ format)
# ---------------------------------------------------------------------------


def test_parse_dict_wet_conditions() -> None:
    raw = {"isWet": True, "tempInCelsius": 18, "windSpeedKm": 25, "description": "Heavy Rain"}
    result = parse_weather(raw)
    assert result.is_wet == 1.0
    assert result.temperature_c == pytest.approx(18.0)
    assert result.wind_kph == pytest.approx(25.0)
    assert result.missing == 0.0


def test_parse_dict_dry_conditions() -> None:
    raw = {"isWet": False, "tempInCelsius": 24, "windSpeedKm": 10, "description": "Fine"}
    result = parse_weather(raw)
    assert result.is_wet == 0.0
    assert result.temperature_c == pytest.approx(24.0)
    assert result.wind_kph == pytest.approx(10.0)


def test_parse_dict_wet_via_description_keyword() -> None:
    # isWet missing but description says "shower"
    raw = {"tempInCelsius": 15, "windSpeedKm": 0, "description": "Shower expected"}
    result = parse_weather(raw)
    assert result.is_wet == 1.0


def test_parse_dict_missing_fields_default_to_zero() -> None:
    result = parse_weather({"description": "Fine"})
    assert result.wind_kph == 0.0
    assert result.temperature_c == 0.0
    assert result.missing == 0.0


# ---------------------------------------------------------------------------
# parse_weather — JSON string (dict serialised as string)
# ---------------------------------------------------------------------------


def test_parse_json_string() -> None:
    import json

    raw = json.dumps({"isWet": True, "tempInCelsius": 12, "windSpeedKm": 35})
    result = parse_weather(raw)
    assert result.is_wet == 1.0
    assert result.temperature_c == pytest.approx(12.0)
    assert result.wind_kph == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# parse_weather — plain-text fallback
# ---------------------------------------------------------------------------


def test_parse_text_fine() -> None:
    result = parse_weather("Fine")
    assert result.is_wet == 0.0
    assert result.missing == 0.0


def test_parse_text_rain() -> None:
    result = parse_weather("Rain, 15°C, 20 km/h")
    assert result.is_wet == 1.0
    assert result.temperature_c == pytest.approx(15.0)
    assert result.wind_kph == pytest.approx(20.0)


def test_parse_text_storm() -> None:
    assert parse_weather("Thunderstorm").is_wet == 1.0


def test_parse_text_wet_field() -> None:
    assert parse_weather("Wet conditions").is_wet == 1.0


def test_parse_text_no_temp_or_wind() -> None:
    result = parse_weather("Partly Cloudy")
    assert result.temperature_c == 0.0
    assert result.wind_kph == 0.0
    assert result.is_wet == 0.0


def test_parse_text_negative_temperature() -> None:
    result = parse_weather("Snow, -2°C, 10 km/h")
    assert result.is_wet == 0.0  # "snow" not in wet keywords
    assert result.temperature_c == pytest.approx(-2.0)


def test_parse_text_celsius_variants() -> None:
    assert parse_weather("22 C clear").temperature_c == pytest.approx(22.0)
    assert parse_weather("22°C clear").temperature_c == pytest.approx(22.0)


def test_parse_text_wind_variants() -> None:
    assert parse_weather("15 km/h breeze").wind_kph == pytest.approx(15.0)
    assert parse_weather("15 kph").wind_kph == pytest.approx(15.0)
    assert parse_weather("15 kmh").wind_kph == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# WeatherInfo helpers
# ---------------------------------------------------------------------------


def test_weather_info_is_named_tuple() -> None:
    w = WeatherInfo(is_wet=1.0, wind_kph=20.0, temperature_c=18.0, missing=0.0)
    assert w.is_wet == 1.0
    assert w[0] == 1.0  # index access (NamedTuple)


# ---------------------------------------------------------------------------
# Integration: venue stats in FeatureBuilder
# ---------------------------------------------------------------------------


def test_venue_stats_start_at_zero() -> None:
    from datetime import datetime

    from fantasy_coach.feature_engineering import FeatureBuilder
    from fantasy_coach.features import MatchRow, TeamRow

    def _team(tid: int, score: int | None = None) -> TeamRow:
        return TeamRow(
            team_id=tid,
            name=f"Team{tid}",
            nick_name=f"T{tid}",
            score=score,
            players=[],
        )

    match = MatchRow(
        match_id=1,
        season=2025,
        round=1,
        start_time=datetime(2025, 3, 1, 19, 0, tzinfo=UTC),
        match_state="FullTime",
        venue="Suncorp Stadium",
        venue_city="Brisbane",
        weather=None,
        home=_team(1, 20),
        away=_team(2, 12),
        team_stats=[],
    )

    builder = FeatureBuilder()
    row = builder.feature_row(match)

    # Indices 13 and 14 are venue_avg_total_points and venue_home_win_rate
    # Before any history, venue_avg_total_points = 0, venue_home_win_rate = 0.5
    assert row[13] == pytest.approx(0.0)
    assert row[14] == pytest.approx(0.5)


def test_venue_stats_update_after_record() -> None:
    from datetime import datetime

    from fantasy_coach.feature_engineering import FeatureBuilder
    from fantasy_coach.features import MatchRow, TeamRow

    def _team(tid: int, score: int | None = None) -> TeamRow:
        return TeamRow(
            team_id=tid,
            name=f"Team{tid}",
            nick_name=f"T{tid}",
            score=score,
            players=[],
        )

    def _match(mid: int, ts: datetime, hscore: int, ascore: int) -> MatchRow:
        return MatchRow(
            match_id=mid,
            season=2025,
            round=1,
            start_time=ts,
            match_state="FullTime",
            venue="Suncorp Stadium",
            venue_city="Brisbane",
            weather=None,
            home=_team(1, hscore),
            away=_team(2, ascore),
            team_stats=[],
        )

    builder = FeatureBuilder()
    t0 = datetime(2025, 3, 1, 19, 0, tzinfo=UTC)
    t1 = datetime(2025, 3, 8, 19, 0, tzinfo=UTC)
    t2 = datetime(2025, 3, 15, 19, 0, tzinfo=UTC)

    m1 = _match(1, t0, 30, 10)  # total=40, home win
    m2 = _match(2, t1, 12, 20)  # total=32, away win

    builder.record(m1)
    builder.record(m2)

    m3 = _match(3, t2, 20, 18)
    row = builder.feature_row(m3)

    # After 2 matches: avg total = (40 + 32) / 2 = 36
    assert row[13] == pytest.approx(36.0)
    # home_win_rate = 1/2 = 0.5
    assert row[14] == pytest.approx(0.5)


def test_weather_features_wired_into_feature_row() -> None:
    from datetime import datetime

    from fantasy_coach.feature_engineering import FeatureBuilder
    from fantasy_coach.features import MatchRow, TeamRow

    def _team(tid: int) -> TeamRow:
        return TeamRow(team_id=tid, name=f"T{tid}", nick_name=f"T{tid}", score=None, players=[])

    import json

    match = MatchRow(
        match_id=9,
        season=2025,
        round=5,
        start_time=datetime(2025, 5, 1, tzinfo=UTC),
        match_state="Upcoming",
        venue="Suncorp Stadium",
        venue_city="Brisbane",
        weather=json.dumps({"isWet": True, "tempInCelsius": 16, "windSpeedKm": 30}),
        home=_team(1),
        away=_team(2),
        team_stats=[],
    )

    builder = FeatureBuilder()
    row = builder.feature_row(match)

    # Indices: 9=is_wet, 10=wind_kph, 11=temperature_c, 12=missing_weather
    assert row[9] == pytest.approx(1.0)  # is_wet
    assert row[10] == pytest.approx(30.0)  # wind_kph
    assert row[11] == pytest.approx(16.0)  # temperature_c
    assert row[12] == pytest.approx(0.0)  # missing_weather
