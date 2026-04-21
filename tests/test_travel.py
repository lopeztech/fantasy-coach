"""Tests for travel & scheduling feature computation."""

from __future__ import annotations

import math

import pytest

from fantasy_coach.travel import (
    _is_brutal,
    _travel_km,
    _tz_delta,
    haversine_km,
    lookup_venue,
    travel_features,
)


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------


def test_haversine_same_point_is_zero() -> None:
    assert haversine_km(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=0.01)


def test_haversine_sydney_to_auckland_approx() -> None:
    # Sydney ↔ Auckland is ~2 155 km great-circle.
    km = haversine_km(-33.87, 151.21, -36.90, 174.78)
    assert 2100 < km < 2200


def test_haversine_sydney_to_perth_approx() -> None:
    # Sydney ↔ Perth is ~3 290 km great-circle.
    km = haversine_km(-33.87, 151.21, -31.95, 115.86)
    assert 3200 < km < 3400


def test_haversine_is_symmetric() -> None:
    a = haversine_km(-33.87, 151.21, -27.47, 153.02)
    b = haversine_km(-27.47, 153.02, -33.87, 151.21)
    assert a == pytest.approx(b, rel=1e-6)


# ---------------------------------------------------------------------------
# lookup_venue
# ---------------------------------------------------------------------------


def test_lookup_venue_returns_info_for_known_venue() -> None:
    info = lookup_venue("Suncorp Stadium")
    assert info is not None
    assert info.city == "Brisbane"
    assert info.tz_offset == 10


def test_lookup_venue_case_insensitive() -> None:
    assert lookup_venue("suncorp stadium") is not None
    assert lookup_venue("SUNCORP STADIUM") is not None


def test_lookup_venue_returns_none_for_unknown() -> None:
    assert lookup_venue("Mystery Ground") is None


def test_lookup_venue_returns_none_for_none_input() -> None:
    assert lookup_venue(None) is None


# ---------------------------------------------------------------------------
# _travel_km
# ---------------------------------------------------------------------------


def test_travel_km_zero_when_no_prev_venue() -> None:
    cur = lookup_venue("Suncorp Stadium")
    assert _travel_km(None, cur) == pytest.approx(0.0)


def test_travel_km_zero_when_no_cur_venue() -> None:
    assert _travel_km("Suncorp Stadium", None) == pytest.approx(0.0)


def test_travel_km_sydney_to_brisbane() -> None:
    cur = lookup_venue("Suncorp Stadium")  # Brisbane
    km = _travel_km("Allianz Stadium", cur)  # Sydney → Brisbane ≈ 735 km
    assert 650 < km < 800


def test_travel_km_sydney_to_auckland() -> None:
    cur = lookup_venue("Go Media Stadium")  # Auckland
    km = _travel_km("Allianz Stadium", cur)  # Sydney → Auckland ≈ 2 155 km
    assert 2000 < km < 2300


# ---------------------------------------------------------------------------
# _tz_delta
# ---------------------------------------------------------------------------


def test_tz_delta_same_timezone_is_zero() -> None:
    cur = lookup_venue("Suncorp Stadium")  # UTC+10
    assert _tz_delta("Allianz Stadium", cur) == 0  # also UTC+10


def test_tz_delta_sydney_to_perth() -> None:
    cur = lookup_venue("Optus Stadium")  # UTC+8
    delta = _tz_delta("Allianz Stadium", cur)  # UTC+10 → +8 = 2 h shift
    assert delta == 2


def test_tz_delta_sydney_to_auckland() -> None:
    cur = lookup_venue("Go Media Stadium")  # UTC+13
    delta = _tz_delta("Allianz Stadium", cur)  # UTC+10 → +13 = 3 h shift
    assert delta == 3


# ---------------------------------------------------------------------------
# _is_brutal
# ---------------------------------------------------------------------------


def test_is_brutal_short_rest_and_long_travel() -> None:
    assert _is_brutal(4.0, 1500.0) == 1


def test_is_brutal_long_rest() -> None:
    assert _is_brutal(7.0, 1500.0) == 0


def test_is_brutal_short_travel() -> None:
    assert _is_brutal(4.0, 500.0) == 0


def test_is_brutal_boundary_rest() -> None:
    # Exactly 6 days rest is NOT short (condition is < 6).
    assert _is_brutal(6.0, 2000.0) == 0


def test_is_brutal_boundary_travel() -> None:
    # Exactly 1 000 km is NOT long (condition is > 1 000).
    assert _is_brutal(3.0, 1000.0) == 0


# ---------------------------------------------------------------------------
# travel_features (integration)
# ---------------------------------------------------------------------------


def test_travel_features_all_zero_when_no_venues() -> None:
    km, tz, bb = travel_features(None, None, None, 7.0, 7.0)
    assert km == pytest.approx(0.0)
    assert tz == pytest.approx(0.0)
    assert bb == pytest.approx(0.0)


def test_travel_features_home_minus_away_sign() -> None:
    # Home team has travelled further (Auckland → Brisbane) than away (Sydney → Brisbane).
    km, _, _ = travel_features(
        "Go Media Stadium",   # Auckland (home came from)
        "Allianz Stadium",    # Sydney   (away came from)
        "Suncorp Stadium",    # Brisbane (current match)
        7.0,
        7.0,
    )
    # Auckland → Brisbane > Sydney → Brisbane, so home_km > away_km → positive diff
    assert km > 0


def test_travel_features_equal_travel_is_zero() -> None:
    km, tz, _ = travel_features(
        "Allianz Stadium",
        "Allianz Stadium",
        "Suncorp Stadium",
        7.0,
        7.0,
    )
    assert km == pytest.approx(0.0)
    assert tz == pytest.approx(0.0)


def test_travel_features_bb_flag_asymmetric() -> None:
    # Home team: short rest + long travel → brutal; Away: fine.
    _, _, bb = travel_features(
        "Go Media Stadium",   # Auckland → Brisbane (> 1 000 km)
        "Suncorp Stadium",    # Brisbane → Brisbane (0 km)
        "Suncorp Stadium",
        4.0,  # home: short rest
        7.0,  # away: long rest
    )
    assert bb == pytest.approx(1.0)  # home is suffering, away is not → +1
