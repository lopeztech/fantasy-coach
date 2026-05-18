"""Tests for ``betting.odds_client.OddsApiClient``."""

from __future__ import annotations

import httpx
import pytest
import respx

from fantasy_coach.betting import odds_client
from fantasy_coach.betting.odds_client import API_BASE_URL, SPORT_KEY, OddsApiClient


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Don't actually sleep between retries during tests."""
    monkeypatch.setattr(odds_client.time, "sleep", lambda _s: None)


def _sample_event(home: str, away: str, line: float = 40.5) -> dict:
    return {
        "id": "abc",
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": "tab",
                "title": "TAB",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": line, "price": 1.90},
                            {"name": "Under", "point": line, "price": 1.90},
                        ],
                    },
                ],
            }
        ],
    }


@respx.mock
def test_fetch_totals_returns_keyed_map() -> None:
    payload = [_sample_event("Melbourne Storm", "Parramatta Eels", 41.5)]
    respx.get(f"{API_BASE_URL}/sports/{SPORT_KEY}/odds").mock(
        return_value=httpx.Response(200, json=payload)
    )

    out = OddsApiClient(api_key="secret").fetch_totals()

    assert ("Storm", "Eels") in out
    line = out[("Storm", "Eels")]
    assert line.line == 41.5
    assert line.overDecimalOdds == 1.90
    assert line.underDecimalOdds == 1.90
    assert line.bookmaker == "TAB"


@respx.mock
def test_fetch_totals_sends_expected_query_params() -> None:
    route = respx.get(f"{API_BASE_URL}/sports/{SPORT_KEY}/odds").mock(
        return_value=httpx.Response(200, json=[])
    )

    OddsApiClient(api_key="secret").fetch_totals()

    sent = route.calls.last.request
    assert "apiKey=secret" in sent.url.query.decode()
    assert "regions=au" in sent.url.query.decode()
    assert "markets=h2h%2Ctotals" in sent.url.query.decode()  # comma is percent-encoded


@respx.mock
def test_fetch_totals_skips_unknown_teams() -> None:
    payload = [_sample_event("Some Touring Side", "Another Touring Side")]
    respx.get(f"{API_BASE_URL}/sports/{SPORT_KEY}/odds").mock(
        return_value=httpx.Response(200, json=payload)
    )

    assert OddsApiClient(api_key="secret").fetch_totals() == {}


@respx.mock
def test_fetch_totals_returns_empty_on_persistent_5xx() -> None:
    """Non-fatal — caller falls back to H2H-only card."""
    respx.get(f"{API_BASE_URL}/sports/{SPORT_KEY}/odds").mock(return_value=httpx.Response(503))

    out = OddsApiClient(api_key="secret", max_retries=2).fetch_totals()

    assert out == {}


@respx.mock
def test_fetch_totals_retries_on_transient_5xx_then_succeeds() -> None:
    payload = [_sample_event("Penrith Panthers", "Sydney Roosters", 39.5)]
    route = respx.get(f"{API_BASE_URL}/sports/{SPORT_KEY}/odds").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=payload),
        ]
    )

    out = OddsApiClient(api_key="secret", max_retries=3).fetch_totals()

    assert ("Panthers", "Roosters") in out
    assert route.call_count == 2


@respx.mock
def test_skips_bookmaker_without_totals_market() -> None:
    payload = [
        {
            "home_team": "Brisbane Broncos",
            "away_team": "Gold Coast Titans",
            "bookmakers": [
                {
                    "key": "x",
                    "title": "X",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Broncos", "price": 1.50}]}],
                }
            ],
        }
    ]
    respx.get(f"{API_BASE_URL}/sports/{SPORT_KEY}/odds").mock(
        return_value=httpx.Response(200, json=payload)
    )

    assert OddsApiClient(api_key="k").fetch_totals() == {}
