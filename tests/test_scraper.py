from __future__ import annotations

import httpx
import pytest
import respx

from fantasy_coach import scraper


@pytest.fixture(autouse=True)
def reset_throttle() -> None:
    scraper._throttle._last_request_at = 0.0


@pytest.fixture(autouse=True)
def sleep_calls(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(scraper.time, "sleep", fake_sleep)
    return calls


@pytest.fixture(autouse=True)
def fast_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FANTASY_COACH_SCRAPE_INTERVAL_SECONDS", "0")


MATCH_URL = "https://www.nrl.com/draw/nrl-premiership/2026/round-8/wests-tigers-v-raiders/data"


@respx.mock
def test_fetch_match_returns_parsed_json() -> None:
    payload = {"matchId": 111, "homeTeam": {"nickName": "Wests Tigers"}}
    respx.get(MATCH_URL).mock(return_value=httpx.Response(200, json=payload))

    result = scraper.fetch_match(2026, 8, "wests-tigers", "raiders")

    assert result == payload


@respx.mock
def test_fetch_match_sends_expected_headers() -> None:
    route = respx.get(MATCH_URL).mock(return_value=httpx.Response(200, json={}))

    scraper.fetch_match(2026, 8, "wests-tigers", "raiders")

    request = route.calls.last.request
    assert "fantasy-coach" in request.headers["user-agent"]
    assert request.headers["accept"] == "application/json"


@respx.mock
def test_fetch_match_returns_none_on_404() -> None:
    respx.get(MATCH_URL).mock(return_value=httpx.Response(404))

    result = scraper.fetch_match(2026, 8, "wests-tigers", "raiders")

    assert result is None


@respx.mock
def test_fetch_match_retries_on_5xx_then_succeeds() -> None:
    payload = {"matchId": 222}
    route = respx.get(MATCH_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(502),
            httpx.Response(200, json=payload),
        ]
    )

    result = scraper.fetch_match(2026, 8, "wests-tigers", "raiders")

    assert result == payload
    assert route.call_count == 3


@respx.mock
def test_fetch_match_raises_when_5xx_exhausts_retries() -> None:
    route = respx.get(MATCH_URL).mock(return_value=httpx.Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        scraper.fetch_match(2026, 8, "wests-tigers", "raiders")

    assert route.call_count == scraper.DEFAULT_MAX_RETRIES


@respx.mock
def test_fetch_match_retries_on_network_error() -> None:
    payload = {"matchId": 333}
    route = respx.get(MATCH_URL).mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json=payload)]
    )

    result = scraper.fetch_match(2026, 8, "wests-tigers", "raiders")

    assert result == payload
    assert route.call_count == 2


@respx.mock
def test_fetch_match_network_error_exhausts() -> None:
    route = respx.get(MATCH_URL).mock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(httpx.ConnectError):
        scraper.fetch_match(2026, 8, "wests-tigers", "raiders")

    assert route.call_count == scraper.DEFAULT_MAX_RETRIES


@respx.mock
def test_fetch_match_respects_throttle_interval(
    monkeypatch: pytest.MonkeyPatch,
    sleep_calls: list[float],
) -> None:
    monkeypatch.setenv("FANTASY_COACH_SCRAPE_INTERVAL_SECONDS", "1.0")
    fake_now = [100.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    monkeypatch.setattr(scraper.time, "monotonic", fake_monotonic)
    respx.get(MATCH_URL).mock(return_value=httpx.Response(200, json={}))

    # First call: no prior request, no throttle sleep.
    scraper.fetch_match(2026, 8, "wests-tigers", "raiders")
    assert sleep_calls == []

    # Second call 0.3s later should sleep ~0.7s to reach the 1s interval.
    fake_now[0] = 100.3
    scraper.fetch_match(2026, 8, "wests-tigers", "raiders")
    throttle_sleeps = [s for s in sleep_calls if s > 0]
    assert len(throttle_sleeps) == 1
    assert throttle_sleeps[0] == pytest.approx(0.7, abs=1e-6)


@respx.mock
def test_fetch_match_uses_provided_client() -> None:
    respx.get(MATCH_URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    client = httpx.Client(base_url=scraper.BASE_URL, timeout=5.0)
    try:
        result = scraper.fetch_match(2026, 8, "wests-tigers", "raiders", client=client)
    finally:
        client.close()
    assert result == {"ok": True}


@respx.mock
def test_fetch_round_returns_payload() -> None:
    payload = {"fixtures": [{"matchCentreUrl": "/draw/x/"}], "byes": []}
    respx.get(
        "https://www.nrl.com/draw/data",
        params={"competition": "111", "round": "8", "season": "2026"},
    ).mock(return_value=httpx.Response(200, json=payload))

    result = scraper.fetch_round(2026, 8)

    assert result == payload


@respx.mock
def test_fetch_round_returns_none_on_404() -> None:
    respx.get("https://www.nrl.com/draw/data").mock(return_value=httpx.Response(404))

    assert scraper.fetch_round(2026, 99) is None


@respx.mock
def test_fetch_match_from_url_appends_data_segment() -> None:
    finals_url = "/draw/nrl-premiership/2024/finals-week-1/game-1/"
    respx.get(f"https://www.nrl.com{finals_url}data").mock(
        return_value=httpx.Response(200, json={"matchId": 1})
    )

    result = scraper.fetch_match_from_url(finals_url)

    assert result == {"matchId": 1}


@respx.mock
def test_fetch_match_from_url_accepts_full_url() -> None:
    full = "https://www.nrl.com/draw/nrl-premiership/2024/round-1/sea-eagles-v-rabbitohs/"
    respx.get(full + "data").mock(return_value=httpx.Response(200, json={"ok": True}))

    assert scraper.fetch_match_from_url(full) == {"ok": True}
