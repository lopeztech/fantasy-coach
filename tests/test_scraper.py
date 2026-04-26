from __future__ import annotations

import httpx
import pytest
import respx

from fantasy_coach import scraper
from fantasy_coach.scraper_cache import SQLiteScraperCache


@pytest.fixture()
def mem_cache() -> SQLiteScraperCache:
    """In-memory SQLite scraper cache for tests."""
    return SQLiteScraperCache(":memory:")


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


# ---------------------------------------------------------------------------
# HTTP caching tests
# ---------------------------------------------------------------------------


@respx.mock
def test_cache_first_fetch_stores_etag(mem_cache: SQLiteScraperCache) -> None:
    """First 200 with an ETag stores it in the cache for the next request."""
    payload = {"matchId": 1}
    respx.get(MATCH_URL).mock(
        return_value=httpx.Response(200, json=payload, headers={"ETag": '"abc123"'})
    )

    result = scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)

    assert result == payload
    entry = mem_cache.get("/draw/nrl-premiership/2026/round-8/wests-tigers-v-raiders/data")
    assert entry is not None
    assert entry.etag == '"abc123"'
    assert entry.last_status == 200


@respx.mock
def test_cache_second_fetch_sends_if_none_match(mem_cache: SQLiteScraperCache) -> None:
    """Second fetch for the same URL sends If-None-Match; 304 returns None."""
    payload = {"matchId": 1}
    route = respx.get(MATCH_URL).mock(
        side_effect=[
            httpx.Response(200, json=payload, headers={"ETag": '"v1"'}),
            httpx.Response(304),
        ]
    )

    # First fetch — populates cache.
    first = scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)
    assert first == payload

    # Second fetch — should send If-None-Match and return None on 304.
    second = scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)
    assert second is None

    # The 304 request must have sent the conditional header.
    second_request = route.calls[1].request
    assert second_request.headers.get("if-none-match") == '"v1"'

    # Cache entry must still be present (last_fetched_at updated).
    entry = mem_cache.get("/draw/nrl-premiership/2026/round-8/wests-tigers-v-raiders/data")
    assert entry is not None
    assert entry.etag == '"v1"'
    assert entry.last_status == 304


@respx.mock
def test_cache_content_hash_fallback_unchanged(mem_cache: SQLiteScraperCache) -> None:
    """For endpoints without validators, content-hash dedup returns None on match."""
    body = b'{"matchId": 99}'
    respx.get(MATCH_URL).mock(
        return_value=httpx.Response(200, content=body, headers={"Content-Type": "application/json"})
    )

    # First fetch — stores content hash.
    first = scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)
    assert first == {"matchId": 99}

    # Second fetch — same body, no ETag/Last-Modified → hash match → None.
    second = scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)
    assert second is None


@respx.mock
def test_cache_etag_mismatch_returns_new_body(mem_cache: SQLiteScraperCache) -> None:
    """When the server returns a new ETag + new body, cache is updated and body returned."""
    route = respx.get(MATCH_URL).mock(
        side_effect=[
            httpx.Response(200, json={"version": 1}, headers={"ETag": '"v1"'}),
            httpx.Response(200, json={"version": 2}, headers={"ETag": '"v2"'}),
        ]
    )

    first = scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)
    assert first == {"version": 1}

    second = scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)
    assert second == {"version": 2}

    # Second request must have sent If-None-Match with old ETag.
    assert route.calls[1].request.headers.get("if-none-match") == '"v1"'

    # Cache entry must have the new ETag.
    entry = mem_cache.get("/draw/nrl-premiership/2026/round-8/wests-tigers-v-raiders/data")
    assert entry is not None
    assert entry.etag == '"v2"'


@respx.mock
def test_cache_no_store_skips_caching(mem_cache: SQLiteScraperCache) -> None:
    """Cache-Control: no-store responses are not cached (no conditional headers on retry)."""
    route = respx.get(MATCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={"live": True},
            headers={"ETag": '"live"', "Cache-Control": "no-store"},
        )
    )

    scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)
    scraper.fetch_match(2026, 8, "wests-tigers", "raiders", cache=mem_cache)

    # Both requests must NOT have sent If-None-Match.
    for call in route.calls:
        assert "if-none-match" not in call.request.headers


@respx.mock
def test_cache_without_cache_param_behaves_as_before() -> None:
    """Callers that don't pass cache= get the original behaviour (no cache interaction)."""
    payload = {"matchId": 5}
    respx.get(MATCH_URL).mock(return_value=httpx.Response(200, json=payload))

    assert scraper.fetch_match(2026, 8, "wests-tigers", "raiders") == payload
