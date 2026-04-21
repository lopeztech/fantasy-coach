"""Tests for ResponseCache, TokenBudget, and CachingGeminiClient."""

from __future__ import annotations

import logging
import time

import pytest

from fantasy_coach.commentary.cache import (
    BudgetExceededError,
    CachingGeminiClient,
    ResponseCache,
    TokenBudget,
    _cache_key,
)
from fantasy_coach.commentary.client import GeminiClient, GeminiResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_RESPONSE = GeminiResponse(text="Panthers win by 10.", input_tokens=40, output_tokens=8)
MODEL = "gemini-2.0-flash-001"
PROJECT = "test-project"


def _make_client(call_count_ref: list[int] | None = None) -> GeminiClient:
    """Return a GeminiClient whose generate() is monkeypatched to avoid HTTP."""
    client = GeminiClient.__new__(GeminiClient)
    client._project = PROJECT
    client._model = MODEL
    client._location = "us-central1"
    client._max_retries = 3
    client._timeout = 30.0
    client._endpoint = "https://mock-endpoint"

    _calls = call_count_ref if call_count_ref is not None else []

    def fake_generate(
        prompt: str, *, system: str = "", max_output_tokens: int = 512
    ) -> GeminiResponse:
        _calls.append(1)
        return FAKE_RESPONSE

    client.generate = fake_generate  # type: ignore[method-assign]
    return client


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------


def test_cache_key_changes_with_model() -> None:
    assert _cache_key("same prompt", "model-a") != _cache_key("same prompt", "model-b")


def test_cache_key_changes_with_prompt() -> None:
    assert _cache_key("prompt-a", MODEL) != _cache_key("prompt-b", MODEL)


def test_cache_key_stable() -> None:
    assert _cache_key("hello", MODEL) == _cache_key("hello", MODEL)


# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------


def test_cache_miss_on_first_call() -> None:
    cache = ResponseCache()
    assert cache.get("some prompt", MODEL) is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_hit_after_set() -> None:
    cache = ResponseCache()
    cache.set("prompt", MODEL, FAKE_RESPONSE)
    result = cache.get("prompt", MODEL)
    assert result == FAKE_RESPONSE
    assert cache.hits == 1


def test_cache_expired_entry_is_a_miss() -> None:
    cache = ResponseCache(default_ttl=0.01)
    cache.set("prompt", MODEL, FAKE_RESPONSE)
    time.sleep(0.02)
    assert cache.get("prompt", MODEL) is None
    assert cache.misses == 1


def test_cache_ttl_override_per_set() -> None:
    cache = ResponseCache(default_ttl=3600)
    cache.set("prompt", MODEL, FAKE_RESPONSE, ttl=0.01)
    time.sleep(0.02)
    assert cache.get("prompt", MODEL) is None


def test_cache_hit_rate_is_zero_with_no_calls() -> None:
    assert ResponseCache().hit_rate == 0.0


def test_cache_hit_rate_after_mixed_calls() -> None:
    cache = ResponseCache()
    cache.set("p", MODEL, FAKE_RESPONSE)
    cache.get("p", MODEL)  # hit
    cache.get("nope", MODEL)  # miss
    assert cache.hit_rate == pytest.approx(0.5)


def test_cache_hit_logs_hit_rate(caplog: pytest.LogCaptureFixture) -> None:
    cache = ResponseCache()
    cache.set("p", MODEL, FAKE_RESPONSE)
    with caplog.at_level(logging.INFO, logger="fantasy_coach.commentary.cache"):
        cache.get("p", MODEL)
    assert "gemini_cache" in caplog.text
    assert "status=hit" in caplog.text


def test_cache_invalidate_removes_entry() -> None:
    cache = ResponseCache()
    cache.set("p", MODEL, FAKE_RESPONSE)
    cache.invalidate("p", MODEL)
    assert cache.get("p", MODEL) is None


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


def test_budget_allows_request_within_limit() -> None:
    budget = TokenBudget(per_request_max=512, daily_limit=10_000)
    budget.check_request(256)  # should not raise


def test_budget_raises_on_per_request_exceeded() -> None:
    budget = TokenBudget(per_request_max=256)
    with pytest.raises(BudgetExceededError, match="per-request"):
        budget.check_request(512)


def test_budget_raises_on_daily_limit_exceeded() -> None:
    budget = TokenBudget(per_request_max=512, daily_limit=100)
    budget.record_usage(90)
    with pytest.raises(BudgetExceededError, match="Daily token budget"):
        budget.check_request(20)


def test_budget_record_usage_accumulates() -> None:
    budget = TokenBudget(daily_limit=1000)
    budget.record_usage(300)
    budget.record_usage(200)
    assert budget.used == 500


def test_budget_logs_usage(caplog: pytest.LogCaptureFixture) -> None:
    budget = TokenBudget(daily_limit=1000)
    with caplog.at_level(logging.INFO, logger="fantasy_coach.commentary.cache"):
        budget.record_usage(100)
    assert "gemini_budget" in caplog.text
    assert "used=100" in caplog.text


def test_budget_window_resets_after_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    import fantasy_coach.commentary.cache as _mod

    now = time.time()
    monkeypatch.setattr(_mod.time, "time", lambda: now)

    budget = TokenBudget(daily_limit=1000)
    budget.record_usage(900)
    assert budget.used == 900

    # Advance clock 25 hours.
    monkeypatch.setattr(_mod.time, "time", lambda: now + 25 * 3600)
    assert budget.used == 0  # reset on access


# ---------------------------------------------------------------------------
# CachingGeminiClient — cache hit path
# ---------------------------------------------------------------------------


def test_caching_client_cache_miss_calls_underlying() -> None:
    calls: list[int] = []
    client = _make_client(calls)
    caching = CachingGeminiClient(client)

    result = caching.generate("Who wins?")

    assert result == FAKE_RESPONSE
    assert len(calls) == 1


def test_caching_client_cache_hit_skips_underlying() -> None:
    calls: list[int] = []
    client = _make_client(calls)
    caching = CachingGeminiClient(client)

    first = caching.generate("Who wins?")
    second = caching.generate("Who wins?")

    assert first == second
    assert len(calls) == 1  # underlying called only once


def test_caching_client_different_prompts_are_separate_entries() -> None:
    calls: list[int] = []
    client = _make_client(calls)
    caching = CachingGeminiClient(client)

    caching.generate("Who wins match A?")
    caching.generate("Who wins match B?")

    assert len(calls) == 2


def test_caching_client_system_prompt_is_part_of_cache_key() -> None:
    calls: list[int] = []
    client = _make_client(calls)
    caching = CachingGeminiClient(client)

    caching.generate("Who wins?", system="Be brief.")
    caching.generate("Who wins?", system="Be verbose.")

    assert len(calls) == 2


# ---------------------------------------------------------------------------
# CachingGeminiClient — budget-exceeded path
# ---------------------------------------------------------------------------


def test_caching_client_raises_on_budget_exceeded() -> None:
    calls: list[int] = []
    client = _make_client(calls)
    budget = TokenBudget(per_request_max=64)
    caching = CachingGeminiClient(client, budget=budget)

    with pytest.raises(BudgetExceededError, match="per-request"):
        caching.generate("Test.", max_output_tokens=128)

    assert len(calls) == 0  # no underlying call should be made


def test_caching_client_budget_not_checked_on_cache_hit() -> None:
    calls: list[int] = []
    client = _make_client(calls)
    # Tiny daily budget that would be exceeded on a second live call.
    budget = TokenBudget(per_request_max=512, daily_limit=10)
    caching = CachingGeminiClient(client, budget=budget)

    # First call: live — records usage.
    caching.generate("Who wins?", max_output_tokens=10)
    # Second identical call: cache hit — budget NOT checked, so no error.
    result = caching.generate("Who wins?", max_output_tokens=10)

    assert result == FAKE_RESPONSE
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# CachingGeminiClient — TTL override
# ---------------------------------------------------------------------------


def test_caching_client_custom_ttl() -> None:
    calls: list[int] = []
    client = _make_client(calls)
    caching = CachingGeminiClient(client, cache=ResponseCache(default_ttl=3600))

    caching.generate("Test.", ttl=0.01)
    time.sleep(0.02)
    caching.generate("Test.")  # should be a miss after expiry

    assert len(calls) == 2
