"""Response caching + token budget guard for GeminiClient.

Layered architecture:
  CachingGeminiClient
    └── ResponseCache  (prompt-hash × model-version → response, with TTL)
    └── TokenBudget    (per-request cap + per-day rolling circuit-breaker)
    └── GeminiClient   (the actual Vertex AI call)

Cache entries are stored in-memory by default.  In production, swap
ResponseCache for a Firestore-backed implementation with the same interface.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from fantasy_coach.commentary.client import GeminiClient, GeminiResponse

logger = logging.getLogger(__name__)

# Default round-level TTL: 7 days.  Predictions don't change between rounds.
DEFAULT_TTL: float = 7 * 24 * 3600.0


class BudgetExceededError(RuntimeError):
    """Raised when a request would breach the configured token budget."""


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    response: GeminiResponse
    expires_at: float  # monotonic clock


# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------


class ResponseCache:
    """In-memory response cache keyed by (prompt_hash, model_version) with TTL.

    The cache key is SHA-256( ``model_version + ":" + combined_prompt`` ) so
    any change to the prompt or model invalidates the entry automatically.

    Swap the backend by subclassing and overriding ``_raw_get`` / ``_raw_set``.
    """

    def __init__(self, default_ttl: float = DEFAULT_TTL) -> None:
        self._default_ttl = default_ttl
        self._store: dict[str, _CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, prompt: str, model_version: str) -> GeminiResponse | None:
        """Return a cached response or ``None`` on a miss / expired entry."""
        key = _cache_key(prompt, model_version)
        entry = self._store.get(key)
        if entry is None or time.monotonic() > entry.expires_at:
            if entry is not None:
                del self._store[key]
            self._misses += 1
            logger.debug("gemini_cache status=miss key_prefix=%.8s", key)
            return None
        self._hits += 1
        logger.info(
            "gemini_cache status=hit hit_rate=%.2f hits=%d misses=%d",
            self.hit_rate,
            self._hits,
            self._misses,
        )
        return entry.response

    def set(
        self,
        prompt: str,
        model_version: str,
        response: GeminiResponse,
        *,
        ttl: float | None = None,
    ) -> None:
        """Store a response in the cache."""
        key = _cache_key(prompt, model_version)
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        self._store[key] = _CacheEntry(response=response, expires_at=expires_at)

    def invalidate(self, prompt: str, model_version: str) -> None:
        key = _cache_key(prompt, model_version)
        self._store.pop(key, None)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


class TokenBudget:
    """Per-request cap + per-day rolling circuit-breaker.

    Two independent limits:
    - ``per_request_max``: hard cap on ``max_output_tokens`` per call.
    - ``daily_limit``: rolling 24 h total-token ceiling (input + output).
    """

    def __init__(
        self,
        *,
        per_request_max: int = 512,
        daily_limit: int = 100_000,
    ) -> None:
        self._per_request_max = per_request_max
        self._daily_limit = daily_limit
        self._used: int = 0
        self._window_start: float = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_request(self, max_output_tokens: int) -> None:
        """Raise BudgetExceededError if the request would breach any limit."""
        if max_output_tokens > self._per_request_max:
            raise BudgetExceededError(
                f"max_output_tokens={max_output_tokens} exceeds per-request "
                f"limit of {self._per_request_max}"
            )
        self._reset_if_expired()
        projected = self._used + max_output_tokens
        if projected > self._daily_limit:
            raise BudgetExceededError(
                f"Daily token budget ({self._daily_limit}) would be exceeded. "
                f"Already used {self._used} tokens in the current window."
            )

    def record_usage(self, total_tokens: int) -> None:
        """Account for tokens consumed by a completed request."""
        self._reset_if_expired()
        self._used += total_tokens
        remaining = max(0, self._daily_limit - self._used)
        logger.info(
            "gemini_budget used=%d daily_limit=%d remaining=%d",
            self._used,
            self._daily_limit,
            remaining,
        )
        if remaining == 0:
            logger.warning("gemini_budget daily_limit_reached limit=%d", self._daily_limit)

    @property
    def used(self) -> int:
        self._reset_if_expired()
        return self._used

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_if_expired(self) -> None:
        if time.time() - self._window_start >= 86_400:
            self._used = 0
            self._window_start = time.time()


# ---------------------------------------------------------------------------
# CachingGeminiClient
# ---------------------------------------------------------------------------


class CachingGeminiClient:
    """GeminiClient wrapper with response caching + token budget enforcement.

    Usage::

        client = CachingGeminiClient(
            GeminiClient(project="my-project"),
            cache=ResponseCache(default_ttl=3600),
            budget=TokenBudget(per_request_max=256, daily_limit=50_000),
        )
        response = client.generate("Who will win?", system="You are an NRL expert.")
        # Second identical call returns instantly from cache:
        response2 = client.generate("Who will win?", system="You are an NRL expert.")
    """

    def __init__(
        self,
        client: GeminiClient,
        *,
        cache: ResponseCache | None = None,
        budget: TokenBudget | None = None,
    ) -> None:
        self._client = client
        self._cache = cache if cache is not None else ResponseCache()
        self._budget = budget if budget is not None else TokenBudget()

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_output_tokens: int = 512,
        ttl: float | None = None,
    ) -> GeminiResponse:
        """Return a (cached) response, enforcing budget before live calls.

        The cache key combines ``system`` + ``prompt`` so changing either
        one is a guaranteed cache miss.  ``ttl`` overrides the cache's
        default TTL for this entry only.
        """
        combined = f"{system}\n\n{prompt}"
        model_version = self._client._model

        cached = self._cache.get(combined, model_version)
        if cached is not None:
            return cached

        self._budget.check_request(max_output_tokens)

        response = self._client.generate(
            prompt,
            system=system,
            max_output_tokens=max_output_tokens,
        )
        self._budget.record_usage(response.total_tokens)
        self._cache.set(combined, model_version, response, ttl=ttl)
        return response

    # Expose underlying metrics for observability.
    @property
    def cache(self) -> ResponseCache:
        return self._cache

    @property
    def budget(self) -> TokenBudget:
        return self._budget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_key(prompt: str, model_version: str) -> str:
    return hashlib.sha256(f"{model_version}:{prompt}".encode()).hexdigest()
