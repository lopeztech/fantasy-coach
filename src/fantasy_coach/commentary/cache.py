"""Response caching + token budget guard for GeminiClient.

Layered architecture:
  CachingGeminiClient
    └── ResponseCache  (prompt-hash × model-version → response, with TTL)
    └── TokenBudget    (per-request cap + per-day rolling circuit-breaker)
    └── GeminiClient   (the actual Vertex AI call)

Cache entries are stored in-memory by default.  In production, swap
ResponseCache for a Firestore-backed implementation with the same interface.

Cache invalidation triggers:
  - Bump CACHE_KEY_VERSION when the prompt template or feature schema changes.
    Every cached entry carries the version at write time; a version bump
    causes all old entries to miss on next access (lazy eviction).
  - Call cache.clear_version_mismatch() for eager eviction (e.g. after a
    bad template rollout rollback).
  - Call cache.clear_stale(max_age_secs) to evict old entries by age.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

from fantasy_coach.commentary.client import GeminiClient, GeminiResponse

logger = logging.getLogger(__name__)

# Bump this whenever the prompt template or FEATURE_NAMES schema changes.
# Old entries whose cache_key_version != CACHE_KEY_VERSION are treated as misses.
CACHE_KEY_VERSION: int = 1

# Default round-level TTL: 7 days.  Predictions don't change between rounds.
DEFAULT_TTL: float = 7 * 24 * 3600.0

# Cost constants for est. spend calculation (Gemini Flash, Vertex AI).
_COST_PER_1K_INPUT = 0.00025
_COST_PER_1K_OUTPUT = 0.0005


class BudgetExceededError(RuntimeError):
    """Raised when a request would breach the configured token budget."""


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    response: GeminiResponse
    expires_at: float         # monotonic clock
    created_at: float         # wall-clock (time.time())
    cache_key_version: int
    feature_snapshot_hash: str = ""   # SHA-8 of the MatchContext inputs
    token_count_in: int = 0
    token_count_out: int = 0


# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------


class ResponseCache:
    """In-memory response cache keyed by (cache_key_version, model_version, prompt_hash).

    The cache key is SHA-256(str(CACHE_KEY_VERSION) + ":" + model_version + ":" + combined_prompt)
    so any change to the version, prompt, or model invalidates the entry automatically.

    Swap the backend by subclassing and overriding ``_raw_get`` / ``_raw_set``.
    """

    def __init__(self, default_ttl: float = DEFAULT_TTL) -> None:
        self._default_ttl = default_ttl
        self._store: dict[str, _CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._tokens_in_total: int = 0
        self._tokens_out_total: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, prompt: str, model_version: str) -> GeminiResponse | None:
        """Return a cached response or ``None`` on a miss / expired entry / version mismatch."""
        key = _cache_key(prompt, model_version)
        entry = self._store.get(key)
        now = time.monotonic()
        if entry is None or now > entry.expires_at or entry.cache_key_version != CACHE_KEY_VERSION:
            if entry is not None:
                del self._store[key]
            self._misses += 1
            logger.debug(
                "commentary_cache_miss key_prefix=%.8s version=%d",
                key,
                CACHE_KEY_VERSION,
            )
            return None
        self._hits += 1
        self._tokens_in_total += entry.token_count_in
        self._tokens_out_total += entry.token_count_out
        logger.info(
            "commentary_cache_hit key_prefix=%.8s hit_rate=%.2f "
            "hits=%d misses=%d feature_hash=%s",
            key,
            self.hit_rate,
            self._hits,
            self._misses,
            entry.feature_snapshot_hash,
        )
        return entry.response

    def set(
        self,
        prompt: str,
        model_version: str,
        response: GeminiResponse,
        *,
        ttl: float | None = None,
        feature_snapshot_hash: str = "",
    ) -> None:
        """Store a response in the cache."""
        key = _cache_key(prompt, model_version)
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        entry = _CacheEntry(
            response=response,
            expires_at=expires_at,
            created_at=time.time(),
            cache_key_version=CACHE_KEY_VERSION,
            feature_snapshot_hash=feature_snapshot_hash,
            token_count_in=response.input_tokens,
            token_count_out=response.output_tokens,
        )
        self._store[key] = entry
        self._tokens_in_total += response.input_tokens
        self._tokens_out_total += response.output_tokens

    def invalidate(self, prompt: str, model_version: str) -> None:
        key = _cache_key(prompt, model_version)
        self._store.pop(key, None)

    def clear_version_mismatch(self) -> int:
        """Eagerly evict all entries whose cache_key_version != CACHE_KEY_VERSION."""
        stale = [k for k, e in self._store.items() if e.cache_key_version != CACHE_KEY_VERSION]
        for k in stale:
            del self._store[k]
        if stale:
            logger.info("commentary_cache_purge count=%d reason=version_mismatch", len(stale))
        return len(stale)

    def clear_stale(self, max_age_secs: float) -> int:
        """Evict all entries older than max_age_secs (wall-clock age)."""
        cutoff = time.time() - max_age_secs
        stale = [k for k, e in self._store.items() if e.created_at < cutoff]
        for k in stale:
            del self._store[k]
        if stale:
            logger.info(
                "commentary_cache_purge count=%d reason=max_age max_age_secs=%.0f",
                len(stale),
                max_age_secs,
            )
        return len(stale)

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

    @property
    def tokens_in_total(self) -> int:
        return self._tokens_in_total

    @property
    def tokens_out_total(self) -> int:
        return self._tokens_out_total

    def estimated_cost_usd(self) -> float:
        """Estimated Gemini Flash cost for tokens consumed this session."""
        return (
            self._tokens_in_total / 1000 * _COST_PER_1K_INPUT
            + self._tokens_out_total / 1000 * _COST_PER_1K_OUTPUT
        )

    def summary(self) -> str:
        """One-line run summary suitable for end-of-precompute logging."""
        total = self._hits + self._misses
        hit_pct = f"{self.hit_rate:.0%}" if total else "n/a"
        cost = self.estimated_cost_usd()
        return (
            f"commentary summary: {total} requests, {self._hits} cache hits "
            f"({hit_pct} hit rate), {self._tokens_in_total} tokens in, "
            f"{self._tokens_out_total} tokens out, est. cost ${cost:.4f}"
        )


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
            "commentary_budget used=%d daily_limit=%d remaining=%d",
            self._used,
            self._daily_limit,
            remaining,
        )
        if remaining == 0:
            logger.warning("commentary_budget daily_limit_reached limit=%d", self._daily_limit)

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
        feature_snapshot_hash: str = "",
    ) -> GeminiResponse:
        """Return a (cached) response, enforcing budget before live calls.

        The cache key combines ``system`` + ``prompt`` so changing either
        one is a guaranteed cache miss.  ``ttl`` overrides the cache's
        default TTL for this entry only.  ``feature_snapshot_hash`` is stored
        in the cache entry for observability (not used as a key dimension —
        the prompt already encodes the feature inputs).
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
        self._cache.set(
            combined,
            model_version,
            response,
            ttl=ttl,
            feature_snapshot_hash=feature_snapshot_hash,
        )
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
    """Include CACHE_KEY_VERSION so a version bump busts all existing entries."""
    return hashlib.sha256(
        f"{CACHE_KEY_VERSION}:{model_version}:{prompt}".encode()
    ).hexdigest()


def context_hash(data: str) -> str:
    """Return an 8-char SHA-256 hex digest of arbitrary string data.

    Used to fingerprint a MatchContext's inputs so cache entries can be
    audited for staleness without re-generating the full prompt.
    """
    return hashlib.sha256(data.encode()).hexdigest()[:8]
