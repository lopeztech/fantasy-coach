"""NRL endpoint scrapers.

Thin wrappers around the two `nrl.com` JSON endpoints documented in
`docs/nrl-endpoints.md`. Throttled to be polite and retrying on transient
failures; 404s return None because a wrong slug order (away-v-home vs
home-v-away) is a common caller bug, not a server error worth crashing on.

HTTP caching (#155): pass a ``ScraperCache`` instance to the fetch functions
to enable ETag / If-None-Match conditional requests and content-hash dedup.
On 304 Not Modified the function returns ``None`` — same as "no new data",
which callers can treat as "skip processing". ``InMemoryScraperCache`` is
the default for local dev and testing; the Cloud Run Job can swap in a
Firestore-backed implementation for durability across cold starts.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.nrl.com"
USER_AGENT = (
    "fantasy-coach/0.1 (+https://github.com/lopeztech/fantasy-coach; "
    "research scraper; contact: joshua.lopez.tech@gmail.com)"
)
DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_RETRIES = 3


def _min_interval_seconds() -> float:
    raw = os.getenv("FANTASY_COACH_SCRAPE_INTERVAL_SECONDS", "1.0")
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid FANTASY_COACH_SCRAPE_INTERVAL_SECONDS=%r; using 1.0", raw)
        return 1.0
    return max(0.0, value)


class _Throttle:
    """Process-wide minimum interval between requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_request_at: float = 0.0

    def wait(self, min_interval: float) -> None:
        if min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = self._last_request_at + min_interval - now
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._last_request_at = now


_throttle = _Throttle()


# ---------------------------------------------------------------------------
# HTTP cache types (#155)
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Metadata persisted between requests for a single URL."""

    url: str
    etag: str | None = None
    last_modified: str | None = None
    # SHA-256 of the response body — used when the endpoint doesn't send
    # ETag or Last-Modified so we can still skip identical payloads.
    content_hash: str | None = None
    last_fetched_at: float = 0.0


@dataclass
class InMemoryScraperCache:
    """In-memory scraper cache. Suitable for single-run dedup and testing.

    Swap for a Firestore-backed implementation in the Cloud Run Job so the
    cache survives across cold starts. Any object with ``get`` / ``put``
    matching the signatures below satisfies the implicit protocol.

    ``stats`` tracks fetch outcomes for the per-run log summary.
    """

    _store: dict[str, CacheEntry] = field(default_factory=dict)
    # Counters incremented by _fetch_json for the per-run summary log.
    new: int = 0
    unchanged: int = 0
    errors: int = 0

    def get(self, url_hash: str) -> CacheEntry | None:
        return self._store.get(url_hash)

    def put(self, url_hash: str, entry: CacheEntry) -> None:
        self._store[url_hash] = entry

    def log_summary(self) -> None:
        total = self.new + self.unchanged + self.errors
        logger.info(
            "scraped %d urls: %d new (200), %d unchanged (304/hash), %d error",
            total,
            self.new,
            self.unchanged,
            self.errors,
        )


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()  # noqa: S324


def _body_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_no_store(response: httpx.Response) -> bool:
    return "no-store" in response.headers.get("cache-control", "").lower()


def _build_cache_key(path: str, params: dict[str, Any] | None) -> str:
    """Canonical URL string used as cache-key input."""
    qs = ("?" + urlencode(sorted((params or {}).items()))) if params else ""
    return BASE_URL + path + qs


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------


def _match_path(year: int, round_: int, home_slug: str, away_slug: str) -> str:
    return f"/draw/nrl-premiership/{year}/round-{round_}/{home_slug}-v-{away_slug}/data"


def fetch_match(
    year: int,
    round_: int,
    home_slug: str,
    away_slug: str,
    *,
    client: httpx.Client | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache: InMemoryScraperCache | None = None,
) -> dict[str, Any] | None:
    """Fetch a single regular-season match's per-match JSON.

    For finals matches use `fetch_match_from_url(matchCentreUrl)` — finals
    slugs (`finals-week-{n}/game-{m}`) don't fit this signature.

    Returns the parsed JSON body on 200, or None on 404 / 304 Not Modified.
    Raises `httpx.HTTPError` after exhausting retries on 5xx / network errors.
    """

    path = _match_path(year, round_, home_slug, away_slug)
    return _fetch_json(path, client=client, max_retries=max_retries, cache=cache)


def fetch_match_from_url(
    match_centre_url: str,
    *,
    client: httpx.Client | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache: InMemoryScraperCache | None = None,
) -> dict[str, Any] | None:
    """Fetch a per-match payload using a `matchCentreUrl` from the fixtures list.

    Accepts either a relative path (`/draw/.../`) or a full URL. Appends `data`
    to the trailing slash. Use this for finals weeks where slugs are
    `finals-week-{n}/game-{m}` rather than home-v-away.
    """

    path = _normalize_match_path(match_centre_url)
    return _fetch_json(path, client=client, max_retries=max_retries, cache=cache)


def fetch_round(
    year: int,
    round_: int,
    *,
    competition: int = 111,
    client: httpx.Client | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache: InMemoryScraperCache | None = None,
) -> dict[str, Any] | None:
    """Fetch the fixtures-list payload for a given round.

    Returns the parsed JSON (with `fixtures`, `byes`, filter metadata) on 200,
    or None on 404 (e.g. a round that doesn't exist for that season).
    """

    path = "/draw/data"
    params = {"competition": competition, "round": round_, "season": year}
    return _fetch_json(path, params=params, client=client, max_retries=max_retries, cache=cache)


def _normalize_match_path(match_centre_url: str) -> str:
    parsed = urlparse(match_centre_url)
    path = parsed.path or match_centre_url
    if not path.startswith("/"):
        path = "/" + path
    if not path.endswith("/"):
        path += "/"
    return path + "data"


def _fetch_json(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache: InMemoryScraperCache | None = None,
) -> dict[str, Any] | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    interval = _min_interval_seconds()

    # Compute cache key and load any stored validators before building the
    # request — etag / last-modified go into request headers.
    cache_key = _build_cache_key(path, params)
    url_hash = _url_hash(cache_key)
    cached = cache.get(url_hash) if cache is not None else None
    if cached is not None:
        if cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified

    owns_client = client is None
    http = client or httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT)
    try:
        for attempt in range(1, max_retries + 1):
            _throttle.wait(interval)
            try:
                response = http.get(path, params=params, headers=headers)
            except httpx.HTTPError as exc:
                if attempt == max_retries:
                    logger.error(
                        "Network error fetching %s after %d attempts: %s",
                        path,
                        attempt,
                        exc,
                    )
                    if cache is not None:
                        cache.errors += 1
                    raise
                delay = _backoff_delay(attempt)
                logger.warning(
                    "Network error fetching %s (attempt %d/%d): %s; retrying in %.2fs",
                    path,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

            if response.status_code == 304:
                logger.debug("304 Not Modified for %s; skipping downstream processing", path)
                # Touch last_fetched_at so we can audit cache freshness.
                if cache is not None and cached is not None:
                    cache.put(
                        url_hash,
                        CacheEntry(
                            url=cache_key,
                            etag=cached.etag,
                            last_modified=cached.last_modified,
                            content_hash=cached.content_hash,
                            last_fetched_at=time.time(),
                        ),
                    )
                    cache.unchanged += 1
                return None

            if response.status_code == 404:
                logger.warning("404 for %s", path)
                return None
            if 500 <= response.status_code < 600:
                if attempt == max_retries:
                    if cache is not None:
                        cache.errors += 1
                    response.raise_for_status()
                delay = _backoff_delay(attempt)
                logger.warning(
                    "%d from %s (attempt %d/%d); retrying in %.2fs",
                    response.status_code,
                    path,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()

            # Update cache unless the server asked us not to.
            if cache is not None and not _is_no_store(response):
                etag = response.headers.get("etag") or None
                last_mod = response.headers.get("last-modified") or None
                body_hash: str | None = None
                if not etag and not last_mod:
                    # Content-hash fallback: compute SHA-256 of the raw body
                    # and skip downstream if unchanged. Warn once so we notice
                    # if NRL ever starts sending validators.
                    logger.debug(
                        "No ETag or Last-Modified from %s; falling back to content-hash dedup",
                        path,
                    )
                    body_hash = _body_hash(response.content)
                    if cached is not None and cached.content_hash == body_hash:
                        logger.debug("Content-hash unchanged for %s; skipping", path)
                        cache.put(
                            url_hash,
                            CacheEntry(
                                url=cache_key,
                                etag=None,
                                last_modified=None,
                                content_hash=body_hash,
                                last_fetched_at=time.time(),
                            ),
                        )
                        cache.unchanged += 1
                        return None
                cache.put(
                    url_hash,
                    CacheEntry(
                        url=cache_key,
                        etag=etag,
                        last_modified=last_mod,
                        content_hash=body_hash,
                        last_fetched_at=time.time(),
                    ),
                )
                cache.new += 1

            return data

        raise RuntimeError(f"fetch retry loop exited without resolution for {path}")
    finally:
        if owns_client:
            http.close()


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: 1s, 2s, 4s (+/- 25% jitter)."""
    base = 2 ** (attempt - 1)
    jitter = random.uniform(-0.25, 0.25) * base
    return max(0.0, base + jitter)
