"""NRL match /data scraper.

Thin wrapper around the per-match endpoint documented in `docs/nrl-endpoints.md`.
Throttled to be polite and retrying on transient failures; 404s return None
because a wrong slug order (away-v-home vs home-v-away) is a common caller bug,
not a server error worth crashing on.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any

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
) -> dict[str, Any] | None:
    """Fetch a single match's per-match JSON.

    Returns the parsed JSON body on 200, or None on 404. Raises `httpx.HTTPError`
    after exhausting retries on 5xx / network errors.

    Slug order must be `home-v-away` (source from the fixtures endpoint —
    wrong order returns 404 and is treated as a missing match, not an error).
    """

    path = _match_path(year, round_, home_slug, away_slug)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    interval = _min_interval_seconds()

    owns_client = client is None
    http = client or httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT)
    try:
        for attempt in range(1, max_retries + 1):
            _throttle.wait(interval)
            try:
                response = http.get(path, headers=headers)
            except httpx.HTTPError as exc:
                if attempt == max_retries:
                    logger.error(
                        "Network error fetching %s after %d attempts: %s",
                        path,
                        attempt,
                        exc,
                    )
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

            if response.status_code == 404:
                logger.warning(
                    "404 for %s — check slug order (expected home-v-away)",
                    path,
                )
                return None
            if 500 <= response.status_code < 600:
                if attempt == max_retries:
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
            return response.json()

        # Loop fell through without returning — should be unreachable because
        # the last attempt either returns, raises, or is a 404 short-circuit.
        raise RuntimeError("fetch_match retry loop exited without resolution")
    finally:
        if owns_client:
            http.close()


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: 1s, 2s, 4s (+/- 25% jitter)."""
    base = 2 ** (attempt - 1)
    jitter = random.uniform(-0.25, 0.25) * base
    return max(0.0, base + jitter)
