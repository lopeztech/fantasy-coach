"""NRL endpoint scrapers.

Thin wrappers around the two `nrl.com` JSON endpoints documented in
`docs/nrl-endpoints.md`. Throttled to be polite and retrying on transient
failures; 404s return None because a wrong slug order (away-v-home vs
home-v-away) is a common caller bug, not a server error worth crashing on.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any
from urllib.parse import urlparse

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
    """Fetch a single regular-season match's per-match JSON.

    For finals matches use `fetch_match_from_url(matchCentreUrl)` — finals
    slugs (`finals-week-{n}/game-{m}`) don't fit this signature.

    Returns the parsed JSON body on 200, or None on 404. Raises
    `httpx.HTTPError` after exhausting retries on 5xx / network errors.
    """

    path = _match_path(year, round_, home_slug, away_slug)
    return _fetch_json(path, client=client, max_retries=max_retries)


def fetch_match_from_url(
    match_centre_url: str,
    *,
    client: httpx.Client | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any] | None:
    """Fetch a per-match payload using a `matchCentreUrl` from the fixtures list.

    Accepts either a relative path (`/draw/.../`) or a full URL. Appends `data`
    to the trailing slash. Use this for finals weeks where slugs are
    `finals-week-{n}/game-{m}` rather than home-v-away.
    """

    path = _normalize_match_path(match_centre_url)
    return _fetch_json(path, client=client, max_retries=max_retries)


def fetch_round(
    year: int,
    round_: int,
    *,
    competition: int = 111,
    client: httpx.Client | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any] | None:
    """Fetch the fixtures-list payload for a given round.

    Returns the parsed JSON (with `fixtures`, `byes`, filter metadata) on 200,
    or None on 404 (e.g. a round that doesn't exist for that season).
    """

    path = "/draw/data"
    params = {"competition": competition, "round": round_, "season": year}
    return _fetch_json(path, params=params, client=client, max_retries=max_retries)


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
) -> dict[str, Any] | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    interval = _min_interval_seconds()

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
                logger.warning("404 for %s", path)
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

        raise RuntimeError(f"fetch retry loop exited without resolution for {path}")
    finally:
        if owns_client:
            http.close()


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: 1s, 2s, 4s (+/- 25% jitter)."""
    base = 2 ** (attempt - 1)
    jitter = random.uniform(-0.25, 0.25) * base
    return max(0.0, base + jitter)
