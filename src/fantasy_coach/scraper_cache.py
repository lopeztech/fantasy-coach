"""HTTP response cache for the NRL scraper.

Stores ETag / Last-Modified validators per scraped URL so that subsequent
requests can send conditional headers (``If-None-Match`` / ``If-Modified-
Since``). On 304 Not Modified the scraper skips JSON parsing and downstream
Firestore writes entirely — the only write is a timestamp bump on the cache
entry.

For endpoints that never send standard validators (``Cache-Control: no-store``
excluded) a SHA-256 content-hash is stored as a fallback; if the body hash
matches the previous scrape, the URL is treated as unchanged.

Two backends, matching the rest of the system:
- ``SQLiteScraperCache`` — local dev / tests; persists a ``scraper_cache``
  table in the same SQLite file as the main DB.
- ``FirestoreScraperCache`` — production (Cloud Run Job); durable across cold
  starts, keyed by SHA-1 of the URL.

TTL is 30 days; call ``prune()`` periodically to remove stale entries.
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

TTL_DAYS = 30

_CREATE_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS scraper_cache (
    url_key         TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    etag            TEXT,
    last_modified   TEXT,
    content_hash    TEXT,
    last_fetched_at TEXT NOT NULL,
    last_status     INTEGER NOT NULL
);
"""


@dataclass
class CacheEntry:
    url: str
    etag: str | None
    last_modified: str | None
    content_hash: str | None
    last_fetched_at: datetime
    last_status: int


class ScraperCache(ABC):
    """Abstract cache for per-URL HTTP validator metadata."""

    @staticmethod
    def url_key(url: str) -> str:
        return hashlib.sha1(url.encode()).hexdigest()

    @abstractmethod
    def get(self, url: str) -> CacheEntry | None:
        """Return the stored entry for ``url``, or None if not cached."""

    @abstractmethod
    def put(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        content_hash: str | None,
        status: int,
    ) -> None:
        """Upsert a cache entry for ``url``."""

    @abstractmethod
    def prune(self) -> int:
        """Delete entries older than TTL_DAYS. Returns row count removed."""


class SQLiteScraperCache(ScraperCache):
    """SQLite-backed scraper cache.

    Pass the path to the existing ``nrl.db`` (or any SQLite file); the
    ``scraper_cache`` table is created on first use.
    """

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_CACHE_TABLE)

    def close(self) -> None:
        self._conn.close()

    def get(self, url: str) -> CacheEntry | None:
        row = self._conn.execute(
            "SELECT * FROM scraper_cache WHERE url_key = ?",
            (self.url_key(url),),
        ).fetchone()
        if row is None:
            return None
        return CacheEntry(
            url=row["url"],
            etag=row["etag"],
            last_modified=row["last_modified"],
            content_hash=row["content_hash"],
            last_fetched_at=datetime.fromisoformat(row["last_fetched_at"]),
            last_status=row["last_status"],
        )

    def put(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        content_hash: str | None,
        status: int,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO scraper_cache
                    (url_key, url, etag, last_modified, content_hash, last_fetched_at, last_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (self.url_key(url), url, etag, last_modified, content_hash, now, status),
            )

    def prune(self) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=TTL_DAYS)).isoformat()
        with self._conn:
            result = self._conn.execute(
                "DELETE FROM scraper_cache WHERE last_fetched_at < ?",
                (cutoff,),
            )
        return result.rowcount


class FirestoreScraperCache(ScraperCache):
    """Firestore-backed scraper cache (production).

    Collection: ``scraper_cache``. Each document is keyed by SHA-1 of the
    scraped URL so Cloud Run Job cold starts share a persistent cache.
    TTL enforcement relies on Firestore TTL policies (30 days on
    ``last_fetched_at``) or manual ``prune()`` calls.
    """

    _COLLECTION = "scraper_cache"

    def __init__(self, client: Any) -> None:
        self._db = client

    def get(self, url: str) -> CacheEntry | None:
        snap = self._db.collection(self._COLLECTION).document(self.url_key(url)).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        return CacheEntry(
            url=data["url"],
            etag=data.get("etag"),
            last_modified=data.get("last_modified"),
            content_hash=data.get("content_hash"),
            last_fetched_at=datetime.fromisoformat(data["last_fetched_at"]),
            last_status=data["last_status"],
        )

    def put(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        content_hash: str | None,
        status: int,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._db.collection(self._COLLECTION).document(self.url_key(url)).set(
            {
                "url": url,
                "etag": etag,
                "last_modified": last_modified,
                "content_hash": content_hash,
                "last_fetched_at": now,
                "last_status": status,
            }
        )

    def prune(self) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=TTL_DAYS)).isoformat()
        batch_size = 500
        deleted = 0
        col = self._db.collection(self._COLLECTION)
        while True:
            docs = col.where("last_fetched_at", "<", cutoff).limit(batch_size).stream()
            batch = self._db.batch()
            count = 0
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            if count == 0:
                break
            batch.commit()
            deleted += count
            if count < batch_size:
                break
        return deleted
