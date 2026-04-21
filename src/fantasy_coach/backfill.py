"""One-shot historical backfill orchestrator.

Walks every round of a season via the fixtures-list endpoint, fetches each
match's per-match payload, extracts a `MatchRow`, and upserts it into a
`Repository`. Idempotent and resumable: a JSON sidecar tracks processed
match URLs, and re-runs skip anything already recorded there.

CLI: `python -m fantasy_coach backfill --season 2024 --db data/nrl.db`
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fantasy_coach.features import extract_match_features
from fantasy_coach.scraper import fetch_match_from_url, fetch_round
from fantasy_coach.storage.repository import Repository

logger = logging.getLogger(__name__)

# Generous upper bound: 27 regular rounds + 4 finals weeks. We stop early
# the first time the fixtures endpoint returns no matches.
MAX_ROUNDS = 31


@dataclass
class RoundSummary:
    season: int
    round: int
    fetched: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass
class BackfillState:
    """Sidecar record of which match URLs have been successfully processed."""

    path: Path
    processed: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> BackfillState:
        if path.exists():
            data = json.loads(path.read_text())
            return cls(path=path, processed=set(data.get("processed", [])))
        return cls(path=path)

    def mark(self, url: str) -> None:
        self.processed.add(url)
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"processed": sorted(self.processed)}, indent=2))


class RetryLog:
    """Append-only log of (url, reason) pairs for matches that failed to ingest."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, url: str, reason: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(f"{url}\t{reason}\n")


def backfill_season(
    season: int,
    repo: Repository,
    state: BackfillState,
    retry_log: RetryLog,
    *,
    fetch_round_fn: Callable[..., dict[str, Any] | None] = fetch_round,
    fetch_match_fn: Callable[..., dict[str, Any] | None] = fetch_match_from_url,
    max_rounds: int = MAX_ROUNDS,
) -> list[RoundSummary]:
    """Run a full-season backfill. Returns a per-round summary list.

    `fetch_round_fn` and `fetch_match_fn` are injectable so tests can drive the
    orchestrator without hitting the network.
    """

    summaries: list[RoundSummary] = []
    for round_ in range(1, max_rounds + 1):
        round_payload = fetch_round_fn(season, round_)
        fixtures = (round_payload or {}).get("fixtures") or []
        if not fixtures:
            # Past the end of the season — finals already played or round
            # doesn't exist for this year.
            logger.info("Season %d round %d: no fixtures, stopping", season, round_)
            break

        summary = RoundSummary(season=season, round=round_)
        for fixture in fixtures:
            url = fixture.get("matchCentreUrl")
            if not url:
                summary.failed += 1
                retry_log.record("<unknown>", "missing matchCentreUrl in fixture")
                continue
            if url in state.processed:
                summary.skipped += 1
                continue
            try:
                raw = fetch_match_fn(url)
            except Exception as exc:  # network / 5xx exhausted
                summary.failed += 1
                retry_log.record(url, f"fetch failed: {exc}")
                continue
            if raw is None:
                summary.failed += 1
                retry_log.record(url, "404 from /data")
                continue
            try:
                row = extract_match_features(raw)
                repo.upsert_match(row)
            except Exception as exc:
                summary.failed += 1
                retry_log.record(url, f"extract/upsert failed: {exc}")
                continue
            state.mark(url)
            summary.fetched += 1

        logger.info(
            "Season %d round %d: fetched=%d skipped=%d failed=%d",
            season,
            round_,
            summary.fetched,
            summary.skipped,
            summary.failed,
        )
        summaries.append(summary)

    return summaries
