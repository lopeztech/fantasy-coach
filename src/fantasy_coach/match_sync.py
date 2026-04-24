"""Re-scrape match docs whose outcome should be known by now.

Background: the precompute Job (#65) only scrapes the *current upcoming*
round. Once a match kicks off and finishes, nothing in the pipeline
re-visits that match row to capture the final state + scores. Rounds
appear ``FullTime`` in Firestore today only because of manual
``copy-matches-to-firestore`` runs after-the-fact — the first round to
actually live through the precompute-only flow (2026 round 8) stayed
stuck on ``Upcoming`` forever, which:

- breaks the ``/accuracy`` endpoint (filters on ``match_state``).
- breaks the retrain loop (#107) — ``split_training_holdout`` drops
  non-FullTime matches, so round N's outcomes silently vanish from
  the holdout the Monday after.

``refresh_stale_matches`` finds every season-X match whose
``start_time`` has already passed but whose stored ``match_state`` is
not ``FullTime``, re-scrapes each via the existing ``fetch_round`` +
``fetch_match_from_url`` pipeline, and upserts. Idempotent: a fresh
re-scrape that still returns ``Upcoming`` (nrl.com hasn't updated yet)
is a no-op; a later run picks it up.

Called from the precompute Job (Tue + Thu) and the retrain Job's
pre-flight (Mon) so a weekend round is guaranteed to reach the retrain
loop with real outcomes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fantasy_coach.features import extract_match_features
from fantasy_coach.scraper import fetch_match_from_url, fetch_round
from fantasy_coach.storage.repository import Repository

logger = logging.getLogger(__name__)


_FULLTIME_STATES = {"FullTime", "FullTimeED"}


def refresh_stale_matches(
    repo: Repository,
    season: int,
    *,
    now: datetime | None = None,
    fetch_round_fn: Callable = fetch_round,
    fetch_match_fn: Callable = fetch_match_from_url,
    max_rounds_back: int = 4,
) -> int:
    """Re-scrape any ``season`` match whose start_time < now and state is not FullTime.

    Groups stale matches by round and re-scrapes each round once (one
    ``fetch_round`` call) to keep request volume low; per-match details
    are fetched individually because the round endpoint's fixture list
    lacks scores / full state.

    ``max_rounds_back`` caps how far back we iterate to avoid re-scraping
    the whole season on a fresh DB. Four rounds covers any realistic gap
    between precompute runs (every 3–4 days).

    Returns count of matches whose docs were upserted (including those
    that came back still ``Upcoming``, which is common for matches in
    states like ``Live`` mid-broadcast).
    """
    now = now or datetime.now(UTC)

    # Pull the full season once — Repository has no "where state != X"
    # accessor, and a per-round pass would duplicate fetches. Season
    # size is O(matches-per-season) which is fine in-memory.
    all_matches = repo.list_matches(season)
    stale_rounds: dict[int, list] = {}
    for m in all_matches:
        if m.match_state in _FULLTIME_STATES:
            continue
        # Ensure both are timezone-aware comparisons. start_time from the
        # repo is timezone-aware ISO 8601; `now` is UTC here.
        if m.start_time >= now:
            continue
        stale_rounds.setdefault(m.round, []).append(m)

    if not stale_rounds:
        logger.info("refresh_stale_matches: no stale matches for %d", season)
        return 0

    # Limit blast radius — stop after max_rounds_back most recent rounds.
    kept_rounds = sorted(stale_rounds)[-max_rounds_back:]
    logger.info(
        "refresh_stale_matches: %d stale matches across rounds %s",
        sum(len(stale_rounds[r]) for r in kept_rounds),
        kept_rounds,
    )

    updated = 0
    for round_ in kept_rounds:
        try:
            round_payload = fetch_round_fn(season, round_)
        except Exception:
            logger.exception("refresh_stale_matches: fetch_round failed for %d r%d", season, round_)
            continue
        if not round_payload:
            continue
        fixtures = round_payload.get("fixtures") or []

        # Only re-scrape matches we already know are stale — saves calls
        # against matches scheduled later in the same round.
        stale_ids = {m.match_id for m in stale_rounds[round_]}
        for fixture in fixtures:
            url = fixture.get("matchCentreUrl")
            if not url:
                continue
            try:
                raw = fetch_match_fn(url)
            except Exception:
                logger.exception("refresh_stale_matches: fetch_match failed for %s", url)
                continue
            if raw is None:
                continue
            try:
                row = extract_match_features(raw)
            except Exception:
                logger.exception("refresh_stale_matches: extract failed for %s", url)
                continue
            if row.match_id not in stale_ids:
                continue
            try:
                repo.upsert_match(row)
            except Exception:
                logger.exception("refresh_stale_matches: upsert failed for %d", row.match_id)
                continue
            updated += 1
            logger.info(
                "refresh_stale_matches: %d r%d match %d state %s",
                season,
                round_,
                row.match_id,
                row.match_state,
            )

    return updated


# ---------------------------------------------------------------------------
# Leaderboard stats sync (#173)
# ---------------------------------------------------------------------------


def sync_leaderboard_stats(season: int, round_id: int) -> int:
    """Update users/{uid}/stats/<season> for every tip on a just-completed round.

    Reads FullTime match results from the repo, fetches all user tips via
    collection-group query, and atomically updates each user's stats doc.
    Returns the number of tips scored. No-op when not running against Firestore.
    """
    project = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        logger.debug("sync_leaderboard_stats: no Firebase project configured, skipping")
        return 0

    try:
        from google.cloud import firestore  # noqa: PLC0415
    except ImportError:
        logger.warning("sync_leaderboard_stats: google-cloud-firestore not installed")
        return 0

    from fantasy_coach.config import get_repository  # noqa: PLC0415

    db = firestore.Client(project=project)
    repo = get_repository()

    try:
        matches = repo.list_matches(season, round_id)
    except Exception as exc:
        logger.error("sync_leaderboard_stats: could not load matches: %s", exc)
        return 0

    results: dict[int, str] = {
        m.match_id: ("home" if m.home.score > m.away.score else "away")
        for m in matches
        if (
            m.match_state in {"FullTime", "FullTimeED"}
            and m.home.score is not None
            and m.away.score is not None
        )
    }

    if not results:
        logger.info("sync_leaderboard_stats: no FullTime matches in round %d", round_id)
        return 0

    tips_by_uid: dict[str, list[dict[str, Any]]] = {}
    for tip_doc in (
        db.collection_group("tips")
        .where("season", "==", season)
        .where("round", "==", round_id)
        .stream()
    ):
        match_id = int(tip_doc.id)
        if match_id not in results:
            continue
        uid = tip_doc.reference.parent.parent.id
        tips_by_uid.setdefault(uid, []).append(
            {"match_id": match_id, "tip": tip_doc.to_dict().get("tip"), "actual": results[match_id]}
        )

    for uid, tips in tips_by_uid.items():
        _update_user_stats(db, uid, season, tips)

    total = sum(len(t) for t in tips_by_uid.values())
    logger.info("sync_leaderboard_stats: scored %d tips for %d users", total, len(tips_by_uid))
    return total


def _update_user_stats(db: Any, uid: str, season: int, new_tips: list[dict[str, Any]]) -> None:
    from google.cloud import firestore  # noqa: PLC0415

    stats_ref = db.collection("users").document(uid).collection("stats").document(str(season))

    @firestore.transactional
    def run(transaction: Any, ref: Any) -> None:
        snap = ref.get(transaction=transaction)
        data = snap.to_dict() if snap.exists else {}
        scored: set[int] = set(data.get("scored_matches", []))
        wins = data.get("wins", 0)
        losses = data.get("losses", 0)
        total = data.get("total_tips", 0)
        streak = data.get("current_streak", 0)
        best = data.get("longest_streak", 0)

        for tip in new_tips:
            mid = tip["match_id"]
            if mid in scored:
                continue
            scored.add(mid)
            total += 1
            if tip["tip"] == tip["actual"]:
                wins += 1
                streak += 1
                best = max(best, streak)
            else:
                losses += 1
                streak = 0

        transaction.set(
            ref,
            {
                "season": season,
                "wins": wins,
                "losses": losses,
                "total_tips": total,
                "accuracy": wins / total if total > 0 else 0.0,
                "margin_points": data.get("margin_points", 0.0),
                "current_streak": streak,
                "longest_streak": best,
                "scored_matches": list(scored),
                "last_updated": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    tx = db.transaction()
    run(tx, stats_ref)
