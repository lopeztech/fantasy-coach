"""Server-side FCM notification sender (#172).

Two notification types:
- ``send_round_published(season, round_id)``: fires at the tail of precompute.
  Targets all opted-in users. Max once per round.
- ``send_kickoff_reminder(match_id, home_name, away_name)``: fires 30 min
  before kickoff for users who have not yet tipped that match.

Both helpers are no-ops when firebase-admin is not configured (local dev
without FIREBASE_PROJECT_ID).

Rate cap: hard server-side limit of 4 notifications per user per week to
prevent misconfiguration flooding. Implemented via a simple Firestore counter
document at ``notification_rate_limits/{uid}`` that tracks sends in a 7-day
rolling window. If sending fails, it is logged but not retried.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

_RATE_LIMIT_COLLECTION = "notification_rate_limits"
_FCM_TOKENS_COLLECTION = "users"  # tokens live at users/{uid}/fcm_tokens/{token}
_MAX_SENDS_PER_WEEK = 4
_QUIET_HOUR_START = 23  # 11 PM
_QUIET_HOUR_END = 7  # 7 AM


def _is_quiet_hours_utc() -> bool:
    """Return True when it's between 23:00 and 07:00 UTC (approximate quiet hours)."""
    h = datetime.now(UTC).hour
    return h >= _QUIET_HOUR_START or h < _QUIET_HOUR_END


def _get_all_tokens(db: object) -> list[tuple[str, str]]:
    """Return ``[(uid, fcm_token), ...]`` for all opted-in users."""
    tokens: list[tuple[str, str]] = []
    # Collection-group query on sub-collection fcm_tokens across all users.
    token_docs = db.collection_group("fcm_tokens").stream()  # type: ignore[attr-defined]
    for doc in token_docs:
        data = doc.to_dict()
        token = data.get("token") or doc.id
        uid = doc.reference.parent.parent.id  # users/{uid}/fcm_tokens/{token}
        if token and uid:
            tokens.append((uid, token))
    return tokens


def _check_and_increment_rate(db: object, uid: str) -> bool:
    """Return True if user is under the weekly rate limit; increment if so."""
    from google.cloud import firestore  # noqa: PLC0415

    now = datetime.now(UTC)
    week_ago = now - timedelta(days=7)
    doc_ref = db.collection(_RATE_LIMIT_COLLECTION).document(uid)  # type: ignore[attr-defined]

    @firestore.transactional  # type: ignore[attr-defined]
    def run_transaction(transaction: object, ref: object) -> bool:
        snap = ref.get(transaction=transaction)  # type: ignore[attr-defined]
        data = snap.to_dict() or {} if snap.exists else {}  # type: ignore[attr-defined]
        # Filter send timestamps within the window.
        sends: list = [s for s in data.get("sends", []) if s.replace(tzinfo=UTC) > week_ago]
        if len(sends) >= _MAX_SENDS_PER_WEEK:
            return False
        sends.append(now)
        transaction.set(ref, {"sends": sends}, merge=True)  # type: ignore[attr-defined]
        return True

    transaction = db.transaction()  # type: ignore[attr-defined]
    return run_transaction(transaction, doc_ref)


def send_round_published(season: int, round_id: int) -> None:
    """Broadcast a 'predictions ready' notification to all opted-in users.

    Safe to call from the precompute Job tail — no-ops gracefully when
    Firebase is not configured or there are no registered tokens.
    """
    project = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        logger.debug("send_round_published: no Firebase project configured, skipping")
        return

    try:
        import firebase_admin  # noqa: PLC0415
        from firebase_admin import messaging  # noqa: PLC0415
        from google.cloud import firestore  # noqa: PLC0415
    except ImportError:
        logger.warning("send_round_published: firebase-admin not installed, skipping")
        return

    if not firebase_admin._apps:  # type: ignore[attr-defined]
        firebase_admin.initialize_app()

    db = firestore.Client(project=project)
    if _is_quiet_hours_utc():
        logger.info("send_round_published: quiet hours, skipping")
        return

    tokens = _get_all_tokens(db)
    if not tokens:
        logger.info("send_round_published: no registered tokens")
        return

    sent = 0
    for uid, token in tokens:
        if not _check_and_increment_rate(db, uid):
            logger.debug("send_round_published: rate limit reached for %s", uid)
            continue
        msg = messaging.Message(
            notification=messaging.Notification(
                title="This week's predictions are ready 🏉",
                body=f"Season {season} Round {round_id} — check your picks",
            ),
            data={"action_url": f"/round/{season}/{round_id}"},
            token=token,
        )
        try:
            messaging.send(msg)
            sent += 1
        except Exception as exc:
            logger.warning("send_round_published: failed to send to %s: %s", token[:16], exc)

    logger.info("send_round_published: sent to %d/%d tokens", sent, len(tokens))


def send_kickoff_reminder(
    match_id: int,
    season: int,
    round_id: int,
    home_name: str,
    away_name: str,
    untipped_uids: list[str],
) -> None:
    """Send a kick-off reminder to users who haven't yet tipped a specific match.

    Caller is responsible for passing only ``untipped_uids`` (users who have
    not tipped ``match_id``). Not called during quiet hours.
    """
    project = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        return

    if _is_quiet_hours_utc():
        logger.info("send_kickoff_reminder: quiet hours, skipping")
        return

    try:
        import firebase_admin  # noqa: PLC0415
        from firebase_admin import messaging  # noqa: PLC0415
        from google.cloud import firestore  # noqa: PLC0415
    except ImportError:
        logger.warning("send_kickoff_reminder: firebase-admin not installed, skipping")
        return

    if not firebase_admin._apps:  # type: ignore[attr-defined]
        firebase_admin.initialize_app()

    db = firestore.Client(project=project)
    all_tokens = dict(_get_all_tokens(db))  # uid → token

    sent = 0
    for uid in untipped_uids:
        token = all_tokens.get(uid)
        if not token:
            continue
        if not _check_and_increment_rate(db, uid):
            continue
        msg = messaging.Message(
            notification=messaging.Notification(
                title=f"{home_name} v {away_name} kicks off in 30 minutes",
                body="Get your tip in before it's too late!",
            ),
            data={"action_url": f"/round/{season}/{round_id}/{match_id}"},
            token=token,
        )
        try:
            messaging.send(msg)
            sent += 1
        except Exception as exc:
            logger.warning("send_kickoff_reminder: failed for %s: %s", uid, exc)

    logger.info("send_kickoff_reminder: sent to %d/%d untipped users", sent, len(untipped_uids))
