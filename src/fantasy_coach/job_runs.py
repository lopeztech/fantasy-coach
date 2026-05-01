"""Audit log for the precompute Job.

Each precompute invocation writes one doc to the ``job_runs`` Firestore
collection: started/finished/status/round, an aggregate ``summary`` of how
predictions moved, and an inline ``changes[]`` of per-match diffs with a
human-readable one-line ``summary`` string. The SPA's /jobs page reads this
to surface "what changed since Tuesday" without users digging through Cloud
Logging.

``predictions`` stays single-version — there is no prediction history
collection. The diff is captured by snapshotting the prev predictions doc
immediately before ``compute_predictions`` overwrites it.

Trigger-signal detection (``team_list`` / ``weather`` / ``retrain`` / null)
compares this run's per-match metadata to the most recent prior ``job_runs``
doc for the same (season, round). The very first time the writer runs there
is no prior doc, so signals are null on that one round; from then on every
subsequent run can label its drivers.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .predictions import PredictionOut

_logger = logging.getLogger(__name__)

_COLLECTION = "job_runs"

_SIGNAL_CLAUSE = {
    "team_list": " — team list update",
    "weather": " — weather update",
    "retrain": " — model retrained",
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class JobRunChange:
    match_id: int
    home_team: str
    away_team: str
    prev_home_prob: float | None
    new_home_prob: float
    delta: float  # new - prev (0.0 when no prev)
    flipped: bool
    trigger_signal: str | None  # team_list | weather | retrain | None
    summary: str
    # Tracked so the next run can detect signals by comparing.
    model_version: str
    team_list_hash: str | None
    weather_fetched_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prev_home_prob": self.prev_home_prob,
            "new_home_prob": self.new_home_prob,
            "delta": self.delta,
            "flipped": self.flipped,
            "trigger_signal": self.trigger_signal,
            "summary": self.summary,
            "model_version": self.model_version,
            "team_list_hash": self.team_list_hash,
            "weather_fetched_at": self.weather_fetched_at,
        }


@dataclass
class JobRunRecord:
    started_at: datetime
    finished_at: datetime | None
    trigger: str  # "scheduled" | "manual"
    status: str  # "success" | "partial" | "failed"
    season: int
    round: int
    matches_processed: int
    model_version: str
    error: str | None
    summary: dict[str, Any]  # flips, mean_abs_delta, max_abs_delta
    changes: list[JobRunChange]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "trigger": self.trigger,
            "status": self.status,
            "season": self.season,
            "round": self.round,
            "matches_processed": self.matches_processed,
            "model_version": self.model_version,
            "error": self.error,
            "summary": self.summary,
            "changes": [c.to_dict() for c in self.changes],
        }


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def render_summary(
    *,
    home_team: str,
    away_team: str,
    prev_home_prob: float | None,
    new_home_prob: float,
    flipped: bool,
    trigger_signal: str | None,
) -> str:
    """Deterministic one-line summary of how a single match's prediction moved."""
    new_pct = round(new_home_prob * 100)

    if prev_home_prob is None:
        fav = home_team if new_home_prob >= 0.5 else away_team
        fav_pct = new_pct if new_home_prob >= 0.5 else 100 - new_pct
        return f"Initial prediction: {fav} {fav_pct}%."

    prev_pct = round(prev_home_prob * 100)
    delta_pct = new_pct - prev_pct
    abs_delta_pct = abs(delta_pct)
    sign = "+" if delta_pct >= 0 else "-"
    signal_clause = _SIGNAL_CLAUSE.get(trigger_signal or "", "")
    abs_delta = abs(new_home_prob - prev_home_prob)

    if flipped:
        new_fav = home_team if new_home_prob >= 0.5 else away_team
        return (
            f"Predicted winner flipped to {new_fav} "
            f"({prev_pct}% → {new_pct}% home, {sign}{abs_delta_pct}pt){signal_clause}."
        )
    if abs_delta >= 0.05:
        verb = "firmed" if delta_pct > 0 else "drifted"
        return (
            f"{home_team} {verb}: {prev_pct}% → {new_pct}% home "
            f"({sign}{abs_delta_pct}pt){signal_clause}."
        )
    if abs_delta >= 0.01:
        return (
            f"Small drift: {prev_pct}% → {new_pct}% home ({sign}{abs_delta_pct}pt){signal_clause}."
        )
    return f"No material change (~{new_pct}% home)."


def detect_trigger_signal(
    *,
    prev_model_version: str | None,
    new_model_version: str,
    prev_team_list_hash: str | None,
    new_team_list_hash: str | None,
    prev_weather_fetched_at: str | None,
    new_weather_fetched_at: str | None,
) -> str | None:
    """Pick the most likely driver of a prediction change.

    Priority retrain > team_list > weather. None means no detectable change
    in tracked inputs (model_version, team list, weather). Signals require
    *both* sides — first-ever observation can't be a "change".
    """
    if prev_model_version is not None and prev_model_version != new_model_version:
        return "retrain"
    if (
        prev_team_list_hash is not None
        and new_team_list_hash is not None
        and prev_team_list_hash != new_team_list_hash
    ):
        return "team_list"
    if (
        prev_weather_fetched_at is not None
        and new_weather_fetched_at is not None
        and prev_weather_fetched_at != new_weather_fetched_at
    ):
        return "weather"
    return None


def build_change_entries(
    *,
    new_predictions: list[PredictionOut],
    prev_predictions_by_match: dict[int, PredictionOut],
    prev_changes_by_match: dict[int, dict[str, Any]],
    new_team_list_hashes: dict[int, str | None],
    new_weather_fetched_at: dict[int, str | None],
) -> list[JobRunChange]:
    """Build one JobRunChange per new prediction.

    ``prev_predictions_by_match`` is the prior ``predictions`` doc, indexed by
    match_id — the source of truth for prev_home_prob and prev_model_version.
    ``prev_changes_by_match`` is the prior ``job_runs`` doc's changes[] indexed
    by match_id — the only place we have prev team_list_hash / weather_fetched_at
    for signal detection.
    """
    out: list[JobRunChange] = []
    for p in new_predictions:
        prev_pred = prev_predictions_by_match.get(p.matchId)
        prev_change = prev_changes_by_match.get(p.matchId)

        prev_home_prob = prev_pred.homeWinProbability if prev_pred is not None else None
        prev_model_version = prev_pred.modelVersion if prev_pred is not None else None
        prev_team_list_hash = prev_change.get("team_list_hash") if prev_change else None
        prev_weather_fetched_at = prev_change.get("weather_fetched_at") if prev_change else None

        new_team_list_hash = new_team_list_hashes.get(p.matchId)
        new_weather = new_weather_fetched_at.get(p.matchId)

        if prev_home_prob is None:
            delta = 0.0
            flipped = False
        else:
            delta = p.homeWinProbability - prev_home_prob
            prev_winner = "home" if prev_home_prob >= 0.5 else "away"
            flipped = prev_winner != p.predictedWinner

        signal = detect_trigger_signal(
            prev_model_version=prev_model_version,
            new_model_version=p.modelVersion,
            prev_team_list_hash=prev_team_list_hash,
            new_team_list_hash=new_team_list_hash,
            prev_weather_fetched_at=prev_weather_fetched_at,
            new_weather_fetched_at=new_weather,
        )

        summary = render_summary(
            home_team=p.home.name,
            away_team=p.away.name,
            prev_home_prob=prev_home_prob,
            new_home_prob=p.homeWinProbability,
            flipped=flipped,
            trigger_signal=signal,
        )

        out.append(
            JobRunChange(
                match_id=p.matchId,
                home_team=p.home.name,
                away_team=p.away.name,
                prev_home_prob=prev_home_prob,
                new_home_prob=p.homeWinProbability,
                delta=round(delta, 4),
                flipped=flipped,
                trigger_signal=signal,
                summary=summary,
                model_version=p.modelVersion,
                team_list_hash=new_team_list_hash,
                weather_fetched_at=new_weather,
            )
        )
    return out


def aggregate_summary(changes: list[JobRunChange]) -> dict[str, Any]:
    """Top-level run summary — drives the list-view row."""
    deltas = [abs(c.delta) for c in changes if c.prev_home_prob is not None]
    flips = sum(1 for c in changes if c.flipped)
    return {
        "flips": flips,
        "mean_abs_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
        "max_abs_delta": round(max(deltas), 4) if deltas else 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers used by the writer in _run_precompute
# ---------------------------------------------------------------------------


def hash_team_lists(team_list_repo: Any, match_id: int) -> str | None:
    """Best-effort short hash of all team-list snapshots for *match_id*.

    Concatenates each snapshot's ``(team_id, scraped_at, players_json)`` tuple
    and hashes — order-stable because ``list_snapshots`` returns rows ordered
    by scraped_at. None on any error or when the repo doesn't expose the
    expected method, so trigger detection gracefully degrades to null.
    """
    if team_list_repo is None or not hasattr(team_list_repo, "list_snapshots"):
        return None
    try:
        snaps = team_list_repo.list_snapshots(match_id)
    except Exception:  # noqa: BLE001
        return None
    if not snaps:
        return None
    h = hashlib.sha256()
    for s in snaps:
        try:
            payload = (
                f"{s.team_id}|{s.scraped_at.isoformat()}|"
                f"{sorted((p.player_id, p.position) for p in s.players)}"
            )
        except Exception:  # noqa: BLE001
            continue
        h.update(payload.encode())
    return h.hexdigest()[:16]


def weather_fetched_at_iso(weather_forecast: Any) -> str | None:
    """Pull a stable timestamp off the WeatherForecast object for diffing.

    None on missing attribute; we treat that as "no weather signal", which is
    equivalent to "not changed" for trigger-signal purposes.
    """
    if weather_forecast is None:
        return None
    fa = getattr(weather_forecast, "fetched_at", None)
    if fa is None:
        return None
    if isinstance(fa, datetime):
        return fa.isoformat()
    return str(fa)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class JobRunStore:
    """Firestore-backed audit log for precompute runs."""

    def __init__(
        self,
        client: Any = None,
        project: str | None = None,
        database: str = "(default)",
    ) -> None:
        if client is not None:
            self._db = client
        else:
            from google.cloud import firestore  # noqa: PLC0415

            self._db = firestore.Client(project=project, database=database)

    def add(self, record: JobRunRecord) -> str:
        """Write a new run record. Returns the auto-generated doc id."""
        ref = self._db.collection(_COLLECTION).document()
        ref.set(record.to_dict())
        return ref.id

    def latest_changes_by_match(self, season: int, round_: int) -> dict[int, dict[str, Any]]:
        """Return the most recent run's changes[] for *(season, round_)*,
        indexed by match_id, so the next run can compute trigger signals.

        Returns an empty dict on miss or any Firestore error. The diff still
        works without it — only signal detection degrades to null.
        """
        col = self._db.collection(_COLLECTION)
        try:
            from google.cloud.firestore_v1 import Query  # noqa: PLC0415

            direction: Any = Query.DESCENDING
        except Exception:  # noqa: BLE001
            direction = "DESCENDING"
        try:
            query = (
                col.where("season", "==", season)
                .where("round", "==", round_)
                .order_by("started_at", direction=direction)
                .limit(1)
            )
            docs = list(query.stream())
        except Exception:  # noqa: BLE001
            _logger.warning("job_runs latest lookup failed; signals will be null", exc_info=True)
            return {}
        if not docs:
            return {}
        doc = docs[0].to_dict() or {}
        return {c["match_id"]: c for c in doc.get("changes", [])}

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent run docs (without changes[]) for the list view.

        Strips the ``changes`` array to keep the API payload small — detail
        callers go through :meth:`get` to load the full doc.
        """
        col = self._db.collection(_COLLECTION)
        try:
            from google.cloud.firestore_v1 import Query  # noqa: PLC0415

            direction: Any = Query.DESCENDING
        except Exception:  # noqa: BLE001
            direction = "DESCENDING"
        try:
            query = col.order_by("started_at", direction=direction).limit(limit)
            docs = list(query.stream())
        except Exception:  # noqa: BLE001
            _logger.warning("job_runs list_recent failed", exc_info=True)
            return []
        out: list[dict[str, Any]] = []
        for d in docs:
            data = d.to_dict() or {}
            data["id"] = d.id
            data.pop("changes", None)
            out.append(data)
        return out

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return the full run doc by id (including changes[])."""
        try:
            snap = self._db.collection(_COLLECTION).document(run_id).get()
        except Exception:  # noqa: BLE001
            _logger.warning("job_runs get failed for %s", run_id, exc_info=True)
            return None
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        data["id"] = run_id
        return data
