"""Tests for the precompute job-run audit log (#244)."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from typing import Any

import pytest

from fantasy_coach.job_runs import (
    JobRunChange,
    JobRunRecord,
    JobRunStore,
    aggregate_summary,
    build_change_entries,
    detect_trigger_signal,
    render_summary,
)
from fantasy_coach.predictions import PredictionOut, TeamInfo


def _pred(
    *,
    match_id: int = 1,
    home_name: str = "Broncos",
    away_name: str = "Storm",
    home_prob: float = 0.6,
    winner: str = "home",
    model_version: str = "mv1",
) -> PredictionOut:
    return PredictionOut(
        matchId=match_id,
        home=TeamInfo(id=10, name=home_name),
        away=TeamInfo(id=20, name=away_name),
        kickoff="2026-05-01T08:00:00+00:00",
        predictedWinner=winner,
        homeWinProbability=home_prob,
        modelVersion=model_version,
        featureHash="fh1",
    )


# ---------------------------------------------------------------------------
# render_summary
# ---------------------------------------------------------------------------


def test_render_summary_initial_prediction_no_prev() -> None:
    out = render_summary(
        home_team="Broncos",
        away_team="Storm",
        prev_home_prob=None,
        new_home_prob=0.62,
        flipped=False,
        trigger_signal=None,
    )
    assert out == "Initial prediction: Broncos 62%."


def test_render_summary_initial_prediction_picks_away_when_underdog() -> None:
    out = render_summary(
        home_team="Broncos",
        away_team="Storm",
        prev_home_prob=None,
        new_home_prob=0.30,
        flipped=False,
        trigger_signal=None,
    )
    assert out == "Initial prediction: Storm 70%."


def test_render_summary_flip_with_signal() -> None:
    out = render_summary(
        home_team="Broncos",
        away_team="Storm",
        prev_home_prob=0.48,
        new_home_prob=0.61,
        flipped=True,
        trigger_signal="team_list",
    )
    assert out == "Predicted winner flipped to Broncos (48% → 61% home, +13pt) — team list update."


def test_render_summary_flip_to_away() -> None:
    out = render_summary(
        home_team="Broncos",
        away_team="Storm",
        prev_home_prob=0.55,
        new_home_prob=0.40,
        flipped=True,
        trigger_signal=None,
    )
    assert out == "Predicted winner flipped to Storm (55% → 40% home, -15pt)."


def test_render_summary_significant_firmed_with_retrain_signal() -> None:
    out = render_summary(
        home_team="Broncos",
        away_team="Storm",
        prev_home_prob=0.55,
        new_home_prob=0.61,
        flipped=False,
        trigger_signal="retrain",
    )
    assert out == "Broncos firmed: 55% → 61% home (+6pt) — model retrained."


def test_render_summary_significant_drifted() -> None:
    out = render_summary(
        home_team="Broncos",
        away_team="Storm",
        prev_home_prob=0.65,
        new_home_prob=0.58,
        flipped=False,
        trigger_signal=None,
    )
    assert out == "Broncos drifted: 65% → 58% home (-7pt)."


def test_render_summary_small_drift_with_weather_signal() -> None:
    out = render_summary(
        home_team="Broncos",
        away_team="Storm",
        prev_home_prob=0.62,
        new_home_prob=0.65,
        flipped=False,
        trigger_signal="weather",
    )
    assert out == "Small drift: 62% → 65% home (+3pt) — weather update."


def test_render_summary_no_material_change_below_threshold() -> None:
    out = render_summary(
        home_team="Broncos",
        away_team="Storm",
        prev_home_prob=0.605,
        new_home_prob=0.610,
        flipped=False,
        trigger_signal=None,
    )
    assert out == "No material change (~61% home)."


# ---------------------------------------------------------------------------
# detect_trigger_signal
# ---------------------------------------------------------------------------


def test_detect_trigger_signal_retrain_wins_priority() -> None:
    assert (
        detect_trigger_signal(
            prev_model_version="mv1",
            new_model_version="mv2",
            prev_team_list_hash="abc",
            new_team_list_hash="def",
            prev_weather_fetched_at="t0",
            new_weather_fetched_at="t1",
        )
        == "retrain"
    )


def test_detect_trigger_signal_team_list_when_only_lineup_changed() -> None:
    assert (
        detect_trigger_signal(
            prev_model_version="mv1",
            new_model_version="mv1",
            prev_team_list_hash="abc",
            new_team_list_hash="def",
            prev_weather_fetched_at="t0",
            new_weather_fetched_at="t0",
        )
        == "team_list"
    )


def test_detect_trigger_signal_weather_when_only_forecast_changed() -> None:
    assert (
        detect_trigger_signal(
            prev_model_version="mv1",
            new_model_version="mv1",
            prev_team_list_hash="abc",
            new_team_list_hash="abc",
            prev_weather_fetched_at="t0",
            new_weather_fetched_at="t1",
        )
        == "weather"
    )


def test_detect_trigger_signal_null_when_first_observation() -> None:
    """No prev side means we can't call it a 'change' — degrade to null."""
    assert (
        detect_trigger_signal(
            prev_model_version=None,
            new_model_version="mv1",
            prev_team_list_hash=None,
            new_team_list_hash="abc",
            prev_weather_fetched_at=None,
            new_weather_fetched_at="t0",
        )
        is None
    )


def test_detect_trigger_signal_null_when_nothing_changed() -> None:
    assert (
        detect_trigger_signal(
            prev_model_version="mv1",
            new_model_version="mv1",
            prev_team_list_hash="abc",
            new_team_list_hash="abc",
            prev_weather_fetched_at="t0",
            new_weather_fetched_at="t0",
        )
        is None
    )


# ---------------------------------------------------------------------------
# build_change_entries
# ---------------------------------------------------------------------------


def test_build_change_entries_no_prev_emits_initial_summary() -> None:
    new = [_pred(match_id=1, home_prob=0.6)]
    changes = build_change_entries(
        new_predictions=new,
        prev_predictions_by_match={},
        prev_changes_by_match={},
        new_team_list_hashes={1: "tlh1"},
        new_weather_fetched_at={1: None},
    )
    assert len(changes) == 1
    c = changes[0]
    assert c.prev_home_prob is None
    assert c.delta == 0.0
    assert c.flipped is False
    assert c.trigger_signal is None
    assert c.summary.startswith("Initial prediction:")
    # Metadata stored for the next run's signal detection
    assert c.team_list_hash == "tlh1"


def test_build_change_entries_flip_detection() -> None:
    prev = _pred(match_id=1, home_prob=0.48, winner="away")
    new = _pred(match_id=1, home_prob=0.61, winner="home")
    changes = build_change_entries(
        new_predictions=[new],
        prev_predictions_by_match={1: prev},
        prev_changes_by_match={
            1: {
                "model_version": "mv1",
                "team_list_hash": "tlh_old",
                "weather_fetched_at": None,
            }
        },
        new_team_list_hashes={1: "tlh_new"},
        new_weather_fetched_at={1: None},
    )
    c = changes[0]
    assert c.flipped is True
    assert c.delta == pytest.approx(0.13)
    assert c.trigger_signal == "team_list"
    assert "flipped to Broncos" in c.summary
    assert "team list update" in c.summary


def test_build_change_entries_retrain_signal_takes_priority() -> None:
    prev = _pred(match_id=1, home_prob=0.55, model_version="mv1")
    new = _pred(match_id=1, home_prob=0.62, model_version="mv2")
    changes = build_change_entries(
        new_predictions=[new],
        prev_predictions_by_match={1: prev},
        prev_changes_by_match={
            1: {
                "model_version": "mv1",
                "team_list_hash": "tlh_old",
                "weather_fetched_at": "t0",
            }
        },
        new_team_list_hashes={1: "tlh_new"},
        new_weather_fetched_at={1: "t1"},
    )
    assert changes[0].trigger_signal == "retrain"
    assert "model retrained" in changes[0].summary


# ---------------------------------------------------------------------------
# aggregate_summary
# ---------------------------------------------------------------------------


def test_aggregate_summary_counts_flips_and_max_delta() -> None:
    changes = [
        JobRunChange(
            match_id=1,
            home_team="A",
            away_team="B",
            prev_home_prob=0.50,
            new_home_prob=0.62,
            delta=0.12,
            flipped=True,
            trigger_signal=None,
            summary="x",
            model_version="mv1",
            team_list_hash=None,
            weather_fetched_at=None,
        ),
        JobRunChange(
            match_id=2,
            home_team="C",
            away_team="D",
            prev_home_prob=0.40,
            new_home_prob=0.43,
            delta=0.03,
            flipped=False,
            trigger_signal=None,
            summary="x",
            model_version="mv1",
            team_list_hash=None,
            weather_fetched_at=None,
        ),
        # Initial prediction — excluded from delta stats.
        JobRunChange(
            match_id=3,
            home_team="E",
            away_team="F",
            prev_home_prob=None,
            new_home_prob=0.55,
            delta=0.0,
            flipped=False,
            trigger_signal=None,
            summary="x",
            model_version="mv1",
            team_list_hash=None,
            weather_fetched_at=None,
        ),
    ]
    summary = aggregate_summary(changes)
    assert summary["flips"] == 1
    assert summary["max_abs_delta"] == 0.12
    # mean over the two non-initial entries
    assert summary["mean_abs_delta"] == pytest.approx(0.075)


def test_aggregate_summary_empty_changes() -> None:
    summary = aggregate_summary([])
    assert summary == {"flips": 0, "mean_abs_delta": 0.0, "max_abs_delta": 0.0}


# ---------------------------------------------------------------------------
# JobRunStore round-trip via in-memory fake Firestore
# ---------------------------------------------------------------------------


class _FakeSnap:
    def __init__(self, doc_id: str, data: dict[str, Any] | None) -> None:
        self.id = doc_id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict[str, Any] | None:
        return self._data


class _FakeDocRef:
    def __init__(self, collection: _FakeCollection, doc_id: str) -> None:
        self._collection = collection
        self.id = doc_id

    def set(self, data: dict[str, Any]) -> None:
        self._collection._docs[self.id] = dict(data)

    def get(self) -> _FakeSnap:
        return _FakeSnap(self.id, self._collection._docs.get(self.id))


class _FakeQuery:
    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection
        self._filters: list[tuple[str, str, Any]] = []
        self._order_by: tuple[str, str] | None = None
        self._limit: int | None = None

    def where(self, field: str, op: str, value: Any) -> _FakeQuery:
        q = _FakeQuery(self._collection)
        q._filters = [*self._filters, (field, op, value)]
        q._order_by = self._order_by
        q._limit = self._limit
        return q

    def order_by(self, field: str, direction: Any = "ASCENDING") -> _FakeQuery:
        q = _FakeQuery(self._collection)
        q._filters = list(self._filters)
        q._order_by = (field, str(direction))
        q._limit = self._limit
        return q

    def limit(self, n: int) -> _FakeQuery:
        q = _FakeQuery(self._collection)
        q._filters = list(self._filters)
        q._order_by = self._order_by
        q._limit = n
        return q

    def stream(self):
        rows = [
            (doc_id, data)
            for doc_id, data in self._collection._docs.items()
            if all(self._eval_filter(data, f) for f in self._filters)
        ]
        if self._order_by is not None:
            field, direction = self._order_by
            reverse = "DESC" in direction.upper()
            rows.sort(key=lambda row: row[1].get(field, ""), reverse=reverse)
        if self._limit is not None:
            rows = rows[: self._limit]
        return iter([_FakeSnap(doc_id, data) for doc_id, data in rows])

    @staticmethod
    def _eval_filter(data: dict[str, Any], f: tuple[str, str, Any]) -> bool:
        field, op, value = f
        if op == "==":
            return data.get(field) == value
        raise NotImplementedError(op)


class _FakeCollection:
    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}
        self._ids = (f"doc_{i}" for i in itertools.count())

    def document(self, doc_id: str | None = None) -> _FakeDocRef:
        if doc_id is None:
            doc_id = next(self._ids)
        return _FakeDocRef(self, doc_id)

    def where(self, field: str, op: str, value: Any) -> _FakeQuery:
        return _FakeQuery(self).where(field, op, value)

    def order_by(self, field: str, direction: Any = "ASCENDING") -> _FakeQuery:
        return _FakeQuery(self).order_by(field, direction)


class _FakeFirestore:
    def __init__(self) -> None:
        self._collections: dict[str, _FakeCollection] = {}

    def collection(self, name: str) -> _FakeCollection:
        return self._collections.setdefault(name, _FakeCollection())


def _record(
    season: int = 2026, round_: int = 9, started_at: datetime | None = None
) -> JobRunRecord:
    return JobRunRecord(
        started_at=started_at or datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 1, 9, 5, tzinfo=UTC),
        trigger="scheduled",
        status="success",
        season=season,
        round=round_,
        matches_processed=1,
        model_version="mv1",
        error=None,
        summary={"flips": 0, "mean_abs_delta": 0.0, "max_abs_delta": 0.0},
        changes=[
            JobRunChange(
                match_id=1,
                home_team="Broncos",
                away_team="Storm",
                prev_home_prob=None,
                new_home_prob=0.6,
                delta=0.0,
                flipped=False,
                trigger_signal=None,
                summary="Initial prediction: Broncos 60%.",
                model_version="mv1",
                team_list_hash="tlh1",
                weather_fetched_at=None,
            )
        ],
    )


def test_job_run_store_add_and_get_round_trip() -> None:
    store = JobRunStore(client=_FakeFirestore())
    run_id = store.add(_record())
    fetched = store.get(run_id)
    assert fetched is not None
    assert fetched["id"] == run_id
    assert fetched["season"] == 2026
    assert fetched["changes"][0]["summary"] == "Initial prediction: Broncos 60%."


def test_job_run_store_list_recent_strips_changes_and_orders_desc() -> None:
    store = JobRunStore(client=_FakeFirestore())
    earlier = _record(started_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC))
    later = _record(started_at=datetime(2026, 5, 3, 6, 0, tzinfo=UTC))
    store.add(earlier)
    later_id = store.add(later)
    rows = store.list_recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["id"] == later_id  # ordered by started_at desc
    assert "changes" not in rows[0]
    assert "summary" in rows[0]


def test_job_run_store_latest_changes_by_match_filters_by_round() -> None:
    store = JobRunStore(client=_FakeFirestore())
    older = _record(round_=8, started_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC))
    target_round = _record(round_=9, started_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC))
    newer_target = _record(round_=9, started_at=datetime(2026, 5, 3, 6, 0, tzinfo=UTC))
    store.add(older)
    store.add(target_round)
    store.add(newer_target)
    out = store.latest_changes_by_match(2026, 9)
    assert set(out.keys()) == {1}
    # Returns the most recent — round-9 entries are identical here, but the
    # function picks the latest started_at and indexes by match_id.
    assert out[1]["team_list_hash"] == "tlh1"


def test_job_run_store_latest_changes_by_match_returns_empty_on_miss() -> None:
    store = JobRunStore(client=_FakeFirestore())
    assert store.latest_changes_by_match(2026, 9) == {}
