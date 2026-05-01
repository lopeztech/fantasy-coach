"""Integration tests for /job-runs endpoints (#244 PR B)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fantasy_coach.app import app
from fantasy_coach.job_runs import JobRunChange, JobRunRecord, JobRunStore

# Reuse the in-memory Firestore stub from the unit tests for the store.
from tests.test_job_runs import _FakeFirestore


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_app_singletons():
    import fantasy_coach.app as app_module

    app_module._store = None
    app_module._job_run_store = None
    yield
    app_module._store = None
    app_module._job_run_store = None


def _record(
    *,
    season: int = 2026,
    round_: int = 9,
    started_at: datetime | None = None,
    flips: int = 1,
    summary_text: str = (
        "Predicted winner flipped to Broncos (48% → 61% home, +13pt) — team list update."
    ),
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
        summary={"flips": flips, "mean_abs_delta": 0.13, "max_abs_delta": 0.13},
        changes=[
            JobRunChange(
                match_id=42,
                home_team="Broncos",
                away_team="Storm",
                prev_home_prob=0.48,
                new_home_prob=0.61,
                delta=0.13,
                flipped=True,
                trigger_signal="team_list",
                summary=summary_text,
                model_version="mv1",
                team_list_hash="tlh1",
                weather_fetched_at=None,
            )
        ],
    )


def _populated_store() -> tuple[JobRunStore, str]:
    store = JobRunStore(client=_FakeFirestore())
    earlier_id = store.add(_record(started_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC), flips=0))
    later_id = store.add(_record(started_at=datetime(2026, 5, 3, 6, 0, tzinfo=UTC), flips=1))
    assert earlier_id != later_id
    return store, later_id


def test_list_job_runs_returns_recent_runs_in_camel_case(client: TestClient) -> None:
    store, latest_id = _populated_store()
    with patch("fantasy_coach.app._get_job_run_store", return_value=store):
        response = client.get("/job-runs?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["runs"], list)
    assert len(body["runs"]) == 2
    # Most recent first
    first = body["runs"][0]
    assert first["id"] == latest_id
    assert first["startedAt"] == "2026-05-03T06:00:00+00:00"
    assert first["matchesProcessed"] == 1
    assert first["modelVersion"] == "mv1"
    # Aggregate summary is camelCase
    assert first["summary"] == {"flips": 1, "meanAbsDelta": 0.13, "maxAbsDelta": 0.13}
    # changes[] is intentionally absent in the list view
    assert "changes" not in first


def test_list_job_runs_empty_when_backend_unavailable(client: TestClient) -> None:
    """Local SQLite dev returns an empty list, not 500."""
    with patch("fantasy_coach.app._get_job_run_store", return_value=None):
        response = client.get("/job-runs")
    assert response.status_code == 200
    assert response.json() == {"runs": []}


def test_get_job_run_returns_full_doc_with_changes(client: TestClient) -> None:
    store, latest_id = _populated_store()
    with patch("fantasy_coach.app._get_job_run_store", return_value=store):
        response = client.get(f"/job-runs/{latest_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == latest_id
    assert body["summary"]["flips"] == 1
    assert len(body["changes"]) == 1
    change = body["changes"][0]
    assert change["matchId"] == 42
    assert change["homeTeam"] == "Broncos"
    assert change["prevHomeProb"] == 0.48
    assert change["newHomeProb"] == 0.61
    assert change["flipped"] is True
    assert change["triggerSignal"] == "team_list"
    assert "team list update" in change["summary"]


def test_get_job_run_returns_404_when_id_unknown(client: TestClient) -> None:
    store, _ = _populated_store()
    with patch("fantasy_coach.app._get_job_run_store", return_value=store):
        response = client.get("/job-runs/does-not-exist")
    assert response.status_code == 404


def test_get_job_run_returns_404_when_backend_unavailable(client: TestClient) -> None:
    """SQLite dev: detail endpoint 404s instead of returning a misleading shape."""
    with patch("fantasy_coach.app._get_job_run_store", return_value=None):
        response = client.get("/job-runs/anything")
    assert response.status_code == 404
