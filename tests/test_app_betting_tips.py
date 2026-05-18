"""Endpoint tests for ``GET /betting-tips``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fantasy_coach import app as app_module
from fantasy_coach.betting.models import BettingTipsOut, Tip, TipLeg


class _FakeStore:
    def __init__(self) -> None:
        self.payload: BettingTipsOut | None = None

    def get(self, season: int, round_: int) -> BettingTipsOut | None:
        if self.payload and self.payload.season == season and self.payload.round == round_:
            return self.payload
        return None


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> _FakeStore:
    fake = _FakeStore()
    # Force the lazy singleton path to use our fake.
    monkeypatch.setattr(app_module, "_betting_tips_store", fake)
    return fake


def test_returns_503_when_cache_empty(fake_store: _FakeStore) -> None:
    client = TestClient(app_module.app)
    res = client.get("/betting-tips?season=2026&round=99")
    assert res.status_code == 503
    assert "Retry-After" in res.headers
    assert res.headers["Retry-After"] == "3600"
    body = res.json()
    assert "season 2026 round 99" in body["detail"]


def test_returns_200_with_payload_when_cache_hit(fake_store: _FakeStore) -> None:
    fake_store.payload = BettingTipsOut(
        season=2026,
        round=7,
        generatedAt="2026-04-29T12:00:00+00:00",
        oddsSnapshotAt=None,
        singles=[
            Tip(
                kind="single",
                legs=[
                    TipLeg(
                        matchId=111,
                        homeName="Storm",
                        awayName="Eels",
                        kickoff="2026-05-01T08:00:00+00:00",
                        market="h2h",
                        selectionCode="home",
                        selection="Storm",
                        line=None,
                        decimalOdds=1.85,
                        modelProb=0.72,
                        impliedProb=0.52,
                        edge=0.20,
                        bookmaker="nrl.com",
                    )
                ],
                combinedOdds=1.85,
                combinedModelProb=0.72,
                combinedImpliedProb=0.52,
                edge=0.20,
                suggestedUnits=1.0,
            )
        ],
        doubles=[],
        trebles=[],
    )

    client = TestClient(app_module.app)
    res = client.get("/betting-tips?season=2026&round=7")
    assert res.status_code == 200
    body = res.json()
    assert body["season"] == 2026
    assert body["round"] == 7
    assert len(body["singles"]) == 1
    leg = body["singles"][0]["legs"][0]
    assert leg["selection"] == "Storm"
    assert leg["market"] == "h2h"
    assert "caveat" in body
