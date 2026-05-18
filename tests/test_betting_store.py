"""Tests for the SQLite-backed ``BettingTipsStore``.

Firestore variant is covered indirectly via ``test_app_betting_tips`` which
substitutes a fake store. The Firestore client is unmockable here without
adding ``mock-firestore`` to the dev deps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fantasy_coach.betting.models import BettingTipsOut, Tip, TipLeg
from fantasy_coach.betting.store import BettingTipsStore


def _sample_tips(season: int = 2026, round_: int = 7) -> BettingTipsOut:
    leg = TipLeg(
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
    single = Tip(
        kind="single",
        legs=[leg],
        combinedOdds=1.85,
        combinedModelProb=0.72,
        combinedImpliedProb=0.52,
        edge=0.20,
        suggestedUnits=1.0,
    )
    return BettingTipsOut(
        season=season,
        round=round_,
        generatedAt="2026-04-29T12:00:00+00:00",
        oddsSnapshotAt=None,
        singles=[single],
        doubles=[],
        trebles=[],
    )


@pytest.fixture
def store(tmp_path: Path) -> BettingTipsStore:
    return BettingTipsStore(path=tmp_path / "tips.db")


def test_get_returns_none_for_missing_round(store: BettingTipsStore) -> None:
    assert store.get(2026, 7) is None


def test_put_and_get_roundtrip(store: BettingTipsStore) -> None:
    tips = _sample_tips()
    store.put(tips)
    loaded = store.get(2026, 7)
    assert loaded is not None
    assert loaded.season == 2026
    assert loaded.round == 7
    assert len(loaded.singles) == 1
    assert loaded.singles[0].legs[0].selection == "Storm"


def test_put_replaces_existing_row(store: BettingTipsStore) -> None:
    store.put(_sample_tips())
    second = _sample_tips()
    second.singles[0].legs[0].selection = "Eels"
    store.put(second)
    loaded = store.get(2026, 7)
    assert loaded is not None
    assert loaded.singles[0].legs[0].selection == "Eels"
