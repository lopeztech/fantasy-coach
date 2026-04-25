from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fantasy_coach.features import (
    MatchRow,
    PlayerMatchStat,
    PlayerRow,
    TeamRow,
    TeamStat,
    extract_match_features,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_extract_2024_full_time_match() -> None:
    raw = _load("match-2024-rd1-sea-eagles-v-rabbitohs.json")

    row = extract_match_features(raw)

    assert isinstance(row, MatchRow)
    assert row.match_id == 20241110110
    assert row.season == 2024
    assert row.round == 1
    assert row.start_time == datetime(2024, 3, 3, 2, 30, tzinfo=UTC)
    assert row.match_state == "FullTime"
    assert row.venue == "Allegiant Stadium"
    assert row.weather is None  # absent from 2024 payloads

    assert isinstance(row.home, TeamRow)
    assert row.home.team_id == 500002
    assert row.home.name == "Manly-Warringah Sea Eagles"
    assert row.home.nick_name == "Sea Eagles"
    assert row.home.score == 36
    assert len(row.home.players) == 18

    assert row.away.team_id == 500005
    assert row.away.score == 24

    fullback = row.home.players[0]
    assert isinstance(fullback, PlayerRow)
    assert fullback.player_id == 501505
    assert fullback.jersey_number == 1
    assert fullback.position == "Fullback"
    assert fullback.first_name == "Tom"
    assert fullback.last_name == "Trbojevic"

    assert row.team_stats, "expected team-level stats for a completed match"
    possession = next(s for s in row.team_stats if s.title == "Possession %")
    assert isinstance(possession, TeamStat)
    assert possession.home_value == 51.0
    assert possession.away_value == 49.0


def test_extract_2026_upcoming_match_has_optional_fields() -> None:
    raw = _load("match-2026-rd8-wests-tigers-v-raiders.json")

    row = extract_match_features(raw)

    assert row.match_id == 20261110810
    assert row.season == 2026
    assert row.round == 8
    assert row.match_state == "Upcoming"
    assert row.weather == "Fine"  # 2026 payloads include weather

    # Upcoming matches have no scores or rosters yet — extractor must accept this.
    assert row.home.score is None
    assert row.away.score is None
    assert row.home.players == []
    assert row.away.players == []
    assert row.home_player_stats == []
    assert row.away_player_stats == []


def test_extract_player_match_stats_fulltime() -> None:
    raw = _load("match-2024-rd1-sea-eagles-v-rabbitohs.json")

    row = extract_match_features(raw)

    assert len(row.home_player_stats) == 18
    assert len(row.away_player_stats) == 18

    fullback = row.home_player_stats[0]
    assert isinstance(fullback, PlayerMatchStat)
    assert fullback.player_id == 501505
    assert fullback.minutes_played == 80
    assert fullback.all_run_metres == 224
    assert fullback.tackles_made == 5
    assert fullback.missed_tackles == 2
    assert fullback.tackle_breaks == 3
    assert fullback.line_breaks == 1
    assert fullback.try_assists == 1
    assert fullback.offloads == 4
    assert fullback.errors == 3
    assert fullback.tries == 0
    assert fullback.tackle_efficiency == pytest.approx(71.43)
    assert fullback.fantasy_points_total == 54


def test_extract_player_match_stats_finals() -> None:
    raw = _load("match-2024-finals-week-1-game-1.json")

    row = extract_match_features(raw)

    # Finals fixtures use the same per-player stats schema as regular rounds.
    assert len(row.home_player_stats) == 18
    assert len(row.away_player_stats) == 18
    assert all(p.player_id > 0 for p in row.home_player_stats)
    assert all(p.minutes_played is not None for p in row.home_player_stats)


def test_unknown_top_level_keys_are_logged_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = _load("match-2024-rd1-sea-eagles-v-rabbitohs.json")
    raw = {**raw, "newMysteryField": {"surprise": True}}

    with caplog.at_level(logging.WARNING, logger="fantasy_coach.features"):
        row = extract_match_features(raw)

    assert isinstance(row, MatchRow)
    assert any("newMysteryField" in r.message for r in caplog.records)


def test_unknown_team_keys_are_logged_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = _load("match-2024-rd1-sea-eagles-v-rabbitohs.json")
    raw = {**raw, "homeTeam": {**raw["homeTeam"], "shinyNewKey": 1}}

    with caplog.at_level(logging.WARNING, logger="fantasy_coach.features"):
        extract_match_features(raw)

    assert any("shinyNewKey" in r.message for r in caplog.records)
