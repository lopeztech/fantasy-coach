"""Tests for referee extraction, storage migration, and feature engineering."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime

import pytest

from fantasy_coach.feature_engineering import FEATURE_NAMES, REF_SHRINKAGE_N, FeatureBuilder
from fantasy_coach.features import MatchRow, TeamRow, _extract_referee_ids, extract_match_features
from fantasy_coach.storage.sqlite import SQLiteRepository

# ---------------------------------------------------------------------------
# _extract_referee_ids
# ---------------------------------------------------------------------------


def test_extract_referee_ids_from_officials_array() -> None:
    officials = [
        {"profileId": 500039, "position": "Referee", "firstName": "Ashley", "lastName": "Klein"},
        {
            "profileId": 500068,
            "position": "Touch Judge",
            "firstName": "Chris",
            "lastName": "Sutton",
        },
        {
            "profileId": 500004,
            "position": "Senior Review Official",
            "firstName": "Gerard",
            "lastName": "Sutton",
        },
    ]
    ref_id, vref_id = _extract_referee_ids(officials)
    assert ref_id == 500039
    assert vref_id == 500004


def test_extract_referee_ids_no_officials() -> None:
    ref_id, vref_id = _extract_referee_ids([])
    assert ref_id is None
    assert vref_id is None


def test_extract_referee_ids_only_touch_judges() -> None:
    officials = [
        {"profileId": 1, "position": "Touch Judge", "firstName": "A", "lastName": "B"},
    ]
    ref_id, vref_id = _extract_referee_ids(officials)
    assert ref_id is None
    assert vref_id is None


def test_extract_referee_ids_missing_profile_id() -> None:
    officials = [{"position": "Referee", "firstName": "Unknown", "lastName": "Official"}]
    ref_id, vref_id = _extract_referee_ids(officials)
    assert ref_id is None


def test_extract_referee_ids_takes_first_referee() -> None:
    officials = [
        {"profileId": 111, "position": "Referee"},
        {"profileId": 222, "position": "Referee"},
    ]
    ref_id, _ = _extract_referee_ids(officials)
    assert ref_id == 111


# ---------------------------------------------------------------------------
# extract_match_features — referee fields
# ---------------------------------------------------------------------------


def test_extract_match_features_captures_referee_ids() -> None:
    raw = {
        "matchId": "1",
        "roundNumber": "1",
        "startTime": "2024-03-03T02:30:00Z",
        "matchState": "FullTime",
        "homeTeam": {"teamId": "1", "name": "Home", "nickName": "H", "score": 20, "players": []},
        "awayTeam": {"teamId": "2", "name": "Away", "nickName": "A", "score": 10, "players": []},
        "officials": [
            {"profileId": 500039, "position": "Referee", "firstName": "X", "lastName": "Y"},
            {
                "profileId": 500004,
                "position": "Senior Review Official",
                "firstName": "A",
                "lastName": "B",
            },
        ],
    }
    row = extract_match_features(raw)
    assert row.referee_id == 500039
    assert row.video_referee_id == 500004


def test_extract_match_features_no_officials_block() -> None:
    raw = {
        "matchId": "2",
        "roundNumber": "1",
        "startTime": "2026-05-01T05:00:00Z",
        "matchState": "Upcoming",
        "homeTeam": {"teamId": "1", "name": "Home", "nickName": "H", "score": None, "players": []},
        "awayTeam": {"teamId": "2", "name": "Away", "nickName": "A", "score": None, "players": []},
    }
    row = extract_match_features(raw)
    assert row.referee_id is None
    assert row.video_referee_id is None


# ---------------------------------------------------------------------------
# SQLite schema migration: v1 → v2
# ---------------------------------------------------------------------------


def _create_v1_db(path: str) -> None:
    """Create a minimal v1 schema DB (without referee columns)."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version VALUES (1);
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY,
            season INTEGER NOT NULL,
            round INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            match_state TEXT NOT NULL,
            venue TEXT, venue_city TEXT, weather TEXT,
            home_team_id INTEGER NOT NULL, home_name TEXT NOT NULL,
            home_nick TEXT NOT NULL, home_score INTEGER,
            away_team_id INTEGER NOT NULL, away_name TEXT NOT NULL,
            away_nick TEXT NOT NULL, away_score INTEGER
        );
        CREATE TABLE IF NOT EXISTS match_players (
            match_id INTEGER NOT NULL, side TEXT NOT NULL,
            player_id INTEGER NOT NULL, jersey_number INTEGER,
            position TEXT, first_name TEXT, last_name TEXT,
            PRIMARY KEY (match_id, side, player_id)
        );
        CREATE TABLE IF NOT EXISTS match_team_stats (
            match_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
            title TEXT NOT NULL, type TEXT NOT NULL,
            units TEXT, home_value REAL, away_value REAL,
            PRIMARY KEY (match_id, ordinal)
        );
    """)
    conn.execute(
        """
        INSERT INTO matches (match_id, season, round, start_time, match_state,
            home_team_id, home_name, home_nick, away_team_id, away_name, away_nick)
        VALUES (42, 2024, 1, '2024-03-03T02:30:00+00:00', 'FullTime', 1, 'H', 'H', 2, 'A', 'A')
        """
    )
    conn.commit()
    conn.close()


def test_sqlite_v1_to_v2_migration() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name

    _create_v1_db(path)
    repo = SQLiteRepository(path)
    match = repo.get_match(42)
    repo.close()

    assert match is not None
    assert match.referee_id is None
    assert match.video_referee_id is None


def test_sqlite_upsert_and_retrieve_referee_ids() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name

    row = MatchRow(
        match_id=99,
        season=2025,
        round=3,
        start_time=datetime(2025, 4, 1, 19, 0, tzinfo=UTC),
        match_state="FullTime",
        venue="Suncorp Stadium",
        venue_city="Brisbane",
        weather=None,
        home=TeamRow(team_id=1, name="Broncos", nick_name="BRI", score=24, players=[]),
        away=TeamRow(team_id=2, name="Storm", nick_name="MEL", score=18, players=[]),
        team_stats=[],
        referee_id=500039,
        video_referee_id=500004,
    )

    repo = SQLiteRepository(path)
    repo.upsert_match(row)
    loaded = repo.get_match(99)
    repo.close()

    assert loaded is not None
    assert loaded.referee_id == 500039
    assert loaded.video_referee_id == 500004


# ---------------------------------------------------------------------------
# FeatureBuilder — referee feature indices and values
# ---------------------------------------------------------------------------

_REF_IDX = list(FEATURE_NAMES).index("ref_avg_total_points")
_PD_IDX = list(FEATURE_NAMES).index("ref_home_penalty_diff")
_MISS_IDX = list(FEATURE_NAMES).index("missing_referee")


def _team(tid: int, score: int | None = None) -> TeamRow:
    return TeamRow(team_id=tid, name=f"T{tid}", nick_name=f"T{tid}", score=score, players=[])


def _match(
    mid: int,
    ts: datetime,
    hscore: int,
    ascore: int,
    referee_id: int | None = None,
) -> MatchRow:
    from fantasy_coach.features import TeamStat

    stats = []
    if referee_id is not None:
        stats = [
            TeamStat(
                title="Penalties Conceded",
                type="Number",
                units=None,
                home_value=4.0,
                away_value=6.0,
            )
        ]
    return MatchRow(
        match_id=mid,
        season=2025,
        round=1,
        start_time=ts,
        match_state="FullTime",
        venue="Suncorp Stadium",
        venue_city="Brisbane",
        weather=None,
        home=_team(1, hscore),
        away=_team(2, ascore),
        team_stats=stats,
        referee_id=referee_id,
    )


def test_missing_referee_flag_when_no_referee_id() -> None:
    builder = FeatureBuilder()
    match = MatchRow(
        match_id=1,
        season=2025,
        round=1,
        start_time=datetime(2025, 3, 1, 19, 0, tzinfo=UTC),
        match_state="Upcoming",
        venue="Suncorp",
        venue_city="Brisbane",
        weather=None,
        home=_team(1),
        away=_team(2),
        team_stats=[],
        referee_id=None,
    )
    row = builder.feature_row(match)
    assert row[_MISS_IDX] == pytest.approx(1.0)


def test_no_missing_flag_when_referee_known() -> None:
    builder = FeatureBuilder()
    t = datetime(2025, 3, 1, 19, 0, tzinfo=UTC)
    m = _match(1, t, 30, 10, referee_id=500039)
    row = builder.feature_row(m)
    assert row[_MISS_IDX] == pytest.approx(0.0)


def test_ref_avg_total_points_starts_at_league_mean() -> None:
    """Before any matches, falls back to league mean (0 when no history at all)."""
    builder = FeatureBuilder()
    t = datetime(2025, 3, 1, 19, 0, tzinfo=UTC)
    row = builder.feature_row(_match(1, t, 0, 0, referee_id=500039))
    assert row[_REF_IDX] == pytest.approx(0.0)


def test_ref_avg_total_points_updates_after_record() -> None:
    builder = FeatureBuilder()
    t0 = datetime(2025, 3, 1, 19, 0, tzinfo=UTC)
    t1 = datetime(2025, 3, 8, 19, 0, tzinfo=UTC)

    m1 = _match(1, t0, 30, 10, referee_id=500039)  # total=40
    builder.record(m1)

    m2 = _match(2, t1, 20, 20, referee_id=500039)
    row = builder.feature_row(m2)
    assert row[_REF_IDX] == pytest.approx(40.0)


def test_ref_avg_shrinks_toward_league_mean_with_few_matches() -> None:
    builder = FeatureBuilder()
    base = datetime(2025, 3, 1, 19, 0, tzinfo=UTC)
    import datetime as dt

    # Record 2 matches with referee 500039 (total=40 each)
    for i in range(2):
        ts = base + dt.timedelta(weeks=i)
        m = _match(i + 1, ts, 30, 10, referee_id=500039)
        builder.record(m)

    # Record 5 matches with a different referee (total=20 each) — these only
    # contribute to the league mean, not to ref 500039's history.
    for i in range(5):
        ts = base + dt.timedelta(weeks=10 + i)
        m = _match(100 + i, ts, 10, 10, referee_id=999)
        builder.record(m)

    # League total deque has 7 matches: 2×40 + 5×20 = 180, mean = 180/7 ≈ 25.71
    # ref 500039 has 2 matches < REF_SHRINKAGE_N=10, so w = 2/10 = 0.2
    # shrunk = 0.2 * 40 + 0.8 * 25.71 ≈ 8.0 + 20.57 ≈ 28.57
    ts_next = base + dt.timedelta(weeks=20)
    m_next = _match(200, ts_next, 0, 0, referee_id=500039)
    row = builder.feature_row(m_next)

    league_mean = (2 * 40 + 5 * 20) / 7
    w = 2 / REF_SHRINKAGE_N
    expected_shrunk = w * 40.0 + (1 - w) * league_mean
    assert row[_REF_IDX] == pytest.approx(expected_shrunk, abs=0.01)


def test_ref_home_penalty_diff_accumulates() -> None:
    builder = FeatureBuilder()
    t0 = datetime(2025, 3, 1, 19, 0, tzinfo=UTC)
    t1 = datetime(2025, 3, 8, 19, 0, tzinfo=UTC)

    # home_penalties=4, away_penalties=6 → diff = -2
    m1 = _match(1, t0, 20, 18, referee_id=500039)
    builder.record(m1)

    m2 = _match(2, t1, 20, 18, referee_id=500039)
    row = builder.feature_row(m2)
    assert row[_PD_IDX] == pytest.approx(-2.0)  # home_value - away_value = 4 - 6


def test_ref_penalty_diff_zero_when_no_stat() -> None:
    builder = FeatureBuilder()
    t = datetime(2025, 3, 1, 19, 0, tzinfo=UTC)
    # no team_stats → no penalty data
    m = MatchRow(
        match_id=1,
        season=2025,
        round=1,
        start_time=t,
        match_state="FullTime",
        venue="V",
        venue_city="C",
        weather=None,
        home=_team(1, 20),
        away=_team(2, 18),
        team_stats=[],
        referee_id=500039,
    )
    builder.record(m)
    m2 = MatchRow(
        match_id=2,
        season=2025,
        round=2,
        start_time=t + __import__("datetime").timedelta(weeks=1),
        match_state="Upcoming",
        venue="V",
        venue_city="C",
        weather=None,
        home=_team(1),
        away=_team(2),
        team_stats=[],
        referee_id=500039,
    )
    row = builder.feature_row(m2)
    assert row[_PD_IDX] == pytest.approx(0.0)


def test_feature_names_has_referee_features() -> None:
    assert "ref_avg_total_points" in FEATURE_NAMES
    assert "ref_home_penalty_diff" in FEATURE_NAMES
    assert "missing_referee" in FEATURE_NAMES
    assert len(FEATURE_NAMES) == 18
