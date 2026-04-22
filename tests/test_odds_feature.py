"""Tests for the bookmaker-odds feature (#26) — extraction + feature math.

The CLI (``merge-closing-lines``) has its own integration test below;
the de-vig helper is already exercised in ``test_closing_lines.py`` so
here we only check the neutral-value / missing-flag behaviour that
the feature wiring relies on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from fantasy_coach.__main__ import main
from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder, _odds_home_win_prob
from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.storage.sqlite import SQLiteRepository

# ---------------------------------------------------------------------------
# _odds_home_win_prob helper
# ---------------------------------------------------------------------------


def test_neutral_and_missing_when_either_side_absent() -> None:
    prob, missing = _odds_home_win_prob(None, 2.0)
    assert prob == 0.5
    assert missing == 1.0
    prob, missing = _odds_home_win_prob(1.8, None)
    assert prob == 0.5
    assert missing == 1.0


def test_devigs_proportionally() -> None:
    # Home 1.50 → implied 0.667; Away 2.50 → implied 0.400. Sum 1.067.
    # De-vigged home prob = 0.667 / 1.067 ≈ 0.6250.
    prob, missing = _odds_home_win_prob(1.5, 2.5)
    assert prob == pytest.approx(0.625, abs=1e-3)
    assert missing == 0.0


def test_invalid_odds_fall_back_to_neutral() -> None:
    # odds <= 1 is invalid (implies certain or better-than-certain); the
    # helper should treat it as missing rather than raise.
    prob, missing = _odds_home_win_prob(0.5, 2.0)
    assert prob == 0.5
    assert missing == 1.0


# ---------------------------------------------------------------------------
# FEATURE_NAMES wiring
# ---------------------------------------------------------------------------


def test_feature_names_include_odds() -> None:
    assert "odds_home_win_prob" in FEATURE_NAMES
    assert "missing_odds" in FEATURE_NAMES
    # Features live adjacent at the end of the tuple so older cached
    # predictions keep their column offsets stable for the first 19 entries.
    idx_prob = FEATURE_NAMES.index("odds_home_win_prob")
    idx_missing = FEATURE_NAMES.index("missing_odds")
    assert idx_missing == idx_prob + 1


def test_feature_row_reads_odds_from_match() -> None:
    """When a MatchRow carries live odds on both TeamRows, the feature row
    should surface a de-vigged probability and a 0 missing flag."""
    builder = FeatureBuilder()
    match = _mini_match(home_odds=1.8, away_odds=2.2)
    row = builder.feature_row(match)
    prob_idx = FEATURE_NAMES.index("odds_home_win_prob")
    missing_idx = FEATURE_NAMES.index("missing_odds")
    # 1/1.8 / (1/1.8 + 1/2.2) ≈ 0.550.
    assert row[prob_idx] == pytest.approx(0.5500, abs=1e-3)
    assert row[missing_idx] == 0.0


def test_feature_row_falls_back_when_odds_missing() -> None:
    builder = FeatureBuilder()
    match = _mini_match(home_odds=None, away_odds=None)
    row = builder.feature_row(match)
    prob_idx = FEATURE_NAMES.index("odds_home_win_prob")
    missing_idx = FEATURE_NAMES.index("missing_odds")
    assert row[prob_idx] == 0.5
    assert row[missing_idx] == 1.0


# ---------------------------------------------------------------------------
# merge-closing-lines CLI
# ---------------------------------------------------------------------------


def test_merge_closing_lines_updates_match_odds(tmp_path: Path) -> None:
    """End-to-end: seed a SQLite DB with one 2024 match (no odds), patch
    ``load_closing_lines`` to return a matching line, run the CLI, and
    verify the row now has both odds populated."""
    db_path = tmp_path / "nrl.db"
    repo = SQLiteRepository(db_path)
    try:
        # Round 1 2024 — Sea Eagles vs Rabbitohs. Names match the
        # bookmaker canonicaliser's known aliases.
        match = _mini_match(
            match_id=20241120240,
            season=2024,
            round_=1,
            start_time=datetime(2024, 3, 8, 8, 0, tzinfo=UTC),
            home_name="Manly-Warringah Sea Eagles",
            home_nick="Sea Eagles",
            away_name="South Sydney Rabbitohs",
            away_nick="Rabbitohs",
            home_odds=None,
            away_odds=None,
        )
        repo.upsert_match(match)
    finally:
        repo.close()

    from fantasy_coach.bookmaker.lines import ClosingLine
    from fantasy_coach.bookmaker.team_names import canonicalize

    home_canon = canonicalize("Sea Eagles")
    away_canon = canonicalize("Rabbitohs")
    assert home_canon is not None and away_canon is not None

    fake_lines = {
        (datetime(2024, 3, 8).date(), home_canon, away_canon): ClosingLine(
            match_date=datetime(2024, 3, 8).date(),
            home_canonical=home_canon,
            away_canonical=away_canon,
            home_odds_close=1.80,
            away_odds_close=2.10,
            p_home_devigged=0.538,
        )
    }

    xlsx_path = tmp_path / "fake.xlsx"
    xlsx_path.write_bytes(b"")  # any non-empty path; load patched below

    with patch("fantasy_coach.bookmaker.lines.load_closing_lines", return_value=fake_lines):
        code = main(
            [
                "merge-closing-lines",
                "--db",
                str(db_path),
                "--xlsx",
                str(xlsx_path),
                "--season",
                "2024",
                "--log-level",
                "WARNING",
            ]
        )
    assert code == 0

    repo = SQLiteRepository(db_path)
    try:
        hydrated = repo.list_matches(2024)[0]
    finally:
        repo.close()
    assert hydrated.home.odds == pytest.approx(1.80)
    assert hydrated.away.odds == pytest.approx(2.10)


def test_merge_closing_lines_errors_when_db_missing(tmp_path: Path) -> None:
    with patch("fantasy_coach.bookmaker.lines.load_closing_lines", return_value={}):
        code = main(
            [
                "merge-closing-lines",
                "--db",
                str(tmp_path / "missing.db"),
                "--xlsx",
                str(tmp_path / "fake.xlsx"),
                "--log-level",
                "WARNING",
            ]
        )
    assert code == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mini_match(
    *,
    match_id: int = 1,
    season: int = 2026,
    round_: int = 1,
    start_time: datetime | None = None,
    home_name: str = "Home",
    home_nick: str = "Home",
    away_name: str = "Away",
    away_nick: str = "Away",
    home_odds: float | None = None,
    away_odds: float | None = None,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round_,
        start_time=start_time or datetime(2026, 4, 10, tzinfo=UTC),
        match_state="Upcoming",
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(
            team_id=100,
            name=home_name,
            nick_name=home_nick,
            score=None,
            players=[],
            odds=home_odds,
        ),
        away=TeamRow(
            team_id=200,
            name=away_name,
            nick_name=away_nick,
            score=None,
            players=[],
            odds=away_odds,
        ),
        team_stats=[],
    )
