from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path

import openpyxl
import pytest

from fantasy_coach.bookmaker import (
    BookmakerPredictor,
    ClosingLine,
    devig_two_way,
    load_closing_lines,
)
from fantasy_coach.bookmaker.team_names import canonicalize
from fantasy_coach.features import MatchRow, TeamRow

# ---------- de-vig math ----------


def test_devig_normalizes_overround() -> None:
    # Bookmaker odds with ~5 % vig: 1/1.85 + 1/2.05 ≈ 1.028
    p_home = devig_two_way(1.85, 2.05)
    p_away = 1.0 - p_home
    # Sum should be exactly 1.
    assert math.isclose(p_home + p_away, 1.0, rel_tol=1e-9)
    # Home should be the favourite (lower odds).
    assert p_home > 0.5


def test_devig_rejects_invalid_odds() -> None:
    with pytest.raises(ValueError):
        devig_two_way(1.0, 2.0)


# ---------- xlsx loader ----------


def _write_test_xlsx(path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    # Banner row (the loader ignores it).
    ws.append(["banner"])
    ws.append(
        [
            "Date",
            "Kick-off (local)",
            "Home Team",
            "Away Team",
            "Venue",
            "Home Score",
            "Away Score",
            "Play Off Game?",
            "Over Time?",
            "Home Odds",
            "Draw Odds",
            "Away Odds",
            "Bookmakers Surveyed",
            "Home Odds Open",
            "Home Odds Min",
            "Home Odds Max",
            "Home Odds Close",
            "Away Odds Open",
            "Away Odds Min",
            "Away Odds Max",
            "Away Odds Close",
        ]
    )
    ws.append(
        [
            datetime(2024, 3, 3),
            None,
            "Manly-Warringah Sea Eagles",
            "South Sydney Rabbitohs",
            "venue",
            36,
            24,
            None,
            None,
            1.85,
            None,
            2.05,
            None,
            None,
            None,
            None,
            1.83,
            None,
            None,
            None,
            2.10,
        ]
    )
    ws.append(
        [
            datetime(2024, 3, 4),
            None,
            "Mystery Team",  # un-mappable → skipped
            "Parramatta Eels",
            "v",
            12,
            18,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            2.0,
            None,
            None,
            None,
            1.85,
        ]
    )
    wb.save(path)


def test_load_closing_lines_parses_and_dedupes(tmp_path: Path) -> None:
    path = tmp_path / "lines.xlsx"
    _write_test_xlsx(path)

    lines = load_closing_lines(path)

    assert len(lines) == 1  # second row dropped (un-mappable home team)
    [(key, line)] = lines.items()
    assert key == (date(2024, 3, 3), "sea-eagles", "rabbitohs")
    assert line.home_odds_close == 1.83
    assert line.away_odds_close == 2.10
    expected = devig_two_way(1.83, 2.10)
    assert math.isclose(line.p_home_devigged, expected, rel_tol=1e-9)


# ---------- team-name canonicalizer ----------


def test_canonicalize_handles_full_and_short_names() -> None:
    assert canonicalize("Manly-Warringah Sea Eagles") == "sea-eagles"
    assert canonicalize("Sea Eagles") == "sea-eagles"
    assert canonicalize("Wests Tigers") == "wests-tigers"
    assert canonicalize("Mystery Team") is None


# ---------- predictor ----------


def _match(*, match_id: int, home_nick: str, away_nick: str, when: datetime) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=when.year,
        round=1,
        start_time=when,
        match_state="FullTime",
        venue=None,
        venue_city=None,
        weather=None,
        home=TeamRow(team_id=1, name=home_nick, nick_name=home_nick, score=0, players=[]),
        away=TeamRow(team_id=2, name=away_nick, nick_name=away_nick, score=0, players=[]),
        team_stats=[],
    )


def test_predictor_returns_devigged_prob_for_known_match() -> None:
    line = ClosingLine(
        match_date=date(2024, 3, 3),
        home_canonical="sea-eagles",
        away_canonical="rabbitohs",
        home_odds_close=1.83,
        away_odds_close=2.10,
        p_home_devigged=devig_two_way(1.83, 2.10),
    )
    predictor = BookmakerPredictor({line.key: line})

    p = predictor.predict_home_win_prob(
        _match(
            match_id=1,
            home_nick="Sea Eagles",
            away_nick="Rabbitohs",
            when=datetime(2024, 3, 3, 6, 30, tzinfo=UTC),
        )
    )

    assert math.isclose(p, line.p_home_devigged, rel_tol=1e-9)
    assert predictor.missing_match_ids == []


def test_predictor_falls_back_when_match_missing() -> None:
    predictor = BookmakerPredictor({})

    p = predictor.predict_home_win_prob(
        _match(
            match_id=42,
            home_nick="Sea Eagles",
            away_nick="Rabbitohs",
            when=datetime(2024, 3, 3, tzinfo=UTC),
        )
    )

    assert p == 0.55
    assert predictor.missing_match_ids == [42]


def test_predictor_tolerates_one_day_drift() -> None:
    line = ClosingLine(
        match_date=date(2024, 3, 3),
        home_canonical="sea-eagles",
        away_canonical="rabbitohs",
        home_odds_close=1.83,
        away_odds_close=2.10,
        p_home_devigged=devig_two_way(1.83, 2.10),
    )
    predictor = BookmakerPredictor({line.key: line})

    # NRL kickoff at 14:00 UTC is March 4 in Sydney time (+10) — predictor
    # should still find the line dated March 3.
    p = predictor.predict_home_win_prob(
        _match(
            match_id=1,
            home_nick="Sea Eagles",
            away_nick="Rabbitohs",
            when=datetime(2024, 3, 2, 18, 0, tzinfo=UTC),
        )
    )
    assert math.isclose(p, line.p_home_devigged, rel_tol=1e-9)
