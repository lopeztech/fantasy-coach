"""Predictor that returns the de-vigged closing-line home win probability.

Looks up each match by (Sydney-local kickoff date, canonical home, canonical
away). Matches that aren't in the closing-lines dataset fall back to 0.55
— the same prior as `HomePickPredictor` — so the metric reflects only
matches the bookmaker actually priced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta, timezone
from typing import ClassVar

from fantasy_coach.bookmaker.lines import ClosingLine
from fantasy_coach.bookmaker.team_names import canonicalize
from fantasy_coach.features import MatchRow

# NRL kickoffs are scheduled in Sydney local time; UTC offsets vary across
# the season due to DST. ±1-day window when looking up by date absorbs that
# without needing a full timezone library.
_DATE_TOLERANCE_DAYS = 1
_AEST_FALLBACK = timezone(timedelta(hours=10))


class BookmakerPredictor:
    name: ClassVar[str] = "bookmaker"

    def __init__(self, closing_lines: Mapping[tuple[date, str, str], ClosingLine]) -> None:
        self._lines = dict(closing_lines)
        self._missing: list[int] = []

    def fit(self, history: Sequence[MatchRow]) -> None:  # noqa: ARG002
        return

    def predict_home_win_prob(self, match: MatchRow) -> float:
        line = self._lookup(match)
        if line is None:
            self._missing.append(match.match_id)
            return 0.55
        return line.p_home_devigged

    @property
    def missing_match_ids(self) -> list[int]:
        return list(self._missing)

    def _lookup(self, match: MatchRow) -> ClosingLine | None:
        home = canonicalize(match.home.nick_name) or canonicalize(match.home.name)
        away = canonicalize(match.away.nick_name) or canonicalize(match.away.name)
        if home is None or away is None:
            return None

        center = match.start_time.astimezone(_AEST_FALLBACK).date()
        for delta in range(-_DATE_TOLERANCE_DAYS, _DATE_TOLERANCE_DAYS + 1):
            line = self._lines.get((center + timedelta(days=delta), home, away))
            if line is not None:
                return line
        return None
