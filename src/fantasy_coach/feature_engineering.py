"""Per-match feature engineering for the logistic-regression baseline.

Walks completed matches in chronological order and emits one feature row per
match using *only* information observable before kickoff. This avoids the
classic time-series leakage where a model "predicts" the past by accidentally
using stats that include the match's own outcome.

Features (all home-minus-away unless noted):
- `elo_diff`: home Elo + home_advantage − away Elo, computed against the rolling
  Elo book that is updated *after* each match is emitted.
- `form_diff_pf`, `form_diff_pa`: rolling-5 points-for and points-against
  averages, home minus away. NaN-padded teams (no prior matches) are filled
  with 0 — equivalent to "no prior info, no edge".
- `days_rest_diff`: home days since last match minus away days since last
  match. First-match-of-season teams are clamped to 14 days (≈ pre-season).
- `h2h_recent_diff`: average home-perspective score margin in the last 3
  meetings (so a team that's beaten its opponent badly recently scores high).
- `is_home_field`: always 1 (home perspective). Constant; included so the
  intercept absorbs it cleanly when fitting.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from fantasy_coach.features import MatchRow
from fantasy_coach.models.elo import Elo

FEATURE_NAMES = (
    "elo_diff",
    "form_diff_pf",
    "form_diff_pa",
    "days_rest_diff",
    "h2h_recent_diff",
    "is_home_field",
)

ROLLING_WINDOW = 5
H2H_WINDOW = 3
DEFAULT_DAYS_REST = 14


@dataclass(frozen=True)
class TrainingFrame:
    """Numpy arrays ready for sklearn."""

    X: np.ndarray
    y: np.ndarray
    match_ids: np.ndarray
    start_times: np.ndarray  # dtype=datetime64[s]
    feature_names: tuple[str, ...] = FEATURE_NAMES


def build_training_frame(
    matches: Iterable[MatchRow],
    *,
    elo: Elo | None = None,
    drop_draws: bool = True,
) -> TrainingFrame:
    """Compute features for every completed match in chronological order.

    Draws are dropped by default — logistic regression wants binary targets
    and NRL has very few draws, so excluding them is cleaner than relabelling.
    """

    elo = elo or Elo()
    completed = sorted(
        (m for m in matches if _is_complete(m)),
        key=lambda m: (m.start_time, m.match_id),
    )

    last_played: dict[int, datetime] = {}
    points_for: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    points_against: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    h2h: dict[tuple[int, int], deque[int]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    rows: list[list[float]] = []
    targets: list[int] = []
    match_ids: list[int] = []
    start_times: list[np.datetime64] = []

    for match in completed:
        h_id, a_id = match.home.team_id, match.away.team_id
        h_score = int(match.home.score or 0)
        a_score = int(match.away.score or 0)

        if drop_draws and h_score == a_score:
            _record_post_match(match, last_played, points_for, points_against, h2h, elo)
            continue

        elo_diff = elo.rating(h_id) + elo.home_advantage - elo.rating(a_id)
        form_pf_h = _avg(points_for[h_id])
        form_pf_a = _avg(points_for[a_id])
        form_pa_h = _avg(points_against[h_id])
        form_pa_a = _avg(points_against[a_id])
        rest_h = _days_since(last_played.get(h_id), match.start_time)
        rest_a = _days_since(last_played.get(a_id), match.start_time)
        # h2h is stored from the lower-team-id's perspective; flip if needed
        # so the feature is always read from the home perspective.
        h2h_avg = _avg(h2h[_h2h_key(h_id, a_id)])
        h2h_recent = h2h_avg if h_id <= a_id else -h2h_avg

        rows.append(
            [
                elo_diff,
                form_pf_h - form_pf_a,
                form_pa_h - form_pa_a,
                rest_h - rest_a,
                h2h_recent,
                1.0,
            ]
        )
        targets.append(1 if h_score > a_score else 0)
        match_ids.append(match.match_id)
        start_times.append(np.datetime64(match.start_time.replace(tzinfo=None), "s"))

        _record_post_match(match, last_played, points_for, points_against, h2h, elo)

    if not rows:
        return TrainingFrame(
            X=np.zeros((0, len(FEATURE_NAMES))),
            y=np.zeros((0,), dtype=int),
            match_ids=np.zeros((0,), dtype=int),
            start_times=np.zeros((0,), dtype="datetime64[s]"),
        )

    return TrainingFrame(
        X=np.asarray(rows, dtype=float),
        y=np.asarray(targets, dtype=int),
        match_ids=np.asarray(match_ids, dtype=int),
        start_times=np.asarray(start_times, dtype="datetime64[s]"),
    )


def _record_post_match(
    match: MatchRow,
    last_played: dict[int, datetime],
    points_for: dict[int, deque[int]],
    points_against: dict[int, deque[int]],
    h2h: dict[tuple[int, int], deque[int]],
    elo: Elo,
) -> None:
    h_id, a_id = match.home.team_id, match.away.team_id
    h_score = int(match.home.score or 0)
    a_score = int(match.away.score or 0)
    points_for[h_id].append(h_score)
    points_for[a_id].append(a_score)
    points_against[h_id].append(a_score)
    points_against[a_id].append(h_score)
    h2h[_h2h_key(h_id, a_id)].append(_signed_h2h(h_id, a_id, h_score, a_score))
    last_played[h_id] = match.start_time
    last_played[a_id] = match.start_time
    elo.update(h_id, a_id, h_score, a_score)


def _is_complete(match: MatchRow) -> bool:
    return (
        match.match_state in {"FullTime", "FullTimeED"}
        and match.home.score is not None
        and match.away.score is not None
    )


def _avg(values: Iterable[int | float]) -> float:
    arr = list(values)
    if not arr:
        return 0.0
    return float(sum(arr)) / len(arr)


def _days_since(prev: datetime | None, now: datetime) -> float:
    if prev is None:
        return float(DEFAULT_DAYS_REST)
    return (now - prev).total_seconds() / 86400.0


def _h2h_key(home_id: int, away_id: int) -> tuple[int, int]:
    """Order-independent matchup key."""
    return (min(home_id, away_id), max(home_id, away_id))


def _signed_h2h(home_id: int, away_id: int, home_score: int, away_score: int) -> int:
    """Score margin from the perspective of the lower team_id, so the same
    matchup key always reads in the same direction."""
    if home_id <= away_id:
        return home_score - away_score
    return away_score - home_score
