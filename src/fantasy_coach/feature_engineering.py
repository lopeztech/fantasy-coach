"""Per-match feature engineering for the logistic-regression baseline.

Walks completed matches in chronological order and emits one feature row per
match using *only* information observable before kickoff. This avoids the
classic time-series leakage where a model "predicts" the past by accidentally
using stats that include the match's own outcome.

`FeatureBuilder` is the underlying state machine — call `feature_row(match)`
to get the pre-kickoff feature vector, then `record(match)` to fold the
outcome into the rolling state. `build_training_frame` is the
fit-time wrapper that drives the builder over a list of completed matches.

Features (all home-minus-away unless noted):
- `elo_diff`: home Elo + home_advantage − away Elo, computed against the rolling
  Elo book that is updated *after* each match is emitted.
- `form_diff_pf`, `form_diff_pa`: rolling-5 points-for and points-against
  averages, home minus away. Empty deques (no prior matches) read as 0
  — equivalent to "no prior info, no edge".
- `days_rest_diff`: home days since last match minus away days since last
  match. First-match-of-season teams are clamped to 14 days (≈ pre-season).
- `h2h_recent_diff`: average home-perspective score margin in the last 3
  meetings (so a team that's beaten its opponent badly recently scores high).
- `is_home_field`: always 1 (home perspective). Constant; included so the
  intercept absorbs it cleanly when fitting.
- `rolling_kick_metres_diff`: rolling-5 kicking metres, home minus away.
  Proxy for halfback kicking game / field-position dominance.
- `rolling_kick_return_metres_diff`: rolling-5 kick return metres, home minus
  away. Proxy for fullback counter-attack effectiveness.
- `rolling_line_breaks_diff`: rolling-5 line breaks, home minus away.
  Measures attack incisiveness.
- `rolling_all_runs_diff`: rolling-5 all runs (hit-ups), home minus away.
  Proxy for forward pack workload.
- `missing_team_stats`: 1.0 when the home team has no entries in any rolling
  team-stat deque (historical gap or upcoming match). 0.0 otherwise.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from fantasy_coach.features import MatchRow, TeamStat
from fantasy_coach.models.elo import Elo

FEATURE_NAMES = (
    "elo_diff",
    "form_diff_pf",
    "form_diff_pa",
    "days_rest_diff",
    "h2h_recent_diff",
    "is_home_field",
    # Team-stat rolling features added in #160 (player-level proxies)
    "rolling_kick_metres_diff",
    "rolling_kick_return_metres_diff",
    "rolling_line_breaks_diff",
    "rolling_all_runs_diff",
    "missing_team_stats",
)

ROLLING_WINDOW = 5
H2H_WINDOW = 3
DEFAULT_DAYS_REST = 14
_TEAM_STATS_WINDOW = 5


@dataclass(frozen=True)
class TrainingFrame:
    """Numpy arrays ready for sklearn."""

    X: np.ndarray
    y: np.ndarray
    match_ids: np.ndarray
    start_times: np.ndarray  # dtype=datetime64[s]
    feature_names: tuple[str, ...] = FEATURE_NAMES


class FeatureBuilder:
    """Stateful per-match feature computation.

    Used at training time by `build_training_frame` and at inference time by
    `LogisticPredictor` so a single match can be scored in O(1) instead of
    rebuilding the whole frame.
    """

    def __init__(self, elo: Elo | None = None) -> None:
        self.elo = elo or Elo()
        self._last_played: dict[int, datetime] = {}
        self._points_for: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self._points_against: dict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=ROLLING_WINDOW)
        )
        self._h2h: dict[tuple[int, int], deque[int]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
        self._current_season: int | None = None
        # Team-stat rolling deques (updated in record() after each completed match)
        self._team_kick_metres: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=_TEAM_STATS_WINDOW)
        )
        self._team_kick_return_metres: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=_TEAM_STATS_WINDOW)
        )
        self._team_line_breaks: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=_TEAM_STATS_WINDOW)
        )
        self._team_all_runs: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=_TEAM_STATS_WINDOW)
        )

    def feature_row(self, match: MatchRow) -> list[float]:
        h_id, a_id = match.home.team_id, match.away.team_id
        elo_diff = self.elo.rating(h_id) + self.elo.home_advantage - self.elo.rating(a_id)
        form_pf_h = _avg(self._points_for[h_id])
        form_pf_a = _avg(self._points_for[a_id])
        form_pa_h = _avg(self._points_against[h_id])
        form_pa_a = _avg(self._points_against[a_id])
        rest_h = _days_since(self._last_played.get(h_id), match.start_time)
        rest_a = _days_since(self._last_played.get(a_id), match.start_time)
        h2h_avg = _avg(self._h2h[_h2h_key(h_id, a_id)])
        h2h_recent = h2h_avg if h_id <= a_id else -h2h_avg

        # Team-stat rolling features (from record() state, no leakage)
        kick_m_h = _avg(self._team_kick_metres[h_id])
        kick_m_a = _avg(self._team_kick_metres[a_id])
        kick_ret_h = _avg(self._team_kick_return_metres[h_id])
        kick_ret_a = _avg(self._team_kick_return_metres[a_id])
        lb_h = _avg(self._team_line_breaks[h_id])
        lb_a = _avg(self._team_line_breaks[a_id])
        runs_h = _avg(self._team_all_runs[h_id])
        runs_a = _avg(self._team_all_runs[a_id])

        # missing_team_stats = 1.0 when the home team has no entries in any
        # rolling team-stat deque (data gap for historical/upcoming matches).
        missing = 1.0 if len(self._team_kick_metres[h_id]) == 0 else 0.0

        return [
            elo_diff,
            form_pf_h - form_pf_a,
            form_pa_h - form_pa_a,
            rest_h - rest_a,
            h2h_recent,
            1.0,
            kick_m_h - kick_m_a,
            kick_ret_h - kick_ret_a,
            lb_h - lb_a,
            runs_h - runs_a,
            missing,
        ]

    def advance_season_if_needed(self, match: MatchRow) -> None:
        """Apply Elo regression-to-mean when the season changes."""
        if self._current_season is None:
            self._current_season = match.season
            return
        if match.season != self._current_season:
            self.elo.regress_to_mean()
            self._current_season = match.season

    def record(self, match: MatchRow) -> None:
        """Fold a completed match's outcome into the rolling state."""
        if match.home.score is None or match.away.score is None:
            return
        h_id, a_id = match.home.team_id, match.away.team_id
        h_score, a_score = int(match.home.score), int(match.away.score)
        self._points_for[h_id].append(h_score)
        self._points_for[a_id].append(a_score)
        self._points_against[h_id].append(a_score)
        self._points_against[a_id].append(h_score)
        self._h2h[_h2h_key(h_id, a_id)].append(_signed_h2h(h_id, a_id, h_score, a_score))
        self._last_played[h_id] = match.start_time
        self._last_played[a_id] = match.start_time
        self.elo.update(h_id, a_id, h_score, a_score)

        # Update team-stat rolling deques (only when team_stats are present)
        kick_m_h = _extract_stat(match.team_stats, "Kicking Metres", "home")
        kick_m_a = _extract_stat(match.team_stats, "Kicking Metres", "away")
        kick_ret_h = _extract_stat(match.team_stats, "Kick Return Metres", "home")
        kick_ret_a = _extract_stat(match.team_stats, "Kick Return Metres", "away")
        lb_h = _extract_stat(match.team_stats, "Line Breaks", "home")
        lb_a = _extract_stat(match.team_stats, "Line Breaks", "away")
        runs_h = _extract_stat(match.team_stats, "All Runs", "home")
        runs_a = _extract_stat(match.team_stats, "All Runs", "away")

        if kick_m_h is not None:
            self._team_kick_metres[h_id].append(kick_m_h)
        if kick_m_a is not None:
            self._team_kick_metres[a_id].append(kick_m_a)
        if kick_ret_h is not None:
            self._team_kick_return_metres[h_id].append(kick_ret_h)
        if kick_ret_a is not None:
            self._team_kick_return_metres[a_id].append(kick_ret_a)
        if lb_h is not None:
            self._team_line_breaks[h_id].append(lb_h)
        if lb_a is not None:
            self._team_line_breaks[a_id].append(lb_a)
        if runs_h is not None:
            self._team_all_runs[h_id].append(runs_h)
        if runs_a is not None:
            self._team_all_runs[a_id].append(runs_a)


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

    completed = sorted(
        (m for m in matches if _is_complete(m)),
        key=lambda m: (m.start_time, m.match_id),
    )

    builder = FeatureBuilder(elo=elo)
    rows: list[list[float]] = []
    targets: list[int] = []
    match_ids: list[int] = []
    start_times: list[np.datetime64] = []

    for match in completed:
        builder.advance_season_if_needed(match)
        h_score = int(match.home.score or 0)
        a_score = int(match.away.score or 0)

        if drop_draws and h_score == a_score:
            builder.record(match)
            continue

        rows.append(builder.feature_row(match))
        targets.append(1 if h_score > a_score else 0)
        match_ids.append(match.match_id)
        start_times.append(np.datetime64(match.start_time.replace(tzinfo=None), "s"))

        builder.record(match)

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


def _is_complete(match: MatchRow) -> bool:
    return (
        match.match_state in {"FullTime", "FullTimeED"}
        and match.home.score is not None
        and match.away.score is not None
    )


def _extract_stat(team_stats: list[TeamStat], title: str, side: str) -> float | None:
    """Return the home or away value for a named team stat, or None if absent."""
    for stat in team_stats:
        if stat.title == title:
            return stat.home_value if side == "home" else stat.away_value
    return None


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
    """Score margin from the perspective of the lower team_id."""
    if home_id <= away_id:
        return home_score - away_score
    return away_score - home_score
