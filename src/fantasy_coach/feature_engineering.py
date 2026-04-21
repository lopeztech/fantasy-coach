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
- `elo_diff`: home Elo + home_advantage − away Elo.
- `form_diff_pf`, `form_diff_pa`: rolling-5 points-for and points-against averages.
- `days_rest_diff`: home days since last match minus away days since last match.
- `h2h_recent_diff`: average home-perspective score margin in the last 3 meetings.
- `is_home_field`: always 1 (home perspective).
- `travel_km_diff`: great-circle km travelled from prev venue to this venue, home − away.
- `timezone_delta_diff`: absolute timezone-shift hours, home − away.
- `back_to_back_short_week_diff`: +1/−1/0 flag for (days_rest < 6 AND travel > 1 000 km).
- `is_wet`, `wind_kph`, `temperature_c`, `missing_weather`: parsed from the weather block.
- `venue_avg_total_points`: rolling-10 average total points at this venue (history-only).
- `venue_home_win_rate`: rolling-20 home win rate at this venue (history-only).
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from fantasy_coach.features import MatchRow
from fantasy_coach.models.elo import Elo
from fantasy_coach.travel import travel_features
from fantasy_coach.weather import parse_weather

FEATURE_NAMES = (
    "elo_diff",
    "form_diff_pf",
    "form_diff_pa",
    "days_rest_diff",
    "h2h_recent_diff",
    "is_home_field",
    "travel_km_diff",
    "timezone_delta_diff",
    "back_to_back_short_week_diff",
    "is_wet",
    "wind_kph",
    "temperature_c",
    "missing_weather",
    "venue_avg_total_points",
    "venue_home_win_rate",
    "ref_avg_total_points",
    "ref_home_penalty_diff",
    "missing_referee",
)

VENUE_TOTAL_WINDOW = 10
VENUE_WIN_WINDOW = 20

ROLLING_WINDOW = 5
H2H_WINDOW = 3
DEFAULT_DAYS_REST = 14

REF_WINDOW = 20  # rolling window for referee stats
REF_SHRINKAGE_N = 10  # shrink toward league mean for fewer than N prior matches


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
        self._last_venue: dict[int, str | None] = {}
        self._points_for: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self._points_against: dict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=ROLLING_WINDOW)
        )
        self._h2h: dict[tuple[int, int], deque[int]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
        self._current_season: int | None = None
        # Venue-level rolling stats (keyed by lower-cased venue name).
        self._venue_total: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=VENUE_TOTAL_WINDOW)
        )
        self._venue_home_wins: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=VENUE_WIN_WINDOW)
        )
        # Referee-level rolling stats (keyed by referee profileId).
        self._ref_total: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=REF_WINDOW))
        self._ref_penalty_diff: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=REF_WINDOW)
        )
        # League-wide rolling totals for shrinkage prior.
        self._league_total: deque[int] = deque(maxlen=REF_WINDOW * 5)

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
        tkm, ttz, tbb = travel_features(
            self._last_venue.get(h_id),
            self._last_venue.get(a_id),
            match.venue,
            rest_h,
            rest_a,
        )
        wx = parse_weather(match.weather)
        vkey = (match.venue or "").lower()
        venue_avg_tp = _avg(self._venue_total[vkey]) if vkey else 0.0
        # Neutral prior (0.5) when no history available for this venue.
        venue_hwr = (
            _avg(self._venue_home_wins[vkey]) if (vkey and self._venue_home_wins[vkey]) else 0.5
        )
        ref_tp, ref_pd, missing_ref = self._referee_features(match.referee_id)
        return [
            elo_diff,
            form_pf_h - form_pf_a,
            form_pa_h - form_pa_a,
            rest_h - rest_a,
            h2h_recent,
            1.0,
            tkm,
            ttz,
            tbb,
            wx.is_wet,
            wx.wind_kph,
            wx.temperature_c,
            wx.missing,
            venue_avg_tp,
            venue_hwr,
            ref_tp,
            ref_pd,
            missing_ref,
        ]

    def _referee_features(self, referee_id: int | None) -> tuple[float, float, float]:
        """Return (ref_avg_total_points, ref_home_penalty_diff, missing_referee).

        Both numeric features are shrunk toward the league mean when the referee
        has fewer than REF_SHRINKAGE_N prior observed matches.
        """
        if referee_id is None:
            league_mean = _avg(self._league_total) if self._league_total else 0.0
            return league_mean, 0.0, 1.0

        history = self._ref_total[referee_id]
        n = len(history)
        league_mean = _avg(self._league_total) if self._league_total else 0.0

        if n == 0:
            ref_avg_tp = league_mean
        elif n < REF_SHRINKAGE_N:
            # Shrink toward league mean: weight = n / REF_SHRINKAGE_N
            w = n / REF_SHRINKAGE_N
            ref_avg_tp = w * _avg(history) + (1 - w) * league_mean
        else:
            ref_avg_tp = _avg(history)

        penalty_hist = self._ref_penalty_diff[referee_id]
        ref_pd = _avg(penalty_hist) if penalty_hist else 0.0

        return ref_avg_tp, ref_pd, 0.0

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
        self._last_venue[h_id] = match.venue
        self._last_venue[a_id] = match.venue
        self.elo.update(h_id, a_id, h_score, a_score)
        # Update venue rolling stats after the match result is known.
        vkey = (match.venue or "").lower()
        if vkey:
            self._venue_total[vkey].append(h_score + a_score)
            self._venue_home_wins[vkey].append(1 if h_score > a_score else 0)
        # Update referee rolling stats.
        total_points = h_score + a_score
        self._league_total.append(total_points)
        if match.referee_id is not None:
            self._ref_total[match.referee_id].append(total_points)
            penalty_diff = _penalty_diff(match)
            if penalty_diff is not None:
                self._ref_penalty_diff[match.referee_id].append(penalty_diff)


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


def _penalty_diff(match: MatchRow) -> float | None:
    """Return home_penalties_conceded - away_penalties_conceded, or None if unavailable."""
    for stat in match.team_stats:
        if (
            stat.title == "Penalties Conceded"
            and stat.home_value is not None
            and stat.away_value is not None
        ):
            return float(stat.home_value) - float(stat.away_value)
    return None
