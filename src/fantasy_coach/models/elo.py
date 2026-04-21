"""Elo rating system — baseline pairwise predictor.

Standard Elo with two NRL-flavoured knobs:
- `home_advantage`: points added to the home team's effective rating before the
  expected-score calc. ~55 points ≈ the historical NRL home edge (~62% home
  win rate, equivalent to ~3–4 spread points).
- `season_regression`: at each season boundary the sweep can pull every team's
  rating fractionally back to the league mean (1500). 0.0 = no regression,
  1.0 = full reset. ~0.25 is a common choice for sports with roster turnover.

Draws are scored as 0.5/0.5. Margin of victory is *not* used here — that's a
follow-up if Elo proves competitive enough to be worth tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_HOME_ADVANTAGE = 55.0
DEFAULT_SEASON_REGRESSION = 0.25


@dataclass
class Elo:
    """In-memory Elo rating book.

    Teams are inserted lazily at first appearance, all starting at
    `initial_rating`. Mutating; not thread-safe.
    """

    k: float = DEFAULT_K
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    initial_rating: float = DEFAULT_INITIAL_RATING
    season_regression: float = DEFAULT_SEASON_REGRESSION

    def __post_init__(self) -> None:
        self._ratings: dict[int, float] = {}

    # ----- read-only -----

    def rating(self, team_id: int) -> float:
        return self._ratings.get(team_id, self.initial_rating)

    def ratings(self) -> dict[int, float]:
        """A defensive copy of the rating book."""
        return dict(self._ratings)

    def predict(self, home_id: int, away_id: int) -> float:
        """Return the home team's win probability (treats draws as half-wins)."""
        home_eff = self.rating(home_id) + self.home_advantage
        away_eff = self.rating(away_id)
        return _expected_score(home_eff, away_eff)

    # ----- mutations -----

    def update(
        self, home_id: int, away_id: int, home_score: int, away_score: int
    ) -> tuple[float, float]:
        """Apply one match. Returns the (home, away) rating deltas."""
        expected_home = self.predict(home_id, away_id)
        actual_home = _result(home_score, away_score)
        delta = self.k * (actual_home - expected_home)

        new_home = self.rating(home_id) + delta
        new_away = self.rating(away_id) - delta
        self._ratings[home_id] = new_home
        self._ratings[away_id] = new_away
        return delta, -delta

    def regress_to_mean(self, weight: float | None = None) -> None:
        """Pull every rating fractionally toward `initial_rating`.

        Call between seasons. `weight=0` no-ops, `weight=1` resets the book.
        Defaults to `self.season_regression`.
        """
        w = self.season_regression if weight is None else weight
        if w <= 0:
            return
        if w > 1:
            raise ValueError(f"regression weight must be in [0, 1], got {w}")
        for team_id, rating in self._ratings.items():
            self._ratings[team_id] = rating + w * (self.initial_rating - rating)


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _result(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5
