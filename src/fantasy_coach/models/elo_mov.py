"""Margin-of-victory weighted Elo (MOV Elo) — 538-style upgrade.

The standard K-factor is multiplied by a MOV term:

    K_eff = K × ln(|margin| + 1) × (2.2 / (elo_diff × 0.001 + 2.2))

- ``ln(|margin| + 1)`` rewards larger wins with diminishing returns.
- The autocorrelation correction (2.2 / …) discounts blowouts when the winner
  was already heavily favoured — a 40-point win over a clear underdog deserves
  less credit than a 40-point upset.

The existing ``Elo`` class is untouched so plain Elo remains a clean A/B
comparison.  ``EloMOV`` is a drop-in replacement: same constructor kwargs,
same ``predict`` / ``update`` / ``regress_to_mean`` interface.
"""

from __future__ import annotations

import math

from fantasy_coach.models.elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_INITIAL_RATING,
    DEFAULT_K,
    DEFAULT_SEASON_REGRESSION,
    Elo,
)


class EloMOV(Elo):
    """MOV-weighted Elo rating book.

    Identical interface to ``Elo``; only ``update`` differs — the K-factor
    is scaled by the 538-style MOV multiplier before each rating update.

    Draws use a plain K (margin = 0 → ln(1) = 0 collapses; we apply K × 1
    for draws to preserve the symmetric update behaviour of the base class).
    """

    def update(
        self, home_id: int, away_id: int, home_score: int, away_score: int
    ) -> tuple[float, float]:
        """Apply one match with MOV-weighted K. Returns (home, away) deltas."""
        expected_home = self.predict(home_id, away_id)
        margin = abs(home_score - away_score)

        if margin == 0:
            # Draw: fall back to plain K so zero-margin doesn't collapse update.
            k_eff = self.k
        else:
            # Elo difference from the winner's perspective (pre-update).
            if home_score > away_score:
                winner_eff = self.rating(home_id) + self.home_advantage
                loser_eff = self.rating(away_id)
            else:
                winner_eff = self.rating(away_id)
                loser_eff = self.rating(home_id) + self.home_advantage

            elo_diff = max(winner_eff - loser_eff, 0.0)
            autocorr = 2.2 / (elo_diff * 0.001 + 2.2)
            k_eff = self.k * math.log(margin + 1) * autocorr

        actual_home: float
        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        delta = k_eff * (actual_home - expected_home)
        self._ratings[home_id] = self.rating(home_id) + delta
        self._ratings[away_id] = self.rating(away_id) - delta
        return delta, -delta
