"""Glicko-2 rating system — higher-fidelity successor to Elo/EloMOV.

Glicko-2 tracks three state variables per team:
- ``mu``: rating on the Glicko-2 scale (default 0 = 1500 in Glicko-1 terms)
- ``phi``: rating deviation (RD) — how uncertain we are about the rating
- ``sigma``: volatility — how much a team's performance fluctuates

The update equations follow Glickman (2012, "Example of the Glicko-2 system"):
https://www.glicko.net/glicko/glicko2.pdf

Scale conversions:
  Glicko-1:  r = 173.7178 * mu + 1500
  Glicko-2:  mu = (r - 1500) / 173.7178

Margin of victory is incorporated via the same approach as ``EloMOV``:
the win-probability score function is augmented by a MOV multiplier so a
large-margin win counts as stronger evidence than a 1-point win.

Interface mirrors ``EloMOV`` (and therefore ``Elo``) exactly:
- ``rating(team_id) -> float``   (returns Glicko-1 scale r)
- ``predict(home_id, away_id) -> float``
- ``update(home_id, away_id, home_score, away_score) -> (home_delta, away_delta)``
- ``regress_to_mean()``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Glicko-2 scale factor: converts Glicko-1 rating to Glicko-2 mu
_SCALE = 173.7178

# Default Glicko-1 parameters, expressed in Glicko-2 internal scale
_DEFAULT_MU = 0.0  # = (1500 - 1500) / 173.7178
_DEFAULT_PHI = 2.0148  # = 350 / 173.7178  (high uncertainty for new teams)
_DEFAULT_SIGMA = 0.06  # volatility — Glickman suggests 0.3–1.2 for most sports

# System constant τ constrains how much σ can change per period.
# Smaller τ → more stable σ. Glickman recommends 0.3–1.2.
_DEFAULT_TAU = 0.5

# Home advantage in Glicko-1 points (same default as Elo / EloMOV)
_DEFAULT_HOME_ADVANTAGE = 55.0
_DEFAULT_INITIAL_RATING = 1500.0
_DEFAULT_SEASON_REGRESSION = 0.25

# Epsilon for Newton–Raphson convergence
_EPSILON = 1e-6


@dataclass
class Glicko2:
    """In-memory Glicko-2 rating book.

    Teams are inserted lazily at first appearance with default (mu, phi, sigma).
    ``update`` applies a single-game update — Glicko-2 is designed for rating
    periods (multiple games), but a single-game update is a valid degenerate
    case that converges to correct ratings over many games.

    ``rating(team_id)`` returns the Glicko-1 scale r value (centred at 1500)
    for compatibility with ``FeatureBuilder.elo_diff`` and display purposes.
    """

    home_advantage: float = _DEFAULT_HOME_ADVANTAGE
    initial_rating: float = _DEFAULT_INITIAL_RATING
    season_regression: float = _DEFAULT_SEASON_REGRESSION
    tau: float = _DEFAULT_TAU

    def __post_init__(self) -> None:
        # Internal state: (mu, phi, sigma) per team
        self._mu: dict[int, float] = {}
        self._phi: dict[int, float] = {}
        self._sigma: dict[int, float] = {}

    # ----- read-only -----

    def _get_state(self, team_id: int) -> tuple[float, float, float]:
        """Return (mu, phi, sigma) for a team, inserting defaults if new."""
        mu = self._mu.get(team_id, _DEFAULT_MU)
        phi = self._phi.get(team_id, _DEFAULT_PHI)
        sigma = self._sigma.get(team_id, _DEFAULT_SIGMA)
        return mu, phi, sigma

    def rating(self, team_id: int) -> float:
        """Return the Glicko-1 scale rating (centred at 1500)."""
        mu = self._mu.get(team_id, _DEFAULT_MU)
        return _SCALE * mu + self.initial_rating

    def predict(self, home_id: int, away_id: int) -> float:
        """Return home team's win probability, accounting for home advantage."""
        home_mu = self._mu.get(home_id, _DEFAULT_MU)
        away_mu = self._mu.get(away_id, _DEFAULT_MU)
        home_phi = self._phi.get(home_id, _DEFAULT_PHI)
        away_phi = self._phi.get(away_id, _DEFAULT_PHI)

        # Convert home advantage from Glicko-1 to Glicko-2 scale
        ha_mu = self.home_advantage / _SCALE

        # Glicko-2 win probability: E(mu_home, mu_away, phi_combined)
        phi_combined = math.sqrt(home_phi**2 + away_phi**2)
        return _E(home_mu + ha_mu, away_mu, phi_combined)

    # ----- mutations -----

    def update(
        self, home_id: int, away_id: int, home_score: int, away_score: int
    ) -> tuple[float, float]:
        """Apply one match with MOV-weighted scores. Returns (home, away) rating deltas.

        Uses the Glicko-2 single-period update applied independently to each team.
        The MOV multiplier scales the effective information weight (inverse variance)
        of the match result — a larger-margin win is treated as if the team played
        ``mov_factor`` games, each with the same binary outcome. This mirrors the
        EloMOV approach of scaling K, but applied in the Glicko-2 information framework.
        """
        # Pre-update ratings for delta computation
        home_r_before = self.rating(home_id)
        away_r_before = self.rating(away_id)

        ha_mu = self.home_advantage / _SCALE
        home_mu, home_phi, home_sigma = self._get_state(home_id)
        away_mu, away_phi, away_sigma = self._get_state(away_id)

        # MOV multiplier (same formula as EloMOV)
        margin = abs(home_score - away_score)
        if margin == 0:
            mov_factor = 1.0
        else:
            elo_diff = max(
                abs(
                    (_SCALE * (home_mu + ha_mu) + self.initial_rating)
                    - (_SCALE * away_mu + self.initial_rating)
                ),
                0.0,
            )
            autocorr = 2.2 / (elo_diff * 0.001 + 2.2)
            mov_factor = math.log(margin + 1) * autocorr

        # Binary actual outcome
        if home_score > away_score:
            s_home = 1.0
        elif home_score < away_score:
            s_home = 0.0
        else:
            s_home = 0.5

        s_away = 1.0 - s_home

        # Update each team using Glicko-2 single-opponent algorithm with MOV scaling.
        # Home team: "played" away team
        new_home_mu, new_home_phi, new_home_sigma = _glicko2_update(
            mu=home_mu,
            phi=home_phi,
            sigma=home_sigma,
            opp_mu=away_mu - ha_mu,  # remove HA from opponent perspective
            opp_phi=away_phi,
            score=s_home,
            tau=self.tau,
            mov_factor=mov_factor,
        )
        # Away team: "played" home team (from away's perspective, home had no HA advantage)
        new_away_mu, new_away_phi, new_away_sigma = _glicko2_update(
            mu=away_mu,
            phi=away_phi,
            sigma=away_sigma,
            opp_mu=home_mu + ha_mu,
            opp_phi=home_phi,
            score=s_away,
            tau=self.tau,
            mov_factor=mov_factor,
        )

        self._mu[home_id] = new_home_mu
        self._phi[home_id] = new_home_phi
        self._sigma[home_id] = new_home_sigma

        self._mu[away_id] = new_away_mu
        self._phi[away_id] = new_away_phi
        self._sigma[away_id] = new_away_sigma

        home_r_after = self.rating(home_id)
        away_r_after = self.rating(away_id)
        return home_r_after - home_r_before, away_r_after - away_r_before

    def regress_to_mean(self, weight: float | None = None) -> None:
        """Apply off-season regression.

        For Glicko-2, "regression" means:
        - Pull mu toward 0 (ratings toward 1500) by `weight`.
        - Increase phi (inflate uncertainty) to model off-season roster changes.

        PHI is inflated by adding a fixed increment (0.5 in Glicko-1 = 0.5/173.7 in Glicko-2)
        rather than by the season_regression weight, because off-season uncertainty
        is additive in Glicko-2 (you're unsure HOW MUCH the team changed, not just that
        it changed directionally).
        """
        w = self.season_regression if weight is None else weight
        if w < 0 or w > 1:
            raise ValueError(f"regression weight must be in [0, 1], got {w}")

        # RD inflation for off-season: add phi increment = 63.2/173.7 ≈ 0.364
        # 63.2 Glicko-1 points is a common off-season inflation choice
        phi_increment = 63.2 / _SCALE

        for team_id in list(self._mu.keys()):
            # Pull rating toward mean
            self._mu[team_id] = self._mu[team_id] * (1 - w)
            # Inflate RD to model off-season uncertainty, capped at initial default
            new_phi = math.sqrt(self._phi[team_id] ** 2 + phi_increment**2)
            self._phi[team_id] = min(new_phi, _DEFAULT_PHI)


# ---------------------------------------------------------------------------
# Glicko-2 core mathematics
# ---------------------------------------------------------------------------


def _g(phi: float) -> float:
    """Glicko-2 g function."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:  # noqa: N802
    """Expected score for player with rating mu against opponent (mu_j, phi_j)."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _glicko2_update(
    mu: float,
    phi: float,
    sigma: float,
    opp_mu: float,
    opp_phi: float,
    score: float,
    tau: float,
    mov_factor: float = 1.0,
) -> tuple[float, float, float]:
    """Apply Glicko-2 update for a single game (single-opponent period).

    Implements the full Glickman (2012) algorithm steps 1-7 for a one-game period.
    Returns updated (mu, phi, sigma).

    ``mov_factor``: margin-of-victory multiplier. Scales the mu update directly —
    a larger-margin win moves the rating further in the winning direction. The phi
    (RD) update is based on standard Glicko-2 variance so RD always decreases
    correctly after a game (we don't inflate uncertainty based on MOV, only scale
    the directional update). Defaults to 1.0 (standard Glicko-2).
    """
    # Step 3: compute g and E for the opponent
    g_j = _g(opp_phi)
    e_j = _E(mu, opp_mu, opp_phi)  # noqa: N806 — uppercase E_j matches Glickman paper

    # Step 4: compute estimated variance v
    v = 1.0 / (g_j**2 * e_j * (1.0 - e_j))

    # Step 5: compute the score-based improvement delta (standard Glicko-2)
    delta = v * g_j * (score - e_j)

    # Step 6: update sigma using Newton-Raphson / Illinois algorithm
    new_sigma = _update_sigma(phi, sigma, delta, v, tau)

    # Step 7: update phi* and then mu, phi
    phi_star = math.sqrt(phi**2 + new_sigma**2)
    new_phi = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
    # MOV multiplier applied to the mu update: larger margin = larger rating shift.
    # phi^2 * g_j * (score - e_j) is the standard update; we scale it by mov_factor.
    new_mu = mu + mov_factor * new_phi**2 * g_j * (score - e_j)

    return new_mu, new_phi, new_sigma


def _update_sigma(
    phi: float,
    sigma: float,
    delta: float,
    v: float,
    tau: float,
) -> float:
    """Update volatility sigma using the Illinois algorithm (Glickman 2012, step 6)."""
    # f function from the Glicko-2 paper
    phi2 = phi**2
    delta2 = delta**2
    a = math.log(sigma**2)

    def f(x: float) -> float:
        ex = math.exp(x)
        tmp = phi2 + v + ex
        return ex * (delta2 - tmp) / (2.0 * tmp**2) - (x - a) / tau**2

    # Initial bracket — variable names match Glickman (2012) algorithm notation
    bracket_a = a  # noqa: N806
    if delta2 > phi2 + v:
        bracket_b = math.log(delta2 - phi2 - v)  # noqa: N806
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        bracket_b = a - k * tau  # noqa: N806

    f_a = f(bracket_a)
    f_b = f(bracket_b)

    # Illinois algorithm iteration
    for _ in range(100):
        bracket_c = bracket_a + (bracket_a - bracket_b) * f_a / (f_b - f_a)  # noqa: N806
        f_c = f(bracket_c)
        if f_c * f_b <= 0:
            bracket_a = bracket_b
            f_a = f_b
        else:
            f_a = f_a / 2.0
        bracket_b = bracket_c
        f_b = f_c
        if abs(bracket_b - bracket_a) < _EPSILON:
            break

    return math.exp(bracket_a / 2.0)
