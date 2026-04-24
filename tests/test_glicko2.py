"""Tests for the Glicko-2 rating system.

Covers the core mathematical properties described in Glickman (2012):
- After a win, rating increases; after a loss, rating decreases.
- Rating deviation (phi) decreases as more matches are played (certainty grows).
- RD increases after season regression (off-season uncertainty).
- Symmetric teams predict ~50/50 (adjusted for home advantage).
- Underdog win produces larger delta than favourite win.
"""

from __future__ import annotations

import math

import pytest

from fantasy_coach.models.glicko2 import (
    _DEFAULT_PHI,
    Glicko2,
    _glicko2_update,
)

# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_new_team_starts_at_initial_rating() -> None:
    g = Glicko2()
    assert g.rating(99) == 1500.0


def test_default_home_advantage() -> None:
    g = Glicko2()
    assert g.home_advantage == 55.0


def test_new_team_has_high_phi() -> None:
    """New teams start with high RD (uncertainty)."""
    g = Glicko2()
    _, phi, sigma = g._get_state(1)
    assert phi == pytest.approx(_DEFAULT_PHI, rel=1e-6)
    assert sigma == 0.06


# ---------------------------------------------------------------------------
# Win/loss update direction
# ---------------------------------------------------------------------------


def test_winner_rating_increases_loser_decreases() -> None:
    g = Glicko2()
    home_r_before = g.rating(1)
    away_r_before = g.rating(2)

    g.update(1, 2, 24, 12)

    assert g.rating(1) > home_r_before  # home won, rating should go up
    assert g.rating(2) < away_r_before  # away lost, rating should go down


def test_loser_as_home_rating_decreases() -> None:
    g = Glicko2()
    home_r_before = g.rating(1)
    away_r_before = g.rating(2)

    g.update(1, 2, 6, 24)  # away wins

    assert g.rating(1) < home_r_before  # home lost
    assert g.rating(2) > away_r_before  # away won


def test_single_win_updates_rd_correctly() -> None:
    """After one game, RD should decrease from the initial high value.

    This is the key Glicko-2 property: playing a game reduces uncertainty
    (phi decreases). The exact value depends on tau and the prior, but
    it must be strictly less than _DEFAULT_PHI.
    """
    g = Glicko2()
    _, phi_before, _ = g._get_state(1)
    assert phi_before == pytest.approx(_DEFAULT_PHI)

    g.update(1, 2, 24, 12)

    _, phi_after, _ = g._get_state(1)
    # After playing one match, RD should have decreased (more certain about rating)
    assert phi_after < phi_before, (
        f"RD should decrease after a game, got {phi_after} >= {phi_before}"
    )


# ---------------------------------------------------------------------------
# Symmetric teams → ~50% win probability
# ---------------------------------------------------------------------------


def test_symmetric_teams_predict_near_50_plus_home_advantage() -> None:
    """With no history, home-advantage should give home > 50% win probability.

    At high initial phi (uncertain teams), the Glicko-2 g-function compresses
    win probabilities toward 0.5 compared to plain Elo. The home advantage still
    gives > 50%, but the margin is smaller than plain Elo would compute.
    """
    g = Glicko2()
    p = g.predict(1, 2)
    # Both at initial rating, home advantage should push probability above 50%
    # Glicko-2 with high initial phi gives ~0.54 (compressed by g-function)
    assert p > 0.50, f"Home advantage should give > 50% home win prob, got {p}"
    assert p < 0.70, f"Win prob should not be excessively high for equal teams, got {p}"


def test_equal_ratings_give_symmetric_predictions() -> None:
    """Two teams with equal ratings: p(1 beats 2) + p(2 beats 1) should both be ~0.5 (excl HA)."""
    g = Glicko2(home_advantage=0.0)  # no home advantage
    p = g.predict(1, 2)
    assert math.isclose(p, 0.5, abs_tol=1e-6), f"Equal teams with no HA should give 0.5, got {p}"


# ---------------------------------------------------------------------------
# MOV weighting — larger margin → larger delta
# ---------------------------------------------------------------------------


def test_larger_margin_produces_larger_rating_delta() -> None:
    """Blowout should move ratings more than narrow win."""
    close = Glicko2()
    blowout = Glicko2()

    home_before_close = close.rating(1)
    home_before_blowout = blowout.rating(1)

    close.update(1, 2, 14, 12)
    blowout.update(1, 2, 40, 0)

    delta_close = close.rating(1) - home_before_close
    delta_blowout = blowout.rating(1) - home_before_blowout

    assert delta_blowout > delta_close, "Blowout should produce larger positive delta"


# ---------------------------------------------------------------------------
# Season regression: RD should increase
# ---------------------------------------------------------------------------


def test_season_regression_increases_phi() -> None:
    """After season regression, phi (RD) should increase to model off-season uncertainty.

    This is the core Glicko-2 off-season property: even if we're certain
    about a team's rating at season end, we become less certain during
    the off-season due to roster changes, coaching changes, etc.
    """
    g = Glicko2()
    # Play some matches to reduce phi
    for _ in range(5):
        g.update(1, 2, 24, 12)
        g.update(2, 1, 18, 10)

    _, phi_mid_season, _ = g._get_state(1)
    assert phi_mid_season < _DEFAULT_PHI  # phi should be less than initial after games

    g.regress_to_mean()

    _, phi_after_regression, _ = g._get_state(1)
    # After regression, phi should increase (more uncertainty)
    assert phi_after_regression > phi_mid_season, (
        f"Phi should increase after season regression: {phi_after_regression} <= {phi_mid_season}"
    )


def test_season_regression_pulls_rating_toward_1500() -> None:
    """Season regression should pull extreme ratings toward 1500."""
    g = Glicko2(season_regression=0.5)

    # Build up a high rating for team 1
    for _ in range(10):
        g.update(1, 2, 30, 0)

    r_before = g.rating(1)
    assert r_before > 1500.0

    g.regress_to_mean()

    r_after = g.rating(1)
    assert 1500.0 < r_after < r_before, "Regression should pull rating toward 1500"


def test_season_regression_invalid_weight_raises() -> None:
    g = Glicko2()
    with pytest.raises(ValueError):
        g.regress_to_mean(weight=2.0)

    with pytest.raises(ValueError):
        g.regress_to_mean(weight=-0.1)


# ---------------------------------------------------------------------------
# Rating scale consistency
# ---------------------------------------------------------------------------


def test_rating_returns_glicko1_scale() -> None:
    """rating() should return values in the Glicko-1 scale (centred at 1500)."""
    g = Glicko2()
    # New team should be exactly 1500
    assert g.rating(1) == pytest.approx(1500.0)


def test_rating_increases_monotonically_with_wins() -> None:
    """More wins against the same opponent should increase rating monotonically."""
    g = Glicko2()
    ratings = [g.rating(1)]
    for _ in range(5):
        g.update(1, 2, 24, 12)
        ratings.append(g.rating(1))
    for i in range(1, len(ratings)):
        assert ratings[i] > ratings[i - 1], f"Rating should increase with each win: {ratings}"


# ---------------------------------------------------------------------------
# Core update function
# ---------------------------------------------------------------------------


def test_glicko2_update_returns_valid_triple() -> None:
    """The core update function should return finite, positive-phi values."""
    new_mu, new_phi, new_sigma = _glicko2_update(
        mu=0.0,
        phi=2.0148,
        sigma=0.06,
        opp_mu=0.0,
        opp_phi=2.0148,
        score=1.0,
        tau=0.5,
    )
    assert math.isfinite(new_mu)
    assert new_phi > 0.0
    assert new_sigma > 0.0


def test_glicko2_update_win_increases_mu() -> None:
    """Winning (score=1.0) against an equal opponent should increase mu."""
    new_mu, _, _ = _glicko2_update(
        mu=0.0,
        phi=2.0148,
        sigma=0.06,
        opp_mu=0.0,
        opp_phi=2.0148,
        score=1.0,
        tau=0.5,
    )
    assert new_mu > 0.0, f"Win should increase mu, got {new_mu}"


def test_glicko2_update_loss_decreases_mu() -> None:
    """Losing (score=0.0) against an equal opponent should decrease mu."""
    new_mu, _, _ = _glicko2_update(
        mu=0.0,
        phi=2.0148,
        sigma=0.06,
        opp_mu=0.0,
        opp_phi=2.0148,
        score=0.0,
        tau=0.5,
    )
    assert new_mu < 0.0, f"Loss should decrease mu, got {new_mu}"


# ---------------------------------------------------------------------------
# Predictor adapter
# ---------------------------------------------------------------------------


def test_glicko2_predictor_name() -> None:
    from fantasy_coach.evaluation.predictors import Glicko2Predictor

    p = Glicko2Predictor()
    assert p.name == "glicko2"


def test_glicko2_predictor_fit_and_predict() -> None:
    from pathlib import Path

    from fantasy_coach.evaluation.predictors import Glicko2Predictor
    from fantasy_coach.storage import SQLiteRepository

    db_path = Path(__file__).parent / "fixtures" / "baseline-nrl.db"
    if not db_path.exists():
        pytest.skip("baseline DB missing")

    repo = SQLiteRepository(db_path)
    try:
        matches = repo.list_matches(2024)
    finally:
        repo.close()

    predictor = Glicko2Predictor()
    predictor.fit(matches[:50])
    m = matches[50]
    prob = predictor.predict_home_win_prob(m)
    assert 0.0 < prob < 1.0


def test_glicko2_predictor_fit_empty_history() -> None:
    from fantasy_coach.evaluation.predictors import Glicko2Predictor

    predictor = Glicko2Predictor()
    predictor.fit([])
    # Should not raise; predict returns the home-advantage prior
    p = predictor.glicko2.predict(1, 2)
    assert 0.5 < p < 1.0  # home advantage should give > 50%
