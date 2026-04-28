"""Unit tests for the Bayesian hierarchical team-strength model.

Tests that do not require PyMC5/JAX run unconditionally.  Tests that invoke
``train_bayesian_hierarchical`` are skipped with ``pytest.importorskip`` when
those packages are not installed in the current environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fantasy_coach.models.bayesian_hierarchical import (
    BayesianHierarchicalModel,
    BayesianTrainingMatch,
    MarginDistribution,
    TeamPosterior,
    _hdi,
    load_bayesian_hierarchical,
    save_bayesian_hierarchical,
    train_bayesian_hierarchical,
)

# ---------------------------------------------------------------------------
# Fixtures — no PyMC required
# ---------------------------------------------------------------------------


def _make_model(
    *,
    n_teams: int = 3,
    use_dixon_coles: bool = True,
    dc_rho: float = 0.05,
    with_trace: bool = False,
) -> BayesianHierarchicalModel:
    """Manually construct a BayesianHierarchicalModel for inference-only tests."""
    ids = tuple(range(1, n_teams + 1))
    index = {t: i for i, t in enumerate(ids)}
    # Team 1 is strongest (highest attack, highest defense).
    # Team 3 is weakest.
    attack = np.array([0.3, 0.0, -0.3][:n_teams])
    defense = np.array([0.3, 0.0, -0.3][:n_teams])
    trace = None
    if with_trace:
        rng = np.random.default_rng(0)
        n = 50
        trace = {
            "attack": rng.normal(attack, 0.05, size=(n, n_teams)),
            "defense": rng.normal(defense, 0.05, size=(n, n_teams)),
            "home_adv": rng.normal(0.1, 0.02, size=n),
            "mu": rng.normal(np.log(18.0), 0.02, size=n),
        }
    return BayesianHierarchicalModel(
        team_ids=ids,
        team_index=index,
        attack_mean=attack,
        attack_std=np.ones(n_teams) * 0.1,
        defense_mean=defense,
        defense_std=np.ones(n_teams) * 0.1,
        home_advantage=0.1,
        mu=np.log(18.0),
        use_dixon_coles=use_dixon_coles,
        dc_rho=dc_rho,
        trace_samples=trace,
    )


def _make_matches(
    n: int = 30,
    *,
    team_ids: tuple[int, int, int] = (1, 2, 3),
) -> list[BayesianTrainingMatch]:
    """Synthetic 3-team round-robin matches (team 1 dominates)."""
    rng = np.random.default_rng(0)
    teams = list(team_ids)
    matches: list[BayesianTrainingMatch] = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        if home == away:
            away = teams[(i + 2) % len(teams)]
        # Team 1 scores ~30, others ~18
        base_h = 30 if home == team_ids[0] else 18
        base_a = 30 if away == team_ids[0] else 18
        hs = int(max(0, rng.normal(base_h, 4)))
        as_ = int(max(0, rng.normal(base_a, 4)))
        matches.append(
            BayesianTrainingMatch(
                home_team_id=home,
                away_team_id=away,
                home_score=hs,
                away_score=as_,
                season=2024,
            )
        )
    return matches


# ---------------------------------------------------------------------------
# _hdi
# ---------------------------------------------------------------------------


def test_hdi_returns_tuple_of_two_ints() -> None:
    pmf = np.array([0.1, 0.4, 0.3, 0.2])
    lo, hi = _hdi(pmf, 0.8)
    assert isinstance(lo, int)
    assert isinstance(hi, int)


def test_hdi_lo_le_hi() -> None:
    rng = np.random.default_rng(42)
    raw = rng.exponential(1, size=101)
    pmf = raw / raw.sum()
    lo, hi = _hdi(pmf, 0.90)
    assert lo <= hi


def test_hdi_full_mass_covers_all() -> None:
    pmf = np.zeros(101)
    pmf[50] = 1.0  # all mass at index 50
    lo, hi = _hdi(pmf, 1.0)
    assert lo == hi  # single-point distribution


# ---------------------------------------------------------------------------
# margin_pmf
# ---------------------------------------------------------------------------


def test_margin_pmf_sums_to_one() -> None:
    model = _make_model()
    pmf = model.margin_pmf(1, 2)
    assert abs(pmf.sum() - 1.0) < 1e-6


def test_margin_pmf_non_negative() -> None:
    model = _make_model()
    pmf = model.margin_pmf(1, 3)
    assert (pmf >= 0).all()


def test_margin_pmf_shape() -> None:
    model = _make_model()
    pmf = model.margin_pmf(1, 2)
    assert pmf.shape == (101,)  # −50 … +50 inclusive


def test_margin_pmf_without_dc_sums_to_one() -> None:
    model = _make_model(use_dixon_coles=False)
    pmf = model.margin_pmf(1, 2)
    assert abs(pmf.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# win_probability
# ---------------------------------------------------------------------------


def test_win_probability_in_unit_interval() -> None:
    model = _make_model()
    for home in (1, 2, 3):
        for away in (1, 2, 3):
            if home == away:
                continue
            p = model.win_probability(home, away)
            assert 0.0 <= p <= 1.0, f"win_prob out of range for {home} vs {away}: {p}"


def test_stronger_home_team_wins_more() -> None:
    """Team 1 (highest attack/defense) should beat team 3 more often than 50%."""
    model = _make_model()
    p_strong_wins = model.win_probability(1, 3)
    assert p_strong_wins > 0.5


def test_weaker_home_team_wins_less() -> None:
    """Team 3 (lowest attack/defense) should lose to team 1 more often than it wins."""
    model = _make_model()
    p_weak_wins = model.win_probability(3, 1)
    assert p_weak_wins < 0.5


def test_symmetric_teams_equal_win_probability() -> None:
    """With equal teams and no home advantage, P(home wins) == P(away wins)."""
    ids = (10, 11)
    index = {10: 0, 11: 1}
    model = BayesianHierarchicalModel(
        team_ids=ids,
        team_index=index,
        attack_mean=np.zeros(2),
        attack_std=np.ones(2) * 0.1,
        defense_mean=np.zeros(2),
        defense_std=np.ones(2) * 0.1,
        home_advantage=0.0,
        mu=np.log(18.0),
        use_dixon_coles=False,
        dc_rho=0.0,
    )
    # Identical teams, no home advantage — symmetric (draws eat into each side equally)
    p_home = model.win_probability(10, 11)
    p_away = model.win_probability(11, 10)
    assert p_home == pytest.approx(p_away, abs=1e-6)


# ---------------------------------------------------------------------------
# margin_distribution
# ---------------------------------------------------------------------------


def test_margin_distribution_probabilities_sum_to_one() -> None:
    model = _make_model()
    dist = model.margin_distribution(1, 2)
    assert abs(dist.home_win_prob + dist.draw_prob + dist.away_win_prob - 1.0) < 1e-6


def test_margin_distribution_hdi_ordered() -> None:
    model = _make_model()
    dist = model.margin_distribution(1, 2)
    lo80, hi80 = dist.ci_80
    lo95, hi95 = dist.ci_95
    assert lo80 <= hi80
    assert lo95 <= hi95
    # 95% interval should be at least as wide as 80%
    assert (hi95 - lo95) >= (hi80 - lo80)


def test_margin_distribution_with_trace() -> None:
    model = _make_model(with_trace=True)
    dist = model.margin_distribution(1, 2, n_samples=20)
    assert isinstance(dist, MarginDistribution)
    assert abs(dist.pmf.sum() - 1.0) < 1e-6


def test_margin_distribution_no_trace_falls_back() -> None:
    model = _make_model(with_trace=False)
    dist = model.margin_distribution(1, 2)
    assert abs(dist.pmf.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# team_posterior
# ---------------------------------------------------------------------------


def test_team_posterior_returns_correct_values() -> None:
    model = _make_model()
    tp = model.team_posterior(1)
    assert isinstance(tp, TeamPosterior)
    assert tp.team_id == 1
    assert tp.attack_mean == pytest.approx(model.attack_mean[0])
    assert tp.defense_mean == pytest.approx(model.defense_mean[0])


# ---------------------------------------------------------------------------
# Persistence — no PyMC required
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = _make_model(with_trace=True)
    path = tmp_path / "bh.joblib"
    save_bayesian_hierarchical(path, model)
    loaded = load_bayesian_hierarchical(path)

    assert loaded.team_ids == model.team_ids
    np.testing.assert_allclose(loaded.attack_mean, model.attack_mean)
    np.testing.assert_allclose(loaded.defense_mean, model.defense_mean)
    assert loaded.home_advantage == pytest.approx(model.home_advantage)
    assert loaded.use_dixon_coles == model.use_dixon_coles
    assert loaded.dc_rho == pytest.approx(model.dc_rho)
    # Trace is preserved
    assert loaded.trace_samples is not None
    np.testing.assert_allclose(
        loaded.trace_samples["attack"],
        model.trace_samples["attack"],  # type: ignore[index]
    )


def test_load_rejects_wrong_model_type(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"model_type": "logistic"}, path)
    with pytest.raises(ValueError, match="bayesian_hierarchical"):
        load_bayesian_hierarchical(path)


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "nested" / "dir" / "bh.joblib"
    save_bayesian_hierarchical(path, model)
    assert path.exists()


# ---------------------------------------------------------------------------
# loader.py dispatch
# ---------------------------------------------------------------------------


def test_loader_dispatches_bayesian_hierarchical(tmp_path: Path) -> None:
    from fantasy_coach.models.loader import load_model

    model = _make_model()
    path = tmp_path / "bh.joblib"
    save_bayesian_hierarchical(path, model)

    # loader.py dispatches by model_type; it returns the raw
    # BayesianHierarchicalModel which is not yet a full Model-protocol object
    # (ensemble integration is PR B).  Just verify no error is raised.
    loaded = load_model(path)
    assert isinstance(loaded, BayesianHierarchicalModel)


# ---------------------------------------------------------------------------
# Training — skipped if PyMC5 / numpyro not installed
# ---------------------------------------------------------------------------


@pytest.fixture()
def _require_pymc() -> None:
    pytest.importorskip("pymc", reason="PyMC5 not installed (training extra)")
    pytest.importorskip("numpyro", reason="numpyro not installed (training extra)")


def test_train_raises_with_too_few_matches(_require_pymc: None) -> None:
    matches = _make_matches(5)
    with pytest.raises(ValueError, match="at least 10"):
        train_bayesian_hierarchical(matches, n_samples=50, n_warmup=50, n_chains=1)


def test_train_returns_model_with_correct_teams(_require_pymc: None) -> None:
    matches = _make_matches(40)
    model = train_bayesian_hierarchical(
        matches,
        n_samples=100,
        n_warmup=100,
        n_chains=1,
        random_seed=0,
    )
    assert set(model.team_ids) == {1, 2, 3}
    assert model.attack_mean.shape == (3,)
    assert model.defense_mean.shape == (3,)


def test_train_team1_stronger_than_team3(_require_pymc: None) -> None:
    """After training on data where team 1 always outscores team 3, the model
    should learn that team 1 has higher attack + defense than team 3."""
    matches = _make_matches(60)
    model = train_bayesian_hierarchical(
        matches,
        n_samples=200,
        n_warmup=200,
        n_chains=1,
        random_seed=42,
    )
    i1 = model.team_index[1]
    i3 = model.team_index[3]
    assert model.attack_mean[i1] > model.attack_mean[i3], (
        "Team 1 attack should be higher than team 3 attack after training"
    )


def test_train_win_prob_reflects_strength(_require_pymc: None) -> None:
    matches = _make_matches(60)
    model = train_bayesian_hierarchical(
        matches,
        n_samples=200,
        n_warmup=200,
        n_chains=1,
        random_seed=42,
    )
    p = model.win_probability(1, 3)
    assert p > 0.5, f"Strong team (1) vs weak team (3): expected >0.5, got {p:.3f}"


def test_train_stores_trace(_require_pymc: None) -> None:
    matches = _make_matches(30)
    model = train_bayesian_hierarchical(
        matches,
        n_samples=100,
        n_warmup=100,
        n_chains=1,
        random_seed=0,
    )
    assert model.trace_samples is not None
    assert model.trace_samples["attack"].shape[1] == 3
    assert model.trace_samples["attack"].shape[0] <= 500


def test_train_power_prior_tightens_posterior(_require_pymc: None) -> None:
    """With a very tight power-prior, the posterior should stay near the prior mean."""
    matches = _make_matches(20)
    # Strong prior: team 1 attack = 1.0 with tiny std (alpha=10 means std/sqrt(10))
    prior = {
        1: TeamPosterior(
            team_id=1, attack_mean=1.0, attack_std=0.01, defense_mean=1.0, defense_std=0.01
        )
    }
    model = train_bayesian_hierarchical(
        matches,
        prior_posteriors=prior,
        power_prior_alpha=10.0,  # very tight prior
        n_samples=100,
        n_warmup=100,
        n_chains=1,
        random_seed=0,
    )
    i1 = model.team_index[1]
    # With a tight prior at 1.0, posterior mean should be pulled toward 1.0
    assert model.attack_mean[i1] > 0.3, (
        f"Expected power-prior to pull team 1 attack toward 1.0, got {model.attack_mean[i1]:.3f}"
    )


def test_train_roundtrip_save_load(_require_pymc: None, tmp_path: Path) -> None:
    matches = _make_matches(30)
    model = train_bayesian_hierarchical(
        matches,
        n_samples=100,
        n_warmup=100,
        n_chains=1,
        random_seed=0,
    )
    path = tmp_path / "bh_trained.joblib"
    save_bayesian_hierarchical(path, model)
    loaded = load_bayesian_hierarchical(path)
    assert loaded.team_ids == model.team_ids
    p_orig = model.win_probability(1, 3)
    p_loaded = loaded.win_probability(1, 3)
    assert p_orig == pytest.approx(p_loaded)
