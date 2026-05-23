"""Tests for the Bayesian hierarchical team-strength model (#144).

These tests require PyMC5 (``uv sync --extra training``). They are
automatically skipped in environments where pymc is not installed (e.g., CI
running ``uv sync --all-groups`` without the training extra).

The synthetic 3-team dataset uses minimal sampling settings (50 tune +
100 draws, 1 chain) to keep wall-clock time under 60 s on a single CPU.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if pymc is unavailable (training extras not installed).
pymc = pytest.importorskip("pymc", reason="pymc not installed; run: uv sync --extra training")

from fantasy_coach.features import MatchRow, TeamRow  # noqa: E402
from fantasy_coach.models.bayesian_hierarchical import (  # noqa: E402
    BayesianHierarchicalModel,
    BayesianMatchData,
    BayesianTrainResult,
    _hdi,
    _skellam_home_win_prob,
    build_bayesian_frame,
    load_bayesian_hierarchical,
    save_bayesian_hierarchical,
    train_bayesian_hierarchical,
)

# ---------------------------------------------------------------------------
# Synthetic 3-team dataset helpers
# ---------------------------------------------------------------------------

_TEAM_STRONG = 1  # always wins big at home
_TEAM_MEDIUM = 2
_TEAM_WEAK = 3

# Approximate expected scores per team
_SCORES: dict[tuple[int, int], tuple[int, int]] = {
    # (home, away): (home_score, away_score)
    (_TEAM_STRONG, _TEAM_MEDIUM): (30, 12),
    (_TEAM_STRONG, _TEAM_WEAK): (36, 8),
    (_TEAM_MEDIUM, _TEAM_WEAK): (22, 14),
    (_TEAM_MEDIUM, _TEAM_STRONG): (14, 26),
    (_TEAM_WEAK, _TEAM_STRONG): (10, 34),
    (_TEAM_WEAK, _TEAM_MEDIUM): (12, 20),
}


def _make_match(
    match_id: int,
    home_score: int,
    away_score: int,
    home_id: int = _TEAM_STRONG,
    away_id: int = _TEAM_MEDIUM,
    offset_hours: int = 0,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=2024,
        round=1 + match_id % 25,
        start_time=datetime(2024, 3, 1, 9 + offset_hours % 10, 0, 0, tzinfo=UTC),
        match_state="FullTime",
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(
            team_id=home_id,
            name=f"Team{home_id}",
            nick_name=f"T{home_id}",
            score=home_score,
            players=[],
        ),
        away=TeamRow(
            team_id=away_id,
            name=f"Team{away_id}",
            nick_name=f"T{away_id}",
            score=away_score,
            players=[],
        ),
        team_stats=[],
    )


def _synthetic_matches(reps_per_pair: int = 5) -> list[MatchRow]:
    """Generate a 3-team round-robin dataset with ``reps_per_pair`` repeats."""
    matches: list[MatchRow] = []
    mid = 0
    for (home_id, away_id), (hs, as_) in _SCORES.items():
        for _r in range(reps_per_pair):
            rng = np.random.default_rng(mid)
            h = int(rng.poisson(hs))
            a = int(rng.poisson(as_))
            matches.append(
                _make_match(mid, h, a, home_id=home_id, away_id=away_id, offset_hours=mid)
            )
            mid += 1
    return matches


def _minimal_data(reps: int = 5) -> BayesianMatchData:
    return build_bayesian_frame(_synthetic_matches(reps))


# Shared trained model (built once per module, sampled at minimal settings).
@pytest.fixture(scope="module")
def trained_result() -> BayesianTrainResult:
    data = _minimal_data(reps=5)  # 30 matches across 3 teams
    return train_bayesian_hierarchical(
        data,
        n_tune=50,
        n_samples=100,
        n_chains=1,
        random_seed=7,
        progressbar=False,
    )


# ---------------------------------------------------------------------------
# build_bayesian_frame
# ---------------------------------------------------------------------------


def test_build_bayesian_frame_shape() -> None:
    matches = _synthetic_matches(reps=3)
    data = build_bayesian_frame(matches)
    n = len(matches)
    assert data.home_idx.shape == (n,)
    assert data.away_idx.shape == (n,)
    assert data.home_score.shape == (n,)
    assert data.away_score.shape == (n,)
    assert len(data.teams) == 3


def test_build_bayesian_frame_team_index_in_range() -> None:
    data = _minimal_data()
    n_teams = len(data.teams)
    assert (data.home_idx >= 0).all()
    assert (data.away_idx >= 0).all()
    assert (data.home_idx < n_teams).all()
    assert (data.away_idx < n_teams).all()


def test_build_bayesian_frame_excludes_incomplete_matches() -> None:
    complete = _make_match(0, 20, 10)
    incomplete = MatchRow(
        match_id=1,
        season=2024,
        round=1,
        start_time=datetime(2024, 3, 2, 9, 0, 0, tzinfo=UTC),
        match_state="Upcoming",
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(team_id=1, name="T1", nick_name="T1", score=None, players=[]),
        away=TeamRow(team_id=2, name="T2", nick_name="T2", score=None, players=[]),
        team_stats=[],
    )
    data = build_bayesian_frame([complete, incomplete])
    assert data.home_score.shape[0] == 1


def test_build_bayesian_frame_empty() -> None:
    data = build_bayesian_frame([])
    assert data.home_idx.shape == (0,)
    assert data.teams == ()


def test_build_bayesian_frame_team_names_populated() -> None:
    data = _minimal_data()
    for team_id in data.teams:
        assert team_id in data.team_names


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------


def test_skellam_home_win_prob_range() -> None:
    for lh, la in [(20, 15), (10, 25), (18, 18)]:
        p = _skellam_home_win_prob(float(lh), float(la))
        assert 0.0 <= p <= 1.0


def test_skellam_home_win_prob_higher_rate_wins_more() -> None:
    assert _skellam_home_win_prob(30.0, 10.0) > _skellam_home_win_prob(10.0, 30.0)


def test_hdi_returns_ordered_bounds() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 5, 1000)
    lo, hi = _hdi(samples, 0.80)
    assert lo <= hi


def test_hdi_80_narrower_than_95() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 5, 1000)
    lo80, hi80 = _hdi(samples, 0.80)
    lo95, hi95 = _hdi(samples, 0.95)
    assert (hi80 - lo80) <= (hi95 - lo95)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_returns_correct_n_train(trained_result: BayesianTrainResult) -> None:
    assert trained_result.n_train == 30
    assert trained_result.n_teams == 3


def test_train_raises_with_too_few_matches() -> None:
    data = _minimal_data(reps=1)  # 6 matches — below the 10-match minimum
    data_small = BayesianMatchData(
        home_idx=data.home_idx[:5],
        away_idx=data.away_idx[:5],
        home_score=data.home_score[:5],
        away_score=data.away_score[:5],
        teams=data.teams,
    )
    with pytest.raises(ValueError, match="at least 10"):
        train_bayesian_hierarchical(data_small, n_tune=10, n_samples=10, n_chains=1)


def test_train_raises_with_one_team() -> None:
    data = BayesianMatchData(
        home_idx=np.zeros(15, dtype=int),
        away_idx=np.zeros(15, dtype=int),
        home_score=np.full(15, 20, dtype=int),
        away_score=np.full(15, 15, dtype=int),
        teams=(1,),
    )
    with pytest.raises(ValueError, match="at least 2"):
        train_bayesian_hierarchical(data, n_tune=10, n_samples=10, n_chains=1)


def test_posterior_samples_keys(trained_result: BayesianTrainResult) -> None:
    ps = trained_result.model.posterior_samples
    for key in ("att", "dfn", "home_adv", "mu"):
        assert key in ps, f"Missing posterior key: {key}"


def test_att_shape_matches_n_teams(trained_result: BayesianTrainResult) -> None:
    att = trained_result.model.posterior_samples["att"]
    assert att.ndim == 2
    assert att.shape[1] == trained_result.n_teams


def test_att_approx_zero_sum(trained_result: BayesianTrainResult) -> None:
    att = trained_result.model.posterior_samples["att"]
    row_sums = att.sum(axis=1)
    # ZeroSumNormal: each draw's team-wise sum should be near zero
    assert np.abs(row_sums).max() < 1e-6, "att posterior rows should sum to 0"


# ---------------------------------------------------------------------------
# BayesianHierarchicalModel predictions
# ---------------------------------------------------------------------------


def test_predict_win_prob_valid_range(trained_result: BayesianTrainResult) -> None:
    model = trained_result.model
    p = model.predict_win_prob(_TEAM_STRONG, _TEAM_WEAK)
    assert 0.0 <= p <= 1.0


def test_stronger_team_wins_more_often(trained_result: BayesianTrainResult) -> None:
    model = trained_result.model
    p_strong_home = model.predict_win_prob(_TEAM_STRONG, _TEAM_WEAK)
    p_weak_home = model.predict_win_prob(_TEAM_WEAK, _TEAM_STRONG)
    assert p_strong_home > p_weak_home


def test_predict_win_prob_unknown_team_returns_half(
    trained_result: BayesianTrainResult,
) -> None:
    model = trained_result.model
    assert model.predict_win_prob(999, _TEAM_STRONG) == pytest.approx(0.5)
    assert model.predict_win_prob(_TEAM_STRONG, 999) == pytest.approx(0.5)


def test_predict_margin_hdi_ordered(trained_result: BayesianTrainResult) -> None:
    model = trained_result.model
    hdi = model.predict_margin_hdi(_TEAM_STRONG, _TEAM_WEAK)
    assert hdi["hdi_80_lo"] <= hdi["hdi_80_hi"]
    assert hdi["hdi_95_lo"] <= hdi["hdi_95_hi"]


def test_predict_margin_hdi_80_inside_95(trained_result: BayesianTrainResult) -> None:
    model = trained_result.model
    hdi = model.predict_margin_hdi(_TEAM_STRONG, _TEAM_WEAK)
    assert hdi["hdi_95_lo"] <= hdi["hdi_80_lo"]
    assert hdi["hdi_80_hi"] <= hdi["hdi_95_hi"]


def test_predict_margin_hdi_unknown_team_returns_default(
    trained_result: BayesianTrainResult,
) -> None:
    model = trained_result.model
    hdi = model.predict_margin_hdi(999, _TEAM_STRONG)
    assert hdi["mean"] == pytest.approx(0.0)


def test_team_strength_summary_keys(trained_result: BayesianTrainResult) -> None:
    model = trained_result.model
    summary = model.team_strength_summary
    assert set(summary.keys()) == set(model.teams)
    for stats in summary.values():
        assert "attack" in stats
        assert "defense" in stats


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path, trained_result: BayesianTrainResult) -> None:
    path = tmp_path / "bayes.joblib"
    save_bayesian_hierarchical(path, trained_result)

    loaded = load_bayesian_hierarchical(path)
    assert isinstance(loaded, BayesianHierarchicalModel)
    assert loaded.teams == trained_result.model.teams

    orig_p = trained_result.model.predict_win_prob(_TEAM_STRONG, _TEAM_WEAK)
    loaded_p = loaded.predict_win_prob(_TEAM_STRONG, _TEAM_WEAK)
    assert orig_p == pytest.approx(loaded_p, abs=1e-9)


def test_load_rejects_wrong_model_type(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"model_type": "logistic"}, path)
    with pytest.raises(ValueError, match="bayesian_hierarchical"):
        load_bayesian_hierarchical(path)
