"""Tests for the Bivariate Poisson + Dixon-Coles model (#209)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.models.bivariate_poisson import (
    BivariatePoissonModel,
    BivariatePrediction,
    BivariateTrainingFrame,
    ScoreLine,
    _dc_factor,
    _log_bp_pmf,
    _score_grid,
    build_bivariate_frame,
    load_bivariate_poisson,
    save_bivariate_poisson,
    train_bivariate_poisson,
)


def _make_match(
    match_id: int,
    home_score: int,
    away_score: int,
    home_id: int = 1,
    away_id: int = 2,
    offset_seconds: int = 0,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=2024,
        round=match_id,
        start_time=datetime(2024, 3, 1, 9, 0, offset_seconds, tzinfo=UTC),
        match_state="FullTime",
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(team_id=home_id, name="Home", nick_name="Home", score=home_score, players=[]),
        away=TeamRow(team_id=away_id, name="Away", nick_name="Away", score=away_score, players=[]),
        team_stats=[],
    )


def _minimal_frame(n: int = 25) -> BivariateTrainingFrame:
    matches = [
        _make_match(i, home_score=20 + i % 6, away_score=14 + i % 4, offset_seconds=i)
        for i in range(n)
    ]
    return build_bivariate_frame(matches)


# ---------------------------------------------------------------------------
# _log_bp_pmf
# ---------------------------------------------------------------------------


def test_log_bp_pmf_sum_to_one() -> None:
    """PMF values over a reasonable score range should sum ≈ 1."""
    lh, la, l3 = 22.0, 18.0, 1.5
    l1, l2 = lh - l3, la - l3
    total = 0.0
    for h in range(60):
        for a in range(60):
            total += np.exp(_log_bp_pmf(h, a, l1, l2, l3))
    assert abs(total - 1.0) < 0.01, f"PMF sums to {total:.4f}, expected ≈ 1.0"


def test_log_bp_pmf_zero_scores() -> None:
    """P(0, 0) should be positive and finite."""
    lp = _log_bp_pmf(0, 0, 5.0, 5.0, 0.5)
    assert np.isfinite(lp)
    assert lp < 0


def test_log_bp_pmf_degenerates_to_independent_when_l3_near_zero() -> None:
    """With λ3 → 0 the bivariate PMF should match the product of Poisson PMFs."""
    from scipy.stats import poisson

    l1, l2, l3 = 20.0, 15.0, 1e-5
    h, a = 18, 12
    bp_log_p = _log_bp_pmf(h, a, l1, l2, l3)
    indep_log_p = poisson.logpmf(h, l1) + poisson.logpmf(a, l2)
    # Should agree to within 0.01 log-units when l3 is tiny
    assert abs(bp_log_p - indep_log_p) < 0.01


# ---------------------------------------------------------------------------
# _dc_factor
# ---------------------------------------------------------------------------


def test_dc_factor_identity_when_tau_zero() -> None:
    for h in range(3):
        for a in range(3):
            assert _dc_factor(h, a, 20.0, 18.0, 0.0) == 1.0


def test_dc_factor_only_fires_for_low_scores() -> None:
    assert _dc_factor(2, 0, 20.0, 18.0, 0.05) == 1.0
    assert _dc_factor(0, 2, 20.0, 18.0, 0.05) == 1.0
    assert _dc_factor(2, 2, 20.0, 18.0, 0.05) == 1.0


def test_dc_factor_positive() -> None:
    """All correction factors must be positive (valid probability multiplier)."""
    for h in range(2):
        for a in range(2):
            assert _dc_factor(h, a, 20.0, 18.0, 0.001) > 0


# ---------------------------------------------------------------------------
# _score_grid
# ---------------------------------------------------------------------------


def test_score_grid_shape() -> None:
    g = _score_grid(22.0, 18.0, 1.5, 0.0)
    assert g.shape == (81, 81)


def test_score_grid_sums_to_one() -> None:
    g = _score_grid(22.0, 18.0, 1.5, 0.001)
    assert abs(g.sum() - 1.0) < 1e-6


def test_score_grid_all_non_negative() -> None:
    g = _score_grid(22.0, 18.0, 1.5, 0.001)
    assert np.all(g >= 0)


def test_score_grid_modal_score_sensible() -> None:
    """The most probable score should be near the expected rates."""
    g = _score_grid(25.0, 20.0, 1.0, 0.0)
    flat_idx = int(np.argmax(g))
    h_mode = flat_idx // 81
    a_mode = flat_idx % 81
    assert 15 <= h_mode <= 35
    assert 10 <= a_mode <= 30


# ---------------------------------------------------------------------------
# build_bivariate_frame
# ---------------------------------------------------------------------------


def test_build_bivariate_frame_shape() -> None:
    frame = _minimal_frame(20)
    assert frame.X.shape == (20, len(frame.feature_names))
    assert len(frame.y_home) == 20
    assert len(frame.y_away) == 20


def test_build_bivariate_frame_includes_draws() -> None:
    matches = [_make_match(i, 18, 18, offset_seconds=i) for i in range(5)]
    frame = build_bivariate_frame(matches)
    assert frame.X.shape[0] == 5


# ---------------------------------------------------------------------------
# train_bivariate_poisson
# ---------------------------------------------------------------------------


def test_train_bivariate_poisson_returns_model() -> None:
    frame = _minimal_frame(25)
    result = train_bivariate_poisson(frame, max_iter=50)
    assert isinstance(result.model, BivariatePoissonModel)
    assert result.n_train == 25


def test_train_bivariate_poisson_raises_too_few_rows() -> None:
    frame = _minimal_frame(5)
    with pytest.raises(ValueError, match="at least 10"):
        train_bivariate_poisson(frame)


def test_train_bivariate_poisson_l3_positive() -> None:
    frame = _minimal_frame(25)
    result = train_bivariate_poisson(frame, max_iter=30)
    assert result.model.l3 > 0


def test_train_bivariate_poisson_nll_finite() -> None:
    frame = _minimal_frame(25)
    result = train_bivariate_poisson(frame, max_iter=30)
    assert np.isfinite(result.final_nll)


# ---------------------------------------------------------------------------
# BivariatePoissonModel
# ---------------------------------------------------------------------------


def test_predict_home_win_prob_range() -> None:
    frame = _minimal_frame(25)
    model = train_bivariate_poisson(frame, max_iter=30).model
    probs = model.predict_home_win_prob(frame.X[:3])
    assert probs.shape == (3,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_predict_distribution_structure() -> None:
    frame = _minimal_frame(25)
    model = train_bivariate_poisson(frame, max_iter=30).model
    pred = model.predict_distribution(frame.X[:1])
    assert isinstance(pred, BivariatePrediction)
    assert pred.score_grid.shape == (81, 81)
    assert 0.0 <= pred.home_win_prob <= 1.0
    assert 0.0 <= pred.draw_prob <= 1.0
    assert pred.home_win_prob + pred.draw_prob <= 1.0
    assert np.isfinite(pred.predicted_margin)
    assert pred.predicted_total > 0
    assert len(pred.top_scorelines) == 5
    assert all(isinstance(s, ScoreLine) for s in pred.top_scorelines)


def test_predict_distribution_probabilities_sum_to_one() -> None:
    frame = _minimal_frame(25)
    model = train_bivariate_poisson(frame, max_iter=30).model
    pred = model.predict_distribution(frame.X[:1])
    total = pred.home_win_prob + pred.draw_prob + (1.0 - pred.home_win_prob - pred.draw_prob)
    assert abs(pred.score_grid.sum() - 1.0) < 1e-5


def test_stronger_home_gives_higher_win_prob() -> None:
    """Matches where home always wins big → higher home win probability."""
    matches = [_make_match(i, 40, 5, home_id=1, away_id=2, offset_seconds=i) for i in range(25)]
    frame = build_bivariate_frame(matches)
    model = train_bivariate_poisson(frame, max_iter=50).model
    pred = model.predict_distribution(frame.X[:1])
    assert pred.home_win_prob > 0.5


def test_top_scorelines_ordered_by_probability() -> None:
    frame = _minimal_frame(25)
    model = train_bivariate_poisson(frame, max_iter=30).model
    pred = model.predict_distribution(frame.X[:1])
    probs = [s.probability for s in pred.top_scorelines]
    assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    frame = _minimal_frame(25)
    result = train_bivariate_poisson(frame, max_iter=30)
    path = tmp_path / "bp.joblib"
    save_bivariate_poisson(path, result)
    loaded = load_bivariate_poisson(path)
    assert isinstance(loaded, BivariatePoissonModel)
    probs_orig = result.model.predict_home_win_prob(frame.X[:2])
    probs_loaded = loaded.predict_home_win_prob(frame.X[:2])
    np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-5)


def test_load_rejects_wrong_model_type(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"model_type": "logistic"}, path)
    with pytest.raises(ValueError, match="bivariate_poisson"):
        load_bivariate_poisson(path)


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------


def test_loader_dispatches_bivariate_poisson(tmp_path: Path) -> None:
    from fantasy_coach.models.loader import load_model

    frame = _minimal_frame(25)
    result = train_bivariate_poisson(frame, max_iter=30)
    path = tmp_path / "bp.joblib"
    save_bivariate_poisson(path, result)
    model = load_model(path)
    assert hasattr(model, "predict_home_win_prob")
    assert hasattr(model, "feature_names")
