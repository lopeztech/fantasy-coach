"""Tests for the weighted / stacked ensemble and its kill switch."""

from __future__ import annotations

import numpy as np
import pytest

from fantasy_coach.models.ensemble import (
    DEFAULT_MIN_IMPROVEMENT,
    EnsembleModel,
    LoadedEnsemble,
    _log_loss,
    fit_ensemble,
    save_ensemble,
)


def _two_informative_bases(rng: np.random.Generator, n: int = 400):
    """Two base predictors that each carry real but imperfect signal.

    Mixing them ought to beat either alone.
    """
    y = rng.integers(0, 2, size=n)
    # Base A is slightly biased toward correct predictions but noisy.
    a = np.where(y == 1, rng.beta(4, 2, n), rng.beta(2, 4, n))
    # Base B is also informative but with a different noise pattern.
    b = np.where(y == 1, rng.beta(3, 2, n), rng.beta(2, 3, n))
    return np.column_stack([a, b]), y


def _uninformative_bases(rng: np.random.Generator, n: int = 400):
    """One informative base, one pure noise — kill switch should fire.

    The best single base is already the informative one; the ensemble can
    at best tie it, not meaningfully beat it, so we expect ``fallback_to_base``.
    """
    y = rng.integers(0, 2, size=n)
    a = np.where(y == 1, rng.beta(4, 2, n), rng.beta(2, 4, n))
    noise = rng.beta(2, 2, n)  # pure noise centred at 0.5
    return np.column_stack([a, noise]), y


# ---------------------------------------------------------------------------
# fit_ensemble — happy paths
# ---------------------------------------------------------------------------


def test_weighted_beats_best_base_on_informative_bases():
    rng = np.random.default_rng(0)
    base_probs, y = _two_informative_bases(rng)
    model = fit_ensemble(
        base_probs, y, mode="weighted", base_model_names=("a", "b"), min_improvement=0.0
    )
    # Weights are on the simplex.
    assert model.weights is not None
    assert model.weights.shape == (2,)
    assert np.all(model.weights >= 0.0)
    assert model.weights.sum() == pytest.approx(1.0)
    # Ensemble log loss is at most the best base — can equal if one weight is 1.0.
    assert model.ensemble_log_loss <= model.base_log_losses[model.best_base_name] + 1e-9


def test_stacked_fits_a_meta_learner():
    rng = np.random.default_rng(1)
    base_probs, y = _two_informative_bases(rng)
    model = fit_ensemble(
        base_probs, y, mode="stacked", base_model_names=("a", "b"), min_improvement=0.0
    )
    assert model.meta_learner is not None
    assert model.meta_learner.coef_.shape == (1, 2)


def test_predict_home_win_prob_shapes_match():
    rng = np.random.default_rng(2)
    base_probs, y = _two_informative_bases(rng)
    model = fit_ensemble(
        base_probs, y, mode="weighted", base_model_names=("a", "b"), min_improvement=0.0
    )
    probs = model.predict_home_win_prob(base_probs)
    assert probs.shape == (base_probs.shape[0],)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_fires_when_ensemble_cannot_beat_best_base():
    rng = np.random.default_rng(3)
    base_probs, y = _uninformative_bases(rng)
    # Pin the threshold high so even marginal wins trigger the fallback —
    # the noise base can't help the informative one enough to clear 0.05.
    model = fit_ensemble(
        base_probs,
        y,
        mode="weighted",
        base_model_names=("informative", "noise"),
        min_improvement=0.05,
    )
    assert model.use_fallback
    assert model.fallback_to_base == "informative"


def test_fallback_prediction_uses_named_base_column():
    rng = np.random.default_rng(4)
    base_probs, y = _uninformative_bases(rng)
    model = fit_ensemble(
        base_probs,
        y,
        mode="stacked",
        base_model_names=("informative", "noise"),
        min_improvement=0.5,  # impossibly strict → fallback always
    )
    assert model.use_fallback
    assert model.fallback_to_base == "informative"
    predicted = model.predict_home_win_prob(base_probs)
    # Fallback path must return the raw informative column, not a reshaped
    # version — single-column pass-through is load-bearing for the eval
    # harness.
    np.testing.assert_array_equal(predicted, base_probs[:, 0])


def test_default_min_improvement_fires_on_truly_equivalent_bases():
    rng = np.random.default_rng(5)
    n = 500
    y = rng.integers(0, 2, size=n)
    # Two identical bases — ensemble has nothing to learn.
    col = np.where(y == 1, rng.beta(4, 2, n), rng.beta(2, 4, n))
    base_probs = np.column_stack([col, col])
    model = fit_ensemble(base_probs, y, mode="weighted", base_model_names=("a", "b"))
    # Both bases tie exactly → ensemble cannot improve → fallback set.
    assert model.use_fallback
    # With the default 0.005 threshold, an identical-base ensemble has zero
    # improvement and should absolutely route to a base.
    improvement = model.base_log_losses[model.best_base_name] - model.ensemble_log_loss
    assert improvement < DEFAULT_MIN_IMPROVEMENT


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_mismatched_column_count_raises():
    rng = np.random.default_rng(6)
    base_probs, y = _two_informative_bases(rng)
    with pytest.raises(ValueError, match="columns"):
        fit_ensemble(base_probs, y, mode="weighted", base_model_names=("a", "b", "c"))


def test_predict_rejects_wrong_column_count():
    rng = np.random.default_rng(7)
    base_probs, y = _two_informative_bases(rng)
    model = fit_ensemble(
        base_probs, y, mode="weighted", base_model_names=("a", "b"), min_improvement=0.0
    )
    with pytest.raises(ValueError, match="base columns"):
        model.predict_home_win_prob(base_probs[:, :1])


def test_mismatched_row_count_raises():
    rng = np.random.default_rng(8)
    base_probs, _ = _two_informative_bases(rng)
    with pytest.raises(ValueError, match="same number of rows"):
        fit_ensemble(base_probs, np.zeros(3), mode="weighted", base_model_names=("a", "b"))


# ---------------------------------------------------------------------------
# Dataclass plumbing
# ---------------------------------------------------------------------------


def test_ensemble_model_dataclass_defaults_are_sane():
    model = EnsembleModel(mode="weighted", base_model_names=("a", "b"))
    assert model.weights is None
    assert model.meta_learner is None
    assert model.base_log_losses == {}
    assert not model.use_fallback
    assert np.isnan(model.ensemble_log_loss)


def test_log_loss_matches_manual_calculation():
    y = np.array([1, 0, 1, 0], dtype=float)
    p = np.array([0.9, 0.2, 0.6, 0.3], dtype=float)
    expected = -(np.log(0.9) + np.log(0.8) + np.log(0.6) + np.log(0.7)) / 4
    assert _log_loss(p, y) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# EnsemblePredictor end-to-end (adapter wiring, not a math test)
# ---------------------------------------------------------------------------


def _synthetic_history(n_matches: int, rng: np.random.Generator):
    """Build a minimal MatchRow history the predictor adapter can fit.

    We avoid the full feature pipeline dependencies (weather, venue, …) by
    combining HomePickPredictor (no history needed) and EloPredictor (needs
    only scores) — enough to exercise the ensemble's fit/predict paths.
    """
    from datetime import UTC, datetime, timedelta

    from fantasy_coach.features import MatchRow, TeamRow

    base = datetime(2024, 3, 1, tzinfo=UTC)
    matches = []
    for i in range(n_matches):
        home_id = 10 + (i % 8) * 2
        away_id = 11 + (i % 8) * 2
        home_score = int(rng.integers(12, 32))
        away_score = int(rng.integers(6, 28))
        matches.append(
            MatchRow(
                match_id=i,
                season=2024,
                round=1 + i // 8,
                start_time=base + timedelta(days=i),
                match_state="FullTime",
                venue=None,
                venue_city=None,
                weather=None,
                referee_id=None,
                home=TeamRow(
                    team_id=home_id,
                    name=str(home_id),
                    nick_name=str(home_id),
                    score=home_score,
                    players=[],
                ),
                away=TeamRow(
                    team_id=away_id,
                    name=str(away_id),
                    nick_name=str(away_id),
                    score=away_score,
                    players=[],
                ),
                team_stats=[],
            )
        )
    return matches


def test_ensemble_predictor_fit_and_predict_end_to_end():
    from fantasy_coach.evaluation.predictors import (
        EloPredictor,
        EnsemblePredictor,
        HomePickPredictor,
    )

    rng = np.random.default_rng(42)
    history = _synthetic_history(60, rng)

    ensemble = EnsemblePredictor(
        base_factories=[HomePickPredictor, EloPredictor],
        mode="weighted",
    )
    ensemble.fit(history)

    # Fit info should record something coherent — either the ensemble
    # trained (disabled=False) or the history was too small (unlikely at n=60).
    assert "disabled" in ensemble.last_fit_info
    assert ensemble.last_fit_info["disabled"] is False
    assert "base_log_losses" in ensemble.last_fit_info

    # Predicting a future-ish match should yield a probability in [0, 1].
    future = _synthetic_history(1, np.random.default_rng(7))[0]
    p = ensemble.predict_home_win_prob(future)
    assert 0.0 <= p <= 1.0


def test_ensemble_predictor_disabled_when_history_too_small():
    from fantasy_coach.evaluation.predictors import (
        EloPredictor,
        EnsemblePredictor,
        HomePickPredictor,
    )

    rng = np.random.default_rng(0)
    history = _synthetic_history(5, rng)

    ensemble = EnsemblePredictor(
        base_factories=[HomePickPredictor, EloPredictor],
        mode="weighted",
        min_meta_rows=30,
    )
    ensemble.fit(history)

    assert ensemble.last_fit_info["disabled"] is True
    # Degraded path still produces a [0,1] probability (falls through to the
    # first base predictor — HomePick's constant 0.55).
    future = _synthetic_history(1, np.random.default_rng(1))[0]
    p = ensemble.predict_home_win_prob(future)
    assert p == pytest.approx(0.55)


def test_ensemble_predictor_empty_bases_rejects():
    from fantasy_coach.evaluation.predictors import EnsemblePredictor

    with pytest.raises(ValueError, match="at least one"):
        EnsemblePredictor(base_factories=[], mode="weighted")


# ---------------------------------------------------------------------------
# Ensemble artifact save/load round-trip (for the prediction API, #84)
# ---------------------------------------------------------------------------


def _two_logistic_blobs(rng: np.random.Generator):
    """Produce two logistic-blob dicts over the live FEATURE_NAMES.

    Both bases are trained on the same synthetic ``FEATURE_NAMES``-shaped frame with
    different random seeds so they disagree — enough for the ensemble's
    weighted combination to differ from either base alone.
    """
    from fantasy_coach.feature_engineering import FEATURE_NAMES, TrainingFrame
    from fantasy_coach.models.logistic import train_logistic

    n = 200
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(int)
    frame = TrainingFrame(
        X=X,
        y=y,
        match_ids=np.arange(n),
        start_times=np.arange(n, dtype=float),
        feature_names=FEATURE_NAMES,
    )
    blobs = []
    for seed in (0, 7):
        result = train_logistic(frame, test_fraction=0.0, random_state=seed)
        blobs.append(
            {
                "model_type": "logistic",
                "pipeline": result.pipeline,
                "feature_names": result.feature_names,
            }
        )
    return blobs


def test_save_ensemble_round_trip_returns_loaded_ensemble(tmp_path):
    from fantasy_coach.feature_engineering import FEATURE_NAMES
    from fantasy_coach.models.loader import Model, load_model

    rng = np.random.default_rng(0)
    base_blobs = _two_logistic_blobs(rng)
    ensemble = EnsembleModel(
        mode="weighted",
        base_model_names=("a", "b"),
        weights=np.array([0.6, 0.4]),
    )

    path = tmp_path / "ensemble.joblib"
    save_ensemble(path, ensemble=ensemble, base_blobs=base_blobs)

    loaded = load_model(path)
    assert isinstance(loaded, LoadedEnsemble)
    assert isinstance(loaded, Model)
    assert loaded.feature_names == FEATURE_NAMES
    assert len(loaded.base_models) == 2


def test_loaded_ensemble_predict_matches_weighted_combination(tmp_path):
    from fantasy_coach.feature_engineering import FEATURE_NAMES
    from fantasy_coach.models.loader import load_model

    rng = np.random.default_rng(1)
    base_blobs = _two_logistic_blobs(rng)
    weights = np.array([0.7, 0.3])
    ensemble = EnsembleModel(
        mode="weighted",
        base_model_names=("a", "b"),
        weights=weights,
    )
    path = tmp_path / "ensemble.joblib"
    save_ensemble(path, ensemble=ensemble, base_blobs=base_blobs)

    loaded = load_model(path)

    X = rng.standard_normal((5, len(FEATURE_NAMES)))
    probs = loaded.predict_home_win_prob(X)

    # Hand-compute expected by pulling each base's probability directly.
    base_a = base_blobs[0]["pipeline"].predict_proba(X)[:, 1]
    base_b = base_blobs[1]["pipeline"].predict_proba(X)[:, 1]
    expected = weights[0] * base_a + weights[1] * base_b

    np.testing.assert_allclose(probs, expected, rtol=1e-9)


def test_loaded_ensemble_respects_kill_switch_fallback(tmp_path):
    from fantasy_coach.feature_engineering import FEATURE_NAMES
    from fantasy_coach.models.loader import load_model

    rng = np.random.default_rng(2)
    base_blobs = _two_logistic_blobs(rng)
    # Kill switch set: predictions should come from base "a" unchanged.
    ensemble = EnsembleModel(
        mode="weighted",
        base_model_names=("a", "b"),
        weights=np.array([0.5, 0.5]),
        fallback_to_base="a",
    )
    path = tmp_path / "ensemble.joblib"
    save_ensemble(path, ensemble=ensemble, base_blobs=base_blobs)

    loaded = load_model(path)
    X = rng.standard_normal((3, len(FEATURE_NAMES)))
    probs = loaded.predict_home_win_prob(X)

    base_a = base_blobs[0]["pipeline"].predict_proba(X)[:, 1]
    np.testing.assert_allclose(probs, base_a, rtol=1e-9)


def test_save_ensemble_rejects_base_count_mismatch(tmp_path):
    rng = np.random.default_rng(3)
    base_blobs = _two_logistic_blobs(rng)
    ensemble = EnsembleModel(
        mode="weighted",
        base_model_names=("a", "b", "c"),
        weights=np.array([0.4, 0.3, 0.3]),
    )
    with pytest.raises(ValueError, match="base blobs"):
        save_ensemble(tmp_path / "ens.joblib", ensemble=ensemble, base_blobs=base_blobs)


def test_loaded_ensemble_rejects_wrong_feature_count(tmp_path):
    from fantasy_coach.models.loader import load_model

    rng = np.random.default_rng(4)
    base_blobs = _two_logistic_blobs(rng)
    ensemble = EnsembleModel(
        mode="weighted",
        base_model_names=("a", "b"),
        weights=np.array([0.5, 0.5]),
    )
    path = tmp_path / "ensemble.joblib"
    save_ensemble(path, ensemble=ensemble, base_blobs=base_blobs)

    loaded = load_model(path)
    with pytest.raises(ValueError, match="features"):
        loaded.predict_home_win_prob(np.zeros((1, 3)))
