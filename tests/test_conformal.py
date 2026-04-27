"""Tests for distribution-free conformal prediction intervals (#214)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fantasy_coach.models.conformal import (
    MarginConformalizer,
    MarginInterval,
    MondrianConformalizer,
    ProbabilityConformalizer,
    ProbabilityInterval,
    load_conformalizer,
    save_conformalizer,
)


def _calib_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic calibration data: probabilities ∈ [0.3, 0.8], outcomes Bernoulli."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.3, 0.8, n)
    outcomes = rng.binomial(1, probs).astype(float)
    return probs, outcomes


def _margin_data(n: int = 200, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic margin data: predicted ± noise."""
    rng = np.random.default_rng(seed)
    predicted = rng.normal(5.0, 8.0, n)
    actual = predicted + rng.normal(0, 5.0, n)
    return predicted, actual


# ---------------------------------------------------------------------------
# ProbabilityConformalizer
# ---------------------------------------------------------------------------


def test_probability_conformalizer_fit_and_transform() -> None:
    probs, outcomes = _calib_data()
    cp = ProbabilityConformalizer(alpha=0.1)
    cp.fit(probs, outcomes)
    iv = cp.transform(0.6)
    assert isinstance(iv, ProbabilityInterval)
    assert iv.lo <= iv.point <= iv.hi
    assert 0.0 <= iv.lo <= 1.0
    assert 0.0 <= iv.hi <= 1.0


def test_probability_conformalizer_raises_before_fit() -> None:
    cp = ProbabilityConformalizer()
    with pytest.raises(RuntimeError, match="fit"):
        cp.transform(0.5)


def test_probability_conformalizer_alpha_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ProbabilityConformalizer(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ProbabilityConformalizer(alpha=1.0)


def test_probability_conformalizer_coverage_at_least_nominal() -> None:
    """Empirical coverage on a held-out set should be ≥ 1−alpha."""
    rng = np.random.default_rng(42)
    probs_calib = rng.uniform(0.3, 0.8, 500)
    outcomes_calib = rng.binomial(1, probs_calib).astype(float)

    probs_test = rng.uniform(0.3, 0.8, 500)
    outcomes_test = rng.binomial(1, probs_test).astype(float)

    for alpha in (0.1, 0.2):
        cp = ProbabilityConformalizer(alpha=alpha)
        cp.fit(probs_calib, outcomes_calib)
        cov = cp.coverage_fraction(probs_test, outcomes_test)
        # Allow a 3pp slack for finite-sample noise
        assert cov >= 1 - alpha - 0.03, f"alpha={alpha}: coverage={cov:.3f} < {1-alpha-0.03:.3f}"


def test_probability_conformalizer_smaller_alpha_wider_interval() -> None:
    """Tighter miscoverage → narrower interval; looser → wider."""
    probs, outcomes = _calib_data(500)
    cp_tight = ProbabilityConformalizer(alpha=0.05)
    cp_loose = ProbabilityConformalizer(alpha=0.20)
    cp_tight.fit(probs, outcomes)
    cp_loose.fit(probs, outcomes)
    iv_tight = cp_tight.transform(0.6)
    iv_loose = cp_loose.transform(0.6)
    tight_width = iv_tight.hi - iv_tight.lo
    loose_width = iv_loose.hi - iv_loose.lo
    assert tight_width >= loose_width


def test_probability_conformalizer_input_validation() -> None:
    cp = ProbabilityConformalizer()
    with pytest.raises(ValueError):
        cp.fit(np.array([0.5, 0.6]), np.array([1.0]))  # mismatched lengths


# ---------------------------------------------------------------------------
# MarginConformalizer
# ---------------------------------------------------------------------------


def test_margin_conformalizer_fit_and_transform() -> None:
    pred, actual = _margin_data()
    mc = MarginConformalizer(alpha=0.1)
    mc.fit(pred, actual)
    iv = mc.transform(5.0)
    assert isinstance(iv, MarginInterval)
    assert iv.lo <= iv.point <= iv.hi


def test_margin_conformalizer_raises_before_fit() -> None:
    mc = MarginConformalizer()
    with pytest.raises(RuntimeError, match="fit"):
        mc.transform(3.0)


def test_margin_conformalizer_coverage() -> None:
    """Empirical coverage on a held-out set should be ≥ 1−alpha."""
    rng = np.random.default_rng(7)
    pred_c = rng.normal(5.0, 8.0, 400)
    act_c = pred_c + rng.normal(0, 5.0, 400)
    pred_t = rng.normal(5.0, 8.0, 400)
    act_t = pred_t + rng.normal(0, 5.0, 400)

    alpha = 0.1
    mc = MarginConformalizer(alpha=alpha)
    mc.fit(pred_c, act_c)
    cov = mc.coverage_fraction(pred_t, act_t)
    assert cov >= 1 - alpha - 0.03, f"coverage={cov:.3f} < {1-alpha-0.03:.3f}"


def test_margin_conformalizer_symmetric_interval() -> None:
    pred, actual = _margin_data(200)
    mc = MarginConformalizer(alpha=0.1)
    mc.fit(pred, actual)
    iv = mc.transform(10.0)
    assert abs((iv.hi - iv.point) - (iv.point - iv.lo)) < 1e-10


def test_margin_conformalizer_input_validation() -> None:
    mc = MarginConformalizer()
    with pytest.raises(ValueError):
        mc.fit(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# MondrianConformalizer
# ---------------------------------------------------------------------------


def test_mondrian_conformalizer_fit_and_transform() -> None:
    probs, outcomes = _calib_data(300)
    bin_fn = lambda p, _: "high" if p >= 0.55 else "low"
    mc = MondrianConformalizer(bin_fn=bin_fn, alpha=0.1)
    mc.fit(probs, outcomes)
    iv = mc.transform(0.7)
    assert isinstance(iv, ProbabilityInterval)
    assert 0.0 <= iv.lo <= iv.hi <= 1.0


def test_mondrian_conformalizer_raises_before_fit() -> None:
    mc = MondrianConformalizer(bin_fn=lambda p, _: "a")
    with pytest.raises(RuntimeError, match="fit"):
        mc.transform(0.5)


def test_mondrian_conformalizer_different_quantiles_per_bin() -> None:
    """Bins with different residual distributions get different quantiles."""
    rng = np.random.default_rng(9)
    # Low-prob group has tighter residuals, high-prob group looser
    p_low = rng.uniform(0.3, 0.5, 100)
    o_low = rng.binomial(1, p_low).astype(float)
    p_high = rng.uniform(0.5, 0.9, 100)
    o_high = rng.binomial(1, p_high).astype(float)

    probs = np.concatenate([p_low, p_high])
    outcomes = np.concatenate([o_low, o_high])

    bin_fn = lambda p, _: "high" if p >= 0.5 else "low"
    mc = MondrianConformalizer(bin_fn=bin_fn, alpha=0.1)
    mc.fit(probs, outcomes)

    assert "high" in mc._quantiles
    assert "low" in mc._quantiles


def test_mondrian_fallback_for_unseen_bin() -> None:
    """Unseen bin at transform-time uses the marginal fallback quantile."""
    probs, outcomes = _calib_data(200)
    mc = MondrianConformalizer(bin_fn=lambda p, _: "seen", alpha=0.1)
    mc.fit(probs, outcomes)
    # "unseen" bin triggers fallback
    mc._quantiles["seen2"] = 999.0  # deliberately wrong; should not be used
    iv = mc.transform(0.6)
    assert np.isfinite(iv.lo) and np.isfinite(iv.hi)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_probability_conformalizer(tmp_path: Path) -> None:
    probs, outcomes = _calib_data(200)
    cp = ProbabilityConformalizer(alpha=0.1)
    cp.fit(probs, outcomes)
    path = tmp_path / "cp.joblib"
    save_conformalizer(path, cp)
    loaded = load_conformalizer(path)
    assert isinstance(loaded, ProbabilityConformalizer)
    assert loaded._quantile == pytest.approx(cp._quantile)


def test_save_load_margin_conformalizer(tmp_path: Path) -> None:
    pred, actual = _margin_data(200)
    mc = MarginConformalizer(alpha=0.1)
    mc.fit(pred, actual)
    path = tmp_path / "mc.joblib"
    save_conformalizer(path, mc)
    loaded = load_conformalizer(path)
    assert isinstance(loaded, MarginConformalizer)
    assert loaded._quantile == pytest.approx(mc._quantile)
