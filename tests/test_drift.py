"""Tests for :mod:`fantasy_coach.drift`."""

from __future__ import annotations

import numpy as np
import pytest

from fantasy_coach.drift import (
    PSI_WARN,
    DriftReport,
    RoundLogLoss,
    compute_psi,
    per_feature_psi,
    psi_warnings,
)


def test_psi_identical_distribution_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    assert compute_psi(x, x) == 0.0


def test_psi_same_population_two_samples_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    assert compute_psi(x, y) < 0.1


def test_psi_mean_shift_is_large():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    y = rng.normal(loc=2.0, size=1000)
    assert compute_psi(x, y) > PSI_WARN


def test_psi_binary_flag_major_shift_detected():
    # is_home_field-shaped binary: 80/20 vs 20/80 flip.
    expected = np.concatenate([np.zeros(800), np.ones(200)])
    actual = np.concatenate([np.zeros(200), np.ones(800)])
    assert compute_psi(expected, actual) > 1.0


def test_psi_empty_returns_zero():
    assert compute_psi(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert compute_psi(np.array([1.0]), np.array([])) == 0.0


def test_psi_constant_feature_is_zero():
    # is_home_field is always 1.0 — if train + recent match, PSI = 0.
    x = np.ones(200)
    assert compute_psi(x, x) == 0.0


def test_psi_accepts_small_bin_count():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    # Small bins shouldn't explode PSI for same-population samples.
    assert compute_psi(x, y, bins=3) < 0.2


def test_per_feature_psi_returns_one_entry_per_feature():
    rng = np.random.default_rng(0)
    training = rng.normal(size=(200, 3))
    recent = rng.normal(size=(50, 3))
    result = per_feature_psi(training, recent, ("a", "b", "c"))
    assert set(result) == {"a", "b", "c"}
    for value in result.values():
        assert value < PSI_WARN


def test_per_feature_psi_col_mismatch_raises():
    training = np.zeros((10, 2))
    recent = np.zeros((10, 2))
    with pytest.raises(ValueError, match="feature matrix"):
        per_feature_psi(training, recent, ("only_one",))


def test_per_feature_psi_recent_col_mismatch_raises():
    training = np.zeros((10, 2))
    recent = np.zeros((10, 3))
    with pytest.raises(ValueError, match="recent matrix"):
        per_feature_psi(training, recent, ("a", "b"))


def test_per_feature_psi_flags_drifting_column_only():
    rng = np.random.default_rng(0)
    training = rng.normal(size=(500, 2))
    drifted_col = rng.normal(loc=2.0, size=(500, 1))
    stable_col = rng.normal(size=(500, 1))
    recent = np.hstack([stable_col, drifted_col])
    result = per_feature_psi(training, recent, ("stable", "drifted"))
    assert result["stable"] < 0.1
    assert result["drifted"] > PSI_WARN


def test_psi_warnings_filters_by_threshold_sorted():
    psi = {"c": 0.5, "a": 0.05, "b": 0.3}
    assert psi_warnings(psi) == ["b", "c"]


def test_psi_warnings_custom_threshold():
    psi = {"a": 0.05, "b": 0.12, "c": 0.5}
    assert psi_warnings(psi, threshold=0.1) == ["b", "c"]


def test_psi_warnings_empty_when_below_threshold():
    assert psi_warnings({"a": 0.05, "b": 0.1}) == []


def test_drift_report_roundtrip():
    report = DriftReport(
        season=2026,
        round=5,
        generated_at="2026-04-24T00:00:00+00:00",
        model_version="abc123def456",
        past_round_accuracy=0.57,
        past_round_log_loss=0.68,
        past_round_brier=0.24,
        rolling_log_loss=[
            RoundLogLoss(season=2026, round=3, n=8, log_loss=0.69, accuracy=0.55),
            RoundLogLoss(season=2026, round=4, n=8, log_loss=0.68, accuracy=0.57),
        ],
        feature_psi={"elo_diff": 0.05, "form_diff_pf": 0.31},
        psi_warnings=["form_diff_pf"],
    )
    restored = DriftReport.from_dict(report.to_dict())
    assert restored == report


def test_drift_report_roundtrip_missing_optional_fields():
    minimal = {
        "season": 2026,
        "round": 1,
        "generated_at": "2026-04-24T00:00:00+00:00",
        "model_version": "x" * 12,
    }
    restored = DriftReport.from_dict(minimal)
    assert restored.past_round_accuracy is None
    assert restored.past_round_log_loss is None
    assert restored.past_round_brier is None
    assert restored.rolling_log_loss == []
    assert restored.feature_psi == {}
    assert restored.psi_warnings == []
