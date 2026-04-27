"""Tests for the OOD detector (#216)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fantasy_coach.models.ood import (
    MahalanobisScorer,
    OODDetector,
    OODResult,
    _percentile_to_flag,
    load_ood_detector,
    save_ood_detector,
)


def _training_data(n: int = 300, n_feat: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, n_feat))


def _ood_data(n: int = 10, n_feat: int = 10, seed: int = 99) -> np.ndarray:
    """Clearly out-of-distribution: drawn from a very different distribution."""
    rng = np.random.default_rng(seed)
    return rng.normal(10, 1, (n, n_feat))  # far from N(0,1) training


# ---------------------------------------------------------------------------
# _percentile_to_flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pct,expected", [
    (0.0, "in_distribution"),
    (50.0, "in_distribution"),
    (89.9, "in_distribution"),
    (90.0, "edge"),
    (95.0, "edge"),
    (98.9, "edge"),
    (99.0, "out_of_distribution"),
    (99.9, "out_of_distribution"),
    (100.0, "out_of_distribution"),
])
def test_percentile_to_flag(pct: float, expected: str) -> None:
    assert _percentile_to_flag(pct) == expected


# ---------------------------------------------------------------------------
# OODDetector
# ---------------------------------------------------------------------------


def test_ood_detector_raises_before_fit() -> None:
    detector = OODDetector()
    with pytest.raises(RuntimeError, match="fit"):
        detector.score(_training_data(1))


def test_ood_detector_fit_does_not_raise() -> None:
    X = _training_data()
    detector = OODDetector()
    detector.fit(X)


def test_ood_detector_score_returns_ood_result() -> None:
    X = _training_data()
    detector = OODDetector()
    detector.fit(X)
    result = detector.score(X[:1])
    assert isinstance(result, OODResult)
    assert 0.0 <= result.percentile <= 100.0
    assert result.flag in {"in_distribution", "edge", "out_of_distribution"}
    assert np.isfinite(result.raw_score)


def test_ood_detector_in_distribution_has_lower_percentile() -> None:
    """In-distribution samples should score lower than clearly OOD samples."""
    X_train = _training_data(300)
    X_ood = _ood_data(50)

    detector = OODDetector()
    detector.fit(X_train)

    in_results = detector.score_batch(X_train[:50])
    ood_results = detector.score_batch(X_ood)

    avg_in_pct = np.mean([r.percentile for r in in_results])
    avg_ood_pct = np.mean([r.percentile for r in ood_results])

    assert avg_ood_pct > avg_in_pct, (
        f"OOD percentile {avg_ood_pct:.1f} should exceed in-dist {avg_in_pct:.1f}"
    )


def test_ood_detector_ood_samples_flagged_mostly_out() -> None:
    """Clearly OOD samples should mostly be flagged 'edge' or 'out_of_distribution'."""
    X_train = _training_data(300)
    X_ood = _ood_data(50)

    detector = OODDetector()
    detector.fit(X_train)
    ood_results = detector.score_batch(X_ood)

    flagged = sum(r.flag != "in_distribution" for r in ood_results)
    assert flagged / len(ood_results) > 0.8, (
        f"Expected > 80% of OOD samples to be flagged, got {flagged}/{len(ood_results)}"
    )


def test_ood_detector_batch_equals_individual() -> None:
    X = _training_data()
    detector = OODDetector()
    detector.fit(X)
    batch = detector.score_batch(X[:5])
    for i, result_b in enumerate(batch):
        result_i = detector.score(X[i : i + 1])
        assert result_b.percentile == pytest.approx(result_i.percentile, abs=1e-6)


def test_ood_detector_percentile_range() -> None:
    X = _training_data(200)
    detector = OODDetector()
    detector.fit(X)
    results = detector.score_batch(X)
    pcts = [r.percentile for r in results]
    assert min(pcts) >= 0.0
    assert max(pcts) <= 100.0


def test_ood_detector_flags_valid() -> None:
    X = _training_data(200)
    detector = OODDetector()
    detector.fit(X)
    results = detector.score_batch(X)
    valid_flags = {"in_distribution", "edge", "out_of_distribution"}
    for r in results:
        assert r.flag in valid_flags


# ---------------------------------------------------------------------------
# MahalanobisScorer
# ---------------------------------------------------------------------------


def test_mahalanobis_scorer_in_dist_higher_than_ood() -> None:
    X_train = _training_data(300)
    X_ood = _ood_data(50)
    scorer = MahalanobisScorer()
    scorer.fit(X_train)
    in_scores = scorer.score_samples(X_train[:50])
    ood_scores = scorer.score_samples(X_ood)
    # Lower (more negative) score = more anomalous, so OOD should be more negative
    assert np.mean(ood_scores) < np.mean(in_scores)


def test_mahalanobis_scorer_raises_before_fit() -> None:
    scorer = MahalanobisScorer()
    with pytest.raises(RuntimeError, match="fit"):
        scorer.score_samples(_training_data(5))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    X = _training_data(200)
    detector = OODDetector()
    detector.fit(X)
    path = tmp_path / "ood.joblib"
    save_ood_detector(path, detector)
    loaded = load_ood_detector(path)
    assert isinstance(loaded, OODDetector)

    r_orig = detector.score(X[:1])
    r_loaded = loaded.score(X[:1])
    assert r_orig.percentile == pytest.approx(r_loaded.percentile)


def test_load_rejects_wrong_type(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"not": "an ood detector"}, path)
    with pytest.raises(ValueError, match="OODDetector"):
        load_ood_detector(path)
