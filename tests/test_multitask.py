"""Tests for the multi-task XGBoost model (#215)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.models.multitask import (
    MultiTaskFrame,
    MultiTaskPrediction,
    MultiTaskXGBoostModel,
    build_multitask_frame,
    load_multitask,
    save_multitask,
    train_multitask,
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


def _minimal_frame(n: int = 30) -> MultiTaskFrame:
    matches = [
        # Alternate home wins (even) and away wins (odd) for class balance
        _make_match(
            i,
            home_score=22 if i % 2 == 0 else 14,
            away_score=14 if i % 2 == 0 else 22,
            offset_seconds=i,
        )
        for i in range(n)
    ]
    return build_multitask_frame(matches)


# ---------------------------------------------------------------------------
# build_multitask_frame
# ---------------------------------------------------------------------------


def test_build_multitask_frame_shape() -> None:
    frame = _minimal_frame(30)
    assert frame.X.shape == (30, len(frame.feature_names))
    assert len(frame.y_winner) == 30
    assert len(frame.y_margin) == 30
    assert len(frame.y_total) == 30


def test_build_multitask_frame_excludes_draws() -> None:
    draws = [_make_match(i, 20, 20, offset_seconds=i) for i in range(5)]
    wins = [_make_match(i + 5, 25, 15, offset_seconds=i + 5) for i in range(5)]
    frame = build_multitask_frame(draws + wins)
    assert frame.X.shape[0] == 5  # only non-draws


def test_build_multitask_frame_winner_binary() -> None:
    frame = _minimal_frame(20)
    assert set(frame.y_winner.tolist()).issubset({0.0, 1.0})


def test_build_multitask_frame_margin_equals_score_diff() -> None:
    # home 25, away 10 → margin = 15, total = 35
    matches = [_make_match(1, 25, 10, offset_seconds=1)]
    frame = build_multitask_frame(matches)
    assert frame.y_margin[0] == pytest.approx(15.0)
    assert frame.y_total[0] == pytest.approx(35.0)


def test_build_multitask_frame_empty() -> None:
    frame = build_multitask_frame([])
    assert frame.X.shape[0] == 0


# ---------------------------------------------------------------------------
# train_multitask
# ---------------------------------------------------------------------------


def test_train_multitask_returns_model() -> None:
    frame = _minimal_frame(30)
    result = train_multitask(frame)
    assert isinstance(result.model, MultiTaskXGBoostModel)
    assert result.n_train == 30


def test_train_multitask_raises_too_few_rows() -> None:
    frame = _minimal_frame(5)
    with pytest.raises(ValueError, match="at least 10"):
        train_multitask(frame)


def test_train_multitask_with_custom_params() -> None:
    frame = _minimal_frame(30)
    result = train_multitask(
        frame,
        winner_params={"n_estimators": 10},
        regressor_params={"n_estimators": 10},
    )
    assert isinstance(result.model, MultiTaskXGBoostModel)


# ---------------------------------------------------------------------------
# MultiTaskXGBoostModel
# ---------------------------------------------------------------------------


def test_predict_home_win_prob_shape() -> None:
    frame = _minimal_frame(30)
    model = train_multitask(frame).model
    probs = model.predict_home_win_prob(frame.X[:5])
    assert probs.shape == (5,)
    assert np.all(probs > 0)
    assert np.all(probs < 1)


def test_predict_returns_multitask_predictions() -> None:
    frame = _minimal_frame(30)
    model = train_multitask(frame).model
    preds = model.predict(frame.X[:3])
    assert len(preds) == 3
    assert all(isinstance(p, MultiTaskPrediction) for p in preds)


def test_predict_total_positive() -> None:
    frame = _minimal_frame(30)
    model = train_multitask(frame).model
    preds = model.predict(frame.X)
    assert all(p.predicted_total >= 0 for p in preds)


def test_coherence_winner_margin_agreement() -> None:
    """Winner side must agree with margin sign on > 99% of predictions."""
    frame = _minimal_frame(30)
    model = train_multitask(frame).model
    coh = model.coherence_fraction(frame.X)
    assert coh > 0.99, f"coherence fraction = {coh:.3f} < 0.99"


def test_stronger_home_higher_win_prob() -> None:
    """Matches where home wins big → home win probability > 0.5."""
    # Mix: 25 dominant home wins + 5 away wins so XGB has both classes
    matches = [_make_match(i, 40, 5, home_id=1, away_id=2, offset_seconds=i) for i in range(25)]
    matches += [_make_match(i + 25, 5, 40, offset_seconds=i + 25) for i in range(5)]
    frame = build_multitask_frame(matches)
    model = train_multitask(frame).model
    prob = model.predict_home_win_prob(frame.X[:1])[0]
    assert prob > 0.5


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    frame = _minimal_frame(30)
    result = train_multitask(frame)
    path = tmp_path / "mt.joblib"
    save_multitask(path, result)
    loaded = load_multitask(path)
    assert isinstance(loaded, MultiTaskXGBoostModel)
    probs_orig = result.model.predict_home_win_prob(frame.X[:3])
    probs_loaded = loaded.predict_home_win_prob(frame.X[:3])
    np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-5)


def test_load_rejects_wrong_model_type(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"model_type": "skellam"}, path)
    with pytest.raises(ValueError, match="multitask"):
        load_multitask(path)


def test_loader_dispatches_multitask(tmp_path: Path) -> None:
    from fantasy_coach.models.loader import load_model

    frame = _minimal_frame(30)
    result = train_multitask(frame)
    path = tmp_path / "mt.joblib"
    save_multitask(path, result)
    model = load_model(path)
    assert hasattr(model, "predict_home_win_prob")
    assert hasattr(model, "feature_names")
