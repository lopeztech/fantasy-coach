"""Tests for the ensemble / stacking module."""

from __future__ import annotations

import numpy as np
import pytest

from fantasy_coach.models.ensemble import (
    EnsembleResult,
    fit_ensemble,
    predict_ensemble,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probs_and_labels(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, y) with alternating classes so every CV fold has both."""
    rng = np.random.default_rng(seed)
    p = np.column_stack(
        [
            rng.uniform(0.3, 0.7, n),
            rng.uniform(0.3, 0.7, n),
            rng.uniform(0.3, 0.7, n),
        ]
    )
    # alternate to ensure both classes in every TimeSeriesSplit fold
    y = np.array([i % 2 for i in range(n)], dtype=int)
    return p, y


# ---------------------------------------------------------------------------
# fit_ensemble
# ---------------------------------------------------------------------------


def test_fit_ensemble_returns_result_stacked() -> None:
    p, y = _probs_and_labels()
    result = fit_ensemble(p, y, ["elo", "logistic", "xgboost"], mode="stacked")
    assert isinstance(result, EnsembleResult)
    assert result.mode == "stacked"
    assert result.base_names == ("elo", "logistic", "xgboost")
    assert result.weights.shape == (3,)


def test_fit_ensemble_returns_result_weighted() -> None:
    p, y = _probs_and_labels()
    result = fit_ensemble(p, y, ["elo", "logistic", "xgboost"], mode="weighted")
    assert result.mode == "weighted"
    # Weights must lie on the unit simplex.
    assert np.all(result.weights >= -1e-9)
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert result.intercept == pytest.approx(0.0)


def test_fit_ensemble_best_base_idx_valid() -> None:
    p, y = _probs_and_labels()
    result = fit_ensemble(p, y, ["elo", "logistic", "xgboost"])
    assert 0 <= result.best_base_idx < 3


def test_fit_ensemble_kill_switch_triggers_on_random_probs() -> None:
    # When all base models output random probabilities, the ensemble cannot
    # beat any single model by 0.5 pp; the kill switch should fire.
    rng = np.random.default_rng(0)
    p = rng.uniform(0.4, 0.6, (100, 3))
    y = rng.integers(0, 2, 100)
    result = fit_ensemble(p, y, ["a", "b", "c"], kill_switch_threshold=0.5)
    assert result.kill_switch is True


def test_fit_ensemble_kill_switch_off_with_strongly_negative_threshold() -> None:
    # With a sufficiently negative threshold the kill switch is always disabled.
    p, y = _probs_and_labels()
    result = fit_ensemble(p, y, ["a", "b", "c"], kill_switch_threshold=-1.0)
    assert result.kill_switch is False


def test_fit_ensemble_base_names_length_mismatch_raises() -> None:
    p, y = _probs_and_labels(40)
    with pytest.raises(AssertionError):
        fit_ensemble(p, y, ["only_two", "names"])


# ---------------------------------------------------------------------------
# predict_ensemble
# ---------------------------------------------------------------------------


def test_predict_ensemble_stacked_shape() -> None:
    p, y = _probs_and_labels()
    result = fit_ensemble(p, y, ["a", "b", "c"], mode="stacked")
    p_new = np.random.default_rng(1).uniform(0.3, 0.7, (10, 3))
    out = predict_ensemble(p_new, result)
    assert out.shape == (10,)


def test_predict_ensemble_stacked_bounds() -> None:
    p, y = _probs_and_labels()
    result = fit_ensemble(p, y, ["a", "b", "c"], mode="stacked")
    p_new = np.random.default_rng(1).uniform(0.0, 1.0, (50, 3))
    out = predict_ensemble(p_new, result)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_predict_ensemble_weighted_shape() -> None:
    p, y = _probs_and_labels()
    result = fit_ensemble(p, y, ["a", "b", "c"], mode="weighted")
    p_new = np.random.default_rng(1).uniform(0.3, 0.7, (10, 3))
    out = predict_ensemble(p_new, result)
    assert out.shape == (10,)


def test_predict_ensemble_kill_switch_returns_best_base() -> None:
    # Construct a result with kill_switch=True and best_base_idx=1.
    result = EnsembleResult(
        mode="stacked",
        base_names=("a", "b", "c"),
        weights=np.array([1.0, 0.0, 0.0]),
        intercept=0.0,
        kill_switch=True,
        best_base_idx=1,
    )
    p = np.array([[0.3, 0.7, 0.5]])
    out = predict_ensemble(p, result)
    assert out[0] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# EnsemblePredictor (integration via predictors.py)
# ---------------------------------------------------------------------------


def test_ensemble_predictor_fits_and_predicts() -> None:
    from datetime import UTC, datetime, timedelta

    from fantasy_coach.evaluation.predictors import EnsemblePredictor
    from fantasy_coach.features import MatchRow, TeamRow

    def _team(tid: int, score: int | None = None) -> TeamRow:
        return TeamRow(team_id=tid, name=f"T{tid}", nick_name=f"T{tid}", score=score, players=[])

    def _match(mid: int, offset_days: int, hscore: int, ascore: int) -> MatchRow:
        return MatchRow(
            match_id=mid,
            season=2025,
            round=1,
            start_time=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=offset_days),
            match_state="FullTime",
            venue="V",
            venue_city="C",
            weather=None,
            home=_team(1, hscore),
            away=_team(2, ascore),
            team_stats=[],
        )

    history = [
        _match(i, i * 7, 22 if i % 2 == 0 else 16, 16 if i % 2 == 0 else 22) for i in range(60)
    ]
    predictor = EnsemblePredictor()
    predictor.fit(history)

    upcoming = MatchRow(
        match_id=999,
        season=2025,
        round=20,
        start_time=datetime(2025, 8, 1, tzinfo=UTC),
        match_state="Upcoming",
        venue="V",
        venue_city="C",
        weather=None,
        home=_team(1),
        away=_team(2),
        team_stats=[],
    )
    p = predictor.predict_home_win_prob(upcoming)
    assert 0.0 <= p <= 1.0


def test_ensemble_predictor_fallback_with_no_history() -> None:
    from datetime import UTC, datetime

    from fantasy_coach.evaluation.predictors import EnsemblePredictor
    from fantasy_coach.features import MatchRow, TeamRow

    predictor = EnsemblePredictor()
    predictor.fit([])
    match = MatchRow(
        match_id=1,
        season=2025,
        round=1,
        start_time=datetime(2025, 1, 1, tzinfo=UTC),
        match_state="Upcoming",
        venue="V",
        venue_city="C",
        weather=None,
        home=TeamRow(team_id=1, name="H", nick_name="H", score=None, players=[]),
        away=TeamRow(team_id=2, name="A", nick_name="A", score=None, players=[]),
        team_stats=[],
    )
    p = predictor.predict_home_win_prob(match)
    assert 0.0 <= p <= 1.0


def test_ensemble_predictor_name() -> None:
    from fantasy_coach.evaluation.predictors import EnsemblePredictor

    assert EnsemblePredictor().name == "ensemble"


def test_ensemble_predictor_weighted_mode() -> None:
    from datetime import UTC, datetime, timedelta

    from fantasy_coach.evaluation.predictors import EnsemblePredictor
    from fantasy_coach.features import MatchRow, TeamRow

    def _team(tid: int, score: int | None = None) -> TeamRow:
        return TeamRow(team_id=tid, name=f"T{tid}", nick_name=f"T{tid}", score=score, players=[])

    history = [
        MatchRow(
            match_id=i,
            season=2025,
            round=1,
            start_time=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i * 7),
            match_state="FullTime",
            venue="V",
            venue_city="C",
            weather=None,
            home=_team(1, 22 if i % 2 == 0 else 16),
            away=_team(2, 16 if i % 2 == 0 else 22),
            team_stats=[],
        )
        for i in range(60)
    ]
    predictor = EnsemblePredictor(mode="weighted")
    predictor.fit(history)

    upcoming = MatchRow(
        match_id=999,
        season=2025,
        round=20,
        start_time=datetime(2025, 8, 1, tzinfo=UTC),
        match_state="Upcoming",
        venue="V",
        venue_city="C",
        weather=None,
        home=_team(1),
        away=_team(2),
        team_stats=[],
    )
    p = predictor.predict_home_win_prob(upcoming)
    assert 0.0 <= p <= 1.0
