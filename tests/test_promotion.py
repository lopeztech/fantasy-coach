"""Tests for :mod:`fantasy_coach.models.promotion`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from fantasy_coach.feature_engineering import FEATURE_NAMES
from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.models.promotion import (
    DEFAULT_MAX_REGRESSION_PCT,
    ShadowMetrics,
    gate_decision,
    shadow_evaluate,
    split_training_holdout,
)


def _match(
    *,
    match_id: int,
    season: int,
    round: int,
    home_id: int,
    away_id: int,
    home_score: int,
    away_score: int,
    when: datetime,
    state: str = "FullTime",
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round,
        start_time=when,
        match_state=state,
        venue=None,
        venue_city=None,
        weather=None,
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


class _FixedProbModel:
    """Minimal ``Model``-protocol implementation returning a constant prob."""

    feature_names = FEATURE_NAMES

    def __init__(self, prob: float) -> None:
        self._prob = prob

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._prob)


# ---------------------------------------------------------------------------
# gate_decision
# ---------------------------------------------------------------------------


def test_gate_promotes_on_tie():
    metrics = ShadowMetrics(n=32, accuracy=0.56, log_loss=0.68, brier=0.24)
    decision = gate_decision(metrics, metrics)
    assert decision.promote is True
    assert decision.log_loss_regression_pct == 0.0
    assert decision.brier_regression_pct == 0.0


def test_gate_promotes_when_candidate_improves():
    inc = ShadowMetrics(n=32, accuracy=0.56, log_loss=0.70, brier=0.25)
    cand = ShadowMetrics(n=32, accuracy=0.58, log_loss=0.65, brier=0.23)
    decision = gate_decision(inc, cand)
    assert decision.promote is True
    assert decision.log_loss_regression_pct < 0  # candidate better = negative
    assert decision.brier_regression_pct < 0


def test_gate_blocks_on_log_loss_regression_above_threshold():
    inc = ShadowMetrics(n=32, accuracy=0.56, log_loss=1.000, brier=0.24)
    cand = ShadowMetrics(n=32, accuracy=0.57, log_loss=1.025, brier=0.24)  # +2.5%
    decision = gate_decision(inc, cand)
    assert decision.promote is False
    assert "log_loss" in decision.reason
    assert decision.log_loss_regression_pct == pytest.approx(2.5)


def test_gate_blocks_on_brier_regression_above_threshold():
    inc = ShadowMetrics(n=32, accuracy=0.56, log_loss=0.68, brier=0.240)
    cand = ShadowMetrics(n=32, accuracy=0.57, log_loss=0.68, brier=0.252)  # +5%
    decision = gate_decision(inc, cand)
    assert decision.promote is False
    assert "brier" in decision.reason


def test_gate_accepts_regression_under_threshold():
    inc = ShadowMetrics(n=32, accuracy=0.56, log_loss=1.000, brier=0.240)
    cand = ShadowMetrics(n=32, accuracy=0.55, log_loss=1.019, brier=0.2445)
    # +1.9% log_loss and +1.875% brier — both under 2%.
    decision = gate_decision(inc, cand)
    assert decision.promote is True


def test_gate_custom_threshold():
    inc = ShadowMetrics(n=32, accuracy=0.56, log_loss=1.00, brier=0.24)
    cand = ShadowMetrics(n=32, accuracy=0.58, log_loss=1.01, brier=0.24)  # +1%
    assert gate_decision(inc, cand).promote is True
    assert gate_decision(inc, cand, max_regression_pct=0.5).promote is False


def test_gate_promotes_when_accuracy_regresses_but_calibration_improves():
    # Accuracy drop ignored; log_loss + brier improve.
    inc = ShadowMetrics(n=32, accuracy=0.60, log_loss=0.80, brier=0.28)
    cand = ShadowMetrics(n=32, accuracy=0.55, log_loss=0.70, brier=0.24)
    decision = gate_decision(inc, cand)
    assert decision.promote is True
    assert decision.accuracy_delta_pct < 0


def test_gate_decision_roundtrip_to_dict():
    inc = ShadowMetrics(n=32, accuracy=0.56, log_loss=0.68, brier=0.24)
    cand = ShadowMetrics(n=32, accuracy=0.57, log_loss=0.65, brier=0.23)
    decision = gate_decision(inc, cand)
    as_dict = decision.to_dict()
    assert as_dict["promote"] is True
    assert as_dict["incumbent"]["log_loss"] == 0.68
    assert as_dict["candidate"]["brier"] == 0.23
    assert "log_loss_regression_pct" in as_dict


def test_gate_default_threshold_is_two_percent():
    # Sanity check that the AC-pinned default hasn't drifted.
    assert DEFAULT_MAX_REGRESSION_PCT == 2.0


# ---------------------------------------------------------------------------
# shadow_evaluate
# ---------------------------------------------------------------------------


def _build_season(n_rounds: int, start: datetime) -> list[MatchRow]:
    matches: list[MatchRow] = []
    mid = 1
    for r in range(1, n_rounds + 1):
        when = start + timedelta(days=7 * (r - 1))
        # Two teams per round with alternating winners so outcomes are mixed.
        matches.append(
            _match(
                match_id=mid,
                season=2024,
                round=r,
                home_id=10,
                away_id=20,
                home_score=24 if r % 2 == 0 else 12,
                away_score=12 if r % 2 == 0 else 24,
                when=when,
            )
        )
        mid += 1
        matches.append(
            _match(
                match_id=mid,
                season=2024,
                round=r,
                home_id=30,
                away_id=40,
                home_score=20,
                away_score=18,
                when=when + timedelta(hours=3),
            )
        )
        mid += 1
    return matches


def test_shadow_evaluate_fixed_prob_model():
    matches = _build_season(n_rounds=6, start=datetime(2024, 3, 1, tzinfo=UTC))
    training, holdout = split_training_holdout(matches, holdout_rounds=2)

    # Model that always predicts 0.9 home win. 4 holdout matches: 2 home
    # wins (round 6 team 10 + both rounds team 30) and 1 away win (round 5
    # team 20). Wait — 2 rounds × 2 matches = 4 holdout matches. Recompute:
    #   round 5 (odd) → home_score=12, away_score=24 (away wins) + team 30
    #   wins at home.
    #   round 6 (even) → home_score=24 (home wins) + team 30 wins.
    # So outcomes (home_win): [0, 1, 1, 1].
    model = _FixedProbModel(prob=0.9)
    metrics = shadow_evaluate(model, training, holdout)

    assert metrics.n == 4
    # Always predicting 0.9: 3 correct ≥ 0.5, 1 wrong. Accuracy 0.75.
    assert metrics.accuracy == 0.75
    # log_loss is positive.
    assert metrics.log_loss > 0


def test_shadow_evaluate_skips_incomplete_and_draws():
    when = datetime(2024, 3, 1, tzinfo=UTC)
    training = [
        _match(
            match_id=1,
            season=2024,
            round=1,
            home_id=10,
            away_id=20,
            home_score=24,
            away_score=18,
            when=when,
        ),
    ]
    holdout = [
        _match(
            match_id=2,
            season=2024,
            round=2,
            home_id=10,
            away_id=20,
            home_score=20,
            away_score=20,
            when=when + timedelta(days=7),
        ),  # draw, skipped
        _match(
            match_id=3,
            season=2024,
            round=2,
            home_id=30,
            away_id=40,
            home_score=None,  # incomplete
            away_score=None,
            when=when + timedelta(days=7, hours=3),
            state="Upcoming",
        ),
        _match(
            match_id=4,
            season=2024,
            round=2,
            home_id=50,
            away_id=60,
            home_score=22,
            away_score=14,
            when=when + timedelta(days=7, hours=6),
        ),
    ]
    model = _FixedProbModel(prob=0.6)
    metrics = shadow_evaluate(model, training, holdout)
    assert metrics.n == 1  # only match 4 is scoreable


def test_shadow_evaluate_empty_holdout_returns_nan_metrics():
    model = _FixedProbModel(prob=0.5)
    metrics = shadow_evaluate(model, [], [])
    assert metrics.n == 0


# ---------------------------------------------------------------------------
# split_training_holdout
# ---------------------------------------------------------------------------


def test_split_training_holdout_basic():
    matches = _build_season(n_rounds=6, start=datetime(2024, 3, 1, tzinfo=UTC))
    training, holdout = split_training_holdout(matches, holdout_rounds=2)

    training_rounds = {(m.season, m.round) for m in training}
    holdout_rounds = {(m.season, m.round) for m in holdout}
    assert holdout_rounds == {(2024, 5), (2024, 6)}
    assert training_rounds == {(2024, r) for r in range(1, 5)}


def test_split_training_holdout_short_history_returns_all_training():
    # Fewer completed rounds than holdout_rounds → no holdout.
    matches = _build_season(n_rounds=2, start=datetime(2024, 3, 1, tzinfo=UTC))
    training, holdout = split_training_holdout(matches, holdout_rounds=4)
    assert len(training) == len(matches)
    assert holdout == []


def test_split_training_holdout_cross_season():
    # Last 2 rounds might span a season boundary.
    matches: list[MatchRow] = []
    matches.extend(_build_season(n_rounds=3, start=datetime(2024, 9, 1, tzinfo=UTC)))
    s2025 = _build_season(n_rounds=3, start=datetime(2025, 3, 1, tzinfo=UTC))
    # Renumber round so each season starts at round 1 again.
    for m in s2025:
        m_new = m.model_copy(update={"season": 2025})
        matches.append(m_new)

    training, holdout = split_training_holdout(matches, holdout_rounds=2)
    holdout_keys = {(m.season, m.round) for m in holdout}
    # Latest two rounds chronologically are the last two of 2025.
    assert holdout_keys == {(2025, 2), (2025, 3)}
