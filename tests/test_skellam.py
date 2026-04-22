"""Unit tests for the Skellam/Poisson margin model."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.models.skellam import (
    MarginDistribution,
    SkellamModel,
    SkellamTrainingFrame,
    build_skellam_frame,
    load_skellam,
    save_skellam,
    train_skellam,
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


def _minimal_frame(n: int = 20) -> SkellamTrainingFrame:
    """Build a minimal training frame from synthetic matches."""
    matches = [
        _make_match(i, home_score=20 + i % 5, away_score=15 + i % 3, offset_seconds=i)
        for i in range(n)
    ]
    return build_skellam_frame(matches)


# ---------------------------------------------------------------------------
# build_skellam_frame
# ---------------------------------------------------------------------------


def test_build_skellam_frame_includes_all_complete_matches() -> None:
    matches = [_make_match(i, 20, 10, offset_seconds=i) for i in range(5)]
    frame = build_skellam_frame(matches)
    assert frame.X.shape == (5, len(frame.feature_names))
    assert len(frame.y_home) == 5
    assert len(frame.y_away) == 5


def test_build_skellam_frame_includes_draws() -> None:
    matches = [_make_match(i, 18, 18, offset_seconds=i) for i in range(5)]
    frame = build_skellam_frame(matches)
    assert frame.X.shape[0] == 5


def test_build_skellam_frame_excludes_incomplete_matches() -> None:
    complete = _make_match(1, 20, 10, offset_seconds=0)
    incomplete = MatchRow(
        match_id=2,
        season=2024,
        round=2,
        start_time=datetime(2024, 3, 2, 9, 0, tzinfo=UTC),
        match_state="Upcoming",
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(team_id=1, name="Home", nick_name="Home", score=None, players=[]),
        away=TeamRow(team_id=2, name="Away", nick_name="Away", score=None, players=[]),
        team_stats=[],
    )
    frame = build_skellam_frame([complete, incomplete])
    assert frame.X.shape[0] == 1


def test_build_skellam_frame_scores_match_targets() -> None:
    matches = [
        _make_match(i, home_score=10 + i, away_score=5 + i, offset_seconds=i) for i in range(3)
    ]
    frame = build_skellam_frame(matches)
    for i, (hs, as_) in enumerate(zip(frame.y_home, frame.y_away, strict=True)):
        assert hs == 10 + i
        assert as_ == 5 + i


def test_build_skellam_frame_empty_returns_zero_shape() -> None:
    frame = build_skellam_frame([])
    assert frame.X.shape[0] == 0
    assert frame.y_home.shape[0] == 0


# ---------------------------------------------------------------------------
# train_skellam
# ---------------------------------------------------------------------------


def test_train_skellam_returns_model() -> None:
    frame = _minimal_frame(20)
    result = train_skellam(frame)
    assert isinstance(result.model, SkellamModel)
    assert result.n_train == 20


def test_train_skellam_raises_with_too_few_rows() -> None:
    frame = _minimal_frame(5)
    with pytest.raises(ValueError, match="at least 10"):
        train_skellam(frame)


# ---------------------------------------------------------------------------
# SkellamModel.predict_margin_distribution
# ---------------------------------------------------------------------------


def test_predict_margin_distribution_returns_valid_distribution() -> None:
    frame = _minimal_frame(30)
    model = train_skellam(frame).model
    x = frame.X[:1]
    dist = model.predict_margin_distribution(x)

    assert isinstance(dist, MarginDistribution)
    assert 0.0 <= dist.home_win_prob <= 1.0
    assert abs(dist.pmf.sum() - 1.0) < 1e-6
    assert dist.lambda_home > 0
    assert dist.lambda_away > 0


def test_predict_margin_distribution_pmf_shape() -> None:
    frame = _minimal_frame(30)
    model = train_skellam(frame).model
    dist = model.predict_margin_distribution(frame.X[:1])
    assert dist.pmf.shape == (81,)  # -40 to +40 inclusive


def test_predict_margin_distribution_ci95_ordered() -> None:
    frame = _minimal_frame(30)
    model = train_skellam(frame).model
    dist = model.predict_margin_distribution(frame.X[:1])
    lo, hi = dist.ci_95
    assert lo <= hi


def test_stronger_home_team_gives_higher_home_win_prob() -> None:
    """Matches where home is historically much stronger → higher home win prob."""
    # Build matches where home always wins big (home team id=1, away id=2)
    matches = [_make_match(i, 40, 5, home_id=1, away_id=2, offset_seconds=i) for i in range(30)]
    frame = build_skellam_frame(matches)
    model = train_skellam(frame).model
    # For a typical feature row from this frame, home win prob should be high
    dist = model.predict_margin_distribution(frame.X[:1])
    assert dist.home_win_prob > 0.5


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_skellam_roundtrip(tmp_path: Path) -> None:
    frame = _minimal_frame(30)
    result = train_skellam(frame)
    path = tmp_path / "skellam.joblib"
    save_skellam(path, result)

    loaded = load_skellam(path)
    assert isinstance(loaded, SkellamModel)

    x = frame.X[:1]
    dist_orig = result.model.predict_margin_distribution(x)
    dist_loaded = loaded.predict_margin_distribution(x)
    assert dist_orig.home_win_prob == pytest.approx(dist_loaded.home_win_prob)
    assert dist_orig.mean == pytest.approx(dist_loaded.mean)


def test_load_skellam_rejects_wrong_model_type(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"model_type": "logistic"}, path)
    with pytest.raises(ValueError, match="skellam"):
        load_skellam(path)
