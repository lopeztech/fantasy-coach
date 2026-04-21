from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from fantasy_coach.feature_engineering import (
    FEATURE_NAMES,
    build_training_frame,
)
from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.models.logistic import (
    load_model,
    save_model,
    train_logistic,
)


def _generate_matches(n: int, *, seed: int = 0) -> list[MatchRow]:
    """Synthesize matches where team 10 is consistently stronger than team 20.

    Lets us assert that the model learns *something* (>50% test accuracy)
    rather than memorising noise.
    """
    rng = random.Random(seed)
    base = datetime(2024, 3, 1, tzinfo=UTC)
    matches: list[MatchRow] = []
    for i in range(n):
        # Alternate home/away so home advantage isn't trivially confounded.
        if i % 2 == 0:
            home_id, away_id = 10, 20
            home_pts = rng.randint(20, 36)
            away_pts = rng.randint(6, 18)
        else:
            home_id, away_id = 20, 10
            home_pts = rng.randint(6, 18)
            away_pts = rng.randint(20, 36)
        matches.append(
            MatchRow(
                match_id=i + 1,
                season=2024,
                round=i + 1,
                start_time=base + timedelta(days=7 * i),
                match_state="FullTime",
                venue=None,
                venue_city=None,
                weather=None,
                home=TeamRow(
                    team_id=home_id,
                    name=str(home_id),
                    nick_name=str(home_id),
                    score=home_pts,
                    players=[],
                ),
                away=TeamRow(
                    team_id=away_id,
                    name=str(away_id),
                    nick_name=str(away_id),
                    score=away_pts,
                    players=[],
                ),
                team_stats=[],
            )
        )
    return matches


def test_train_returns_a_fitted_pipeline() -> None:
    frame = build_training_frame(_generate_matches(50))
    result = train_logistic(frame, test_fraction=0.2)
    assert result.n_train + result.n_test == 50
    # On a clean signal, a logistic model should beat 50/50 by a clear margin.
    assert result.test_accuracy > 0.7


def test_train_refuses_too_few_rows() -> None:
    frame = build_training_frame(_generate_matches(5))
    with pytest.raises(ValueError, match="at least 10 rows"):
        train_logistic(frame)


def test_save_load_round_trip(tmp_path: Path) -> None:
    frame = build_training_frame(_generate_matches(50))
    result = train_logistic(frame)
    path = tmp_path / "model.joblib"
    save_model(result, path)

    loaded = load_model(path)
    assert loaded.feature_names == FEATURE_NAMES

    # Prediction probability should match the in-memory pipeline within numerical tolerance.
    sample = frame.X[:5]
    expected = result.pipeline.predict_proba(sample)[:, 1]
    actual = loaded.predict_home_win_prob(sample)
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_load_rejects_models_with_stale_feature_names(tmp_path: Path) -> None:
    frame = build_training_frame(_generate_matches(50))
    result = train_logistic(frame)
    path = tmp_path / "stale.joblib"

    import joblib

    joblib.dump(
        {"pipeline": result.pipeline, "feature_names": ("only_one",)},
        path,
    )

    with pytest.raises(RuntimeError, match="Retrain"):
        load_model(path)


def test_predict_rejects_wrong_feature_count(tmp_path: Path) -> None:
    frame = build_training_frame(_generate_matches(50))
    result = train_logistic(frame)
    path = tmp_path / "ok.joblib"
    save_model(result, path)
    loaded = load_model(path)

    bad = np.zeros((1, len(FEATURE_NAMES) - 1))
    with pytest.raises(ValueError, match="Expected"):
        loaded.predict_home_win_prob(bad)
