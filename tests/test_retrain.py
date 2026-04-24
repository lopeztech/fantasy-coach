"""Tests for :mod:`fantasy_coach.retrain`."""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from fantasy_coach.drift import DriftReport
from fantasy_coach.feature_engineering import FEATURE_NAMES, build_training_frame
from fantasy_coach.features import MatchRow, TeamRow
from fantasy_coach.models.promotion import GateDecision
from fantasy_coach.models.xgboost_model import save_model as save_xgb
from fantasy_coach.models.xgboost_model import train_xgboost
from fantasy_coach.retrain import (
    RetrainResult,
    _build_drift_report,
    default_gcs_uploader,
    run_retrain,
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
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round,
        start_time=when,
        match_state="FullTime",
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


def _synthetic_matches(
    *,
    n_rounds: int,
    teams: Sequence[int] = (10, 20, 30, 40, 50, 60),
    start: datetime,
    season: int,
    seed: int = 0,
) -> list[MatchRow]:
    rng = random.Random(seed)
    rows: list[MatchRow] = []
    mid = (season % 100) * 1000  # keep match_ids unique across seasons
    for r in range(1, n_rounds + 1):
        when = start + timedelta(days=7 * (r - 1))
        shuffled = list(teams)
        rng.shuffle(shuffled)
        for h, a in zip(shuffled[0::2], shuffled[1::2], strict=True):
            # Favour higher team id — gives the model a learnable signal.
            if h > a:
                home, away = rng.randint(18, 30), rng.randint(6, 18)
            else:
                home, away = rng.randint(6, 18), rng.randint(18, 30)
            rows.append(
                _match(
                    match_id=mid,
                    season=season,
                    round=r,
                    home_id=h,
                    away_id=a,
                    home_score=home,
                    away_score=away,
                    when=when + timedelta(hours=rng.randint(0, 24)),
                )
            )
            mid += 1
    return rows


class _FakeRepo:
    def __init__(self, matches: list[MatchRow]) -> None:
        self._by_season: dict[int, list[MatchRow]] = {}
        for m in matches:
            self._by_season.setdefault(m.season, []).append(m)

    def list_matches(self, season: int, round: int | None = None) -> list[MatchRow]:
        rows = self._by_season.get(season, [])
        return [m for m in rows if round is None or m.round == round]

    # Unused by retrain but satisfies Repository protocol.
    def upsert_match(self, row: MatchRow) -> None:  # pragma: no cover
        self._by_season.setdefault(row.season, []).append(row)

    def get_match(self, match_id: int) -> MatchRow | None:  # pragma: no cover
        for rows in self._by_season.values():
            for m in rows:
                if m.match_id == match_id:
                    return m
        return None


# ---------------------------------------------------------------------------
# _build_drift_report unit test with a fake model
# ---------------------------------------------------------------------------


class _FixedProbModel:
    feature_names = FEATURE_NAMES

    def __init__(self, prob: float = 0.5) -> None:
        self._prob = prob

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._prob)


def test_build_drift_report_populates_all_fields(tmp_path: Path):
    matches = _synthetic_matches(
        n_rounds=8,
        start=datetime(2025, 3, 1, tzinfo=UTC),
        season=2025,
    )
    training = [m for m in matches if m.round <= 4]
    holdout = [m for m in matches if m.round > 4]

    # Write a dummy file so model_version computation has something to hash.
    dummy_path = tmp_path / "incumbent.joblib"
    dummy_path.write_bytes(b"not-a-real-artefact")

    report = _build_drift_report(
        incumbent_model=_FixedProbModel(0.6),
        incumbent_path=dummy_path,
        training=training,
        holdout=holdout,
    )

    assert isinstance(report, DriftReport)
    assert report.season == 2025
    assert report.round == 8  # last holdout round
    assert report.past_round_accuracy is not None
    assert report.past_round_log_loss is not None
    assert report.past_round_brier is not None
    assert len(report.rolling_log_loss) == 4  # rounds 5..8
    assert set(report.feature_psi) == set(FEATURE_NAMES)
    assert len(report.model_version) == 12


def test_build_drift_report_empty_holdout_returns_zeros(tmp_path: Path):
    report = _build_drift_report(
        incumbent_model=_FixedProbModel(0.5),
        incumbent_path=tmp_path / "missing.joblib",
        training=[],
        holdout=[],
    )
    assert report.rolling_log_loss == []
    assert report.past_round_accuracy is None
    assert report.model_version == "unknown"


# ---------------------------------------------------------------------------
# run_retrain end-to-end with fake side effects
# ---------------------------------------------------------------------------


@pytest.fixture
def many_matches() -> list[MatchRow]:
    """30 matches across 3 seasons — enough for XGBoost fallback path."""
    matches = []
    matches.extend(
        _synthetic_matches(
            n_rounds=10,
            start=datetime(2023, 3, 1, tzinfo=UTC),
            season=2023,
            seed=1,
        )
    )
    matches.extend(
        _synthetic_matches(
            n_rounds=10,
            start=datetime(2024, 3, 1, tzinfo=UTC),
            season=2024,
            seed=2,
        )
    )
    matches.extend(
        _synthetic_matches(
            n_rounds=10,
            start=datetime(2025, 3, 1, tzinfo=UTC),
            season=2025,
            seed=3,
        )
    )
    return matches


def _seed_incumbent(incumbent_path: Path, training_matches: list[MatchRow]) -> None:
    """Train a throwaway XGBoost on training_matches, save to ``incumbent_path``.

    Gives the retrain flow a real incumbent to compare against.
    """
    frame = build_training_frame(training_matches)
    result = train_xgboost(frame)
    save_xgb(result, incumbent_path)


def test_run_retrain_happy_path_promotes_and_uploads(tmp_path: Path, many_matches: list[MatchRow]):
    incumbent_path = tmp_path / "incumbent.joblib"
    candidate_path = tmp_path / "candidate.joblib"
    # Seed an incumbent from matches strictly before the holdout window.
    pre_holdout = [m for m in many_matches if (m.season, m.round) < (2025, 7)]
    _seed_incumbent(incumbent_path, pre_holdout)

    drift_calls: list[DriftReport] = []
    uploads: list[tuple[Path, str]] = []
    issue_calls: list[tuple[GateDecision, DriftReport]] = []

    result = run_retrain(
        _FakeRepo(many_matches),
        incumbent_path=incumbent_path,
        candidate_out_path=candidate_path,
        gcs_uri="gs://fake-bucket/latest.joblib",
        seasons=(2023, 2024, 2025),
        holdout_rounds=4,
        drift_writer=drift_calls.append,
        gcs_uploader=lambda p, u: uploads.append((p, u)),
        issue_opener=lambda d, r: (issue_calls.append((d, r)), 999)[1],
    )

    assert isinstance(result, RetrainResult)
    assert candidate_path.exists()
    assert len(drift_calls) == 1
    assert drift_calls[0].season == 2025
    # On identical training inputs the candidate matches incumbent, so we
    # expect promotion — the gate only blocks on regression.
    assert result.promoted is True
    assert result.gcs_uploaded is True
    assert uploads == [(candidate_path, "gs://fake-bucket/latest.joblib")]
    assert issue_calls == []  # not called on promote


def test_run_retrain_no_holdout_raises(tmp_path: Path, many_matches: list[MatchRow]):
    incumbent_path = tmp_path / "incumbent.joblib"
    _seed_incumbent(incumbent_path, many_matches[:20])

    with pytest.raises(RuntimeError, match="Not enough completed rounds"):
        run_retrain(
            _FakeRepo(many_matches[:5]),  # only 5 matches → ≤ 1 round
            incumbent_path=incumbent_path,
            candidate_out_path=tmp_path / "cand.joblib",
            seasons=(2023,),
            holdout_rounds=10,
        )


def test_run_retrain_skips_gcs_upload_when_no_uri(tmp_path: Path, many_matches: list[MatchRow]):
    incumbent_path = tmp_path / "incumbent.joblib"
    _seed_incumbent(
        incumbent_path,
        [m for m in many_matches if (m.season, m.round) < (2025, 7)],
    )
    uploads: list[tuple[Path, str]] = []

    result = run_retrain(
        _FakeRepo(many_matches),
        incumbent_path=incumbent_path,
        candidate_out_path=tmp_path / "cand.joblib",
        gcs_uri=None,  # <- no upload target
        seasons=(2023, 2024, 2025),
        holdout_rounds=4,
        gcs_uploader=lambda p, u: uploads.append((p, u)),
    )
    assert result.promoted is True
    assert result.gcs_uploaded is False
    assert uploads == []


def test_run_retrain_blocks_and_opens_issue(tmp_path: Path, many_matches: list[MatchRow]):
    incumbent_path = tmp_path / "incumbent.joblib"
    # Seed the incumbent with *all* matches — including the holdout — so its
    # shadow metrics dominate the (strictly training-only) candidate.
    _seed_incumbent(incumbent_path, many_matches)

    drift_calls: list[DriftReport] = []
    issue_calls: list[tuple[GateDecision, DriftReport]] = []

    result = run_retrain(
        _FakeRepo(many_matches),
        incumbent_path=incumbent_path,
        candidate_out_path=tmp_path / "cand.joblib",
        seasons=(2023, 2024, 2025),
        holdout_rounds=4,
        # Absurdly tight gate — any regression blocks, so we reliably trigger
        # the block branch regardless of stochastic training outcomes.
        max_regression_pct=-100.0,
        drift_writer=drift_calls.append,
        issue_opener=lambda d, r: (issue_calls.append((d, r)), 42)[1],
    )
    assert result.promoted is False
    assert result.issue_number == 42
    assert len(issue_calls) == 1
    assert len(drift_calls) == 1  # drift report always written


# ---------------------------------------------------------------------------
# default_gcs_uploader argument validation
# ---------------------------------------------------------------------------


def test_default_gcs_uploader_rejects_bad_uri(tmp_path: Path):
    p = tmp_path / "x.joblib"
    p.write_bytes(b"ok")
    with pytest.raises(ValueError, match="must start with gs://"):
        default_gcs_uploader(p, "s3://not-gcs/x")
    with pytest.raises(ValueError, match="gs://<bucket>/<blob>"):
        default_gcs_uploader(p, "gs://bucket-only")
