"""Tests for :mod:`fantasy_coach.github_issue`."""

from __future__ import annotations

import httpx
import pytest
import respx

from fantasy_coach.drift import DriftReport, RoundLogLoss
from fantasy_coach.github_issue import (
    API_BASE,
    DEFAULT_LABEL,
    TOKEN_ENV,
    open_github_model_drift_issue,
    render_issue_body,
)
from fantasy_coach.models.promotion import GateDecision, ShadowMetrics


def _decision(promote: bool = False) -> GateDecision:
    return GateDecision(
        promote=promote,
        reason="log_loss regression +3.00% exceeds threshold +2.00%",
        incumbent=ShadowMetrics(n=32, accuracy=0.56, log_loss=0.68, brier=0.24),
        candidate=ShadowMetrics(n=32, accuracy=0.57, log_loss=0.70, brier=0.245),
        log_loss_regression_pct=3.0,
        brier_regression_pct=2.08,
        accuracy_delta_pct=1.79,
    )


def _report() -> DriftReport:
    return DriftReport(
        season=2026,
        round=7,
        generated_at="2026-04-24T00:00:00+00:00",
        model_version="abc123def456",
        past_round_accuracy=0.55,
        past_round_log_loss=0.72,
        past_round_brier=0.25,
        rolling_log_loss=[
            RoundLogLoss(season=2026, round=4, n=8, log_loss=0.69, accuracy=0.55),
            RoundLogLoss(season=2026, round=5, n=8, log_loss=0.70, accuracy=0.56),
            RoundLogLoss(season=2026, round=6, n=8, log_loss=0.71, accuracy=0.55),
            RoundLogLoss(season=2026, round=7, n=8, log_loss=0.72, accuracy=0.55),
        ],
        feature_psi={"elo_diff": 0.05, "form_diff_pf": 0.31, "odds_home_win_prob": 0.28},
        psi_warnings=["form_diff_pf", "odds_home_win_prob"],
    )


def test_render_body_contains_metrics_table():
    body = render_issue_body(_decision(), _report())
    assert "| accuracy |" in body
    assert "| log_loss |" in body
    assert "| brier    |" in body
    assert "+3.00%" in body  # regression pct surfaced


def test_render_body_lists_psi_warnings():
    body = render_issue_body(_decision(), _report())
    assert "`form_diff_pf`" in body
    assert "`odds_home_win_prob`" in body


def test_render_body_handles_no_psi_warnings():
    report = DriftReport(
        season=2026,
        round=7,
        generated_at="2026-04-24T00:00:00+00:00",
        model_version="abc123def456",
        past_round_accuracy=0.55,
        past_round_log_loss=0.72,
        past_round_brier=0.25,
        rolling_log_loss=[],
        feature_psi={},
        psi_warnings=[],
    )
    body = render_issue_body(_decision(), report)
    assert "No PSI warnings" in body


def test_render_body_includes_rolling_trend_rows():
    body = render_issue_body(_decision(), _report())
    assert "| 2026 | 4 | 8 |" in body
    assert "| 2026 | 7 | 8 |" in body


def test_open_issue_raises_on_passing_decision():
    with pytest.raises(ValueError, match="passing gate decision"):
        open_github_model_drift_issue(_decision(promote=True), _report())


def test_open_issue_returns_none_when_token_missing(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    result = open_github_model_drift_issue(_decision(), _report())
    assert result is None


@respx.mock
def test_open_issue_posts_to_github_returns_number(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "ghp_faketoken")
    route = respx.post(f"{API_BASE}/repos/lopeztech/fantasy-coach/issues").mock(
        return_value=httpx.Response(201, json={"number": 4242})
    )

    number = open_github_model_drift_issue(_decision(), _report())

    assert number == 4242
    assert route.called
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer ghp_faketoken"
    body = request.content.decode()
    assert DEFAULT_LABEL in body
    assert "retrain blocked" in body


@respx.mock
def test_open_issue_returns_none_on_api_error(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "ghp_faketoken")
    respx.post(f"{API_BASE}/repos/lopeztech/fantasy-coach/issues").mock(
        return_value=httpx.Response(500, json={"message": "boom"})
    )
    assert open_github_model_drift_issue(_decision(), _report()) is None


@respx.mock
def test_open_issue_returns_none_on_network_error(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "ghp_faketoken")
    respx.post(f"{API_BASE}/repos/lopeztech/fantasy-coach/issues").mock(
        side_effect=httpx.ConnectError("no route")
    )
    assert open_github_model_drift_issue(_decision(), _report()) is None


@respx.mock
def test_open_issue_uses_explicit_token_and_repo(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    route = respx.post(f"{API_BASE}/repos/other/repo/issues").mock(
        return_value=httpx.Response(201, json={"number": 1}),
    )
    number = open_github_model_drift_issue(
        _decision(),
        _report(),
        repo="other/repo",
        token="explicit",
    )
    assert number == 1
    assert route.called
    assert route.calls[0].request.headers["Authorization"] == "Bearer explicit"
