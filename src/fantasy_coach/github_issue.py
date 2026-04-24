"""GitHub issue creation for blocked-retrain events (#107).

When the promotion gate blocks a candidate, the retrain loop calls
``open_github_model_drift_issue`` to file an issue tagged ``model-drift``.
The token comes from env var ``GITHUB_MODEL_DRIFT_TOKEN`` — mounted from
Secret Manager on the Cloud Run Job via platform-infra.

If the token is absent the function logs + returns ``None``; the retrain
pipeline still writes the drift report to Firestore so the signal isn't
lost. This keeps local development (where no token is configured)
unaffected by a would-be block.
"""

from __future__ import annotations

import logging
import os

import httpx

from fantasy_coach.drift import DriftReport
from fantasy_coach.models.promotion import GateDecision

logger = logging.getLogger(__name__)

TOKEN_ENV = "GITHUB_MODEL_DRIFT_TOKEN"
DEFAULT_REPO = "lopeztech/fantasy-coach"
DEFAULT_LABEL = "model-drift"
API_BASE = "https://api.github.com"
_TIMEOUT_SECONDS = 10.0


def open_github_model_drift_issue(
    decision: GateDecision,
    report: DriftReport,
    *,
    repo: str = DEFAULT_REPO,
    token: str | None = None,
    client: httpx.Client | None = None,
) -> int | None:
    """Open a ``model-drift`` issue describing the blocked promotion.

    Returns the issue number on success, ``None`` if no token is
    configured (graceful no-op for local dev) or the API call fails.
    ``client`` is injectable for tests.
    """
    if decision.promote:
        raise ValueError("open_github_model_drift_issue called for a passing gate decision")

    token = token or os.getenv(TOKEN_ENV)
    if not token:
        logger.warning(
            "%s not set; skipping GitHub issue creation. "
            "Drift report was still written to Firestore.",
            TOKEN_ENV,
        )
        return None

    title = _title(report)
    body = render_issue_body(decision, report)
    payload = {"title": title, "body": body, "labels": [DEFAULT_LABEL]}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "fantasy-coach-retrain/1.0",
    }

    _client = client or httpx.Client(timeout=_TIMEOUT_SECONDS)
    try:
        response = _client.post(
            f"{API_BASE}/repos/{repo}/issues",
            json=payload,
            headers=headers,
        )
    except httpx.HTTPError:
        logger.exception("failed to call GitHub issues API")
        return None
    finally:
        if client is None:
            _client.close()

    if response.status_code >= 300:
        logger.error(
            "GitHub issue creation returned %d: %s",
            response.status_code,
            response.text[:500],
        )
        return None

    return int(response.json().get("number"))


def _title(report: DriftReport) -> str:
    return f"Model drift: retrain blocked for season {report.season} round {report.round}"


def render_issue_body(decision: GateDecision, report: DriftReport) -> str:
    """Markdown body for the drift issue — pure function for easy testing."""
    inc = decision.incumbent
    cand = decision.candidate

    lines: list[str] = [
        f"The weekly retrain pipeline (#107) trained a candidate model for "
        f"season {report.season} round {report.round} but the promotion gate "
        f"**blocked** it from going live.",
        "",
        f"**Reason:** {decision.reason}",
        "",
        f"**Incumbent model version:** `{report.model_version}`",
        f"**Drift report Firestore doc:** `model_drift_reports/{report.season}-{report.round:02d}`",
        "",
        "## Shadow-eval metrics (last 4 rounds)",
        "",
        "| Metric | Incumbent | Candidate | Δ % |",
        "|---|---:|---:|---:|",
        (
            f"| accuracy | {inc.accuracy:.4f} | {cand.accuracy:.4f} | "
            f"{decision.accuracy_delta_pct:+.2f}% |"
        ),
        (
            f"| log_loss | {inc.log_loss:.4f} | {cand.log_loss:.4f} | "
            f"{decision.log_loss_regression_pct:+.2f}% |"
        ),
        (
            f"| brier    | {inc.brier:.4f} | {cand.brier:.4f} | "
            f"{decision.brier_regression_pct:+.2f}% |"
        ),
        f"| n        | {inc.n} | {cand.n} | — |",
        "",
        "## Rolling log-loss trend",
        "",
        "| Season | Round | n | Log-loss | Accuracy |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in report.rolling_log_loss:
        lines.append(f"| {r.season} | {r.round} | {r.n} | {r.log_loss:.4f} | {r.accuracy:.4f} |")

    if report.psi_warnings:
        lines += [
            "",
            f"## PSI warnings (> {', '.join(report.psi_warnings)})",
            "",
            "| Feature | PSI |",
            "|---|---:|",
        ]
        for feat in report.psi_warnings:
            lines.append(f"| `{feat}` | {report.feature_psi.get(feat, 0.0):.3f} |")
    else:
        lines += ["", "No PSI warnings (all features under threshold)."]

    lines += [
        "",
        "## Rollback / next steps",
        "",
        (
            "- The incumbent artefact at "
            "`gs://fantasy-coach-lcd-models/logistic/latest.joblib` is unchanged."
        ),
        (
            "- Inspect the candidate artefact saved on the Cloud Run Job's "
            "ephemeral disk for post-mortem — re-run the retrain Job with "
            "`--dry-run` to reproduce locally."
        ),
        (
            "- If the regression is a known expected shift (e.g. finals series), "
            "close this issue with `reason: not planned`. The next week's "
            "retrain re-evaluates from scratch."
        ),
        "",
        "_Filed automatically by the retrain Cloud Run Job._",
    ]
    return "\n".join(lines)
