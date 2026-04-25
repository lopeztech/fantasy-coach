"""Render walk-forward results into a markdown comparison table."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fantasy_coach.evaluation.harness import EvaluationResult
from fantasy_coach.models.calibration import reliability_bins

if TYPE_CHECKING:
    from fantasy_coach.evaluation.profit import CLVReport


def render_markdown(
    results: Sequence[EvaluationResult],
    *,
    seasons: Sequence[int],
    generated_at: datetime | None = None,
    clv_reports: Sequence[CLVReport] | None = None,
) -> str:
    generated_at = generated_at or datetime.now()
    lines = [
        "# Model evaluation",
        "",
        f"Generated: {generated_at.isoformat(timespec='seconds')}",
        f"Seasons: {', '.join(str(s) for s in sorted(seasons))}",
        "",
        "Walk-forward: for each round, train on every prior completed match, "
        "predict that round, score against actuals. Draws are dropped from "
        "scoring (binary metrics).",
        "",
        "| Model | n predictions | Accuracy | Log loss | Brier | ECE |",
        "|-------|---------------|----------|----------|-------|-----|",
    ]
    for result in results:
        m = result.metrics()
        ece_val = m.get("ece", float("nan"))
        ece_str = f"{ece_val:.3f}" if ece_val == ece_val else "n/a"  # NaN check
        lines.append(
            f"| {result.predictor_name} | {result.n} | "
            f"{m['accuracy']:.3f} | {m['log_loss']:.3f} | {m['brier']:.3f} | {ece_str} |"
        )
    lines.append("")

    # Reliability diagrams — one section per model.
    lines.append("## Reliability diagrams")
    lines.append("")
    lines.append(
        "Each row is a probability bin. A well-calibrated model has "
        "`mean_confidence ≈ mean_accuracy` in every bin."
    )
    lines.append("")

    for result in results:
        if not result.predictions:
            continue
        bins = reliability_bins(result.probs, result.actuals)
        lines.append(f"### {result.predictor_name}")
        lines.append("")
        lines.append("| Bin | Mean confidence | Mean accuracy | n |")
        lines.append("|-----|----------------|---------------|---|")
        for b in bins:
            conf = f"{b['mean_confidence']:.3f}" if b["mean_confidence"] is not None else "—"
            acc = f"{b['mean_accuracy']:.3f}" if b["mean_accuracy"] is not None else "—"
            lines.append(f"| {b['lo']:.1f}–{b['hi']:.1f} | {conf} | {acc} | {b['n']} |")
        lines.append("")

    if clv_reports:
        lines.append("## Market efficiency (CLV)")
        lines.append("")
        lines.append(
            "CLV = model probability − de-vigged closing line probability.  "
            "Positive mean CLV indicates the model consistently finds value "
            "the market corrects to — the long-run edge signal."
        )
        lines.append("")
        lines.append("| Model | n (with odds) | Mean CLV | Win rate | ROI flat |")
        lines.append("|-------|--------------|----------|----------|----------|")
        for report in clv_reports:
            lines.append(
                f"| {report.predictor_name} | {report.n} | "
                f"{report.mean_clv:+.4f} | {report.win_rate:.3f} | "
                f"{report.roi_flat:+.3f} |"
            )
        lines.append("")

        for report in clv_reports:
            lines.append(f"### {report.predictor_name} — cumulative CLV curve")
            lines.append("")
            lines.append("| # | Season | Round | CLV | Cumulative CLV |")
            lines.append("|---|--------|-------|-----|----------------|")
            cum = report.cumulative_clv
            for i, (c, cumulative) in enumerate(zip(report.match_clvs, cum, strict=True), start=1):
                lines.append(f"| {i} | {c.season} | {c.round} | {c.clv:+.4f} | {cumulative:+.4f} |")
            lines.append("")

    return "\n".join(lines)


def write_markdown(
    path: Path,
    results: Sequence[EvaluationResult],
    *,
    seasons: Sequence[int],
    clv_reports: Sequence[CLVReport] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(results, seasons=seasons, clv_reports=clv_reports))
