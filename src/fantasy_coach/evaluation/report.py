"""Render walk-forward results into a markdown comparison table."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from fantasy_coach.evaluation.harness import EvaluationResult


def render_markdown(
    results: Sequence[EvaluationResult],
    *,
    seasons: Sequence[int],
    generated_at: datetime | None = None,
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
        "| Model | n predictions | Accuracy | Log loss | Brier |",
        "|-------|---------------|----------|----------|-------|",
    ]
    for result in results:
        m = result.metrics()
        lines.append(
            f"| {result.predictor_name} | {result.n} | "
            f"{m['accuracy']:.3f} | {m['log_loss']:.3f} | {m['brier']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    results: Sequence[EvaluationResult],
    *,
    seasons: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(results, seasons=seasons))
