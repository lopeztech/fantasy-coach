"""Ablation: referee features with populated referee data.

Usage: uv run python scripts/ablate_referee_features.py

Compares walk-forward metrics (logistic + XGBoost) on the baseline fixture DB
with referee features enabled (current state) vs disabled (forced to the
"referee unknown" fallback — equivalent to referee_id = NULL for every match).

Fixture coverage: 2023 213/213, 2024 213/213, 2025 210/213, 2026 64/204.
This is a substantial improvement over the #57 ablation where all rows had
referee_id = NULL.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fantasy_coach import feature_engineering as fe
from fantasy_coach.evaluation import LogisticPredictor, XGBoostPredictor
from fantasy_coach.evaluation.harness import walk_forward_from_repo
from fantasy_coach.storage import SQLiteRepository

BASELINE_DB = Path(__file__).parent.parent / "tests" / "fixtures" / "baseline-nrl.db"
SEASONS = (2023, 2024, 2025, 2026)
PREDICTORS = [LogisticPredictor, XGBoostPredictor]


def run(label: str, patch_ctx=None) -> None:
    print(f"\n=== {label} ===")
    print(f"{'Predictor':<12} {'n':>4} {'accuracy':>10} {'log_loss':>10} {'brier':>8}")
    print("-" * 50)
    for cls in PREDICTORS:
        repo = SQLiteRepository(BASELINE_DB)
        try:
            if patch_ctx is not None:
                with patch_ctx:
                    result = walk_forward_from_repo(repo, SEASONS, cls)
            else:
                result = walk_forward_from_repo(repo, SEASONS, cls)
        finally:
            repo.close()
        m = result.metrics()
        print(
            f"{cls.__name__:<12} {result.n:>4} {m['accuracy']:>10.4f} "
            f"{m['log_loss']:>10.4f} {m['brier']:>8.4f}"
        )


def _no_ref(self, referee_id):  # noqa: ANN001
    """Force referee_id = None path for every match."""
    league_mean = fe._avg(self._league_total) if self._league_total else 0.0
    return league_mean, 0.0, 1.0


if __name__ == "__main__":
    run("WITH referee features (populated: 2023-100%, 2024-100%, 2025-99%)")
    run(
        "WITHOUT referee features (forced missing_referee=1.0 for all matches)",
        patch.object(fe.FeatureBuilder, "_referee_features", _no_ref),
    )
    print()
