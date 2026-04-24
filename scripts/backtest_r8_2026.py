"""One-off R8 2026 backtest for #203.

Reads R8 fixtures + history from `data/nrl.db`, predicts each fixture with
the production XGBoost artefact (capped PSD applied at feature_row time),
applies the new market shrinkage, and prints old-vs-new probabilities for
the three completed games. Used to confirm the cap + shrinkage flips
Tigers/Raiders before merging the PR.

Run:
    uv run python scripts/backtest_r8_2026.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fantasy_coach.feature_engineering import (
    FEATURE_NAMES,
    PLAYER_STRENGTH_DIFF_CAP,
    FeatureBuilder,
)
from fantasy_coach.models.loader import load_model
from fantasy_coach.predictions import MARKET_SHRINKAGE_WEIGHT, _apply_market_shrinkage
from fantasy_coach.storage import SQLiteRepository

# Original Firestore predictions for the 3 completed R8 games (homeP, predicted).
ORIG = {
    20261110810: ("Wests Tigers", "Canberra Raiders", 0.457, "away", "Tigers 33-14"),
    20261110820: (
        "North Queensland Cowboys",
        "Cronulla-Sutherland Sharks",
        0.486,
        "away",
        "Cowboys 46-34",
    ),
    20261110830: (
        "Brisbane Broncos",
        "Canterbury-Bankstown Bulldogs",
        0.409,
        "away",
        "Broncos 32-12",
    ),
}


def main() -> None:
    repo = SQLiteRepository(Path("data/nrl.db"))
    try:
        history: list = []
        for season in (2023, 2024, 2025, 2026):
            try:
                history.extend(repo.list_matches(season))
            except Exception:
                continue
        # Treat anything before R8 2026 as history for inference state.
        round_8 = [m for m in history if m.season == 2026 and m.round == 8]
        prior = [
            m
            for m in history
            if (m.season < 2026) or (m.season == 2026 and m.round < 8 and m.home.score is not None)
        ]
    finally:
        repo.close()

    builder = FeatureBuilder()
    for match in sorted(prior, key=lambda m: (m.start_time, m.match_id)):
        if match.home.score is None or match.away.score is None:
            continue
        builder.advance_season_if_needed(match)
        builder.record(match)

    loaded = load_model(Path("artifacts/xgboost.joblib"))
    feature_names = (
        loaded.feature_names if isinstance(loaded.feature_names, tuple) else FEATURE_NAMES
    )
    psd_idx = feature_names.index("player_strength_diff")
    odds_idx = feature_names.index("odds_home_win_prob")
    missing_idx = feature_names.index("missing_odds")

    print(f"PLAYER_STRENGTH_DIFF_CAP = {PLAYER_STRENGTH_DIFF_CAP}")
    print(f"MARKET_SHRINKAGE_WEIGHT  = {MARKET_SHRINKAGE_WEIGHT}")
    print()
    print(
        f"{'matchId':>12}  {'PSD':>7}  {'odds':>5}  {'raw':>5}  "
        f"{'final':>5}  {'pick':>4}  {'orig':>5}  result"
    )
    print("-" * 80)

    for match in sorted(round_8, key=lambda m: (m.start_time, m.match_id)):
        if match.match_id not in ORIG:
            continue
        x = np.asarray([builder.feature_row(match)], dtype=float)
        psd = float(x[0, psd_idx])
        odds = float(x[0, odds_idx]) if x[0, missing_idx] < 0.5 else None
        raw_prob = float(loaded.predict_home_win_prob(x)[0])
        final_prob, market = _apply_market_shrinkage(round(raw_prob, 4), x, feature_names)
        pick = "home" if final_prob >= 0.5 else "away"
        orig = ORIG[match.match_id]
        print(
            f"{match.match_id:>12}  {psd:>+7.0f}  "
            f"{(f'{odds:.3f}' if odds is not None else '  -  '):>5}  "
            f"{raw_prob:>.3f}  {final_prob:>.3f}  {pick:>4}  "
            f"{orig[2]:>.3f}  {orig[4]} (orig pick: {orig[3]})"
        )


if __name__ == "__main__":
    main()
