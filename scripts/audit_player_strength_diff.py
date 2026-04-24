"""Audit player_strength_diff directional validity (#166).

Answers 4 questions:
1. Directional correlation on holdout: does higher player_strength_diff correlate
   with home wins? Bucket by quintile.
2. Partial dependence: show the correlation bucket table as a proxy for PDP.
3. Magnitude check: what fraction of predictions have |player_strength_diff| > 500?
4. Market comparison: when player_strength_diff direction and odds_home_win_prob
   disagree, how often does each get it right?

Usage:
    uv run python scripts/audit_player_strength_diff.py
    uv run python scripts/audit_player_strength_diff.py --db tests/fixtures/baseline-nrl.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running the script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from fantasy_coach.feature_engineering import FEATURE_NAMES, FeatureBuilder
from fantasy_coach.features import MatchRow
from fantasy_coach.storage import SQLiteRepository

SEASONS = (2024, 2025, 2026)

# Feature indices
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
PSD_IDX = _IDX["player_strength_diff"]
ODDS_IDX = _IDX["odds_home_win_prob"]
MISSING_PSD_IDX = _IDX["missing_player_strength"]
MISSING_ODDS_IDX = _IDX["missing_odds"]


@dataclass
class AuditRow:
    match_id: int
    season: int
    round: int
    player_strength_diff: float
    odds_home_win_prob: float
    missing_psd: float
    missing_odds: float
    actual_home_win: int


def collect_feature_rows(
    repo: SQLiteRepository,
    seasons: tuple[int, ...],
) -> list[AuditRow]:
    """Walk the baseline DB and collect feature rows + outcomes using walk-forward state."""
    rows: list[AuditRow] = []

    all_matches: list[MatchRow] = []
    for season in seasons:
        all_matches.extend(repo.list_matches(season))
    all_matches.sort(key=lambda m: (m.start_time, m.match_id))

    # Group into (season, round) for walk-forward logic
    by_round: dict[tuple[int, int], list[MatchRow]] = defaultdict(list)
    for m in all_matches:
        by_round[(m.season, m.round)].append(m)

    # Walk forward: for each round, compute features from history
    history: list[MatchRow] = []
    builder = FeatureBuilder()
    current_season: int | None = None

    for (season, _round), round_matches in sorted(by_round.items()):
        # Check for season transition
        if current_season is None:
            current_season = season
        elif season != current_season:
            builder.elo.regress_to_mean()
            builder.player_ratings.regress_to_mean()
            current_season = season

        completed = [
            m
            for m in round_matches
            if m.home.score is not None
            and m.away.score is not None
            and m.home.score != m.away.score  # skip draws
            and m.match_state in {"FullTime", "FullTimeED"}
        ]

        for match in sorted(completed, key=lambda m: (m.start_time, m.match_id)):
            features = builder.feature_row(match)
            rows.append(
                AuditRow(
                    match_id=match.match_id,
                    season=match.season,
                    round=match.round,
                    player_strength_diff=features[PSD_IDX],
                    odds_home_win_prob=features[ODDS_IDX],
                    missing_psd=features[MISSING_PSD_IDX],
                    missing_odds=features[MISSING_ODDS_IDX],
                    actual_home_win=1 if (match.home.score or 0) > (match.away.score or 0) else 0,
                )
            )

        # Record outcomes into state
        for match in sorted(round_matches, key=lambda m: (m.start_time, m.match_id)):
            if match.home.score is None or match.away.score is None:
                continue
            builder.advance_season_if_needed(match)
            builder.record(match)
        history.extend(round_matches)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit player_strength_diff directional validity")
    parser.add_argument(
        "--db",
        default="tests/fixtures/baseline-nrl.db",
        help="Path to SQLite database",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    repo = SQLiteRepository(db_path)
    try:
        rows = collect_feature_rows(repo, SEASONS)
    finally:
        repo.close()

    print("=" * 70)
    print("player_strength_diff Directional Audit (#166)")
    print("=" * 70)
    print(f"\nTotal predictions (non-draw): {len(rows)}")
    print(f"Seasons: {SEASONS}")

    # Subset with valid PSD (non-missing)
    valid = [r for r in rows if r.missing_psd == 0.0]
    missing_count = sum(1 for r in rows if r.missing_psd > 0.0)
    print(f"\nValid PSD rows: {len(valid)} ({len(valid) / len(rows):.1%})")
    print(f"Missing PSD rows: {missing_count} ({missing_count / len(rows):.1%})")

    psd_values = np.array([r.player_strength_diff for r in valid])
    print("\nPSD stats on valid rows:")
    print(f"  min:    {psd_values.min():>10.1f}")
    print(f"  p25:    {np.percentile(psd_values, 25):>10.1f}")
    print(f"  median: {np.median(psd_values):>10.1f}")
    print(f"  p75:    {np.percentile(psd_values, 75):>10.1f}")
    print(f"  max:    {psd_values.max():>10.1f}")
    print(f"  mean:   {psd_values.mean():>10.1f}")
    print(f"  std:    {psd_values.std():>10.1f}")

    # -----------------------------------------------------------------------
    # Q1: Directional correlation by quintile
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Q1 — Directional Correlation by Quintile")
    print("=" * 70)
    print("""
Does higher player_strength_diff predict home wins?
A well-calibrated feature should show monotone increasing home win rate
from the lowest quintile (away-team-advantaged) to highest (home-advantaged).
""")

    if len(valid) > 0:
        quintiles = np.percentile(psd_values, [20, 40, 60, 80])
        bins = [-np.inf] + list(quintiles) + [np.inf]
        labels = ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]

        print(f"{'Quintile':<15} {'PSD range':<28} {'n':>4} {'home_win_rate':>14}")
        print("-" * 65)
        for i in range(5):
            lo, hi = bins[i], bins[i + 1]
            bucket = [r for r in valid if lo < r.player_strength_diff <= hi]
            if not bucket:
                continue
            win_rate = sum(r.actual_home_win for r in bucket) / len(bucket)
            range_str = (
                f"[{lo:.0f}, {hi:.0f}]"
                if lo != -np.inf and hi != np.inf
                else (f"(−∞, {hi:.0f}]" if lo == -np.inf else f"({lo:.0f}, +∞)")
            )
            print(f"{labels[i]:<15} {range_str:<28} {len(bucket):>4} {win_rate:>14.4f}")

        # Overall correlation
        wins = np.array([r.actual_home_win for r in valid])
        corr = float(np.corrcoef(psd_values, wins)[0, 1])
        print(f"\nPearson r(player_strength_diff, home_win): {corr:+.4f}")

        if corr > 0:
            print(
                "→ Positive correlation: higher PSD (home stronger) → more home wins."
                " CORRECT direction."
            )
        else:
            print("→ NEGATIVE correlation: PSD is ANTI-correlated with home wins. CHECK SIGN.")

    # -----------------------------------------------------------------------
    # Q2: Pseudo-PDP (bucket table as proxy)
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Q2 — Pseudo Partial Dependence (Quintile Bucket Table as Proxy)")
    print("=" * 70)
    print("""
Without a saved production model, we can't compute a true PDP.
The quintile table above serves as a proxy: it shows the marginal relationship
between PSD and home win rate, averaging over the real data distribution.

Interpretation:
- A monotone increasing pattern from Q1→Q5 confirms PSD is directionally valid.
- Non-monotone or negative slope indicates noise or sign inversion.
""")

    # -----------------------------------------------------------------------
    # Q3: Magnitude check — |PSD| > 500
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Q3 — Magnitude Check: |player_strength_diff| > 500")
    print("=" * 70)
    print("""
Large absolute values of PSD indicate that one team's composite lineup strength
greatly exceeds the other. Values > 500 Elo points are extreme and worth checking:
- Could indicate uncapped player ratings diverging over many seasons.
- Could indicate a bug where default ratings are summed asymmetrically.
""")

    if len(valid) > 0:
        extreme = [r for r in valid if abs(r.player_strength_diff) > 500]
        extreme_frac = len(extreme) / len(valid)
        print(f"Rows with |PSD| > 500: {len(extreme)} / {len(valid)} = {extreme_frac:.1%}")

        thresholds = [100, 200, 500, 1000, 2000]
        print(f"\n{'|PSD| threshold':<20} {'count':>8} {'fraction':>10}")
        print("-" * 42)
        for threshold in thresholds:
            count = sum(1 for r in valid if abs(r.player_strength_diff) > threshold)
            print(f"{threshold:<20} {count:>8} {count / len(valid):>10.1%}")

        if extreme_frac > 0.10:
            print(
                f"\n⚠ WARNING: {extreme_frac:.1%} of rows have |PSD| > 500"
                " — consider capping the feature."
            )
        elif extreme_frac > 0.01:
            print(
                f"\n→ {extreme_frac:.1%} of rows have |PSD| > 500"
                " — within acceptable range (< 10%)."
            )
        else:
            print(
                f"\n✓ Only {extreme_frac:.1%} of rows have |PSD| > 500"
                " — feature scale is healthy."
            )

    # -----------------------------------------------------------------------
    # Q4: Market comparison — disagreement analysis
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Q4 — Market Comparison: Disagreements Between PSD and Odds")
    print("=" * 70)
    print("""
When player_strength_diff direction (PSD > 0 → home stronger)
disagrees with odds_home_win_prob direction (< 0.5 → market favours away),
how often does each get it right?

This quantifies whether PSD adds information beyond what the market knows.
If PSD is right more often than odds in disagreement cases, it captures
roster signal the market misses.
""")

    # Valid rows with both PSD and odds
    both_valid = [r for r in rows if r.missing_psd == 0.0 and r.missing_odds == 0.0]
    print(f"Rows with both PSD and Odds valid: {len(both_valid)}")

    if len(both_valid) > 0:
        # PSD prediction: >0 → home, <0 → away, =0 → abstain
        # Odds prediction: >0.5 → home, <0.5 → away

        agree = [
            r
            for r in both_valid
            if (r.player_strength_diff > 0 and r.odds_home_win_prob > 0.5)
            or (r.player_strength_diff < 0 and r.odds_home_win_prob < 0.5)
        ]
        disagree = [
            r
            for r in both_valid
            if (r.player_strength_diff > 0 and r.odds_home_win_prob < 0.5)
            or (r.player_strength_diff < 0 and r.odds_home_win_prob > 0.5)
        ]
        neither = [
            r for r in both_valid if r.player_strength_diff == 0.0 or r.odds_home_win_prob == 0.5
        ]

        print(f"\nAgreement cases: {len(agree)} ({len(agree) / len(both_valid):.1%})")
        print(f"Disagreement cases: {len(disagree)} ({len(disagree) / len(both_valid):.1%})")
        print(f"Neither/zero cases: {len(neither)} ({len(neither) / len(both_valid):.1%})")

        if agree:
            agree_win = sum(r.actual_home_win for r in agree) / len(agree)
            print(f"\nAgreement case accuracy (both predict same): {agree_win:.4f}")

        if disagree:
            # In disagreement cases, was PSD right?
            psd_right = sum(
                1
                for r in disagree
                if (r.player_strength_diff > 0 and r.actual_home_win == 1)
                or (r.player_strength_diff < 0 and r.actual_home_win == 0)
            )
            odds_right = sum(
                1
                for r in disagree
                if (r.odds_home_win_prob > 0.5 and r.actual_home_win == 1)
                or (r.odds_home_win_prob < 0.5 and r.actual_home_win == 0)
            )
            n = len(disagree)
            print(f"\nIn {n} disagreement cases:")
            print(f"  PSD was right:  {psd_right}/{n} = {psd_right / n:.4f}")
            print(f"  Odds was right: {odds_right}/{n} = {odds_right / n:.4f}")

            if psd_right > odds_right:
                print(f"\n→ PSD wins disagreements ({psd_right / n:.1%} vs {odds_right / n:.1%})")
                print("  PSD captures roster signal the market may under-react to.")
            elif odds_right > psd_right:
                print(
                    f"\n→ Market wins disagreements ({odds_right / n:.1%} vs {psd_right / n:.1%})"
                )
                print("  Odds encodes better/more-timely roster signal than PSD.")
            else:
                print(
                    f"\n→ Tied ({psd_right / n:.1%} each)"
                    " — PSD adds no edge over market in disagreements."
                )

    # -----------------------------------------------------------------------
    # Summary / recommendation
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 70)

    if len(valid) > 0:
        corr = float(
            np.corrcoef(
                np.array([r.player_strength_diff for r in valid]),
                np.array([r.actual_home_win for r in valid]),
            )[0, 1]
        )
        extreme_frac = sum(1 for r in valid if abs(r.player_strength_diff) > 500) / len(valid)

        print("\nKey numbers:")
        print(f"  Correlation (PSD vs home win): {corr:+.4f}")
        print(f"  Fraction |PSD| > 500:          {extreme_frac:.1%}")
        print(f"  Valid PSD coverage:             {len(valid) / len(rows):.1%}")

        print()
        if corr > 0.02:
            direction_ok = True
            print("✓ PSD is directionally correct (positive correlation with home wins).")
        elif corr > 0:
            direction_ok = True
            print("≈ PSD has weak positive correlation — directionally correct but noisy.")
        else:
            direction_ok = False
            print("✗ PSD has NEGATIVE correlation — check feature sign or construction.")

        if extreme_frac > 0.10:
            print("⚠ Many extreme values — consider capping at ±500 or normalizing.")
        else:
            print("✓ PSD magnitude distribution is healthy.")

        print()
        print("RECOMMENDATION:")
        if direction_ok and extreme_frac <= 0.10:
            print("  Keep player_strength_diff as-is.")
            print(
                "  The feature has correct directional signal with healthy magnitude distribution."
            )
            print("  For POSITION_WEIGHTS, see the sweep results (issue #159) —")
            print("  flat weights slightly improve XGBoost, but expert weights are defensible.")
        elif direction_ok and extreme_frac > 0.10:
            print("  Consider capping: clip player_strength_diff to ±500 or ±1000.")
            print("  Directional signal is valid but extreme outliers may be hurting calibration.")
        else:
            print("  Investigate sign of the feature — negative correlation is unexpected.")
            print("  Check FeatureBuilder._player_strength() and PlayerRatings.composite().")


if __name__ == "__main__":
    main()
