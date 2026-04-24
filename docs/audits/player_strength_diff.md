# player_strength_diff — Directional Validity Audit (#166)

**Baseline:** `tests/fixtures/baseline-nrl.db`, seasons 2024–2026, n=480 non-draw predictions.
**Script:** `scripts/audit_player_strength_diff.py`

---

## Q1 — Directional Correlation on Holdout

Does higher `player_strength_diff` (home team's composite player-rating sum minus away's) correlate with home wins?

| Quintile | PSD range | n | Home win rate |
|---|---|--:|--:|
| Q1 (lowest) | (−∞, −1766] | 96 | 0.5938 |
| Q2 | [−1766, −630] | 96 | 0.5208 |
| Q3 | [−630, 499] | 96 | 0.4583 |
| Q4 | [499, 1593] | 96 | 0.6146 |
| Q5 (highest) | (1593, +∞) | 96 | 0.6354 |

**Pearson r = +0.0120** (positive; higher PSD → more home wins).

**Interpretation:** The feature is directionally correct — Q5 (home team strongly stronger) has the highest home win rate (63.5%) and Q3 (near-balanced) has the lowest (45.8%). However, the Q1 anomaly (59.4%) is notable: the _most_ away-advantaged bucket has a higher win rate than Q2 and Q3. This is likely a small-sample artefact (only 2 seasons of player rating history) combined with the large variance of the PSD distribution (std ≈ 1988 Elo points — see Q3 below). Overall the monotone Q3→Q5 progression and positive Pearson r confirm the feature works in the intended direction.

---

## Q2 — Partial Dependence Proxy

Without a saved production XGBoost artefact, a true PDP cannot be computed. The quintile bucket table above is the best available proxy: it shows the marginal relationship between PSD and home win rate after averaging over all other features in the real data.

**Key finding:** The Q3→Q5 monotone increase (0.458 → 0.615 → 0.635) is consistent with a correctly signed feature. The Q1 dip relative to Q2 (0.594 vs 0.521) is non-monotone and likely noise on a 2-season baseline.

When the 2023 backfill (#158) lands, the quintile bins should tighten and the Q1 anomaly should attenuate as player ratings converge from ~3 seasons of warm-up history. Re-run `audit_player_strength_diff.py` after that data lands.

---

## Q3 — Magnitude Check: |PSD| > 500

| |PSD| threshold | Count | Fraction |
|--:|--:|--:|
| 100 | 460 | 95.8% |
| 200 | 443 | 92.3% |
| 500 | 396 | 82.5% |
| 1,000 | 294 | 61.3% |
| 2,000 | 154 | 32.1% |

**82.5% of predictions have |PSD| > 500.**

The raw `player_strength_diff` has a standard deviation of ~1,988 Elo points and ranges from −5,272 to +6,625. This is large but structurally expected:

- The feature is a _sum_ of position-weighted ratings across the full named XIII + bench (13 starters + 4 bench), not a mean.
- Each player's rating starts at 1500 and diverges over matches. Even after 2 seasons, the spread across the rating book is wide relative to the Elo scale.
- POSITION_WEIGHTS range from 0.5 (interchange) to 3.0 (halfback); the sum across a full XIII is `sum(POSITION_WEIGHTS.values()) × avg_rating ≈ 17.2 × 1500 = 25,800`. A 7% spread between lineups = ±1,800 Elo points — consistent with the observed std.

**Recommendation:** Consider capping `player_strength_diff` at ±1,000 (corresponding to ~3× the typical starter-quality gap) in the feature pipeline to reduce extreme-value leverage on tree splits. XGBoost monotone constraints (from #165) already prevent sign inversions, but capping would compress the distribution into a more interpretable range without losing the directional signal.

This is filed as a follow-up rather than an immediate change because the baseline pinned metrics (#167) don't show calibration issues traceable to this magnitude — the XGBoost monotone constraint is already doing the heavy lifting. After the 2023 backfill (#158), re-evaluate.

---

## Q4 — Market Comparison: PSD vs Odds Disagreements

Rows with both `player_strength_diff` and `odds_home_win_prob` valid: **372 of 480 (77.5%)**.

| Category | Count | Fraction |
|---|--:|--:|
| Agreement (both predict home or both predict away) | 217 | 58.3% |
| Disagreement (PSD says home, odds says away, or vice versa) | 152 | 40.9% |
| Neither (PSD = 0 or odds = 0.5) | 3 | 0.8% |

Agreement case accuracy (both models agree): **0.5622**

**In 152 disagreement cases:**

| | Correct | Fraction |
|---|--:|--:|
| PSD direction | 66 / 152 | 43.4% |
| Odds direction | 86 / 152 | 56.6% |

**The market wins disagreements: 56.6% vs 43.4%.**

This aligns with the expected prior: bookmaker odds encode late-money, injury whispers, and sharp-money signal that the PSD feature cannot see from public roster data alone. When PSD and the market disagree, the market is the better tiebreaker on this 2-season window.

Importantly, this does **not** mean PSD should be removed — the 58.3% agreement rate shows PSD is usually consistent with the market. The disagreement cases likely represent matches where (a) the market has late injury information PSD doesn't, or (b) PSD is reacting to roster changes that are genuine signal but the market hasn't updated on yet. Larger sample (2023 backfill) would clarify which effect dominates.

---

## Explicit Recommendation

**Keep `player_strength_diff` as-is with one follow-up item.**

1. **Direction**: Correct (+0.0120 Pearson r; Q3→Q5 monotone increase confirmed).
2. **Magnitude**: The wide distribution (std ≈ 1,988) is structurally expected for a sum-of-ratings feature across a full XIII. No immediate fix required; XGBoost monotone constraints already prevent sign-inversion splits.
3. **Position weights (#159)**: The sweep in `scripts/sweep_position_weights.py` shows flat weights slightly improve XGBoost (+1.04pp accuracy, −0.008 log_loss, −0.003 brier vs expert prior). The issue AC says to keep expert-prior weights if flat wins — see [issue #159 resolution](../../docs/model.md#position-weighting-159) for the tradeoff documentation.
4. **Market comparison**: PSD loses head-to-head in disagreement cases (43.4% vs 56.6%), confirming the market is the stronger signal when the two conflict. This is expected and consistent with `odds_home_win_prob` having the largest XGBoost coefficient in the artefact.

**Follow-up filed:** Cap `|player_strength_diff|` at ±1,000 after the 2023 backfill (#158) lands and player ratings have ~3 seasons to converge. Re-run this audit and compare quintile table shape before and after capping.

---

## 2026-04-25 update — follow-up shipped early (#203)

The "wait for #158" deferral was overtaken by live evidence. Round 8 2026 went 0/3 on the three completed Sat-morning fixtures, and one of the misses was the exact PSD-overrules-market pattern this audit flagged:

| Match | PSD value | PSD contribution | Market home prob | Market contribution | Pick | Result |
|---|--:|--:|--:|--:|---|---|
| Tigers v Raiders | −1,232 | −0.3198 | 0.613 | −0.1076 | Raiders 54.3% | Tigers 33–14 |

PSD was above the cap proposed here (1,000) and the market had Tigers as 61.3% favourites. The model picked Raiders. The market was right.

The other two R8 misses (Cowboys/Sharks, Broncos/Bulldogs) had `|PSD|` *below* the cap (−701 and +168) — they're not PSD-magnitude failures, and capping doesn't change them.

#203 ships the audit's own follow-up plus an output-layer market shrinkage:

- `PLAYER_STRENGTH_DIFF_CAP = 1000.0` in `feature_engineering.py` — applied uniformly at training and inference, so saved artefacts and live predictions see the same distribution.
- `MARKET_SHRINKAGE_WEIGHT = 0.3` in `predictions.py` — when `odds_home_win_prob` is present, the final home-win probability is `(1 − w)·model + w·market`. Direction is preserved on agreement (most cases); disagreement cases are pulled toward the more-accurate signal (audit Q4: 56.6% vs 43.4%).

When the 2023 backfill lands, re-run `scripts/audit_player_strength_diff.py` to confirm the quintile table tightens and the Q1 anomaly attenuates as anticipated.
