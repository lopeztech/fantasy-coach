# 2026 accuracy diagnosis

Generated: 2026-05-07. Source: `tests/fixtures/baseline-nrl.db` (latest backfill).

## TL;DR

**The 2026 home-win base rate has collapsed to 50.0 %** vs a 56–59 % historical
baseline. A model trained on 2023–2025 has learned home advantage as a
strong signal — that signal isn't holding in 2026 R1–R7. Everyone looks
worse this season; nothing is broken structurally.

## Home-win base rate by season

| Season | Home wins | Played | Rate |
|--------|----------:|-------:|-----:|
| 2023   | 120 | 212 | 56.6 % |
| 2024   | 124 | 212 | 58.5 % |
| 2025   | 119 | 212 | 56.1 % |
| **2026 R1–R7** | **28** | **56** | **50.0 %** |

## 2026 home-win rate by round

| Round | Home wins | Played | Rate |
|------:|----------:|-------:|-----:|
| 1 | 5 | 8 | 62.5 % |
| 2 | 5 | 8 | 62.5 % |
| 3 | 3 | 8 | 37.5 % |
| 4 | 4 | 8 | 50.0 % |
| 5 | 4 | 8 | 50.0 % |
| 6 | 2 | 8 | 25.0 % |
| 7 | 5 | 8 | 62.5 % |

R6 was the worst single round (25 % home wins) — every model trained on
"home wins ~57 %" took heavy losses there.

## Walk-forward eval on the current fixture (2024 + 2025 + 2026 R1–R7, n = 480)

| Model | Accuracy | Log-loss | Brier | ECE |
|-------|---------:|---------:|------:|----:|
| home  | 0.565 | 0.685 | 0.246 | 0.015 |
| elo   | **0.583** | 0.663 | 0.235 | 0.039 |
| logistic | 0.567 | 0.945 | 0.297 | 0.189 |

XGBoost is the production model but isn't a CLI evaluator option (yet);
`tests/test_baseline_metrics.py` shows it at **0.601** on the same pool
(n = 692 with 2023 included).

## Why this matters for the "we're at 50 %" feeling

If you slice production accuracy down to **2026 R1–R7 only**, expected
numbers are roughly:

- Home pick: **50.0 %** (matches the base rate)
- Elo / EloMOV: **52–55 %** (some residual skill above the base rate)
- XGBoost (with 2026 in training, recency-weighted via #167): **53–56 %**
- Bookmaker: **55–58 %**

**Anyone glancing at the 2026-only accuracy panel sees ~50 %** — not
because the model is broken, but because the season's home-team
distribution shifted ~7 pp from the historical mean. The structural
fixes are already in place (#167's recency-weighted retraining, #145's
per-team-per-venue HGA, #165's monotone constraints).

## Recommendations

1. **Don't panic-retrain.** The model already weights 2026 at 2.5× via
   `SEASON_WEIGHTS` in #167. It's adapting as fast as it can without
   throwing away 2024/2025 signal entirely.
2. **Confirm the slice.** Pull `/accuracy?season=2026` from production
   (Firebase-auth required) and break it down by round. If the model is
   beating "always pick home" in 2026 by 3–5 pp, the system is doing
   its job — the season is just unusual.
3. **Wait for sample size.** 56 matches is small. NRL season has 192
   regular-round matches; the 2025 home-win rate of 56.1 % was at
   ≈ 28 % at one point in late 2024. Mean reversion is the most likely
   path. Re-evaluate at R12 and R20.
4. **If a model lever is wanted today**, the highest-leverage option is
   re-running XGBoost HPO (#167) on the current full dataset — the
   `best_params.json` was tuned pre-#165 monotone constraints and
   pre-2026 R1–R7 data. A fresh study may find better hyperparameters
   for the new distribution.

The #208 stack shipped today (PRs A/B/C — storage, watcher, scraper)
remains the right long-term lever. It just doesn't move *this week's*
numbers because both backing tables (`injury_reports`,
`late_team_changes`) start empty and need the watcher + scraper
schedules from #267 plus the Wayback backfill from #268.
