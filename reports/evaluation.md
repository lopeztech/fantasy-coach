# Model evaluation

Generated: 2026-04-21
Seasons: 2024, 2025 (213 matches each, 426 total; 424 after dropping 2 draws)
Walk-forward: for each round, train on every prior completed match, predict that round, score against actuals.
Bookmaker source: [aussportsbetting.com NRL xlsx](https://www.aussportsbetting.com/historical_data/nrl.xlsx) (2,888 historical lines; 329 of our 424 scored matches were priced).

## Headline

| Model     | Accuracy | Log loss | Brier  | n   |
|-----------|---------:|---------:|-------:|----:|
| home      | 0.575    | 0.683    | 0.245  | 329 |
| elo       | 0.608    | 0.650    | 0.229  | 329 |
| logistic  | 0.581    | 0.686    | 0.244  | 329 |
| **bookmaker** | **0.614** | **0.642** | **0.225** | 329 |

Numbers above are **on the bookmaker-priced subset** (n=329). This is the only honest like-for-like comparison — scoring our models on matches the bookmaker didn't price would inflate or deflate the gap depending on which matches were missing.

## Reading the result

- **Bookmaker is the ceiling, as expected.** It's the best model on every metric. The market encodes everything public knowledge can see, including team lists, weather, and late money.
- **Elo is right behind it.** 0.608 vs 0.614 accuracy — within a coin-flip of matching the market on a 329-game sample. Log loss and Brier are also very close. For a 200-line algorithm with one tunable parameter (K) it's a strong baseline.
- **Logistic isn't earning its complexity.** 0.581 accuracy is barely above always-pick-home (0.575), and its calibration (log loss 0.686, Brier 0.244) is *worse* than home — meaning its probabilities are less honest than a flat 0.55. With only ~424 training examples and walk-forward retraining (so early rounds train on tens of matches), the rolling form / rest / h2h features aren't pulling weight yet.

## Implications for #25 (XGBoost)

Don't reach for a non-linear model to beat Elo here. The signal-to-noise ratio at 200 matches/season is too low for tree-based models to differentiate themselves from a regularised linear baseline. Either feed the model more data (more seasons, player-level features) or accept that Elo + bookmaker is the right ensemble until something materially new shows up.

## Coverage note

95 of the 424 scored matches (22 %) weren't in the bookmaker dataset — typically lower-profile early-round and mid-season fixtures. The aussportsbetting xlsx covers a subset of NRL fixtures and isn't exhaustive; this isn't a join bug, and `BookmakerPredictor.missing_match_ids` surfaces the gap explicitly.

## Methodology

### Walk-forward, not random split

Every model is fit on history strictly before the round being predicted, then refit one round later with the additional data. No future information ever leaks into a prediction.

### Draws dropped

NRL had 2 draws across 2024–2025. They're dropped from scoring because the binary metrics expect `{0, 1}` outcomes; including them as 0.5 outcomes biases log loss and Brier in a way that obscures real ranking differences.

### Bookmaker fallback

Matches that aren't in the closing-lines dataset get a 0.55 (home prior) prediction from the `BookmakerPredictor`. Those predictions are excluded from the table above so the comparison is on the same 329-match subset for every model.

### Reproducing

```bash
uv run python -m fantasy_coach backfill --season 2024 --db data/nrl.db
uv run python -m fantasy_coach backfill --season 2025 --db data/nrl.db
curl -L -o data/nrl-odds.xlsx https://www.aussportsbetting.com/historical_data/nrl.xlsx
uv run python -m fantasy_coach evaluate \
    --model home --model elo --model logistic --model bookmaker \
    --closing-lines data/nrl-odds.xlsx \
    --seasons 2024,2025 \
    --report reports/evaluation.md
```

The CLI's report writer doesn't yet split full-set vs priced-subset — this file was hand-curated from the same numbers. A follow-up to `evaluation/report.py` should add that view so the report regenerates faithfully from the CLI alone.
