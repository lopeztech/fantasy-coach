# Prediction model

Two baselines live in `fantasy_coach.models`:

- **Elo** (`models.elo.Elo`) — pairwise rating, only knows prior wins/losses.
- **Logistic regression** (`models.logistic`) — blends Elo with rolling form,
  rest, and head-to-head context. Feature pipeline lives in
  `feature_engineering.build_training_frame`.

Both consume the `MatchRow` rows produced by `features.extract_match_features`
and stored via `storage.SQLiteRepository`.

## Logistic regression features

All features are home-minus-away unless the name says otherwise. Computed for
each match using only matches whose `start_time` precedes it — no leakage.

| Feature           | Definition                                                                                              | Why                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `elo_diff`        | `home_elo + home_advantage − away_elo`, evaluated against the rolling Elo book at kickoff.              | Bakes in long-run team strength + the historical home edge.               |
| `form_diff_pf`    | Rolling-5 average **points scored** by home minus away.                                                 | Recent attacking form. Catches injury-driven slumps faster than Elo.      |
| `form_diff_pa`    | Rolling-5 average **points conceded** by home minus away.                                               | Recent defensive form. Pairs with `form_diff_pf` to triangulate strength. |
| `days_rest_diff`  | Days between this match and the team's previous match, home minus away. First-of-season clamped to 14.  | Short-week games (4–5 days) measurably underperform 7-day games in the NRL. |
| `h2h_recent_diff` | Average score margin in the last 3 head-to-head matchups, from the home team's perspective.             | Stylistic mismatches show up in repeated head-to-heads (e.g. forward packs that consistently beat smaller teams). |
| `is_home_field`   | Constant `1.0`.                                                                                         | Lets the intercept absorb a stable home-field bias rather than letting it leak into other features. |
| `travel_km_diff`  | Great-circle km the home team travelled from their last venue minus the same for the away team.         | Long-haul trips (e.g. Perth-to-Brisbane) impose measurable fatigue when combined with a short week. |
| `timezone_delta_diff` | Absolute timezone-shift hours between the team's last venue and the current venue, home minus away. | Eastward shifts disrupt circadian rhythms more than equivalent westward shifts. |
| `back_to_back_short_week_diff` | `+1` if home team has `rest < 6 days AND travel > 1 000 km`; `-1` for away; `0` otherwise. | Captures the specific "brutal" scenario; orthogonal to `travel_km_diff` and `days_rest_diff`. |
| `is_wet`          | `1.0` if weather is wet/rainy (from structured NRL weather block or keyword match); else `0.0`.         | Wet conditions suppress high-scoring, favouring defensively disciplined teams. |
| `wind_kph`        | Wind speed in km/h; `0.0` when data absent.                                                             | Strong wind suppresses kicking game and scoring. |
| `temperature_c`   | Temperature in Celsius; `0.0` when data absent.                                                         | Extreme heat/cold affects player endurance and injury rates. |
| `missing_weather` | `1.0` when the weather block is absent (pre-2026 historical data).                                      | Explicit missing-data flag so the model learns a separate intercept rather than imputing zeros. |
| `venue_avg_total_points` | Rolling-10 average total points at this venue (history-only, no current match).               | Some grounds (windy, small) consistently produce tighter games regardless of teams. |
| `venue_home_win_rate` | Rolling-20 home win rate at this venue; defaults to 0.5 before any history.                       | Captures venue-specific home advantage — fortress grounds vs neutral-feeling venues. |
| `ref_avg_total_points` | Rolling-20 average total points for matches officiated by this referee; shrunk toward league mean for < 10 prior matches. | Some referees blow more penalties/restarts affecting scoring pace. |
| `ref_home_penalty_diff` | Rolling-20 average (home − away) Penalties Conceded for this referee; `0.0` when unavailable. | Captures whether a referee tends to penalise the home or away team more often. |
| `missing_referee` | `1.0` when referee ID is absent (upcoming fixtures or pre-2026 data).                                   | Explicit missing-data flag. |

### Ablation notes — referee features (#57)

Walk-forward evaluation on the 2024–2025 baseline DB (424 predictions) after
adding `ref_avg_total_points`, `ref_home_penalty_diff`, and `missing_referee`:

| Metric   | Before #57 | After #57 | Δ      |
|----------|------------|-----------|--------|
| accuracy | 0.5637     | 0.5660    | +0.23pp |
| log_loss | 0.7636     | 0.7640    | −0.0004 |
| brier    | 0.2655     | 0.2654    | +0.0001 |

**Result: no meaningful signal** — the change is within noise. The root cause is
that the 2024–2025 baseline DB was built at schema v1 (before referee IDs were
extracted), so `referee_id = NULL` for every historical match; all predictions
fall back to `missing_referee = 1.0` and the league-mean prior for
`ref_avg_total_points`. The features are structurally correct and will accumulate
signal as new rounds are ingested with referee data. They remain active in the
feature vector; revisit with at least one full season of referee-annotated matches.

### What's deliberately *not* in here

- **Margin of victory in Elo** — kept simple to make the Elo baseline a
  clean comparison point. If logistic noticeably beats Elo, MOV-weighted
  Elo is a follow-up.
- **Bookmaker odds** — high-signal but not a feature we can train on
  historically (odds drop out of the fixtures payload after kickoff). See
  issues #13 (benchmark vs closing lines) and #26 (live odds feature).
- **Team-list / injury status** — issues #24 (parsing) and #27 (modelling).
- **Player-level stats** — kept out of the baseline. Will come in once
  XGBoost (#25) makes nonlinear interactions worth modelling.

## XGBoost model (#25)

An XGBClassifier is available at `fantasy_coach.models.xgboost_model` as an alternative
to logistic regression. It uses the same 18-feature input and is trained with
time-series-aware hyperparameter search (`GridSearchCV` + `TimeSeriesSplit(n_splits=3)`)
over `max_depth ∈ {3, 4, 5}`, `n_estimators ∈ {100, 200}`, `learning_rate ∈ {0.05, 0.1}`.

### Comparison (2024–2025 walk-forward baseline, 424 predictions)

| Model    | Accuracy | Log-loss | Brier  |
|----------|----------|----------|--------|
| Elo      | 0.5943   | 0.6570   | 0.2325 |
| Logistic | 0.5660   | 0.7640   | 0.2654 |
| XGBoost  | 0.5448   | 0.7599   | 0.2721 |

**Decision: keep logistic as default.** XGBoost's log-loss improvement is 0.41pp,
below the 1-point threshold stated in the issue AC. On this 2-season baseline, XGBoost
is worse on accuracy and brier. Both models suffer from limited referee data (see
referee ablation above). Recommendation: re-evaluate once 3+ seasons of data — including
referee and injury features — are backfilled; gradient boosting typically needs ≥ 5k rows
to clearly outperform a linear baseline.

The XGBoost model is serialised with the same joblib interface as logistic
(`save_model` / `load_model`), keyed by `"model_type": "xgboost"`. The prediction
API can be switched by swapping the artefact path in config.

## Ensemble (#56)

`fantasy_coach.models.ensemble` blends Elo, logistic, and XGBoost via a stacked
logistic-regression meta-learner (or optionally a simplex-constrained weighted average).

**Training protocol (no future-leak):**
1. First 75 % of completed history → train the three base models.
2. Remaining 25 % → generate held-out probabilities; fit the meta-learner on these.
3. Re-fit all base models on 100 % of history for inference.

**Kill switch:** if the ensemble's cross-validated log-loss does not beat the best
single base model by 0.5 pp, `predict_ensemble` falls back to that best base model.

### Comparison (2024–2025 walk-forward baseline, 424 predictions)

| Model    | Accuracy | Log-loss | Brier  |
|----------|----------|----------|--------|
| Elo      | 0.5943   | 0.6570   | 0.2325 |
| Logistic | 0.5660   | 0.7640   | 0.2654 |
| XGBoost  | 0.5448   | 0.7599   | 0.2721 |
| Ensemble | 0.6014   | 0.6782   | 0.2390 |

**Result: ensemble improves accuracy (+0.71 pp vs Elo) but worsens log-loss (+2.12 pp).** The
kill switch fires for most early rounds because Elo's well-calibrated probabilities dominate.
When the kill switch is inactive (larger history, more data), the ensemble finds accuracy gains
by combining logistic/XGBoost signal — but those models' miscalibrated probabilities push
log-loss up. Net effect: better discrimination, worse calibration than Elo alone.

**Recommendation:** calibrate the ensemble output (Platt or isotonic, as in `models.calibration`)
before using as default. Until calibrated ensemble log-loss beats Elo by ≥ 1 pp, logistic
remains the default. Re-evaluate with ≥ 3 seasons of data and full referee/injury features.

## Train / test split

Time-ordered, never random. The most recent 20 % of completed matches form
the test set; the rest is training. This mirrors how the model is actually
used: predict the next round given everything before.

## Artefact format

`save_model` writes a joblib blob containing:

```
{"pipeline": Pipeline(StandardScaler → LogisticRegression),
 "feature_names": (..., ...)}
```

`load_model` refuses to load if `feature_names` doesn't match the current
`feature_engineering.FEATURE_NAMES` — schema drift in the feature list
must force a retrain rather than silently mis-aligning columns.

## CLI

```
python -m fantasy_coach train-logistic \
    --season 2024 --season 2025 \
    --db data/nrl.db \
    --out artifacts/logistic.joblib
```

Multiple `--season` flags pool matches across seasons before splitting.
