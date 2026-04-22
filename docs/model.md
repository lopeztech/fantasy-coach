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
| `key_absence_diff` | Position-weighted count of this team's "regular" starters missing from the current XIII, home minus away. See "Position weighting" below. | A team missing its halfback or hooker measurably underperforms — a far bigger deal than missing a bench forward. |
| `form_diff_pf_adjusted` | Rolling-5 average of (home PF − opponent's rolling-10 PA baseline) minus the same for away. Opponent baseline is pre-match state only. | Strips out opponent quality: "points scored above what this opponent usually concedes". Kept alongside raw `form_diff_pf`. |
| `form_diff_pa_adjusted` | Rolling-5 average of (home PA − opponent's rolling-10 PF baseline) minus the same for away. | Strips out opponent quality: "points conceded relative to what this opponent usually scores". Kept alongside raw `form_diff_pa`. |

### Position weighting (#27)

`feature_engineering.POSITION_WEIGHTS` assigns per-position importance used
by `key_absence_diff`. The ratios are expert-prior, informed by consensus
rugby-league analytics that primary playmakers (7, 9) and last-line defenders
(1) are the highest-leverage positions on the field; exact values are
low-stakes because the logistic coefficient normalises scale, but the
*ratios* matter.

| Position | Weight | Why |
|----------|--------|-----|
| Halfback (7) | 3.0 | Primary playmaker — sets attacking shape, controls kicks, most irreplaceable. |
| Hooker (9) | 2.5 | Dummy-half distribution + middle defensive reads. |
| Fullback (1) | 2.5 | Last line of defence + kick-return / counter-attack engine. |
| Five-Eighth (6) | 2.0 | Secondary playmaker, often carries the running game. |
| Lock (13) | 1.5 | Middle forward engine, frequently team captain / leader. |
| Centre (3, 4) | 1.5 | Defensive reads + attacking shape on the edges. |
| 2nd Row (11, 12) | 1.2 | Middle forward workload; more interchangeable than locks. |
| Prop (8, 10) | 1.0 | Rotated role — bench cover is plentiful. |
| Winger (2, 5) | 1.0 | Impactful on the day but replaceable across rounds. |
| Interchange (14–17) | 0.5 | Bench is inherently rotation-heavy; less signal per change. |

"Regular starter" = a player who started in ≥ 2 of the team's last 5
completed matches (`KEY_ABSENCE_REGULAR_MIN_STARTS` / `KEY_ABSENCE_WINDOW`).
Each regular carries their *most common* starting position in that window;
that's the position the weight table looks up. Feature returns `0.0` before
a team has enough history (first few rounds) and when the current scrape
has no `is_on_field` flag (pre-team-list-drop — no signal rather than
false signal).

### Ablation notes — key-absence feature (#27)

Walk-forward evaluation on the 2024+2025 refreshed DB (424 predictions),
comparing logistic with/without the `key_absence_diff` column using the
same DB state (column zeroed in the "without" run rather than physically
dropped, so FEATURE_NAMES order is stable):

| Metric   | Without | With (position-weighted) | Δ vs without |
|----------|--------:|-------------------------:|-------------:|
| accuracy | 0.5401  | 0.5519                   | **+0.0118** |
| log_loss | 0.7831  | 0.7965                   | +0.0133 (worse) |
| brier    | 0.2710  | 0.2740                   | +0.0030 (worse) |

**Result: mixed tradeoff — accuracy up, calibration down.** The feature
converts some close calls into correct picks (~5 additional correct out of
424) but when the model is wrong, it's more wrong — the classic
"bolder-but-spikier" signature of a high-coefficient binary-ish feature.

Coefficient inspection on the retrained model puts `key_absence_diff` at
rank #4 in magnitude (−0.109) with the expected negative sign — the model
*did* learn a real signal from the feature, it just spends that signal on
being more decisive.

A secondary ablation with **flat weights** (all positions = 1.0, i.e. the
feature degenerates to "count of missing regular starters") gave:

| Metric   | Without | With (flat) | Δ vs without |
|----------|--------:|------------:|-------------:|
| accuracy | 0.5401  | 0.5495      | +0.0094 |
| log_loss | 0.7831  | 0.7928      | +0.0097 (worse) |
| brier    | 0.2710  | 0.2725      | +0.0015 (worse) |

Flat weights sit on a slightly better point in the accuracy-vs-calibration
tradeoff, but still regress log-loss. The position-weighted scheme is kept
because (a) the issue AC explicitly asks for it, and (b) the extra
accuracy is load-bearing for the SPA's "Pick: X" headline.

**Known limitation / follow-up:** 424 predictions is a small sample, and
walk-forward refits from scratch per round, so early-season rounds — when
every team's "regular XIII" is still stabilising — noise the training
signal. Revisit once we have a second full season of is-on-field data
(currently 2024+2025 only; 2023 would add another 200 matches of warm-up
history). Weight ratio tuning is filed as a follow-up.

### Ablation notes — bookmaker odds feature (#26)

Adds `odds_home_win_prob` (de-vigged market-implied home win probability)
and `missing_odds`. Historical matches are populated via the new
`merge-closing-lines` CLI reading the aussportsbetting.com NRL xlsx; live
matches use the odds already present in the scraped `homeTeam.odds` /
`awayTeam.odds` decimal-odds fields.

Same-DB walk-forward on `baseline-nrl.db` (424 predictions), column masked
to neutral 0.5 + missing-flag in the "without" run:

| Predictor | Metric | Without | With | Δ |
|-----------|--------|--------:|-----:|---:|
| logistic  | accuracy | 0.5519 | **0.5566** | **+0.005** |
| logistic  | log_loss | 0.8026 | **0.8017** | **−0.001** |
| logistic  | brier    | 0.2750 | **0.2735** | **−0.002** |
| **xgboost** | accuracy | 0.5660 | **0.5755** | **+0.009** |
| **xgboost** | log_loss | 0.7551 | **0.7490** | **−0.006** |
| **xgboost** | brier    | 0.2663 | **0.2625** | **−0.004** |

**First feature in this release to lift both models across all three
metrics cleanly.** The odds feature is orthogonal enough to the existing
rating/form signal that even logistic gets small, uniformly-signed
improvements — unlike the #27 / #109 features where multicollinearity
hurt logistic. The model learns the correct (positive) coefficient; on
the retrained artefact, `odds_home_win_prob` has the **largest
coefficient in the entire feature set** (+0.391), narrowly beating
`form_diff_pa` and `form_diff_pa_adjusted`.

Magnitude is small because odds already encode Elo + form + public news,
so adding them on top of those features captures only the *extra* signal
(late money, injury whispers, sharp opinion). The issue's caveat stands:
"if odds become our strongest feature, we're partly predicting the
market" — we now confirm that empirically.

Historical coverage of 77% (373 of 484 completed 2024+2025 matches);
unmatched rows tend to be pre-season, finals, or scheduling date slips
that `canonicalize()` couldn't resolve. Remaining slips are a clean-up
target if we revisit.

### Ablation notes — player strength feature (#109)

The `player_strength_diff` / `missing_player_strength` pair wraps a
per-player Elo-style rating system (see `models/player_ratings.py`) into
the existing linear feature set. Feature value is Σ(rating × position_weight
× bench_factor) for the named XIII + bench, home − away — so a rookie
halfback contributes less than a veteran at the same position, which the
#27 absence feature can't distinguish.

Same-DB walk-forward on `baseline-nrl.db` (424 predictions), column zeroed
in the "without" run:

| Predictor | Metric | Without | With | Δ |
|-----------|--------|--------:|-----:|---:|
| logistic | accuracy | 0.5637 | 0.5519 | −0.012 (worse) |
| logistic | log_loss | 0.7978 | 0.8026 | +0.005 (worse) |
| logistic | brier    | 0.2744 | 0.2750 | +0.001 (flat) |
| **xgboost** | accuracy | 0.5542 | **0.5755** | **+0.021** |
| **xgboost** | log_loss | 0.7776 | **0.7657** | **−0.012** |
| **xgboost** | brier    | 0.2747 | **0.2699** | **−0.005** |

**Logistic regresses slightly; XGBoost wins all three metrics.** Same pattern
as the #27 absence feature, more pronounced: linear models struggle to
combine a quality composite with the independent absence/form signals
(coefficients clash), while tree splits capture the non-linear interactions
("strong lineup + home advantage + good ref" is a different prediction
surface than the linear sum).

Logistic stays the production default for now — switching to XGBoost is a
separate decision (compounding effect over multiple features suggests it's
close to time). Meanwhile, the player_strength contribution is live for
XGBoost via the same feature vector. See `test_baseline_metrics.py` for the
pinned metrics; the multi-metric XGBoost win tightens the case for issue
#25's revisit once the third season lands.

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

## MOV-weighted Elo (EloMOV) — #106

`models.elo_mov.EloMOV` is a drop-in replacement for `Elo` that scales the
K-factor by a margin-of-victory term before each rating update:

```
K_eff = K × ln(|margin| + 1) × (2.2 / (elo_diff × 0.001 + 2.2))
```

- `ln(|margin| + 1)` rewards larger wins with diminishing returns.
- The autocorrelation correction `(2.2 / …)` discounts a blowout when the
  winner was already heavily favoured — a 40-point win over a clear underdog
  earns less credit than a 40-point upset.

### Ablation (#106) — 2024–2025 baseline, 424 predictions

| Model      | Accuracy | Log-loss | Brier  |
|------------|----------|----------|--------|
| Plain Elo  | 0.5943   | 0.6570   | 0.2325 |
| **MOV Elo**| **0.6179** | 0.6578 | 0.2323 |
| Δ          | **+2.36pp** | +0.0008 | −0.0002 |

**Result: promotion gate passes** (≥ 0.5 pp accuracy improvement). MOV Elo
improves accuracy by 2.36 pp with negligible log-loss change — the model
correctly weights blowouts as stronger evidence of team-strength gaps.

**Decision:** `FeatureBuilder` now defaults to `EloMOV` so the `elo_diff`
feature used by logistic regression and XGBoost reflects MOV-adjusted ratings.
Plain `Elo` remains available via `Elo()` for A/B comparisons; `EloPredictor`
still uses it so the standalone Elo walk-forward baseline is unchanged.

### Ablation notes — opponent-adjusted form features (#108)

Walk-forward evaluation on the 2024–2025 baseline DB (424 predictions) adding
`form_diff_pf_adjusted` and `form_diff_pa_adjusted` alongside the raw form
features, with EloMOV as the default rater (combined with #106).

| Metric   | Before #108 (EloMOV only) | After #108 | Δ          |
|----------|--------------------------|------------|------------|
| logistic accuracy | 0.5613          | 0.5637     | +0.24pp    |
| logistic log_loss | 0.7926          | 0.7978     | +0.005 (worse) |
| xgboost accuracy  | 0.5613          | 0.5637     | +0.24pp    |
| xgboost log_loss  | 0.7718          | 0.7687     | −0.003 ✓   |

**Result: small accuracy gain with logistic calibration regression.**
XGBoost log-loss improved. The logistic log_loss regression is consistent
with sparse opponent-history in early rounds — each team only plays 16 unique
opponents over two seasons, so the rolling-10 opponent baseline is often thin.

**Decision: keep both raw and adjusted features.** Signal is expected to
improve as the DB accumulates more history. Raw `form_diff_pf`/`pa` are kept
so the logistic can learn to down-weight the adjusted versions when they are noisy.

### What's deliberately *not* in here

- **Glicko-2** — full rating + RD + volatility. More code for marginal gain
  over MOV Elo; deferred until MOV-weighted Elo proves itself over a second
  full season.
- **Bookmaker odds** — high-signal but not a feature we can train on
  historically (odds drop out of the fixtures payload after kickoff). See
  issues #13 (benchmark vs closing lines) and #26 (live odds feature).
- **Team-list / injury status** — issues #24 (parsing) and #27 (modelling).
- **Player-level stats** — kept out of the baseline. Will come in once
  XGBoost (#25) makes nonlinear interactions worth modelling.
- **Glicko-2** — deferred; MOV-weighted Elo (#106) captures the main gain.
  Revisit if a second season shows MOV Elo plateauing.

## XGBoost model (#25)

An XGBClassifier is available at `fantasy_coach.models.xgboost_model` as an alternative
to logistic regression. It uses the same feature set (see the table above) and is trained with
time-series-aware hyperparameter search (`GridSearchCV` + `TimeSeriesSplit(n_splits=3)`)
over `max_depth ∈ {3, 4, 5}`, `n_estimators ∈ {100, 200}`, `learning_rate ∈ {0.05, 0.1}`.

### Production model: XGBoost (switched 2026-04-22, #136)

The comparison table below is from the pre-#136 state. After the bookmaker-
odds feature (#26) landed, XGBoost's edge over logistic compounded enough
— and logistic's multicollinearity-driven wrong-sign coefficient on
``player_strength_diff`` (#109) became misleading enough on per-feature
attribution — that we flipped the production artefact.

**What changed:** ``artifacts/xgboost.joblib`` is now uploaded to
``gs://fantasy-coach-lcd-models/logistic/latest.joblib`` (the path keeps
the old name for now — renaming is a tiny follow-up but needs a paired
deploy-workflow edit). ``models.loader.load_model`` dispatches by
``model_type`` embedded in the joblib blob, so the same path serves either
model without code changes.

**What stayed:** logistic training still works (``python -m fantasy_coach
train-logistic``); the comparison baseline below is still the source of
truth for ablation reporting; the EXPECTED dict in ``test_baseline_metrics``
pins walk-forward numbers for *both* models so regressions on either side
are caught.

**Contribution attribution:** ``_compute_contributions`` in
``predictions.py`` now dispatches by model type:
- logistic: ``coef × (x − mean) / scale`` (unchanged).
- XGBoost: booster ``predict(pred_contribs=True)`` — returns per-feature
  margin contributions (log-odds for binary classification), drops the
  bias column. Output shape matches logistic so the sentinel filter +
  detail enrichment + UI rendering all work without branching.

### Comparison (2024–2025 walk-forward baseline, 424 predictions)

| Model    | Accuracy | Log-loss | Brier  |
|----------|----------|----------|--------|
| Elo      | 0.5943   | 0.6570   | 0.2325 |
| Logistic | 0.5519   | 0.7965   | 0.2740 |
| XGBoost  | 0.5708   | 0.7708   | 0.2717 |

Numbers refreshed in #27 — the new `key_absence_diff` feature was
*especially* useful for XGBoost (accuracy +2.6pp, 0.5448 → 0.5708, biggest
absolute jump of any model) because tree splits can capture position-specific
thresholds the logistic can't. XGBoost now beats logistic on accuracy by
1.9pp and on log-loss by 0.026.

**Decision: keep logistic as default** for now — Elo still owns log-loss
(0.66 vs XGBoost 0.77), and the SPA's "Pick: X" headline is accuracy-facing
where Elo also still wins. Worth re-evaluating once a third season of
backfilled data lands (would bring the walk-forward sample past ~600
predictions, where gradient boosting typically starts to pull ahead).

The XGBoost model is serialised with the same joblib interface as logistic
(`save_model` / `load_model`), keyed by `"model_type": "xgboost"`. The prediction
API can be switched by swapping the artefact path in config.

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

## Skellam margin model — #110

Models each team's score as an independent Poisson process (λ_home, λ_away),
fit via Poisson GLM with log-link using the same `FEATURE_NAMES` feature set.
The score difference (home − away) follows a Skellam(λ_home, λ_away)
distribution, giving three coherent outputs from a single model:

| Output | Description |
|---|---|
| `home_win_prob` | P(margin > 0) — sum of Skellam PMF over margins 1..80 |
| `predicted_margin` | E[home_score − away_score] = λ_home − λ_away |
| `margin_ci_95` | (lo, hi) covering 95 % of the PMF mass at 2.5/97.5 pct |

Feature convention: the home model uses features as-is (home − away
differences); the away model sees negated features so positive values
consistently mean "better away team". Both models share a `StandardScaler`
pre-processing stage and are regularised with L2 penalty α = 200 — strong
regularisation is needed because Poisson GLM with log-link can extrapolate
to near-0 / near-1 win probabilities for extreme feature rows.

### Walk-forward ablation — 2024–2025 baseline, 424 predictions

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Home pick | 0.5731 | 0.6835 | 0.2452 |
| Elo (plain) | 0.5943 | 0.6570 | 0.2325 |
| EloMOV | **0.6179** | 0.6578 | **0.2323** |
| Logistic | 0.5637 | 0.7978 | 0.2744 |
| XGBoost | 0.5637 | 0.7687 | 0.2720 |
| **Skellam** | 0.5684 | **0.7110** | 0.2534 |

**Observations:**
- Skellam improves log_loss and Brier vs both logistic and XGBoost, indicating
  better probability calibration — the distribution-level training objective
  (mean Poisson deviance) avoids the "push probabilities toward 0/1" tendency
  of discriminative classifiers.
- Accuracy (0.5684) is slightly above logistic (0.5637) but well below EloMOV
  (0.6179); EloMOV remains the best single model.
- The predicted margin output (`predicted_margin = λ_home − λ_away`) is a
  purely additive UI feature — existing API clients that only read
  `homeWinProbability` are unaffected.

**Decision:** Skellam is added as a secondary model. It is not promoted to
replace EloMOV as the ensemble's primary signal; the margin and CI outputs
are surfaced as optional fields on `PredictionOut` for display purposes only.
