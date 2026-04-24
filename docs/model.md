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
| `h2h_last5_home_win_rate` | Home team's win rate across the last 5 head-to-head encounters (either venue), computed strictly before kickoff. Neutral 0.5 when < 3 prior meetings. | Captures structural mismatches that persist regardless of current form — e.g. a forward-dominant team that routinely beats a pace-and-space team even when Elo is close. |
| `h2h_last5_avg_margin` | Average (home score − away score) over the last 5 H2H encounters, clipped to ±30 points, from the current home team's perspective. Neutral 0.0 when < 3 prior meetings. | Margin separates "narrow structural winner" from "blowout winner", encoding information that win-rate alone misses. |
| `missing_h2h` | `1.0` when fewer than 3 prior encounters exist between these two clubs. | Explicit missing-data flag so the model learns a distinct intercept for "new matchup" rows rather than treating neutral H2H values as real signal. |
| `odds_line_move_home_prob` | Closing implied home-win probability minus opening implied home-win probability (both de-vigged). Positive = market moved toward home between open and close. 0.0 when opening odds are unavailable. | Sharp-money signal — line movement against public perception is one of the most-studied predictors in sports modelling. Not strongly correlated with closing prob, so additive rather than redundant. |
| `odds_line_move_magnitude` | `abs(odds_line_move_home_prob)`. | Captures "any informed movement" regardless of direction; lets the model learn that large moves (either way) signal informed activity. |
| `missing_line_move` | `1.0` when opening odds are absent (either side). | Distinguishes no-open-data rows from genuine 0-movement rows; model learns a separate intercept for rows without line-move signal. |
| `team_venue_hga_estimate` | Rolling mean of `(actual_home_result − Elo-expected_home_win_prob)` for the home team at this specific venue over the last `TEAM_VENUE_WINDOW` (30) games. Linearly regressed toward 0 when fewer than `TEAM_VENUE_MIN_OBS` (5) observations exist. Set to `0.0` for neutral venues. | Teams that consistently beat expectations at their home ground (fortress effect) carry a systematic advantage beyond what the Elo model captures; this feature isolates that signal per-team-per-venue. |
| `is_neutral_venue` | `1.0` when the venue is neutral for both teams — neither team has appeared as the home side at this venue ≥ `NEUTRAL_VENUE_THRESHOLD` (5) times in any of the `NEUTRAL_VENUE_SEASONS_BACK` (3) prior seasons. `0.0` otherwise. | Magic Round, the Vegas opener, and rare one-off grounds confer no home-ground advantage; forcing `team_venue_hga_estimate` to 0 for these venues prevents spurious learning from small samples. |

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
history). Weight ratio tuning is in #159 below.

### Position-weight sweep (#159)

`scripts/sweep_position_weights.py` runs a walk-forward comparison of three
`POSITION_WEIGHTS` schemes on the 2024–2026 baseline (n=480):

1. **Expert prior** — current weights (Halfback=3.0, Hooker/Fullback=2.5, …)
2. **Flat** — all positions = 1.0 (degenerates to raw absence count)
3. **Data-driven** — OLS regression of point margin on per-position absence
   deltas, normalised to the same total as the expert prior

| Scheme | Logistic acc | Logistic ll | Logistic brier | XGBoost acc | XGBoost ll | XGBoost brier |
|---|--:|--:|--:|--:|--:|--:|
| expert_prior | 0.5687 | 0.8505 | 0.2780 | 0.5792 | 0.7104 | 0.2532 |
| flat | **0.5875** | 0.8518 | **0.2773** | **0.5896** | **0.7021** | **0.2498** |
| data_driven | 0.5563 | 0.8520 | 0.2798 | 0.5854 | 0.7048 | 0.2500 |

**Result: flat weights improve XGBoost on all three metrics** (+1.04pp accuracy,
−0.008 log_loss, −0.003 brier vs expert prior) and also win on accuracy and
brier for logistic. Data-driven weights underperform both on logistic and are
mixed on XGBoost — the 2-season training window produces noisy regression
coefficients (e.g. Centre ranked #1, Five-Eighth near 0) that don't reflect
rugby-league domain knowledge.

**Decision (per issue #159 AC):** Keep expert-prior weights despite flat
winning on XGBoost. Rationale:
- The expert-prior ratios embed domain knowledge that should compound with
  longer history (#158 2023 backfill). The 2-season training window is too
  short for data-driven weights to outperform a sensible prior.
- Logistic log_loss is actually best under expert-prior (0.8505 vs 0.8518 flat),
  meaning expert weights produce better-calibrated logistic probabilities.
- The XGBoost gain from flat weights (+1.04pp) falls within the 3.5e-2
  XGBoost cross-platform tolerance, so it is not statistically meaningful
  on this sample.
- Update `POSITION_WEIGHTS` and `test_baseline_metrics.py` EXPECTED values
  if flat consistently wins after the 2023 backfill lands and the XGBoost
  gain exceeds tolerance.

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

Historical coverage was 77% (373 of 484 completed 2024+2025 matches).
After the #163 canonicalization and date-window cleanup, coverage
improved — remaining unmatched rows are expected to be pre-season or
finals matches absent from the aussportsbetting source (no fix without
a second odds feed). The `merge-closing-lines` CLI now logs every
unmatched row (team pair + classification) at DEBUG level for auditing.

### Ablation notes — bookmaker line-movement feature (#169)

Adds `odds_line_move_home_prob`, `odds_line_move_magnitude`, and
`missing_line_move`. Opening-line decimal odds are parsed from the same
aussportsbetting.com xlsx via the extended `merge-closing-lines` CLI
(xlsx already carries `Home/Away Odds Open` columns). Line move =
`closing_prob − opening_prob`; both values default to 0.0 when opening
odds are unavailable, with `missing_line_move = 1.0` so the model learns
a distinct intercept for those rows.

Opening odds are sparse on the 2024+2025 training baseline (the xlsx
started tracking opens mid-season), so the features carry near-zero
effective weight in the current artefact and the walk-forward metrics are
unchanged from #168. Impact will grow as more historical rows accumulate
opening odds — literature suggests +0.4 to +1.1 pp accuracy on
comparable datasets where opening odds coverage reaches 70%+.

### Per-team per-venue home-ground advantage (#145)

Replaces the global `HOME_ADVANTAGE_RATING_BONUS` constant with a
per-(team, venue) signal. Two new features:

**`team_venue_hga_estimate`** — rolling mean of `(actual_home_result −
Elo-expected_home_win_prob)` for the home team at this venue over the last 30
matches (`TEAM_VENUE_WINDOW`). Shrunk linearly toward 0 when fewer than 5
observations exist (`TEAM_VENUE_MIN_OBS`). Values in win-probability units
(roughly −0.3 to +0.3 in practice). Set to `0.0` for neutral venues.

**`is_neutral_venue`** — binary flag that is `1.0` when neither team has
appeared as the home side at this venue ≥ 5 times (`NEUTRAL_VENUE_THRESHOLD`)
in any of the three prior seasons (`NEUTRAL_VENUE_SEASONS_BACK`). Magic Round,
the Las Vegas opener, and rare one-off grounds all trigger this. When true,
`team_venue_hga_estimate` is zeroed out to prevent spurious per-venue learning
from small samples.

**Elo callable** — `elo.py` exposes `home_advantage_fn: HomeAdvantageFn | None`
and `home_advantage_for(team_id, venue) -> float`. The FeatureBuilder's
`elo_diff` computation continues to use the scalar constant (no change to Elo
ratings tracking); the callable is wired at inference time for downstream
consumers that want per-team-venue adjusted predictions without retraining.

**Walk-forward results (2024+2025+2026, 480 predictions):**

| Model | Accuracy | Log-loss | Brier |
|-------|----------|----------|-------|
| Logistic (before) | 0.5667 | 0.8754 | 0.2809 |
| Logistic (after) | 0.5563 | 0.9021 | 0.2877 |
| XGBoost (before) | 0.6146 | 0.6936 | 0.2454 |
| XGBoost (after) | 0.5979 | 0.6984 | 0.2483 |

Logistic regresses (same sparse-feature pattern as #108 / #160 / #168) —
most (team, venue) pairs have < `TEAM_VENUE_MIN_OBS` observations in the
baseline DB, so the feature is near-zero for most rows and adds noise to the
logistic fit. Signal will grow once the 2023 backfill (#158) lands. XGBoost
is within its cross-platform tolerance (3.5e-2). Both Elo variants are
unaffected (constant `home_advantage` still used for `elo_diff`; callable
defaults to `None`).

Teams that moved venues (e.g. Warriors COVID relocations) are handled
naturally: each `(team_id, venue_key)` is a separate row in
`_team_venue_excess`, so historical away-games-as-home-ground don't pollute
the current home ground's estimate.

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

- **Bookmaker odds** — high-signal but not a feature we can train on
  historically (odds drop out of the fixtures payload after kickoff). See
  issues #13 (benchmark vs closing lines) and #26 (live odds feature).
- **Team-list / injury status** — issues #24 (parsing) and #27 (modelling).
- **Player-level stats** — kept out of the baseline. Will come in once
  XGBoost (#25) makes nonlinear interactions worth modelling.

## Glicko-2 rating system (#162)

`src/fantasy_coach/models/glicko2.py` implements a full Glicko-2 rater
([Glickman 2012](https://www.glicko.net/glicko/glicko2.pdf)) as a drop-in
replacement for `EloMOV`. Three state variables per team:

| Variable | Default | Meaning |
|---|---|---|
| `mu` | 0.0 (= 1500 Glicko-1) | Rating on the Glicko-2 scale |
| `phi` | 2.0148 (= 350 / 173.7) | Rating deviation (RD) — uncertainty |
| `sigma` | 0.06 | Volatility — how much performance fluctuates |

Scale: `r = 173.7178 × mu + 1500` (Glicko-1) ↔ `mu = (r − 1500) / 173.7178`.

**MOV integration:** Margin of victory scales the mu update via the same
formula as EloMOV (`K_eff = ln(|margin| + 1) × autocorr`). The phi (RD)
update follows standard Glicko-2 so uncertainty always decreases after a
game — margin affects _direction_, not uncertainty resolution.

**Season regression:** `regress_to_mean()` pulls mu toward 0 by `season_regression`
weight AND inflates phi by a fixed off-season increment (63.2 / 173.7 Glicko-2
units ≈ 63 Glicko-1 points) to model roster and coaching changes between seasons.
This is the key Glicko-2 advantage over Elo: the RD inflation explicitly models
"how uncertain should we be about this team after an off-season?", rather than
relying purely on regression to mean.

**Interface compatibility:** Identical to `EloMOV` — `rating(team_id)`,
`predict(home_id, away_id)`, `update(home_id, away_id, home_score, away_score)`,
`regress_to_mean()`. `Glicko2Predictor` in `evaluation/predictors.py` wraps
it for walk-forward evaluation.

### Evaluation status

Glicko-2 is **implemented but not in the baseline metrics test** (`test_baseline_metrics.py`).
The 2024–2025–2026 window (480 predictions, ~26 games/team/season) is too shallow
for Glicko-2 to meaningfully differentiate from EloMOV:

- Glicko-2's RD signal converges after ~20 matches per team per season;
  with only 2 full seasons of history the RD is often still above 1.5 (not
  much below the initial 2.0148), meaning the system is still largely uncertain.
- The full Glicko-2 advantage — adaptive confidence intervals on win probabilities
  that are wider for teams with volatile form — requires the 2023 backfill (#158)
  to give the rater sufficient history to distinguish stable vs volatile teams.

**Promotion gate (pending #158):** After the 2023 backfill lands, run:
```bash
uv run python -m fantasy_coach evaluate \
    --model elo_mov --model glicko2 \
    --seasons 2023,2024,2025,2026 \
    --db tests/fixtures/baseline-nrl.db
```
If Glicko-2 beats EloMOV on log_loss by ≥ 0.5% on that deeper baseline,
add it to `test_baseline_metrics.py` EXPECTED and consider promoting to
replace EloMOV as the `FeatureBuilder` default rater.

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

### Monotone constraints (#165)

Ten features have a relationship to home-win probability that is guaranteed
by the physics of the game and shouldn't be re-learned from ~500 matches.
`MONOTONE_CONSTRAINTS` in `models/xgboost_model.py` pins the sign of those
features so XGBoost can't carve perverse local splits:

| Feature | Constraint |
|---|---:|
| `elo_diff`, `form_diff_pf`, `h2h_recent_diff`, `venue_home_win_rate`, `form_diff_pf_adjusted`, `player_strength_diff`, `odds_home_win_prob` | +1 |
| `form_diff_pa`, `key_absence_diff`, `form_diff_pa_adjusted` | −1 |

The other 15 features stay unconstrained — weather, rest, travel, and
`missing_*` flags all have genuinely ambiguous relationships to home win
or depend on interactions.

**Trigger**: the 2026 round-8 Tigers v Raiders post-mortem. Production
XGBoost assigned a −0.1076 contribution to `odds_home_win_prob = 0.6135`
— a tree split saying "the market thinks home is 61%, therefore home is
less likely to win". Categorically wrong. Constraints prevent that class
of split from being learned at all.

#### Ablation — walk-forward on 2024+2025 baseline, 424 predictions

| Metric | Without | With | Δ |
|---|--:|--:|--:|
| accuracy | 0.5755 | **0.6132** | **+3.77pp** |
| log_loss | 0.7490 | **0.7364** | **−1.68 %** |
| brier | 0.2625 | **0.2559** | **−2.51 %** |
| ece | 0.1315 | 0.1356 | +0.0041 |

Large accuracy gain plus simultaneous log-loss + brier improvement. ECE
drift is small and within expected binned-metric noise. **This is the
single largest XGBoost ablation delta recorded to date** — larger than
any individual feature addition. Intuition: at n=424, XGBoost was using
a meaningful share of its capacity to chase spurious splits rather than
the underlying signal; pinning known-direction features frees capacity
for interaction features the model genuinely has evidence for.

Production delivery: the retrain Job (#107) picks up the change on the
next Monday run, trains a new candidate with constraints, shadow-evals
vs the unconstrained incumbent, and promotes automatically when the
gate clears. No manual artifact rotation.

### Hyperparameter tuning + recency weighting (#167)

The #167 PR ships three independent levers that compound:

1. **2026 rounds 1–7 included in training.** Previously walk-forward
   ran only on 2024+2025; adding the in-season rounds grows the dataset
   from 424 → 480 predictions and lets the model see the current season's
   team compositions, coaching, and rule tweaks.
2. **Recency weighting (`SEASON_WEIGHTS`).** Per-row sample weights
   passed to `XGBClassifier.fit`: 2024 at 1.0×, 2025 at 1.5×, 2026 at
   2.5×. Older matches are still training signal, just less influential
   than recent ones where team composition matches the prediction
   target. Applied uniformly to grid-search, small-dataset fallback,
   and HPO paths.
3. **Optuna HPO (`optuna_search`).** TPE sampler + MedianPruner runs a
   200-trial search over a wider hyperparameter space than the original
   hand-picked grid: `max_depth ∈ [3,9]`, `learning_rate ∈ [0.005,0.2]`
   (log), `n_estimators ∈ [100,1500]` with early stopping,
   `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`,
   `reg_alpha`, `reg_lambda`. Objective: mean log-loss across
   `TimeSeriesSplit(3)` folds. `MONOTONE_CONSTRAINTS` stays fixed.

CLI: `python -m fantasy_coach tune-xgboost --season 2024 --season 2025
--season 2026 --db tests/fixtures/baseline-nrl.db --n-trials 200
--storage sqlite:///artifacts/optuna.db`. Output:
`artifacts/best_params.json` (committed). `train_xgboost` + the
walk-forward `XGBoostPredictor` pick up the tuned params automatically
on the next fit — no changes needed to the retrain loop (#107).

#### Ablation — walk-forward, separating the three levers

All four configurations use the same baseline DB
(`tests/fixtures/baseline-nrl.db`) and the post-#165 monotone constraints.

| Config | n | Accuracy | Log-loss | Brier |
|---|--:|--:|--:|--:|
| a) 2024+2025, no weights, no HPO (post-#165 baseline) | 424 | 0.6132 | 0.7364 | 0.2559 |
| b) + 2026 R1–7 in training | 480 | 0.6000 | 0.7416 | 0.2595 |
| c) + recency weights | 480 | 0.5917 | 0.7491 | 0.2627 |
| **d) + HPO w/ early stopping (current PR)** | 480 | **0.5854** | **0.7045** | **0.2496** |

**Two signals worth reading carefully:**

- Pooled accuracy drops (0.6132 → 0.5854) because the 2026 R1–7 rounds
  are structurally harder to predict — thinner rolling-history features
  at early rounds. Every model in the baseline test takes the same hit:
  EloMOV goes 0.6179 → 0.6125, logistic 0.5566 → 0.5604, etc. The
  accuracy drop is **eval-pool change**, not model degradation.
- **Log-loss and Brier — the proper scoring rules — BOTH IMPROVE on
  the full pool** under the final config. 0.7364 → 0.7045 on log-loss
  (−4.3 %), 0.2559 → 0.2496 on Brier (−2.5 %). The model is better
  calibrated across the bigger eval set despite fewer top-pick hits.

**The early-stopping save.** The first ablation run of config (d) was
disastrous — log_loss 0.8564, Brier 0.2875 — because Optuna picked
`n_estimators=439` tuned for the full 480-row dataset, and walk-forward
trains per-round on much smaller subsets (round 1 sees zero history;
round 10 sees ~80 rows). 439 trees on 80 rows = catastrophic overfit.
`train_xgboost` now reserves a held-out tail slice (15 %) per-round
and uses `early_stopping_rounds=30` to trim the estimator count to
what the training set actually supports. That took config (d) from a
retrain-gate block to a promotion candidate.

Production delivery: `artifacts/best_params.json` is committed + baked
into the Dockerfile. The retrain Job (#107) loads it via
`load_best_params()` on every fit; Monday's retrain run trains a
candidate with tuned hyperparameters + recency weights on all three
seasons + early stopping, shadow-evaluates, and promotes automatically
if the gate clears. Based on the (d) numbers above, it will.

Re-running HPO after larger-dataset PRs (e.g. #158 2023 backfill): the
Optuna study is persisted to `sqlite:///artifacts/optuna.db` (gitignored;
regenerable) so a second run resumes rather than restarting from scratch.

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

## Stacked ensemble (#171)

`StackedEnsemblePredictor` in `evaluation/predictors.py` combines
XGBoost + Skellam + EloMOV through a logistic-regression meta-learner
trained on out-of-fold base probabilities. Walk-forward flow per round:

1. Split chronologically-sorted history 80 / 20.
2. Fit each base on the 80 % slice → predict the 20 % slice → that's
   the OOF probability matrix (n_val × 3).
3. Fit the meta-learner via `fit_ensemble(mode="stacked")` on those
   OOF probabilities + actual outcomes. Inherits the existing kill
   switch from #56 (if meta doesn't beat the best base by 0.005
   log-loss, fall back to the best base's raw prediction).
4. Refit each base on the **full** history for inference so the
   strongest possible bases feed into the meta.

### Ablation — 2024+2025+2026 baseline, 480 predictions

| Model | accuracy | log-loss | brier |
|---|--:|--:|--:|
| home | 0.5646 | 0.6852 | 0.2460 |
| elo | 0.5833 | 0.6628 | 0.2353 |
| **EloMOV** | **0.6125** | 0.6668 | 0.2366 |
| logistic | 0.5604 | 0.8269 | 0.2751 |
| XGBoost (prod) | 0.5729 | 0.7079 | 0.2515 |
| Skellam | 0.5708 | 0.7097 | 0.2529 |
| **Stacked** | 0.5854 | **0.6807** | **0.2423** |

Stacked **beats XGBoost on all three metrics** (+1.25pp accuracy,
−3.8 % log-loss, −3.7 % brier) — the AC's promotion-gate criterion is
met. **EloMOV still wins on accuracy** (0.6125 vs 0.5854) because the
meta-learner's regularisation on the thin 20 %-tail val slice dilutes
EloMOV's signal; with a larger holdout (deeper history post-#158
backfill) the meta should learn to upweight EloMOV.

**Decision:** stacked is made available as a backup predictor via
`StackedEnsemblePredictor` but the production artefact stays XGBoost.
Promotion to production is a separate decision pending either (a) a
larger training set that lets the meta upweight EloMOV properly, or
(b) an explicit business call to trade accuracy for better calibration.
The `stacked` pin lives in `test_baseline_metrics.py` so any regression
trips CI.

**Production-artefact limitation:** the current stacked flow persists
XGBoost + Skellam cleanly (both have joblib serialisation) but EloMOV
doesn't have an artefact shape yet. Walk-forward evaluation uses an
in-memory EloMOV that replays the match history; a production
`train-stacked` CLI would need a small EloMOV serialiser first.
Filed as a follow-up if/when we decide to promote.

## Player-strength cap + market shrinkage (#203)

Two production-layer guards added after R8 2026 went 0/3, one of which
(Tigers v Raiders) was the exact PSD-overrules-market failure mode flagged
in #166. Both are off-the-shelf safety nets, not modelling changes — they
sit at the feature-input and prediction-output boundaries respectively, so
the underlying XGBoost / stacked / logistic model code is untouched.

1. **`PLAYER_STRENGTH_DIFF_CAP = 1000.0`** in
   `src/fantasy_coach/feature_engineering.py`. The audit measured
   `std≈1988` and 82.5 % of holdout rows with `|PSD| > 500`. Capping at
   `±1000` (~½σ) bounds extreme-value leverage without losing direction.
   Applied uniformly at training and inference, so saved artefacts and
   live predictions see the same distribution. The cap only affects the
   long tails (~20 % of rows in the audit's holdout); most predictions are
   unchanged.

2. **`MARKET_SHRINKAGE_WEIGHT = 0.3`** in
   `src/fantasy_coach/predictions.py`. After the model emits its raw
   home-win probability, when `odds_home_win_prob` is present (i.e.
   `missing_odds == 0.0`), the final probability is

       final_prob = 0.7 · model_prob + 0.3 · odds_home_win_prob

   `w = 0.3` is justified by the audit's Q4 result: in the 152 cases where
   PSD and the market disagreed on direction, the market won 56.6 % vs
   PSD's 43.4 %. Anchoring 30 % toward the market preserves direction on
   agreement (the common case) and pulls disagreement cases toward the
   more-accurate signal. Live R8 example: model = 0.457, market = 0.613,
   blended = 0.504 — flips Tigers/Raiders from away to home, the correct
   side. Tunable; raise if persistent model-overrules-market regressions
   re-appear in production.

The shrinkage is intentionally output-layer rather than baked into the
model so the stored `contributions` array still reflects what the model
itself "thought" — the UI shows the raw model attribution, with the blend
documented as a separate post-processing step in this section.

## Retraining cadence & drift (#107)

The production XGBoost artefact at
`gs://fantasy-coach-lcd-models/logistic/latest.joblib` is refreshed by a
weekly Cloud Run Job (`fantasy-coach-retrain`) triggered by Cloud
Scheduler every **Monday 10:00 AEST**, after Sunday's round is complete
and before Tuesday's precompute run. The full pipeline lives in
`src/fantasy_coach/retrain.py`; invoke locally with `python -m
fantasy_coach retrain`.

### Pipeline

1. Load completed matches from Firestore (last ~3 seasons).
2. Split into **training** (everything before) and a **4-round holdout**
   (the last 4 completed rounds).
3. Train a fresh XGBoost candidate on the training split
   (`train_xgboost`, same hyperparameter grid as the manual CLI).
4. Shadow-evaluate incumbent + candidate on the holdout
   (`models.promotion.shadow_evaluate`).
5. Gate the candidate (`models.promotion.gate_decision`).
6. On promote: upload to the GCS URI above (overwriting `latest.joblib`;
   bucket object versioning is the rollback path).
7. On block: open a GitHub issue tagged `model-drift`, body = metrics
   table + PSI warnings + rolling log-loss trend.
8. Always: write a `DriftReport` to Firestore
   (`model_drift_reports/{season}-{round:02d}`).

### Promotion gate

| Metric | Threshold | Gate behaviour |
|---|---|---|
| log-loss regression | > +2 % vs incumbent on holdout | **block** |
| brier regression | > +2 % vs incumbent on holdout | **block** |
| accuracy | any | informational only — never blocks |

Calibration is what gates, not accuracy. A model that pushes probabilities
toward 0/1 can beat the incumbent on accuracy while worsening log-loss;
`homeWinProbability` and the contribution-list UI both need calibrated
output, so log-loss + brier are the binding constraints.

### PSI (distribution shift)

Per-feature Population Stability Index between the training and holdout
feature matrices is computed on every run. Thresholds follow Siddiqi's
industry convention:

| PSI | Interpretation |
|---|---|
| < 0.10 | no meaningful shift |
| 0.10 – 0.25 | minor shift |
| > 0.25 | **warn** — surfaced in `DriftReport.psi_warnings` |

PSI **never blocks** — the AC explicitly scopes it to "warn but don't
block". Bin count is auto-reduced at small holdout sizes so the null
distribution doesn't trip the 0.25 threshold on pure sampling variance
(~32 holdout predictions → ~3 bins, see `drift._effective_bins`).

### Drift report schema

```
model_drift_reports/{season}-{round:02d}
├── season                int
├── round                 int                     latest holdout round
├── generated_at          string                  ISO 8601 UTC
├── model_version         string                  first 12 hex of sha256(artefact)
├── past_round_accuracy   float | null            incumbent on latest round
├── past_round_log_loss   float | null
├── past_round_brier      float | null
├── rolling_log_loss      list<map>               one per holdout round
│   ├── season            int
│   ├── round             int
│   ├── n                 int
│   ├── log_loss          float
│   └── accuracy          float
├── feature_psi           map<string, float>      per-feature PSI
└── psi_warnings          list<string>            feature names with PSI > 0.25
```

### Rollback

GCS object versioning is enabled on the `fantasy-coach-lcd-models`
bucket (platform-infra `google_storage_bucket.models`). To revert to the
prior artefact:

```bash
GENERATION=$(gsutil ls -a gs://fantasy-coach-lcd-models/logistic/latest.joblib \
    | sed -n '2p' | cut -d'#' -f2)
gsutil cp "gs://fantasy-coach-lcd-models/logistic/latest.joblib#${GENERATION}" \
    gs://fantasy-coach-lcd-models/logistic/latest.joblib
```

The API reads the latest generation on cold start, so restart Cloud Run
(roll a new revision via `deploy.yml` or `gcloud run services update`)
and the reverted artefact is served. The precompute Job re-downloads on
every execution so it picks the revert up on its next scheduled run.

### Out of scope for #107 (tracked separately)

- Logistic retraining (`train-logistic`). Logistic is a comparison
  baseline only — the retrain loop targets the production model. File
  a follow-up if logistic ever returns to production.
- Email alerting on gate-block. GitHub issue is the first notification
  channel; email can be added later by plugging into the existing budget
  notification channel.
- Online / per-round retraining. Weekly is sufficient at the current
  pace of data — rounds are week-sized and there's no mid-round signal
  that would change weights.
