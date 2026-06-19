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
| `is_origin_round` | `1.0` when the NRL club round overlaps a State of Origin game week (Rounds 13, 16, 19 for 2024–2026). `0.0` otherwise. Hard-coded calendar in `representative.py`. | Origin camps pull up to 17 players per state from their clubs the week of each game, causing systematic roster disruption that Elo and rolling form cannot see. |
| `is_magic_round` | `1.0` when this is the Magic Round (all 16 teams at Suncorp, single weekend). `0.0` otherwise. Hard-coded calendar in `representative.py`. | The logistical bunching and atypical travel pattern adds a signal on top of `is_neutral_venue`. |
| `origin_callups_diff` | Home minus away count of players named in the Origin squad for this fixture window (from `representative_callups` DB table). `0.0` until squad data is backfilled. | Granular per-match disruption index — a team missing 6 Origin players has a quantifiably different disadvantage than one missing 1. |
| `is_test_window` | `1.0` when the match date falls within an international Test window (e.g. Pacific Championships, Oct–Nov). `0.0` otherwise. | Representative-country players pulled from club squads late-season; affects finals warm-up matches. |

### NRL calendar effects (#211)

`representative.py` holds hard-coded calendar data for 2024–2026 (the
seasons with training data). Each function is a pure lookup with no I/O:

- `is_origin_round(season, round_)` — True when the NRL round overlaps an Origin week.
- `is_magic_round(season, round_)` — True for the single-venue Magic Round weekend.
- `is_test_window(match_date)` — True when the date falls within a Pacific Championships / Test window.
- `origin_game_number(season, round_)` — Returns 1, 2, or 3 for the corresponding Origin game, or None.

`representative_callups` is a SQLite/Firestore table (schema v6) populated by
the precompute job when squad announcements are scraped. Until squads are
backfilled, `origin_callups_diff` defaults to `0.0` — the model degrades
gracefully to the binary `is_origin_round` flag.

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

### Ablation notes — referee features (#57, revisited #161)

**Original (#57) ablation — 2024–2025 only, 424 predictions, referee_id = NULL for all rows:**

| Metric   | Before #57 | After #57 | Δ       |
|----------|------------|-----------|---------|
| accuracy | 0.5637     | 0.5660    | +0.23pp |
| log_loss | 0.7636     | 0.7640    | −0.0004 |
| brier    | 0.2655     | 0.2654    | +0.0001 |

Inconclusive: all `referee_id = NULL` in that DB snapshot — every match fell
back to the league-mean prior so the features carried no information.

**Revisit (#161) — 2023–2026 baseline, 692 predictions, referee data populated:**

Fixture coverage at time of ablation: 2023 100%, 2024 100%, 2025 99%, 2026 31%.
Ablation method: `scripts/ablate_referee_features.py`. "Without" condition forces
`_referee_features` to return the `referee_id = None` fallback for every match
(league-mean total-points, `ref_home_penalty_diff = 0`, `missing_referee = 1`).

| Model    | Metric   | With refs | Without refs | Δ       |
|----------|----------|-----------|--------------|---------|
| Logistic | accuracy | 0.5694    | 0.5679       | +0.15pp |
| Logistic | log_loss | 0.8906    | 0.8825       | +0.0081 ↑ (worse) |
| Logistic | brier    | 0.2831    | 0.2814       | +0.0017 ↑ (worse) |
| XGBoost  | accuracy | 0.6055    | 0.5838       | +2.17pp |
| XGBoost  | log_loss | 0.6938    | 0.6854       | +0.0084 ↑ (worse) |
| XGBoost  | brier    | 0.2476    | 0.2438       | +0.0038 ↑ (worse) |

**Result: accuracy signal exists (especially XGBoost +2.17pp), but proper
scoring rules worsen for both models.** The features are teaching the model
referee-specific correlations that flip some borderline predictions correctly
while widening probability estimates beyond what the data supports — a
calibration cost that log_loss and brier both capture.

The most likely cause is data sparsity per referee: even with referee IDs
populated for 2023–2025, the rolling-20 window per referee hits the
`REF_SHRINKAGE_N` threshold for many referees, so the shrinkage toward league
mean is aggressive. As referee-annotated data accumulates (2+ full seasons of
refs with ≥ 20 observed matches each), calibration should improve.

**Decision (post #161):** features remain active. The XGBoost accuracy signal
is non-trivial; removing the features now would cost 2.17pp on the evaluation
pool. Revisit calibration specifically for referee-driven predictions once at
least 2 full seasons (2024 + 2025 complete) have ≥ 20 obs per active referee.
Track via the `missing_referee` rate in production predictions — when it drops
below 20%, the calibration concern diminishes.

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
- **Player-level stats** — kept out of the baseline. Will come in once
  XGBoost (#25) makes nonlinear interactions worth modelling.

(Team-list availability shipped under #27 as `key_absence_diff` /
`player_strength_diff`; structured injury *health* status shipped under #269 —
see "Injury severity feature" below.)

### Injury severity feature (#208 / #269)

The availability features (`key_absence_diff`, `player_strength_diff`) only know
whether a player is on the team list — binary in-or-out. The injury feature adds
*health status* the team list can't express, sourced from the scraped weekly NRL
injury list (`injury_reports`) — parsed from prose by Gemini and backfilled
historically via the Wayback Machine (#268), which lives outside the walk-forward
match history.

| Feature | Definition | Why |
|---------|-----------|-----|
| `injury_severity_diff` | Per team, a **position-weighted count** of currently-listed players: Σ `status_weight × POSITION_WEIGHTS[pos]` over active reports (OUT/SUSPENDED 1.0, TEST 0.5, 21-man-squad 0.25; RETURNING excluded). The injured player's position comes from the walk-forward `player_ratings` book. Home − away, clamped to ±`INJURY_SEVERITY_DIFF_CAP` (50). | Carrying multiple injured spine players is a real strength signal the team list misses (an injured player named on the list still counts as "available" there). |
| `missing_injury_data` | 1.0 when no injury list was scraped for this `(season, round)`. | Mirrors the other `missing_*` flags so the model learns a separate intercept for "no injury signal" rather than reading a quiet week as a zeroed-out one. A scraped week with no injuries for these teams is **not** missing. |

**Why a count, not a severity × duration.** #269 originally specified
`Σ status_weight × weeks_out` plus a returning-player count and a late-withdrawal
count. A walk-forward ablation (2024–2025, `scripts/ab_injury_backtest.py`) killed
all three:

- The full 4-feature set **degraded** the model (−1.4pp accuracy, +0.008 log-loss).
- The Gemini-estimated `weeks_out` was the culprit — dropping it (a binary count)
  flipped the result to net-positive. The returning count was net-harmful (the
  "rust" hypothesis didn't hold) and `late_withdrawal_diff` is ~always 0.0
  historically (the kickoff-hour watcher only runs live), so both were cut.
- Final position-weighted binary severity: **+0.7pp accuracy, −0.0016 log-loss,
  −0.0008 Brier** — net-positive on all three, clearing #208's acceptance gate.

The returning and late-withdrawal data is still collected (`injury_reports`
RETURNING rows; the `watch-team-lists` watcher's `late_team_changes`) for a future
revisit once live late-change history accumulates.

**Plumbing.** The data is supplied to `FeatureBuilder` via an `InjuryIndex`
(keyed by `(season, round, team_id)`). Because the evaluation harness and training
CLIs build `FeatureBuilder` instances deep inside predictor classes that never see
a repo, the index is set as a process-level "active" index for the duration of a
`train-xgboost` / `evaluate` run (`active_injury_index(...)`);
`compute_predictions` instead passes it explicitly for the round being served. A
*single* static index over all scraped reports is leakage-safe for walk-forward:
scoring round R only reads `(season, R, team)` keys, whose reports were scraped
before R's kickoff. When no index is set the features are neutral
(`missing_injury_data = 1.0`), so the default/library path is unchanged.

**Retrain.** These two names are appended to `FEATURE_NAMES`, which bumps the
`load_model` schema check — the production XGBoost artefact must be retrained and
re-uploaded to GCS (per the "retrain when FEATURE_NAMES changes" rule).

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

**Promotion gate result (run 2026-06-11, post-#158 backfill):** walk-forward
on the 2023–2026 baseline (692 predictions):

| Rater | Accuracy | Log-loss | Brier | ECE |
|---|---|---|---|---|
| EloMOV | **0.6272** | **0.6566** | **0.2315** | **0.0396** |
| Glicko-2 | 0.6055 | 0.7524 | 0.2553 | 0.1381 |

The gate required Glicko-2 to beat EloMOV log-loss by ≥ 0.5%; it instead
regressed by 14.6%. **Gate fails decisively — EloMOV stays the default
rater.** The high ECE suggests the RD-driven probability widening is
miscalibrated on NRL-sized samples (17 teams, ~26 games/season); revisit
only if a much deeper backfill (2019+) lands or the RD/volatility priors
are re-tuned for short seasons.

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
``predictions.py`` dispatches by model type:
- **logistic**: ``coef × (x − mean) / scale`` (unchanged, exact).
- **XGBoost / gradient-boosting**: TreeSHAP via
  ``models/explainability.py::shap_contributions`` — delegates to
  ``Booster.predict(pred_contribs=True)``.  Returns per-feature contributions
  in log-odds space satisfying the exact sum invariant:
  ``sum(shap_contributions) + bias == raw_log_odds``.  The bias column is
  dropped so the output aligns with ``FEATURE_NAMES``. Output shape matches
  logistic so the sentinel filter + detail enrichment + UI rendering all work
  without branching. For XGBoost artefacts, ``shap_interactions`` also
  enriches each contribution's ``detail`` with the dominant interaction partner
  (``detail.interaction.{partner, magnitude}``), surfaced in the SPA as a
  "× feature_name" sub-row.

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
| `elo_diff`, `form_diff_pf`, `h2h_recent_diff`, `venue_home_win_rate`, `form_diff_pf_adjusted`, `player_strength_diff`, `odds_home_win_prob`, `halves_strength_diff`, `forwards_strength_diff`, `hooker_strength_diff`, `outside_backs_strength_diff`, `halves_x_forwards_diff` | +1 |
| `form_diff_pa`, `key_absence_diff`, `form_diff_pa_adjusted` | −1 |

The other features stay unconstrained — weather, rest, travel, and
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

### No-signal XGBoost column sampling weights

Some `FEATURE_NAMES` columns are deliberately present before their backing
data is available in the baseline snapshot: line movement, representative
callups/minutes, forecast-only weather fields, and the constant `is_home_field`
tree no-op. Dropping those columns would break model-artifact compatibility, so
XGBoost instead keeps the schema and gives those columns near-zero
`feature_weights` for column sampling.

Walk-forward on the 2023–2026 baseline (n=692):

| Predictor | Metric | Before | After | Δ |
|---|---:|---:|---:|---:|
| XGBoost | accuracy | 0.5882 | **0.5968** | **+0.0087** |
| XGBoost | log_loss | 0.6916 | **0.6889** | **−0.0027** |
| XGBoost | brier | 0.2465 | **0.2451** | **−0.0014** |
| XGBoost | ECE | 0.0422 | 0.0487 | +0.0065 |
| Stacked | accuracy | 0.5896 | 0.5896 | 0.0000 |
| Stacked | log_loss | **0.6794** | 0.6814 | +0.0020 |
| Stacked | brier | **0.2421** | 0.2431 | +0.0010 |

The 2026 slice improves materially for the production model (accuracy 0.5357
→ 0.6071, log_loss 0.7441 → 0.6874, brier 0.2686 → 0.2456). Remove a column
from `NO_SIGNAL_COLUMN_SAMPLE_FEATURES` once its data is backfilled and the
feature has real variance in walk-forward training.

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

## Stacked ensemble (#171, reworked 2026-06)

`StackedEnsemblePredictor` in `evaluation/predictors.py` combines
XGBoost + Skellam + EloMOV through a convex logit-space combiner trained
on **accumulated walk-forward out-of-fold rows**.

The original (#171) design split each round's history 80/20 and fit a
LogReg meta-learner on the 20 % tail — a thin single-window sample that
systematically diluted the strongest base (EloMOV). The rework exploits
the harness contract: `fit(history)` is called per round on the *same*
predictor instance with an extending history, so matches that are new
since the previous call were predicted by bases trained strictly before
them. Those are genuine OOF rows; the predictor accumulates them across
the whole walk and refits the combiner (`fit_ensemble(mode=
"logit_weighted")`, convex weights over base log-odds) on the full pool
each round — hundreds of rows by season's end instead of a few dozen.
Bases are refit on the full history at the end of every `fit()` for
inference. Below 40 accumulated OOF rows the predictor returns the
EloMOV base unchanged.

Logit-space mixing matters: probability-space convex mixing
systematically under-weights confident bases near 0/1. On the 2023–2026
offline combiner study, the logit variant beat probability mixing on
log-loss for every base combination tried.

### Results — 2023–2026 baseline, 692 predictions, full odds coverage

See `test_baseline_metrics.py` EXPECTED for the authoritative pinned
numbers (this rework + the closing-line coverage fix landed together).
Versus the old design on the same data the rework improves the stacked
row on all three metrics, and the new `blended` production-parity
predictor (see "Production probability blend" below) is the strongest
pinned row overall.

## Position-pair matchup features (#210)

`player_strength_diff` collapses the full roster into a single scalar. The
position-pair features disaggregate it into four position-group differentials
(home − away sum of per-player Elo ratings for starters in each group):

| Feature | Positions |
|---|---|
| `halves_strength_diff` | Halfback, Five-Eighth |
| `forwards_strength_diff` | Prop, Lock, 2nd Row |
| `hooker_strength_diff` | Hooker |
| `outside_backs_strength_diff` | Fullback, Winger, Centre |
| `halves_x_forwards_diff` | Interaction: `fwds_diff` when `halves_diff` and `fwds_diff` share the same sign, else 0 |

The interaction term captures "dominant on both axes" — a team with both an
elite halves pairing and a dominant forward pack compounds their advantages in
a way linear combinations don't express. XGBoost can exploit the disaggregated
signals via axis-aligned splits that `player_strength_diff` alone prevents.

All five features use `POSITION_GROUPS` from `models/player_ratings.py` and the
same `PlayerRatings` book as `player_strength_diff`. They default to `0.0` when
no `is_on_field` data exists (same no-data contract as `missing_player_strength`).
Monotone constraints (+1) are applied to all five: positive diff = home group
stronger = monotonically higher home-win probability.

Walk-forward ablation pending retrain with new feature schema.

## Player-strength cap (#203)

Production-layer guard added after R8 2026 went 0/3 (the Tigers v Raiders
PSD-overrules-market failure mode flagged in #166):

**`PLAYER_STRENGTH_DIFF_CAP = 1000.0`** in
`src/fantasy_coach/feature_engineering.py`. The audit measured
`std≈1988` and 82.5 % of holdout rows with `|PSD| > 500`. Capping at
`±1000` (~½σ) bounds extreme-value leverage without losing direction.
Applied uniformly at training and inference, so saved artefacts and
live predictions see the same distribution. The cap only affects the
long tails (~20 % of rows in the audit's holdout); most predictions are
unchanged.

#203 also introduced a linear output-layer market shrink
(`MARKET_SHRINKAGE_WEIGHT = 0.3`); that has been superseded by the
logit-space probability blend below.

## Closing-line coverage fix (2026-06)

The aussportsbetting xlsx has closing lines for **every NRL season back to
2009**, but the baseline DB's `merge-closing-lines` runs were stale: 2023
had never been merged at all, and 2024/2025 sat at ~77 % from an older
canonicalisation pass. Re-running the merge brought every completed match
in the 2023–2026 baseline to full coverage (213/213 per completed season;
2026 covers all completed rounds):

| Season | Before | After |
|---|---|---|
| 2023 | 0/213 | 213/213 |
| 2024 | 164/213 | 213/213 |
| 2025 | 166/213 | 213/213 |
| 2026 (completed) | 43/56 | 56/56 |

Since `odds_home_win_prob` is XGBoost's single strongest feature, the data
fix alone moved walk-forward XGBoost accuracy 0.5968 → 0.6301 (+3.3 pp)
with log-loss and brier both improving. Skellam improved on all three
metrics too. **When refreshing the baseline fixture, always re-run
`merge-closing-lines` for every season before copying the DB into
`tests/fixtures/`** — the docstring in `test_baseline_metrics.py` includes
the step.

The same merge was applied to `data/nrl.db` and synced to the production
Firestore `matches` collection so the weekly retrain (#107) trains on the
same coverage.

## Production probability blend (logit space)

`models/blend.py` defines the output layer applied by
`_apply_probability_blend` in `predictions.py` after the primary model
emits its raw probability:

    final = σ( 0.24·logit(p_model) + 0.36·logit(p_elo_mov) + 0.40·logit(p_market) )

- `p_model` — the loaded artefact's output (XGBoost in production).
- `p_elo_mov` — the `elo_mov_home_win_prob` feature already present in the
  prediction-time feature row.
- `p_market` — the `odds_home_win_prob` feature (absent when
  `missing_odds` is set, or out of (0, 1) — both treated as missing).

Missing signals renormalise the remaining weights (no market → model 0.4 /
EloMOV 0.6; nothing available → raw model probability). Logit space is the
correct mixing geometry: linear blending under-weights confident signals
near 0/1.

**Evidence** (offline combiner study over walk-forward base probabilities,
2023–2026 baseline, n=692, full closing-line coverage): fixed logit-space
weights beat every per-round-refit meta-learner tried (convex weights,
LogReg meta, market-augmented stacking), and the optimum is flat across
market weights 0.3–0.6 (accuracy 0.643–0.646, log-loss 0.634–0.641), so
the chosen point is not a knife-edge fit:

| Output layer | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Raw XGBoost | 0.6301 | 0.6802 | 0.2404 |
| Old: linear 0.3 market shrink | 0.6431 | 0.6534 | 0.2298 |
| **New: logit blend (0.24/0.36/0.40)** | **0.6445** | **0.6380** | **0.2232** |
| Pure closing line (reference) | 0.6445 | 0.6311 | 0.2199 |

The blend is pinned in `test_baseline_metrics.py` as the `blended`
predictor (production parity), so a regression in the *served* probability
trips CI even when the raw model is unchanged. The stored `contributions`
array still reflects the raw model attribution; the blend remains a
documented post-processing step.

Caveat: historical evaluation uses **closing** lines, while the Tue/Thu
precompute sees odds days before kickoff — live blend inputs are slightly
noisier than the backtest's. The EloMOV leg is unaffected.

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

## Market efficiency: CLV + profit-based evaluation (#212)

`src/fantasy_coach/evaluation/profit.py` adds a second evaluation axis:

**Closing-line value (CLV)** — does the model beat the closing line?

    clv = p_model − p_close

where `p_close` is the de-vigged closing-line probability (from
`MatchRow.home.odds` / `.away.odds`, backfilled via `merge-closing-lines`).

A consistently positive mean CLV indicates the model finds value the market
later corrects to — the near-unfakeable long-run profitability signal.

**Functions:**

| Function | Description |
|----------|-------------|
| `compute_clv(eval_result, matches)` | Join predictions to closing-line data; return a `CLVReport` with per-match CLV, mean CLV, win rate, and flat-stake ROI. Returns `None` when fewer than 10 covered matches exist. |
| `kelly_stake(p_model, decimal_odds, bankroll, kelly_fraction=0.25)` | Quarter-Kelly stake sizing. Returns 0 when the model has no edge (`p × odds ≤ 1`). |
| `simulate_pnl(match_clvs, strategy, starting_bankroll)` | Simulate bankroll evolution across all covered matches. Strategies: `"flat"` (1 unit/bet) or `"quarter_kelly"`. |

**Usage:**

```bash
python -m fantasy_coach evaluate \
  --model xgboost --model bookmaker \
  --seasons 2024,2025 \
  --closing-lines data/odds/nrl.xlsx \
  --profit
```

The `--profit` flag appends a "Market efficiency" section to the markdown
report with mean CLV per model and a cumulative CLV curve table.

**Interpretation note** documented in the report: positive CLV does **not**
equal positive PnL on a small sample — both are reported side by side.
Statistical significance requires ≥ 400 predictions; Wald-test p < 0.05
is the criterion used to declare a model "statistically edge-positive".

### Betting-tip evidence (2026-06, full closing-line coverage, n=692)

Walk-forward backtest of flat-stake betting on each predictor's pick at the
closing line, plus value-filtered betting (only bet when
`p_model × decimal_odds − 1 > threshold`):

| Strategy | n bets | Hit rate | Flat ROI |
|---|--:|--:|--:|
| Blended pick, every match | 692 | — | −6.4 % |
| Raw XGBoost pick, every match | 692 | — | −2.0 % |
| Raw XGBoost, edge > 0 only | 299 | 0.518 | +0.5 % |
| **Raw XGBoost, edge > 5 %** | **242** | **0.529** | **+7.1 %** |
| Raw XGBoost, edge > 10 % | 195 | 0.497 | +7.3 % |

Two practical conclusions baked into how the product should present tips:

1. **Winner picks and probabilities come from the blend** — it has the best
   accuracy (0.6445) and calibration (ECE 0.028) of anything we run. But
   because it anchors 40 % to the market, betting its pick at market odds
   pays the vig with little disagreement left to exploit — it is not a
   staking signal.
2. **Value flags should come from the raw model's edge vs the market.**
   The +7.1 % ROI at the 5 % edge threshold is *suggestive, not proven*
   (SE ≈ 6.4 % at n=242, t ≈ 1.1 — well short of Wald p < 0.05), and the
   backtest bets at closing prices the Tue/Thu precompute can't actually
   get. Treat the edge flag as "the model sees value here", never as a
   profitability promise.

## Prediction uncertainty (#146)

Every prediction from `compute_predictions` now carries four optional fields
that expose the model's confidence in the output:

| Field | Type | Description |
|-------|------|-------------|
| `baseModelSpread` | `float` | `max − min` probability across XGBoost + logistic + bookmaker for this match. Higher = more model disagreement. |
| `winProbability80ci` | `[float, float]` | Proxy 80% interval `[prob − spread/2, prob + spread/2]` clipped to `[0, 1]`. Not a Bayesian credible interval — use it as a rough disagreement band pending calibration. |
| `trainingDataSimilarity` | `float` | Cosine similarity between this match's feature vector and the mean training-set feature vector. Values close to 1.0 indicate an in-distribution matchup; values below ~0.5 flag unusual team/context combinations. |
| `confidenceBand` | `"low"` \| `"medium"` \| `"high"` | 3-level label derived from `baseModelSpread` and `trainingDataSimilarity`. |

### Confidence band thresholds

These are heuristic — pending held-out calibration against realised outcomes.

| Band | Condition |
|------|-----------|
| `"high"` | `spread ≤ 0.10` **and** `ood_similarity ≥ 0.80` |
| `"medium"` | `spread ≤ 0.20` **and** `ood_similarity ≥ 0.50` |
| `"low"` | any other case |

### Storage

The four fields are serialised together as a single `uncertainty` JSON column
in the SQLite `predictions` table (Firestore stores the full `model_dump()`).
Predictions written before this feature shipped deserialise with all four
fields as `None` — the response schema is additive.

### Known limitations

- The 80% CI is a proxy (disagreement band), not a calibrated interval. A
  proper Bayesian or bootstrap interval would require re-running the model
  many times or holding out a calibration set by season.
- `trainingDataSimilarity` uses cosine similarity in raw feature space (no
  whitening). Features with large absolute magnitudes (e.g. Elo ratings in
  the hundreds) dominate the dot product. A PCA-whitened space or
  Mahalanobis distance would be more principled.
- `baseModelSpread` reflects the number of secondary models available: a
  single-model deployment (no logistic + no odds) always has spread = 0 and
  will appear `"high"` confidence regardless of the true uncertainty.

## Bayesian hierarchical model (#144)

`src/fantasy_coach/models/bayesian_hierarchical.py`

Requires `uv sync --extra training` (adds `pymc>=5.0`, `numpyro>=0.15`).

### Model specification

Poisson component form (Dixon-Coles 1997 lineage):

```
log(λ_H) = μ + att[home] + dfn[away] + home_adv
log(λ_A) = μ + att[away] + dfn[home]
home_score ~ Poisson(λ_H)
away_score ~ Poisson(λ_A)
```

Latent parameters (all per season):

| Parameter | Description | Prior |
|-----------|-------------|-------|
| `att[t]` | Team attack strength in log-rate space | `ZeroSumNormal(σ_att)`, σ_att ~ HalfNormal(0.3) |
| `dfn[t]` | Team defense strength (negative = concede more) | `ZeroSumNormal(σ_dfn)`, σ_dfn ~ HalfNormal(0.3) |
| `home_adv` | Shared home-field log-rate boost | Normal(0.1, 0.3) |
| `μ` | Log-scale intercept ≈ log(typical NRL score) | Normal(log(20), 0.5) |

ZeroSumNormal enforces the sum-to-zero constraint (Σ att = Σ dfn = 0),
making attack and defense parameters identifiable without a corner
constraint.

Optional **Dixon-Coles low-score correction** (`use_dc_correction=True`): a
τ-parameterised log-likelihood adjustment for cells {(0,0), (0,1), (1,0),
(1,1)}. For NRL where sub-6-point totals are rare, τ typically converges
near zero; left off by default.

### Sampling

- NUTS via PyMC5; numpyro NUTS backend selected automatically when numpyro
  is installed (~5× faster on CPU).
- Default: 500 warmup + 500 draws, 2 chains. Produces ~1 000 raw samples,
  thinned to 500 for persistence.
- A single fit on 400 matches takes ~3–4 min on 2 vCPU (Cloud Run Job
  default). Bump to 4 vCPU if the Bayesian model is added to the weekly
  precompute pipeline.

### Posterior persistence

`save_bayesian_hierarchical` / `load_bayesian_hierarchical` store a 500-sample
trimmed trace as a plain numpy dict inside a joblib blob. No PyMC import is
required at inference time — `load_bayesian_hierarchical` rebuilds
`BayesianHierarchicalModel` from the saved arrays.

### Prior choices (rationale)

- **Power-prior cold start**: season-over-season, the previous season's
  posterior mean is a sensible warm-start for the new season. Use
  `α ≈ 0.4–0.6` as the power prior exponent (stronger shrinkage for teams
  with high roster turnover). Current implementation uses the unconditional
  HalfNormal prior; the power-prior extension is a v2 improvement.
- **Laplace shortcut for weekly updates**: between precompute runs, re-fit
  using the last posterior summary (mean + covariance) as a Laplace
  approximation of the prior. Skips the random-walk latent and is ~10×
  faster than full NUTS re-sampling. Suitable when adding a single round's
  results.

### Walk-forward integration

`BayesianPredictor` in `src/fantasy_coach/evaluation/predictors.py`
implements the `Predictor` protocol for the walk-forward harness:

```python
from fantasy_coach.evaluation import BayesianPredictor, walk_forward_from_repo

result = walk_forward_from_repo(repo, seasons, BayesianPredictor)
metrics = result.metrics()  # accuracy, log_loss, brier, ece
```

Falls back to p=0.5 when pymc is not installed or history < 10 matches.

### Coverage evaluation

The Bayesian model exposes a posterior predictive margin distribution that
can be checked for empirical coverage — whether the actual home-minus-away
score margin falls inside the HDI at the stated credible level.

```python
from fantasy_coach.evaluation import walk_forward_bayesian_coverage

coverage = walk_forward_bayesian_coverage(matches_by_round, BayesianPredictor)
# {"n": 424, "coverage_80": 0.812, "coverage_95": 0.946}
```

**Coverage targets** (from the #144 acceptance criteria):
- 80%-HDI: ≥ 78% empirical coverage
- 95%-HDI: ≥ 93% empirical coverage

Slight under-coverage is expected and honest; the HDI uses boundary-inclusive
counting (the integer boundary of a discrete Skellam distribution is
included). If coverage is materially below target (e.g. < 70% at 80%-CI),
the model is miscalibrated and should not ship — add a post-hoc isotonic
calibration layer on the posterior quantiles.

**Note on HDI vs equal-tailed interval**: `predict_margin_hdi` uses the
minimum-width HDI (highest density interval), not the equal-tailed interval.
For asymmetric posteriors (large home-field advantage or one-sided injuries)
the HDI is shorter and covers the mass more faithfully.

### Ensemble registration

`BayesianPredictor` can be used as a base model in `EnsemblePredictor`
alongside the existing Logistic/Skellam/XGBoost bases:

```python
from fantasy_coach.evaluation import BayesianPredictor, EnsemblePredictor, LogisticPredictor

ensemble = EnsemblePredictor(
    [LogisticPredictor, BayesianPredictor],
    mode="weighted",
    name="logistic+bayesian",
)
```

The existing `fallback_to_base` kill-switch in `fit_ensemble` handles the
case where the Bayesian model does not improve on the best base — it routes
predictions through the better base automatically.

**Note**: adding `BayesianPredictor` to `StackedEnsemblePredictor` (which
uses XGBoost + Skellam + EloMOV) would double the fitting time per round.
Evaluate in a separate ablation before promoting to the production stack.

### Ablation vs logistic and Skellam

Pending walk-forward evaluation on 2024+2025 data. Hypotheses from the
literature and the #144 design discussion:

| Metric | Expected vs Logistic | Expected vs Skellam |
|--------|----------------------|---------------------|
| Log-loss rounds 1–4 | Better (partial pooling shrinks cold-start predictions) | Roughly equal |
| Log-loss mid-season | Roughly equal | Slightly better (posterior mean ≈ Skellam point estimate) |
| Brier score | Roughly equal | Roughly equal |
| Margin 80%-CI coverage | N/A (new capability) | Better (HDI vs point-estimate-derived CI) |

### When to prefer the Bayesian model

- **Early season (rounds 1–4)**: the ZeroSumNormal shrinkage toward zero
  prevents extreme rating divergence before teams have played enough matches.
  Logistic/XGBoost use rolling features that are sparse in early rounds.
- **Novel matchups**: new expansion teams or franchises with < 5 head-to-head
  records benefit from the pooled league-mean prior.
- **Uncertainty quantification**: when the SPA needs a credible interval for
  the margin (e.g. the "try-it-yourself" explorer), only the Bayesian model
  provides a coherent posterior predictive distribution.
- **Do not prefer** for mid-season predictions where XGBoost with bookmaker
  odds dominates — the Bayesian model does not ingest the odds feature and
  cannot match its accuracy on well-priced matches.
