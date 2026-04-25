# NRL endpoints

Reference for the two `nrl.com` JSON endpoints this project depends on. No auth required; both are plain `GET` with no cookies or special headers.

## Fixtures list (per round)

```
GET https://www.nrl.com/draw/data?competition=111&round={N}&season={YEAR}
```

Returns all matches in a given round, plus byes and dropdown filter metadata.

**Params**

| Name          | Example  | Notes                                                                |
|---------------|----------|----------------------------------------------------------------------|
| `competition` | `111`    | NRL Telstra Premiership. Other values available in `filterCompetitions` of the response (e.g. `161` Women's, `119` Pre-Season) |
| `round`       | `1`–`27` | Regular season. `28+` returns finals weeks                           |
| `season`      | `2024`   | Historically valid range per `filterSeasons` (1908–2026)             |

**Key fields (per fixture)**

- `matchCentreUrl` — `/draw/nrl-premiership/{year}/round-{n}/{home}-v-{away}/`. Append `/data` for the per-match endpoint below.
- `homeTeam.teamId`, `awayTeam.teamId` — stable integer team IDs.
- `homeTeam.theme.key`, `awayTeam.theme.key` — slug form of the team name (e.g. `wests-tigers`, `sea-eagles`). Matches the slug used in `matchCentreUrl`.
- `homeTeam.nickName`, `awayTeam.nickName` — display names.
- `homeTeam.odds`, `awayTeam.odds` — decimal bookmaker odds. **Populated only for upcoming fixtures in the current season; dropped once matches finish. Cannot be used as a historical training feature.**
- `homeTeam.teamPosition`, `awayTeam.teamPosition` — ladder position at request time (e.g. `"3rd"`).
- `clock.kickOffTimeLong` — ISO 8601 UTC timestamp.
- `venue`, `venueCity`.
- `matchState` — observed values: `Upcoming`, `Pre`, `Live`, `FullTime`. Likely others.

**Finals rounds**

For `round >= 28`, fixture slugs are numbered rather than team-based:

```
/draw/nrl-premiership/2024/finals-week-1/game-1/
/draw/nrl-premiership/2024/finals-week-1/game-2/
```

Scrapers must handle both `round-{n}/{home}-v-{away}` and `finals-week-{n}/game-{m}` formats.

**Byes**

Returned in a separate top-level `byes` array, one entry per team on bye that round.

## Per-match detail

```
GET https://www.nrl.com/draw/nrl-premiership/{year}/{round-slug}/{match-slug}/data
```

- `{round-slug}` = `round-{n}` or `finals-week-{n}`
- `{match-slug}` = `{home}-v-{away}` for regular rounds, `game-{n}` for finals weeks

Returns rich per-match JSON: `matchId`, `homeTeam`/`awayTeam` with player lists + stats, `timeline`, `officials`, `venue`, `stats`, etc.

**Schema stability**: consistent between 2024 and 2026. 2026 adds a `weather` key absent in 2024 — treat optional keys defensively.

**Slug ordering**: `home-v-away`, not alphabetical. Wrong order returns 404. Always source slugs from the fixtures endpoint rather than constructing them manually.

### Per-player game stats (`stats.players`)

Completed matches (`matchState == "FullTime"`) include a per-player stats block at:

```
stats.players.homeTeam[]   # 18 entries (13 starters + 5 bench)
stats.players.awayTeam[]   # 18 entries
stats.players.meta[]       # group metadata for the in-app UI; not used here
```

Each entry has `playerId` plus ~58 numeric stat fields. The fields the project currently persists (#142):

| Key | Type | Notes |
|-----|------|-------|
| `playerId` | int | Stable across seasons |
| `minutesPlayed` | int | 0-80 (regular time) |
| `allRunMetres` | int | Total run metres incl. dummy-half + kick returns |
| `tacklesMade` | int | Successful tackles |
| `missedTackles` | int | |
| `tackleBreaks` | int | "Tackle busts" in commentary |
| `lineBreaks` | int | |
| `tryAssists` | int | |
| `offloads` | int | |
| `errors` | int | All handling/general errors |
| `tries` | int | |
| `tackleEfficiency` | float | Percentage 0-100, pre-computed by NRL |
| `fantasyPointsTotal` | int | NRL Fantasy points; useful as composite signal |

Other published-but-not-persisted fields include `hitUps`, `hitUpRunMetres`, `dummyHalfRuns`, `dummyHalfRunMetres`, `kicks`, `kickMetres`, `kickReturnMetres`, `passes`, `receipts`, `postContactMetres`, `lineBreakAssists`, `lineEngagedRuns`, `playTheBallTotal`, `playTheBallAverageSpeed`, `tackleEfficiency`, `bombKicks`, `crossFieldKicks`, `grubberKicks`, `forcedDropOutKicks`, `fortyTwentyKicks`, `goals`, `conversions`, `fieldGoals`, `points`, etc. Add to `PlayerMatchStat` and the SQLite schema (with a v6→v7 migration) when needed.

**Upcoming matches** carry an empty `stats.players: {}` — extractor must default both arrays to `[]`.

## Rate limits

No documented rate limit. No `robots.txt` permission statement for these endpoints. Throttle to ~1 req/sec out of politeness.

## Samples

- [`samples/draw-round-8-2026.json`](samples/draw-round-8-2026.json) — trimmed fixtures-endpoint response
