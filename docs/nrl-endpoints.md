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

## Rate limits

No documented rate limit. No `robots.txt` permission statement for these endpoints. Throttle to ~1 req/sec out of politeness.

## Samples

- [`samples/draw-round-8-2026.json`](samples/draw-round-8-2026.json) — trimmed fixtures-endpoint response
