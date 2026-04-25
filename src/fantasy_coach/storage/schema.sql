-- Schema version 6.
--
-- One row per match in `matches`, children (`match_players`, `match_team_stats`)
-- keyed by match_id + side ('home' | 'away'). Upserts are done by deleting the
-- existing match rows and re-inserting, so children stay consistent.
-- v2 adds referee_id and video_referee_id columns to matches (NRL officials block).
-- v3 adds is_on_field to match_players + a team_list_snapshots table (#24).
-- v4 adds home_odds / away_odds decimal closing-line columns (#26).
-- v5 adds home_odds_open / away_odds_open opening-line columns (#169).
-- v6 adds representative_callups for State of Origin / Test squad tracking (#211).

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS matches (
    match_id         INTEGER PRIMARY KEY,
    season           INTEGER NOT NULL,
    round            INTEGER NOT NULL,
    start_time       TEXT    NOT NULL,  -- ISO 8601 UTC
    match_state      TEXT    NOT NULL,
    venue            TEXT,
    venue_city       TEXT,
    weather          TEXT,
    home_team_id     INTEGER NOT NULL,
    home_name        TEXT    NOT NULL,
    home_nick        TEXT    NOT NULL,
    home_score       INTEGER,
    away_team_id     INTEGER NOT NULL,
    away_name        TEXT    NOT NULL,
    away_nick        TEXT    NOT NULL,
    away_score       INTEGER,
    referee_id       INTEGER,   -- NRL profileId for position="Referee"
    video_referee_id INTEGER,   -- NRL profileId for position="Senior Review Official"
    home_odds        REAL,      -- decimal odds from scrape OR merged closing lines
    away_odds        REAL,
    home_odds_open   REAL,      -- opening-line decimal odds from xlsx (#169)
    away_odds_open   REAL
);

CREATE INDEX IF NOT EXISTS idx_matches_season_round
    ON matches(season, round);

CREATE TABLE IF NOT EXISTS match_players (
    match_id       INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    side           TEXT    NOT NULL CHECK (side IN ('home', 'away')),
    player_id      INTEGER NOT NULL,
    jersey_number  INTEGER,
    position       TEXT,
    first_name     TEXT,
    last_name      TEXT,
    is_on_field    INTEGER,   -- 1/0/NULL; starting XIII flag from NRL's isOnField
    PRIMARY KEY (match_id, side, player_id)
);

CREATE TABLE IF NOT EXISTS team_list_snapshots (
    snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    season        INTEGER NOT NULL,
    round         INTEGER NOT NULL,
    match_id      INTEGER NOT NULL,
    team_id       INTEGER NOT NULL,
    scraped_at    TEXT    NOT NULL,   -- ISO 8601 UTC
    players_json  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_team_list_match_time
    ON team_list_snapshots (match_id, team_id, scraped_at);

CREATE INDEX IF NOT EXISTS idx_team_list_season_time
    ON team_list_snapshots (season, scraped_at);

-- Representative callups: State of Origin, Test, and Pacific Championship
-- squads. One row per (player_id, fixture) pair. ``fixture`` is an enum:
-- origin1, origin2, origin3, test_au, test_nz, test_pac.
-- Populated manually from squad announcements; used to derive the
-- origin_callups_diff and is_test_window_diff model features (#211).
CREATE TABLE IF NOT EXISTS representative_callups (
    player_id    INTEGER NOT NULL,
    season       INTEGER NOT NULL,
    fixture      TEXT    NOT NULL CHECK (fixture IN (
                     'origin1', 'origin2', 'origin3',
                     'test_au', 'test_nz', 'test_pac'
                 )),
    fixture_date TEXT    NOT NULL,  -- ISO 8601 date of the rep game
    state        TEXT,              -- 'NSW', 'QLD', 'AUS', 'NZ', 'PAC', etc.
    PRIMARY KEY (player_id, season, fixture)
);

CREATE INDEX IF NOT EXISTS idx_representative_callups_season
    ON representative_callups (season, fixture);

CREATE TABLE IF NOT EXISTS match_team_stats (
    match_id    INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    title       TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    units       TEXT,
    home_value  REAL,
    away_value  REAL,
    PRIMARY KEY (match_id, ordinal)
);
