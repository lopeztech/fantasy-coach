-- Schema version 6.
--
-- One row per match in `matches`, children (`match_players`, `match_team_stats`,
-- `match_player_stats`) keyed by match_id + side ('home' | 'away'). Upserts are
-- done by deleting the existing match rows and re-inserting, so children stay
-- consistent.
-- v2 adds referee_id and video_referee_id columns to matches (NRL officials block).
-- v3 adds is_on_field to match_players + a team_list_snapshots table (#24).
-- v4 adds home_odds / away_odds decimal closing-line columns (#26).
-- v5 adds home_odds_open / away_odds_open opening-line columns (#169).
-- v6 adds representative_callups (#211) and match_player_stats (#142) tables.

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

-- Representative callups for State of Origin and Test matches (#211).
-- One row per player per fixture window.  Populated by the precompute Job
-- via `representative.fetch_origin_squads` after squad announcement.
-- fixture: "origin1" | "origin2" | "origin3" | "test_au" | "test_nz" | "test_pac"
-- state:   "nsw" | "qld" | NULL (for non-Origin callups)
CREATE TABLE IF NOT EXISTS representative_callups (
    player_id    INTEGER NOT NULL,
    season       INTEGER NOT NULL,
    fixture      TEXT    NOT NULL,
    fixture_date TEXT    NOT NULL,  -- ISO 8601 date (Sunday of announcement week)
    nrl_team_id  INTEGER NOT NULL,
    state        TEXT,
    PRIMARY KEY (player_id, season, fixture)
);

CREATE INDEX IF NOT EXISTS idx_rep_callups_season_fixture
    ON representative_callups (season, fixture);

CREATE TABLE IF NOT EXISTS match_player_stats (
    match_id             INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    side                 TEXT    NOT NULL CHECK (side IN ('home', 'away')),
    ordinal              INTEGER NOT NULL,
    player_id            INTEGER NOT NULL,
    minutes_played       INTEGER,
    all_run_metres       INTEGER,
    tackles_made         INTEGER,
    missed_tackles       INTEGER,
    tackle_breaks        INTEGER,
    line_breaks          INTEGER,
    try_assists          INTEGER,
    offloads             INTEGER,
    errors               INTEGER,
    tries                INTEGER,
    tackle_efficiency    REAL,
    fantasy_points_total INTEGER,
    PRIMARY KEY (match_id, side, player_id)
);

CREATE INDEX IF NOT EXISTS idx_match_player_stats_player_match
    ON match_player_stats (player_id, match_id);
