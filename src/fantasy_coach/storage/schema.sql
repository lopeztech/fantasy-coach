-- Schema version 1.
--
-- One row per match in `matches`, children (`match_players`, `match_team_stats`)
-- keyed by match_id + side ('home' | 'away'). Upserts are done by deleting the
-- existing match rows and re-inserting, so children stay consistent.

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS matches (
    match_id      INTEGER PRIMARY KEY,
    season        INTEGER NOT NULL,
    round         INTEGER NOT NULL,
    start_time    TEXT    NOT NULL,  -- ISO 8601 UTC
    match_state   TEXT    NOT NULL,
    venue         TEXT,
    venue_city    TEXT,
    weather       TEXT,
    home_team_id  INTEGER NOT NULL,
    home_name     TEXT    NOT NULL,
    home_nick     TEXT    NOT NULL,
    home_score    INTEGER,
    away_team_id  INTEGER NOT NULL,
    away_name     TEXT    NOT NULL,
    away_nick     TEXT    NOT NULL,
    away_score    INTEGER
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
    PRIMARY KEY (match_id, side, player_id)
);

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
