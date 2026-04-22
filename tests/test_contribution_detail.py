"""Tests for contribution sentinel filtering and key_absence_diff detail (#124).

Covers:
- ``_compute_contributions`` drops sentinel-value features before top-K ranking
  so that "Weather data available" / "Venue 0 pts" never consume a top slot.
- ``_compute_contributions`` populates ``detail`` for ``key_absence_diff`` when
  ``builder`` and ``match`` are supplied.
- ``FeatureBuilder._key_absence_detail`` returns the correct player rows.
- Old predictions that lack ``detail`` still deserialise correctly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fantasy_coach.feature_engineering import (
    FEATURE_NAMES,
    KEY_ABSENCE_REGULAR_MIN_STARTS,
    POSITION_WEIGHTS,
    FeatureBuilder,
)
from fantasy_coach.features import MatchRow, PlayerRow, TeamRow
from fantasy_coach.predictions import FeatureContribution, _compute_contributions

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_loaded(feature_names: tuple[str, ...], dominant_idx: int = 0):
    """Return a LoadedModel-shaped object for the given feature set.

    The dominant feature is given a far-larger X range so it always ranks
    first by |contribution| — we can assert ordering in tests.
    """
    from fantasy_coach.models.logistic import LoadedModel

    n = len(feature_names)
    rng = np.random.default_rng(42)
    x_data = rng.standard_normal((100, n))
    x_data[:, dominant_idx] *= 5
    y = (x_data[:, dominant_idx] > 0).astype(int)

    pipeline = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    pipeline.fit(x_data, y)
    return LoadedModel(pipeline=pipeline, feature_names=feature_names)


def _player(
    player_id: int,
    *,
    position: str,
    first_name: str | None = None,
    last_name: str | None = None,
    is_on_field: bool = True,
) -> PlayerRow:
    return PlayerRow(
        player_id=player_id,
        jersey_number=player_id,
        position=position,
        first_name=first_name or f"F{player_id}",
        last_name=last_name or f"L{player_id}",
        is_on_field=is_on_field,
    )


def _standard_xiii(base: int = 100) -> list[PlayerRow]:
    positions = [
        (1, "Fullback"),
        (2, "Winger"),
        (3, "Centre"),
        (4, "Centre"),
        (5, "Winger"),
        (6, "Five-Eighth"),
        (7, "Halfback"),
        (8, "Prop"),
        (9, "Hooker"),
        (10, "Prop"),
        (11, "2nd Row"),
        (12, "2nd Row"),
        (13, "Lock"),
    ]
    return [_player(base + j, position=pos) for j, pos in positions]


def _match_row(
    match_id: int,
    *,
    season: int = 2024,
    round_: int = 1,
    start_time: datetime,
    home_team_id: int,
    away_team_id: int,
    home_players: list[PlayerRow],
    away_players: list[PlayerRow],
    home_score: int = 0,
    away_score: int = 0,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        season=season,
        round=round_,
        start_time=start_time,
        match_state="FullTime",
        venue="Stadium",
        venue_city="Sydney",
        weather=None,
        home=TeamRow(
            team_id=home_team_id,
            name=f"Team{home_team_id}",
            nick_name=f"T{home_team_id}",
            score=home_score,
            players=home_players,
        ),
        away=TeamRow(
            team_id=away_team_id,
            name=f"Team{away_team_id}",
            nick_name=f"T{away_team_id}",
            score=away_score,
            players=away_players,
        ),
        team_stats=[],
    )


# ---------------------------------------------------------------------------
# Sentinel filtering
# ---------------------------------------------------------------------------


def _full_feature_loaded(dominant_feature: str):
    """LoadedModel with all FEATURE_NAMES, dominant_feature ranked #1."""
    idx = FEATURE_NAMES.index(dominant_feature)
    return _make_loaded(FEATURE_NAMES, dominant_idx=idx)


def _raw_for(feature_values: dict[str, float]) -> np.ndarray:
    """Build a (1, n_features) array from a sparse dict of feature overrides."""
    row = np.zeros(len(FEATURE_NAMES), dtype=float)
    for fname, v in feature_values.items():
        row[FEATURE_NAMES.index(fname)] = v
    # is_home_field is always 1.0 in real predictions.
    row[FEATURE_NAMES.index("is_home_field")] = 1.0
    return row.reshape(1, -1)


def test_sentinel_is_home_field_never_in_contributions() -> None:
    """is_home_field is always 1.0 and constant — must be excluded always."""
    loaded = _full_feature_loaded("elo_diff")
    # Give is_home_field a large raw value to make it a strong contributor if
    # not filtered.  It must still not appear.
    raw = _raw_for({"elo_diff": 100.0, "is_home_field": 1.0})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    features = {c.feature for c in contribs}
    assert "is_home_field" not in features


def test_sentinel_missing_weather_zero_excluded() -> None:
    """missing_weather=0 means weather IS available — no signal, drop it."""
    loaded = _full_feature_loaded("elo_diff")
    raw = _raw_for({"elo_diff": 5.0, "missing_weather": 0.0})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    features = {c.feature for c in contribs}
    assert "missing_weather" not in features


def test_sentinel_missing_weather_one_included() -> None:
    """missing_weather=1 (no data) IS informative — must not be filtered."""
    loaded = _full_feature_loaded("missing_weather")
    raw = _raw_for({"missing_weather": 1.0})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    features = {c.feature for c in contribs}
    assert "missing_weather" in features


def test_sentinel_venue_avg_total_points_zero_excluded() -> None:
    loaded = _full_feature_loaded("elo_diff")
    raw = _raw_for({"elo_diff": 5.0, "venue_avg_total_points": 0.0})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    features = {c.feature for c in contribs}
    assert "venue_avg_total_points" not in features


def test_sentinel_venue_avg_total_points_nonzero_included() -> None:
    loaded = _full_feature_loaded("venue_avg_total_points")
    raw = _raw_for({"venue_avg_total_points": 48.5})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    assert any(c.feature == "venue_avg_total_points" for c in contribs)


def test_sentinel_venue_home_win_rate_half_excluded() -> None:
    """Neutral prior (0.5) means no venue history — filter it."""
    loaded = _full_feature_loaded("elo_diff")
    raw = _raw_for({"elo_diff": 5.0, "venue_home_win_rate": 0.5})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    features = {c.feature for c in contribs}
    assert "venue_home_win_rate" not in features


def test_sentinel_ref_features_excluded_when_missing_referee() -> None:
    """ref_avg_total_points and ref_home_penalty_diff are defaults when referee unknown."""
    loaded = _full_feature_loaded("elo_diff")
    raw = _raw_for({"elo_diff": 5.0, "missing_referee": 1.0, "ref_avg_total_points": 42.0})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    features = {c.feature for c in contribs}
    assert "ref_avg_total_points" not in features
    assert "ref_home_penalty_diff" not in features


def test_sentinel_ref_features_included_when_referee_known() -> None:
    """ref features carry real signal when missing_referee=0."""
    loaded = _full_feature_loaded("ref_avg_total_points")
    raw = _raw_for({"ref_avg_total_points": 55.0, "missing_referee": 0.0})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    assert any(c.feature == "ref_avg_total_points" for c in contribs)


def test_sentinel_filtering_fills_top_k_from_remaining() -> None:
    """Filtered slots must not create gaps — top_k slots from non-sentinel features."""
    loaded = _full_feature_loaded("elo_diff")
    # Set several sentinels; dominant non-sentinel is elo_diff.
    raw = _raw_for(
        {
            "elo_diff": 80.0,
            "is_home_field": 1.0,
            "missing_weather": 0.0,
            "venue_avg_total_points": 0.0,
            "venue_home_win_rate": 0.5,
        }
    )
    contribs = _compute_contributions(loaded, raw, top_k=3)
    assert contribs is not None
    # Should still get 3 rows (from the non-sentinel features).
    assert len(contribs) == 3
    features = {c.feature for c in contribs}
    assert "is_home_field" not in features
    assert "missing_weather" not in features


# ---------------------------------------------------------------------------
# key_absence_diff detail population
# ---------------------------------------------------------------------------


def _builder_with_history(
    team_id: int,
    n_matches: int,
    players_fn,
) -> FeatureBuilder:
    """Return a builder with ``n_matches`` of history for ``team_id``."""
    builder = FeatureBuilder()
    for r in range(n_matches):
        match = _match_row(
            match_id=r,
            round_=r + 1,
            start_time=datetime(2024, 3, r + 1, tzinfo=UTC),
            home_team_id=team_id,
            away_team_id=999,
            home_players=players_fn(r),
            away_players=_standard_xiii(base=500),
        )
        builder.advance_season_if_needed(match)
        builder.record(match)
    return builder


def test_key_absence_detail_missing_halfback_has_name() -> None:
    """Missing halfback is listed in detail with their name and position."""
    team_id = 10
    builder = _builder_with_history(
        team_id,
        n_matches=KEY_ABSENCE_REGULAR_MIN_STARTS + 1,
        players_fn=lambda _r: _standard_xiii(base=100),
    )

    # Today: halfback (player 107) is absent.
    today_home = [p for p in _standard_xiii(base=100) if p.player_id != 107]
    detail = builder._key_absence_detail(team_id, today_home)

    assert len(detail) == 1
    row = detail[0]
    assert row["player_id"] == 107
    assert row["position"] == "Halfback"
    assert row["weight"] == pytest.approx(POSITION_WEIGHTS["Halfback"])
    # Name constructed from first_name + last_name in _player helper.
    assert row["name"] == "F107 L107"


def test_key_absence_detail_empty_when_no_history() -> None:
    builder = FeatureBuilder()
    detail = builder._key_absence_detail(10, _standard_xiii())
    assert detail == []


def test_key_absence_detail_empty_when_no_is_on_field() -> None:
    team_id = 10
    builder = _builder_with_history(
        team_id,
        n_matches=KEY_ABSENCE_REGULAR_MIN_STARTS + 1,
        players_fn=lambda _r: _standard_xiii(base=100),
    )
    no_flags = [
        PlayerRow(
            player_id=p.player_id,
            jersey_number=p.jersey_number,
            position=p.position,
            first_name=p.first_name,
            last_name=p.last_name,
            is_on_field=None,
        )
        for p in _standard_xiii(base=100)
    ]
    assert builder._key_absence_detail(team_id, no_flags) == []


def test_compute_contributions_populates_key_absence_detail() -> None:
    """_compute_contributions fills detail when builder + match supplied."""
    team_id = 10
    builder = _builder_with_history(
        team_id,
        n_matches=KEY_ABSENCE_REGULAR_MIN_STARTS + 1,
        players_fn=lambda _r: _standard_xiii(base=100),
    )

    # Upcoming match: home missing halfback, away intact.
    today_home = [p for p in _standard_xiii(base=100) if p.player_id != 107]
    today_away = _standard_xiii(base=200)
    match = _match_row(
        match_id=99,
        round_=99,
        start_time=datetime(2024, 4, 1, tzinfo=UTC),
        home_team_id=team_id,
        away_team_id=20,
        home_players=today_home,
        away_players=today_away,
    )

    # Build a loaded model with key_absence_diff as the dominant feature.
    loaded = _full_feature_loaded("key_absence_diff")
    raw = np.asarray([builder.feature_row(match)], dtype=float)

    contribs = _compute_contributions(loaded, raw, builder=builder, match=match)
    assert contribs is not None

    ka_contrib = next((c for c in contribs if c.feature == "key_absence_diff"), None)
    assert ka_contrib is not None, "key_absence_diff should be in contributions"
    assert ka_contrib.detail is not None
    home_missing = ka_contrib.detail["home_missing"]
    assert isinstance(home_missing, list)
    assert len(home_missing) == 1
    assert home_missing[0]["position"] == "Halfback"
    assert home_missing[0]["name"] == "F107 L107"


def test_compute_contributions_no_detail_without_builder() -> None:
    """Without builder/match, key_absence_diff has no detail."""
    loaded = _full_feature_loaded("key_absence_diff")
    raw = _raw_for({"key_absence_diff": 3.0})
    contribs = _compute_contributions(loaded, raw)
    assert contribs is not None
    ka = next((c for c in contribs if c.feature == "key_absence_diff"), None)
    if ka is not None:
        assert ka.detail is None


# ---------------------------------------------------------------------------
# Backward-compat: old FeatureContribution without detail still loads
# ---------------------------------------------------------------------------


def test_feature_contribution_detail_defaults_to_none() -> None:
    c = FeatureContribution(feature="elo_diff", value=10.0, contribution=0.3)
    assert c.detail is None


def test_feature_contribution_roundtrips_detail() -> None:
    c = FeatureContribution(
        feature="key_absence_diff",
        value=3.0,
        contribution=-0.5,
        detail={
            "home_missing": [
                {"player_id": 107, "name": "Joe Smith", "position": "Halfback", "weight": 3.0}
            ],
            "away_missing": [],
        },
    )
    dumped = c.model_dump()
    c2 = FeatureContribution(**dumped)
    assert c2.detail is not None
    assert c2.detail["home_missing"][0]["name"] == "Joe Smith"
