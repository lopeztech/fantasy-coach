"""Tests for PreviewGenerator and prompt-building helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from fantasy_coach.commentary.cache import CachingGeminiClient, ResponseCache, TokenBudget
from fantasy_coach.commentary.client import GeminiClient, GeminiResponse
from fantasy_coach.commentary.preview import (
    MatchContext,
    PreviewGenerator,
    _build_prompt,
    _top_drivers,
)
from fantasy_coach.feature_engineering import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_caching_client(text: str = "Broncos to win.") -> CachingGeminiClient:
    """Return a CachingGeminiClient whose underlying GeminiClient is mocked."""
    mock_client = MagicMock(spec=GeminiClient)
    mock_client._model = "gemini-2.0-flash-001"
    mock_client.generate.return_value = GeminiResponse(text=text, input_tokens=80, output_tokens=20)
    return CachingGeminiClient(
        mock_client,
        cache=ResponseCache(default_ttl=3600),
        budget=TokenBudget(per_request_max=200, daily_limit=10_000),
    )


def _ctx(
    *,
    match_id: int = 1,
    home: str = "Broncos",
    away: str = "Storm",
    winner: str = "home",
    prob: float = 0.63,
    version: str = "abc123",
    venue: str | None = "Suncorp Stadium",
    feature_row: list[float] | None = None,
) -> MatchContext:
    return MatchContext(
        match_id=match_id,
        home_name=home,
        away_name=away,
        predicted_winner=winner,
        home_win_prob=prob,
        model_version=version,
        venue=venue,
        feature_row=feature_row,
    )


# ---------------------------------------------------------------------------
# MatchContext
# ---------------------------------------------------------------------------


def test_match_context_defaults() -> None:
    ctx = MatchContext(
        match_id=5,
        home_name="Raiders",
        away_name="Panthers",
        predicted_winner="away",
        home_win_prob=0.42,
        model_version="v1",
    )
    assert ctx.venue is None
    assert ctx.feature_row is None
    assert list(ctx.feature_names) == list(FEATURE_NAMES)


def test_match_context_custom_feature_names() -> None:
    ctx = MatchContext(
        match_id=1,
        home_name="A",
        away_name="B",
        predicted_winner="home",
        home_win_prob=0.6,
        model_version="v1",
        feature_names=["f1", "f2"],
    )
    assert list(ctx.feature_names) == ["f1", "f2"]


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


def test_prompt_contains_team_names() -> None:
    ctx = _ctx(home="Broncos", away="Storm")
    prompt = _build_prompt(ctx)
    assert "Broncos" in prompt
    assert "Storm" in prompt


def test_prompt_contains_match_id_and_model_version() -> None:
    ctx = _ctx(match_id=42, version="deadbeef")
    prompt = _build_prompt(ctx)
    assert "42" in prompt
    assert "deadbeef" in prompt


def test_prompt_contains_venue_when_set() -> None:
    ctx = _ctx(venue="Suncorp Stadium")
    assert "Suncorp Stadium" in _build_prompt(ctx)


def test_prompt_omits_venue_when_none() -> None:
    ctx = _ctx(venue=None)
    assert "Venue" not in _build_prompt(ctx)


def test_prompt_home_winner_confidence() -> None:
    ctx = _ctx(winner="home", prob=0.63, home="Broncos")
    prompt = _build_prompt(ctx)
    assert "Broncos" in prompt
    assert "63%" in prompt


def test_prompt_away_winner_confidence() -> None:
    ctx = _ctx(winner="away", prob=0.38, away="Storm")
    prompt = _build_prompt(ctx)
    assert "Storm" in prompt
    assert "62%" in prompt  # 1 - 0.38


def test_prompt_includes_drivers_when_feature_row_present() -> None:
    # elo_diff=10 favours home
    row = [0.0] * len(FEATURE_NAMES)
    row[0] = 10.0  # elo_diff
    ctx = _ctx(feature_row=row, winner="home")
    prompt = _build_prompt(ctx)
    assert "Elo rating advantage" in prompt


def test_prompt_no_drivers_section_without_feature_row() -> None:
    ctx = _ctx(feature_row=None)
    assert "Key factors" not in _build_prompt(ctx)


# ---------------------------------------------------------------------------
# _top_drivers
# ---------------------------------------------------------------------------


def _feature_row(**kwargs: float) -> list[float]:
    """Build a feature row with named overrides; all others zero."""
    names = list(FEATURE_NAMES)
    row = [0.0] * len(names)
    for k, v in kwargs.items():
        row[names.index(k)] = v
    return row


def test_top_drivers_home_pick_elo() -> None:
    row = _feature_row(elo_diff=50.0)
    ctx = _ctx(feature_row=row, winner="home")
    drivers = _top_drivers(ctx)
    assert "Elo rating advantage" in drivers


def test_top_drivers_away_pick_elo() -> None:
    # elo_diff negative → home Elo < away Elo → away advantage
    row = _feature_row(elo_diff=-30.0)
    ctx = _ctx(feature_row=row, winner="away")
    drivers = _top_drivers(ctx)
    assert "Elo rating advantage" in drivers


def test_top_drivers_opposing_feature_excluded() -> None:
    # elo_diff positive favours home; predicted winner is away → should NOT appear
    row = _feature_row(elo_diff=50.0)
    ctx = _ctx(feature_row=row, winner="away")
    drivers = _top_drivers(ctx)
    assert "Elo rating advantage" not in drivers


def test_top_drivers_pa_polarity() -> None:
    # form_diff_pa polarity=-1: negative value = home concedes less = home advantage
    row = _feature_row(form_diff_pa=-5.0)
    ctx = _ctx(feature_row=row, winner="home")
    drivers = _top_drivers(ctx)
    assert "recent defensive form" in drivers


def test_top_drivers_skips_is_home_field_and_weather_flags() -> None:
    row = _feature_row(is_home_field=1.0, missing_weather=1.0, is_wet=1.0, wind_kph=30.0)
    ctx = _ctx(feature_row=row, winner="home")
    drivers = _top_drivers(ctx)
    assert not drivers  # all polarity-0 features, nothing to rank


def test_top_drivers_returns_at_most_n() -> None:
    row = _feature_row(elo_diff=50.0, form_diff_pf=10.0, days_rest_diff=3.0, h2h_recent_diff=8.0)
    ctx = _ctx(feature_row=row, winner="home")
    assert len(_top_drivers(ctx, n=3)) <= 3


def test_top_drivers_ranked_by_magnitude() -> None:
    row = _feature_row(elo_diff=100.0, form_diff_pf=5.0)
    ctx = _ctx(feature_row=row, winner="home")
    drivers = _top_drivers(ctx, n=2)
    assert drivers[0] == "Elo rating advantage"
    assert drivers[1] == "recent scoring form"


def test_top_drivers_empty_when_no_feature_row() -> None:
    ctx = _ctx(feature_row=None)
    assert _top_drivers(ctx) == []


# ---------------------------------------------------------------------------
# PreviewGenerator — happy path + caching
# ---------------------------------------------------------------------------


def test_preview_generator_returns_text() -> None:
    client = _fake_caching_client("Brisbane Broncos should win comfortably.")
    gen = PreviewGenerator(client)
    result = gen.generate(_ctx())
    assert result == "Brisbane Broncos should win comfortably."


def test_preview_generator_strips_whitespace() -> None:
    client = _fake_caching_client("  Preview text.  \n")
    gen = PreviewGenerator(client)
    assert gen.generate(_ctx()) == "Preview text."


def test_preview_generator_caches_identical_context() -> None:
    mock_inner = MagicMock(spec=GeminiClient)
    mock_inner._model = "gemini-2.0-flash-001"
    mock_inner.generate.return_value = GeminiResponse(
        text="Cached.", input_tokens=50, output_tokens=10
    )
    caching = CachingGeminiClient(
        mock_inner,
        cache=ResponseCache(default_ttl=3600),
        budget=TokenBudget(per_request_max=200, daily_limit=10_000),
    )
    gen = PreviewGenerator(caching)
    ctx = _ctx(match_id=7, version="v1")

    gen.generate(ctx)
    gen.generate(ctx)  # second call — should hit cache

    assert mock_inner.generate.call_count == 1  # live call only once


def test_preview_generator_different_match_ids_no_cache_collision() -> None:
    mock_inner = MagicMock(spec=GeminiClient)
    mock_inner._model = "gemini-2.0-flash-001"
    mock_inner.generate.return_value = GeminiResponse(
        text="Text.", input_tokens=50, output_tokens=10
    )
    caching = CachingGeminiClient(
        mock_inner,
        cache=ResponseCache(default_ttl=3600),
        budget=TokenBudget(per_request_max=200, daily_limit=10_000),
    )
    gen = PreviewGenerator(caching)

    gen.generate(_ctx(match_id=1, version="v1"))
    gen.generate(_ctx(match_id=2, version="v1"))

    assert mock_inner.generate.call_count == 2


def test_preview_generator_different_model_versions_no_cache_collision() -> None:
    mock_inner = MagicMock(spec=GeminiClient)
    mock_inner._model = "gemini-2.0-flash-001"
    mock_inner.generate.return_value = GeminiResponse(
        text="Text.", input_tokens=50, output_tokens=10
    )
    caching = CachingGeminiClient(
        mock_inner,
        cache=ResponseCache(default_ttl=3600),
        budget=TokenBudget(per_request_max=200, daily_limit=10_000),
    )
    gen = PreviewGenerator(caching)

    gen.generate(_ctx(match_id=1, version="v1"))
    gen.generate(_ctx(match_id=1, version="v2"))

    assert mock_inner.generate.call_count == 2


def test_preview_generator_custom_token_cap() -> None:
    mock_inner = MagicMock(spec=GeminiClient)
    mock_inner._model = "gemini-2.0-flash-001"
    mock_inner.generate.return_value = GeminiResponse(
        text="Short.", input_tokens=50, output_tokens=10
    )
    caching = CachingGeminiClient(
        mock_inner,
        cache=ResponseCache(),
        budget=TokenBudget(per_request_max=200, daily_limit=10_000),
    )
    gen = PreviewGenerator(caching, max_output_tokens=100)
    gen.generate(_ctx())
    _, kwargs = mock_inner.generate.call_args
    assert kwargs["max_output_tokens"] == 100
