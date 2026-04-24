"""Match preview generation via Gemini.

Builds a structured prompt from prediction + feature context, calls
CachingGeminiClient, and returns a 2–3 sentence natural-language preview.

Cache key: the prompt incorporates matchId + modelVersion, so the SHA-256
cache key in ResponseCache implicitly acts as a (matchId, modelVersion) cache.

Budget: 150 output tokens per match ≈ 2–3 sentences.  Eight matches/round ×
150 tokens ≈ 1 200 tokens output, well under $0.01/round at Flash pricing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from fantasy_coach.commentary.cache import CachingGeminiClient, context_hash
from fantasy_coach.feature_engineering import FEATURE_NAMES

_SYSTEM = (
    "You are a concise NRL match analyst. "
    "Write exactly 2–3 sentence match previews using only the facts provided — "
    "never invent statistics. Active voice. Direct. No filler phrases."
)

# Tokens per call — enforces <$0.01/round of 8 matches at Gemini Flash pricing.
_MAX_OUTPUT_TOKENS = 150

# 7-day TTL: predictions for a round don't change once generated.
_TTL = 7 * 24 * 3600.0

# +1 = higher value favours home, -1 = higher value favours away, 0 = skip.
_FEATURE_POLARITY: dict[str, int] = {
    "elo_diff": 1,
    "form_diff_pf": 1,
    "form_diff_pa": -1,
    "days_rest_diff": 1,
    "h2h_recent_diff": 1,
    "is_home_field": 0,
    "travel_km_diff": -1,
    "timezone_delta_diff": -1,
    "back_to_back_short_week_diff": -1,
    "is_wet": 0,
    "wind_kph": 0,
    "temperature_c": 0,
    "missing_weather": 0,
    "venue_avg_total_points": 0,
    "venue_home_win_rate": 1,
}

_FEATURE_LABELS: dict[str, str] = {
    "elo_diff": "Elo rating advantage",
    "form_diff_pf": "recent scoring form",
    "form_diff_pa": "recent defensive form",
    "days_rest_diff": "rest advantage",
    "h2h_recent_diff": "head-to-head record",
    "travel_km_diff": "travel burden",
    "timezone_delta_diff": "timezone adjustment",
    "back_to_back_short_week_diff": "back-to-back travel fatigue",
    "venue_home_win_rate": "venue home-win rate",
}


@dataclass
class MatchContext:
    """All facts available to the preview prompt — no fabrication beyond these."""

    match_id: int
    home_name: str
    away_name: str
    predicted_winner: str  # "home" | "away"
    home_win_prob: float
    model_version: str
    venue: str | None = None
    feature_row: Sequence[float] | None = None
    feature_names: Sequence[str] = field(default_factory=lambda: list(FEATURE_NAMES))


class PreviewGenerator:
    """Generate a cached 2–3 sentence match preview via Gemini.

    Usage::

        gen = PreviewGenerator(caching_client)
        text = gen.generate(ctx)
    """

    def __init__(
        self,
        client: CachingGeminiClient,
        *,
        max_output_tokens: int = _MAX_OUTPUT_TOKENS,
        ttl: float | None = _TTL,
    ) -> None:
        self._client = client
        self._max_output_tokens = max_output_tokens
        self._ttl = ttl

    def generate(self, ctx: MatchContext) -> str:
        """Return a preview string; result is cached by matchId + modelVersion."""
        prompt = _build_prompt(ctx)
        feature_hash = _context_hash(ctx)
        response = self._client.generate(
            prompt,
            system=_SYSTEM,
            max_output_tokens=self._max_output_tokens,
            ttl=self._ttl,
            feature_snapshot_hash=feature_hash,
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(ctx: MatchContext) -> str:
    winner_name = ctx.home_name if ctx.predicted_winner == "home" else ctx.away_name
    confidence = ctx.home_win_prob if ctx.predicted_winner == "home" else 1 - ctx.home_win_prob

    lines: list[str] = [
        f"[match_id={ctx.match_id} model={ctx.model_version}]",
        f"Teams: {ctx.home_name} (home) vs {ctx.away_name} (away)",
    ]
    if ctx.venue:
        lines.append(f"Venue: {ctx.venue}")

    lines.append(f"Model pick: {winner_name} ({confidence:.0%} win probability)")

    drivers = _top_drivers(ctx)
    if drivers:
        lines.append("Key factors favouring the pick:")
        for label in drivers:
            lines.append(f"  - {label}")

    lines.append(
        f"\nWrite a 2–3 sentence preview explaining why {winner_name} is favoured. "
        "Use only the facts listed above."
    )
    return "\n".join(lines)


def _top_drivers(ctx: MatchContext, n: int = 3) -> list[str]:
    """Return up to n human-readable driver labels ranked by strength."""
    if not ctx.feature_row:
        return []

    home_pick = ctx.predicted_winner == "home"
    scored: list[tuple[float, str]] = []

    for fname, val in zip(ctx.feature_names, ctx.feature_row, strict=False):
        polarity = _FEATURE_POLARITY.get(fname, 0)
        if polarity == 0:
            continue
        label = _FEATURE_LABELS.get(fname)
        if label is None:
            continue
        # Effective strength: how much does this feature favour the predicted winner?
        effective = val * polarity * (1.0 if home_pick else -1.0)
        if effective > 0:
            scored.append((effective, label))

    scored.sort(reverse=True)
    return [label for _, label in scored[:n]]


def _context_hash(ctx: MatchContext) -> str:
    """SHA-8 fingerprint of MatchContext inputs for cache-entry auditing."""
    payload = json.dumps(
        {
            "match_id": ctx.match_id,
            "home": ctx.home_name,
            "away": ctx.away_name,
            "predicted_winner": ctx.predicted_winner,
            "home_win_prob": round(ctx.home_win_prob, 4),
            "model_version": ctx.model_version,
            "venue": ctx.venue,
            "feature_row": list(ctx.feature_row) if ctx.feature_row else None,
        },
        sort_keys=True,
    )
    return context_hash(payload)
