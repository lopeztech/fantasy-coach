"""Pydantic schemas for the Betting Tips feature.

camelCase field names match the SPA's other typed responses (PredictionOut etc.).
The N815 lint rule is suppressed module-wide via the field-level ``noqa: N815``
markers, following the convention established in ``predictions.py``.

Tips are produced by the precompute Job (Tue/Thu) and read back by the
``/betting-tips`` endpoint — see ``betting.generator`` for the selection logic
and ``betting.store`` for the cache layout.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Market = Literal["h2h", "totals"]
H2HSelection = Literal["home", "away"]
TotalsSelection = Literal["over", "under"]


class TipLeg(BaseModel):
    """One leg inside a single, double, or treble.

    ``selection`` is the human-readable side (team name for H2H, "Over 41.5"
    for totals). ``selectionCode`` is the machine-readable enum the SPA can
    use to render variant-specific iconography.
    """

    matchId: int  # noqa: N815
    homeName: str  # noqa: N815
    awayName: str  # noqa: N815
    kickoff: str  # noqa: N815  ISO 8601
    market: Market
    selectionCode: H2HSelection | TotalsSelection  # noqa: N815
    selection: str  # noqa: N815  display string, e.g. "Storm" or "Over 41.5"
    line: float | None = None  # totals line; None for H2H
    decimalOdds: float  # noqa: N815  bookmaker price for this leg
    modelProb: float  # noqa: N815  model's probability the leg lands
    impliedProb: float  # noqa: N815  de-vigged bookmaker probability
    edge: float  # modelProb − impliedProb
    bookmaker: str  # short label e.g. "nrl.com" or "TAB"


class Tip(BaseModel):
    """A single bet (1 leg) or multi (2+ legs)."""

    kind: Literal["single", "double", "treble"]
    legs: list[TipLeg]
    combinedOdds: float  # noqa: N815  product of leg decimal odds
    combinedModelProb: float  # noqa: N815  product of leg modelProbs (independence assumption)
    combinedImpliedProb: float  # noqa: N815  product of leg impliedProbs
    edge: float  # combinedModelProb − combinedImpliedProb
    suggestedUnits: float  # noqa: N815  flat 1.0 for v1; future: Kelly fraction


class BettingTipsOut(BaseModel):
    """Top-level response for ``GET /betting-tips``.

    Always 2 singles + 2 doubles + 2 trebles when the prediction pool is rich
    enough. Sections fall short when there aren't enough qualifying legs — the
    SPA renders a "not enough confident picks this round" empty state.
    """

    season: int
    round: int
    generatedAt: str  # noqa: N815  ISO 8601 UTC — when the tips were computed
    oddsSnapshotAt: str | None = None  # noqa: N815  when external odds were fetched (totals only)
    caveat: str = (
        "Multi probabilities assume each leg lands independently. "
        "Real outcomes are correlated, so true joint probabilities are usually lower."
    )
    singles: list[Tip]
    doubles: list[Tip]
    trebles: list[Tip]


# ---------------------------------------------------------------------------
# the-odds-api ingest shapes (internal — not part of the public response)
# ---------------------------------------------------------------------------


class TotalsLine(BaseModel):
    """A single totals (over/under) line for a match, from the-odds-api."""

    line: float  # e.g. 41.5
    overDecimalOdds: float  # noqa: N815
    underDecimalOdds: float  # noqa: N815
    bookmaker: str
