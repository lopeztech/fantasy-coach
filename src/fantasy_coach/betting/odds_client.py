"""Thin httpx client for the-odds-api.com (NRL totals markets).

Used only by the precompute Job. The API is **never called on the hot path**;
results are computed once per precompute run, blended with cached predictions,
and persisted to Firestore for the ``/betting-tips`` read-through.

Free tier covers ``rugby_league_nrl`` with regions=``au`` and markets=``h2h,totals``
at 500 requests/month — a Tue+Thu cadence uses ~8/month. The key is read from
``FANTASY_COACH_ODDS_API_KEY`` (Secret Manager in production); without it the
client is never constructed and the precompute step gracefully skips totals.

Retries follow the same exponential-backoff shape as ``commentary.client``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import httpx

from fantasy_coach.betting.models import TotalsLine

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.the-odds-api.com/v4"
# the-odds-api sport key. Confirmed against the public sports listing: the
# slug is a single word "rugbyleague" — using "rugby_league_nrl" returns 404.
SPORT_KEY = "rugbyleague_nrl"
REGIONS = "au"
MARKETS = "h2h,totals"

# Raise httpx's INFO-level logger to WARNING. The default INFO log writes the
# full request URL on every call, which includes the API key in this client's
# query string ("?apiKey=...&regions=au..."). Without this, every odds-api
# call leaks the secret into Cloud Logging. We can't move the key into a
# header — the-odds-api only accepts the key via query string. WARNING is a
# safe global setting (no other httpx caller in this codebase relies on the
# INFO line for behaviour).
logging.getLogger("httpx").setLevel(logging.WARNING)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Map nrl.com team names to the-odds-api team names. the-odds-api uses long
# canonical names ("Brisbane Broncos"); nrl.com uses short nicknames ("Broncos").
# Built on first call from a small static map — small enough to keep in code.
_TEAM_ALIASES: dict[str, frozenset[str]] = {
    "Broncos": frozenset({"brisbane broncos", "broncos"}),
    "Raiders": frozenset({"canberra raiders", "raiders"}),
    "Bulldogs": frozenset({"canterbury bulldogs", "canterbury-bankstown bulldogs", "bulldogs"}),
    "Sharks": frozenset({"cronulla sharks", "cronulla-sutherland sharks", "sharks"}),
    "Dolphins": frozenset({"dolphins"}),
    "Titans": frozenset({"gold coast titans", "titans"}),
    "Sea Eagles": frozenset({"manly sea eagles", "manly warringah sea eagles", "sea eagles"}),
    "Storm": frozenset({"melbourne storm", "storm"}),
    "Knights": frozenset({"newcastle knights", "knights"}),
    "Cowboys": frozenset({"north queensland cowboys", "cowboys"}),
    "Eels": frozenset({"parramatta eels", "eels"}),
    "Panthers": frozenset({"penrith panthers", "panthers"}),
    "Rabbitohs": frozenset({"south sydney rabbitohs", "rabbitohs"}),
    "Dragons": frozenset({"st george illawarra dragons", "dragons"}),
    "Roosters": frozenset({"sydney roosters", "roosters"}),
    "Warriors": frozenset({"new zealand warriors", "nz warriors", "warriors"}),
    "Wests Tigers": frozenset({"wests tigers", "tigers"}),
}


def _match_team(api_team: str) -> str | None:
    """Map a the-odds-api team name back to the nrl.com short nickname.

    Returns None when the name doesn't fit any known NRL side — happens for
    pre-season exhibitions or rep games that occasionally appear in the feed.
    """
    needle = api_team.strip().lower()
    for nrl_name, aliases in _TEAM_ALIASES.items():
        if needle in aliases:
            return nrl_name
    return None


class OddsApiClient:
    """Fetch NRL totals lines from the-odds-api.com.

    Constructed only inside the precompute job. Tests pass ``transport`` to
    override the HTTP layer with respx.
    """

    def __init__(
        self,
        api_key: str,
        *,
        max_retries: int = 3,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout
        self._transport = transport

    def fetch_totals(self) -> dict[tuple[str, str], TotalsLine]:
        """Return a ``{(home_nickname, away_nickname): TotalsLine}`` map.

        Picks the first bookmaker that quotes a totals market for each event,
        sorted by the order the-odds-api returns (typically by issue time).
        Multi-bookmaker line shopping is a future enhancement.

        Returns an empty dict on any error — totals are an enhancement, not a
        blocker, so the caller can always fall back to H2H-only tips.
        """
        try:
            data = self._get(
                f"{API_BASE_URL}/sports/{SPORT_KEY}/odds",
                params={
                    "apiKey": self._api_key,
                    "regions": REGIONS,
                    "markets": MARKETS,
                    "oddsFormat": "decimal",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("odds-api fetch failed (non-fatal): %s", exc)
            return {}

        out: dict[tuple[str, str], TotalsLine] = {}
        for event in data or []:
            home_raw = event.get("home_team") or ""
            away_raw = event.get("away_team") or ""
            home = _match_team(home_raw)
            away = _match_team(away_raw)
            if home is None or away is None:
                logger.debug("odds-api: skipping unknown teams %r vs %r", home_raw, away_raw)
                continue
            line = self._first_totals_line(event.get("bookmakers") or [])
            if line is not None:
                out[(home, away)] = line
        logger.info("odds-api: fetched totals for %d matches", len(out))
        return out

    @staticmethod
    def _first_totals_line(bookmakers: Iterable[dict]) -> TotalsLine | None:
        """Extract the first complete (over+under) totals line from a bookmaker list.

        Skips bookmakers that only quote one side — the de-vig math requires
        both prices to be meaningful.
        """
        for bm in bookmakers:
            for market in bm.get("markets") or []:
                if market.get("key") != "totals":
                    continue
                over: dict | None = None
                under: dict | None = None
                for outcome in market.get("outcomes") or []:
                    name = (outcome.get("name") or "").lower()
                    if name == "over":
                        over = outcome
                    elif name == "under":
                        under = outcome
                if over and under and over.get("point") == under.get("point"):
                    try:
                        return TotalsLine(
                            line=float(over["point"]),
                            overDecimalOdds=float(over["price"]),
                            underDecimalOdds=float(under["price"]),
                            bookmaker=str(bm.get("title") or bm.get("key") or "unknown"),
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        return None

    def _get(self, url: str, *, params: dict) -> list:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            if attempt:
                time.sleep(2**attempt)
            client_kwargs: dict = {"timeout": self._timeout}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            try:
                with httpx.Client(**client_kwargs) as client:
                    resp = client.get(url, params=params)
                if resp.status_code in _RETRYABLE_STATUS:
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    logger.warning(
                        "odds-api retry attempt=%d status=%d", attempt + 1, resp.status_code
                    )
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                logger.warning("odds-api network error attempt=%d: %s", attempt + 1, exc)
        raise RuntimeError(
            f"odds-api request failed after {self._max_retries} attempts"
        ) from last_exc
