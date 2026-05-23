import csv
import html as _html
import logging
import os
import pathlib
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from fantasy_coach import __version__
from fantasy_coach.betting.models import BettingTipsOut
from fantasy_coach.betting.store import (
    BettingTipsStore,
    FirestoreBettingTipsStore,
    get_betting_tips_store,
)
from fantasy_coach.config import get_repository
from fantasy_coach.job_runs import JobRunStore
from fantasy_coach.predictions import (
    FeatureContribution,
    FirestorePredictionStore,
    PredictionOut,
    PredictionStore,
    compute_bayesian_uncertainty,
    compute_whatif_base,
    get_prediction_store,
)
from fantasy_coach.storage.repository import Repository

logger = logging.getLogger(__name__)

_ACCURACY_THRESHOLD = 0.55


class RoundAccuracy(BaseModel):
    season: int
    round: int
    modelVersion: str
    total: int
    correct: int
    accuracy: float


class ModelVersionAccuracy(BaseModel):
    modelVersion: str
    total: int
    correct: int
    accuracy: float


class TeamOption(BaseModel):
    id: int
    name: str


class VenueOut(BaseModel):
    name: str
    city: str


class AccuracyOut(BaseModel):
    rounds: list[RoundAccuracy]
    byModelVersion: list[ModelVersionAccuracy]
    overallAccuracy: float | None
    belowThreshold: bool
    threshold: float
    scoredMatches: int
    # Filter-option catalogues — always reflects the full season (unfiltered)
    # so the frontend can populate dropdowns without a separate API call.
    teams: list[TeamOption]
    venues: list[str]
    modelVersions: list[str]


class TeamFormEntry(BaseModel):
    round: int
    matchId: int
    opponentId: int
    opponentName: str
    isHome: bool
    result: str  # "win" | "loss" | "draw"
    score: int
    opponentScore: int
    eloAfter: float
    eloDelta: float
    kickoff: str  # ISO 8601 UTC


class TeamFormHistory(BaseModel):
    teamId: int
    teamName: str
    season: int
    matches: list[TeamFormEntry]


class TeamProfile(BaseModel):
    teamId: int
    teamName: str
    season: int
    currentRecord: dict  # {"wins": n, "losses": n, "draws": n}
    currentElo: float
    eloTrend: str  # "up" | "down" | "flat"
    recentForm: list[str]  # last 10 results as "W"/"L"/"D"
    nextFixture: dict | None  # next upcoming match details
    allFixtures: list[dict]  # all fixtures in season


class DashboardOut(BaseModel):
    season: int
    currentRound: int | None
    favouriteTeamId: int | None
    nextFixture: dict | None
    untippedMatchIds: list[int]
    seasonAccuracy: float | None
    totalTips: int
    correctTips: int


class JobRunChangeOut(BaseModel):
    matchId: int
    homeTeam: str
    awayTeam: str
    prevHomeProb: float | None
    newHomeProb: float
    delta: float
    flipped: bool
    triggerSignal: str | None
    summary: str


class JobRunSummaryOut(BaseModel):
    flips: int
    meanAbsDelta: float
    maxAbsDelta: float


class JobRunListItem(BaseModel):
    id: str
    startedAt: str
    finishedAt: str | None
    trigger: str
    status: str
    season: int
    round: int
    matchesProcessed: int
    modelVersion: str
    error: str | None
    summary: JobRunSummaryOut


class JobRunListOut(BaseModel):
    runs: list[JobRunListItem]


class JobRunDetail(JobRunListItem):
    changes: list[JobRunChangeOut]


class UncertaintyOut(BaseModel):
    """Bayesian posterior predictive uncertainty for a single match (#144)."""

    matchId: int  # noqa: N815
    homeTeamId: int  # noqa: N815
    awayTeamId: int  # noqa: N815
    winProbability: float  # noqa: N815  posterior mean P(home wins)
    margin80ci: list[int]  # noqa: N815  [lo, hi] 80% HDI integers
    margin95ci: list[int]  # noqa: N815  [lo, hi] 95% HDI integers
    source: str  # "bayesian" | "fallback"


ALLOWED_ORIGINS_ENV = "FANTASY_COACH_ALLOWED_ORIGINS"
DEFAULT_ALLOWED_ORIGINS = (
    "https://fantasy.lopezcloud.dev,http://localhost:5173,http://localhost:4173"
)


def _allowed_origins() -> list[str]:
    raw = os.getenv(ALLOWED_ORIGINS_ENV, DEFAULT_ALLOWED_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def _prefetch_current_rounds() -> None:
    """Warm the Firestore prediction cache on startup with the 2 most recent rounds.

    Non-fatal: the regular per-request Firestore path still works on failure.
    Only runs when STORAGE_BACKEND=firestore; SQLite dev is a no-op.
    """
    try:
        store = _get_store()
        if isinstance(store, FirestorePredictionStore):
            store.prefetch_recent(n_rounds=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("prefetch_current_rounds failed (non-fatal): %s", exc)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _prefetch_current_rounds()
    yield


app = FastAPI(
    title="Fantasy Coach",
    version=__version__,
    description="NRL match prediction API. All non-health endpoints require a Firebase ID token.",
    lifespan=_lifespan,
)

# Enable Firebase token verification when a project ID is configured.
# Omitting FIREBASE_PROJECT_ID disables the check (useful for local SQLite dev).
#
# Middleware ordering: the last-added middleware runs first on inbound requests,
# so CORSMiddleware must be added after FirebaseAuthMiddleware. This lets the
# browser's preflight OPTIONS request short-circuit inside CORS before auth
# sees it — auth middleware only understands Bearer tokens and would 401 the
# preflight otherwise.
if os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"):
    # Lazy import: firebase_admin (~85 ms) is only needed when auth is enabled.
    # This keeps /healthz cold-start below 100 ms in environments without auth.
    from fantasy_coach.auth import FirebaseAuthMiddleware  # noqa: PLC0415

    app.add_middleware(FirebaseAuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

# Module-level singletons — created lazily on first use.
_store: PredictionStore | FirestorePredictionStore | None = None
_repo: Repository | None = None
_job_run_store: JobRunStore | None = None
_betting_tips_store: BettingTipsStore | FirestoreBettingTipsStore | None = None

# In-process cache for team profile responses: key=(team_id, season), value=(timestamp, data)
_PROFILE_CACHE_TTL = 60  # seconds
_profile_cache: dict[tuple[int, int], tuple[float, TeamProfile]] = {}

# In-process cache for simulation results: key=season, value=(timestamp, result_dict)
_SIM_CACHE_TTL = 300  # 5 minutes
_sim_cache: dict[int, tuple[float, dict]] = {}

# In-process cache for dashboard responses: key=(uid, season), value=(timestamp, data)
_DASHBOARD_CACHE_TTL = 60  # seconds
_dashboard_cache: dict[tuple[str, int], tuple[float, DashboardOut]] = {}


def _get_store() -> PredictionStore | FirestorePredictionStore:
    global _store
    if _store is None:
        _store = get_prediction_store()
    return _store


def _get_repo() -> Repository:
    global _repo
    if _repo is None:
        _repo = get_repository()
    return _repo


def _get_betting_tips_store() -> BettingTipsStore | FirestoreBettingTipsStore:
    global _betting_tips_store
    if _betting_tips_store is None:
        _betting_tips_store = get_betting_tips_store()
    return _betting_tips_store


def _get_job_run_store() -> JobRunStore | None:
    """Return the audit-log store, or None when running on SQLite (local dev).

    The job_runs collection is Firestore-only (#244 PR A) so we can't surface
    it from a local sqlite dev server — the endpoints respond with an empty
    list / 404 in that case.
    """
    global _job_run_store
    if os.getenv("STORAGE_BACKEND", "sqlite").lower() != "firestore":
        return None
    if _job_run_store is None:
        _job_run_store = JobRunStore(project=os.getenv("FIREBASE_PROJECT_ID"))
    return _job_run_store


def _job_run_doc_to_list_item(doc: dict) -> dict:
    """Map a Firestore doc (snake_case) to the camelCase API shape, no changes[]."""
    summary = doc.get("summary") or {}
    return {
        "id": doc["id"],
        "startedAt": doc.get("started_at", ""),
        "finishedAt": doc.get("finished_at"),
        "trigger": doc.get("trigger", "scheduled"),
        "status": doc.get("status", "success"),
        "season": doc.get("season", 0),
        "round": doc.get("round", 0),
        "matchesProcessed": doc.get("matches_processed", 0),
        "modelVersion": doc.get("model_version", ""),
        "error": doc.get("error"),
        "summary": {
            "flips": summary.get("flips", 0),
            "meanAbsDelta": summary.get("mean_abs_delta", 0.0),
            "maxAbsDelta": summary.get("max_abs_delta", 0.0),
        },
    }


def _job_run_change_to_out(change: dict) -> dict:
    return {
        "matchId": change.get("match_id", 0),
        "homeTeam": change.get("home_team", ""),
        "awayTeam": change.get("away_team", ""),
        "prevHomeProb": change.get("prev_home_prob"),
        "newHomeProb": change.get("new_home_prob", 0.0),
        "delta": change.get("delta", 0.0),
        "flipped": bool(change.get("flipped", False)),
        "triggerSignal": change.get("trigger_signal"),
        "summary": change.get("summary", ""),
    }


def _annotate_results(
    predictions: list[PredictionOut], season: int, round_: int
) -> list[PredictionOut]:
    """Attach actualWinner from the match repo (FullTime matches only)."""
    try:
        matches = {m.match_id: m for m in _get_repo().list_matches(season, round_)}
    except Exception:
        return predictions  # repo unavailable — return predictions as-is
    result = []
    for p in predictions:
        m = matches.get(p.matchId)
        if (
            m
            and m.match_state == "FullTime"
            and m.home.score is not None
            and m.away.score is not None
        ):
            winner = "home" if m.home.score > m.away.score else "away"
            p = p.model_copy(update={"actualWinner": winner})
        result.append(p)
    return result


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get(
    "/predictions",
    response_model=list[PredictionOut],
    summary="Get predictions for a season/round",
    description=(
        "Returns predicted winner and home-win probability for every match in "
        "the requested round. Predictions are precomputed twice a week by a "
        "Cloud Run Job (Tue 09:00 and Thu 06:00 AEST); this endpoint is a "
        "cache read only. Returns 503 with a retry hint if the cache is empty."
    ),
)
def get_predictions(
    season: int = Query(..., description="NRL season year, e.g. 2026"),
    round: int = Query(..., description="Round number, e.g. 7", alias="round"),
) -> list[PredictionOut]:
    cached = _get_store().get(season, round)
    if not cached:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No cached predictions for season {season} round {round}. "
                "The precompute job runs Tue 09:00 AEST and Thu 06:00 AEST. "
                "Retry in a few minutes or trigger it manually with "
                "`gcloud run jobs execute fantasy-coach-precompute`."
            ),
        )
    return _annotate_results(cached, season, round)


@app.get(
    "/betting-tips",
    response_model=BettingTipsOut,
    summary="Top betting tips for a season/round",
    description=(
        "Returns the precomputed 2-singles + 2-doubles + 2-trebles card for the "
        "requested round. Tips combine cached model probabilities with bookmaker "
        "odds (head-to-head from the nrl.com scrape; totals from the-odds-api "
        "when configured). Computed twice a week by the Cloud Run Job; this "
        "endpoint is a cache read only. Returns 503 with a retry hint when the "
        "cache is empty. Picks are educational, not financial advice — gamble "
        "responsibly."
    ),
)
def get_betting_tips(
    season: int = Query(..., description="NRL season year, e.g. 2026"),
    round: int = Query(..., description="Round number, e.g. 7", alias="round"),
) -> BettingTipsOut:
    tips = _get_betting_tips_store().get(season, round)
    if tips is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No cached betting tips for season {season} round {round}. "
                "The precompute job runs Tue 09:00 AEST and Thu 06:00 AEST. "
                "Retry in a few minutes or trigger it manually with "
                "`gcloud run jobs execute fantasy-coach-precompute`."
            ),
            headers={"Retry-After": "3600"},
        )
    return tips


@app.get(
    "/predictions/{match_id}/uncertainty",
    response_model=UncertaintyOut,
    summary="Bayesian posterior predictive uncertainty for a match",
    description=(
        "Returns the Bayesian hierarchical model's posterior predictive win probability "
        "and 80%/95% margin HDI for the given match. Requires a saved Bayesian model "
        "artefact at FANTASY_COACH_BAYESIAN_MODEL_PATH (or artifacts/bayesian.joblib). "
        "Falls back to the stored XGBoost win probability and wide symmetric intervals "
        "when the Bayesian model is unavailable."
    ),
)
def get_match_uncertainty(
    match_id: int,
    season: int = Query(..., description="NRL season year, e.g. 2026"),
) -> UncertaintyOut:
    try:
        matches = _get_repo().list_matches(season)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Match store unavailable: {exc}") from exc

    match = next((m for m in matches if m.match_id == match_id), None)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Match {match_id} not found in season {season}.",
        )

    result = compute_bayesian_uncertainty(match.home.team_id, match.away.team_id)
    if result is not None:
        return UncertaintyOut(
            matchId=match_id,
            homeTeamId=match.home.team_id,
            awayTeamId=match.away.team_id,
            winProbability=result["win_probability"],
            margin80ci=result["margin_80ci"],
            margin95ci=result["margin_95ci"],
            source="bayesian",
        )

    # Bayesian model unavailable — fall back to cached XGBoost prediction.
    fallback_prob = 0.5
    try:
        stored = _get_store().get(season, match.round)
        pred = next((p for p in stored if p.matchId == match_id), None)
        if pred is not None:
            fallback_prob = pred.homeWinProbability
    except Exception:
        pass

    return UncertaintyOut(
        matchId=match_id,
        homeTeamId=match.home.team_id,
        awayTeamId=match.away.team_id,
        winProbability=fallback_prob,
        margin80ci=[-15, 15],
        margin95ci=[-25, 25],
        source="fallback",
    )


@app.get(
    "/accuracy",
    response_model=AccuracyOut,
    summary="Rolling model accuracy over recent rounds",
    description=(
        "Returns per-round and per-model-version accuracy for the last N completed "
        "rounds of the given season. A round is considered complete when all its "
        "matches have match_state=FullTime in the match store."
    ),
)
def get_accuracy(
    season: int = Query(..., description="NRL season year, e.g. 2026"),
    last_n_rounds: int = Query(
        default=10, ge=1, le=27, description="How many recent completed rounds to include"
    ),
    team_id: int | None = Query(
        default=None, description="Filter to matches involving this team ID"
    ),
    venue: str | None = Query(
        default=None, description="Filter to matches at this venue (case-insensitive substring)"
    ),
    model_version: str | None = Query(
        default=None,
        description="Filter predictions by model version prefix (first 8+ hex chars)",
    ),
) -> AccuracyOut:
    try:
        matches = _get_repo().list_matches(season)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Match store unavailable: {exc}") from exc

    # Collect filter-option catalogues from all season matches (unfiltered).
    teams_seen: dict[int, str] = {}
    venues_seen: set[str] = set()
    for m in matches:
        teams_seen[m.home.team_id] = m.home.name
        teams_seen[m.away.team_id] = m.away.name
        if m.venue:
            venues_seen.add(m.venue)

    teams_catalogue = sorted(
        [TeamOption(id=tid, name=name) for tid, name in teams_seen.items()],
        key=lambda t: t.name,
    )
    venues_catalogue = sorted(venues_seen)

    # Build result set respecting team/venue filters.
    venue_lower = venue.lower() if venue else None
    results: dict[int, str] = {}
    completed_match_ids_by_round: dict[int, list[int]] = {}
    for m in matches:
        if team_id is not None and m.home.team_id != team_id and m.away.team_id != team_id:
            continue
        if venue_lower is not None and (m.venue is None or venue_lower not in m.venue.lower()):
            continue
        if m.match_state == "FullTime" and m.home.score is not None and m.away.score is not None:
            winner = "home" if m.home.score > m.away.score else "away"
            results[m.match_id] = winner
            completed_match_ids_by_round.setdefault(m.round, []).append(m.match_id)

    # Take the most-recent N rounds that have at least one FullTime result
    completed_rounds = sorted(completed_match_ids_by_round.keys(), reverse=True)[:last_n_rounds]
    completed_rounds.reverse()  # oldest-first for charting

    round_accuracy_list: list[RoundAccuracy] = []
    mv_stats: dict[str, dict[str, int]] = {}
    mv_catalogue: set[str] = set()
    total_correct = 0
    total_scored = 0

    for round_ in completed_rounds:
        preds = _get_store().get(season, round_)
        if not preds:
            continue

        # Collect all model versions seen (pre-filter) for the catalogue.
        for p in preds:
            mv_catalogue.add(p.modelVersion)

        # Apply model-version filter if specified.
        if model_version is not None:
            preds = [p for p in preds if p.modelVersion.startswith(model_version)]

        # Dominant model version for this round (by plurality of predictions)
        mv_counts: dict[str, int] = {}
        for p in preds:
            mv_counts[p.modelVersion] = mv_counts.get(p.modelVersion, 0) + 1
        if not mv_counts:
            continue
        dominant_mv = max(mv_counts, key=lambda k: mv_counts[k])

        scored = [(p, results[p.matchId]) for p in preds if p.matchId in results]
        n_total = len(scored)
        if n_total == 0:
            continue
        n_correct = sum(1 for p, actual in scored if p.predictedWinner == actual)
        accuracy = n_correct / n_total

        round_accuracy_list.append(
            RoundAccuracy(
                season=season,
                round=round_,
                modelVersion=dominant_mv,
                total=n_total,
                correct=n_correct,
                accuracy=accuracy,
            )
        )

        for p, actual in scored:
            mv = p.modelVersion
            if mv not in mv_stats:
                mv_stats[mv] = {"total": 0, "correct": 0}
            mv_stats[mv]["total"] += 1
            if p.predictedWinner == actual:
                mv_stats[mv]["correct"] += 1

        total_correct += n_correct
        total_scored += n_total

    overall_accuracy = total_correct / total_scored if total_scored > 0 else None

    by_model_version = [
        ModelVersionAccuracy(
            modelVersion=mv,
            total=s["total"],
            correct=s["correct"],
            accuracy=s["correct"] / s["total"] if s["total"] > 0 else 0.0,
        )
        for mv, s in mv_stats.items()
    ]

    return AccuracyOut(
        rounds=round_accuracy_list,
        byModelVersion=by_model_version,
        overallAccuracy=overall_accuracy,
        belowThreshold=(overall_accuracy is not None and overall_accuracy < _ACCURACY_THRESHOLD),
        threshold=_ACCURACY_THRESHOLD,
        scoredMatches=total_scored,
        teams=teams_catalogue,
        venues=venues_catalogue,
        modelVersions=sorted(mv_catalogue),
    )


@app.get(
    "/teams/{team_id}/form",
    response_model=TeamFormHistory,
    summary="Team form history with Elo progression",
    description=(
        "Returns the last N completed matches for a team, with per-match results "
        "and EloMOV rating after each match. Elo is computed by replaying the full "
        "season history in chronological order so all opponents' ratings are accurate."
    ),
)
def get_team_form(
    team_id: int,
    season: int = Query(..., description="NRL season year, e.g. 2026"),
    last: int = Query(default=20, ge=1, le=40, description="Number of recent matches to return"),
) -> TeamFormHistory:
    try:
        all_matches = _get_repo().list_matches(season)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Match store unavailable: {exc}") from exc

    completed = sorted(
        [
            m
            for m in all_matches
            if m.match_state == "FullTime" and m.home.score is not None and m.away.score is not None
        ],
        key=lambda m: (m.start_time, m.match_id),
    )

    from fantasy_coach.models.elo_mov import EloMOV  # noqa: PLC0415

    elo = EloMOV()
    entries: list[TeamFormEntry] = []
    team_name = ""

    for m in completed:
        home_delta, away_delta = elo.update(
            m.home.team_id,
            m.away.team_id,
            m.home.score,
            m.away.score,  # type: ignore[arg-type]
        )

        is_home = m.home.team_id == team_id
        is_away = m.away.team_id == team_id
        if not (is_home or is_away):
            continue

        if is_home:
            score, opp_score = m.home.score, m.away.score
            opp_id, opp_name = m.away.team_id, m.away.name
            delta = home_delta
            if not team_name:
                team_name = m.home.name
        else:
            score, opp_score = m.away.score, m.home.score
            opp_id, opp_name = m.home.team_id, m.home.name
            delta = away_delta
            if not team_name:
                team_name = m.away.name

        assert score is not None and opp_score is not None  # narrowed above
        if score > opp_score:
            result = "win"
        elif score < opp_score:
            result = "loss"
        else:
            result = "draw"

        entries.append(
            TeamFormEntry(
                round=m.round,
                matchId=m.match_id,
                opponentId=opp_id,
                opponentName=opp_name,
                isHome=is_home,
                result=result,
                score=score,
                opponentScore=opp_score,
                eloAfter=round(elo.rating(team_id), 1),
                eloDelta=round(delta, 1),
                kickoff=m.start_time.isoformat(),
            )
        )

    if not team_name:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found in season {season}")

    return TeamFormHistory(
        teamId=team_id,
        teamName=team_name,
        season=season,
        matches=entries[-last:],
    )


@app.get(
    "/teams/{team_id}/profile",
    response_model=TeamProfile,
    summary="Team profile: record, Elo trend, recent form, and fixtures",
    description=(
        "Returns a team's season record, current Elo rating with trend, "
        "last-10-match form string, next upcoming fixture (with prediction if cached), "
        "and a full list of season fixtures."
    ),
)
def get_team_profile(
    team_id: int,
    season: int = Query(..., description="NRL season year, e.g. 2026"),
) -> TeamProfile:
    # Serve from in-process cache if still fresh.
    cache_key = (team_id, season)
    cached_entry = _profile_cache.get(cache_key)
    if cached_entry is not None:
        ts, cached_profile = cached_entry
        if time.monotonic() - ts < _PROFILE_CACHE_TTL:
            return cached_profile

    try:
        all_matches = _get_repo().list_matches(season)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Match store unavailable: {exc}") from exc

    # Sort chronologically to walk forward for Elo computation.
    sorted_matches = sorted(all_matches, key=lambda m: (m.start_time, m.match_id))

    from fantasy_coach.models.elo_mov import EloMOV  # noqa: PLC0415

    elo = EloMOV()
    team_name = ""
    wins = losses = draws = 0
    form_entries: list[str] = []  # "W"/"L"/"D" for completed matches
    all_fixtures: list[dict] = []
    completed_entries: list[dict] = []  # for Elo trend (last 3)

    for m in sorted_matches:
        is_home = m.home.team_id == team_id
        is_away = m.away.team_id == team_id
        if not (is_home or is_away):
            # Still update Elo for all teams to keep ratings accurate.
            if (
                m.match_state == "FullTime"
                and m.home.score is not None
                and m.away.score is not None
            ):
                elo.update(m.home.team_id, m.away.team_id, m.home.score, m.away.score)
            continue

        # Resolve team name on first encounter.
        if not team_name:
            team_name = m.home.name if is_home else m.away.name

        opp_id = m.away.team_id if is_home else m.home.team_id
        opp_name = m.away.name if is_home else m.home.name

        if m.match_state == "FullTime" and m.home.score is not None and m.away.score is not None:
            elo.update(m.home.team_id, m.away.team_id, m.home.score, m.away.score)

            score = m.home.score if is_home else m.away.score
            opp_score = m.away.score if is_home else m.home.score

            if score > opp_score:
                wins += 1
                form_char = "W"
            elif score < opp_score:
                losses += 1
                form_char = "L"
            else:
                draws += 1
                form_char = "D"

            form_entries.append(form_char)
            elo_after = round(elo.rating(team_id), 1)
            completed_entries.append({"elo": elo_after})

            all_fixtures.append(
                {
                    "round": m.round,
                    "matchId": m.match_id,
                    "opponent": opp_name,
                    "opponentId": opp_id,
                    "isHome": is_home,
                    "kickoff": m.start_time.isoformat(),
                    "result": form_char,
                    "score": score,
                    "opponentScore": opp_score,
                }
            )
        else:
            all_fixtures.append(
                {
                    "round": m.round,
                    "matchId": m.match_id,
                    "opponent": opp_name,
                    "opponentId": opp_id,
                    "isHome": is_home,
                    "kickoff": m.start_time.isoformat(),
                    "result": None,
                    "score": None,
                    "opponentScore": None,
                }
            )

    if not team_name:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found in season {season}")

    # Elo trend based on last 3 completed matches.
    current_elo = round(elo.rating(team_id), 1)
    if len(completed_entries) >= 2:
        elo_now = completed_entries[-1]["elo"]
        elo_before = (
            completed_entries[-3]["elo"]
            if len(completed_entries) >= 3
            else completed_entries[-2]["elo"]
        )
        if elo_now > elo_before + 1:
            elo_trend = "up"
        elif elo_now < elo_before - 1:
            elo_trend = "down"
        else:
            elo_trend = "flat"
    else:
        elo_trend = "flat"

    # Last 10 results as form pills.
    recent_form = form_entries[-10:]

    # Next fixture: first match not yet FullTime for this team.
    next_fixture: dict | None = None
    for fixture in all_fixtures:
        if fixture["result"] is None:
            pred_winner = None
            pred_prob = None
            try:
                preds = _get_store().get(season, fixture["round"])
                for pred in preds:
                    if pred.matchId == fixture["matchId"]:
                        pred_winner = pred.predictedWinner
                        pred_prob = pred.homeWinProbability
                        break
            except Exception:
                pass
            next_fixture = {
                "matchId": fixture["matchId"],
                "round": fixture["round"],
                "opponent": fixture["opponent"],
                "opponentId": fixture["opponentId"],
                "isHome": fixture["isHome"],
                "kickoff": fixture["kickoff"],
                "predWinner": pred_winner,
                "predProb": pred_prob,
            }
            break

    # Enrich upcoming fixtures in allFixtures with predictions.
    try:
        upcoming_rounds: dict[int, list[dict]] = {}
        for fixture in all_fixtures:
            if fixture["result"] is None:
                upcoming_rounds.setdefault(fixture["round"], []).append(fixture)
        for round_num, fixtures_in_round in upcoming_rounds.items():
            try:
                preds = _get_store().get(season, round_num)
                pred_map = {p.matchId: p for p in preds}
                for fixture in fixtures_in_round:
                    pred = pred_map.get(fixture["matchId"])
                    if pred:
                        fixture["predWinner"] = pred.predictedWinner
                        fixture["predProb"] = pred.homeWinProbability
            except Exception:
                pass
    except Exception:
        pass

    profile = TeamProfile(
        teamId=team_id,
        teamName=team_name,
        season=season,
        currentRecord={"wins": wins, "losses": losses, "draws": draws},
        currentElo=current_elo,
        eloTrend=elo_trend,
        recentForm=recent_form,
        nextFixture=next_fixture,
        allFixtures=all_fixtures,
    )

    _profile_cache[cache_key] = (time.monotonic(), profile)
    return profile


def _require_auth(request: Request) -> str:
    uid: str | None = getattr(request.state, "uid", None)
    return uid if uid is not None else "__dev__"


class NotificationSubscribeIn(BaseModel):
    token: str
    platform: str = "web"
    timezone: str | None = None  # IANA timezone string, e.g. "Australia/Sydney"


# ---------------------------------------------------------------------------
# Groups / Leaderboard models
# ---------------------------------------------------------------------------


class GroupIn(BaseModel):
    name: str


class GroupOut(BaseModel):
    gid: str
    name: str
    inviteCode: str
    ownerUid: str
    memberCount: int
    createdAt: str | None = None


class JoinGroupIn(BaseModel):
    inviteCode: str


class LeaderboardEntry(BaseModel):
    rank: int
    uid: str
    displayName: str
    wins: int
    losses: int
    totalTips: int
    accuracy: float
    marginPoints: float
    currentStreak: int
    longestStreak: int


class LeaderboardOut(BaseModel):
    season: int
    groupId: str | None
    entries: list[LeaderboardEntry]


@app.get(
    "/me/dashboard",
    response_model=DashboardOut,
    summary="Personalised dashboard for the authenticated user",
    description=(
        "Returns the current round, upcoming untipped matches, next fixture "
        "for the user's favourite team, and season-to-date tipping accuracy."
    ),
)
def get_dashboard(
    request: Request,
    season: int = Query(..., description="NRL season year, e.g. 2026"),
    favourite_team_id: int | None = Query(
        default=None, description="Team ID to use for next-fixture lookup"
    ),
) -> DashboardOut:
    uid = _require_auth(request)
    cache_key = (uid, season)
    cached_entry = _dashboard_cache.get(cache_key)
    if cached_entry is not None:
        ts, cached_dash = cached_entry
        if time.monotonic() - ts < _DASHBOARD_CACHE_TTL:
            return cached_dash

    try:
        matches = _get_repo().list_matches(season)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Match store unavailable: {exc}") from exc

    # Determine current round: lowest round with at least one non-FullTime match,
    # falling back to the highest round seen when all matches are complete.
    rounds_with_upcoming: list[int] = []
    all_rounds: list[int] = []
    for m in matches:
        all_rounds.append(m.round)
        if m.match_state != "FullTime":
            rounds_with_upcoming.append(m.round)

    if rounds_with_upcoming:
        current_round: int | None = min(rounds_with_upcoming)
    elif all_rounds:
        current_round = max(all_rounds)
    else:
        current_round = None

    # User's own tips for this season — keyed by match_id so we can
    # diff against the fixture list and against actual results.
    # Empty when uid is the dev fallback or when STORAGE_BACKEND != firestore
    # (tips are only persisted client-side via the Firebase JS SDK).
    user_tips = _fetch_user_tips(uid, season)

    # Untipped = current-round matches the *user* hasn't picked yet.
    # (Earlier this was wired to the prediction cache, which silently
    # collapsed once the precompute job populated predictions.)
    untipped_match_ids: list[int] = []
    if current_round is not None:
        current_round_matches = [m for m in matches if m.round == current_round]
        untipped_match_ids = [
            m.match_id for m in current_round_matches if m.match_id not in user_tips
        ]

    # Season-to-date tipping accuracy: compare *user tips* to FullTime results.
    result_map: dict[int, str] = {
        m.match_id: ("home" if m.home.score > m.away.score else "away")  # type: ignore[operator]
        for m in matches
        if m.match_state == "FullTime" and m.home.score is not None and m.away.score is not None
    }
    total_tips = 0
    correct_tips = 0
    for mid, choice in user_tips.items():
        if mid in result_map:
            total_tips += 1
            if choice == result_map[mid]:
                correct_tips += 1

    season_accuracy = correct_tips / total_tips if total_tips > 0 else None

    # Next fixture for favourite team.
    next_fixture: dict | None = None
    if favourite_team_id is not None:
        sorted_matches = sorted(matches, key=lambda m: (m.start_time, m.match_id))
        for m in sorted_matches:
            is_home = m.home.team_id == favourite_team_id
            is_away = m.away.team_id == favourite_team_id
            if not (is_home or is_away):
                continue
            if m.match_state == "FullTime":
                continue
            opp_name = m.away.name if is_home else m.home.name
            opp_id = m.away.team_id if is_home else m.home.team_id
            pred_winner = None
            pred_prob = None
            try:
                preds = _get_store().get(season, m.round)
                for pred in preds:
                    if pred.matchId == m.match_id:
                        pred_winner = pred.predictedWinner
                        pred_prob = pred.homeWinProbability
                        break
            except Exception:
                pass
            next_fixture = {
                "matchId": m.match_id,
                "round": m.round,
                "season": season,
                "opponent": opp_name,
                "opponentId": opp_id,
                "isHome": is_home,
                "kickoff": m.start_time.isoformat(),
                "predWinner": pred_winner,
                "predProb": pred_prob,
            }
            break

    dashboard = DashboardOut(
        season=season,
        currentRound=current_round,
        favouriteTeamId=favourite_team_id,
        nextFixture=next_fixture,
        untippedMatchIds=untipped_match_ids,
        seasonAccuracy=season_accuracy,
        totalTips=total_tips,
        correctTips=correct_tips,
    )

    _dashboard_cache[cache_key] = (time.monotonic(), dashboard)
    return dashboard


@app.get(
    "/season/{season}/simulation",
    summary="Monte Carlo season simulation",
    description=(
        "Runs (or serves a cached) 10 000-simulation Monte Carlo season forecast. "
        "Returns per-team probabilities for top-8 / top-4 / minor-premiership / "
        "grand-final / premiership, plus the current regular-season ladder derived "
        "from completed matches. Cached for 5 minutes in-process."
    ),
)
def get_season_simulation(season: int) -> dict:
    cache_hit = _sim_cache.get(season)
    if cache_hit is not None:
        ts, cached = cache_hit
        if time.monotonic() - ts < _SIM_CACHE_TTL:
            return cached

    from datetime import UTC  # noqa: PLC0415
    from datetime import datetime as _dt  # noqa: PLC0415

    from fantasy_coach.simulation import simulate_season  # noqa: PLC0415

    try:
        all_matches = _get_repo().list_matches(season)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Match store unavailable: {exc}") from exc

    # Collect precomputed win probabilities from the prediction store.
    predictions: dict[int, float] = {}
    rounds = {m.round for m in all_matches}
    for round_ in rounds:
        try:
            for p in _get_store().get(season, round_):
                predictions[p.matchId] = p.homeWinProbability
        except Exception:
            pass

    sim = simulate_season(season, all_matches, predictions)

    # Build current standings from completed matches.
    standings_data: dict[int, dict] = {}
    for m in all_matches:
        for tid, name in ((m.home.team_id, m.home.name), (m.away.team_id, m.away.name)):
            if tid not in standings_data:
                standings_data[tid] = {
                    "teamId": tid,
                    "teamName": name,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "points": 0,
                    "pctFor": 0,
                    "pctAgainst": 0,
                }
        if m.home.score is not None and m.away.score is not None:
            h, a = standings_data[m.home.team_id], standings_data[m.away.team_id]
            hs, as_ = int(m.home.score), int(m.away.score)
            h["pctFor"] += hs
            h["pctAgainst"] += as_
            a["pctFor"] += as_
            a["pctAgainst"] += hs
            if hs > as_:
                h["wins"] += 1
                h["points"] += 2
                a["losses"] += 1
            elif hs < as_:
                a["wins"] += 1
                a["points"] += 2
                h["losses"] += 1
            else:
                h["draws"] += 1
                h["points"] += 1
                a["draws"] += 1
                a["points"] += 1

    def _pct(row: dict) -> float:
        return (row["pctFor"] / row["pctAgainst"] * 100.0) if row["pctAgainst"] else 0.0

    standings = sorted(
        standings_data.values(),
        key=lambda r: (-r["points"], -_pct(r)),
    )
    for pos, row in enumerate(standings, start=1):
        row["position"] = pos
        row["percentage"] = round(_pct(row), 1)

    out = {
        **sim.as_dict(),
        "computedAt": _dt.now(UTC).isoformat(),
        "standings": standings,
    }
    _sim_cache[season] = (time.monotonic(), out)
    return out


# ---------------------------------------------------------------------------
# What-if prediction explorer (#150)
# ---------------------------------------------------------------------------

# Feature names the what-if endpoint accepts as overrides. Deliberately
# restricted to the human-interpretable subset that has a direct UI knob.
_WHATIF_ALLOWED_OVERRIDES = frozenset(
    {
        "is_wet",
        "wind_kph",
        "temperature_c",
        "rain_intensity",
        "days_rest_diff",
    }
)

# Per-user rate limiter: at most 20 requests per 60-second sliding window.
_WHATIF_RATE_LIMIT = 20
_WHATIF_RATE_WINDOW = 60.0
_whatif_rate: dict[str, deque[float]] = {}

# Short-lived result cache keyed by (match_id, sorted_overrides_str).
_WHATIF_RESULT_CACHE_TTL = 300.0
_whatif_result_cache: dict[tuple[int, str], tuple[float, "WhatIfOut"]] = {}


class WhatIfIn(BaseModel):
    season: int
    round: int  # noqa: A003
    overrides: dict[str, float] = {}


class WhatIfOut(BaseModel):  # noqa: N815 (camelCase fields match API convention)
    baseHomeWinProbability: float
    homeWinProbability: float
    predictedWinner: str
    contributions: list[FeatureContribution] | None = None


@app.post(
    "/predictions/{match_id}/whatif",
    response_model=WhatIfOut,
    summary="What-if prediction explorer",
    description=(
        "Re-runs the model for a specific match with one or more feature overrides "
        "applied. Accepts weather and rest-days adjustments and returns the updated "
        "home-win probability alongside the per-feature contribution breakdown. "
        "Throttled to 20 calls/minute per authenticated user. "
        "Results are cached for 5 minutes by (match_id, overrides)."
    ),
)
def get_whatif_prediction(
    match_id: int,
    body: WhatIfIn,
    request: Request,
) -> WhatIfOut:
    uid = _require_auth(request)

    # Sliding-window rate limit.
    now = time.monotonic()
    window = _whatif_rate.setdefault(uid, deque())
    while window and now - window[0] > _WHATIF_RATE_WINDOW:
        window.popleft()
    if len(window) >= _WHATIF_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded: 20 what-if requests per minute.",
        )
    window.append(now)

    # Reject unknown override keys immediately so callers get a clear error.
    unknown = set(body.overrides) - _WHATIF_ALLOWED_OVERRIDES
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown override feature(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_WHATIF_ALLOWED_OVERRIDES)}",
        )

    # Check short-lived result cache.
    overrides_key = str(sorted(body.overrides.items()))
    result_key = (match_id, overrides_key)
    cached_result = _whatif_result_cache.get(result_key)
    if cached_result is not None and now - cached_result[0] < _WHATIF_RESULT_CACHE_TTL:
        return cached_result[1]

    # We require a stored base prediction to (a) confirm the match exists in the
    # prediction cache and (b) provide the baseline probability for the response.
    stored_preds = _get_store().get(body.season, body.round)
    base_pred = next((p for p in stored_preds if p.matchId == match_id), None)
    if base_pred is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stored prediction for match {match_id} in {body.season} r{body.round}.",
        )

    from fantasy_coach.feature_engineering import FEATURE_NAMES  # noqa: PLC0415
    from fantasy_coach.models.loader import load_model  # noqa: PLC0415
    from fantasy_coach.predictions import (  # noqa: PLC0415
        DEFAULT_MODEL_PATH,
        MODEL_PATH_ENV,
        _apply_market_shrinkage,
        _compute_contributions,
    )

    model_path = Path(os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))

    try:
        x, _ = compute_whatif_base(match_id, body.season, body.round, _get_repo(), model_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    loaded = load_model(model_path)
    feature_names: tuple[str, ...] = getattr(loaded, "feature_names", None) or FEATURE_NAMES

    x_overridden = x.copy()
    weather_overridden = any(
        k in body.overrides for k in ("is_wet", "wind_kph", "temperature_c", "rain_intensity")
    )

    for feat, value in body.overrides.items():
        if feat in feature_names:
            x_overridden[0, feature_names.index(feat)] = float(value)

    # When any weather feature is overridden, clear missing_weather so the
    # model treats the provided values as real data, not imputed defaults.
    if weather_overridden and "missing_weather" in feature_names:
        x_overridden[0, feature_names.index("missing_weather")] = 0.0

    raw_prob = round(float(loaded.predict_home_win_prob(x_overridden)[0]), 4)
    new_prob, _ = _apply_market_shrinkage(raw_prob, x_overridden, feature_names)

    contribs = _compute_contributions(loaded, x_overridden)

    result = WhatIfOut(
        baseHomeWinProbability=base_pred.homeWinProbability,
        homeWinProbability=new_prob,
        predictedWinner="home" if new_prob >= 0.5 else "away",
        contributions=contribs,
    )

    _whatif_result_cache[result_key] = (now, result)
    return result


@app.get(
    "/job-runs",
    response_model=JobRunListOut,
    summary="List recent precompute job runs",
    description=(
        "Returns the most recent precompute Job invocations with their aggregate "
        "summary (flips, mean/max absolute delta) but without the per-match "
        "changes[] array. Use GET /job-runs/{run_id} for the detail view. "
        "Empty list when running on local SQLite — the audit log is Firestore-only."
    ),
)
def list_job_runs(
    limit: int = Query(20, ge=1, le=100, description="Max number of runs to return."),
) -> JobRunListOut:
    store = _get_job_run_store()
    if store is None:
        return JobRunListOut(runs=[])
    rows = store.list_recent(limit=limit)
    return JobRunListOut(runs=[JobRunListItem(**_job_run_doc_to_list_item(r)) for r in rows])


@app.get(
    "/job-runs/{run_id}",
    response_model=JobRunDetail,
    summary="Get a single precompute job run with per-match change diffs",
    description=(
        "Returns the full job_runs doc, including the inline changes[] array. "
        "Each change carries prev/new home win probability, delta, a flipped "
        "flag, an optional trigger_signal label, and a deterministic one-line "
        "human-readable summary."
    ),
)
def get_job_run(run_id: str) -> JobRunDetail:
    store = _get_job_run_store()
    if store is None:
        raise HTTPException(
            status_code=404,
            detail="Job-run audit log is unavailable on this backend.",
        )
    doc = store.get(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Job run {run_id} not found.")
    list_item = _job_run_doc_to_list_item(doc)
    changes = [JobRunChangeOut(**_job_run_change_to_out(c)) for c in doc.get("changes", [])]
    return JobRunDetail(**list_item, changes=changes)


_VENUES_CSV = pathlib.Path(__file__).parents[3] / "data" / "venues.csv"

# Lazily populated teams cache: maps season → sorted TeamOption list.
_teams_cache: dict[int, list[TeamOption]] = {}


@app.get(
    "/venues",
    response_model=list[VenueOut],
    summary="List all known NRL venues",
    description=(
        "Returns venue name and city from the bundled venues.csv."
        " Used by the client-side search index."
    ),
)
def get_venues() -> list[VenueOut]:
    if not _VENUES_CSV.exists():
        return []
    with _VENUES_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        return [VenueOut(name=row["name"], city=row["city"]) for row in reader]


@app.get(
    "/teams",
    response_model=list[TeamOption],
    summary="List all NRL teams seen in a season",
    description=(
        "Returns `{id, name}` for every team seen in the given season. "
        "Derived from the match store — same data as the teams catalogue in /accuracy."
    ),
)
def get_teams(
    season: int = Query(..., description="NRL season year, e.g. 2026"),
) -> list[TeamOption]:
    if season in _teams_cache:
        return _teams_cache[season]
    try:
        all_matches = _get_repo().list_matches(season)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Match store unavailable: {exc}") from exc
    seen: dict[int, str] = {}
    for m in all_matches:
        seen[m.home.team_id] = m.home.name
        seen[m.away.team_id] = m.away.name
    teams = sorted(
        [TeamOption(id=tid, name=name) for tid, name in seen.items()],
        key=lambda t: t.name,
    )
    _teams_cache[season] = teams
    return teams


@app.post(
    "/notifications/subscribe",
    summary="Register an FCM token for push notifications",
    description=(
        "Persists the caller's FCM device token under `users/{uid}/fcm_tokens/{token}` "
        "in Firestore. Idempotent — re-subscribing with the same token is a no-op. "
        "Requires Firestore (STORAGE_BACKEND=firestore)."
    ),
    status_code=204,
)
def notifications_subscribe(
    request: Request,
    body: NotificationSubscribeIn,
) -> None:
    uid = _require_auth(request)
    backend = os.getenv("STORAGE_BACKEND", "sqlite").lower()
    if backend != "firestore":
        # Local SQLite dev — silently accept so the client doesn't 500.
        return

    try:
        from google.cloud import firestore as _fs  # noqa: PLC0415
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Firestore not available") from exc

    project = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    db = _fs.Client(project=project)
    token_ref = db.collection("users").document(uid).collection("fcm_tokens").document(body.token)
    token_ref.set(
        {
            "token": body.token,
            "platform": body.platform,
            "timezone": body.timezone,
            "created_at": _fs.SERVER_TIMESTAMP,
            "last_seen_at": _fs.SERVER_TIMESTAMP,
        },
        merge=True,
    )


# ---------------------------------------------------------------------------
# Groups + Leaderboard endpoints (#173)
# ---------------------------------------------------------------------------

_GROUP_MAX_SIZE = 100


def _require_firestore(operation: str):
    """Raise 503 if not running against Firestore."""
    if os.getenv("STORAGE_BACKEND", "sqlite").lower() != "firestore":
        raise HTTPException(
            status_code=503,
            detail=f"{operation} requires STORAGE_BACKEND=firestore",
        )


def _get_firestore_client():
    try:
        from google.cloud import firestore  # noqa: PLC0415

        project = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        return firestore.Client(project=project)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Firestore SDK not available") from exc


def _make_invite_code() -> str:
    import random  # noqa: PLC0415
    import string  # noqa: PLC0415

    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _fetch_user_tips(uid: str, season: int) -> dict[int, str]:
    """Return ``{match_id: tip_choice}`` for this user/season from Firestore.

    Tips live at ``users/{uid}/tips/{matchId}`` (written by the SPA via the
    Firebase JS SDK). Returns ``{}`` when there's no real authenticated user,
    when the storage backend isn't Firestore, or on any Firestore error —
    callers should treat that as "no tips" rather than 5xx.
    """
    if uid == "__dev__":
        return {}
    if os.getenv("STORAGE_BACKEND", "sqlite").lower() != "firestore":
        return {}
    try:
        db = _get_firestore_client()
        tips_q = (
            db.collection("users").document(uid).collection("tips").where("season", "==", season)
        )
        out: dict[int, str] = {}
        for snap in tips_q.stream():
            try:
                mid = int(snap.id)
            except (ValueError, TypeError):
                continue
            data = snap.to_dict() or {}
            choice = data.get("tip")
            if choice in ("home", "away"):
                out[mid] = choice
        return out
    except Exception:
        return {}


@app.post(
    "/groups",
    response_model=GroupOut,
    summary="Create a new tipping group",
    status_code=201,
)
def create_group(request: Request, body: GroupIn) -> GroupOut:
    _require_firestore("create_group")
    uid = _require_auth(request)
    db = _get_firestore_client()
    invite_code = _make_invite_code()
    from google.cloud import firestore as _fs  # noqa: PLC0415

    _, ref = db.collection("groups").add(
        {
            "name": body.name,
            "owner_uid": uid,
            "invite_code": invite_code,
            "member_count": 1,
            "created_at": _fs.SERVER_TIMESTAMP,
        }
    )
    gid = ref.id
    # Add owner as first member.
    ref.collection("members").document(uid).set(
        {"joined_at": _fs.SERVER_TIMESTAMP, "display_name_snapshot": ""}
    )
    # Mirror to user's groups sub-collection.
    db.collection("users").document(uid).collection("groups").document(gid).set(
        {"name": body.name, "joined_at": _fs.SERVER_TIMESTAMP}
    )
    return GroupOut(
        gid=gid,
        name=body.name,
        inviteCode=invite_code,
        ownerUid=uid,
        memberCount=1,
    )


@app.post(
    "/groups/{gid}/join",
    response_model=GroupOut,
    summary="Join a group by invite code",
)
def join_group(request: Request, gid: str, body: JoinGroupIn) -> GroupOut:
    _require_firestore("join_group")
    uid = _require_auth(request)
    db = _get_firestore_client()
    from google.cloud import firestore as _fs  # noqa: PLC0415

    group_ref = db.collection("groups").document(gid)
    group_snap = group_ref.get()
    if not group_snap.exists:
        raise HTTPException(status_code=404, detail="Group not found")
    data = group_snap.to_dict()
    if data.get("invite_code") != body.inviteCode:
        raise HTTPException(status_code=403, detail="Invalid invite code")
    if data.get("member_count", 0) >= _GROUP_MAX_SIZE:
        raise HTTPException(status_code=422, detail="Group is full (max 100 members)")
    member_ref = group_ref.collection("members").document(uid)
    if member_ref.get().exists:
        # Already a member — idempotent.
        return GroupOut(
            gid=gid,
            name=data["name"],
            inviteCode=data["invite_code"],
            ownerUid=data["owner_uid"],
            memberCount=data.get("member_count", 1),
        )
    member_ref.set({"joined_at": _fs.SERVER_TIMESTAMP, "display_name_snapshot": ""})
    group_ref.update({"member_count": _fs.Increment(1)})
    db.collection("users").document(uid).collection("groups").document(gid).set(
        {"name": data["name"], "joined_at": _fs.SERVER_TIMESTAMP}
    )
    return GroupOut(
        gid=gid,
        name=data["name"],
        inviteCode=data["invite_code"],
        ownerUid=data["owner_uid"],
        memberCount=data.get("member_count", 1) + 1,
    )


@app.post(
    "/groups/{gid}/leave",
    summary="Leave a group",
    status_code=204,
)
def leave_group(request: Request, gid: str) -> None:
    _require_firestore("leave_group")
    uid = _require_auth(request)
    db = _get_firestore_client()
    from google.cloud import firestore as _fs  # noqa: PLC0415

    group_ref = db.collection("groups").document(gid)
    group_snap = group_ref.get()
    if not group_snap.exists:
        raise HTTPException(status_code=404, detail="Group not found")
    member_ref = group_ref.collection("members").document(uid)
    if member_ref.get().exists:
        member_ref.delete()
        group_ref.update({"member_count": _fs.Increment(-1)})
    db.collection("users").document(uid).collection("groups").document(gid).delete()


@app.get(
    "/leaderboard",
    response_model=LeaderboardOut,
    summary="Tipping leaderboard",
    description=(
        "Returns the top 50 tippers for the season ordered by accuracy (tiebreak: "
        "margin_points ascending, then display name). Reads from pre-aggregated "
        "`users/{uid}/stats` documents written by match_sync when results land. "
        "Optionally filter to a group via `group_id`."
    ),
)
def get_leaderboard(
    request: Request,
    season: int = Query(..., description="NRL season year"),
    group_id: str | None = Query(default=None),
) -> LeaderboardOut:
    _require_firestore("get_leaderboard")
    _require_auth(request)
    db = _get_firestore_client()

    if group_id:
        # Collect UIDs in this group, then look up each member's stats doc by
        # known path. The schema (match_sync._update_user_stats) writes stats
        # to users/{uid}/stats/{season}, so a direct get_all is faster than a
        # collection_group query and doesn't need a composite index.
        member_docs = db.collection("groups").document(group_id).collection("members").stream()
        member_uids = [d.id for d in member_docs]
        if not member_uids:
            return LeaderboardOut(season=season, groupId=group_id, entries=[])
        refs = [
            db.collection("users").document(uid).collection("stats").document(str(season))
            for uid in member_uids
        ]
        stats_docs = [snap for snap in db.get_all(refs) if snap.exists]
    else:
        # Global top-50.
        stats_docs = list(
            db.collection_group("stats")
            .where("season", "==", season)
            .order_by("accuracy", direction="DESCENDING")
            .limit(50)
            .stream()
        )

    entries: list[LeaderboardEntry] = []
    for doc in stats_docs:
        d = doc.to_dict()
        total = d.get("total_tips", 0)
        wins = d.get("wins", 0)
        acc = wins / total if total > 0 else 0.0
        entries.append(
            LeaderboardEntry(
                rank=0,  # assigned after sort
                uid=doc.reference.parent.parent.id,
                displayName=d.get("display_name", "Tipster"),
                wins=wins,
                losses=d.get("losses", 0),
                totalTips=total,
                accuracy=acc,
                marginPoints=d.get("margin_points", 0.0),
                currentStreak=d.get("current_streak", 0),
                longestStreak=d.get("longest_streak", 0),
            )
        )

    entries.sort(key=lambda e: (-e.accuracy, e.marginPoints, e.displayName))
    for i, entry in enumerate(entries[:50], start=1):
        entry = entry.model_copy(update={"rank": i})
        entries[i - 1] = entry

    return LeaderboardOut(season=season, groupId=group_id, entries=entries[:50])


# ---------------------------------------------------------------------------
# Social sharing: OG image card + HTML shell (#175)
# ---------------------------------------------------------------------------

# In-process cache for rendered OG PNG bytes: key=(match_id, season, round_num).
_og_png_cache: dict[tuple[int, int, int], bytes] = {}
_OG_PNG_CACHE_MAX = 50

_SHARE_BASE_URL_ENV = "FANTASY_COACH_PUBLIC_BASE_URL"
_DEFAULT_SHARE_BASE = "https://fantasy.lopezcloud.dev"


def _share_base_url() -> str:
    return os.getenv(_SHARE_BASE_URL_ENV, _DEFAULT_SHARE_BASE)


@app.get("/og/match/{match_id}.png", include_in_schema=False)
def get_og_image(
    match_id: int,
    season: int = Query(..., description="Season year, e.g. 2026"),
    round: int = Query(..., description="Round number, e.g. 7", alias="round"),
) -> Response:
    """Render a 1200x630 OG image card for the match. Public — no auth required."""
    cache_key = (match_id, season, round)
    if cache_key in _og_png_cache:
        return Response(
            content=_og_png_cache[cache_key],
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    preds = _get_store().get(season, round)
    pred = next((p for p in preds if p.matchId == match_id), None)
    if not pred:
        raise HTTPException(status_code=404, detail="Match not found")

    try:
        from fantasy_coach.og_image import render_card  # noqa: PLC0415

        png = render_card(
            home_name=pred.home.name,
            away_name=pred.away.name,
            home_win_prob=pred.homeWinProbability,
            kickoff_iso=pred.kickoff,
            round_label=f"Round {round}, {season}",
        )
    except Exception as exc:
        logger.exception("OG image render failed for match %s", match_id)
        raise HTTPException(status_code=500, detail="Image rendering failed") from exc

    if len(_og_png_cache) >= _OG_PNG_CACHE_MAX:
        _og_png_cache.clear()
    _og_png_cache[cache_key] = png

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@app.get("/share/match/{match_id}", include_in_schema=False)
def get_share_page(
    match_id: int,
    season: int = Query(..., description="Season year, e.g. 2026"),
    round: int = Query(..., description="Round number, e.g. 7", alias="round"),
) -> HTMLResponse:
    """HTML shell with OG meta tags for social-embed preview. Public — no auth required.

    Social crawlers (Twitterbot, Slackbot) can't execute JavaScript, so the SPA
    URL won't have populated meta tags. This endpoint returns a minimal server-side
    rendered page that sets og:title / og:image, then immediately redirects the
    browser to the SPA via <meta http-equiv="refresh">.
    """
    preds = _get_store().get(season, round)
    pred = next((p for p in preds if p.matchId == match_id), None)
    if not pred:
        raise HTTPException(status_code=404, detail="Match not found")

    home_pct = round(pred.homeWinProbability * 100)
    away_pct = 100 - home_pct
    winner = pred.home.name if pred.predictedWinner == "home" else pred.away.name
    winner_pct = home_pct if pred.predictedWinner == "home" else away_pct

    # Absolute URLs — og:image and og:url must be absolute for most crawlers.
    api_base = _share_base_url().rstrip("/")
    # Use the API server for og:image (it renders the PNG); use the SPA for og:url.
    # The API URL is the Cloud Run service; the SPA URL is Firebase Hosting.
    api_url = os.getenv("FANTASY_COACH_API_BASE_URL", api_base)
    og_image = f"{api_url}/og/match/{match_id}.png?season={season}&round={round}"
    spa_url = f"{api_base}/round/{season}/{round}/{match_id}"

    # Escape all user-derived strings for safe embedding in HTML attributes/content.
    e = _html.escape
    title = e(f"{pred.home.name} vs {pred.away.name} — {winner} {winner_pct}% — Fantasy Coach")
    description = e(
        f"{pred.home.name} {home_pct}% vs {pred.away.name} {away_pct}%  ·  Round {round}, {season}"
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <meta name="description" content="{description}">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{title}">
  <meta property="og:description" content="{description}">
  <meta property="og:image" content="{e(og_image)}">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:url" content="{e(spa_url)}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{title}">
  <meta name="twitter:description" content="{description}">
  <meta name="twitter:image" content="{e(og_image)}">
  <link rel="canonical" href="{e(spa_url)}">
  <meta http-equiv="refresh" content="0;url={e(spa_url)}">
</head>
<body>
  <p>Redirecting to <a href="{e(spa_url)}">Fantasy Coach</a>&hellip;</p>
</body>
</html>"""

    return HTMLResponse(page, headers={"Cache-Control": "public, max-age=300"})
