import csv
import os
import pathlib
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fantasy_coach import __version__
from fantasy_coach.auth import FirebaseAuthMiddleware
from fantasy_coach.config import get_repository
from fantasy_coach.models.elo_mov import EloMOV
from fantasy_coach.predictions import (
    FirestorePredictionStore,
    PredictionOut,
    PredictionStore,
    get_prediction_store,
)
from fantasy_coach.storage.repository import Repository

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


ALLOWED_ORIGINS_ENV = "FANTASY_COACH_ALLOWED_ORIGINS"
DEFAULT_ALLOWED_ORIGINS = (
    "https://fantasy.lopezcloud.dev,http://localhost:5173,http://localhost:4173"
)


def _allowed_origins() -> list[str]:
    raw = os.getenv(ALLOWED_ORIGINS_ENV, DEFAULT_ALLOWED_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Fantasy Coach",
    version=__version__,
    description="NRL match prediction API. All non-health endpoints require a Firebase ID token.",
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
    app.add_middleware(FirebaseAuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

# Module-level singletons — created lazily on first use.
_store: PredictionStore | FirestorePredictionStore | None = None
_repo: Repository | None = None

# In-process cache for team profile responses: key=(team_id, season), value=(timestamp, data)
_PROFILE_CACHE_TTL = 60  # seconds
_profile_cache: dict[tuple[int, int], tuple[float, TeamProfile]] = {}

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

    # Untipped = current-round matches that have no stored prediction.
    untipped_match_ids: list[int] = []
    if current_round is not None:
        current_round_matches = [m for m in matches if m.round == current_round]
        try:
            preds = _get_store().get(season, current_round)
            tipped_ids = {p.matchId for p in preds}
        except Exception:
            tipped_ids = set()
        untipped_match_ids = [
            m.match_id for m in current_round_matches if m.match_id not in tipped_ids
        ]

    # Season-to-date tipping accuracy: compare stored predictions to FullTime results.
    total_tips = 0
    correct_tips = 0
    completed = [
        m
        for m in matches
        if m.match_state == "FullTime" and m.home.score is not None and m.away.score is not None
    ]
    completed_rounds = {m.round for m in completed}
    for rnd in completed_rounds:
        try:
            preds = _get_store().get(season, rnd)
        except Exception:
            continue
        result_map = {
            m.match_id: ("home" if m.home.score > m.away.score else "away")  # type: ignore[operator]
            for m in completed
            if m.round == rnd
        }
        for p in preds:
            if p.matchId in result_map:
                total_tips += 1
                if p.predictedWinner == result_map[p.matchId]:
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
