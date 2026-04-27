"""Monte Carlo season-ladder + finals simulator (#217).

Runs N simulations of the NRL season from the current state:
1. Completed matches use actual results.
2. Upcoming fixtures use the precomputed ``homeWinProbability`` where available;
   other future rounds fall back to ``DEFAULT_HOME_WIN_PROB``.
3. Each simulation resolves the full regular season to a final ladder, then
   runs the NRL McIntyre Final 8 bracket to determine a champion.

Vectorised numpy implementation — 10k sims × ~100 remaining matches × 17 teams
runs in < 2 seconds on Cloud Run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fantasy_coach.features import MatchRow

logger = logging.getLogger(__name__)

# Neutral home win advantage used when we have no prediction for a fixture.
DEFAULT_HOME_WIN_PROB = 0.55

# Approximate NRL average points scored per team per match — used to simulate
# a plausible percentage (points-for / points-against) for tiebreaking.
_AVG_WINNER_SCORE = 24.0
_AVG_LOSER_SCORE = 18.0
_SCORE_NOISE_STD = 6.0

# Teams that finish in the top 8 after the regular season make the finals.
FINALS_SPOTS = 8


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class SeasonOutcome:
    """Per-team simulation aggregate probabilities."""

    team_id: int
    team_name: str
    playoff_prob: float  # top 8
    top_4_prob: float
    top_2_prob: float
    minor_premiership_prob: float  # finish 1st on the ladder
    grand_final_prob: float
    premiership_prob: float


@dataclass
class SimulationResult:
    """Full result returned by simulate_season."""

    season: int
    n_simulations: int
    teams: list[SeasonOutcome] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "season": self.season,
            "nSimulations": self.n_simulations,
            "teams": [
                {
                    "teamId": t.team_id,
                    "teamName": t.team_name,
                    "playoffProb": round(t.playoff_prob, 4),
                    "top4Prob": round(t.top_4_prob, 4),
                    "top2Prob": round(t.top_2_prob, 4),
                    "minorPremiershipProb": round(t.minor_premiership_prob, 4),
                    "grandFinalProb": round(t.grand_final_prob, 4),
                    "premiershipProb": round(t.premiership_prob, 4),
                }
                for t in sorted(self.teams, key=lambda t: -t.playoff_prob)
            ],
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def simulate_season(
    season: int,
    all_matches: list[MatchRow],
    predictions: dict[int, float],
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Run Monte Carlo season simulation.

    Args:
        season: NRL season year.
        all_matches: All matches in the season (completed + upcoming).
        predictions: ``{match_id: homeWinProbability}`` for upcoming fixtures
            where we have a precomputed prediction.
        n_simulations: Number of Monte Carlo draws.
        rng: Optional RNG for deterministic testing.

    Returns:
        ``SimulationResult`` with per-team probabilities.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Index all teams and assign integer slots.
    team_ids: list[int] = []
    team_names: dict[int, str] = {}
    for m in all_matches:
        for tid, name in ((m.home.team_id, m.home.name), (m.away.team_id, m.away.name)):
            if tid not in team_names:
                team_ids.append(tid)
                team_names[tid] = name
    team_ids.sort()
    team_idx: dict[int, int] = {tid: i for i, tid in enumerate(team_ids)}
    n_teams = len(team_ids)

    # Separate completed from upcoming matches.
    completed = [m for m in all_matches if m.home.score is not None and m.away.score is not None]
    upcoming = [m for m in all_matches if m.home.score is None or m.away.score is None]

    # --- Compute current standings from completed matches ---
    base_comp_pts = np.zeros(n_teams, dtype=np.float64)
    base_pts_for = np.zeros(n_teams, dtype=np.float64)
    base_pts_against = np.zeros(n_teams, dtype=np.float64)

    for m in completed:
        hi = team_idx[m.home.team_id]
        ai = team_idx[m.away.team_id]
        hs, as_ = int(m.home.score), int(m.away.score)
        base_pts_for[hi] += hs
        base_pts_against[hi] += as_
        base_pts_for[ai] += as_
        base_pts_against[ai] += hs
        if hs > as_:
            base_comp_pts[hi] += 2
        elif hs < as_:
            base_comp_pts[ai] += 2
        else:
            base_comp_pts[hi] += 1
            base_comp_pts[ai] += 1

    # --- Build upcoming match arrays ---
    n_upcoming = len(upcoming)
    if n_upcoming == 0:
        # Season complete — just run finals on actual standings.
        result = SimulationResult(season=season, n_simulations=1)
        _add_playoff_outcomes(
            result,
            team_ids,
            team_names,
            team_idx,
            base_comp_pts.reshape(1, -1),
            base_pts_for.reshape(1, -1),
            base_pts_against.reshape(1, -1),
            rng,
            n_simulations=1,
        )
        return result

    home_idxs = np.array([team_idx[m.home.team_id] for m in upcoming], dtype=np.int32)
    away_idxs = np.array([team_idx[m.away.team_id] for m in upcoming], dtype=np.int32)
    home_probs = np.array(
        [predictions.get(m.match_id, DEFAULT_HOME_WIN_PROB) for m in upcoming],
        dtype=np.float64,
    )

    # --- Monte Carlo simulation ---
    # shape: (n_simulations, n_upcoming)
    random_draws = rng.random((n_simulations, n_upcoming))
    home_wins = random_draws < home_probs[np.newaxis, :]  # (n_sims, n_upcoming)

    # Simulate scores for percentage computation.
    winner_noise = rng.normal(0, _SCORE_NOISE_STD, (n_simulations, n_upcoming))
    loser_noise = rng.normal(0, _SCORE_NOISE_STD, (n_simulations, n_upcoming))
    winner_scores = np.maximum(1, _AVG_WINNER_SCORE + winner_noise)
    loser_scores = np.maximum(0, np.minimum(winner_scores - 1, _AVG_LOSER_SCORE + loser_noise))

    # Competition points per simulation: (n_sims, n_teams)
    comp_pts = np.tile(base_comp_pts, (n_simulations, 1))
    pts_for = np.tile(base_pts_for, (n_simulations, 1))
    pts_against = np.tile(base_pts_against, (n_simulations, 1))

    for j in range(n_upcoming):
        hi = home_idxs[j]
        ai = away_idxs[j]
        hw = home_wins[:, j]
        ws = winner_scores[:, j]
        ls = loser_scores[:, j]

        # Home wins
        comp_pts[:, hi] += hw.astype(np.float64) * 2
        pts_for[:, hi] += np.where(hw, ws, ls)
        pts_against[:, hi] += np.where(hw, ls, ws)

        # Away wins
        aw = ~hw
        comp_pts[:, ai] += aw.astype(np.float64) * 2
        pts_for[:, ai] += np.where(aw, ws, ls)
        pts_against[:, ai] += np.where(aw, ls, ws)

    result = SimulationResult(season=season, n_simulations=n_simulations)
    _add_playoff_outcomes(
        result,
        team_ids,
        team_names,
        team_idx,
        comp_pts,
        pts_for,
        pts_against,
        rng,
        n_simulations=n_simulations,
    )
    return result


# ---------------------------------------------------------------------------
# Private: finals simulation
# ---------------------------------------------------------------------------


def _add_playoff_outcomes(
    result: SimulationResult,
    team_ids: list[int],
    team_names: dict[int, str],
    team_idx: dict[int, int],
    comp_pts: np.ndarray,  # (n_sims, n_teams)
    pts_for: np.ndarray,
    pts_against: np.ndarray,
    rng: np.random.Generator,
    n_simulations: int,
) -> None:
    n_teams = len(team_ids)

    # Counters for outcomes.
    playoff_count = np.zeros(n_teams, dtype=np.int64)
    top4_count = np.zeros(n_teams, dtype=np.int64)
    top2_count = np.zeros(n_teams, dtype=np.int64)
    minor_count = np.zeros(n_teams, dtype=np.int64)
    gf_count = np.zeros(n_teams, dtype=np.int64)
    premier_count = np.zeros(n_teams, dtype=np.int64)

    # Vectorise the ranking: sort by (-comp_pts, -percentage, noise).
    pct = np.where(pts_against > 0, pts_for / pts_against * 100.0, 0.0)
    noise = rng.random(comp_pts.shape) * 1e-6  # break remaining ties randomly

    # Sort key: higher comp_pts, higher pct, small noise.
    sort_key = comp_pts * 10_000 + pct + noise
    # ranks[s, t] = position of team t in simulation s (0 = top = 1st)
    ranks = (n_teams - 1) - np.argsort(np.argsort(sort_key, axis=1), axis=1)

    playoff_mask = ranks < FINALS_SPOTS  # top 8
    top4_mask = ranks < 4
    top2_mask = ranks < 2
    minor_mask = ranks == 0

    playoff_count += playoff_mask.sum(axis=0)
    top4_count += top4_mask.sum(axis=0)
    top2_count += top2_mask.sum(axis=0)
    minor_count += minor_mask.sum(axis=0)

    # Get seedings: shape (n_sims, 8) — index of team in each seed position (0-indexed)
    # Sort teams by rank within each sim, take top 8.
    ordered = np.argsort(-sort_key, axis=1)  # (n_sims, n_teams) descending
    seeds = ordered[:, :FINALS_SPOTS]  # (n_sims, 8), each col = team idx in that seed

    # Simple finals simulation: higher seed hosts (0.55 home advantage).
    home_adv = 0.55
    finals_randoms = rng.random((n_simulations, 9))  # 9 finals matches max

    # Track GF participants and premiers.
    for s in range(n_simulations):
        bracket = list(seeds[s])
        r = finals_randoms[s]
        gf_teams, premier = _run_finals_tracked(bracket, r, home_adv)
        for t in gf_teams:
            gf_count[t] += 1
        premier_count[premier] += 1

    n = n_simulations or 1
    for i, tid in enumerate(team_ids):
        result.teams.append(
            SeasonOutcome(
                team_id=tid,
                team_name=team_names[tid],
                playoff_prob=playoff_count[i] / n,
                top_4_prob=top4_count[i] / n,
                top_2_prob=top2_count[i] / n,
                minor_premiership_prob=minor_count[i] / n,
                grand_final_prob=gf_count[i] / n,
                premiership_prob=premier_count[i] / n,
            )
        )


def _match_winner(
    team_a: int, team_b: int, a_is_higher_seed: bool, rand: float, home_adv: float
) -> int:
    """Return winning team index. Higher seed hosts (has home_adv)."""
    prob_a_wins = home_adv if a_is_higher_seed else (1 - home_adv)
    return team_a if rand < prob_a_wins else team_b


def _run_finals(bracket: list[int], randoms: np.ndarray, home_adv: float) -> list[int]:
    """Return [GF_winner] — not tracked, just for counting."""
    _, premier = _run_finals_tracked(bracket, randoms, home_adv)
    return [premier]


def _run_finals_tracked(
    bracket: list[int], randoms: np.ndarray, home_adv: float
) -> tuple[list[int], int]:
    """Simulate NRL McIntyre Final 8; return (gf_participants, champion).

    bracket[i] = team_idx of seed i+1 (0-indexed, 0=1st seed).
    randoms: at least 9 values used as match random numbers.
    """
    # Week 1 — Qualifying finals (top 4) + Elimination finals (bottom 4)
    # QF1: seed1 (0) vs seed4 (3) — seed1 hosts
    qf1_winner = _match_winner(bracket[0], bracket[3], True, randoms[0], home_adv)
    qf1_loser = bracket[3] if qf1_winner == bracket[0] else bracket[0]

    # QF2: seed2 (1) vs seed3 (2) — seed2 hosts
    qf2_winner = _match_winner(bracket[1], bracket[2], True, randoms[1], home_adv)
    qf2_loser = bracket[2] if qf2_winner == bracket[1] else bracket[1]

    # EF1: seed5 (4) vs seed8 (7) — seed5 hosts
    ef1_winner = _match_winner(bracket[4], bracket[7], True, randoms[2], home_adv)

    # EF2: seed6 (5) vs seed7 (6) — seed6 hosts
    ef2_winner = _match_winner(bracket[5], bracket[6], True, randoms[3], home_adv)

    # Week 2 — Semi finals
    # SF1: QF1 loser vs EF2 winner — QF loser considered higher seed (had a bye of sorts)
    sf1_winner = _match_winner(qf1_loser, ef2_winner, True, randoms[4], home_adv)

    # SF2: QF2 loser vs EF1 winner
    sf2_winner = _match_winner(qf2_loser, ef1_winner, True, randoms[5], home_adv)

    # Week 3 — Preliminary finals
    # PF1: QF1 winner vs SF1 winner — QF1 winner hosts
    pf1_winner = _match_winner(qf1_winner, sf1_winner, True, randoms[6], home_adv)

    # PF2: QF2 winner vs SF2 winner — QF2 winner hosts
    pf2_winner = _match_winner(qf2_winner, sf2_winner, True, randoms[7], home_adv)

    # Week 4 — Grand Final (neutral venue)
    gf_winner = pf1_winner if randoms[8] < 0.5 else pf2_winner

    return [pf1_winner, pf2_winner], gf_winner
