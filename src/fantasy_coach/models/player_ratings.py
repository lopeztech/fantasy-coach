"""Per-player Elo-style ratings — the availability-aware strength primitive.

Mirrors ``models.elo.Elo`` but keyed on ``player_id``. Each player's rating
is updated after every match they appear in, with the player treated as
"playing against" the opponent team's composite rating (average of that
team's starting XIII ratings). Starters get a full K-update; bench players
get a fractional update — a proxy for minutes played since the NRL data
feed doesn't carry on/off timings.

Why this lives inside ``FeatureBuilder`` state (not Firestore, as #109's
issue body proposes):

- ``EloMOV`` already walks all historical matches on builder construction.
- Player ratings can walk the same loop at O(XIII × matches) — cheap.
- Keeping state transient matches the no-leakage contract: the walk-forward
  harness refits per round and each refit sees ratings built only from
  strictly-earlier matches.
- No new Firestore collection to maintain, version, or backfill.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K_STARTER = 20.0
DEFAULT_K_BENCH = 10.0  # bench plays ~20-30 minutes on average — half-weight update
DEFAULT_SEASON_REGRESSION = 0.25

# Position-group membership for the matchup-differential features (#210).
# Groups are disjoint subsets of the NRL starting XIII.
# Bench / Interchange players are not included — we only want the starters
# contesting each position battle.
POSITION_GROUPS: dict[str, frozenset[str]] = {
    "halves": frozenset({"Halfback", "Five-Eighth"}),
    "forwards": frozenset({"Prop", "Lock", "2nd Row"}),
    "hooker": frozenset({"Hooker"}),
    "outside_backs": frozenset({"Fullback", "Winger", "Centre"}),
}


@dataclass
class PlayerRatings:
    """In-memory player-rating book.

    Players are inserted lazily at first appearance. A secondary book
    tracks each player's most-common position so the composite feature
    can look up the right ``POSITION_WEIGHTS`` entry without consulting
    the match row a second time.
    """

    k_starter: float = DEFAULT_K_STARTER
    k_bench: float = DEFAULT_K_BENCH
    initial_rating: float = DEFAULT_INITIAL_RATING
    season_regression: float = DEFAULT_SEASON_REGRESSION

    # Populated by ``update`` and read by ``rating`` / ``position``.
    _ratings: dict[int, float] = field(default_factory=dict)
    _position_counts: dict[int, Counter[str]] = field(default_factory=dict)

    # ----- read-only -----

    def rating(self, player_id: int) -> float:
        return self._ratings.get(player_id, self.initial_rating)

    def position(self, player_id: int) -> str | None:
        """Return the player's most-common observed starting position.

        Used by the composite feature to apply position weights even when
        the current match row happens to field the player in a different
        slot — their "regular" position is the rating-level signal.
        """
        counts = self._position_counts.get(player_id)
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def composite(
        self,
        starters: Iterable[tuple[int, str | None]],
        bench: Iterable[tuple[int, str | None]] = (),
        *,
        position_weights: dict[str, float] | None = None,
        bench_weight: float = 0.3,
    ) -> float:
        """Sum of ``rating × position_weight`` across a named XIII + bench.

        ``position_weights`` keys are NRL position strings (``"Halfback"``,
        ``"Hooker"``, …). When absent, every position weight defaults to 1.0.
        Bench players contribute ``bench_weight × rating × position_weight``
        — rough proxy for their shorter minutes — and are skipped entirely
        when ``bench_weight == 0``.
        """
        weights = position_weights or {}
        total = 0.0
        for player_id, position in starters:
            w = weights.get(position or "", 1.0)
            total += w * self.rating(player_id)
        if bench_weight > 0:
            for player_id, position in bench:
                w = weights.get(position or "", 1.0)
                total += bench_weight * w * self.rating(player_id)
        return total

    # ----- mutations -----

    def update(
        self,
        home_players: Iterable[tuple[int, str | None, bool]],
        away_players: Iterable[tuple[int, str | None, bool]],
        home_score: int,
        away_score: int,
    ) -> None:
        """Fold one match into the rating book.

        Each ``*_players`` iterable yields ``(player_id, position, is_on_field)``.
        Players with unknown ``is_on_field`` (i.e. ``None`` in the raw data)
        are skipped — no signal is better than wrong signal. The team
        composite used as the "opponent rating" is built from the starting
        XIII only (bench composites would be noisier and dilute the update).
        """
        home_list = [p for p in home_players if p[2] is not None]
        away_list = [p for p in away_players if p[2] is not None]
        if not home_list or not away_list:
            return

        home_starters = [(pid, pos) for pid, pos, on in home_list if on]
        away_starters = [(pid, pos) for pid, pos, on in away_list if on]
        if not home_starters or not away_starters:
            return

        home_rating = _avg_rating(self, home_starters)
        away_rating = _avg_rating(self, away_starters)

        actual_home = _result(home_score, away_score)
        expected_home = _expected_score(home_rating, away_rating)
        actual_away = 1.0 - actual_home
        expected_away = 1.0 - expected_home

        # Each player in the home XIII/bench "played against" the away
        # composite rating; away side mirrors. Position counts update
        # every appearance, not only starters, so the rater learns a
        # player's usual slot even if they got a bench game here.
        for pid, pos, on in home_list:
            k = self.k_starter if on else self.k_bench
            self._ratings[pid] = self.rating(pid) + k * (actual_home - expected_home)
            if pos:
                self._position_counts.setdefault(pid, Counter())[pos] += 1
        for pid, pos, on in away_list:
            k = self.k_starter if on else self.k_bench
            self._ratings[pid] = self.rating(pid) + k * (actual_away - expected_away)
            if pos:
                self._position_counts.setdefault(pid, Counter())[pos] += 1

    def regress_to_mean(self, weight: float | None = None) -> None:
        """Pull every rating fractionally toward ``initial_rating``.

        Call between seasons, same contract as ``Elo.regress_to_mean``.
        """
        w = self.season_regression if weight is None else weight
        if w <= 0:
            return
        if w > 1:
            raise ValueError(f"regression weight must be in [0, 1], got {w}")
        for pid, r in self._ratings.items():
            self._ratings[pid] = r + w * (self.initial_rating - r)


def _avg_rating(book: PlayerRatings, players: list[tuple[int, str | None]]) -> float:
    if not players:
        return book.initial_rating
    return sum(book.rating(pid) for pid, _ in players) / len(players)


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _result(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5
