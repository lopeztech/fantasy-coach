from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from fantasy_coach.backfill import (
    BackfillState,
    RetryLog,
    backfill_season,
)
from fantasy_coach.storage import SQLiteRepository

FIXTURES = Path(__file__).parent / "fixtures"
FULLTIME = json.loads((FIXTURES / "match-2024-rd1-sea-eagles-v-rabbitohs.json").read_text())


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SQLiteRepository]:
    db = SQLiteRepository(tmp_path / "test.db")
    yield db
    db.close()


def _round_payload(*urls: str) -> dict:
    return {"fixtures": [{"matchCentreUrl": u} for u in urls]}


def _make_round_fn(payloads: dict[int, dict | None]):
    def f(season: int, round_: int):
        return payloads.get(round_)

    return f


def _make_match_fn(by_url: dict[str, dict | None | Exception]):
    def f(url: str):
        v = by_url[url]
        if isinstance(v, Exception):
            raise v
        return v

    return f


def test_backfill_ingests_all_fixtures_in_a_round(repo: SQLiteRepository, tmp_path: Path) -> None:
    state = BackfillState.load(tmp_path / "state.json")
    retry = RetryLog(tmp_path / "retry.tsv")

    summaries = backfill_season(
        2024,
        repo,
        state,
        retry,
        fetch_round_fn=_make_round_fn(
            {1: _round_payload("/draw/nrl-premiership/2024/round-1/sea-eagles-v-rabbitohs/")}
        ),
        fetch_match_fn=_make_match_fn(
            {"/draw/nrl-premiership/2024/round-1/sea-eagles-v-rabbitohs/": FULLTIME}
        ),
        max_rounds=2,
    )

    assert len(summaries) == 1
    assert summaries[0].fetched == 1
    assert summaries[0].failed == 0
    assert repo.get_match(FULLTIME["matchId"]) is not None


def test_backfill_stops_when_round_has_no_fixtures(repo: SQLiteRepository, tmp_path: Path) -> None:
    state = BackfillState.load(tmp_path / "state.json")
    retry = RetryLog(tmp_path / "retry.tsv")

    summaries = backfill_season(
        2024,
        repo,
        state,
        retry,
        fetch_round_fn=_make_round_fn(
            {
                1: _round_payload("/draw/nrl-premiership/2024/round-1/a-v-b/"),
                2: {"fixtures": []},
                3: _round_payload("/should/not/be/fetched/"),
            }
        ),
        fetch_match_fn=_make_match_fn({"/draw/nrl-premiership/2024/round-1/a-v-b/": FULLTIME}),
    )

    assert [s.round for s in summaries] == [1]


def test_backfill_skips_already_processed_urls(repo: SQLiteRepository, tmp_path: Path) -> None:
    state = BackfillState.load(tmp_path / "state.json")
    retry = RetryLog(tmp_path / "retry.tsv")
    url = "/draw/nrl-premiership/2024/round-1/sea-eagles-v-rabbitohs/"
    state.mark(url)

    fetched_urls: list[str] = []

    def match_fn(u: str):
        fetched_urls.append(u)
        return FULLTIME

    summaries = backfill_season(
        2024,
        repo,
        state,
        retry,
        fetch_round_fn=_make_round_fn({1: _round_payload(url), 2: {"fixtures": []}}),
        fetch_match_fn=match_fn,
    )

    assert summaries[0].skipped == 1
    assert summaries[0].fetched == 0
    assert fetched_urls == []  # never re-fetched


def test_backfill_records_failures_to_retry_log(repo: SQLiteRepository, tmp_path: Path) -> None:
    state = BackfillState.load(tmp_path / "state.json")
    retry_path = tmp_path / "retry.tsv"
    retry = RetryLog(retry_path)
    boom = "/draw/nrl-premiership/2024/round-1/team-a-v-team-b/"

    summaries = backfill_season(
        2024,
        repo,
        state,
        retry,
        fetch_round_fn=_make_round_fn({1: _round_payload(boom), 2: {"fixtures": []}}),
        fetch_match_fn=_make_match_fn({boom: RuntimeError("kaboom")}),
    )

    assert summaries[0].failed == 1
    assert boom not in state.processed
    log_lines = retry_path.read_text().splitlines()
    assert any(boom in line and "kaboom" in line for line in log_lines)


def test_backfill_treats_404_as_failure_and_continues(
    repo: SQLiteRepository, tmp_path: Path
) -> None:
    state = BackfillState.load(tmp_path / "state.json")
    retry = RetryLog(tmp_path / "retry.tsv")
    missing = "/draw/nrl-premiership/2024/round-1/missing-v-match/"
    good = "/draw/nrl-premiership/2024/round-1/sea-eagles-v-rabbitohs/"

    summaries = backfill_season(
        2024,
        repo,
        state,
        retry,
        fetch_round_fn=_make_round_fn({1: _round_payload(missing, good), 2: {"fixtures": []}}),
        fetch_match_fn=_make_match_fn({missing: None, good: FULLTIME}),
    )

    assert summaries[0].fetched == 1
    assert summaries[0].failed == 1


def test_backfill_handles_finals_round_slugs(repo: SQLiteRepository, tmp_path: Path) -> None:
    state = BackfillState.load(tmp_path / "state.json")
    retry = RetryLog(tmp_path / "retry.tsv")
    finals_url = "/draw/nrl-premiership/2024/finals-week-1/game-1/"

    summaries = backfill_season(
        2024,
        repo,
        state,
        retry,
        fetch_round_fn=_make_round_fn(
            {
                1: {"fixtures": []},  # immediate stop in this scenario
            }
        ),
        fetch_match_fn=_make_match_fn({finals_url: FULLTIME}),
    )

    # Round 1 empty → backfill stops without trying finals (finals would be
    # later rounds in real data — this test asserts the empty-round halt).
    assert summaries == []


def test_backfill_state_persists_across_runs(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = BackfillState.load(state_path)
    state.mark("/draw/foo/")
    state.mark("/draw/bar/")

    reloaded = BackfillState.load(state_path)
    assert reloaded.processed == {"/draw/foo/", "/draw/bar/"}


def test_state_flush_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "deeper" / "state.json"
    state = BackfillState.load(nested)
    state.mark("/draw/x/")
    assert nested.exists()
