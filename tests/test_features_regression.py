"""Regression test against snapshot extracted MatchRows.

Pairs each `tests/fixtures/match-*.json` (raw `/data` payload) with a
snapshot in `tests/fixtures/extracted/extracted-*.json`. If the
extractor's output shape changes — added/removed fields, type drift,
re-ordering of stable lists — these tests fail until the snapshot is
intentionally regenerated.

To regenerate after a deliberate change:
  uv run python -c "
import json
from pathlib import Path
from fantasy_coach.features import extract_match_features
src = Path('tests/fixtures')
for raw in src.glob('match-*.json'):
    row = extract_match_features(json.loads(raw.read_text()))
    out = src / 'extracted' / raw.name.replace('match-', 'extracted-')
    out.write_text(row.model_dump_json(indent=2))
"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fantasy_coach.features import MatchRow, extract_match_features

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOTS = FIXTURES / "extracted"

PAIRS = sorted(
    (raw, SNAPSHOTS / raw.name.replace("match-", "extracted-"))
    for raw in FIXTURES.glob("match-*.json")
)


@pytest.mark.parametrize(
    ("raw_path", "snapshot_path"),
    PAIRS,
    ids=[raw.stem for raw, _ in PAIRS],
)
def test_extracted_matchrow_matches_snapshot(raw_path: Path, snapshot_path: Path) -> None:
    assert snapshot_path.exists(), (
        f"Missing snapshot for {raw_path.name}; regenerate per the module docstring."
    )
    raw = json.loads(raw_path.read_text())
    expected = MatchRow.model_validate_json(snapshot_path.read_text())
    actual = extract_match_features(raw)
    assert actual == expected
