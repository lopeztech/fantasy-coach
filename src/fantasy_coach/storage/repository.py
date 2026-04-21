"""Storage abstraction for match rows.

Local dev uses `SQLiteRepository`; production will add a Firestore impl.
Kept as a `Protocol` so the swap is mechanical and callers depend on the
interface rather than the concrete class.
"""

from __future__ import annotations

from typing import Protocol

from fantasy_coach.features import MatchRow


class Repository(Protocol):
    def upsert_match(self, row: MatchRow) -> None: ...

    def get_match(self, match_id: int) -> MatchRow | None: ...

    def list_matches(self, season: int, round: int | None = None) -> list[MatchRow]: ...
