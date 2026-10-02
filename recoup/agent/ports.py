"""Protocol seams: the graph depends on behaviour, not implementations.

Each port is a ``Protocol`` so a demo adapter (offline, deterministic) and a
real adapter (Razorpay, SQLite, Claude) are interchangeable without the nodes
knowing which they hold.  This is what lets the whole agent run offline in CI
and against the real test-mode account with the same code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Protocol

from config.taxonomy import Intervention


class Clock(Protocol):
    def now_iso(self) -> str: ...
    def hours_between(self, earlier_iso: str, later_iso: str) -> float: ...
    def next_high_liquidity_iso(
        self, from_iso: str, salary_day: int, min_delay_hours: float
    ) -> str: ...


class Executor(Protocol):
    def execute(
        self,
        record: Dict[str, Any],
        intervention: Intervention,
        *,
        run_id: str,
        attempt_number: int,
        attempt_at_iso: str,
        outstanding_paise: int,
    ) -> Dict[str, Any]:
        """Fire the action.  Returns a dict with at least ``outcome``,
        ``amount_recovered_paise``, ``api_called``, ``was_mocked``,
        ``razorpay_entity_id``, ``idempotency_key``, and ``settles_async``."""
        ...


class AuditLog(Protocol):
    def write(self, row: Dict[str, Any]) -> None: ...


class Narrator(Protocol):
    def narrate(self, state: Dict[str, Any]) -> str:
        """Write-only prose for the audit row.  Must swallow its own errors and
        never influence control flow."""
        ...
