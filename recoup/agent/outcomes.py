"""The async resolution seam: ``ingest_outcome`` and a pure ``resolve_outcome``.

Human outcomes (a customer updates a card, re-auths a mandate) take real-world
time and arrive later.  Both a real PayPal webhook route and the test-mode
``OutcomeSimulator`` call ``ingest_outcome`` with a normalised event.  It is
idempotent by ``event_id`` -- webhooks arrive at-least-once and out of order --
and the classification of an event into a resolution is the pure function
``resolve_outcome``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

from recoup.agent.state import TerminalStatus


class ResolvedOutcome(str, Enum):
    RECOVERED = "recovered"
    LAPSED = "lapsed"          # the customer never acted -> escalate/abandon
    IGNORED = "ignored"        # duplicate or irrelevant


@dataclass(frozen=True)
class NormalisedEvent:
    """What both the simulator and a real webhook produce."""

    event_id: str
    record_id: str
    customer_id: str
    kind: str                  # "customer_action" | "timeout" | ...
    acted: bool
    amount_recovered_cents: int
    occurred_at_iso: str


def resolve_outcome(event: NormalisedEvent) -> ResolvedOutcome:
    """Pure map from an event to a resolution.  No I/O, easy to test."""
    if event.kind == "customer_action" and event.acted and event.amount_recovered_cents > 0:
        return ResolvedOutcome.RECOVERED
    if event.kind in ("timeout", "customer_action") and not event.acted:
        return ResolvedOutcome.LAPSED
    return ResolvedOutcome.IGNORED


@dataclass
class OutcomeResolution:
    record_id: str
    resolution: ResolvedOutcome
    terminal_status: str
    amount_recovered_cents: int


class OutcomeIngest:
    """Idempotent front door for async outcomes.

    Dedups by ``event_id`` (webhooks fire twice), applies ``resolve_outcome``,
    and hands the resolution to a caller-supplied ``apply`` callback (which the
    scheduler uses to close the record and record recovered money).
    """

    def __init__(self, apply: Callable[[OutcomeResolution], None]) -> None:
        self._seen: Set[str] = set()
        self._apply = apply

    def ingest_outcome(self, event: NormalisedEvent) -> Optional[OutcomeResolution]:
        if event.event_id in self._seen:
            return None  # duplicate delivery -> no double count
        self._seen.add(event.event_id)

        resolution = resolve_outcome(event)
        if resolution is ResolvedOutcome.RECOVERED:
            terminal = TerminalStatus.RECOVERED.value
            amount = event.amount_recovered_cents
        elif resolution is ResolvedOutcome.LAPSED:
            terminal = TerminalStatus.ESCALATED.value
            amount = 0
        else:
            return None

        out = OutcomeResolution(event.record_id, resolution, terminal, amount)
        self._apply(out)
        return out
