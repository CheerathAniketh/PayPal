"""The outcome event, and a monotonic order-independent fold.

Why two axes and not one ``outcome`` field
-------------------------------------------
A single ``outcome`` word cannot express "the API accepted the call but the
money did not settle" -- which is a normal event, not a contradiction.  This
mirrors the real-vs-simulated split already in the executor: the gateway answers
*did the API call succeed*, the world answers *did we recover the money*.  So an
event carries ``api_result`` and ``financial_result`` separately, plus an
amount.

Why a staged key
----------------
An attempt legitimately emits more than one event over time: ``pending`` at API
return, then ``resolved`` at settlement.  An attempt-level unique index would
drop the resolution -- every pending charge stuck forever, recovered dollars
silently undercounting.  Stage-in-the-key fixes it with one extra field.

The monotonic fold invariant
-----------------------------
``financial_result`` may move ``pending -> recovered / not_recovered / partial``
but never backward and never between two resolved states.  A late or duplicate
event that contradicts a resolved attempt is a no-op.  This is the
out-of-order-webhook defence, and it is what makes the fold order-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from recoup.idempotency import staged_key


class Stage(str, Enum):
    FIRED = "fired"        # intent recorded, before the API returns
    PENDING = "pending"    # API accepted, settlement not yet known
    RESOLVED = "resolved"  # the world has decided


class ApiResult(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"
    NONE = "none"          # no API involved (mocked / internal)


class FinancialResult(str, Enum):
    PENDING = "pending"
    RECOVERED = "recovered"
    NOT_RECOVERED = "not_recovered"
    PARTIAL = "partial"


_RESOLVED_FINANCIAL = {
    FinancialResult.RECOVERED,
    FinancialResult.NOT_RECOVERED,
    FinancialResult.PARTIAL,
}


class IllegalEvent(ValueError):
    """An event that cannot logically exist.  Raised at construction, so an
    illegal event is unconstructable rather than merely discouraged."""


@dataclass(frozen=True)
class OutcomeEvent:
    run_id: str
    record_id: str
    customer_id: str
    attempt_n: int
    stage: Stage
    intervention: str
    is_contact: bool
    amount_cents: int
    occurred_at: str            # when it happened -> drives ALL window/ordering
    ingested_at: str            # when we heard about it -> provenance only
    api_result: ApiResult
    financial_result: FinancialResult
    amount_recovered_cents: int = 0
    failure_reason: str = ""
    source: str = "simulator"   # "simulator" | "webhook" -- NEVER branched on
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attempt_n < 1:
            raise IllegalEvent(f"attempt_n is 1-based, got {self.attempt_n}")
        if self.amount_recovered_cents < 0:
            raise IllegalEvent("amount_recovered_cents cannot be negative")
        if self.amount_recovered_cents > self.amount_cents:
            raise IllegalEvent(
                "recovered cannot exceed the amount attempted "
                f"({self.amount_recovered_cents} > {self.amount_cents})"
            )
        if (
            self.financial_result is FinancialResult.RECOVERED
            and self.amount_recovered_cents <= 0
        ):
            raise IllegalEvent("a RECOVERED event must recover a positive amount")
        if (
            self.financial_result is FinancialResult.NOT_RECOVERED
            and self.amount_recovered_cents != 0
        ):
            raise IllegalEvent("a NOT_RECOVERED event cannot recover money")

    @property
    def key(self) -> str:
        """The staged idempotency key -- unique per (run, record, attempt,
        stage).  Exact duplicates collide on this; distinct-but-stale events are
        handled by the fold, not the key."""
        return staged_key(self.run_id, self.record_id, self.attempt_n, self.stage.value)

    def to_row(self) -> Dict[str, Any]:
        import json

        return {
            "idempotency_key": self.key,
            "run_id": self.run_id,
            "record_id": self.record_id,
            "customer_id": self.customer_id,
            "attempt_n": self.attempt_n,
            "stage": self.stage.value,
            "intervention": self.intervention,
            "is_contact": int(self.is_contact),
            "amount_cents": self.amount_cents,
            "occurred_at": self.occurred_at,
            "ingested_at": self.ingested_at,
            "api_result": self.api_result.value,
            "financial_result": self.financial_result.value,
            "amount_recovered_cents": self.amount_recovered_cents,
            "failure_reason": self.failure_reason,
            "source": self.source,
            "raw": json.dumps(self.raw, default=str),
        }


@dataclass
class RecordOutcome:
    """The read model derived from folding an attempt's events.

    This does NOT replace the authoritative RecordStatus machine; it feeds it.
    ``PARTIAL`` means the same thing on both sides.
    """

    record_id: str
    attempt_n: int
    financial_result: FinancialResult = FinancialResult.PENDING
    amount_recovered_cents: int = 0
    resolved: bool = False

    @property
    def is_resolved(self) -> bool:
        return self.financial_result in _RESOLVED_FINANCIAL


def fold(events: Iterable[OutcomeEvent]) -> Dict[int, RecordOutcome]:
    """Fold a stream of events into per-attempt outcomes.

    Order-independent and monotonic: shuffling the events, or delivering a stale
    one late, yields the identical result.  Once an attempt is resolved, only a
    *different resolved state for the same attempt* would be a contradiction --
    and contradictions are ignored (first resolution wins), so a duplicate
    resolution cannot double-count recovered money.
    """
    outcomes: Dict[int, RecordOutcome] = {}
    for event in events:
        current = outcomes.get(event.attempt_n)
        if current is None:
            current = RecordOutcome(event.record_id, event.attempt_n)
            outcomes[event.attempt_n] = current

        if current.is_resolved:
            # Terminal: a late/duplicate event cannot regress or overwrite it.
            continue

        if event.financial_result in _RESOLVED_FINANCIAL:
            current.financial_result = event.financial_result
            current.amount_recovered_cents = event.amount_recovered_cents
            current.resolved = True
        # A PENDING event on a still-pending attempt leaves it pending.
    return outcomes


def total_recovered(outcomes: Dict[int, RecordOutcome]) -> int:
    return sum(o.amount_recovered_cents for o in outcomes.values())
