"""Attempt keys and the record lifecycle state machine.

Recoup exists to re-attempt payments that already failed.  That makes ambiguous
failures its *defining hazard*, not an edge case:

    The agent fires a debit.  The network times out.  No response arrives.  Did
    the debit land?  Unknown.  The naive reaction is to retry -- and now a
    customer whose original complaint was "my payment failed" has been charged
    twice.

That is strictly worse than any metric being slightly off, and it is the single
worst outcome this system can produce.  Razorpay's own guidance is explicit:
retries after a timeout must reuse the same idempotency key *and* the same
request body.

The bug this module replaced: the executor used ``receipt=f"recoup_{record_id}"``
-- identical for every retry of a record, so attempts were not distinguishable
at all.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Mapping, Set

# Razorpay's `receipt` field is capped at 40 characters.
PAYPAL_REFERENCE_LIMIT = 40


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------
def attempt_key(run_id: str, record_id: str, attempt_number: int) -> str:
    """The canonical idempotency key for one attempt.

    Three properties, each load-bearing:

    * **Deterministic** -- a crashed run that resumes derives the identical key
      and therefore cannot double-charge.
    * **Unique per attempt** -- attempt 2 is a genuinely different operation and
      must be allowed to proceed.
    * **Scoped per run** -- re-running the batch is a new set of operations,
      not a replay of the old one.
    """
    if attempt_number < 1:
        raise ValueError(f"attempt_number is 1-based, got {attempt_number}")
    return f"recoup:{run_id}:{record_id}:attempt:{attempt_number}"


def staged_key(run_id: str, record_id: str, attempt_number: int, stage: str) -> str:
    """Attempt key extended with a settlement stage.

    One attempt legitimately emits more than one event over time (``pending`` at
    API return, then ``resolved`` at settlement).  An attempt-level unique index
    would drop the resolution -- every pending charge stuck forever, recovered
    dollars silently undercounting.  Stage-in-the-key is a one-field extension of
    the scheme above and needs no new schema.
    """
    return f"{attempt_key(run_id, record_id, attempt_number)}:{stage}"


def receipt_id(run_id: str, record_id: str, attempt_number: int) -> str:
    """Compress the key to fit Razorpay's 40-character ``receipt`` limit while
    preserving the (run, record, attempt) triple."""
    full = attempt_key(run_id, record_id, attempt_number)
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:16]
    receipt = f"rcp_{digest}_a{attempt_number}"
    if len(receipt) > PAYPAL_REFERENCE_LIMIT:  # pragma: no cover - guard
        receipt = receipt[:PAYPAL_REFERENCE_LIMIT]
    return receipt


# --------------------------------------------------------------------------
# Record lifecycle
# --------------------------------------------------------------------------
class RecordStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SCHEDULED = "scheduled"
    RECOVERED = "recovered"
    PARTIALLY_RECOVERED = "partially_recovered"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"


TERMINAL_STATUSES: Set[RecordStatus] = {
    RecordStatus.RECOVERED,
    RecordStatus.ESCALATED,
    RecordStatus.ABANDONED,
}

# `partially_recovered` is deliberately NOT terminal: a partial debit leaves a
# residual balance still at risk, so the agent may keep working the remainder.
_ALLOWED: Mapping[RecordStatus, Set[RecordStatus]] = {
    # A first attempt may simply succeed. The original table forced a detour
    # through IN_PROGRESS, which the state machine correctly refused at runtime
    # on the agent's very first run -- the machine working, the table wrong.
    RecordStatus.PENDING: {
        RecordStatus.IN_PROGRESS,
        RecordStatus.SCHEDULED,
        RecordStatus.RECOVERED,
        RecordStatus.PARTIALLY_RECOVERED,
        RecordStatus.ESCALATED,
        RecordStatus.ABANDONED,
    },
    RecordStatus.IN_PROGRESS: {
        RecordStatus.IN_PROGRESS,
        RecordStatus.SCHEDULED,
        RecordStatus.RECOVERED,
        RecordStatus.PARTIALLY_RECOVERED,
        RecordStatus.ESCALATED,
        RecordStatus.ABANDONED,
    },
    RecordStatus.SCHEDULED: {
        RecordStatus.IN_PROGRESS,
        RecordStatus.SCHEDULED,
        RecordStatus.RECOVERED,
        RecordStatus.PARTIALLY_RECOVERED,
        RecordStatus.ESCALATED,
        RecordStatus.ABANDONED,
    },
    RecordStatus.PARTIALLY_RECOVERED: {
        RecordStatus.IN_PROGRESS,
        RecordStatus.SCHEDULED,
        RecordStatus.RECOVERED,
        RecordStatus.PARTIALLY_RECOVERED,
        RecordStatus.ESCALATED,
        RecordStatus.ABANDONED,
    },
    # Terminal states have no outgoing edges at all.
    RecordStatus.RECOVERED: set(),
    RecordStatus.ESCALATED: set(),
    RecordStatus.ABANDONED: set(),
}


class IllegalTransition(RuntimeError):
    """Raised when a record is asked to leave a terminal state.

    Reopening a stopped record would mean contacting a customer the agent has
    already decided to stop contacting -- precisely what the stopping rules
    exist to prevent.
    """


def is_terminal(status: RecordStatus) -> bool:
    return status in TERMINAL_STATUSES


def can_transition(current: RecordStatus, target: RecordStatus) -> bool:
    return target in _ALLOWED[current]


def assert_transition(current: RecordStatus, target: RecordStatus) -> RecordStatus:
    if not can_transition(current, target):
        raise IllegalTransition(
            f"{current.value} -> {target.value} is not a legal transition"
            + (" (source state is terminal)" if is_terminal(current) else "")
        )
    return target
