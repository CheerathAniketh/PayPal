"""The per-customer contact cap: a rolling window, keyed on the person.

The blocker this closes
-----------------------
Phase 1's batch controller was a sorted loop with a per-record limit.  The
moment two records shared a ``customer_id``, per-record limits let the customer
be contacted many times over -- a live counterexample to "zero compliance
violations".  So the cap is **global per customer**: all customer-facing
recovery contacts across all of that customer's records count toward one shared
budget.

Three decisions, each deliberate
--------------------------------
* **Flat cap for everyone.**  Customer value affects priority and expected value,
  never how many times you may contact them.  Value-scaled contact limits are a
  customer-treatment liability.
* **Always a rolling window** (option C).  The cap protects a person, not a
  product, so it does not branch on subscription status.  The billing cycle
  still governs retry *timing* via the clock; it just no longer governs the
  contact counter's reset.
* **Rolling means a query, not a counter.**  "How many contact events in the
  last 30 days?" is evaluated fresh each time.  A counter that resets at a
  boundary silently reintroduces the reset-burst problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol


class ContactCapDecision(str, Enum):
    ALLOWED = "allowed"
    CAP_REACHED = "cap_reached"


@dataclass(frozen=True)
class ContactCapPolicy:
    max_contacts: int = 3           # flat, for everyone
    window_days: int = 30           # one rolling window


@dataclass(frozen=True)
class ContactCapResult:
    decision: ContactCapDecision
    used: int
    limit: int
    window_start_iso: str

    @property
    def allowed(self) -> bool:
        return self.decision is ContactCapDecision.ALLOWED

    @property
    def reason(self) -> str:
        if self.allowed:
            return (
                f"{self.used}/{self.limit} customer contacts used in the "
                f"rolling window"
            )
        return (
            f"contact cap reached: {self.used}/{self.limit} in the rolling "
            f"window since {self.window_start_iso}"
        )


class ContactCounter(Protocol):
    """Anything that can count a customer's contacts since a timestamp -- the
    EventStore satisfies this, so the cap has no direct DB dependency."""

    def contact_count(self, customer_id: str, since_iso: str) -> int: ...


def contact_cap_result(
    counter: ContactCounter,
    customer_id: str,
    now: datetime,
    policy: ContactCapPolicy = ContactCapPolicy(),
) -> ContactCapResult:
    """Would one more customer-facing contact be permitted right now?"""
    window_start = now - timedelta(days=policy.window_days)
    window_start_iso = window_start.isoformat()
    used = counter.contact_count(customer_id, window_start_iso)
    decision = (
        ContactCapDecision.ALLOWED
        if used < policy.max_contacts
        else ContactCapDecision.CAP_REACHED
    )
    return ContactCapResult(
        decision=decision,
        used=used,
        limit=policy.max_contacts,
        window_start_iso=window_start_iso,
    )
