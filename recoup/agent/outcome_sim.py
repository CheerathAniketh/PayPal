"""Test-mode outcome simulator: deterministic, reproducible customer actions.

Stands in for real webhook traffic behind the ``ingest_outcome`` seam.  When a
customer-facing contact is parked (a payment link, a re-auth), the simulator
decides -- deterministically, from the hidden environment -- whether that
customer acts before a timeout, and emits the corresponding normalised event.

It never branches on ``source``; the events it produces are indistinguishable
from real ones to everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config.taxonomy import Intervention
from recoup.agent.outcomes import NormalisedEvent
from recoup.environment import RecoveryEnvironment
from recoup.models import Customer, FailedRecord, LatentTruth


@dataclass
class _Parked:
    record: FailedRecord
    customer: Customer
    latent: LatentTruth
    intervention: Intervention
    attempt_n: int
    parked_at: datetime
    deadline: datetime
    outstanding_cents: int


class OutcomeSimulator:
    """Parks customer-facing contacts and resolves them at their deadline."""

    def __init__(
        self,
        environment: RecoveryEnvironment,
        customers: Dict[str, Customer],
        latent: Dict[str, LatentTruth],
        response_window: timedelta = timedelta(days=3),
    ) -> None:
        self.environment = environment
        self.customers = customers
        self.latent = latent
        self.response_window = response_window
        self._parked: List[_Parked] = []

    def park(
        self,
        record: FailedRecord,
        intervention: Intervention,
        attempt_n: int,
        now: datetime,
        outstanding_cents: int,
    ) -> None:
        self._parked.append(
            _Parked(
                record=record,
                customer=self.customers[record.customer_id],
                latent=self.latent[record.record_id],
                intervention=intervention,
                attempt_n=attempt_n,
                parked_at=now,
                deadline=now + self.response_window,
                outstanding_cents=outstanding_cents,
            )
        )

    def next_deadline(self) -> Optional[datetime]:
        return min((p.deadline for p in self._parked), default=None)

    def resolve_due(self, now: datetime) -> List[NormalisedEvent]:
        """Emit events for every parked contact whose deadline has passed."""
        events: List[NormalisedEvent] = []
        still: List[_Parked] = []
        for parked in self._parked:
            if parked.deadline <= now:
                events.append(self._event_for(parked))
            else:
                still.append(parked)
        self._parked = still
        return events

    def resolve_all(self, now: datetime) -> List[NormalisedEvent]:
        events = [self._event_for(p) for p in self._parked]
        self._parked = []
        return events

    def _event_for(self, parked: _Parked) -> NormalisedEvent:
        # The customer "acts" iff the hidden world says the recovery happens.
        acted = self.environment.sample_outcome(
            parked.record,
            parked.customer,
            parked.latent,
            parked.intervention,
            parked.deadline,
            parked.attempt_n,
        )
        event_id = (
            f"evt:{parked.record.record_id}:attempt:{parked.attempt_n}:"
            f"{parked.intervention.value}"
        )
        return NormalisedEvent(
            event_id=event_id,
            record_id=parked.record.record_id,
            customer_id=parked.record.customer_id,
            kind="customer_action",
            acted=acted,
            amount_recovered_cents=parked.outstanding_cents if acted else 0,
            occurred_at_iso=parked.deadline.isoformat(),
        )
