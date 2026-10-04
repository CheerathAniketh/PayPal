"""The async execution seam: fire, suspend, resolve later.

The central honesty point of Phase 2: ``execute_and_suspend`` fires an action,
records the *intent* (a FIRED event), and returns *before the money result is
known*.  Charges settle later; mandate debits settle T+1.  A synchronous
``execute`` that returned an outcome would model a fiction a payments engineer
spots immediately.

Settlement then arrives through the SAME ``ingest_outcome`` a real webhook would
call -- so the simulator and production share one code path, and ``source`` is
provenance only, never branched on.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from config.taxonomy import Intervention, spec
from recoup.environment import RecoveryEnvironment
from recoup.event_store import EventStore
from recoup.events import (
    ApiResult,
    FinancialResult,
    OutcomeEvent,
    Stage,
)
from recoup.executor import TEST_MODE_SUPPORT
from recoup.models import Customer, FailedRecord, LatentTruth
from recoup.money import split_cents


@dataclass
class SuspendedAttempt:
    """A fired attempt awaiting settlement."""

    record_id: str
    attempt_n: int
    fired_at: str


def execute_and_suspend(
    store: EventStore,
    record: FailedRecord,
    intervention: Intervention,
    *,
    run_id: str,
    attempt_n: int,
    now: datetime,
    outstanding_cents: Optional[int] = None,
) -> SuspendedAttempt:
    """Fire the action, write a FIRED event, and return.  No outcome yet.

    The FIRED event is what the contact cap counts (customer-facing, at
    ``occurred_at = now``), and what a later resolution folds against.
    """
    action = spec(intervention)
    outstanding = record.amount_cents if outstanding_cents is None else outstanding_cents
    amount = (
        split_cents(outstanding, action.recovery_fraction)
        if action.recovery_fraction < 1.0
        else outstanding
    )
    support = TEST_MODE_SUPPORT[intervention]
    api = ApiResult.ACCEPTED if support.supported else ApiResult.NONE

    event = OutcomeEvent(
        run_id=run_id,
        record_id=record.record_id,
        customer_id=record.customer_id,
        attempt_n=attempt_n,
        stage=Stage.FIRED,
        intervention=intervention.value,
        is_contact=action.contacts_customer,
        amount_cents=amount,
        occurred_at=now.isoformat(),
        ingested_at=now.isoformat(),
        api_result=api,
        financial_result=FinancialResult.PENDING,
        source="simulator",
        raw={"phase": "fired"},
    )
    store.ingest_outcome(event)
    return SuspendedAttempt(record.record_id, attempt_n, now.isoformat())


@dataclass(order=True)
class _Scheduled:
    settle_at: datetime
    seq: int
    attempt: "SuspendedAttempt" = field(compare=False)
    record: FailedRecord = field(compare=False)
    customer: Customer = field(compare=False)
    latent: LatentTruth = field(compare=False)
    intervention: Intervention = field(compare=False)
    run_id: str = field(compare=False)


class SimulatedGateway:
    """Queues settlement and resolves it later through ``ingest_outcome``.

    Stands in for the real webhook infrastructure behind the same seam.  A real
    PayPal webhook route is a new *caller* of ``ingest_outcome``, not a
    rewrite of this.
    """

    def __init__(
        self,
        store: EventStore,
        environment: RecoveryEnvironment,
        settle_delay: timedelta = timedelta(hours=24),
    ) -> None:
        self.store = store
        self.environment = environment
        self.settle_delay = settle_delay
        self._queue: List[_Scheduled] = []
        self._seq = 0

    def fire(
        self,
        record: FailedRecord,
        customer: Customer,
        latent: LatentTruth,
        intervention: Intervention,
        *,
        run_id: str,
        attempt_n: int,
        now: datetime,
        outstanding_cents: Optional[int] = None,
    ) -> SuspendedAttempt:
        """Fire and queue the settlement for ``now + settle_delay``."""
        attempt = execute_and_suspend(
            self.store, record, intervention,
            run_id=run_id, attempt_n=attempt_n, now=now,
            outstanding_cents=outstanding_cents,
        )
        self._seq += 1
        heapq.heappush(
            self._queue,
            _Scheduled(
                settle_at=now + self.settle_delay,
                seq=self._seq,
                attempt=attempt,
                record=record,
                customer=customer,
                latent=latent,
                intervention=intervention,
                run_id=run_id,
            ),
        )
        return attempt

    def next_settlement_time(self) -> Optional[datetime]:
        return self._queue[0].settle_at if self._queue else None

    def settle_due(self, now: datetime) -> int:
        """Resolve every attempt whose settlement time has arrived.

        Returns how many were resolved.  Each resolution is a RESOLVED event
        pushed through ``ingest_outcome`` -- the identical entry point a webhook
        uses.
        """
        resolved = 0
        while self._queue and self._queue[0].settle_at <= now:
            item = heapq.heappop(self._queue)
            self._resolve(item, now)
            resolved += 1
        return resolved

    def settle_all(self, now: datetime) -> int:
        """Resolve everything still queued, regardless of time.  For batch end."""
        resolved = 0
        while self._queue:
            item = heapq.heappop(self._queue)
            self._resolve(item, max(now, item.settle_at))
            resolved += 1
        return resolved

    def _resolve(self, item: _Scheduled, now: datetime) -> None:
        action = spec(item.intervention)
        recovered = self.environment.sample_outcome(
            item.record, item.customer, item.latent, item.intervention,
            item.settle_at, item.attempt.attempt_n,
        )
        # Re-derive the fired amount so the resolution matches the intent.
        outstanding_ok = self.store.events_for_record(item.record.record_id)
        fired_amount = next(
            (e.amount_cents for e in outstanding_ok
             if e.attempt_n == item.attempt.attempt_n and e.stage is Stage.FIRED),
            item.record.amount_cents,
        )
        if recovered:
            fin = (
                FinancialResult.PARTIAL
                if action.recovery_fraction < 1.0
                else FinancialResult.RECOVERED
            )
            amount_recovered = fired_amount
        else:
            fin = FinancialResult.NOT_RECOVERED
            amount_recovered = 0

        event = OutcomeEvent(
            run_id=item.run_id,
            record_id=item.record.record_id,
            customer_id=item.record.customer_id,
            attempt_n=item.attempt.attempt_n,
            stage=Stage.RESOLVED,
            intervention=item.intervention.value,
            is_contact=action.contacts_customer,
            amount_cents=fired_amount,
            occurred_at=item.settle_at.isoformat(),
            ingested_at=now.isoformat(),
            api_result=ApiResult.ACCEPTED,
            financial_result=fin,
            amount_recovered_cents=amount_recovered,
            source="simulator",
            raw={"phase": "resolved"},
        )
        self.store.ingest_outcome(event)
