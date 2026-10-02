"""The runtime container and its demo adapters.

The nodes read their dependencies from a single module-level ``RUNTIME`` object,
configured once at startup via :func:`configure`.  The defaults are demo
adapters -- fully offline, deterministic -- so ``import graph`` works with no
keys, no network, and no database, which is what makes the smoke drivers and CI
possible.

Wiring real infrastructure is a one-call swap:

    configure(executor=RealRazorpayExecutor(), clock=SimClock(),
              audit=SqliteAudit(), narrator=ClaudeNarrator(), policy=Policy(...))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config.taxonomy import Intervention, spec
from recoup.agent.control import DEFAULT_POLICY, Policy
from recoup.agent.ports import AuditLog, Clock, Executor, Narrator
from recoup.clock import SimulatedClock
from recoup.environment import RecoveryEnvironment
from recoup.executor import Executor as PaymentsExecutor
from recoup.models import ExecutionMode, Outcome
from recoup.money import format_inr, split_paise


# --------------------------------------------------------------------------
# Demo clock
# --------------------------------------------------------------------------
class DemoClock:
    """A clock over a mutable "current" ISO timestamp, defaulting to now.

    The scheduler owns the true simulated time and passes ``now_iso`` into each
    invoke; this adapter is defaults-and-arithmetic only.
    """

    def __init__(self, start: Optional[datetime] = None) -> None:
        self._now = start or datetime(2026, 1, 16, 9, 0, 0)

    def now_iso(self) -> str:
        return self._now.isoformat()

    def hours_between(self, earlier_iso: str, later_iso: str) -> float:
        earlier = datetime.fromisoformat(earlier_iso)
        later = datetime.fromisoformat(later_iso)
        return max(0.0, (later - earlier).total_seconds() / 3600.0)

    def next_high_liquidity_iso(
        self, from_iso: str, salary_day: int, min_delay_hours: float
    ) -> str:
        clock = SimulatedClock(datetime.fromisoformat(from_iso))
        return clock.next_high_liquidity_day(salary_day, min_delay_hours).isoformat()


# --------------------------------------------------------------------------
# Demo executor: wraps the Phase-1 simulated executor, needs latent truth
# --------------------------------------------------------------------------
class DemoExecutor:
    """Runs interventions through the Phase-1 SIMULATED executor.

    It needs the hidden environment + latent truth to decide outcomes, which the
    demo supplies from the frozen batch.  Machine outcomes (retries) resolve
    in-band; human outcomes (links) return ``settles_async=True`` and no money
    yet -- the async seam resolves them later.
    """

    def __init__(
        self,
        environment: RecoveryEnvironment,
        customers: Dict[str, Any],
        latent: Dict[str, Any],
    ) -> None:
        self._payments = PaymentsExecutor(environment, mode=ExecutionMode.SIMULATED)
        self._customers = customers
        self._latent = latent

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
        from recoup.models import FailedRecord

        action = spec(intervention)
        contacts = action.contacts_customer

        # A customer-facing contact is fired and then parked: the outcome is the
        # customer's future action, resolved out-of-band. We do not sample it
        # synchronously -- and so it needs neither the hidden environment nor
        # latent truth, which is why those lookups come after this branch.
        if contacts:
            key = f"recoup:{run_id}:{record['record_id']}:attempt:{attempt_number}"
            return {
                "outcome": "in_progress",
                "amount_recovered_paise": 0,
                "amount_attempted_paise": outstanding_paise,
                "api_called": False,
                "was_mocked": not spec(intervention).contacts_customer,
                "mock_reason": "",
                "razorpay_entity_id": None,
                "idempotency_key": key,
                "settles_async": True,
            }

        # Synchronous (silent-retry) path: now we need the typed record, the
        # customer, and the latent truth to sample the outcome.
        clean = {k: v for k, v in record.items() if not k.startswith("_")}
        record_obj = FailedRecord.from_json(clean) if "failed_at" in clean else None
        customer = self._customers[record["customer_id"]]
        latent = self._latent[record["record_id"]]
        attempt_at = datetime.fromisoformat(attempt_at_iso)

        result = self._payments.execute(
            record_obj,
            customer,
            latent,
            intervention,
            run_id=run_id,
            attempt_number=attempt_number,
            attempt_at=attempt_at,
            outstanding_paise=outstanding_paise,
        )
        return {
            "outcome": result.outcome.value,
            "amount_recovered_paise": result.amount_recovered_paise,
            "amount_attempted_paise": result.amount_attempted_paise,
            "api_called": result.api_called,
            "was_mocked": result.was_mocked,
            "mock_reason": result.mock_reason,
            "razorpay_entity_id": result.razorpay_entity_id,
            "idempotency_key": result.idempotency_key,
            "settles_async": False,
        }


# --------------------------------------------------------------------------
# Demo audit log + narrator
# --------------------------------------------------------------------------
class DemoAuditLog:
    """Collects audit rows in memory.  A real adapter writes to the Phase-1
    append-only SQLite store instead."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def write(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)


class TemplateNarrator:
    """Deterministic, offline narration.  Write-only: it reads state and returns
    prose, swallowing any error, and has no edge back into routing."""

    def narrate(self, state: Dict[str, Any]) -> str:
        try:
            record = state.get("record", {})
            decision = state.get("decision", {})
            execution = state.get("execution", {})
            guardrail = state.get("guardrail", {})
            status = state.get("terminal_status", "?")
            amount = format_inr(int(record.get("amount_paise", 0)))
            who = record.get("customer_id", "?")

            if status == "recovered":
                got = format_inr(int(execution.get("amount_recovered_paise", 0)))
                return (
                    f"Recovered {got} of {amount} from {who} via "
                    f"{decision.get('chosen', '?')}."
                )
            if status == "in_progress":
                if execution.get("settles_async"):
                    return (
                        f"Sent {decision.get('chosen', '?')} to {who} for "
                        f"{amount}; awaiting customer action."
                    )
                got = int(execution.get("amount_recovered_paise", 0))
                if got > 0:
                    return (
                        f"Partially recovered {format_inr(got)} of {amount} from "
                        f"{who}; residual remains at risk."
                    )
                return (
                    f"{decision.get('chosen', '?')} on {who}'s {amount} did not "
                    f"recover this attempt; record stays open."
                )
            if status == "escalated":
                return (
                    f"Escalated {who}'s {amount}: {guardrail.get('reason') or decision.get('rationale', '')}"
                )
            if status == "abandoned":
                return f"Gave up on {who}'s {amount}: {decision.get('rationale', '')}"
            if status == "scheduled":
                return f"Scheduled a retry for {who}'s {amount}."
            return f"{status}: {who} {amount}"
        except Exception:  # noqa: BLE001 - narration must never break the graph
            return ""


# --------------------------------------------------------------------------
# The container
# --------------------------------------------------------------------------
@dataclass
class Runtime:
    clock: Clock = field(default_factory=DemoClock)
    executor: Optional[Executor] = None       # set by configure(); demo has none by default
    audit: AuditLog = field(default_factory=DemoAuditLog)
    narrator: Narrator = field(default_factory=TemplateNarrator)
    policy: Policy = field(default_factory=lambda: DEFAULT_POLICY)


RUNTIME = Runtime()


def configure(
    *,
    clock: Optional[Clock] = None,
    executor: Optional[Executor] = None,
    audit: Optional[AuditLog] = None,
    narrator: Optional[Narrator] = None,
    policy: Optional[Policy] = None,
) -> Runtime:
    """Swap any subset of the runtime's adapters.  Called once at startup."""
    if clock is not None:
        RUNTIME.clock = clock
    if executor is not None:
        RUNTIME.executor = executor
    if audit is not None:
        RUNTIME.audit = audit
    if narrator is not None:
        RUNTIME.narrator = narrator
    if policy is not None:
        RUNTIME.policy = policy
    return RUNTIME
