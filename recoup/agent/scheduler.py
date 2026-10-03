"""The outer batch controller.

Owns everything cross-record: EV-ordered dispatch, the global intervention-cost
budget, the per-customer contact cap, cooldown / high-liquidity re-enqueue, and
simulated time.  It is a second *caller* of the brain (``control.classify`` /
``control.decide`` reading the same ``Policy``), never a reimplementation.

Propose / dispose: the scheduler uses cooldown + caps to decide *when to offer*
an attempt; the graph's guardrail is the final authority on whether it is
*allowed*.  Budget and contacts are decremented from the graph's actual output,
so a guardrail block spends nothing.

Store is truth: each pass re-derives the eligible set, so the scheduler is
effectively stateless between ticks and a run is crash-resumable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from config.taxonomy import Intervention, classify_reason, spec
from recoup.agent.control import DEFAULT_POLICY, Policy, decide
from recoup.agent.graph import graph
from recoup.agent.outcomes import (
    NormalisedEvent,
    OutcomeIngest,
    OutcomeResolution,
)
from recoup.agent.state import TerminalStatus
from recoup.money import format_inr


# --------------------------------------------------------------------------
# Config + summary
# --------------------------------------------------------------------------
@dataclass
class SchedulerConfig:
    run_id: str = "run_batch"
    start: datetime = field(default_factory=lambda: datetime(2026, 1, 16, 9, 0, 0))
    batch_budget_cents: int = 10_000_00   # intervention-cost ceiling (Rs 10k)
    max_passes: int = 40
    tick_hours: float = 24.0
    policy: Policy = field(default_factory=lambda: DEFAULT_POLICY)


@dataclass
class RunSummary:
    run_id: str
    n_records: int
    total_at_risk_cents: int
    recovered_cents: int
    recovered_count: int
    escalated_count: int
    abandoned_count: int
    in_progress_count: int
    attempts: int
    contacts_made: int
    budget_spent_cents: int
    passes: int

    @property
    def recovery_rate(self) -> float:
        return (
            self.recovered_cents / self.total_at_risk_cents
            if self.total_at_risk_cents
            else 0.0
        )

    def render(self) -> str:
        return (
            f"[{self.run_id}] {self.recovered_count} recovered "
            f"({format_inr(self.recovered_cents)}) | "
            f"{self.escalated_count} escalated | {self.abandoned_count} abandoned | "
            f"{self.in_progress_count} awaiting | {self.attempts} attempts | "
            f"{self.recovery_rate:.1%} of {format_inr(self.total_at_risk_cents)}"
        )


# --------------------------------------------------------------------------
# Per-record scheduling state (the scheduler's own bookkeeping)
# --------------------------------------------------------------------------
@dataclass
class _RecordState:
    record: Dict[str, Any]
    outstanding_cents: int
    attempts: int = 0
    status: str = "open"                 # open | recovered | escalated | abandoned | awaiting
    last_attempt_iso: Optional[str] = None
    ready_at: Optional[datetime] = None  # cooldown / salary-window gate
    recovered_cents: int = 0


class BatchScheduler:
    def __init__(
        self,
        records: List[Dict[str, Any]],
        config: SchedulerConfig = SchedulerConfig(),
        *,
        on_park=None,                    # callback(record_dict, intervention, attempt, now, outstanding)
    ) -> None:
        self.config = config
        self.policy = config.policy
        self.now = config.start
        self.states: Dict[str, _RecordState] = {
            r["record_id"]: _RecordState(
                record=dict(r), outstanding_cents=int(r["amount_cents"])
            )
            for r in records
        }
        self.contacts_by_customer: Dict[str, List[datetime]] = {}
        self.budget_spent = 0
        self.attempts = 0
        self.contacts_made = 0
        self.passes = 0
        self._on_park = on_park
        self._audit_rows: List[Dict[str, Any]] = []
        # The async front door closes records when their outcome resolves.
        self.ingest = OutcomeIngest(self._apply_resolution)

    # ------------------------------------------------------------------
    # Public: async outcome ingestion (webhook / simulator both call this)
    # ------------------------------------------------------------------
    def ingest_outcome(self, event: NormalisedEvent) -> Optional[OutcomeResolution]:
        return self.ingest.ingest_outcome(event)

    def _apply_resolution(self, resolution: OutcomeResolution) -> None:
        state = self.states.get(resolution.record_id)
        if state is None or state.status in ("recovered", "escalated", "abandoned"):
            return
        if resolution.terminal_status == TerminalStatus.RECOVERED.value:
            amount = min(resolution.amount_recovered_cents, state.outstanding_cents)
            state.recovered_cents += amount
            state.outstanding_cents -= amount
            state.status = "recovered" if state.outstanding_cents <= 0 else "open"
        elif resolution.terminal_status == TerminalStatus.ESCALATED.value:
            state.status = "escalated"

    # ------------------------------------------------------------------
    # The main loop
    # ------------------------------------------------------------------
    def run(self) -> RunSummary:
        for _ in range(self.config.max_passes):
            if self.budget_spent >= self.config.batch_budget_cents:
                break
            eligible = self._eligible_now()
            if not eligible:
                if not self._advance_time():
                    break
                continue
            self.passes += 1
            # EV-ordered dispatch: highest net-EV records first.
            eligible.sort(key=self._net_ev_key, reverse=True)
            progressed = False
            for state in eligible:
                if self.budget_spent >= self.config.batch_budget_cents:
                    break
                if self._dispatch(state):
                    progressed = True
            if not progressed and not self._advance_time():
                break
        return self._summarise()

    # ------------------------------------------------------------------
    def _eligible_now(self) -> List[_RecordState]:
        out = []
        for state in self.states.values():
            if state.status != "open":
                continue
            if state.ready_at is not None and state.ready_at > self.now:
                continue
            out.append(state)
        return out

    def _net_ev_key(self, state: _RecordState) -> int:
        prior = self._prior_contacts(state.record["customer_id"])
        rec = dict(state.record)
        rec["amount_cents"] = state.outstanding_cents
        decision = decide(rec, prior_contacts=prior, policy=self.policy)
        return decision.net_ev_cents

    def _dispatch(self, state: _RecordState) -> bool:
        record = state.record
        customer_id = record["customer_id"]
        prior_contacts = self._prior_contacts(customer_id)
        attempt = state.attempts + 1

        # Build the invoke record with the control fields the nodes read.
        rec = dict(record)
        rec["amount_cents"] = int(record["amount_cents"])  # keep original for taxonomy
        rec["_outstanding_cents"] = state.outstanding_cents
        rec["_prior_contacts"] = prior_contacts
        rec["_hours_since_last"] = (
            None
            if state.last_attempt_iso is None
            else max(
                0.0,
                (self.now - datetime.fromisoformat(state.last_attempt_iso)).total_seconds()
                / 3600.0,
            )
        )

        # Propose gate: if the record's contact-cap budget is exhausted AND the
        # only sane action is a contact, we still let the graph decide and log a
        # clean escalation. So we simply invoke.
        result = graph.invoke(
            {
                "run_id": self.config.run_id,
                "record": rec,
                "attempt": attempt,
                "now_iso": self.now.isoformat(),
            }
        )
        self.attempts += 1
        state.attempts = attempt
        state.last_attempt_iso = self.now.isoformat()

        decision = result.get("decision", {})
        execution = result.get("execution", {})
        terminal = result.get("terminal_status")
        chosen = decision.get("chosen")
        self._audit_rows.append(result.get("audit_row", {}))

        # Dispose: spend budget/contacts from the ACTUAL action taken.
        action_intervention = _safe_intervention(chosen)
        if action_intervention is not None and terminal not in ("escalated", "abandoned"):
            action = spec(action_intervention)
            if action.contacts_customer:
                self.contacts_by_customer.setdefault(customer_id, []).append(self.now)
                self.contacts_made += 1
                self.budget_spent += _contact_cost(prior_contacts, self.policy)

        # Apply the attempt's terminal effect to scheduler state.
        return self._apply_terminal(state, terminal, execution, action_intervention)

    def _apply_terminal(self, state, terminal, execution, intervention) -> bool:
        if terminal == TerminalStatus.RECOVERED.value:
            recovered = min(int(execution.get("amount_recovered_cents", 0)),
                            state.outstanding_cents)
            state.recovered_cents += recovered
            state.outstanding_cents -= recovered
            state.status = "recovered" if state.outstanding_cents <= 0 else "open"
            return True
        if terminal == TerminalStatus.ESCALATED.value:
            state.status = "escalated"
            return True
        if terminal == TerminalStatus.ABANDONED.value:
            state.status = "abandoned"
            return True
        if terminal == TerminalStatus.IN_PROGRESS.value:
            if execution.get("settles_async"):
                # A contact was fired; park it for async resolution and stop
                # re-sending every tick (fire-once contacts).
                state.status = "awaiting"
                if self._on_park is not None and intervention is not None:
                    self._on_park(
                        state.record, intervention, state.attempts, self.now,
                        state.outstanding_cents,
                    )
                return True
            # A failed silent retry: gate the next attempt behind cooldown or
            # the next high-liquidity day, then leave it open.
            partial = int(execution.get("amount_recovered_cents", 0))
            if partial > 0:
                state.recovered_cents += partial
                state.outstanding_cents -= partial
            self._set_ready_at(state, intervention)
            # Attempt cap reached? then this record can never progress silently.
            klass = classify_reason(state.record.get("error_reason", ""))
            cap = self.policy.max_attempts.get(klass, 0)
            if state.attempts >= cap:
                state.status = "escalated"
            return True
        return False

    def _set_ready_at(self, state: _RecordState, intervention) -> None:
        from recoup.clock import SimulatedClock

        cooldown = timedelta(hours=self.policy.min_cooldown_hours)
        base = self.now + cooldown
        if intervention is Intervention.RETRY_SALARY_WINDOW:
            clock = SimulatedClock(self.now)
            salary_day = int(state.record.get("customer_salary_day", 1))
            base = clock.next_high_liquidity_day(
                salary_day, self.policy.min_cooldown_hours
            )
        state.ready_at = base

    def _advance_time(self) -> bool:
        """Jump to the next moment something can happen: the earliest ready_at,
        else a fixed tick.  Returns False when nothing is left to do."""
        open_states = [s for s in self.states.values() if s.status == "open"]
        awaiting = [s for s in self.states.values() if s.status == "awaiting"]
        if not open_states and not awaiting:
            return False
        future_ready = [
            s.ready_at for s in open_states if s.ready_at and s.ready_at > self.now
        ]
        if future_ready:
            self.now = min(future_ready)
            return True
        # Nothing gated in the future and nothing ready now -> only awaiting
        # records remain, which the async layer resolves; stop advancing.
        if not any(s.ready_at is None for s in open_states):
            return False
        self.now = self.now + timedelta(hours=self.config.tick_hours)
        return True

    # ------------------------------------------------------------------
    def _prior_contacts(self, customer_id: str) -> int:
        window = timedelta(hours=self.policy.contact_window_hours)
        cutoff = self.now - window
        times = self.contacts_by_customer.get(customer_id, [])
        return sum(1 for t in times if t >= cutoff)

    def _summarise(self) -> RunSummary:
        recovered = [s for s in self.states.values() if s.status == "recovered"]
        escalated = [s for s in self.states.values() if s.status == "escalated"]
        abandoned = [s for s in self.states.values() if s.status == "abandoned"]
        awaiting = [s for s in self.states.values() if s.status in ("awaiting", "open")]
        return RunSummary(
            run_id=self.config.run_id,
            n_records=len(self.states),
            total_at_risk_cents=sum(
                int(s.record["amount_cents"]) for s in self.states.values()
            ),
            recovered_cents=sum(s.recovered_cents for s in self.states.values()),
            recovered_count=len(recovered),
            escalated_count=len(escalated),
            abandoned_count=len(abandoned),
            in_progress_count=len(awaiting),
            attempts=self.attempts,
            contacts_made=self.contacts_made,
            budget_spent_cents=self.budget_spent,
            passes=self.passes,
        )

    @property
    def audit_rows(self) -> List[Dict[str, Any]]:
        return self._audit_rows


def _safe_intervention(value) -> Optional[Intervention]:
    try:
        return Intervention(value)
    except (ValueError, TypeError):
        return None


def _contact_cost(prior_contacts: int, policy: Policy) -> int:
    from recoup.economics import contact_cost_cents

    return contact_cost_cents(prior_contacts, True, policy.economic)
