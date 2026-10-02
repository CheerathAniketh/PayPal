"""Thin LangGraph nodes and the routers between them.

Each node is deliberately thin: it calls the pure control functions or a port,
writes its slice of state, and returns.  No node decides policy inline -- that
all lives in ``control.py`` -- so a node is easy to read and the logic is easy
to test without a graph.

Every path sets ``terminal_status`` BEFORE ``narrate``, so ``log`` always writes
one complete audit row with the narration attached.
"""

from __future__ import annotations

from typing import Any, Dict

from config.taxonomy import Intervention, spec
from recoup.agent.control import check_guardrails, classify, decide
from recoup.agent.runtime import RUNTIME
from recoup.agent.state import AgentState, TerminalStatus
from recoup.money import split_paise


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------
def ingest(state: AgentState) -> Dict[str, Any]:
    """Normalise the record and stamp defaults the controller may have left."""
    record = dict(state["record"])
    now_iso = state.get("now_iso") or RUNTIME.clock.now_iso()
    return {
        "record": record,
        "attempt": int(state.get("attempt", 1)),
        "now_iso": now_iso,
    }


# --------------------------------------------------------------------------
# diagnose
# --------------------------------------------------------------------------
def diagnose(state: AgentState) -> Dict[str, Any]:
    klass = classify(state["record"])
    return {
        "diagnosis": {
            "failure_class": klass.value,
            "reason": state["record"].get("error_reason", ""),
            "method": "deterministic reason-code lookup",
        }
    }


# --------------------------------------------------------------------------
# decide
# --------------------------------------------------------------------------
def decide_node(state: AgentState) -> Dict[str, Any]:
    prior_contacts = int(state["record"].get("_prior_contacts", 0))
    decision = decide(state["record"], prior_contacts=prior_contacts,
                      policy=RUNTIME.policy)
    out: Dict[str, Any] = {
        "decision": {
            "failure_class": decision.failure_class.value,
            "chosen": decision.chosen.value,
            "net_ev_paise": decision.net_ev_paise,
            "p_recover": round(decision.p_recover, 4),
            "ranked": decision.ranked,
            "stop": decision.stop,
            "stop_reason": decision.stop_reason,
            "rationale": decision.rationale,
        }
    }
    # A decision to abandon/escalate is terminal set HERE, before narrate.
    if decision.chosen is Intervention.GIVE_UP:
        out["terminal_status"] = TerminalStatus.ABANDONED.value
    elif decision.chosen is Intervention.ESCALATE:
        out["terminal_status"] = TerminalStatus.ESCALATED.value
    return out


def route_after_decide(state: AgentState) -> str:
    chosen = state["decision"]["chosen"]
    if chosen in (Intervention.GIVE_UP.value, Intervention.ESCALATE.value):
        return "abandon"
    return "guardrail_check"


# --------------------------------------------------------------------------
# guardrail_check
# --------------------------------------------------------------------------
def guardrail_check(state: AgentState) -> Dict[str, Any]:
    record = state["record"]
    intervention = Intervention(state["decision"]["chosen"])
    hours_since = record.get("_hours_since_last")
    result = check_guardrails(
        record,
        intervention,
        attempt=int(state["attempt"]),
        hours_since_last=hours_since,
        prior_contacts=int(record.get("_prior_contacts", 0)),
        policy=RUNTIME.policy,
    )
    out: Dict[str, Any] = {
        "guardrail": {
            "allowed": result.allowed,
            "checks": result.checks,
            "reason": result.reason,
        }
    }
    if result.blocked:
        # Blocked BEFORE execution -> escalated, terminal set here.
        out["terminal_status"] = TerminalStatus.ESCALATED.value
    return out


def route_after_guardrail(state: AgentState) -> str:
    return "execute" if state["guardrail"]["allowed"] else "abandon"


# --------------------------------------------------------------------------
# execute
# --------------------------------------------------------------------------
def execute(state: AgentState) -> Dict[str, Any]:
    record = state["record"]
    intervention = Intervention(state["decision"]["chosen"])
    outstanding = int(record.get("_outstanding_paise", record.get("amount_paise", 0)))

    if RUNTIME.executor is None:
        # No executor wired (pure control test): synthesise a no-op result.
        return {
            "execution": {
                "outcome": "failed",
                "amount_recovered_paise": 0,
                "api_called": False,
                "settles_async": spec(intervention).contacts_customer,
            },
            "terminal_status": (
                TerminalStatus.IN_PROGRESS.value
                if spec(intervention).contacts_customer
                else TerminalStatus.RECOVERED.value  # placeholder; overwritten below
            ),
        }

    result = RUNTIME.executor.execute(
        record,
        intervention,
        run_id=state["run_id"],
        attempt_number=int(state["attempt"]),
        attempt_at_iso=state["now_iso"],
        outstanding_paise=outstanding,
    )

    # Cap recovery at the outstanding balance -- never over-collect. (This is the
    # §14 over-collection fix: a partial debit followed by a full retry must not
    # recover more than is owed.)
    recovered = min(int(result.get("amount_recovered_paise", 0)), outstanding)
    result["amount_recovered_paise"] = recovered

    if result.get("settles_async"):
        terminal = TerminalStatus.IN_PROGRESS.value
    elif result.get("outcome") == "recovered" and recovered >= outstanding:
        terminal = TerminalStatus.RECOVERED.value
    elif result.get("outcome") == "recovered" and 0 < recovered < outstanding:
        terminal = TerminalStatus.IN_PROGRESS.value  # partial: residual remains
    else:
        terminal = TerminalStatus.IN_PROGRESS.value  # failed attempt; may retry

    return {"execution": result, "terminal_status": terminal}


# --------------------------------------------------------------------------
# narrate  (write-only LLM sidecar)
# --------------------------------------------------------------------------
def narrate(state: AgentState) -> Dict[str, Any]:
    prose = RUNTIME.narrator.narrate(dict(state))
    return {"narration": prose}


# --------------------------------------------------------------------------
# log
# --------------------------------------------------------------------------
def log(state: AgentState) -> Dict[str, Any]:
    record = state["record"]
    decision = state.get("decision", {})
    execution = state.get("execution", {})
    guardrail = state.get("guardrail", {})
    row = {
        "run_id": state["run_id"],
        "record_id": record.get("record_id"),
        "customer_id": record.get("customer_id"),   # on the row: cap query is a lookup
        "timestamp": state["now_iso"],
        "attempt_number": int(state["attempt"]),
        "chosen_action": decision.get("chosen"),
        "rationale": decision.get("rationale", ""),
        "guardrail_checks": guardrail.get("checks", {}),
        "model_score": decision.get("p_recover"),
        "outcome": state.get("terminal_status"),
        "amount_recovered_paise": int(execution.get("amount_recovered_paise", 0)),
        "idempotency_key": execution.get("idempotency_key", ""),
        "execution_mode": "simulated",
        "api_called": bool(execution.get("api_called", False)),
        "razorpay_entity_id": execution.get("razorpay_entity_id"),
        "was_mocked": bool(execution.get("was_mocked", False)),
        "mock_reason": execution.get("mock_reason", ""),
        "narration": state.get("narration", ""),
        "settles_async": bool(execution.get("settles_async", False)),
    }
    RUNTIME.audit.write(row)
    return {"audit_row": row}
