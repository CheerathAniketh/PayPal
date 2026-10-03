"""The compiled graph: routing, terminal-before-narrate, one audit row per pass.

These run the real module-level ``graph`` with demo adapters wired to the frozen
batch, so they exercise the actual LangGraph assembly, not a mock of it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from config.taxonomy import FailureClass, Intervention, classify_reason
from recoup.agent.graph import graph
from recoup.agent.runtime import DemoAuditLog, DemoExecutor, configure
from recoup.environment import RecoveryEnvironment
from recoup.generator import hydrate_customers, load_frozen, load_latent


@pytest.fixture(scope="module")
def wired():
    records, customers = load_frozen()
    customers = hydrate_customers(customers)
    latent = load_latent()
    audit = DemoAuditLog()
    configure(executor=DemoExecutor(RecoveryEnvironment(), customers, latent),
              audit=audit)
    return records, latent, audit


def _run(records, klass, *, now=datetime(2026, 1, 27, 9), attempt=1, overrides=None):
    record = next(r for r in records if classify_reason(r.error_reason) is klass)
    rec = record.to_json()
    rec["_outstanding_cents"] = rec["amount_cents"]
    if overrides:
        rec.update(overrides)
    return graph.invoke(
        {"run_id": "run_test", "record": rec, "attempt": attempt,
         "now_iso": now.isoformat()}
    )


def test_all_five_classes_route_and_reach_a_terminal(wired):
    records, _latent, _audit = wired
    for klass in (
        FailureClass.INSUFFICIENT_FUNDS,
        FailureClass.BANK_DOWNTIME,
        FailureClass.CARD_EXPIRED,
        FailureClass.MANDATE_BROKEN,
        FailureClass.DO_NOT_HONOUR,
    ):
        out = _run(records, klass)
        assert out["diagnosis"]["failure_class"] == klass.value
        assert out["terminal_status"] in (
            "recovered", "in_progress", "escalated", "abandoned", "scheduled"
        )


def test_terminal_status_is_set_before_narrate(wired):
    """Every path stamps terminal_status, and narration is attached to it."""
    records, _latent, _audit = wired
    out = _run(records, FailureClass.BANK_DOWNTIME)
    assert "terminal_status" in out
    assert "narration" in out
    assert out["narration"]  # non-empty prose


def test_every_pass_writes_exactly_one_audit_row(wired):
    records, _latent, audit = wired
    before = len(audit.rows)
    _run(records, FailureClass.INSUFFICIENT_FUNDS)
    assert len(audit.rows) == before + 1
    row = audit.rows[-1]
    # customer_id on the row so the contact-cap query is a lookup, not a join.
    assert row["customer_id"]
    assert row["chosen_action"]
    assert row["outcome"] == "in_progress" or row["outcome"] in (
        "recovered", "escalated", "abandoned"
    )


def test_class_five_tries_once_then_escalates(wired):
    """The graceful give-up case: one controlled retry, then a human."""
    records, latent, _audit = wired
    dead = next(
        r for r in records
        if classify_reason(r.error_reason) is FailureClass.DO_NOT_HONOUR
        and latent[r.record_id].is_truly_dead
    )
    rec = dead.to_json(); rec["_outstanding_cents"] = rec["amount_cents"]
    # Attempt 2 exceeds the class-5 cap of 1 -> guardrail escalates.
    out = graph.invoke(
        {"run_id": "run_test", "record": rec, "attempt": 2,
         "now_iso": datetime(2026, 1, 27, 9).isoformat()}
    )
    assert out["terminal_status"] == "escalated"
    assert out["guardrail"]["checks"]["attempt_cap"] == "fail"


def test_guardrail_blocks_cancelled_mandate_retry(wired):
    """A cancelled mandate cannot be silently retried; it escalates.

    We force the decision toward a silent retry by presenting a cancelled
    mandate as an insufficient_funds failure (so a retry is the argmax), which
    the mandate guardrail then refuses.
    """
    records, _latent, _audit = wired
    out = _run(
        records, FailureClass.INSUFFICIENT_FUNDS,
        overrides={"is_mandate_debit": True, "subscription_status": "cancelled"},
    )
    assert out["terminal_status"] == "escalated"
    assert out["guardrail"]["checks"]["mandate_active"] == "fail"


def test_amount_over_cap_escalates(wired):
    records, _latent, _audit = wired
    out = _run(
        records, FailureClass.INSUFFICIENT_FUNDS,
        overrides={"amount_cents": 900_000, "_outstanding_cents": 900_000},
    )
    assert out["terminal_status"] == "escalated"
    assert out["guardrail"]["checks"]["amount_gate"] == "fail"


def test_recovery_never_exceeds_outstanding(wired):
    """The §14 over-collection fix: recovered is capped at the outstanding
    balance even if the executor reports more."""
    records, _latent, _audit = wired
    out = _run(
        records, FailureClass.BANK_DOWNTIME,
        overrides={"_outstanding_cents": 10_000},  # tiny residual
    )
    recovered = int(out["execution"].get("amount_recovered_cents", 0))
    assert recovered <= 10_000
