"""Financial integrity: money exactness, idempotency, run isolation, lifecycle.

The strongest test in the file is
``database_physically_rejects_duplicate_idempotency_key``: it proves
double-charging is impossible, not merely unlikely.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from config.taxonomy import Intervention
from recoup.idempotency import (
    IllegalTransition,
    RecordStatus,
    assert_transition,
    attempt_key,
    is_terminal,
    receipt_id,
    staged_key,
    PAYPAL_REFERENCE_LIMIT,
)
from recoup.models import AuditEntry, ExecutionMode, Outcome
from recoup.money import (
    format_usd,
    cents_to_dollars,
    dollars_to_cents,
    split_cents,
    total_cents,
)
from tests.conftest import make_record


# --------------------------------------------------------------------------
# Money is exact
# --------------------------------------------------------------------------
def test_money_is_exact_where_float_would_drift():
    """0.1 + 0.2 == 0.3, in cents."""
    assert dollars_to_cents("0.1") + dollars_to_cents("0.2") == dollars_to_cents("0.3")
    assert 0.1 + 0.2 != 0.3  # the hazard this removes


def test_summing_many_amounts_stays_exact():
    """Totals match the sum of parts.

    A reviewer who totals the audit log by hand must get the same number the
    report prints.
    """
    amounts = [dollars_to_cents(f"{v}.33") for v in range(1, 200)]
    assert total_cents(amounts) == sum(amounts)
    # And no float ever sneaks in.
    with pytest.raises(TypeError):
        total_cents([100, 2.5])


def test_all_stored_amounts_are_integers():
    assert isinstance(dollars_to_cents(736.0), int)
    assert isinstance(dollars_to_cents("99.99"), int)
    assert dollars_to_cents("736.00") == 73_600


def test_partial_debit_rounds_down_never_up():
    """Rounding cannot inflate recovery.

    When a rounding direction must be chosen, choose the one that cannot inflate
    the headline recovery figure.
    """
    assert split_cents(73_601, 0.5) == 36_800  # 36800.5 -> floor
    assert split_cents(101, 0.5) == 50         # 50.5 -> floor
    assert split_cents(100, 1.0) == 100
    with pytest.raises(ValueError):
        split_cents(100, 1.5)


def test_display_helpers_are_display_only():
    assert format_usd(73_600) == "$736.00"
    assert format_usd(-5000) == "-$50.00"
    assert cents_to_dollars(73_600) == 736.0


# --------------------------------------------------------------------------
# Idempotency keys
# --------------------------------------------------------------------------
def test_key_is_deterministic():
    """A resumed run cannot double-charge: it derives the identical key."""
    a = attempt_key("run_1", "rec_0001", 1)
    b = attempt_key("run_1", "rec_0001", 1)
    assert a == b == "recoup:run_1:rec_0001:attempt:1"


def test_key_differs_per_attempt_and_per_run():
    """Legitimate retries are not blocked; re-running is a new set of ops."""
    base = attempt_key("run_1", "rec_0001", 1)
    assert attempt_key("run_1", "rec_0001", 2) != base
    assert attempt_key("run_2", "rec_0001", 1) != base
    # The staged key extends, never collides across stages.
    assert staged_key("run_1", "rec_0001", 1, "pending") != staged_key(
        "run_1", "rec_0001", 1, "resolved"
    )


def test_receipt_fits_razorpay_limit():
    """Uniqueness survives the 40-character squeeze."""
    seen = set()
    for run in ("run_1", "run_2"):
        for rec in ("rec_0001", "rec_9999"):
            for attempt in (1, 2, 3):
                r = receipt_id(run, rec, attempt)
                assert len(r) <= PAYPAL_REFERENCE_LIMIT
                seen.add(r)
    assert len(seen) == 12  # all distinct


# --------------------------------------------------------------------------
# The database guarantee
# --------------------------------------------------------------------------
def _entry(store_run="run_1", record_id="rec_0001", attempt=1, key=None):
    return AuditEntry(
        run_id=store_run,
        record_id=record_id,
        timestamp=datetime(2026, 1, 16, 9),
        attempt_number=attempt,
        chosen_action=Intervention.RETRY_NOW,
        rationale="test",
        guardrail_checks={},
        model_score=None,
        outcome=Outcome.FAILED,
        amount_recovered_cents=0,
        idempotency_key=key or attempt_key(store_run, record_id, attempt),
        execution_mode=ExecutionMode.SIMULATED,
        api_called=False,
    )


def test_database_physically_rejects_duplicate_idempotency_key(store):
    """**Double-charging is impossible, not unlikely.**

    A unique index on ``idempotency_key`` means a duplicate attempt raises
    IntegrityError rather than quietly writing a second charge.
    """
    store.append_audit(_entry())
    with pytest.raises(sqlite3.IntegrityError):
        store.append_audit(_entry())  # identical key


def test_runs_do_not_blend_metrics(store):
    """Re-running does not double every number.

    Each audit row carries ``run_id``; a metric for one run is isolated by it.
    """
    store.append_audit(_entry(store_run="run_1"))
    store.append_audit(_entry(store_run="run_2"))
    assert len(store.audit_for_run("run_1")) == 1
    assert len(store.audit_for_run("run_2")) == 1


# --------------------------------------------------------------------------
# Record lifecycle
# --------------------------------------------------------------------------
def test_records_start_pending_and_track_progress(store):
    record = make_record()
    store.upsert_record(record, detected_class="insufficient_funds")
    assert store.get_status(record.record_id) is RecordStatus.PENDING
    store.set_status(record.record_id, RecordStatus.IN_PROGRESS)
    store.set_status(record.record_id, RecordStatus.RECOVERED)
    assert store.get_status(record.record_id) is RecordStatus.RECOVERED


def test_pending_can_recover_directly(store):
    """A first attempt may simply succeed.

    The original table forced a detour through IN_PROGRESS; the state machine
    refused that at runtime, which was the machine working and the table wrong.
    """
    assert assert_transition(RecordStatus.PENDING, RecordStatus.RECOVERED)


def test_terminal_records_cannot_be_reopened(store):
    """Stopped means stopped.

    Reopening a terminal record would mean contacting a customer the agent
    already decided to stop contacting.
    """
    record = make_record()
    store.upsert_record(record)
    store.set_status(record.record_id, RecordStatus.ESCALATED)
    assert is_terminal(RecordStatus.ESCALATED)
    with pytest.raises(IllegalTransition):
        store.set_status(record.record_id, RecordStatus.IN_PROGRESS)


def test_partial_recovery_is_not_terminal():
    """Residual balance stays workable.

    A partial debit leaves money still at risk, so the agent may keep working
    the remainder.
    """
    assert not is_terminal(RecordStatus.PARTIALLY_RECOVERED)
    assert assert_transition(
        RecordStatus.PARTIALLY_RECOVERED, RecordStatus.RECOVERED
    )


def test_outstanding_balance_tracks_partial_recovery(store):
    record = make_record(amount_cents=73_600)
    store.upsert_record(record)
    assert store.outstanding_cents(record.record_id) == 73_600
    store.add_recovered(record.record_id, 36_800)
    assert store.outstanding_cents(record.record_id) == 36_800
