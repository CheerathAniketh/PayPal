"""The customer contact cap, the event store, and the async gateway seam.

These lock gap-2 (the cap is shared per customer, and the window rolls) and the
gap-3 seam (execute returns before the outcome is known; settlement folds
through the same ingest path).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import sqlite3

import pytest

from config.taxonomy import FailureClass, Intervention, PaymentMethod
from recoup.contact_cap import (
    ContactCapPolicy,
    contact_cap_result,
)
from recoup.environment import RecoveryEnvironment
from recoup.event_store import EventStore
from recoup.events import ApiResult, FinancialResult, OutcomeEvent, Stage
from recoup.gateway import SimulatedGateway
from recoup.models import Customer, FailedRecord, LatentTruth
from tests.conftest import make_customer, make_latent, make_record


def _store() -> EventStore:
    return EventStore(":memory:")


def _contact_event(store, customer_id, record_id, attempt_n, occurred):
    store.ingest_outcome(
        OutcomeEvent(
            run_id="run_1",
            record_id=record_id,
            customer_id=customer_id,
            attempt_n=attempt_n,
            stage=Stage.FIRED,
            intervention="update_payment_method",
            is_contact=True,
            amount_cents=50_000,
            occurred_at=occurred.isoformat(),
            ingested_at=occurred.isoformat(),
            api_result=ApiResult.ACCEPTED,
            financial_result=FinancialResult.PENDING,
        )
    )


# --------------------------------------------------------------------------
# The event store's idempotency
# --------------------------------------------------------------------------
def test_duplicate_event_is_a_no_op():
    """Ingesting the same event twice writes once and reports the duplicate."""
    store = _store()
    now = datetime(2026, 1, 16, 9)
    event = OutcomeEvent(
        run_id="run_1", record_id="rec_1", customer_id="cust_1", attempt_n=1,
        stage=Stage.RESOLVED, intervention="retry_now", is_contact=False,
        amount_cents=73_600, occurred_at=now.isoformat(), ingested_at=now.isoformat(),
        api_result=ApiResult.ACCEPTED, financial_result=FinancialResult.RECOVERED,
        amount_recovered_cents=73_600,
    )
    assert store.ingest_outcome(event) is True
    assert store.ingest_outcome(event) is False  # duplicate
    assert store.total_recovered_cents() == 73_600  # not doubled
    store.close()


def test_event_store_is_append_only():
    store = _store()
    now = datetime(2026, 1, 16, 9)
    _contact_event(store, "cust_1", "rec_1", 1, now)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("UPDATE outcome_events SET stage = 'resolved'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("DELETE FROM outcome_events")
    store.close()


# --------------------------------------------------------------------------
# The cap is shared per customer
# --------------------------------------------------------------------------
def test_cap_is_shared_across_a_customers_records():
    """The literal fix for the sorted-loop bug.

    Two different records for the same customer draw down ONE shared budget.
    """
    store = _store()
    now = datetime(2026, 2, 1, 9)
    policy = ContactCapPolicy(max_contacts=3, window_days=30)

    # Three contacts across three DIFFERENT records, same customer.
    _contact_event(store, "cust_1", "rec_a", 1, now - timedelta(days=1))
    _contact_event(store, "cust_1", "rec_b", 1, now - timedelta(days=2))
    _contact_event(store, "cust_1", "rec_c", 1, now - timedelta(days=3))

    result = contact_cap_result(store, "cust_1", now, policy)
    assert result.allowed is False
    assert result.used == 3
    store.close()


def test_a_different_customer_has_a_separate_budget():
    store = _store()
    now = datetime(2026, 2, 1, 9)
    policy = ContactCapPolicy(max_contacts=3, window_days=30)
    for i in range(3):
        _contact_event(store, "cust_1", f"rec_{i}", 1, now - timedelta(days=i + 1))
    assert contact_cap_result(store, "cust_1", now, policy).allowed is False
    assert contact_cap_result(store, "cust_2", now, policy).allowed is True
    store.close()


def test_old_contacts_age_out_of_the_rolling_window():
    """The window genuinely rolls; it is not a counter with a reset."""
    store = _store()
    now = datetime(2026, 3, 1, 9)
    policy = ContactCapPolicy(max_contacts=2, window_days=30)

    # Two old contacts (45 days ago) and none recent.
    _contact_event(store, "cust_1", "rec_a", 1, now - timedelta(days=45))
    _contact_event(store, "cust_1", "rec_b", 1, now - timedelta(days=40))
    result = contact_cap_result(store, "cust_1", now, policy)
    assert result.allowed is True   # both aged out
    assert result.used == 0

    # One recent contact still counts.
    _contact_event(store, "cust_1", "rec_c", 1, now - timedelta(days=5))
    assert contact_cap_result(store, "cust_1", now, policy).used == 1
    store.close()


def test_silent_retries_do_not_count_against_the_cap():
    """Only customer-facing contacts consume the budget."""
    store = _store()
    now = datetime(2026, 2, 1, 9)
    # A silent retry event (is_contact=False) for the same customer.
    store.ingest_outcome(
        OutcomeEvent(
            run_id="run_1", record_id="rec_1", customer_id="cust_1", attempt_n=1,
            stage=Stage.FIRED, intervention="retry_now", is_contact=False,
            amount_cents=50_000, occurred_at=now.isoformat(),
            ingested_at=now.isoformat(), api_result=ApiResult.ACCEPTED,
            financial_result=FinancialResult.PENDING,
        )
    )
    assert contact_cap_result(store, "cust_1", now, ContactCapPolicy()).used == 0
    store.close()


# --------------------------------------------------------------------------
# The async gateway seam
# --------------------------------------------------------------------------
def _isf_record() -> tuple[FailedRecord, Customer, LatentTruth]:
    customer = make_customer(income_regularity=0.9, salary_day=1)
    record = make_record(customer=customer, reason="insufficient_funds",
                         amount_cents=73_600)
    latent = make_latent(base_logodds=6.0, income_regularity=0.9)  # will recover
    return record, customer, latent


def test_execute_returns_before_the_outcome_is_known():
    """After firing, no resolved outcome exists yet -- the money result is in
    the future, resolved out-of-band."""
    store = _store()
    gw = SimulatedGateway(store, RecoveryEnvironment())
    now = datetime(2026, 1, 16, 9)
    record, customer, latent = _isf_record()

    gw.fire(record, customer, latent, Intervention.RETRY_NOW,
            run_id="run_1", attempt_n=1, now=now)

    outcomes = store.fold_record(record.record_id)
    assert outcomes[1].financial_result is FinancialResult.PENDING
    assert store.total_recovered_cents() == 0  # nothing settled yet
    store.close()


def test_settlement_resolves_through_the_same_ingest_path():
    """Advancing past the settle delay resolves the attempt into recovered
    money -- via ingest_outcome, the identical entry point a webhook uses."""
    store = _store()
    gw = SimulatedGateway(store, RecoveryEnvironment(),
                          settle_delay=timedelta(hours=24))
    now = datetime(2026, 1, 16, 9)
    record, customer, latent = _isf_record()
    gw.fire(record, customer, latent, Intervention.RETRY_NOW,
            run_id="run_1", attempt_n=1, now=now)

    # Nothing due yet.
    assert gw.settle_due(now + timedelta(hours=1)) == 0
    # Past the delay it settles.
    assert gw.settle_due(now + timedelta(hours=25)) == 1

    outcomes = store.fold_record(record.record_id)
    assert outcomes[1].financial_result is FinancialResult.RECOVERED
    assert store.total_recovered_cents() == 73_600
    store.close()


def test_settlement_is_idempotent_if_replayed():
    """Even if a resolution is somehow delivered twice, money is counted once."""
    store = _store()
    gw = SimulatedGateway(store, RecoveryEnvironment())
    now = datetime(2026, 1, 16, 9)
    record, customer, latent = _isf_record()
    gw.fire(record, customer, latent, Intervention.RETRY_NOW,
            run_id="run_1", attempt_n=1, now=now)
    gw.settle_all(now + timedelta(hours=25))
    total_once = store.total_recovered_cents()

    # Re-ingest the resolved event directly (a duplicate webhook).
    resolved = [e for e in store.events_for_record(record.record_id)
                if e.stage is Stage.RESOLVED][0]
    store.ingest_outcome(resolved)
    assert store.total_recovered_cents() == total_once
    store.close()
