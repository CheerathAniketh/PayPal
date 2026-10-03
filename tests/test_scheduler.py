"""The outer batch scheduler: EV order, shared contact cap, cooldown re-enqueue,
budget, and the async lift.

These use small hand-built batches so the assertions are exact, plus one run
over the frozen batch to prove it terminates and produces a coherent summary.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.taxonomy import Intervention
from recoup.agent.outcome_sim import OutcomeSimulator
from recoup.agent.runtime import DemoAuditLog, DemoExecutor, configure
from recoup.agent.scheduler import BatchScheduler, SchedulerConfig
from recoup.environment import RecoveryEnvironment
from recoup.generator import hydrate_customers, load_frozen, load_latent
from recoup.models import FailedRecord
from tests.conftest import make_customer, make_record


@pytest.fixture(scope="module")
def batch_data():
    records, customers = load_frozen()
    customers = hydrate_customers(customers)
    latent = load_latent()
    return records, customers, latent


def _wire(customers, latent):
    configure(executor=DemoExecutor(RecoveryEnvironment(), customers, latent),
              audit=DemoAuditLog())


def test_full_batch_run_terminates_and_is_coherent(batch_data):
    records, customers, latent = batch_data
    _wire(customers, latent)
    config = SchedulerConfig(run_id="run_a", start=datetime(2026, 1, 16, 9))
    summary = BatchScheduler([r.to_json() for r in records], config).run()

    # Every record ends somewhere; the counts add up to the batch size.
    assert (
        summary.recovered_count
        + summary.escalated_count
        + summary.abandoned_count
        + summary.in_progress_count
        == summary.n_records
    )
    assert summary.total_at_risk_cents > 0
    assert 0.0 <= summary.recovery_rate <= 1.0
    # It respected the budget ceiling.
    assert summary.budget_spent_cents <= config.batch_budget_cents


def test_recovered_money_never_exceeds_total_at_risk(batch_data):
    records, customers, latent = batch_data
    _wire(customers, latent)
    summary = BatchScheduler(
        [r.to_json() for r in records],
        SchedulerConfig(run_id="run_b", start=datetime(2026, 1, 16, 9)),
    ).run()
    assert summary.recovered_cents <= summary.total_at_risk_cents


def test_contact_cap_bites_across_a_single_customers_records(batch_data):
    """One customer with several card_expired records cannot be contacted past
    the cap, no matter how many records they own."""
    _records, customers, latent = batch_data
    _wire(customers, latent)

    # Three card_expired records for ONE customer -> each wants a contact.
    cid = "cust_0000"
    recs = []
    for i in range(4):
        r = make_record(
            record_id=f"caprec_{i}",
            customer=make_customer(customer_id=cid, avg_payment_cents=300_000),
            amount_cents=400_000,           # big enough that a contact is worth it
            reason="card_expired",
        )
        d = r.to_json()
        recs.append(d)
    # These customer_ids are not in the demo executor's map; but card_expired
    # contacts park without calling the executor's environment, so that's fine.
    customers[cid] = make_customer(customer_id=cid, avg_payment_cents=300_000)
    latent[recs[0]["record_id"]] = None  # not used for parked contacts

    config = SchedulerConfig(run_id="run_cap", start=datetime(2026, 1, 16, 9))
    scheduler = BatchScheduler(recs, config)
    scheduler.run()
    # At most `max_contacts_per_window` contacts went out for this customer.
    assert scheduler.contacts_made <= config.policy.max_contacts_per_window


def test_async_resolution_lifts_the_recovery_number(batch_data):
    """The honest-metrics story: parked contacts resolved through ingest_outcome
    raise the recovered total above the synchronous-only figure."""
    records, customers, latent = batch_data
    _wire(customers, latent)

    sim = OutcomeSimulator(RecoveryEnvironment(), customers, latent,
                           response_window=timedelta(days=3))

    def on_park(record_dict, intervention, attempt_n, now, outstanding):
        clean = {k: v for k, v in record_dict.items() if not k.startswith("_")}
        sim.park(FailedRecord.from_json(clean), intervention, attempt_n, now,
                 outstanding)

    config = SchedulerConfig(run_id="run_async", start=datetime(2026, 1, 16, 9))
    scheduler = BatchScheduler([r.to_json() for r in records], config,
                               on_park=on_park)
    before = scheduler.run()

    events = sim.resolve_all(config.start + timedelta(days=30))
    for event in events:
        scheduler.ingest_outcome(event)
    after = scheduler._summarise()

    assert after.in_progress_count < before.in_progress_count
    assert after.recovered_cents >= before.recovered_cents


def test_ingest_outcome_is_idempotent(batch_data):
    """A webhook delivered twice cannot double-count the recovery."""
    records, customers, latent = batch_data
    _wire(customers, latent)
    from recoup.agent.outcomes import NormalisedEvent

    # A minimal one-record batch we can drive directly.
    rec = next(r.to_json() for r in records)
    scheduler = BatchScheduler([rec], SchedulerConfig(run_id="run_idem"))
    event = NormalisedEvent(
        event_id="evt_1",
        record_id=rec["record_id"],
        customer_id=rec["customer_id"],
        kind="customer_action",
        acted=True,
        amount_recovered_cents=10_000,
        occurred_at_iso=datetime(2026, 1, 20, 9).isoformat(),
    )
    first = scheduler.ingest_outcome(event)
    second = scheduler.ingest_outcome(event)  # duplicate
    assert first is not None
    assert second is None
    assert scheduler.states[rec["record_id"]].recovered_cents <= int(rec["amount_cents"])
