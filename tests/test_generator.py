"""The ML evaluation story, defended by tests.

These are not checking that code runs.  Each one guards a claim that would
otherwise be unfalsifiable -- most importantly that the customer-grouped split
is load-bearing rather than something nice to write in a README.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from config.taxonomy import (
    ALL_CLASSES,
    CANDIDATES,
    REASON_MAP,
    FailureClass,
    Intervention,
    classify_reason,
    spec,
)
from recoup.clock import SimulatedClock
from recoup.environment import RecoveryEnvironment, liquidity
from recoup.generator import GeneratorConfig, generate
from recoup.ingest import detect
from recoup.models import AuditEntry, ExecutionMode, Outcome
from tests.conftest import make_customer, make_latent, make_record


# --------------------------------------------------------------------------
# The batch itself
# --------------------------------------------------------------------------
def test_deterministic():
    """Same seed -> identical batch.  The frozen batch in the repo is exactly
    reproducible, which is itself a hiring signal."""
    a = generate(GeneratorConfig(seed=99))
    b = generate(GeneratorConfig(seed=99))
    assert [r.to_json() for r in a.records] == [r.to_json() for r in b.records]
    assert a.holdout_customer_ids == b.holdout_customer_ids
    assert {k: v.to_json() for k, v in a.latent.items()} == {
        k: v.to_json() for k, v in b.latent.items()
    }


def test_all_classes_present(batch):
    """Full taxonomy coverage: a class with no records cannot be demonstrated."""
    seen = {detect(r) for r in batch.records}
    assert set(ALL_CLASSES) <= seen
    assert FailureClass.UNKNOWN not in seen


def test_customers_recur(batch):
    """GroupKFold needs groups larger than one.

    Without repeat customers, grouping by customer_id is identical to a random
    row split and the whole leakage defence is decorative.
    """
    assert len(batch.customers) < len(batch.records)
    assert batch.recurring_customer_count() >= 30


def test_holdout_is_customer_disjoint(batch):
    """No customer appears in both splits.  Asserted, not trusted."""
    train = {r.customer_id for r in batch.train_records}
    held = {r.customer_id for r in batch.holdout_records}
    assert train & held == set()
    assert held == set(batch.holdout_customer_ids)
    assert len(batch.train_records) + len(batch.holdout_records) == len(batch.records)


def test_seeded_dead_records_exist(batch, env):
    """Graceful give-up is demonstrable rather than asserted.

    A record seeded dead recovers essentially never, whatever you try -- so the
    agent stopping on it is a correct decision that can be *shown*.
    """
    dead = [rid for rid, truth in batch.latent.items() if truth.is_truly_dead]
    assert len(dead) >= 8
    from_class_five = [
        rid
        for rid in dead
        if batch.latent[rid].seeded_class is FailureClass.DO_NOT_HONOUR
    ]
    assert len(from_class_five) >= 5

    record = next(r for r in batch.records if r.record_id == dead[0])
    customer = batch.customers[record.customer_id]
    truth = batch.latent[record.record_id]
    for intervention in (Intervention.RETRY_NOW, Intervention.UPDATE_PAYMENT_METHOD):
        p = env.true_prob(record, customer, truth, intervention, record.failed_at)
        assert p < 0.01


# --------------------------------------------------------------------------
# The correlation band: three tests lock it in place
# --------------------------------------------------------------------------
def _trait_arrays(batch):
    tenure = np.array([c.tenure_days for c in batch.customers.values()], dtype=float)
    engagement = np.array([c.engagement for c in batch.customers.values()])
    regularity = np.array(
        [c.income_regularity for c in batch.customers.values()]
    )
    return tenure, engagement, regularity


def test_latent_traits_are_partially_but_not_fully_inferable(batch):
    """**GroupKFold is load-bearing, not theatre.**

    At correlation ~0.0 the traits are unlearnable noise and the model has
    nothing to find.  At ~1.0 they are perfectly recoverable from observables,
    two records from the same customer become conditionally independent given
    the features, and grouping by customer removes no shortcut at all.

    In between -- and only in between -- a customer carries persistent hidden
    information a model can only partly infer.
    """
    tenure, engagement, _ = _trait_arrays(batch)
    corr = float(np.corrcoef(tenure, engagement)[0, 1])
    assert 0.35 < corr < 0.85, f"engagement/tenure correlation {corr:.3f} outside band"

    # And the residual is real: an optimal linear read-off of every observable
    # customer feature still leaves most of the variance unexplained.
    design = np.column_stack(
        [
            np.ones_like(tenure),
            tenure,
            np.array([c.avg_payment_cents for c in batch.customers.values()], float),
            np.array([c.prior_failures for c in batch.customers.values()], float),
            np.array(
                [float(c.is_subscriber) for c in batch.customers.values()]
            ),
        ]
    )
    coef, *_ = np.linalg.lstsq(design, engagement, rcond=None)
    resid = engagement - design @ coef
    r2 = 1.0 - resid.var() / engagement.var()
    assert 0.05 < r2 < 0.80, f"engagement R^2 from observables is {r2:.3f}"


def test_income_regularity_sits_in_the_same_band(batch):
    """Regularity is mostly latent -- weaker than engagement, still not noise."""
    tenure, _, regularity = _trait_arrays(batch)
    corr = float(np.corrcoef(tenure, regularity)[0, 1])
    assert 0.15 < corr < 0.60, f"regularity/tenure correlation {corr:.3f} outside band"


def test_same_observables_still_give_different_traits(batch):
    """Two customers who look identical to the model are not identical.

    This is the concrete form of the band: if the traits were a deterministic
    function of observables, this spread would be zero.
    """
    buckets: dict[int, list[float]] = {}
    for c in batch.customers.values():
        buckets.setdefault(c.tenure_days // 120, []).append(c.engagement)
    spreads = [max(v) - min(v) for v in buckets.values() if len(v) >= 3]
    assert spreads, "expected several customers in at least one tenure bucket"
    assert max(spreads) > 0.25


# --------------------------------------------------------------------------
# Why a model is necessary at all
# --------------------------------------------------------------------------
def test_optimal_intervention_varies_by_customer_trait(env):
    """The propensity model is necessary, not decorative.

    If ``income_regularity`` only *scaled* the reward, the argmax intervention
    would be identical for every customer and a model would merely re-order one
    queue.  Because the trait changes **which** intervention wins, the agent has
    to learn a per-customer policy.
    """
    now = datetime(2026, 1, 16, 9)
    salary_day = 1
    winners = {}
    for label, regularity in (("regular", 1.0), ("irregular", 0.16)):
        customer = make_customer(salary_day=salary_day, income_regularity=regularity)
        record = make_record(customer=customer, amount_cents=70_000, failed_at=now)
        latent = make_latent(income_regularity=regularity)
        salary_window = SimulatedClock(now).next_high_liquidity_day(salary_day, 24.0)

        scored = {}
        for intervention in CANDIDATES[FailureClass.INSUFFICIENT_FUNDS]:
            if spec(intervention).terminal:
                continue
            when = (
                salary_window
                if intervention is Intervention.RETRY_SALARY_WINDOW
                else now
            )
            scored[intervention] = env.true_prob(
                record, customer, latent, intervention, when
            )
        winners[label] = max(scored, key=scored.get)

    assert winners["regular"] is Intervention.RETRY_SALARY_WINDOW
    assert winners["irregular"] is not Intervention.RETRY_SALARY_WINDOW


def test_salary_window_is_smooth():
    """No memorisable cliff.

    The first draft was a hard step (day 27 -> 0.0, day 28 -> 1.0).  A model
    could ace that by learning one boundary, which would be a bug dressed as a
    result.  Real liquidity ramps.
    """
    values = [liquidity(datetime(2026, 3, d, 9), 28) for d in range(1, 32)]
    steps = [abs(b - a) for a, b in zip(values, values[1:])]
    assert max(steps) < 0.40, f"largest one-day jump was {max(steps):.3f}"
    assert len({round(v, 3) for v in values}) >= 15
    assert min(values) > 0.0 and max(values) <= 1.0


def test_timing_changes_outcomes(batch, env):
    """The core thesis: *when* you retry changes whether you get paid."""
    flips = 0
    biggest_gap = 0.0
    for record in batch.records:
        if detect(record) is not FailureClass.INSUFFICIENT_FUNDS:
            continue
        customer = batch.customers[record.customer_id]
        latent = batch.latent[record.record_id]
        if latent.is_truly_dead:
            continue
        clock = SimulatedClock(record.failed_at)
        good = clock.next_high_liquidity_day(customer.salary_day, 24.0)
        bad = good + timedelta(days=13)
        p_good = env.true_prob(record, customer, latent, Intervention.RETRY_NOW, good)
        p_bad = env.true_prob(record, customer, latent, Intervention.RETRY_NOW, bad)
        biggest_gap = max(biggest_gap, p_good - p_bad)
        if env.sample_outcome(
            record, customer, latent, Intervention.RETRY_NOW, good
        ) != env.sample_outcome(
            record, customer, latent, Intervention.RETRY_NOW, bad
        ):
            flips += 1
    assert biggest_gap > 0.20, f"best timing gap was only {biggest_gap:.3f}"
    assert flips >= 3, "timing never flipped a sampled outcome"


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def test_detection_matches_labels(batch):
    """Taxonomy and generator agree.

    A self-consistency check: if the two ever drift apart, it surfaces here
    rather than at metrics time, when a wrong number has already reached a
    slide.
    """
    for record in batch.records:
        assert detect(record) is batch.latent[record.record_id].seeded_class


def test_classify_unknown_escalates():
    """Unmapped signals do not get guessed at.

    ``do_not_honour`` and ``gateway_error`` are the specific trap: both sound
    like PayPal reason codes, neither is one.
    """
    assert classify_reason("do_not_honour") is FailureClass.UNKNOWN
    assert classify_reason("gateway_error") is FailureClass.UNKNOWN
    assert classify_reason("invalid_card") is FailureClass.UNKNOWN
    assert classify_reason("") is FailureClass.UNKNOWN
    assert CANDIDATES[FailureClass.UNKNOWN] == (Intervention.ESCALATE,)


def test_taxonomy_reason_strings_are_unique():
    """One reason cannot map to two classes, or detection is ambiguous."""
    all_reasons = [r for reasons in REASON_MAP.values() for r in reasons]
    assert len(all_reasons) == len(set(all_reasons))
    assert all(r == r.lower().strip() for r in all_reasons)


# --------------------------------------------------------------------------
# The receipts
# --------------------------------------------------------------------------
def _entry(**overrides) -> AuditEntry:
    defaults = dict(
        run_id="run_1",
        record_id="rec_0001",
        timestamp=datetime(2026, 1, 16, 9),
        attempt_number=1,
        chosen_action=Intervention.RETRY_NOW,
        rationale="test",
        guardrail_checks={"cooldown": "pass"},
        model_score=0.42,
        outcome=Outcome.RECOVERED,
        amount_recovered_cents=73_600,
        idempotency_key="recoup:run_1:rec_0001:attempt:1",
        execution_mode=ExecutionMode.SIMULATED,
        api_called=False,
    )
    defaults.update(overrides)
    return AuditEntry(**defaults)


def test_audit_is_append_only(store):
    """Receipts are tamper-evident.

    Evidence that could be silently rewritten -- by a buggy node, or by a
    developer tidying up a bad run before recording the demo -- is not evidence.
    """
    import sqlite3

    store.append_audit(_entry())
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("UPDATE audit_log SET outcome = 'recovered'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("DELETE FROM audit_log")
    assert len(store.audit_for_run("run_1")) == 1


def test_audit_enums_serialise_to_strings(store):
    """Metric queries will not break silently.

    Every headline number is a GROUP BY over these columns.  If an enum reached
    SQLite as a repr, the grouping would fragment and the report would be wrong
    without anything raising.
    """
    store.append_audit(_entry())
    row = store.audit_for_run("run_1")[0]
    assert row["outcome"] == "recovered"
    assert row["chosen_action"] == "retry_now"
    assert row["execution_mode"] == "simulated"
    assert isinstance(row["guardrail_checks"], str)
    assert store.outcome_histogram("run_1") == {"recovered": 1}
