"""The two-axis outcome and the monotonic fold.

These lock gap-3 and gap-4: the fold is order-independent, a late pending cannot
regress a resolved attempt, a duplicate is a no-op, and source never changes the
fold.
"""

from __future__ import annotations

import random
from datetime import datetime

import pytest

from recoup.events import (
    ApiResult,
    FinancialResult,
    IllegalEvent,
    OutcomeEvent,
    Stage,
    fold,
    total_recovered,
)

T0 = datetime(2026, 1, 16, 9, 0, 0).isoformat()
T1 = datetime(2026, 1, 17, 9, 0, 0).isoformat()
T2 = datetime(2026, 1, 18, 9, 0, 0).isoformat()


def _event(stage, financial, *, attempt_n=1, recovered=0, amount=73_600,
           source="simulator", occurred=T0):
    return OutcomeEvent(
        run_id="run_1",
        record_id="rec_0001",
        customer_id="cust_0001",
        attempt_n=attempt_n,
        stage=stage,
        intervention="retry_now",
        is_contact=False,
        amount_paise=amount,
        occurred_at=occurred,
        ingested_at=occurred,
        api_result=ApiResult.ACCEPTED,
        financial_result=financial,
        amount_recovered_paise=recovered,
        source=source,
    )


# --------------------------------------------------------------------------
# Construction guards: illegal events are unconstructable
# --------------------------------------------------------------------------
def test_recovered_more_than_attempted_is_illegal():
    with pytest.raises(IllegalEvent):
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, recovered=80_000)


def test_recovered_event_must_recover_positive_money():
    with pytest.raises(IllegalEvent):
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, recovered=0)


def test_not_recovered_event_cannot_carry_money():
    with pytest.raises(IllegalEvent):
        _event(Stage.RESOLVED, FinancialResult.NOT_RECOVERED, recovered=100)


# --------------------------------------------------------------------------
# The fold
# --------------------------------------------------------------------------
def test_pending_then_resolved_recovers():
    events = [
        _event(Stage.PENDING, FinancialResult.PENDING),
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, recovered=73_600, occurred=T1),
    ]
    outcomes = fold(events)
    assert outcomes[1].financial_result is FinancialResult.RECOVERED
    assert outcomes[1].amount_recovered_paise == 73_600
    assert total_recovered(outcomes) == 73_600


def test_fold_is_order_independent():
    """Shuffle the events; the folded state is identical every time."""
    events = [
        _event(Stage.FIRED, FinancialResult.PENDING),
        _event(Stage.PENDING, FinancialResult.PENDING, occurred=T1),
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, recovered=73_600, occurred=T2),
    ]
    reference = fold(events)[1]
    for _ in range(20):
        shuffled = events[:]
        random.shuffle(shuffled)
        result = fold(shuffled)[1]
        assert result.financial_result is reference.financial_result
        assert result.amount_recovered_paise == reference.amount_recovered_paise


def test_late_pending_after_resolved_is_ignored():
    """An out-of-order webhook cannot regress a resolved attempt."""
    events = [
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, recovered=73_600),
        _event(Stage.PENDING, FinancialResult.PENDING, occurred=T2),  # arrives late
    ]
    outcomes = fold(events)
    assert outcomes[1].financial_result is FinancialResult.RECOVERED
    assert outcomes[1].amount_recovered_paise == 73_600


def test_duplicate_resolution_does_not_double_count():
    """Recovered rupees cannot double-count on a retried webhook."""
    events = [
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, recovered=73_600),
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, recovered=73_600),
    ]
    assert total_recovered(fold(events)) == 73_600


def test_contradictory_late_resolution_is_ignored():
    """First resolution wins; a contradicting one is a no-op."""
    events = [
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, recovered=73_600),
        _event(Stage.RESOLVED, FinancialResult.NOT_RECOVERED, recovered=0, occurred=T2),
    ]
    outcomes = fold(events)
    assert outcomes[1].financial_result is FinancialResult.RECOVERED


def test_source_does_not_change_the_fold():
    """Simulator and webhook events fold identically."""
    sim = fold([_event(Stage.RESOLVED, FinancialResult.RECOVERED,
                        recovered=73_600, source="simulator")])
    hook = fold([_event(Stage.RESOLVED, FinancialResult.RECOVERED,
                        recovered=73_600, source="webhook")])
    assert sim[1].financial_result is hook[1].financial_result
    assert sim[1].amount_recovered_paise == hook[1].amount_recovered_paise


def test_partial_resolution_is_tracked():
    events = [_event(Stage.RESOLVED, FinancialResult.PARTIAL, recovered=36_800)]
    outcomes = fold(events)
    assert outcomes[1].financial_result is FinancialResult.PARTIAL
    assert outcomes[1].amount_recovered_paise == 36_800


def test_separate_attempts_fold_separately():
    events = [
        _event(Stage.RESOLVED, FinancialResult.NOT_RECOVERED, attempt_n=1, recovered=0),
        _event(Stage.RESOLVED, FinancialResult.RECOVERED, attempt_n=2, recovered=73_600),
    ]
    outcomes = fold(events)
    assert outcomes[1].financial_result is FinancialResult.NOT_RECOVERED
    assert outcomes[2].financial_result is FinancialResult.RECOVERED
    assert total_recovered(outcomes) == 73_600
