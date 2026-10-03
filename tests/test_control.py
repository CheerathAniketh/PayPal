"""The control plane: classify / decide / check_guardrails as pure functions.

The point of these is that the brain is testable with zero graph, zero I/O, zero
model -- which is the whole reason the decision logic lives in ``control.py`` and
not inside the nodes.
"""

from __future__ import annotations

import pytest

from config.taxonomy import FailureClass, Intervention
from recoup.agent.control import (
    DEFAULT_POLICY,
    Policy,
    _p_recover,
    check_guardrails,
    classify,
    decide,
)
from recoup.money import rupees_to_cents


def _record(**overrides):
    base = {
        "record_id": "rec_1",
        "customer_id": "cust_1",
        "amount_cents": rupees_to_cents(2000),
        "error_reason": "insufficient_funds",
        "prior_retries": 0,
        "customer_avg_payment_cents": rupees_to_cents(1500),
        "is_mandate_debit": False,
        "pre_debit_notified": True,
        "subscription_status": "active",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------
def test_classify_is_a_lookup_not_a_guess():
    assert classify(_record(error_reason="insufficient_funds")) is FailureClass.INSUFFICIENT_FUNDS
    assert classify(_record(error_reason="card_expired")) is FailureClass.CARD_EXPIRED
    # A non-Razorpay string is UNKNOWN, never coerced to a plausible class.
    assert classify(_record(error_reason="do_not_honour")) is FailureClass.UNKNOWN


# --------------------------------------------------------------------------
# decide
# --------------------------------------------------------------------------
def test_decide_picks_argmax_net_ev():
    decision = decide(_record(amount_cents=rupees_to_cents(5000)))
    assert decision.chosen is not Intervention.GIVE_UP
    assert decision.stop is False
    # The chosen row is the top of the ranked list by net EV.
    evs = [r["net_ev_cents"] for r in decision.ranked]
    assert evs == sorted(evs, reverse=True)
    assert decision.net_ev_cents == evs[0]


def test_decide_abandons_when_only_a_costly_contact_is_left():
    """A tiny card_expired charge gives up: the only candidate is a paid contact
    whose cost exceeds the expected recovery on so small an amount.

    (A do_not_honour retry, by contrast, is a FREE silent retry and is always
    worth making -- so the abandon case must be one with no free option.)
    """
    decision = decide(_record(
        error_reason="card_expired",  # only candidate: update_payment_method
        amount_cents=rupees_to_cents(150),
    ))
    assert decision.chosen is Intervention.GIVE_UP
    assert decision.stop is True


def test_decide_escalates_unknown():
    decision = decide(_record(error_reason="totally_unmapped_reason"))
    assert decision.chosen is Intervention.ESCALATE
    assert decision.failure_class is FailureClass.UNKNOWN


def test_p_recover_is_the_single_swap_point():
    """The Phase-4 hook: one function, called per candidate, returns a [0,1]
    probability -- swappable for model.predict_proba with no change to decide().
    """
    p = _p_recover(_record(), Intervention.RETRY_SALARY_WINDOW)
    assert 0.0 <= p <= 1.0
    # Retry fatigue lowers it.
    p_tired = _p_recover(_record(prior_retries=3), Intervention.RETRY_SALARY_WINDOW)
    assert p_tired < p


# --------------------------------------------------------------------------
# check_guardrails  -- compliance, separate from economics
# --------------------------------------------------------------------------
def test_cooldown_blocks_a_too_soon_retry():
    result = check_guardrails(
        _record(), Intervention.RETRY_NOW,
        attempt=1, hours_since_last=2.0, prior_contacts=0,
    )
    assert result.blocked
    assert result.checks["cooldown"] == "fail"


def test_amount_gate_blocks_a_large_autonomous_retry():
    result = check_guardrails(
        _record(amount_cents=rupees_to_cents(9000)), Intervention.RETRY_NOW,
        attempt=1, hours_since_last=None, prior_contacts=0,
    )
    assert result.blocked
    assert result.checks["amount_gate"] == "fail"


def test_cancelled_mandate_cannot_be_silently_retried():
    result = check_guardrails(
        _record(is_mandate_debit=True, subscription_status="cancelled"),
        Intervention.RETRY_NOW,
        attempt=1, hours_since_last=None, prior_contacts=0,
    )
    assert result.blocked
    assert result.checks["mandate_active"] == "fail"


def test_emandate_without_pre_debit_notice_is_blocked():
    result = check_guardrails(
        _record(is_mandate_debit=True, pre_debit_notified=False,
                subscription_status="active"),
        Intervention.RETRY_NOW,
        attempt=1, hours_since_last=None, prior_contacts=0,
    )
    assert result.blocked
    assert result.checks["pre_debit_notice"] == "fail"


def test_contact_cap_blocks_a_further_contact():
    result = check_guardrails(
        _record(error_reason="card_expired"), Intervention.UPDATE_PAYMENT_METHOD,
        attempt=1, hours_since_last=None,
        prior_contacts=DEFAULT_POLICY.max_contacts_per_window,
    )
    assert result.blocked
    assert result.checks["contact_cap"] == "fail"


def test_a_clean_retry_passes_every_guardrail():
    result = check_guardrails(
        _record(), Intervention.RETRY_NOW,
        attempt=1, hours_since_last=48.0, prior_contacts=0,
    )
    assert result.allowed
    assert all(v == "pass" for v in result.checks.values())


def test_economics_and_compliance_are_separate_predicates():
    """A record can be economically worth it AND compliance-blocked -- the two
    verdicts are independent, for different logged causes."""
    record = _record(amount_cents=rupees_to_cents(9000))  # big -> worth it
    decision = decide(record)
    assert decision.stop is False  # economically worth trying
    guard = check_guardrails(
        record, Intervention(decision.chosen),
        attempt=1, hours_since_last=None, prior_contacts=0,
    )
    assert guard.blocked  # but the amount gate refuses it
