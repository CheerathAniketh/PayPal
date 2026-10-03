"""Net EV and the derived threshold.

These lock the gap-1 claims: the threshold is derived from a policy anchor, big
charges justify attempts at lower probability, and the escalating contact cost
flips "worth it" to "not worth it" before the hard cap ever trips.
"""

from __future__ import annotations

import pytest

from config.taxonomy import FailureClass, Intervention
from recoup.economics import (
    DEFAULT_POLICY,
    EconomicPolicy,
    EconomicStopReason,
    break_even_probability,
    contact_cost_cents,
    net_ev_cents,
    rank_interventions,
    should_stop_economic,
)
from recoup.money import rupees_to_cents


def test_break_even_at_reference_amount_equals_target():
    """The threshold is derived from a policy anchor, not a magic constant.

    At the reference amount, break-even lands exactly on ``p_star_target``.
    """
    policy = DEFAULT_POLICY
    p_star = break_even_probability(
        policy.reference_amount_cents, prior_contacts=0, policy=policy
    )
    assert p_star == pytest.approx(policy.p_star_target, abs=1e-3)


def test_break_even_falls_as_amount_rises():
    """Big charges justify attempts at lower probability.

    This is the entire headline behaviour of net EV over gross EV.
    """
    small = break_even_probability(rupees_to_cents(500))
    medium = break_even_probability(rupees_to_cents(2000))
    large = break_even_probability(rupees_to_cents(10000))
    assert small > medium > large
    assert large < 0.05


def test_tiny_charge_is_never_worth_a_contact():
    """p* > 1 for a small charge: no probability justifies the contact cost."""
    p_star = break_even_probability(rupees_to_cents(150))
    assert p_star > 1.0


def test_gateway_fee_is_inside_the_p_weighted_term():
    """The fee is only paid on success, so at p=0 net EV is just -contact_cost."""
    amount = rupees_to_cents(1000)
    ev_zero = net_ev_cents(0.0, amount, is_contact=True)
    assert ev_zero == -contact_cost_cents(0, True)


def test_silent_retry_is_free_and_never_escalates():
    """A silent retry does not touch the customer: zero cost, no ladder."""
    assert contact_cost_cents(0, is_contact=False) == 0
    assert contact_cost_cents(5, is_contact=False) == 0
    # A free attempt has non-negative net EV for any p > 0.
    assert net_ev_cents(0.01, rupees_to_cents(1000), is_contact=False) >= 0


def test_escalating_contact_cost_flips_worth_it_to_not_worth_it():
    """The gap-1 x gap-2 hinge: the economic brake fires before the hard cap.

    Hold probability and amount fixed; only the number of prior contacts rises.
    Net EV must cross from positive to non-positive as the contact cost climbs.
    """
    amount = rupees_to_cents(900)
    p = 0.35
    first = net_ev_cents(p, amount, prior_contacts=0, is_contact=True)
    later = net_ev_cents(p, amount, prior_contacts=6, is_contact=True)
    assert first > 0
    assert later < first
    # And there exists a contact index where it goes non-positive.
    flipped = any(
        net_ev_cents(p, amount, prior_contacts=n, is_contact=True) <= 0
        for n in range(0, 12)
    )
    assert flipped


def test_contact_cost_escalates_linearly():
    """cost(n) = base * (1 + k*n)."""
    policy = EconomicPolicy()
    base = policy.base_contact_cost_cents()
    assert contact_cost_cents(0, True, policy) == base
    c1 = contact_cost_cents(1, True, policy)
    c2 = contact_cost_cents(2, True, policy)
    # Equal successive increments -> linear.
    assert (c2 - c1) == pytest.approx(c1 - base, abs=1)


def test_rank_prefers_higher_net_ev_not_higher_probability():
    """A partial debit with higher p can still lose to a full debit.

    ``retry_smaller_amount`` collects half; ranking on probability would prefer
    it, ranking on net EV need not.
    """
    amount = rupees_to_cents(3000)
    p = {
        Intervention.RETRY_SALARY_WINDOW: 0.45,
        Intervention.RETRY_NOW: 0.40,
        Intervention.RETRY_SMALLER_AMOUNT: 0.75,  # highest p, half the money
    }
    ranked = rank_interventions(
        FailureClass.INSUFFICIENT_FUNDS, amount, p
    )
    top = ranked[0]
    # The half-money option should not automatically win despite highest p.
    assert top.intervention is Intervention.RETRY_SALARY_WINDOW
    # net EV ordering is respected
    evs = [c.net_ev_cents for c in ranked]
    assert evs == sorted(evs, reverse=True)


def test_should_stop_when_no_candidate_beats_tau():
    amount = rupees_to_cents(1000)
    p = {Intervention.RETRY_SALARY_WINDOW: 0.001, Intervention.RETRY_NOW: 0.0}
    ranked = rank_interventions(FailureClass.INSUFFICIENT_FUNDS, amount, p)
    # Silent retries are free, so a positive-p silent retry keeps it alive; use
    # a policy with a positive tau to force a stop decision on weak EV.
    policy = EconomicPolicy(tau_cents=rupees_to_cents(5))
    stop, reason, best = should_stop_economic(ranked, policy)
    assert stop is True
    assert reason is EconomicStopReason.NO_POSITIVE_CANDIDATE


def test_should_continue_when_a_candidate_is_worth_it():
    amount = rupees_to_cents(5000)
    p = {Intervention.RETRY_SALARY_WINDOW: 0.5, Intervention.RETRY_NOW: 0.3}
    ranked = rank_interventions(FailureClass.INSUFFICIENT_FUNDS, amount, p)
    stop, reason, best = should_stop_economic(ranked)
    assert stop is False
    assert reason is EconomicStopReason.WORTH_IT
    assert best.net_ev_cents > 0


def test_empty_candidate_set_stops_with_nothing_to_try():
    ranked = rank_interventions(FailureClass.UNKNOWN, rupees_to_cents(1000), {})
    stop, reason, _ = should_stop_economic(ranked)
    assert stop is True
    assert reason is EconomicStopReason.NOTHING_TO_TRY
