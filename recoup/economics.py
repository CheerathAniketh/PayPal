"""Net expected value, and a stopping threshold that is derived rather than
chosen.

The formula
-----------
    net_ev = p * (amount - gateway_fee) - contact_cost

* The gateway fee sits INSIDE the p-weighted term -- it is only charged when the
  charge succeeds.
* The contact cost sits OUTSIDE it -- you pinged the customer whether or not the
  money came back.

The threshold is derived, not chosen.  Setting ``net_ev = 0`` and solving:

    p* = contact_cost / (amount - fee)

So a Rs 10,000 charge justifies an attempt at a far lower probability than a
Rs 200 charge.  That amount-sensitivity is the entire reason for moving off
gross EV.

Anchoring the cost without inventing a number
---------------------------------------------
Rather than hand-picking paise, a policy anchor ``P_STAR_TARGET`` (the minimum
success probability worth acting on at a reference amount) is chosen, and the
base contact cost is back-solved from it.  The story stays honest: the number
comes from a stated policy, not from a coefficient pulled out of the air.

Pure module: no I/O, no DB, no clock, no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config.taxonomy import CANDIDATES, FailureClass, Intervention, spec
from recoup.money import rupees_to_paise


@dataclass(frozen=True)
class EconomicPolicy:
    """Every economic knob, in one auditable place."""

    # Gateway fee: a hard channel cost, charged only on a successful capture.
    # Razorpay's standard MDR is ~2%; a per-txn floor keeps tiny debits honest.
    gateway_fee_bps: int = 200                # 2.00%
    gateway_fee_floor_paise: int = 200        # Rs 2 minimum

    # The policy anchor. "At a reference charge of Rs 1,000, do not bother
    # contacting a customer unless the success probability is at least this."
    p_star_target: float = 0.20
    reference_amount_paise: int = field(default_factory=lambda: rupees_to_paise(1000))

    # Linear cost escalation (option 2 of the three considered). One clean knob
    # `k`: the nth customer-facing contact costs base * (1 + k*n). Makes the
    # economic brake back off *before* the hard cap trips.
    escalation_k: float = 0.6

    # A silent retry does not touch the customer, so its contact cost is ~zero
    # and it never escalates. Only customer-facing asks climb the ladder.
    silent_retry_cost_paise: int = 0

    # Abandon when the best net EV is at or below this. Zero by default: an
    # attempt that does not beat break-even is not worth making.
    tau_paise: int = 0

    def gateway_fee(self, amount_paise: int) -> int:
        """The fee charged on a successful capture of ``amount_paise``."""
        pct = (amount_paise * self.gateway_fee_bps) // 10_000
        return max(pct, self.gateway_fee_floor_paise)

    def base_contact_cost_paise(self) -> int:
        """Back-solve the base contact cost from the policy anchor.

            base = p_star_target * (reference_amount - fee(reference_amount))

        This is the cost such that, at the reference amount, break-even lands
        exactly on ``p_star_target``.
        """
        net = self.reference_amount_paise - self.gateway_fee(self.reference_amount_paise)
        return int(round(self.p_star_target * net))


DEFAULT_POLICY = EconomicPolicy()


class EconomicStopReason(str, Enum):
    WORTH_IT = "worth_it"
    BELOW_BREAK_EVEN = "below_break_even"          # p too low for this amount
    NO_POSITIVE_CANDIDATE = "no_positive_candidate"  # every option nets <= tau
    NOTHING_TO_TRY = "nothing_to_try"              # only terminal options left


def contact_cost_paise(
    prior_contacts: int, is_contact: bool, policy: EconomicPolicy = DEFAULT_POLICY
) -> int:
    """Cost of the *next* action.

    ``prior_contacts`` is how many customer-facing contacts have already gone to
    this customer.  A silent retry (``is_contact=False``) is free and does not
    escalate.
    """
    if not is_contact:
        return policy.silent_retry_cost_paise
    base = policy.base_contact_cost_paise()
    return int(round(base * (1.0 + policy.escalation_k * prior_contacts)))


def net_ev_paise(
    p_recover: float,
    amount_paise: int,
    *,
    recovery_fraction: float = 1.0,
    prior_contacts: int = 0,
    is_contact: bool = False,
    policy: EconomicPolicy = DEFAULT_POLICY,
) -> int:
    """net_ev = p * (collected - fee) - contact_cost, in integer paise.

    ``recovery_fraction`` handles partial debits: a smaller-amount retry
    collects less on success, so its whole p-weighted term shrinks -- which is
    exactly why ranking on probability alone would over-prefer it.
    """
    collected = int(amount_paise * recovery_fraction)
    fee = policy.gateway_fee(collected)
    gross = p_recover * (collected - fee)
    cost = contact_cost_paise(prior_contacts, is_contact, policy)
    return int(round(gross - cost))


def break_even_probability(
    amount_paise: int,
    *,
    prior_contacts: int = 0,
    is_contact: bool = True,
    recovery_fraction: float = 1.0,
    policy: EconomicPolicy = DEFAULT_POLICY,
) -> float:
    """p* = contact_cost / (collected - fee).

    Returns a value that may exceed 1.0 -- meaning no probability justifies the
    action at this amount (a tiny charge against a real contact cost).
    """
    collected = int(amount_paise * recovery_fraction)
    fee = policy.gateway_fee(collected)
    denom = collected - fee
    cost = contact_cost_paise(prior_contacts, is_contact, policy)
    if denom <= 0:
        return float("inf")
    return cost / denom


@dataclass
class RankedCandidate:
    intervention: Intervention
    p_recover: float
    net_ev_paise: int
    is_contact: bool


def rank_interventions(
    failure_class: FailureClass,
    amount_paise: int,
    p_by_intervention: Dict[Intervention, float],
    *,
    prior_contacts: int = 0,
    policy: EconomicPolicy = DEFAULT_POLICY,
) -> List[RankedCandidate]:
    """Score every non-terminal candidate for the class by net EV, descending.

    ``p_by_intervention`` is where the propensity model plugs in later: today
    the rules policy fills it, in Phase 4 the model does, and nothing here
    changes.
    """
    ranked: List[RankedCandidate] = []
    for intervention in CANDIDATES.get(failure_class, ()):
        action = spec(intervention)
        if action.terminal:
            continue
        p = p_by_intervention.get(intervention, 0.0)
        ev = net_ev_paise(
            p,
            amount_paise,
            recovery_fraction=action.recovery_fraction,
            prior_contacts=prior_contacts,
            is_contact=action.contacts_customer,
            policy=policy,
        )
        ranked.append(
            RankedCandidate(intervention, p, ev, action.contacts_customer)
        )
    ranked.sort(key=lambda c: c.net_ev_paise, reverse=True)
    return ranked


def should_stop_economic(
    ranked: List[RankedCandidate], policy: EconomicPolicy = DEFAULT_POLICY
) -> Tuple[bool, EconomicStopReason, Optional[RankedCandidate]]:
    """Stop when the best available net EV does not beat tau.

    Note the deliberate asymmetry with silent retries: a free silent retry has
    non-negative net EV for any p > 0, so this will never stop a record while a
    silent retry is a valid candidate.  That is correct -- a free attempt is
    always worth making -- and it means the retry *count* bound lives in the
    guardrail layer, not here.
    """
    if not ranked:
        return True, EconomicStopReason.NOTHING_TO_TRY, None
    best = ranked[0]
    if best.net_ev_paise <= policy.tau_paise:
        return True, EconomicStopReason.NO_POSITIVE_CANDIDATE, best
    return False, EconomicStopReason.WORTH_IT, best
