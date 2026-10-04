"""The failure taxonomy: the contract every other module reads.

Five failure classes, a deterministic reason-code lookup, and the bounded set of
interventions the agent may choose from.

Why classification is a lookup and not a model
----------------------------------------------
PayPal hands you the reason in its error response (``details[].issue`` on the
Orders API), which can be handled programmatically.  Putting a classifier
on top of a lookup table is ML-as-decoration.  The ML in this project is
reserved for the genuinely uncertain question -- *will this recover?* -- not the
solved one -- *what broke?*

What was verified, and how: through the PayPal sandbox's negative-testing
header (``PayPal-Mock-Response``, see ``scripts/probe_mock_errors.py``) the
capture endpoint accepted INSTRUMENT_DECLINED, TRANSACTION_REFUSED,
PAYER_ACTION_REQUIRED, ORDER_NOT_APPROVED, PAYER_ACCOUNT_RESTRICTED and
TRANSACTION_LIMIT_EXCEEDED and returned them as ``details[0].issue``.
INTERNAL_SERVER_ERROR returns HTTP 500 with no ``issue``.  The mock bodies are
canned, so this confirms the vocabulary, not live decline behaviour.

Two reasons used in the synthetic data, insufficient_funds and card_expired,
were rejected by the sandbox mock mechanism.  They come from the generator
only and are not confirmed PayPal behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Tuple


# --------------------------------------------------------------------------
# Failure classes
# --------------------------------------------------------------------------
class FailureClass(str, Enum):
    """The five recoverable-failure archetypes, plus an explicit unknown."""

    INSUFFICIENT_FUNDS = "insufficient_funds"
    BANK_DOWNTIME = "bank_downtime"
    CARD_EXPIRED = "card_expired"
    MANDATE_BROKEN = "mandate_broken"
    DO_NOT_HONOUR = "do_not_honour"
    UNKNOWN = "unknown"


class PaymentMethod(str, Enum):
    CARD = "card"
    PAYPAL_BALANCE = "paypal_balance"
    BANK_ACCOUNT = "bank_account"


# --------------------------------------------------------------------------
# Reason-code lookup
# --------------------------------------------------------------------------
# Source: PayPal Orders v2 `details[].issue` values. INSTRUMENT_DECLINED,
# TRANSACTION_REFUSED, PAYER_ACTION_REQUIRED, ORDER_NOT_APPROVED,
# PAYER_ACCOUNT_RESTRICTED, TRANSACTION_LIMIT_EXCEEDED and HTTP 500
# INTERNAL_SERVER_ERROR were all returned by the sandbox
# (scripts/probe_mock_errors.py). CARD_EXPIRED is in PayPal's docs but the
# sandbox mock header rejects it (403). "insufficient_funds" is an INTERNAL
# label: PayPal returned no such issue code in testing, so it is only
# reachable from synthetic data.
# Keys are lowercase because classify_reason() lowercases its input.
# payer_account_restricted is deliberately unmapped: retrying a restricted
# account is pointless, so it falls through to UNKNOWN -> ESCALATE.
REASON_MAP: Mapping[FailureClass, Tuple[str, ...]] = {
    FailureClass.INSUFFICIENT_FUNDS: (
        "insufficient_funds",
        "transaction_limit_exceeded",
    ),
    FailureClass.BANK_DOWNTIME: (
        "internal_server_error",
    ),
    FailureClass.CARD_EXPIRED: (
        "card_expired",
    ),
    FailureClass.MANDATE_BROKEN: (
        # Old name kept for now: "the buyer must act again before money can
        # move", which is what these two PayPal issues mean.
        "payer_action_required",
        "order_not_approved",
    ),
    FailureClass.DO_NOT_HONOUR: (
        "instrument_declined",
        "transaction_refused",
    ),
}

_REASON_INDEX: Dict[str, FailureClass] = {
    reason: klass for klass, reasons in REASON_MAP.items() for reason in reasons
}

# The `source` vocabulary is per payment method, not global.
# Synthetic attribute: PayPal's Orders v2 error bodies carry no `source` field
# that we verified, so these labels only describe where a simulated failure
# originated.
SOURCE_VOCABULARY: Mapping[PaymentMethod, Tuple[str, ...]] = {
    PaymentMethod.CARD: ("issuer_bank", "gateway", "internal"),
    PaymentMethod.PAYPAL_BALANCE: ("paypal", "gateway", "internal"),
    PaymentMethod.BANK_ACCOUNT: ("issuer_bank", "paypal", "internal"),
}


def classify_reason(reason: str) -> FailureClass:
    """Map a PayPal reason code to a failure class.

    Returns :data:`FailureClass.UNKNOWN` for anything unmapped rather than
    guessing at the nearest class.  Escalation is the default for anything
    unrecognised -- graceful failure is wired in at step one, not bolted on.
    """
    if not reason:
        return FailureClass.UNKNOWN
    return _REASON_INDEX.get(reason.strip().lower(), FailureClass.UNKNOWN)


def known_reasons() -> Tuple[str, ...]:
    return tuple(sorted(_REASON_INDEX))


# --------------------------------------------------------------------------
# Interventions
# --------------------------------------------------------------------------
class Intervention(str, Enum):
    RETRY_NOW = "retry_now"
    RETRY_DELAYED = "retry_delayed"
    RETRY_SALARY_WINDOW = "retry_salary_window"
    RETRY_SMALLER_AMOUNT = "retry_smaller_amount"
    UPDATE_PAYMENT_METHOD = "update_payment_method"
    RE_AUTH_MANDATE = "re_auth_mandate"
    ESCALATE = "escalate"
    GIVE_UP = "give_up"


@dataclass(frozen=True)
class InterventionSpec:
    """Flags the guardrail and economics layers read.

    ``recovery_fraction`` is what lets the agent rank by expected *money*
    rather than by probability: ``retry_smaller_amount`` collects half the
    outstanding balance when it succeeds, so a straight probability argmax
    would systematically over-prefer it.
    """

    intervention: Intervention
    description: str
    contacts_customer: bool          # counts against the contact cap
    touches_instrument: bool         # a debit attempt on the instrument
    requires_consent: bool           # must not run without fresh consent
    terminal: bool                   # ends processing for the record
    recovery_fraction: float         # share of outstanding balance collected


INTERVENTIONS: Mapping[Intervention, InterventionSpec] = {
    Intervention.RETRY_NOW: InterventionSpec(
        Intervention.RETRY_NOW,
        "Re-attempt the debit immediately.",
        contacts_customer=False, touches_instrument=True,
        requires_consent=False, terminal=False, recovery_fraction=1.0,
    ),
    Intervention.RETRY_DELAYED: InterventionSpec(
        Intervention.RETRY_DELAYED,
        "Re-attempt the debit after a cooldown, once transient errors clear.",
        contacts_customer=False, touches_instrument=True,
        requires_consent=False, terminal=False, recovery_fraction=1.0,
    ),
    Intervention.RETRY_SALARY_WINDOW: InterventionSpec(
        Intervention.RETRY_SALARY_WINDOW,
        "Schedule the debit for the customer's next high-liquidity day.",
        contacts_customer=False, touches_instrument=True,
        requires_consent=False, terminal=False, recovery_fraction=1.0,
    ),
    Intervention.RETRY_SMALLER_AMOUNT: InterventionSpec(
        Intervention.RETRY_SMALLER_AMOUNT,
        "Debit a partial amount; the residual balance stays at risk.",
        contacts_customer=False, touches_instrument=True,
        requires_consent=False, terminal=False, recovery_fraction=0.5,
    ),
    Intervention.UPDATE_PAYMENT_METHOD: InterventionSpec(
        Intervention.UPDATE_PAYMENT_METHOD,
        "Send the customer a link to update their payment instrument.",
        contacts_customer=True, touches_instrument=False,
        requires_consent=False, terminal=False, recovery_fraction=1.0,
    ),
    Intervention.RE_AUTH_MANDATE: InterventionSpec(
        Intervention.RE_AUTH_MANDATE,
        "Send the customer a re-authorisation / re-consent link.",
        contacts_customer=True, touches_instrument=False,
        requires_consent=True, terminal=False, recovery_fraction=1.0,
    ),
    Intervention.ESCALATE: InterventionSpec(
        Intervention.ESCALATE,
        "Hand the record to a human with a reason.",
        contacts_customer=False, touches_instrument=False,
        requires_consent=False, terminal=True, recovery_fraction=0.0,
    ),
    Intervention.GIVE_UP: InterventionSpec(
        Intervention.GIVE_UP,
        "Stop working the record deliberately; judged not worth another ask.",
        contacts_customer=False, touches_instrument=False,
        requires_consent=False, terminal=True, recovery_fraction=0.0,
    ),
}


def spec(intervention: Intervention) -> InterventionSpec:
    return INTERVENTIONS[intervention]


# --------------------------------------------------------------------------
# Class -> bounded candidate set
# --------------------------------------------------------------------------
# The candidate set is where "never retry an expired card" and "never silently
# retry a broken mandate" become structural: those interventions are simply not
# offered for those classes, so the guardrail is a second line of defence
# rather than the only one.
PRIMARY_INTERVENTION: Mapping[FailureClass, Intervention] = {
    FailureClass.INSUFFICIENT_FUNDS: Intervention.RETRY_SALARY_WINDOW,
    FailureClass.BANK_DOWNTIME: Intervention.RETRY_DELAYED,
    FailureClass.CARD_EXPIRED: Intervention.UPDATE_PAYMENT_METHOD,
    FailureClass.MANDATE_BROKEN: Intervention.RE_AUTH_MANDATE,
    FailureClass.DO_NOT_HONOUR: Intervention.RETRY_NOW,
    FailureClass.UNKNOWN: Intervention.ESCALATE,
}

CANDIDATES: Mapping[FailureClass, Tuple[Intervention, ...]] = {
    FailureClass.INSUFFICIENT_FUNDS: (
        Intervention.RETRY_SALARY_WINDOW,
        Intervention.RETRY_NOW,
        Intervention.RETRY_SMALLER_AMOUNT,
        Intervention.ESCALATE,
    ),
    FailureClass.BANK_DOWNTIME: (
        Intervention.RETRY_DELAYED,
        Intervention.RETRY_NOW,
        Intervention.ESCALATE,
    ),
    FailureClass.CARD_EXPIRED: (
        # No card retry: it is pointless AND it increments the failure counter
        # at the issuer.
        Intervention.UPDATE_PAYMENT_METHOD,
        Intervention.ESCALATE,
    ),
    FailureClass.MANDATE_BROKEN: (
        # Never a silent retry against a broken mandate: re-consent only.
        Intervention.RE_AUTH_MANDATE,
        Intervention.ESCALATE,
    ),
    FailureClass.DO_NOT_HONOUR: (
        Intervention.RETRY_NOW,
        Intervention.ESCALATE,
    ),
    FailureClass.UNKNOWN: (
        Intervention.ESCALATE,
    ),
}

# Per-class attempt caps. Class 3 and 4 are 0 because the instrument must not
# be re-debited at all -- the recovery path is a customer contact.
MAX_ATTEMPTS: Mapping[FailureClass, int] = {
    FailureClass.INSUFFICIENT_FUNDS: 3,
    FailureClass.BANK_DOWNTIME: 3,
    FailureClass.CARD_EXPIRED: 0,
    FailureClass.MANDATE_BROKEN: 0,
    FailureClass.DO_NOT_HONOUR: 1,
    FailureClass.UNKNOWN: 0,
}

ALL_CLASSES: Tuple[FailureClass, ...] = (
    FailureClass.INSUFFICIENT_FUNDS,
    FailureClass.BANK_DOWNTIME,
    FailureClass.CARD_EXPIRED,
    FailureClass.MANDATE_BROKEN,
    FailureClass.DO_NOT_HONOUR,
)
