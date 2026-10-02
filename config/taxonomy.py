"""The failure taxonomy: the contract every other module reads.

Five failure classes, a deterministic reason-code lookup, and the bounded set of
interventions the agent may choose from.

Why classification is a lookup and not a model
----------------------------------------------
Razorpay hands you the reason code in its error response; its own documentation
says the ``reason`` field can be handled programmatically.  Putting a classifier
on top of a lookup table is ML-as-decoration.  The ML in this project is
reserved for the genuinely uncertain question -- *will this recover?* -- not the
solved one -- *what broke?*

Reason strings below were verified against Razorpay's published error tables
(List of Errors: Bad Request + Gateway Errors, plus the Cards and UPI method
pages).  Notably ``do_not_honour`` and ``gateway_error`` are NOT Razorpay
strings -- they are ISO-8583 / colloquial terminology.
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
    NETBANKING = "netbanking"
    UPI = "upi"
    EMANDATE = "emandate"


# --------------------------------------------------------------------------
# Reason-code lookup
# --------------------------------------------------------------------------
# Verified against Razorpay's live error tables. Every string here is one
# Razorpay actually returns; nothing is written from memory.
REASON_MAP: Mapping[FailureClass, Tuple[str, ...]] = {
    FailureClass.INSUFFICIENT_FUNDS: (
        "insufficient_funds",
    ),
    FailureClass.BANK_DOWNTIME: (
        "bank_technical_error",
        "gateway_technical_error",
        "bank_not_available",
        "bank_cutoff_in_progress",
        "server_error",
    ),
    FailureClass.CARD_EXPIRED: (
        "card_expired",
    ),
    FailureClass.MANDATE_BROKEN: (
        "mandate_creation_declined",
        "mandate_creation_failed",
        "mandate_creation_expired",
        "mandate_creation_timeout",
        "reqauth_mandate_not_acknowledged",
    ),
    FailureClass.DO_NOT_HONOUR: (
        # 'do_not_honour' is ISO-8583 terminology and is NEVER returned by
        # Razorpay. Hard declines surface as these three strings.
        "card_declined",
        "payment_declined",
        "payment_risk_check_failed",
    ),
}

_REASON_INDEX: Dict[str, FailureClass] = {
    reason: klass for klass, reasons in REASON_MAP.items() for reason in reasons
}

# The `source` vocabulary is per payment method, not global.
SOURCE_VOCABULARY: Mapping[PaymentMethod, Tuple[str, ...]] = {
    PaymentMethod.CARD: ("issuer_bank", "gateway", "internal"),
    PaymentMethod.NETBANKING: ("issuer_bank", "gateway", "internal"),
    PaymentMethod.EMANDATE: ("issuer_bank", "bank", "gateway", "internal"),
    PaymentMethod.UPI: (
        "customer_psp",
        "network",
        "beneficiary_bank",
        "gateway",
        "internal",
    ),
}


def classify_reason(reason: str) -> FailureClass:
    """Map a Razorpay reason code to a failure class.

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
