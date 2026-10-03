"""The brain: pure classify / decide / check_guardrails + Policy.

No I/O, no clock, no network, no model.  The only LLM in the system is the
``narrate`` node, which is write-only.  Everything that *decides* is here, and
every function is a pure map from data to a structured verdict -- which is what
makes the whole control plane unit-testable without a graph.

The Phase-4 hook
----------------
``decide()`` scores every candidate through a single call site,
``_p_recover(record, intervention)``.  Replacing that one function with
``model.predict_proba(record, intervention)`` swaps the rules policy for the
learned model with no other change -- same signature, same call site, nothing
else moves.  The gap between the rules policy and the oracle is the ML story,
and it is *measured* (``scripts/measure_policy_gap.py``), never asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.taxonomy import (
    CANDIDATES,
    MAX_ATTEMPTS,
    FailureClass,
    Intervention,
    classify_reason,
    spec,
)
from recoup.economics import (
    EconomicPolicy,
    EconomicStopReason,
    net_ev_cents,
    rank_interventions,
    should_stop_economic,
)


# --------------------------------------------------------------------------
# Policy: every control knob, in one auditable place
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    economic: EconomicPolicy = field(default_factory=EconomicPolicy)

    # Compliance knobs.
    min_cooldown_hours: float = 24.0
    max_contacts_per_window: int = 3
    contact_window_hours: float = 720.0            # 30 days
    amount_auto_cap_cents: int = 500_000           # Rs 5,000: above -> human gate

    # Per-class attempt caps come from the taxonomy; mirrored here for clarity.
    max_attempts: Dict[FailureClass, int] = field(
        default_factory=lambda: dict(MAX_ATTEMPTS)
    )


DEFAULT_POLICY = Policy()


# --------------------------------------------------------------------------
# Diagnose
# --------------------------------------------------------------------------
def classify(record: Dict[str, Any]) -> FailureClass:
    """Pure lookup on the reason Razorpay handed us.  Never ML."""
    return classify_reason(record.get("error_reason", ""))


# --------------------------------------------------------------------------
# The propensity estimate -- the Phase-4 swap point
# --------------------------------------------------------------------------
# A rules-based recovery-probability heuristic. It is deliberately a decent-but-
# imperfect read of the observable features, so that the learned model has real
# room to beat it -- the measured gap is the ML story. Swap this one function
# for model.predict_proba and decide() is unchanged.
_CLASS_PRIOR: Dict[FailureClass, float] = {
    FailureClass.INSUFFICIENT_FUNDS: 0.45,
    FailureClass.BANK_DOWNTIME: 0.72,
    FailureClass.CARD_EXPIRED: 0.50,
    FailureClass.MANDATE_BROKEN: 0.45,
    FailureClass.DO_NOT_HONOUR: 0.10,
    FailureClass.UNKNOWN: 0.05,
}

_INTERVENTION_ADJ: Dict[Intervention, float] = {
    Intervention.RETRY_SALARY_WINDOW: 1.15,
    Intervention.RETRY_DELAYED: 1.10,
    Intervention.RETRY_NOW: 0.90,
    Intervention.RETRY_SMALLER_AMOUNT: 1.05,
    Intervention.UPDATE_PAYMENT_METHOD: 1.0,
    Intervention.RE_AUTH_MANDATE: 1.0,
}


# --------------------------------------------------------------------------
# Phase-4 model injection -- opt-in, never the default
# --------------------------------------------------------------------------
# The learned model is wired in by *replacing the estimate at this one call
# site*, not by editing decide().  It is a module-global rather than a
# constructor argument because control is a pure-function plane with no object to
# hang state on; configure_model / clear_model make the swap explicit and
# reversible.  Default is None: with nothing configured, the rules heuristic
# below runs and the submission never depends on the model existing.
_MODEL: Optional[Any] = None


def configure_model(model: Any) -> None:
    """Route :func:`_p_recover` through ``model.predict_proba(record, intervention)``.

    ``model`` must expose ``predict_proba(record: dict, intervention) -> float``.
    """
    global _MODEL
    _MODEL = model


def clear_model() -> None:
    """Revert to the rules heuristic.  Tests call this to stay isolated."""
    global _MODEL
    _MODEL = None


def using_model() -> bool:
    return _MODEL is not None


def _p_recover(record: Dict[str, Any], intervention: Intervention) -> float:
    """Estimate of P(recover | record, intervention), in [0, 1].

    THE PHASE-4 HOOK.  If a model has been configured it answers; otherwise the
    rules heuristic below runs.  ``decide()`` calls this and is unchanged either
    way -- the entire model swap lives in this function.
    """
    if _MODEL is not None:
        p = float(_MODEL.predict_proba(record, intervention))
        return max(0.0, min(1.0, p))

    klass = classify(record)
    p = _CLASS_PRIOR.get(klass, 0.05)
    p *= _INTERVENTION_ADJ.get(intervention, 1.0)

    # Retry fatigue: each prior retry makes the observable case look worse.
    prior = int(record.get("prior_retries", 0))
    p *= 0.82 ** prior

    # Amount pressure: a debit large relative to the customer's usual is harder.
    avg = max(int(record.get("customer_avg_payment_cents", 1)), 1)
    ratio = int(record.get("amount_cents", 0)) / avg
    if ratio > 1.5:
        p *= 0.85

    # Tenure as an OBSERVABLE PROXY for the latent engagement/regularity traits.
    # A real rules policy exploits every observable it has; tenure correlates
    # ~0.6 with engagement, so a long-tenured customer is a better bet. It is
    # only a proxy, though -- the residual latent signal is exactly the headroom
    # a learned model captures, which is why a gap survives this term.
    tenure = int(record.get("customer_tenure_days", 0))
    tenure_factor = 0.85 + 0.30 * min(tenure / 900.0, 1.0)   # 0.85 .. 1.15
    p *= tenure_factor
    return max(0.0, min(1.0, p))


# --------------------------------------------------------------------------
# Decide
# --------------------------------------------------------------------------
@dataclass
class Decision:
    failure_class: FailureClass
    chosen: Intervention
    net_ev_cents: int
    p_recover: float
    ranked: List[Dict[str, Any]]
    stop: bool
    stop_reason: str
    rationale: str


def decide(
    record: Dict[str, Any],
    *,
    prior_contacts: int = 0,
    policy: Policy = DEFAULT_POLICY,
) -> Decision:
    """Score every candidate by net EV and pick the argmax -- or abandon.

    Economic question only: *is this worth doing?*  Whether it is *allowed* is a
    separate predicate (:func:`check_guardrails`).
    """
    klass = classify(record)
    amount = int(record.get("amount_cents", 0))

    if not CANDIDATES.get(klass) or klass is FailureClass.UNKNOWN:
        return Decision(
            failure_class=klass,
            chosen=Intervention.ESCALATE,
            net_ev_cents=0,
            p_recover=0.0,
            ranked=[],
            stop=True,
            stop_reason=EconomicStopReason.NOTHING_TO_TRY.value,
            rationale=f"{klass.value}: no automated intervention; escalating.",
        )

    p_by_intervention = {
        intervention: _p_recover(record, intervention)
        for intervention in CANDIDATES[klass]
        if not spec(intervention).terminal
    }
    ranked = rank_interventions(
        klass, amount, p_by_intervention,
        prior_contacts=prior_contacts, policy=policy.economic,
    )
    stop, reason, best = should_stop_economic(ranked, policy.economic)

    if stop or best is None:
        return Decision(
            failure_class=klass,
            chosen=Intervention.GIVE_UP,
            net_ev_cents=best.net_ev_cents if best else 0,
            p_recover=best.p_recover if best else 0.0,
            ranked=[_ranked_row(c) for c in ranked],
            stop=True,
            stop_reason=reason.value,
            rationale=(
                f"{klass.value}: best net EV "
                f"{(best.net_ev_cents if best else 0)} cents does not beat the "
                f"stop threshold; giving up."
            ),
        )

    return Decision(
        failure_class=klass,
        chosen=best.intervention,
        net_ev_cents=best.net_ev_cents,
        p_recover=best.p_recover,
        ranked=[_ranked_row(c) for c in ranked],
        stop=False,
        stop_reason=reason.value,
        rationale=(
            f"{klass.value}: {best.intervention.value} has the highest net EV "
            f"({best.net_ev_cents} cents, p={best.p_recover:.2f})."
        ),
    )


def _ranked_row(c) -> Dict[str, Any]:
    return {
        "intervention": c.intervention.value,
        "p_recover": round(c.p_recover, 4),
        "net_ev_cents": c.net_ev_cents,
        "is_contact": c.is_contact,
    }


# --------------------------------------------------------------------------
# Guardrails -- the compliance predicate, separate from economics
# --------------------------------------------------------------------------
@dataclass
class GuardrailResult:
    allowed: bool
    checks: Dict[str, str]
    reason: str

    @property
    def blocked(self) -> bool:
        return not self.allowed


def check_guardrails(
    record: Dict[str, Any],
    intervention: Intervention,
    *,
    attempt: int,
    hours_since_last: Optional[float],
    prior_contacts: int,
    policy: Policy = DEFAULT_POLICY,
) -> GuardrailResult:
    """Is this action *allowed*?  Distinct from whether it is *worth it*.

    Each check returns a structured pass/fail so the audit row can say exactly
    which rule blocked, and a block is logged differently from a give-up.
    """
    checks: Dict[str, str] = {}
    action = spec(intervention)
    klass = classify(record)

    # 1. Per-class attempt cap. Card/mandate classes are 0: never re-debit.
    cap = policy.max_attempts.get(klass, 0)
    if action.touches_instrument and attempt > cap:
        checks["attempt_cap"] = "fail"
        return _blocked(checks, f"attempt {attempt} exceeds cap {cap} for {klass.value}")
    checks["attempt_cap"] = "pass"

    # 2. Cooldown between retries.
    if action.touches_instrument and hours_since_last is not None:
        if hours_since_last < policy.min_cooldown_hours:
            checks["cooldown"] = "fail"
            return _blocked(
                checks,
                f"cooldown not elapsed: {hours_since_last:.1f}h < "
                f"{policy.min_cooldown_hours}h",
            )
    checks["cooldown"] = "pass"

    # 3. Contact cap (customer-facing actions only).
    if action.contacts_customer and prior_contacts >= policy.max_contacts_per_window:
        checks["contact_cap"] = "fail"
        return _blocked(
            checks,
            f"contact cap reached: {prior_contacts}/{policy.max_contacts_per_window}",
        )
    checks["contact_cap"] = "pass"

    # 4. Amount gate: a large debit needs a human, not an autonomous retry.
    if action.touches_instrument and int(record.get("amount_cents", 0)) > policy.amount_auto_cap_cents:
        checks["amount_gate"] = "fail"
        return _blocked(
            checks,
            f"amount {record.get('amount_cents')} over auto-cap "
            f"{policy.amount_auto_cap_cents}: needs human sign-off",
        )
    checks["amount_gate"] = "pass"

    # 5. Mandate rules: never a silent retry against a broken/cancelled mandate.
    status = record.get("subscription_status", "none")
    if action.touches_instrument and record.get("is_mandate_debit") and status in (
        "cancelled", "halted", "paused",
    ):
        checks["mandate_active"] = "fail"
        return _blocked(
            checks,
            f"mandate is {status}: a debit needs re-consent, not a retry",
        )
    checks["mandate_active"] = "pass"

    # 6. E-mandate pre-debit notification (RBI): a debit requires notice.
    if action.touches_instrument and record.get("is_mandate_debit") and not record.get(
        "pre_debit_notified", False
    ):
        checks["pre_debit_notice"] = "fail"
        return _blocked(
            checks,
            "e-mandate debit without the required pre-debit notification",
        )
    checks["pre_debit_notice"] = "pass"

    return GuardrailResult(allowed=True, checks=checks, reason="all guardrails passed")


def _blocked(checks: Dict[str, str], reason: str) -> GuardrailResult:
    return GuardrailResult(allowed=False, checks=checks, reason=reason)
