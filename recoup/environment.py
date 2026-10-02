"""The hidden recovery world.

Given a record, a customer, its hidden truth, an intervention and *when* the
agent acts, this module samples whether the money came back.  It is what the
propensity model will later try to approximate, which makes it the
highest-stakes file in the project.

Why it cannot be a lookup table
-------------------------------
The batch is synthetic.  If the environment were a lookup keyed on observable
features, the model would just re-learn the generator, evaluation would look
perfect, and it would prove nothing.  The entire ML story depends on this being
a *fair problem*: a customer must carry persistent hidden information that a
model can only partly infer.

The failure this file originally had
------------------------------------
The first draft generated ``Customer.engagement`` and
``Customer.income_regularity`` and then never used them -- the environment never
even received a ``Customer``.  Instead it *reconstructed* engagement as a
deterministic function of observable features.  That sounds harmless.  It is
not: if engagement is fully determined by observables, two records from the same
customer are conditionally independent given the features, nothing hidden
remains to memorise, and ``GroupKFold(customer_id)`` does exactly nothing.  The
grouped split would have been methodological theatre -- the right thing to write
in a README, doing zero actual work.

Calibration
-----------
The coefficients are not invented.  Aggregate behaviour is checked against
published dunning benchmarks (see ``scripts/calibrate_environment.py``):
soft declines recover at 40-70%; hard declines cap out at 20-30%; card-updater
flows recover roughly 70% of expired cards.
"""

from __future__ import annotations

import calendar
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Tuple

from config.taxonomy import (
    CANDIDATES,
    FailureClass,
    Intervention,
    classify_reason,
    spec,
)
from recoup.models import Customer, FailedRecord, LatentTruth

# A record seeded as truly dead recovers essentially never, whatever you try.
DEAD_PROB = 0.0003


@dataclass(frozen=True)
class EnvironmentConfig:
    """Every coefficient in the world, in one auditable place."""

    # Per-class baseline log-odds, calibrated to published recovery benchmarks.
    class_base: Dict[FailureClass, float] = field(
        default_factory=lambda: {
            FailureClass.INSUFFICIENT_FUNDS: 0.91,
            FailureClass.BANK_DOWNTIME: 2.79,
            FailureClass.CARD_EXPIRED: 1.00,
            FailureClass.MANDATE_BROKEN: 0.78,
            FailureClass.DO_NOT_HONOUR: -1.91,
            FailureClass.UNKNOWN: -1.50,
        }
    )

    # Intervention fit. Only where the intervention changes the *mechanism*,
    # never as a second copy of an effect already modelled below.
    fit: Dict[Tuple[FailureClass, Intervention], float] = field(
        default_factory=lambda: {
            # Waiting is the actual fix for a transient outage.
            (FailureClass.BANK_DOWNTIME, Intervention.RETRY_DELAYED): 0.80,
            # Hammering during the outage is close to pointless.
            (FailureClass.BANK_DOWNTIME, Intervention.RETRY_NOW): -0.60,
        }
    )
    off_menu_penalty: float = -2.50   # an intervention this class never offers

    # Liquidity: money lands on salary day and depletes afterwards.
    salary_floor: float = 0.60        # see note in `_liquidity_term`
    salary_gain: float = 1.80
    ramp_days: float = 2.5            # smooth rise approaching salary day
    decay_days: float = 6.0           # decay after it

    amount_pressure: float = 0.90     # a big debit relative to the usual is harder
    smaller_amount_penalty: float = 0.35   # variable-amount debit friction
    retry_fatigue: float = 0.60       # per prior retry
    delay_decay: float = 0.05         # per day a failed payment sits unresolved
    engagement_gain: float = 2.60     # drives customer-contact interventions

    jitter_scale: float = 0.35        # per-intervention aleatoric wobble


DEFAULT_CONFIG = EnvironmentConfig()


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def liquidity(attempt_at: datetime, salary_day: int) -> float:
    """How much money is plausibly in the account, in [0, 1].

    Originally a hard step (day 27 -> 0.0, day 28 -> 1.0).  That is wrong twice
    over: real liquidity ramps rather than snapping, and a cliff is a trivially
    memorisable feature that would let a model ace the problem for the wrong
    reason.  This is a smooth two-sided exponential -- a short ramp approaching
    the salary date, a longer decay after it.
    """
    month_len = calendar.monthrange(attempt_at.year, attempt_at.month)[1]
    anchor = min(max(salary_day, 1), month_len)
    day = attempt_at.day + attempt_at.hour / 24.0
    delta = day - anchor
    # Wrap to the nearest occurrence: day 1 is "just after" a salary on day 28.
    if delta < -month_len / 2.0:
        delta += month_len
    elif delta > month_len / 2.0:
        delta -= month_len
    if delta <= 0.0:
        return math.exp(delta / DEFAULT_CONFIG.ramp_days)
    return math.exp(-delta / DEFAULT_CONFIG.decay_days)


def _jitter(record_id: str, intervention: Intervention, scale: float) -> float:
    """Per-(record, intervention) wobble.

    Correlated within a record via ``LatentTruth.base_logodds`` and jittered per
    intervention here, so that a perfect model still tops out below AUC 1.0 --
    genuine aleatoric uncertainty, as in real dunning logs.
    """
    digest = hashlib.sha256(
        f"jit|{record_id}|{intervention.value}".encode("utf-8")
    ).digest()
    # Two bytes -> a roughly standard-normal draw via the Irwin-Hall trick.
    u = int.from_bytes(digest[:4], "big") / 2**32
    v = int.from_bytes(digest[4:8], "big") / 2**32
    u = min(max(u, 1e-9), 1 - 1e-9)
    normal = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
    return scale * normal


class RecoveryEnvironment:
    """The hidden world.  Only the generator and the evaluator may consult it."""

    def __init__(self, config: EnvironmentConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # FOR LABELING / EVAL ONLY
    # ------------------------------------------------------------------
    def true_prob(
        self,
        record: FailedRecord,
        customer: Customer,
        latent: LatentTruth,
        intervention: Intervention,
        attempt_at: datetime,
    ) -> float:
        """The ground-truth recovery probability.

        FOR LABELING / EVAL ONLY.  Kept as a separate function from
        :meth:`sample_outcome` so that misuse -- an agent peeking at the answer
        -- is visible in review rather than buried inside a sampler.
        """
        cfg = self.config
        action = spec(intervention)
        if action.terminal:
            return 0.0
        if latent.is_truly_dead:
            return DEAD_PROB

        klass = classify_reason(record.error_reason)
        z = cfg.class_base.get(klass, cfg.class_base[FailureClass.UNKNOWN])

        if intervention not in CANDIDATES.get(klass, ()):
            z += cfg.off_menu_penalty
        z += cfg.fit.get((klass, intervention), 0.0)

        if action.touches_instrument:
            z += self._liquidity_term(customer, latent, attempt_at)
            ratio = record.amount_paise / max(customer.avg_payment_paise, 1)
            # A partial debit asks for less money, so it feels less pressure.
            z -= cfg.amount_pressure * ratio * action.recovery_fraction
            if intervention is Intervention.RETRY_SMALLER_AMOUNT:
                # Debiting an amount other than the registered one carries real
                # friction under a fixed mandate.
                z -= cfg.smaller_amount_penalty
        else:
            # A customer-facing ask: whether they act is about engagement.
            z += cfg.engagement_gain * (latent.engagement - 0.5)

        z -= cfg.retry_fatigue * record.prior_retries

        # The longer a failed payment sits, the less recoverable it becomes.
        # This is what makes waiting for a salary window a genuine trade-off
        # rather than a free option.
        days_waited = max(
            0.0, (attempt_at - record.failed_at).total_seconds() / 86400.0
        )
        z -= cfg.delay_decay * days_waited

        z += latent.base_logodds
        z += _jitter(record.record_id, intervention, cfg.jitter_scale)
        return _sigmoid(z)

    def _liquidity_term(
        self, customer: Customer, latent: LatentTruth, attempt_at: datetime
    ) -> float:
        """``0.6 + 1.8 * income_regularity * liquidity``.

        The floor matters more than it looks.  A pure multiplication would say
        an irregular-income customer is *insensitive to timing*.  That is wrong:
        their liquidity is unpredictable, not absent.  They still get paid --
        you just cannot guess when.

        The consequence is the whole reason a model is needed here.  Because
        this trait changes *which* intervention wins rather than merely scaling
        the reward, the agent must learn a per-customer policy instead of
        re-ordering one queue.
        """
        cfg = self.config
        liq = liquidity(attempt_at, customer.salary_day)
        return cfg.salary_floor + cfg.salary_gain * latent.income_regularity * liq

    # ------------------------------------------------------------------
    # The sampler the executor actually calls
    # ------------------------------------------------------------------
    def sample_outcome(
        self,
        record: FailedRecord,
        customer: Customer,
        latent: LatentTruth,
        intervention: Intervention,
        attempt_at: datetime,
        attempt_number: int = 1,
    ) -> bool:
        """Did the money come back?  Deterministic given the same inputs."""
        p = self.true_prob(record, customer, latent, intervention, attempt_at)
        draw = _uniform(record.record_id, intervention, attempt_number)
        return draw < p

    def best_intervention(
        self,
        record: FailedRecord,
        customer: Customer,
        latent: LatentTruth,
        attempt_at: datetime,
    ) -> Tuple[Intervention, float]:
        """Oracle argmax over the class's candidate set, by probability."""
        klass = classify_reason(record.error_reason)
        options = [
            i for i in CANDIDATES.get(klass, ()) if not spec(i).terminal
        ]
        if not options:
            return Intervention.ESCALATE, 0.0
        scored = [
            (i, self.true_prob(record, customer, latent, i, attempt_at))
            for i in options
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[0]


def _uniform(record_id: str, intervention: Intervention, attempt_number: int) -> float:
    digest = hashlib.sha256(
        f"draw|{record_id}|{intervention.value}|{attempt_number}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
