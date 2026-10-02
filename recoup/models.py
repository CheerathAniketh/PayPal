"""Typed records, with observable data and hidden ground truth kept apart.

The central structural decision of the project lives here: ``FailedRecord``
holds everything the agent and the propensity model may see, and ``LatentTruth``
holds the parameters the environment uses to decide outcomes.  They are separate
objects rather than separate naming conventions, because that makes **leakage a
type error instead of a code-review catch**.  You cannot accidentally pass the
answer key into a feature dict; the object is simply not in scope.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from config.taxonomy import FailureClass, Intervention, PaymentMethod


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------
class Outcome(str, Enum):
    """What happened to one attempt.

    This was originally a bare ``str`` with the valid values listed only in a
    comment.  Every headline metric -- recovery rate, escalation count,
    false-effort avoided -- is a GROUP BY over this column, so a typo like
    ``"recoverd"`` would not raise.  It would silently produce a wrong number
    in the final report, discovered at the worst possible moment.
    """

    RECOVERED = "recovered"   # money came back
    FAILED = "failed"         # attempt ran, did not recover
    ESCALATED = "escalated"   # guardrail tripped -> handed to a human
    GAVE_UP = "gave_up"       # judged dead, stopped deliberately
    SCHEDULED = "scheduled"   # queued for a later window
    BLOCKED = "blocked"       # guardrail refused BEFORE execution


# BLOCKED is deliberately distinct from ESCALATED: "refused before it ran" and
# "tried, then handed over" are different facts, and the false-effort-avoided
# metric needs to tell them apart.


class ExecutionMode(str, Enum):
    SIMULATED = "simulated"
    TEST_MODE = "test_mode"


# --------------------------------------------------------------------------
# Observable side of the wall
# --------------------------------------------------------------------------
@dataclass
class Customer:
    """A customer.

    ``engagement`` and ``income_regularity`` are **latent traits**: they live on
    this object because they are properties of the person, but they are never
    returned by :meth:`FailedRecord.to_features` and never written to the
    observable batch file.  They persist across a customer's records, which is
    exactly what makes ``GroupKFold(customer_id)`` load-bearing.
    """

    customer_id: str
    tenure_days: int
    avg_payment_paise: int
    salary_day: int                 # observable: billing/credit anchor day
    is_subscriber: bool
    prior_failures: int

    # ---- latent, never a feature -------------------------------------
    engagement: float = 0.5
    income_regularity: float = 0.5

    def observable(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "tenure_days": self.tenure_days,
            "avg_payment_paise": self.avg_payment_paise,
            "salary_day": self.salary_day,
            "is_subscriber": self.is_subscriber,
            "prior_failures": self.prior_failures,
        }


@dataclass
class FailedRecord:
    """One failed payment: everything the agent may see, and nothing else."""

    record_id: str
    customer_id: str
    amount_paise: int
    method: PaymentMethod
    error_reason: str
    error_source: str
    failed_at: datetime
    prior_retries: int
    is_mandate_debit: bool
    pre_debit_notified: bool
    subscription_status: str        # active | paused | halted | cancelled | none
    customer_tenure_days: int
    customer_avg_payment_paise: int
    customer_salary_day: int
    customer_prior_failures: int
    customer_is_subscriber: bool

    def to_features(self) -> Dict[str, Any]:
        """The model's entire view of the world.

        Deliberately hand-written rather than ``asdict``: a new field on this
        dataclass should be an explicit decision to expose it, not an accident.
        """
        return {
            "amount_paise": self.amount_paise,
            "method": self.method.value,
            "error_reason": self.error_reason,
            "error_source": self.error_source,
            "prior_retries": self.prior_retries,
            "is_mandate_debit": int(self.is_mandate_debit),
            "pre_debit_notified": int(self.pre_debit_notified),
            "subscription_status": self.subscription_status,
            "customer_tenure_days": self.customer_tenure_days,
            "customer_avg_payment_paise": self.customer_avg_payment_paise,
            "customer_salary_day": self.customer_salary_day,
            "customer_prior_failures": self.customer_prior_failures,
            "customer_is_subscriber": int(self.customer_is_subscriber),
            "amount_ratio": (
                self.amount_paise / max(self.customer_avg_payment_paise, 1)
            ),
        }

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["method"] = self.method.value
        payload["failed_at"] = self.failed_at.isoformat()
        return payload

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "FailedRecord":
        data = dict(payload)
        data["method"] = PaymentMethod(data["method"])
        data["failed_at"] = datetime.fromisoformat(data["failed_at"])
        return cls(**data)


# --------------------------------------------------------------------------
# Hidden side of the wall
# --------------------------------------------------------------------------
@dataclass
class LatentTruth:
    """The answer key for one record.  Never a feature, never in the store.

    Keyed by ``record_id`` and stored *beside* the record, in its own file, so
    that the separation is visible on the filesystem and not just in the type
    system.
    """

    record_id: str
    base_logodds: float             # irreducible per-record noise
    is_truly_dead: bool             # recovery probability ~0 whatever you try
    seeded_class: FailureClass      # the generator's label, for self-consistency
    engagement: float               # copy of the customer trait, for convenience
    income_regularity: float

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["seeded_class"] = self.seeded_class.value
        return payload

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "LatentTruth":
        data = dict(payload)
        data["seeded_class"] = FailureClass(data["seeded_class"])
        return cls(**data)


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------
@dataclass
class RunRecord:
    """One agent run.  Without this, running the batch twice interleaves both
    runs in the audit log with no way to separate them, and every metric --
    each a GROUP BY over that table -- silently double-counts."""

    run_id: str
    seed: int
    mode: ExecutionMode
    batch_size: int
    config_json: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    notes: str = ""


@dataclass
class AuditEntry:
    """One receipt line.  Append-only; see :mod:`recoup.db`."""

    run_id: str
    record_id: str
    timestamp: datetime
    attempt_number: int
    chosen_action: Intervention      # typed: only real interventions are loggable
    rationale: str
    guardrail_checks: Dict[str, Any]
    model_score: Optional[float]
    outcome: Outcome
    amount_recovered_paise: int
    idempotency_key: str
    execution_mode: ExecutionMode
    api_called: bool
    razorpay_entity_id: Optional[str] = None
    was_mocked: bool = False
    mock_reason: str = ""
    api_error: str = ""
    customer_id: str = ""

    def to_row(self) -> Dict[str, Any]:
        """Flatten for SQLite.  Enums become their string values so that metric
        queries are plain SQL and do not break silently on a repr change."""
        return {
            "run_id": self.run_id,
            "record_id": self.record_id,
            "customer_id": self.customer_id,
            "timestamp": self.timestamp.isoformat(),
            "attempt_number": self.attempt_number,
            "chosen_action": self.chosen_action.value,
            "rationale": self.rationale,
            "guardrail_checks": json.dumps(self.guardrail_checks, default=str),
            "model_score": self.model_score,
            "outcome": self.outcome.value,
            "amount_recovered_paise": self.amount_recovered_paise,
            "idempotency_key": self.idempotency_key,
            "execution_mode": self.execution_mode.value,
            "api_called": int(self.api_called),
            "razorpay_entity_id": self.razorpay_entity_id,
            "was_mocked": int(self.was_mocked),
            "mock_reason": self.mock_reason,
            "api_error": self.api_error,
        }


@dataclass
class BatchStats:
    """Summary of a frozen batch, for the ingest report."""

    n_records: int
    n_customers: int
    n_recurring_customers: int
    total_at_risk_paise: int
    class_histogram: Dict[str, int] = field(default_factory=dict)
    n_truly_dead: int = 0
