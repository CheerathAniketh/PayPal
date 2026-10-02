"""The execution layer.

The central design decision
--------------------------
    **Razorpay answers "did the API call succeed?"
    The environment answers "did we recover the money?"**

Test mode cannot know whether a customer's salary landed, whether they clicked
an update-card link, or whether their bank's outage cleared.  Those are
properties of the simulated world.

So real test-mode calls prove the agent can drive a payments API and produce
real entity ids for the audit trail -- but **reporting API success as revenue
recovered would be a fabricated number**, precisely what the honest-metrics bar
screens for.  This must be said out loud in the pitch, not buried here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from config.taxonomy import Intervention, spec
from recoup.environment import RecoveryEnvironment
from recoup.idempotency import attempt_key, receipt_id
from recoup.models import (
    Customer,
    ExecutionMode,
    FailedRecord,
    LatentTruth,
    Outcome,
)
from recoup.money import split_paise


@dataclass(frozen=True)
class TestModeSupport:
    """Whether an intervention has an honest real-API equivalent in test mode.

    Every intervention must declare one.  A test asserts the mapping is total,
    so no intervention can ever run without an explicit real-or-mock decision.
    """

    supported: bool
    entity: str = ""
    mock_reason: str = ""


TEST_MODE_SUPPORT: Mapping[Intervention, TestModeSupport] = {
    Intervention.RETRY_NOW: TestModeSupport(True, "order"),
    Intervention.RETRY_DELAYED: TestModeSupport(True, "order"),
    Intervention.RETRY_SALARY_WINDOW: TestModeSupport(True, "order"),
    Intervention.RETRY_SMALLER_AMOUNT: TestModeSupport(True, "order"),
    Intervention.UPDATE_PAYMENT_METHOD: TestModeSupport(True, "payment_link"),
    Intervention.RE_AUTH_MANDATE: TestModeSupport(
        False,
        mock_reason=(
            "mandate re-consent requires a real customer authentication "
            "journey; no honest test-mode equivalent exists"
        ),
    ),
    Intervention.ESCALATE: TestModeSupport(
        False, mock_reason="internal state; no payments API involved"
    ),
    Intervention.GIVE_UP: TestModeSupport(
        False, mock_reason="internal state; no payments API involved"
    ),
}


@dataclass
class ExecutionResult:
    intervention: Intervention
    outcome: Outcome
    amount_attempted_paise: int
    amount_recovered_paise: int
    idempotency_key: str
    receipt: str
    execution_mode: ExecutionMode
    attempted_at: datetime
    api_called: bool = False
    razorpay_entity_id: Optional[str] = None
    was_mocked: bool = False
    mock_reason: str = ""
    api_error: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


class TestModeRequiresClient(RuntimeError):
    """Raised when TEST_MODE is requested without a Razorpay client.

    Silently downgrading to SIMULATED would mean the audit trail claims real
    calls that never happened.
    """


class Executor:
    """Runs one intervention for one record.

    ``run_id`` and ``attempt_number`` are REQUIRED, never defaulted.  Defaulting
    them would let two different attempts derive the same idempotency key, which
    is exactly the condition that causes a double charge.
    """

    def __init__(
        self,
        environment: RecoveryEnvironment,
        mode: ExecutionMode = ExecutionMode.SIMULATED,
        client: Any = None,
    ) -> None:
        if mode is ExecutionMode.TEST_MODE and client is None:
            raise TestModeRequiresClient(
                "TEST_MODE needs a Razorpay client; refusing to downgrade "
                "silently to SIMULATED"
            )
        self.environment = environment
        self.mode = mode
        self.client = client

    # ------------------------------------------------------------------
    def execute(
        self,
        record: FailedRecord,
        customer: Customer,
        latent: LatentTruth,
        intervention: Intervention,
        *,
        run_id: str,
        attempt_number: int,
        attempt_at: datetime,
        outstanding_paise: Optional[int] = None,
    ) -> ExecutionResult:
        action = spec(intervention)
        key = attempt_key(run_id, record.record_id, attempt_number)
        receipt = receipt_id(run_id, record.record_id, attempt_number)
        outstanding = (
            record.amount_paise if outstanding_paise is None else outstanding_paise
        )

        # ---- terminal bookkeeping states -----------------------------
        if action.terminal:
            return ExecutionResult(
                intervention=intervention,
                outcome=(
                    Outcome.GAVE_UP
                    if intervention is Intervention.GIVE_UP
                    else Outcome.ESCALATED
                ),
                amount_attempted_paise=0,
                amount_recovered_paise=0,
                idempotency_key=key,
                receipt=receipt,
                execution_mode=self.mode,
                attempted_at=attempt_at,
                api_called=False,
                was_mocked=False,
                mock_reason=TEST_MODE_SUPPORT[intervention].mock_reason,
            )

        # Never ask for more than is still owed. Without this cap a partial
        # debit followed by a full retry over-collects.
        amount_attempted = min(
            outstanding, split_paise(outstanding, action.recovery_fraction)
        ) if action.recovery_fraction < 1.0 else outstanding

        result = ExecutionResult(
            intervention=intervention,
            outcome=Outcome.FAILED,
            amount_attempted_paise=amount_attempted,
            amount_recovered_paise=0,
            idempotency_key=key,
            receipt=receipt,
            execution_mode=self.mode,
            attempted_at=attempt_at,
        )

        # ---- the API side: did the CALL succeed? ---------------------
        if self.mode is ExecutionMode.TEST_MODE:
            self._call_razorpay(record, intervention, amount_attempted, receipt, result)
        else:
            support = TEST_MODE_SUPPORT[intervention]
            result.was_mocked = not support.supported
            result.mock_reason = support.mock_reason

        # ---- the world side: did the MONEY come back? ----------------
        # An API error is recorded, but the environment still decides. A flaky
        # network must not be able to quietly deflate the recovery metrics.
        recovered = self.environment.sample_outcome(
            record, customer, latent, intervention, attempt_at, attempt_number
        )
        if recovered:
            result.amount_recovered_paise = amount_attempted
            result.outcome = Outcome.RECOVERED
        else:
            result.amount_recovered_paise = 0
            result.outcome = Outcome.FAILED
        return result

    # ------------------------------------------------------------------
    def _call_razorpay(
        self,
        record: FailedRecord,
        intervention: Intervention,
        amount_paise: int,
        receipt: str,
        result: ExecutionResult,
    ) -> None:
        support = TEST_MODE_SUPPORT[intervention]
        if not support.supported:
            # Flagged, not hidden: the limitation travels into the audit row.
            result.was_mocked = True
            result.mock_reason = support.mock_reason
            result.api_called = False
            return

        try:
            if support.entity == "order":
                entity = self.client.order.create(
                    {
                        "amount": amount_paise,
                        "currency": "INR",
                        "receipt": receipt,
                        "payment_capture": 1,
                        "notes": {
                            "recoup_record_id": record.record_id,
                            "recoup_intervention": intervention.value,
                        },
                    }
                )
            elif support.entity == "payment_link":
                entity = self.client.payment_link.create(
                    {
                        "amount": amount_paise,
                        "currency": "INR",
                        "description": "Update your payment method to clear a "
                        "failed subscription charge",
                        "reference_id": receipt,
                        "notes": {
                            "recoup_record_id": record.record_id,
                            "recoup_intervention": intervention.value,
                        },
                    }
                )
            else:  # pragma: no cover - guarded by the totality test
                raise RuntimeError(f"unknown entity {support.entity!r}")
            result.api_called = True
            result.razorpay_entity_id = entity.get("id")
        except Exception as exc:  # noqa: BLE001 - the error is data, not control
            result.api_called = True
            result.api_error = f"{type(exc).__name__}: {exc}"
