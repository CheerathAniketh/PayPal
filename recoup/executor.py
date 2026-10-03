"""The execution layer.

The central design decision
--------------------------
    **PayPal answers "did the API call succeed, and with what error?"
    The environment answers "did we recover the money?"**

The PayPal sandbox cannot know whether a customer's salary landed, whether they
approved an order, or whether their bank's outage cleared.  Those are properties
of the simulated world.

What the sandbox *can* do, and what this module uses it for:

* create real orders, so the audit trail carries real PayPal order ids;
* return real PayPal error bodies on capture, forced with the sandbox-only
  ``PayPal-Mock-Response`` header (see ``SANDBOX_MOCKABLE``).

The failures are **injected**, not organic: an order nobody approved can never
be captured, so a retry the environment says failed is shown with a genuine
PayPal error body rather than a made-up one.  Reporting API success as revenue
recovered would be a fabricated number, so ``recovered`` is always decided by
the environment, never by the API.  This must be said out loud in the pitch.
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
from recoup.money import split_cents
from recoup.paypal.orders import PayPalApiError

CURRENCY = "USD"

# Reasons the sandbox accepted as PayPal-Mock-Response codes when probed with
# scripts/probe_mock_errors.py.  insufficient_funds and card_expired were
# rejected (HTTP 403) and payer_cannot_pay returned 503, so none has an entry.
SANDBOX_MOCKABLE = frozenset(
    {
        "instrument_declined",
        "transaction_refused",
        "payer_action_required",
        "order_not_approved",
        "transaction_limit_exceeded",
        "internal_server_error",
    }
)
# PayPal's generic "the payment method was declined" -- used when the record's
# own reason cannot be injected.  The substitution is recorded in the audit row.
DEFAULT_RETRY_MOCK = "INSTRUMENT_DECLINED"


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
    # An order whose approve link is the "update your payment method" link.
    Intervention.UPDATE_PAYMENT_METHOD: TestModeSupport(True, "order_link"),
    Intervention.RE_AUTH_MANDATE: TestModeSupport(
        False,
        mock_reason=(
            "re-approval means the buyer completing PayPal's payer-action "
            "flow in a browser; the sandbox cannot do that unattended"
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
    amount_attempted_cents: int
    amount_recovered_cents: int
    idempotency_key: str
    receipt: str
    execution_mode: ExecutionMode
    attempted_at: datetime
    api_called: bool = False
    paypal_entity_id: Optional[str] = None
    was_mocked: bool = False
    mock_reason: str = ""
    api_error: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


class TestModeRequiresClient(RuntimeError):
    """Raised when TEST_MODE is requested without a PayPal client.

    Silently downgrading to SIMULATED would mean the audit trail claims real
    calls that never happened.
    """


def _summarise_error(exc: PayPalApiError) -> Dict[str, Any]:
    """The audit-worthy parts of a PayPal error body."""
    body = exc.body if isinstance(exc.body, dict) else {}
    details = body.get("details") or [{}]
    return {
        "http_status": exc.status,
        "name": body.get("name"),
        "issue": details[0].get("issue"),
        "debug_id": body.get("debug_id"),
    }


class Executor:
    """Runs one intervention for one record.

    ``run_id`` and ``attempt_number`` are REQUIRED, never defaulted.  Defaulting
    them would let two different attempts derive the same idempotency key, which
    is exactly the condition that causes a double charge.

    ``client`` is a ``recoup.paypal.orders.PayPalOrders`` (or anything with the
    same ``create_order`` / ``capture_order`` methods).
    """

    def __init__(
        self,
        environment: RecoveryEnvironment,
        mode: ExecutionMode = ExecutionMode.SIMULATED,
        client: Any = None,
    ) -> None:
        if mode is ExecutionMode.TEST_MODE and client is None:
            raise TestModeRequiresClient(
                "TEST_MODE needs a PayPal client; refusing to downgrade "
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
        outstanding_cents: Optional[int] = None,
    ) -> ExecutionResult:
        action = spec(intervention)
        key = attempt_key(run_id, record.record_id, attempt_number)
        receipt = receipt_id(run_id, record.record_id, attempt_number)
        outstanding = (
            record.amount_cents if outstanding_cents is None else outstanding_cents
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
                amount_attempted_cents=0,
                amount_recovered_cents=0,
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
            outstanding, split_cents(outstanding, action.recovery_fraction)
        ) if action.recovery_fraction < 1.0 else outstanding

        result = ExecutionResult(
            intervention=intervention,
            outcome=Outcome.FAILED,
            amount_attempted_cents=amount_attempted,
            amount_recovered_cents=0,
            idempotency_key=key,
            receipt=receipt,
            execution_mode=self.mode,
            attempted_at=attempt_at,
        )

        # ---- the world side first: did the MONEY come back? ----------
        # Decided before the API call so a failure injected into the sandbox
        # always agrees with the outcome the environment chose.
        recovered = self.environment.sample_outcome(
            record, customer, latent, intervention, attempt_at, attempt_number
        )

        # ---- the API side: did the CALL succeed? ---------------------
        # An API error is recorded, but the environment still decides. A flaky
        # network must not be able to quietly deflate the recovery metrics.
        if self.mode is ExecutionMode.TEST_MODE:
            self._call_paypal(
                record, intervention, amount_attempted, receipt, recovered, result
            )
        else:
            support = TEST_MODE_SUPPORT[intervention]
            result.was_mocked = not support.supported
            result.mock_reason = support.mock_reason

        if recovered:
            result.amount_recovered_cents = amount_attempted
            result.outcome = Outcome.RECOVERED
        else:
            result.amount_recovered_cents = 0
            result.outcome = Outcome.FAILED
        return result

    # ------------------------------------------------------------------
    def _call_paypal(
        self,
        record: FailedRecord,
        intervention: Intervention,
        amount_cents: int,
        receipt: str,
        will_recover: bool,
        result: ExecutionResult,
    ) -> None:
        support = TEST_MODE_SUPPORT[intervention]
        if not support.supported:
            # Flagged, not hidden: the limitation travels into the audit row.
            result.was_mocked = True
            result.mock_reason = support.mock_reason
            result.api_called = False
            return

        # 1. A real sandbox order. PayPal-Request-Id makes a replay harmless.
        try:
            order = self.client.create_order(
                amount_cents=amount_cents,
                currency=CURRENCY,
                reference_id=receipt,
                idempotency_key=f"{receipt}-create",
                description=f"Recoup {intervention.value} for {record.record_id}",
            )
            result.api_called = True
            result.paypal_entity_id = order.get("id")
            result.detail["order_status"] = order.get("status")
            for link in order.get("links", []):
                if link.get("rel") in ("approve", "payer-action"):
                    result.detail["approve_url"] = link.get("href")
        except Exception as exc:  # noqa: BLE001 - the error is data, not control
            result.api_called = True
            result.api_error = f"{type(exc).__name__}: {exc}"
            return

        # 2. A retry the world says FAILED gets a genuine PayPal error body.
        #    A retry that recovered is not captured: capture needs the buyer to
        #    approve the order, which the sandbox cannot do unattended.
        if support.entity != "order" or will_recover or not result.paypal_entity_id:
            return
        reason = (record.error_reason or "").strip().lower()
        mock = reason.upper() if reason in SANDBOX_MOCKABLE else DEFAULT_RETRY_MOCK
        result.detail["mock_code"] = mock
        result.detail["original_reason"] = record.error_reason
        try:
            resp = self.client.capture_order(
                result.paypal_entity_id,
                idempotency_key=f"{receipt}-capture",
                mock_code=mock,
            )
            result.detail["capture_status"] = resp.get("status")
        except PayPalApiError as exc:
            result.detail["paypal_error"] = _summarise_error(exc)
        except Exception as exc:  # noqa: BLE001
            result.api_error = f"{type(exc).__name__}: {exc}"
