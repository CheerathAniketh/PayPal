"""Map PayPal webhook events onto what Recoup understands. Pure, no I/O.

UNVERIFIED against real payloads: where the order id lives. For capture events
it is read from resource.supplementary_data.related_ids.order_id; for
CHECKOUT.ORDER.APPROVED it is resource.id. /api/webhook-events logs every
verified event so the first real one confirms or corrects this.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from recoup.agent.outcomes import NormalisedEvent
from recoup.api.registry import OrderRef

ORDER_APPROVED = "CHECKOUT.ORDER.APPROVED"
CAPTURE_COMPLETED = "PAYMENT.CAPTURE.COMPLETED"
CAPTURE_DENIED = "PAYMENT.CAPTURE.DENIED"


def order_id_of(event: Dict[str, Any]) -> Optional[str]:
    resource = event.get("resource") or {}
    if event.get("event_type") == ORDER_APPROVED:
        return resource.get("id")
    related = (resource.get("supplementary_data") or {}).get("related_ids") or {}
    return related.get("order_id")


def _to_cents(value: Any) -> int:
    return int((Decimal(str(value)) * 100).to_integral_value())


def normalise(event: Dict[str, Any], ref: OrderRef) -> Optional[NormalisedEvent]:
    etype = event.get("event_type")
    if etype not in (CAPTURE_COMPLETED, CAPTURE_DENIED):
        return None
    resource = event.get("resource") or {}
    recovered = etype == CAPTURE_COMPLETED
    return NormalisedEvent(
        event_id=str(event.get("id", "")),
        record_id=ref.record_id,
        customer_id=ref.customer_id,
        kind="customer_action",
        # A denied capture is not recovery; "not acted" resolves to LAPSED,
        # which escalates the record to a human.
        acted=recovered,
        amount_recovered_cents=(
            _to_cents((resource.get("amount") or {}).get("value", "0"))
            if recovered else 0
        ),
        occurred_at_iso=str(event.get("create_time", "")),
    )
