"""FastAPI app: PayPal webhook receiver plus a guarded live-demo endpoint."""
from __future__ import annotations

import hmac
import json
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from recoup.api.normalise import ORDER_APPROVED, normalise, order_id_of
from recoup.api.registry import OrderRef, OrderRegistry
from recoup.paypal.orders import PayPalApiError
from recoup.paypal.webhook_verify import WebhookNotConfigured, verify_webhook


class DemoOrder(BaseModel):
    record_id: str
    customer_id: str = "demo"
    amount_cents: int = Field(default=100, ge=100, le=10_000)


def create_app(
    *,
    settings: Any,
    auth: Any,
    orders: Any,
    registry: OrderRegistry,
    ingest: Any,
    verifier: Callable[..., bool] = verify_webhook,
) -> FastAPI:
    app = FastAPI(title="Recoup")
    log: Deque[Dict[str, Any]] = deque(maxlen=500)
    app.state.event_log = log

    @app.get("/healthz")
    def healthz() -> Dict[str, bool]:
        return {"ok": True}

    @app.get("/api/webhook-events")
    def webhook_events() -> list:
        return list(log)

    @app.post("/webhook")
    async def webhook(request: Request) -> Dict[str, Any]:
        raw = await request.body()  # raw bytes first; the signature covers them
        try:
            ok = await run_in_threadpool(
                verifier, settings, auth, request.headers, raw
            )
        except WebhookNotConfigured:
            raise HTTPException(503, "webhook verification not configured")
        except Exception:
            raise HTTPException(503, "verification unavailable, retry later")
        if not ok:
            raise HTTPException(401, "invalid webhook signature")

        event = json.loads(raw)
        etype = event.get("event_type")
        order_id = order_id_of(event)
        ref = registry.lookup(order_id) if order_id else None
        entry: Dict[str, Any] = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "event_id": event.get("id"),
            "event_type": etype,
            "order_id": order_id,
            "matched": ref is not None,
        }
        log.append(entry)

        if ref is None:
            entry["result"] = "ignored_unknown_order"
            return {"status": "ignored"}

        if etype == ORDER_APPROVED:
            try:
                await run_in_threadpool(
                    lambda: orders.capture_order(
                        order_id, idempotency_key=f"{order_id}-capture-approved"
                    )
                )
            except PayPalApiError as exc:
                entry["result"] = f"capture_rejected_{exc.status}"
                return {"status": "capture_rejected"}
            except Exception:
                raise HTTPException(503, "capture unavailable, retry later")
            entry["result"] = "capture_requested"
            return {"status": "capture_requested"}

        normalised = normalise(event, ref)
        if normalised is None:
            entry["result"] = "ignored_event_type"
            return {"status": "ignored"}
        resolution = ingest.ingest_outcome(normalised)
        if resolution is None:
            entry["result"] = "duplicate"
            return {"status": "duplicate"}
        entry["result"] = resolution.resolution.value
        return {"status": resolution.resolution.value}

    @app.post("/api/demo/order")
    async def demo_order(body: DemoOrder, request: Request) -> Dict[str, Any]:
        token = os.getenv("DEMO_TOKEN", "")
        if not token:
            raise HTTPException(404)  # disabled unless explicitly enabled
        supplied = request.headers.get("x-demo-token", "")
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(401)
        order = await run_in_threadpool(
            lambda: orders.create_order(
                amount_cents=body.amount_cents,
                currency="USD",
                reference_id=body.record_id,
                idempotency_key=f"demo-{uuid.uuid4().hex}",
                description="Recoup live demo order",
            )
        )
        registry.register(order["id"], OrderRef(body.record_id, body.customer_id))
        approve = next(
            (l["href"] for l in order.get("links", [])
             if l.get("rel") in ("approve", "payer-action")),
            None,
        )
        return {"order_id": order["id"], "approve_url": approve}

    return app
