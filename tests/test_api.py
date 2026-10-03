"""Webhook app: signature gate, capture on approval, idempotent recovery."""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from recoup.agent.outcomes import OutcomeIngest
from recoup.api.app import create_app
from recoup.api.registry import OrderRef, OrderRegistry
from recoup.paypal.orders import PayPalApiError
from recoup.paypal.webhook_verify import WebhookNotConfigured


class FakeOrders:
    def __init__(self):
        self.captured = []
        self.capture_error = None

    def capture_order(self, order_id, *, idempotency_key, mock_code=None):
        if self.capture_error:
            raise self.capture_error
        self.captured.append((order_id, idempotency_key))
        return {"status": "COMPLETED"}

    def create_order(self, **kwargs):
        return {
            "id": "NEWORDER",
            "links": [{"rel": "approve", "href": "https://example/approve"}],
        }


def build(verifier=lambda *a, **k: True):
    resolutions = []
    registry = OrderRegistry()
    registry.register("ORDER1", OrderRef("rec_0001", "cust_0001"))
    orders = FakeOrders()
    app = create_app(
        settings=SimpleNamespace(),
        auth=SimpleNamespace(),
        orders=orders,
        registry=registry,
        ingest=OutcomeIngest(apply=resolutions.append),
        verifier=verifier,
    )
    return TestClient(app), orders, resolutions, registry


def capture_event(event_id="WH-1", etype="PAYMENT.CAPTURE.COMPLETED",
                  order="ORDER1", value="36.80"):
    return {
        "id": event_id,
        "event_type": etype,
        "create_time": "2026-10-03T10:00:00Z",
        "resource": {
            "id": "CAP1",
            "amount": {"currency_code": "USD", "value": value},
            "supplementary_data": {"related_ids": {"order_id": order}},
        },
    }


def post(client, event):
    return client.post("/webhook", content=json.dumps(event))


def test_bad_signature_is_401_and_nothing_is_ingested():
    client, _, resolutions, _ = build(verifier=lambda *a, **k: False)
    assert post(client, capture_event()).status_code == 401
    assert resolutions == []


def test_unconfigured_webhook_is_503():
    def boom(*a, **k):
        raise WebhookNotConfigured("not set")

    client, *_ = build(verifier=boom)
    assert post(client, capture_event()).status_code == 503


def test_capture_completed_recovers_the_exact_amount():
    client, _, resolutions, _ = build()
    assert post(client, capture_event(value="36.80")).status_code == 200
    assert len(resolutions) == 1
    assert resolutions[0].record_id == "rec_0001"
    assert resolutions[0].amount_recovered_cents == 3680


def test_duplicate_delivery_is_not_double_counted():
    client, _, resolutions, _ = build()
    post(client, capture_event(event_id="WH-9"))
    again = post(client, capture_event(event_id="WH-9"))
    assert again.json()["status"] == "duplicate"
    assert len(resolutions) == 1


def test_order_approved_triggers_capture_not_recovery():
    client, orders, resolutions, _ = build()
    event = {"id": "WH-2", "event_type": "CHECKOUT.ORDER.APPROVED",
             "resource": {"id": "ORDER1"}}
    assert post(client, event).json()["status"] == "capture_requested"
    assert orders.captured == [("ORDER1", "ORDER1-capture-approved")]
    assert resolutions == []


def test_capture_rejected_by_paypal_is_acknowledged_not_retried():
    client, orders, _, _ = build()
    orders.capture_error = PayPalApiError(422, {"name": "UNPROCESSABLE_ENTITY"})
    event = {"id": "WH-3", "event_type": "CHECKOUT.ORDER.APPROVED",
             "resource": {"id": "ORDER1"}}
    r = post(client, event)
    assert r.status_code == 200
    assert r.json()["status"] == "capture_rejected"


def test_unknown_order_is_acknowledged_and_ignored():
    client, _, resolutions, _ = build()
    r = post(client, capture_event(order="UNKNOWN"))
    assert r.status_code == 200
    assert r.json()["status"] == "ignored"
    assert resolutions == []


def test_capture_denied_lapses_with_zero_recovered():
    client, _, resolutions, _ = build()
    post(client, capture_event(etype="PAYMENT.CAPTURE.DENIED"))
    assert resolutions[0].resolution.value == "lapsed"
    assert resolutions[0].amount_recovered_cents == 0


def test_demo_endpoint_is_disabled_without_a_token(monkeypatch):
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    client, *_ = build()
    assert client.post("/api/demo/order", json={"record_id": "rec_1"}).status_code == 404


def test_demo_endpoint_creates_and_registers_an_order(monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "s3cret")
    client, _, _, registry = build()
    assert client.post("/api/demo/order", json={"record_id": "rec_1"}).status_code == 401
    r = client.post(
        "/api/demo/order",
        json={"record_id": "rec_1", "amount_cents": 500},
        headers={"X-Demo-Token": "s3cret"},
    )
    assert r.status_code == 200
    assert r.json()["approve_url"] == "https://example/approve"
    assert registry.lookup("NEWORDER").record_id == "rec_1"
