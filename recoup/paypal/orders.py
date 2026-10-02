"""PayPal Orders v2 calls. The idempotency key goes in PayPal-Request-Id."""
from __future__ import annotations

import requests

from config.paypal_settings import PayPalSettings
from recoup.paypal.auth import PayPalAuth


class PayPalApiError(RuntimeError):
    def __init__(self, status: int, body: dict | str):
        super().__init__(f"PayPal API error {status}: {body}")
        self.status = status
        self.body = body


class PayPalOrders:
    def __init__(self, settings: PayPalSettings, auth: PayPalAuth,
                 session: requests.Session | None = None):
        self._s = settings
        self._auth = auth
        self._http = session or requests.Session()

    def _headers(self, request_id: str | None = None) -> dict:
        h = {
            "Authorization": f"Bearer {self._auth.token()}",
            "Content-Type": "application/json",
        }
        if request_id:
            h["PayPal-Request-Id"] = request_id
        return h

    def create_order(self, *, amount_cents: int, currency: str, reference_id: str,
                     idempotency_key: str, description: str = "") -> dict:
        value = f"{amount_cents // 100}.{amount_cents % 100:02d}"
        payload = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "reference_id": reference_id,
                "description": description,
                "amount": {"currency_code": currency, "value": value},
            }],
        }
        resp = self._http.post(f"{self._s.base_url}/v2/checkout/orders",
                               json=payload, headers=self._headers(idempotency_key), timeout=20)
        body = resp.json() if resp.content else {}
        if resp.status_code not in (200, 201):
            raise PayPalApiError(resp.status_code, body)
        return body

    def get_order(self, order_id: str) -> dict:
        resp = self._http.get(f"{self._s.base_url}/v2/checkout/orders/{order_id}",
                              headers=self._headers(), timeout=20)
        body = resp.json() if resp.content else {}
        if resp.status_code != 200:
            raise PayPalApiError(resp.status_code, body)
        return body
