"""PayPal Orders v2 calls. The idempotency key goes in PayPal-Request-Id."""
from __future__ import annotations

import json

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

    def capture_order(self, order_id: str, *, idempotency_key: str,
                      mock_code: str | None = None) -> dict:
        """Capture an approved order.

        ``mock_code`` sets the sandbox-only PayPal-Mock-Response header so the
        sandbox returns that PayPal error. PayPalSettings refuses live mode, so
        this can never reach production.
        """
        headers = self._headers(idempotency_key)
        if mock_code:
            headers["PayPal-Mock-Response"] = json.dumps(
                {"mock_application_codes": mock_code}
            )
        resp = self._http.post(
            f"{self._s.base_url}/v2/checkout/orders/{order_id}/capture",
            data=b"{}", headers=headers, timeout=20,
        )
        body = resp.json() if resp.content else {}
        if resp.status_code not in (200, 201):
            raise PayPalApiError(resp.status_code, body)
        return body
