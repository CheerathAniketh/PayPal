"""PayPal OAuth2 client-credentials token, cached until shortly before expiry."""
from __future__ import annotations

import threading
import time

import requests

from config.paypal_settings import PayPalSettings

_EXPIRY_MARGIN_S = 60


class PayPalAuthError(RuntimeError):
    pass


class PayPalAuth:
    def __init__(self, settings: PayPalSettings, session: requests.Session | None = None):
        self._s = settings
        self._http = session or requests.Session()
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at - _EXPIRY_MARGIN_S:
                return self._token
            resp = self._http.post(
                f"{self._s.base_url}/v1/oauth2/token",
                auth=(self._s.client_id, self._s.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if resp.status_code != 200:
                raise PayPalAuthError(f"token request failed: {resp.status_code} {resp.text[:300]}")
            body = resp.json()
            self._token = body["access_token"]
            self._expires_at = time.time() + int(body.get("expires_in", 300))
            return self._token
