"""Webhook signature verification via PayPal's verify-webhook-signature API.

PayPal does the cryptography server-side, so this module never fetches the
certificate URL itself (no server-side-request-forgery surface).

Contract with the HTTP layer:
* returns True  -> PayPal reported verification_status == SUCCESS
* returns False -> the event is not authentic (bad/missing headers, bad JSON,
                   FAILURE, or a non-200 answer from PayPal); respond 401
* raises        -> verification could not be performed (network error, or
                   WebhookNotConfigured); respond 503 so PayPal retries

The raw request bytes must be passed in, captured before any framework parses
them.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import requests

VERIFY_PATH = "/v1/notifications/verify-webhook-signature"

REQUIRED_HEADERS = (
    "paypal-auth-algo",
    "paypal-cert-url",
    "paypal-transmission-id",
    "paypal-transmission-sig",
    "paypal-transmission-time",
)


class WebhookNotConfigured(RuntimeError):
    """PAYPAL_WEBHOOK_ID is missing; refusing to accept unverified events."""


def verify_webhook(
    settings: Any,
    auth: Any,
    headers: Mapping[str, str],
    raw_body: bytes,
    session: Optional[requests.Session] = None,
) -> bool:
    if not getattr(settings, "webhook_id", ""):
        raise WebhookNotConfigured(
            "PAYPAL_WEBHOOK_ID is not set; refusing to accept unverified webhooks"
        )

    h = {k.lower(): v for k, v in headers.items()}
    if any(not h.get(k) for k in REQUIRED_HEADERS):
        return False

    try:
        event = json.loads(raw_body)
    except ValueError:
        return False

    payload = {
        "auth_algo": h["paypal-auth-algo"],
        "cert_url": h["paypal-cert-url"],
        "transmission_id": h["paypal-transmission-id"],
        "transmission_sig": h["paypal-transmission-sig"],
        "transmission_time": h["paypal-transmission-time"],
        "webhook_id": settings.webhook_id,
        "webhook_event": event,
    }
    http = session or requests.Session()
    resp = http.post(
        f"{settings.base_url}{VERIFY_PATH}",
        json=payload,
        headers={
            "Authorization": f"Bearer {auth.token()}",
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    if resp.status_code != 200:
        return False
    return resp.json().get("verification_status") == "SUCCESS"
