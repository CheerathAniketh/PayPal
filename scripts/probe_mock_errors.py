"""Probe which PayPal-Mock-Response codes the sandbox accepts on order capture.

Sandbox only. Creates one $1.00 order, then tries to capture it once per mock
code. A mocked error never captures the order, so one order serves every probe.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "https://api-m.sandbox.paypal.com"

CODES = [
    "INSTRUMENT_DECLINED",
    "TRANSACTION_REFUSED",
    "INSUFFICIENT_FUNDS",
    "CARD_EXPIRED",
    "PAYER_ACTION_REQUIRED",
    "ORDER_NOT_APPROVED",
    "PAYER_CANNOT_PAY",
    "PAYER_ACCOUNT_RESTRICTED",
    "TRANSACTION_LIMIT_EXCEEDED",
    "INTERNAL_SERVER_ERROR",
]


def load_env(path: str = ".env") -> dict:
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def call(method: str, url: str, headers: dict, data: bytes | None = None):
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = {"raw": raw[:200].decode(errors="replace")}
    return status, body


def main() -> None:
    env = load_env()
    if env.get("PAYPAL_ENV", "sandbox").lower() != "sandbox":
        raise SystemExit("refusing to run: PAYPAL_ENV must be sandbox")
    cid, secret = env["PAYPAL_CLIENT_ID"], env["PAYPAL_CLIENT_SECRET"]

    basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    status, body = call(
        "POST",
        f"{BASE}/v1/oauth2/token",
        {
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        b"grant_type=client_credentials",
    )
    if status != 200:
        raise SystemExit(f"token failed: {status} {body}")
    auth = {
        "Authorization": f"Bearer {body['access_token']}",
        "Content-Type": "application/json",
    }

    order_req = {
        "intent": "CAPTURE",
        "purchase_units": [
            {"amount": {"currency_code": "USD", "value": "1.00"}}
        ],
    }
    status, order = call(
        "POST",
        f"{BASE}/v2/checkout/orders",
        {**auth, "PayPal-Request-Id": str(uuid.uuid4())},
        json.dumps(order_req).encode(),
    )
    if status not in (200, 201):
        raise SystemExit(f"order failed: {status} {order}")
    order_id = order["id"]
    print(f"order {order_id} created\n")

    sample_printed = False
    for code in CODES:
        status, body = call(
            "POST",
            f"{BASE}/v2/checkout/orders/{order_id}/capture",
            {
                **auth,
                "PayPal-Request-Id": str(uuid.uuid4()),
                "PayPal-Mock-Response": json.dumps(
                    {"mock_application_codes": code}
                ),
            },
            b"{}",
        )
        name = body.get("name", "?")
        details = body.get("details") or [{}]
        issue = details[0].get("issue", "-")
        print(f"{code:28s} HTTP {status}  name={name}  issue={issue}")
        if not sample_printed and issue == code:
            print("  sample body:", json.dumps(body)[:600])
            sample_printed = True


if __name__ == "__main__":
    main()
