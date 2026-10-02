"""PayPal credentials. Sandbox only: live mode is structurally refused."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

SANDBOX_BASE_URL = "https://api-m.sandbox.paypal.com"


class LiveModeRefused(RuntimeError):
    """Raised if anything other than the sandbox is requested."""


class MissingCredentials(RuntimeError):
    """Raised when PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET are not set."""


@dataclass(frozen=True)
class PayPalSettings:
    client_id: str
    client_secret: str
    webhook_id: str = ""
    base_url: str = SANDBOX_BASE_URL


def load_settings() -> PayPalSettings:
    load_dotenv()
    env = os.getenv("PAYPAL_ENV", "sandbox").strip().lower()
    if env != "sandbox":
        raise LiveModeRefused(f"PAYPAL_ENV={env!r} refused: this project is sandbox-only.")
    client_id = os.getenv("PAYPAL_CLIENT_ID", "").strip()
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise MissingCredentials("Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET in .env")
    return PayPalSettings(
        client_id=client_id,
        client_secret=client_secret,
        webhook_id=os.getenv("PAYPAL_WEBHOOK_ID", "").strip(),
    )
