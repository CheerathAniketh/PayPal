"""Credentials, and a refusal to start on a live key.

Recoup autonomously retries debits.  "Never touches real money" has to be
structural, not careful -- so a key that does not start with ``rzp_test_``
raises at import/config time rather than warning.  ``.env`` is gitignored: this
is a public submission, and a leaked key, even a test one, is a bad look.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

TEST_KEY_PREFIX = "rzp_test_"
LIVE_KEY_PREFIX = "rzp_live_"


class LiveModeRefused(RuntimeError):
    """Raised when a live Razorpay key is supplied.  Not a warning."""


class MissingCredentials(RuntimeError):
    pass


def load_dotenv(path: Path = ENV_PATH) -> dict:
    """Minimal .env reader.  Deliberately dependency-free so Phase 1 needs
    nothing but numpy and pytest."""
    values: dict = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    key_id: str
    key_secret: str

    @property
    def masked_secret(self) -> str:
        if len(self.key_secret) <= 8:
            return "*" * len(self.key_secret)
        return f"{self.key_secret[:4]}{'*' * (len(self.key_secret) - 8)}{self.key_secret[-4:]}"

    def client(self):  # pragma: no cover - requires the razorpay package
        import razorpay

        client = razorpay.Client(auth=(self.key_id, self.key_secret))
        client.set_app_details({"title": "Recoup", "version": "1.0"})
        return client


def validate_key_id(key_id: str) -> str:
    if key_id.startswith(LIVE_KEY_PREFIX):
        raise LiveModeRefused(
            "refusing to start with a LIVE Razorpay key. Recoup autonomously "
            "retries debits; it must never be pointed at real money."
        )
    if not key_id.startswith(TEST_KEY_PREFIX):
        raise LiveModeRefused(
            f"key_id must start with {TEST_KEY_PREFIX!r}; got {key_id[:12]!r}"
        )
    return key_id


def get_settings(
    env: Optional[dict] = None, path: Path = ENV_PATH
) -> Settings:
    source = dict(load_dotenv(path))
    source.update({k: v for k, v in os.environ.items() if k.startswith("RAZORPAY_")})
    if env:
        source.update(env)

    key_id = source.get("RAZORPAY_KEY_ID", "")
    key_secret = source.get("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise MissingCredentials(
            "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set "
            f"(looked in {path} and the environment)"
        )
    validate_key_id(key_id)
    return Settings(key_id=key_id, key_secret=key_secret)
