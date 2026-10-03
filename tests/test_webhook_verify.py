"""Webhook verification: fail closed, no network, fakes only."""
import json
from types import SimpleNamespace

import pytest

from recoup.paypal.webhook_verify import (
    VERIFY_PATH,
    WebhookNotConfigured,
    verify_webhook,
)

SETTINGS = SimpleNamespace(
    webhook_id="WH123", base_url="https://api-m.sandbox.paypal.com"
)
HEADERS = {
    "PayPal-Auth-Algo": "SHA256withRSA",
    "PayPal-Cert-Url": "https://api.sandbox.paypal.com/cert.pem",
    "PayPal-Transmission-Id": "tid-1",
    "PayPal-Transmission-Sig": "sig",
    "PayPal-Transmission-Time": "2026-10-03T10:00:00Z",
}
BODY = json.dumps({"id": "WH-1", "event_type": "PAYMENT.CAPTURE.DENIED"}).encode()


class FakeAuth:
    def token(self):
        return "tok"


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        return self.resp


def test_success_sends_the_full_payload():
    sess = FakeSession(FakeResp(200, {"verification_status": "SUCCESS"}))
    assert verify_webhook(SETTINGS, FakeAuth(), HEADERS, BODY, sess) is True
    url, payload, headers = sess.calls[0]
    assert url.endswith(VERIFY_PATH)
    assert payload["webhook_id"] == "WH123"
    assert payload["webhook_event"]["event_type"] == "PAYMENT.CAPTURE.DENIED"
    assert payload["transmission_time"] == "2026-10-03T10:00:00Z"
    assert headers["Authorization"] == "Bearer tok"


def test_failure_status_is_rejected():
    sess = FakeSession(FakeResp(200, {"verification_status": "FAILURE"}))
    assert verify_webhook(SETTINGS, FakeAuth(), HEADERS, BODY, sess) is False


def test_non_200_from_paypal_is_rejected():
    sess = FakeSession(FakeResp(400, {"name": "VALIDATION_ERROR"}))
    assert verify_webhook(SETTINGS, FakeAuth(), HEADERS, BODY, sess) is False


def test_missing_header_is_rejected_without_calling_paypal():
    sess = FakeSession(FakeResp(200, {"verification_status": "SUCCESS"}))
    bad = {k: v for k, v in HEADERS.items() if k != "PayPal-Transmission-Sig"}
    assert verify_webhook(SETTINGS, FakeAuth(), bad, BODY, sess) is False
    assert sess.calls == []


def test_invalid_json_is_rejected_without_calling_paypal():
    sess = FakeSession(FakeResp(200, {"verification_status": "SUCCESS"}))
    assert verify_webhook(SETTINGS, FakeAuth(), HEADERS, b"not json", sess) is False
    assert sess.calls == []


def test_no_webhook_id_fails_closed():
    unset = SimpleNamespace(webhook_id="", base_url=SETTINGS.base_url)
    with pytest.raises(WebhookNotConfigured):
        verify_webhook(unset, FakeAuth(), HEADERS, BODY, FakeSession(FakeResp()))
