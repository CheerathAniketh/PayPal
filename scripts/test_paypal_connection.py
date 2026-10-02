"""Connectivity check: fetch a sandbox token and create one $1.00 test order."""
from config.paypal_settings import load_settings
from recoup.paypal.auth import PayPalAuth
from recoup.paypal.orders import PayPalOrders


def main() -> None:
    s = load_settings()
    auth = PayPalAuth(s)
    print("token ok:", auth.token()[:12] + "...")
    order = PayPalOrders(s, auth).create_order(
        amount_cents=100, currency="USD", reference_id="smoke-001",
        idempotency_key="smoke-001-attempt-1", description="connectivity check",
    )
    print("order id:", order["id"], "| status:", order["status"])
    for link in order.get("links", []):
        if link.get("rel") == "approve":
            print("approve url:", link["href"])


if __name__ == "__main__":
    main()
