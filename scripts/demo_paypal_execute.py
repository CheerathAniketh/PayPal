"""Live sandbox demo: run records through the TEST_MODE executor.

Creates real sandbox orders.  Run:  python -m scripts.demo_paypal_execute
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime

from config.paypal_settings import load_settings
from config.taxonomy import FailureClass, PRIMARY_INTERVENTION, classify_reason
from recoup.environment import RecoveryEnvironment
from recoup.executor import Executor
from recoup.generator import hydrate_customers, load_frozen, load_latent
from recoup.models import ExecutionMode
from recoup.paypal.auth import PayPalAuth
from recoup.paypal.orders import PayPalOrders

PER_CLASS = 2


def main() -> None:
    settings = load_settings()
    client = PayPalOrders(settings, PayPalAuth(settings))
    records, customers = load_frozen()
    customers = hydrate_customers(customers)
    latent = load_latent()
    executor = Executor(
        RecoveryEnvironment(), mode=ExecutionMode.TEST_MODE, client=client
    )

    seen: Counter = Counter()
    for rec in records:
        klass = classify_reason(rec.error_reason)
        if klass is FailureClass.UNKNOWN or seen[klass] >= PER_CLASS:
            continue
        seen[klass] += 1
        intervention = PRIMARY_INTERVENTION[klass]
        res = executor.execute(
            rec,
            customers[rec.customer_id],
            latent[rec.record_id],
            intervention,
            run_id="demo-live",
            attempt_number=1,
            attempt_at=datetime(2026, 1, 27, 9),
        )
        print(
            f"{klass.name:18s} {intervention.value:22s} "
            f"{res.outcome.value:10s} entity={res.paypal_entity_id} "
            f"api_called={res.api_called} mocked={res.was_mocked}"
        )
        if res.api_error:
            print("   api_error:", res.api_error)
        if res.detail:
            print("   detail:", res.detail)


if __name__ == "__main__":
    main()
