"""Shared fixtures.

Kept deliberately small: most tests want a hand-built record whose properties
they control, not a slice of the batch.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from config.taxonomy import FailureClass, PaymentMethod
from recoup.db import Store
from recoup.environment import RecoveryEnvironment
from recoup.generator import BATCH_ANCHOR, generate
from recoup.models import Customer, FailedRecord, LatentTruth

NOW = datetime(2026, 1, 16, 9, 0, 0)


def make_customer(
    customer_id: str = "cust_test",
    *,
    salary_day: int = 1,
    avg_payment_cents: int = 50_000,
    engagement: float = 0.5,
    income_regularity: float = 0.5,
    tenure_days: int = 600,
) -> Customer:
    return Customer(
        customer_id=customer_id,
        tenure_days=tenure_days,
        avg_payment_cents=avg_payment_cents,
        salary_day=salary_day,
        is_subscriber=True,
        prior_failures=1,
        engagement=engagement,
        income_regularity=income_regularity,
    )


def make_record(
    record_id: str = "rec_test",
    *,
    customer: Customer | None = None,
    amount_cents: int = 73_600,
    reason: str = "insufficient_funds",
    method: PaymentMethod = PaymentMethod.CARD,
    prior_retries: int = 0,
    failed_at: datetime = NOW,
    is_mandate_debit: bool = False,
    pre_debit_notified: bool = True,
    subscription_status: str = "active",
) -> FailedRecord:
    customer = customer or make_customer()
    return FailedRecord(
        record_id=record_id,
        customer_id=customer.customer_id,
        amount_cents=amount_cents,
        method=method,
        error_reason=reason,
        error_source="issuer_bank",
        failed_at=failed_at,
        prior_retries=prior_retries,
        is_mandate_debit=is_mandate_debit,
        pre_debit_notified=pre_debit_notified,
        subscription_status=subscription_status,
        customer_tenure_days=customer.tenure_days,
        customer_avg_payment_cents=customer.avg_payment_cents,
        customer_salary_day=customer.salary_day,
        customer_prior_failures=customer.prior_failures,
        customer_is_subscriber=customer.is_subscriber,
    )


def make_latent(
    record_id: str = "rec_test",
    *,
    base_logodds: float = 0.0,
    is_truly_dead: bool = False,
    seeded_class: FailureClass = FailureClass.INSUFFICIENT_FUNDS,
    engagement: float = 0.5,
    income_regularity: float = 0.5,
) -> LatentTruth:
    return LatentTruth(
        record_id=record_id,
        base_logodds=base_logodds,
        is_truly_dead=is_truly_dead,
        seeded_class=seeded_class,
        engagement=engagement,
        income_regularity=income_regularity,
    )


@pytest.fixture
def store() -> Store:
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def env() -> RecoveryEnvironment:
    return RecoveryEnvironment()


@pytest.fixture(scope="session")
def batch():
    return generate()


@pytest.fixture(scope="session")
def anchor() -> datetime:
    return BATCH_ANCHOR
