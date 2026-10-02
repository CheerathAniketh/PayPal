"""The synthetic batch: customers, records, latent truth, and a grouped holdout.

Everything flows from one seed, so the frozen batch in the repo is exactly
reproducible.  Reproducibility is itself a hiring signal.

Freezing writes three separate files -- observable records, the answer key, and
the split -- for the same reason ``models.py`` splits the objects: it should be
*structurally obvious* that the answer key is not training input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config.taxonomy import (
    ALL_CLASSES,
    REASON_MAP,
    SOURCE_VOCABULARY,
    FailureClass,
    PaymentMethod,
)
from recoup.models import Customer, FailedRecord, LatentTruth
from recoup.money import rupees_to_paise

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FROZEN_PATH = DATA_DIR / "batch_frozen.json"
LATENT_PATH = DATA_DIR / "batch_latent.json"
SPLIT_PATH = DATA_DIR / "batch_split.json"

# The batch's reference "now". A fixed anchor, so the demo hits the same salary
# windows regardless of what day the video is recorded.
BATCH_ANCHOR = datetime(2026, 1, 10, 9, 0, 0)


@dataclass(frozen=True)
class GeneratorConfig:
    n_records: int = 180
    n_customers: int = 75
    seed: int = 20260251

    # Class mix, weighted the way a real dunning queue looks.
    class_weights: Dict[FailureClass, float] = field(
        default_factory=lambda: {
            FailureClass.INSUFFICIENT_FUNDS: 0.34,
            FailureClass.BANK_DOWNTIME: 0.20,
            FailureClass.CARD_EXPIRED: 0.18,
            FailureClass.MANDATE_BROKEN: 0.16,
            FailureClass.DO_NOT_HONOUR: 0.12,
        }
    )

    # Amounts are lognormal: real payment distributions are right-skewed, not
    # uniform. Mandate debits skew larger.
    amount_mu: float = 6.24
    amount_mu_mandate: float = 6.64
    amount_sigma: float = 0.55

    # Seeded-dead records make graceful give-up demonstrable rather than
    # asserted.
    dead_rate_do_not_honour: float = 0.65
    dead_rate_other: float = 0.03

    # Latent-trait priors. The slopes are tuned so the traits land in the
    # correlation band the tests lock down: learnable in part, not recoverable
    # exactly.
    engagement_intercept: float = 0.22
    engagement_slope: float = 0.64
    engagement_k: float = 6.0
    regularity_intercept: float = 0.36
    regularity_slope: float = 0.34
    regularity_k: float = 4.0

    n_holdout_customers: int = 18


SALARY_DAYS = (1, 2, 3, 5, 7, 25, 27, 28, 30)
SUBSCRIPTION_STATUSES = ("active", "paused", "halted", "cancelled")

# Method coherence: a card_expired record is always a card; a mandate failure
# happens on a mandate rail.
CLASS_METHODS: Dict[FailureClass, Tuple[PaymentMethod, ...]] = {
    FailureClass.INSUFFICIENT_FUNDS: (
        PaymentMethod.CARD, PaymentMethod.NETBANKING,
        PaymentMethod.UPI, PaymentMethod.EMANDATE,
    ),
    FailureClass.BANK_DOWNTIME: (
        PaymentMethod.CARD, PaymentMethod.NETBANKING,
        PaymentMethod.UPI, PaymentMethod.EMANDATE,
    ),
    FailureClass.CARD_EXPIRED: (PaymentMethod.CARD,),
    FailureClass.MANDATE_BROKEN: (PaymentMethod.EMANDATE, PaymentMethod.UPI),
    FailureClass.DO_NOT_HONOUR: (PaymentMethod.CARD, PaymentMethod.UPI),
}


@dataclass
class Batch:
    records: List[FailedRecord]
    customers: Dict[str, Customer]
    latent: Dict[str, LatentTruth]
    holdout_customer_ids: List[str]
    seed: int

    # ---- convenience -------------------------------------------------
    @property
    def holdout_records(self) -> List[FailedRecord]:
        held = set(self.holdout_customer_ids)
        return [r for r in self.records if r.customer_id in held]

    @property
    def train_records(self) -> List[FailedRecord]:
        held = set(self.holdout_customer_ids)
        return [r for r in self.records if r.customer_id not in held]

    def total_at_risk_paise(self) -> int:
        return sum(r.amount_paise for r in self.records)

    def recurring_customer_count(self) -> int:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record.customer_id] = counts.get(record.customer_id, 0) + 1
        return sum(1 for n in counts.values() if n > 1)


def _beta_around(rng: np.random.Generator, prior: float, k: float) -> float:
    """Draw a trait from a Beta centred on an observable-driven prior.

    Concentration ``k`` is deliberately modest so real per-customer variation
    survives.  The whole point is a correlation *band*: near 0 the traits are
    unlearnable noise and the model has nothing to find; near 1 they are
    perfectly recoverable and the grouped split is theatre.  In between, a
    customer carries persistent hidden information a model can only partly
    infer -- and grouping removes a real shortcut.
    """
    prior = float(np.clip(prior, 0.05, 0.95))
    return float(rng.beta(k * prior, k * (1.0 - prior)))


def generate(config: GeneratorConfig = GeneratorConfig()) -> Batch:
    rng = np.random.default_rng(config.seed)

    # ---- customers ---------------------------------------------------
    customers: Dict[str, Customer] = {}
    for i in range(config.n_customers):
        cid = f"cust_{i:04d}"
        tenure = int(np.clip(rng.lognormal(mean=5.55, sigma=0.85), 14, 2200))
        avg_payment = rupees_to_paise(
            float(np.clip(rng.lognormal(mean=6.2, sigma=0.5), 90, 12000))
        )
        engagement = _beta_around(
            rng,
            config.engagement_intercept
            + config.engagement_slope * min(tenure / 900.0, 1.0),
            config.engagement_k,
        )
        regularity = _beta_around(
            rng,
            config.regularity_intercept
            + config.regularity_slope * min(tenure / 900.0, 1.0),
            config.regularity_k,
        )
        customers[cid] = Customer(
            customer_id=cid,
            tenure_days=tenure,
            avg_payment_paise=avg_payment,
            salary_day=int(rng.choice(SALARY_DAYS)),
            is_subscriber=bool(rng.random() < 0.72),
            prior_failures=int(rng.poisson(0.8)),
            engagement=engagement,
            income_regularity=regularity,
        )

    # ---- who owns which record --------------------------------------
    # Every customer gets one record so that all 75 actually appear; the
    # remainder is distributed by a weight that rises with a customer's prior
    # failure history, because customers who fail tend to fail again. That
    # concentration is what produces recurring customers with several records
    # -- without them, GroupKFold(customer_id) would have groups of size one
    # and would be meaningless.
    ids = list(customers)
    owners: List[str] = list(ids)
    extras = config.n_records - config.n_customers
    weights = np.array(
        [customers[cid].prior_failures + 0.35 for cid in ids], dtype=float
    )
    weights /= weights.sum()
    owners.extend(rng.choice(ids, size=extras, p=weights).tolist())
    rng.shuffle(owners)

    # ---- records + latent -------------------------------------------
    classes = list(config.class_weights)
    class_p = np.array([config.class_weights[c] for c in classes], dtype=float)
    class_p /= class_p.sum()

    records: List[FailedRecord] = []
    latent: Dict[str, LatentTruth] = {}

    for idx, cid in enumerate(owners):
        customer = customers[cid]
        klass = classes[int(rng.choice(len(classes), p=class_p))]
        method = CLASS_METHODS[klass][int(rng.integers(len(CLASS_METHODS[klass])))]
        reasons = REASON_MAP[klass]
        reason = reasons[int(rng.integers(len(reasons)))]
        sources = SOURCE_VOCABULARY[method]
        source = sources[int(rng.integers(len(sources)))]

        mu = (
            config.amount_mu_mandate
            if klass is FailureClass.MANDATE_BROKEN
            else config.amount_mu
        )
        amount = rupees_to_paise(
            float(np.clip(rng.lognormal(mean=mu, sigma=config.amount_sigma), 49, 40000))
        )

        is_mandate_debit = method is PaymentMethod.EMANDATE or (
            method is PaymentMethod.UPI and customer.is_subscriber
        )
        if klass is FailureClass.MANDATE_BROKEN:
            status = str(
                rng.choice(
                    SUBSCRIPTION_STATUSES, p=[0.10, 0.30, 0.25, 0.35]
                )
            )
        elif customer.is_subscriber:
            status = str(rng.choice(SUBSCRIPTION_STATUSES, p=[0.88, 0.07, 0.03, 0.02]))
        else:
            status = "none"

        # Records arrive mid-journey, not all fresh.
        prior_retries = int(rng.choice([0, 0, 0, 1, 1, 2]))
        failed_at = BATCH_ANCHOR - timedelta(
            hours=float(rng.uniform(6.0, 24.0 * 7.0))
        )

        record_id = f"rec_{idx:04d}"
        records.append(
            FailedRecord(
                record_id=record_id,
                customer_id=cid,
                amount_paise=amount,
                method=method,
                error_reason=reason,
                error_source=source,
                failed_at=failed_at,
                prior_retries=prior_retries,
                is_mandate_debit=is_mandate_debit,
                pre_debit_notified=(not is_mandate_debit) or bool(rng.random() < 0.85),
                subscription_status=status,
                customer_tenure_days=customer.tenure_days,
                customer_avg_payment_paise=customer.avg_payment_paise,
                customer_salary_day=customer.salary_day,
                customer_prior_failures=customer.prior_failures,
                customer_is_subscriber=customer.is_subscriber,
            )
        )

        dead_rate = (
            config.dead_rate_do_not_honour
            if klass is FailureClass.DO_NOT_HONOUR
            else config.dead_rate_other
        )
        latent[record_id] = LatentTruth(
            record_id=record_id,
            base_logodds=float(rng.normal(0.0, 0.9)),
            is_truly_dead=bool(rng.random() < dead_rate),
            seeded_class=klass,
            engagement=customer.engagement,
            income_regularity=customer.income_regularity,
        )

    # ---- customer-grouped holdout ------------------------------------
    holdout = sorted(
        rng.choice(ids, size=config.n_holdout_customers, replace=False).tolist()
    )

    return Batch(
        records=records,
        customers=customers,
        latent=latent,
        holdout_customer_ids=holdout,
        seed=config.seed,
    )


# --------------------------------------------------------------------------
# Freezing / thawing
# --------------------------------------------------------------------------
def freeze(batch: Batch, data_dir: Path = DATA_DIR) -> Dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)

    frozen = {
        "anchor": BATCH_ANCHOR.isoformat(),
        "seed": batch.seed,
        "records": [r.to_json() for r in batch.records],
        # Customers appear here WITHOUT their latent traits.
        "customers": [c.observable() for c in batch.customers.values()],
    }
    latent = {
        "seed": batch.seed,
        "latent": [t.to_json() for t in batch.latent.values()],
        "customer_traits": {
            cid: {
                "engagement": c.engagement,
                "income_regularity": c.income_regularity,
            }
            for cid, c in batch.customers.items()
        },
    }
    split = {
        "seed": batch.seed,
        "holdout_customer_ids": batch.holdout_customer_ids,
        "holdout_record_ids": [r.record_id for r in batch.holdout_records],
        "train_record_ids": [r.record_id for r in batch.train_records],
    }

    paths = {
        "frozen": data_dir / FROZEN_PATH.name,
        "latent": data_dir / LATENT_PATH.name,
        "split": data_dir / SPLIT_PATH.name,
    }
    paths["frozen"].write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    paths["latent"].write_text(json.dumps(latent, indent=2), encoding="utf-8")
    paths["split"].write_text(json.dumps(split, indent=2), encoding="utf-8")
    return paths


def load_frozen(data_dir: Path = DATA_DIR) -> Tuple[List[FailedRecord], Dict[str, Customer]]:
    """Load the OBSERVABLE batch.  Deliberately cannot return latent truth."""
    payload = json.loads((data_dir / FROZEN_PATH.name).read_text(encoding="utf-8"))
    records = [FailedRecord.from_json(r) for r in payload["records"]]
    customers = {
        c["customer_id"]: Customer(**c) for c in payload["customers"]
    }
    return records, customers


def load_latent(data_dir: Path = DATA_DIR) -> Dict[str, LatentTruth]:
    """Load the ANSWER KEY.  Only the environment and the evaluator call this."""
    payload = json.loads((data_dir / LATENT_PATH.name).read_text(encoding="utf-8"))
    return {t["record_id"]: LatentTruth.from_json(t) for t in payload["latent"]}


def load_split(data_dir: Path = DATA_DIR) -> Dict[str, List[str]]:
    payload = json.loads((data_dir / SPLIT_PATH.name).read_text(encoding="utf-8"))
    return {
        "holdout_customer_ids": payload["holdout_customer_ids"],
        "holdout_record_ids": payload["holdout_record_ids"],
        "train_record_ids": payload["train_record_ids"],
    }


def hydrate_customers(
    customers: Dict[str, Customer], data_dir: Path = DATA_DIR
) -> Dict[str, Customer]:
    """Re-attach latent traits to observable customers.  EVAL ONLY."""
    payload = json.loads((data_dir / LATENT_PATH.name).read_text(encoding="utf-8"))
    traits = payload["customer_traits"]
    for cid, customer in customers.items():
        if cid in traits:
            customer.engagement = traits[cid]["engagement"]
            customer.income_regularity = traits[cid]["income_regularity"]
    return customers
