"""The feature contract: observable record + intervention -> fixed vector.

Everything the model may see enters through this one function.  It reads only
from the observable record dict (the shape of :meth:`FailedRecord.to_features`
and :meth:`FailedRecord.to_json`, which share their raw keys) plus the candidate
intervention.  It *cannot* reach the latent traits: they are not passed in.
That is the leakage wall, enforced here at the single entry point rather than
trusted to reviewer vigilance.

Design choices
--------------
* **Fixed categorical vocabularies, hand-rolled one-hot.**  The vocabularies come
  from the taxonomy, not from whatever happens to appear in a given split.  A
  train/holdout split that never sees ``bank_downtime`` still produces the same
  columns in the same order, so a model trained on one split scores correctly on
  any record.  This is the property a stateful ``OneHotEncoder`` would have to
  promise and could silently break.
* **Derived ``failure_class`` instead of the raw reason string.**  The class is a
  deterministic function of the observable reason code (Razorpay hands it to us),
  and within a class the specific reason carries no additional recovery signal in
  this environment.  Feeding the class keeps the vector low-cardinality and
  honest; the raw ``error_reason`` / ``error_source`` strings are deliberately
  dropped as noise.
* **Log-scaled money.**  Amounts are lognormal; ``log1p`` gives the tree evenly
  spaced split points instead of a long right tail.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from config.taxonomy import (
    FailureClass,
    Intervention,
    classify_reason,
    spec,
)

# The interventions the model ever scores: the non-terminal ones.  Terminal
# actions (escalate / give_up) are decisions, not debit/contact attempts, and
# have no recovery probability to predict.
SCORED_INTERVENTIONS: Sequence[Intervention] = tuple(
    i for i in Intervention if not spec(i).terminal
)

# Fixed vocabularies -> stable one-hot column order.
_FAILURE_CLASSES: Sequence[str] = tuple(fc.value for fc in FailureClass)
_METHODS: Sequence[str] = ("card", "netbanking", "upi", "emandate")
_SUB_STATUSES: Sequence[str] = ("active", "paused", "halted", "cancelled", "none")
_INTERVENTIONS: Sequence[str] = tuple(i.value for i in SCORED_INTERVENTIONS)

# Latent trait keys that must NEVER appear in a feature dict.  The test-suite
# asserts none of these is a feature name; kept here so the wall is declared in
# one obvious place.
FORBIDDEN_KEYS: frozenset = frozenset(
    {"engagement", "income_regularity", "base_logodds", "is_truly_dead", "seeded_class"}
)

_NUMERIC_NAMES: List[str] = [
    "log_amount",
    "amount_ratio",
    "prior_retries",
    "customer_tenure_days",
    "log_customer_avg_payment",
    "customer_salary_day",
    "customer_prior_failures",
    "customer_is_subscriber",
    "is_mandate_debit",
    "pre_debit_notified",
]


def _one_hot(value: str, vocabulary: Sequence[str], prefix: str) -> Dict[str, float]:
    return {f"{prefix}={cat}": (1.0 if value == cat else 0.0) for cat in vocabulary}


def _amount_ratio(record: Dict[str, Any]) -> float:
    avg = max(int(record.get("customer_avg_payment_paise", 1)), 1)
    return int(record.get("amount_paise", 0)) / avg


def feature_names() -> List[str]:
    """The ordered feature names.  Stable across processes and splits."""
    names = list(_NUMERIC_NAMES)
    names += [f"class={c}" for c in _FAILURE_CLASSES]
    names += [f"method={m}" for m in _METHODS]
    names += [f"substatus={s}" for s in _SUB_STATUSES]
    names += [f"intervention={i}" for i in _INTERVENTIONS]
    return names


def feature_dict(record: Dict[str, Any], intervention: Intervention) -> Dict[str, float]:
    """Named feature map for one (record, intervention) pair.

    ``record`` may be either a :meth:`FailedRecord.to_json` payload or a
    :meth:`FailedRecord.to_features` payload -- they share every key this reads.
    """
    klass = classify_reason(record.get("error_reason", "")) if record.get(
        "error_reason"
    ) is not None else FailureClass.UNKNOWN

    feats: Dict[str, float] = {
        "log_amount": math.log1p(float(record.get("amount_paise", 0))),
        "amount_ratio": _amount_ratio(record),
        "prior_retries": float(record.get("prior_retries", 0)),
        "customer_tenure_days": float(record.get("customer_tenure_days", 0)),
        "log_customer_avg_payment": math.log1p(
            float(record.get("customer_avg_payment_paise", 0))
        ),
        "customer_salary_day": float(record.get("customer_salary_day", 0)),
        "customer_prior_failures": float(record.get("customer_prior_failures", 0)),
        "customer_is_subscriber": float(int(bool(record.get("customer_is_subscriber", 0)))),
        "is_mandate_debit": float(int(bool(record.get("is_mandate_debit", 0)))),
        "pre_debit_notified": float(int(bool(record.get("pre_debit_notified", False)))),
    }
    feats.update(_one_hot(klass.value, _FAILURE_CLASSES, "class"))
    feats.update(_one_hot(str(record.get("method", "")), _METHODS, "method"))
    feats.update(
        _one_hot(str(record.get("subscription_status", "none")), _SUB_STATUSES, "substatus")
    )
    feats.update(_one_hot(intervention.value, _INTERVENTIONS, "intervention"))
    return feats


def feature_vector(record: Dict[str, Any], intervention: Intervention) -> List[float]:
    """The ordered float vector matching :func:`feature_names`."""
    fd = feature_dict(record, intervention)
    return [fd[name] for name in feature_names()]
