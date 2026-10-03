"""Batch-wide policy comparison, with the learned model scored out-of-sample.

Extends ``measure_policy_gap`` from a rules-vs-oracle gap to a five-policy
ladder, adding the Phase-4 model:

    do_nothing  -- attempt nothing (the floor)
    blind       -- always the class's primary intervention, no economics, no
                   stopping (a value ceiling that ignores discipline)
    rules       -- decide() with the rules _p_recover
    model       -- decide() with the learned _p_recover (SAME decide, SAME
                   guardrails; only the probability estimate changes)
    oracle      -- the environment's best intervention at the true probability

Two honesty rules make this a fair ablation:

* Every policy is scored on the same records, at the same attempt-times, on a
  single attempt, by the *true* environment probability.  Apples to apples.
* The model is **cross-fitted**: each record is scored by a model trained only on
  OTHER customers (GroupKFold on customer_id).  No record is ever scored by a
  model that saw its own customer, so the batch-wide figure cannot be inflated by
  memorisation.

The headline is the rules->model->oracle line: because rules and model differ
*only* in the probability estimate, the value the model adds over rules is
attributable to the estimate and nothing else.

``blind`` will often post a high gross value here -- it never gives up, so it
banks small recoveries the economic stop declines.  That is not a win: the value
metric ignores the contact costs, wasted dead-record debits, and compliance
breaches blind incurs, which the disciplined agent avoids.  Those show up in the
Phase-5 discipline scorecard, not in gross value.

Run:  python -m scripts.compare_policies
"""

from __future__ import annotations

import numpy as np

from config.taxonomy import (
    CANDIDATES,
    PRIMARY_INTERVENTION,
    FailureClass,
    Intervention,
    classify_reason,
    spec,
)
from recoup.agent import control
from recoup.agent.control import decide
from recoup.environment import RecoveryEnvironment
from recoup.generator import BATCH_ANCHOR, generate
from recoup.ml.dataset import attempt_time, build_dataset
from recoup.ml.model import train_model
from recoup.money import format_inr
from sklearn.model_selection import GroupKFold

CALIBRATION = "platt"


def _true_value(env, batch, record, intervention, now):
    if intervention in (Intervention.GIVE_UP, Intervention.ESCALATE):
        return 0
    customer = batch.customers[record.customer_id]
    latent = batch.latent[record.record_id]
    when = attempt_time(intervention, record.customer_salary_day, now)
    p = env.true_prob(record, customer, latent, intervention, when)
    return int(p * record.amount_cents * spec(intervention).recovery_fraction)


def _scorable(record):
    klass = classify_reason(record.error_reason)
    return bool(CANDIDATES.get(klass)) and klass is not FailureClass.UNKNOWN


def _crossfit_model_choices(batch, now):
    """record_id -> chosen intervention, each from a model blind to its customer."""
    records = [r for r in batch.records if _scorable(r)]
    ids = np.array([r.record_id for r in records])
    groups = np.array([r.customer_id for r in records], dtype=object)
    choices = {}
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    dummy_X = np.zeros((len(records), 1))
    for tr, te in gkf.split(dummy_X, np.zeros(len(records)), groups):
        train_customer_ids = set(groups[tr])
        train_record_ids = [
            r.record_id for r in batch.records
            if r.customer_id in train_customer_ids and _scorable(r)
        ]
        ds = build_dataset(batch, now=now, record_ids=train_record_ids)
        model = train_model(ds, calibration=CALIBRATION)
        control.configure_model(model)
        try:
            for i in te:
                rec = records[i]
                choices[rec.record_id] = decide(rec.to_json()).chosen
        finally:
            control.clear_model()
    return choices


def main() -> None:
    batch = generate()
    env = RecoveryEnvironment()
    now = BATCH_ANCHOR
    records = [r for r in batch.records if _scorable(r)]

    # do_nothing
    v_nothing = 0

    # blind: always the class primary, no stopping
    v_blind = sum(
        _true_value(env, batch, r, PRIMARY_INTERVENTION[classify_reason(r.error_reason)], now)
        for r in records
    )

    # rules
    control.clear_model()
    v_rules = sum(_true_value(env, batch, r, decide(r.to_json()).chosen, now) for r in records)

    # model, cross-fitted
    choices = _crossfit_model_choices(batch, now)
    v_model = sum(_true_value(env, batch, r, choices[r.record_id], now) for r in records)

    # oracle
    v_oracle = 0
    for r in records:
        c = batch.customers[r.customer_id]
        l = batch.latent[r.record_id]
        oi, op = env.best_intervention(r, c, l, now)
        v_oracle += int(op * r.amount_cents * spec(oi).recovery_fraction)

    print("Five-policy comparison -- single-attempt recoverable value, whole batch")
    print("=" * 70)
    print(f"records scored   {len(records)}   (model cross-fitted: each record scored")
    print("                 by a model trained only on OTHER customers)")
    print("-" * 70)
    for name, v in [
        ("do_nothing", v_nothing),
        ("blind", v_blind),
        ("rules", v_rules),
        ("model", v_model),
        ("oracle", v_oracle),
    ]:
        pct = v / v_oracle if v_oracle else 0.0
        print(f"  {name:11s} {format_inr(v):>14s}   {pct:6.1%} of oracle")
    print("-" * 70)
    if v_oracle > v_rules:
        closed = (v_model - v_rules) / (v_oracle - v_rules)
        print(f"rules gap to oracle : {(v_oracle - v_rules) / v_oracle:5.1%}")
        print(f"model gap to oracle : {(v_oracle - v_model) / v_oracle:5.1%}")
        print(f"headroom closed     : {closed:5.1%} of the rules->oracle gap")
    print("-" * 70)
    print("rules vs model differ ONLY in _p_recover; the delta is the estimate.")
    print("blind's gross value ignores the compliance + effort costs it incurs.")


if __name__ == "__main__":
    main()
