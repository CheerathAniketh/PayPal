"""Build the labelled training table from the batch and the hidden environment.

This is the *only* module in the ML package that touches ``LatentTruth`` or the
``RecoveryEnvironment``.  It produces labels; it never leaks the parameters that
generated them into ``X``.  Features come from :mod:`recoup.ml.features`, which
can only see observable data; labels come from the environment here.  The wall
is that these are two different function calls on two different objects.

What a training row is
----------------------
One row per (record, candidate intervention) -- an **all-arms potential-outcomes
table**.  For each record we ask the environment, for every non-terminal
intervention its class offers, "would the money have come back?"  This gives the
model a fair counterfactual: it sees each arm's outcome and can learn *which*
intervention wins for *which* customer, not merely re-rank one queue.

Why label at the agent's real attempt-time
------------------------------------------
The environment's probability depends on *when* the agent acts, but the Phase-4
hook ``_p_recover(record, intervention)`` receives no clock.  So we label each
arm at exactly the time the agent would act on it: a salary-window retry is
labelled at the next high-liquidity day, everything else at ``now``.  Train-time
and inference-time timing therefore match, and the model learns the timing
effect implicitly through ``(intervention, salary_day)`` -- which is the whole
reason a tree model that captures interactions is the right tool here.

Production note (stated plainly in the README)
----------------------------------------------
A real system never observes all arms -- only the one it played.  There you would
fit on logged bandit feedback under an exploration policy and correct the
confounding with IPS / doubly-robust estimation.  The all-arms table is the
clean synthetic analogue; it isolates whether the *decision pipeline* is correct
from the separate, harder question of off-policy estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Tuple

import numpy as np

from config.taxonomy import CANDIDATES, FailureClass, Intervention, classify_reason, spec
from recoup.clock import SimulatedClock
from recoup.environment import RecoveryEnvironment
from recoup.generator import Batch
from recoup.ml.features import feature_names, feature_vector


def attempt_time(intervention: Intervention, salary_day: int, now: datetime) -> datetime:
    """When the agent would actually run ``intervention``.

    Canonical definition, shared by the trainer and the policy comparison so the
    two never drift.  A salary-window retry lands on the next high-liquidity day;
    every other action runs now.
    """
    if intervention is Intervention.RETRY_SALARY_WINDOW:
        return SimulatedClock(now).next_high_liquidity_day(salary_day, 24.0)
    return now


@dataclass
class Dataset:
    """A materialised training table, leakage-safe by construction."""

    X: np.ndarray                 # (n_rows, n_features) float
    y: np.ndarray                 # (n_rows,) binary
    groups: np.ndarray            # (n_rows,) customer_id -> grouped CV
    record_ids: List[str]
    interventions: List[Intervention]
    feature_names: List[str]

    def __len__(self) -> int:
        return int(self.X.shape[0])


def build_dataset(
    batch: Batch,
    *,
    now: datetime,
    env: RecoveryEnvironment | None = None,
    record_ids: Sequence[str] | None = None,
) -> Dataset:
    """Materialise the all-arms table for the given records.

    ``record_ids`` restricts to a subset (e.g. the training customers); ``None``
    uses the whole batch.  Labels are sampled from ``env`` -- deterministic given
    the batch seed, so the table is reproducible.
    """
    env = env or RecoveryEnvironment()
    wanted = set(record_ids) if record_ids is not None else None

    names = feature_names()
    rows: List[List[float]] = []
    labels: List[int] = []
    groups: List[str] = []
    rec_ids: List[str] = []
    interventions: List[Intervention] = []

    for record in batch.records:
        if wanted is not None and record.record_id not in wanted:
            continue
        klass = classify_reason(record.error_reason)
        if klass is FailureClass.UNKNOWN:
            continue
        customer = batch.customers[record.customer_id]
        latent = batch.latent[record.record_id]
        rec_json = record.to_json()

        for intervention in CANDIDATES.get(klass, ()):
            if spec(intervention).terminal:
                continue
            when = attempt_time(intervention, record.customer_salary_day, now)
            recovered = env.sample_outcome(
                record, customer, latent, intervention, when, attempt_number=1
            )
            rows.append(feature_vector(rec_json, intervention))
            labels.append(int(recovered))
            groups.append(record.customer_id)
            rec_ids.append(record.record_id)
            interventions.append(intervention)

    return Dataset(
        X=np.asarray(rows, dtype=float),
        y=np.asarray(labels, dtype=int),
        groups=np.asarray(groups, dtype=object),
        record_ids=rec_ids,
        interventions=interventions,
        feature_names=names,
    )
