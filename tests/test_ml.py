"""Phase 4 -- the learned propensity model.

The tests that matter here are not "does LightGBM run" but the properties the ML
story depends on: the leakage wall holds, splits are grouped, the calibrated
score is a probability, and the learned estimate actually beats the rules
estimate on customers it never trained on.  If any of those breaks, the Phase-4
claim is theatre.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from config.taxonomy import CANDIDATES, Intervention, classify_reason, spec
from recoup.agent import control
from recoup.agent.control import decide
from recoup.environment import RecoveryEnvironment
from recoup.generator import BATCH_ANCHOR, generate
from recoup.ml.dataset import attempt_time, build_dataset
from recoup.ml.features import (
    FORBIDDEN_KEYS,
    SCORED_INTERVENTIONS,
    feature_dict,
    feature_names,
    feature_vector,
)
from recoup.ml.model import PropensityModel, train_model


@pytest.fixture(scope="module")
def batch():
    return generate()


@pytest.fixture(scope="module")
def trained(batch):
    train_ids = [r.record_id for r in batch.train_records]
    ds = build_dataset(batch, now=BATCH_ANCHOR, record_ids=train_ids)
    return train_model(ds, calibration="platt")


@pytest.fixture(autouse=True)
def _isolate_control():
    """No test may leak a configured model into another."""
    control.clear_model()
    yield
    control.clear_model()


# --------------------------------------------------------------------------
# The leakage wall
# --------------------------------------------------------------------------
def test_no_latent_trait_is_a_feature_name():
    names = feature_names()
    for name in names:
        for forbidden in FORBIDDEN_KEYS:
            assert forbidden not in name, f"latent trait {forbidden!r} leaked into {name!r}"


def test_feature_dict_never_emits_latent_keys(batch):
    record = batch.records[0].to_json()
    fd = feature_dict(record, Intervention.RETRY_NOW)
    assert FORBIDDEN_KEYS.isdisjoint(fd.keys())


def test_feature_dict_ignores_injected_latent_fields(batch):
    """Even if a latent trait is smuggled into the record dict, it is not read."""
    record = batch.records[0].to_json()
    clean = feature_vector(dict(record), Intervention.RETRY_NOW)
    poisoned = dict(record)
    poisoned["engagement"] = 0.999
    poisoned["income_regularity"] = 0.001
    poisoned["base_logodds"] = 5.0
    assert feature_vector(poisoned, Intervention.RETRY_NOW) == clean


# --------------------------------------------------------------------------
# Feature vector shape / determinism
# --------------------------------------------------------------------------
def test_feature_vector_matches_names_length(batch):
    record = batch.records[0].to_json()
    assert len(feature_vector(record, Intervention.RETRY_NOW)) == len(feature_names())


def test_feature_vector_is_deterministic(batch):
    record = batch.records[3].to_json()
    a = feature_vector(record, Intervention.RETRY_SALARY_WINDOW)
    b = feature_vector(record, Intervention.RETRY_SALARY_WINDOW)
    assert a == b


def test_intervention_changes_the_vector(batch):
    record = batch.records[0].to_json()
    a = feature_vector(record, Intervention.RETRY_NOW)
    b = feature_vector(record, Intervention.RETRY_SALARY_WINDOW)
    assert a != b


def test_scored_interventions_are_non_terminal():
    for i in SCORED_INTERVENTIONS:
        assert not spec(i).terminal


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------
def test_train_and_holdout_customers_are_disjoint(batch):
    train_ids = [r.record_id for r in batch.train_records]
    hold_ids = [r.record_id for r in batch.holdout_records]
    ds_tr = build_dataset(batch, now=BATCH_ANCHOR, record_ids=train_ids)
    ds_ho = build_dataset(batch, now=BATCH_ANCHOR, record_ids=hold_ids)
    assert set(ds_tr.groups).isdisjoint(set(ds_ho.groups))


def test_dataset_is_all_arms(batch):
    """Every non-terminal candidate of a scorable record's class appears."""
    ds = build_dataset(batch, now=BATCH_ANCHOR)
    by_record = {}
    for rid, interv in zip(ds.record_ids, ds.interventions):
        by_record.setdefault(rid, set()).add(interv)
    # pick a record and check its arm set matches the taxonomy
    sample = ds.record_ids[0]
    rec = next(r for r in batch.records if r.record_id == sample)
    klass = classify_reason(rec.error_reason)
    expected = {i for i in CANDIDATES[klass] if not spec(i).terminal}
    assert by_record[sample] == expected


# --------------------------------------------------------------------------
# Probability contract
# --------------------------------------------------------------------------
def test_predict_proba_in_unit_interval(batch, trained):
    record = batch.holdout_records[0].to_json()
    for interv in SCORED_INTERVENTIONS:
        p = trained.predict_proba(record, interv)
        assert 0.0 <= p <= 1.0


def test_batch_and_single_predictions_agree(batch, trained):
    ds = build_dataset(batch, now=BATCH_ANCHOR, record_ids=[batch.holdout_records[0].record_id])
    batch_p = trained.predict_proba_batch(ds.X)
    single_p = [
        trained.predict_proba(
            next(r for r in batch.records if r.record_id == rid).to_json(), interv
        )
        for rid, interv in zip(ds.record_ids, ds.interventions)
    ]
    assert np.allclose(batch_p, single_p, atol=1e-9)


# --------------------------------------------------------------------------
# The claim: the learned estimate beats the rules estimate out-of-sample
# --------------------------------------------------------------------------
def _holdout_value(batch, chooser):
    env = RecoveryEnvironment()
    now = BATCH_ANCHOR
    total = 0
    for rec in batch.holdout_records:
        klass = classify_reason(rec.error_reason)
        if not CANDIDATES.get(klass):
            continue
        interv = chooser(rec)
        if interv in (Intervention.GIVE_UP, Intervention.ESCALATE):
            continue
        c = batch.customers[rec.customer_id]
        l = batch.latent[rec.record_id]
        when = attempt_time(interv, rec.customer_salary_day, now)
        p = env.true_prob(rec, c, l, interv, when)
        total += int(p * rec.amount_paise * spec(interv).recovery_fraction)
    return total


def test_model_recovers_more_than_rules_on_holdout(batch, trained):
    control.clear_model()
    rules_value = _holdout_value(batch, lambda rec: decide(rec.to_json()).chosen)

    control.configure_model(trained)
    model_value = _holdout_value(batch, lambda rec: decide(rec.to_json()).chosen)
    control.clear_model()

    assert model_value > rules_value


# --------------------------------------------------------------------------
# The wiring: opt-in, reversible, and it actually routes through the model
# --------------------------------------------------------------------------
def test_configure_and_clear_model_toggles_the_hook(trained):
    assert not control.using_model()
    control.configure_model(trained)
    assert control.using_model()
    control.clear_model()
    assert not control.using_model()


def test_model_changes_at_least_one_decision(batch, trained):
    control.clear_model()
    rules_choices = {r.record_id: decide(r.to_json()).chosen for r in batch.holdout_records}
    control.configure_model(trained)
    model_choices = {r.record_id: decide(r.to_json()).chosen for r in batch.holdout_records}
    control.clear_model()
    assert rules_choices != model_choices


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def test_save_load_round_trip(batch, trained, tmp_path):
    path = trained.save(tmp_path / "m.joblib")
    reloaded = PropensityModel.load(path)
    record = batch.holdout_records[0].to_json()
    for interv in SCORED_INTERVENTIONS:
        assert reloaded.predict_proba(record, interv) == pytest.approx(
            trained.predict_proba(record, interv), abs=1e-9
        )
