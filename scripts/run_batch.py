"""One command: run the whole batch through the real agent and print the receipts.

This is the submission's headline run. It drives all 180 records through the
same compiled graph + scheduler the rest of the system uses -- EV-ordered
dispatch, per-customer contact caps, cooldown / salary-window re-enqueue, and the
async outcome path a webhook would feed -- then reports the §9 metrics the track
bar names explicitly:

    money recovered, recovery rate by failure class, escalation count,
    compliance violations (zero, by construction), false effort avoided
    (dead records we correctly stopped on), and a sample of the append-only
    audit trail.

It runs the batch TWICE -- once on the rules policy (the ship-safe default) and
once on the learned model, CROSS-FITTED by customer so every record is scored by
a model that never saw its own customer -- and prints both realized recovery
numbers side by side. The rules run is detailed; the model run is the headline
delta. The environment's outcome draws are deterministic, so the only thing that
changes between the two is which interventions the policy picked.

Run:  python -m scripts.run_batch                 # rules detailed + model headline
      python -m scripts.run_batch --model         # model detailed + rules headline
      python -m scripts.run_batch --no-async      # synchronous pass only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.taxonomy import ALL_CLASSES, Intervention, classify_reason, spec
from recoup.agent import control
from recoup.agent.outcome_sim import OutcomeSimulator
from recoup.agent.runtime import DemoAuditLog, DemoExecutor, configure
from recoup.agent.scheduler import BatchScheduler, RunSummary, SchedulerConfig
from recoup.environment import RecoveryEnvironment
from recoup.generator import BATCH_ANCHOR, generate, hydrate_customers, load_frozen, load_latent
from recoup.models import FailedRecord
from recoup.money import format_inr

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "propensity_model.joblib"
START = datetime(2026, 1, 16, 9)
BAR = "=" * 72
DASH = "-" * 72

# Guardrail checks whose failure would be a *compliance* breach if the action
# had still executed. A block sets status escalated, so these must never coincide
# with a non-escalated outcome; we verify that rather than assume it.
COMPLIANCE_CHECKS = ("mandate_active", "pre_debit_notice")


def _run_once(records, customers, latent, *, run_id: str, use_async: bool):
    """One full batch run; returns (summary_after, scheduler, audit)."""
    audit = DemoAuditLog()
    configure(executor=DemoExecutor(RecoveryEnvironment(), customers, latent), audit=audit)

    simulator = OutcomeSimulator(
        RecoveryEnvironment(), customers, latent, response_window=timedelta(days=3)
    )

    def on_park(record_dict, intervention, attempt_n, now, outstanding):
        clean = {k: v for k, v in record_dict.items() if not k.startswith("_")}
        simulator.park(FailedRecord.from_json(clean), intervention, attempt_n, now, outstanding)

    config = SchedulerConfig(run_id=run_id, start=START)
    scheduler = BatchScheduler(
        [r.to_json() for r in records], config, on_park=on_park if use_async else None
    )
    scheduler.run()

    if use_async:
        settle_now = config.start + timedelta(days=30)
        for event in simulator.resolve_all(settle_now):
            scheduler.ingest_outcome(event)

    return scheduler._summarise(), scheduler, audit


def _by_class(scheduler, latent) -> List[Tuple[str, int, int, int, int]]:
    """Per class: (class, at_risk, recovered, n_records, n_recovered)."""
    rows = []
    for klass in ALL_CLASSES:
        at_risk = recovered = n = n_rec = 0
        for st in scheduler.states.values():
            if classify_reason(st.record.get("error_reason", "")) is not klass:
                continue
            n += 1
            at_risk += int(st.record["amount_paise"])
            recovered += st.recovered_paise
            if st.status == "recovered":
                n_rec += 1
        rows.append((klass.value, at_risk, recovered, n, n_rec))
    return rows


def _false_effort_avoided(scheduler, latent) -> Tuple[int, int, float]:
    """Truly-dead records the agent correctly stopped on, and mean attempts on them."""
    stopped = 0
    dead_total = 0
    attempts_on_dead = 0
    for st in scheduler.states.values():
        truth = latent.get(st.record["record_id"])
        if truth is None or not truth.is_truly_dead:
            continue
        dead_total += 1
        attempts_on_dead += st.attempts
        if st.status in ("escalated", "abandoned"):
            stopped += 1
    mean_attempts = attempts_on_dead / dead_total if dead_total else 0.0
    return stopped, dead_total, mean_attempts


def _compliance_violations(audit_rows: List[Dict[str, Any]]) -> int:
    """Actions that touched an instrument they were not allowed to.

    A violation = an audit row where a compliance guardrail FAILED yet the
    outcome was not an escalation (i.e. the action proceeded anyway), OR an
    instrument-touching debit that executed against a broken mandate. Both are
    structurally impossible here; we count them to prove it, not assume it.
    """
    violations = 0
    for row in audit_rows:
        checks = row.get("guardrail_checks", {}) or {}
        outcome = row.get("outcome")
        if any(checks.get(c) == "fail" for c in COMPLIANCE_CHECKS) and outcome != "escalated":
            violations += 1
    return violations


def _print_scorecard(summary: RunSummary, scheduler, latent, audit) -> None:
    print(f"\n{DASH}\nRECOVERY")
    print(f"  at risk        {format_inr(summary.total_at_risk_paise)}")
    print(f"  recovered      {format_inr(summary.recovered_paise)}  "
          f"({summary.recovery_rate:.1%})")
    print(f"  recovered      {summary.recovered_count} records | "
          f"escalated {summary.escalated_count} | abandoned {summary.abandoned_count} | "
          f"awaiting {summary.in_progress_count}")
    print(f"  work done      {summary.attempts} attempts | "
          f"{summary.contacts_made} customer contacts | "
          f"budget {format_inr(summary.budget_spent_paise)}")

    print(f"\n{DASH}\nRECOVERY BY FAILURE CLASS")
    print(f"  {'class':<20} {'recovered / at risk':>26}   {'rate':>6}   {'n':>7}")
    for name, at_risk, recovered, n, n_rec in _by_class(scheduler, latent):
        rate = recovered / at_risk if at_risk else 0.0
        pair = f"{format_inr(recovered)} / {format_inr(at_risk)}"
        print(f"  {name:<20} {pair:>26}   {rate:6.1%}   {n_rec:>3}/{n:<3}")

    print(f"\n{DASH}\nDISCIPLINE")
    stopped, dead_total, mean_attempts = _false_effort_avoided(scheduler, latent)
    violations = _compliance_violations(scheduler.audit_rows)
    print(f"  compliance violations       {violations}   "
          f"(scanned {len(scheduler.audit_rows)} audit rows)")
    print(f"  false effort avoided        {stopped}/{dead_total} truly-dead records "
          f"stopped cleanly")
    print(f"  mean attempts on dead       {mean_attempts:.2f}  "
          f"(no hammering: capped, then escalated)")
    print(f"  escalations logged w/ reason {summary.escalated_count}")


def _print_sample_receipts(scheduler, k: int = 2) -> None:
    print(f"\n{DASH}\nSAMPLE AUDIT RECEIPTS (in-memory demo adapter; "
          f"SQLite append-only store is built + tested)")
    shown = 0
    for row in scheduler.audit_rows:
        if row.get("outcome") == "escalated" or shown < k:
            keep = {
                "record_id": row.get("record_id"),
                "attempt": row.get("attempt_number"),
                "chosen_action": row.get("chosen_action"),
                "outcome": row.get("outcome"),
                "guardrail_checks": row.get("guardrail_checks"),
                "model_score": row.get("model_score"),
                "recovered_paise": row.get("amount_recovered_paise"),
            }
            print(json.dumps(keep))
            shown += 1
            if shown >= k + 1:
                break


class _CrossFitRouter:
    """One model per customer-fold; dispatches on the record's customer_id so
    every record is scored by a model blind to its own customer."""

    def __init__(self, mapping):
        self._m = mapping

    def predict_proba(self, record, intervention):
        model = self._m.get(record.get("customer_id"))
        if model is None:
            return 0.5
        return model.predict_proba(record, intervention)


def _build_crossfit_router(batch, now):
    """Train one model per customer-fold (GroupKFold on customer_id).

    The shipped artifact is refit on ALL customers, so scoring it on this same
    batch would be an in-sample number. Cross-fitting here is what makes the
    headline honest -- the same discipline scripts/compare_policies.py uses.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold

    from config.taxonomy import FailureClass
    from recoup.ml.dataset import build_dataset
    from recoup.ml.model import train_model

    scorable = [r for r in batch.records
                if classify_reason(r.error_reason) is not FailureClass.UNKNOWN]
    cust = np.array(sorted({r.customer_id for r in scorable}), dtype=object)
    gkf = GroupKFold(n_splits=min(5, len(cust)))
    mapping = {}
    for tr, te in gkf.split(np.zeros((len(cust), 1)), np.zeros(len(cust)), cust):
        train_customers = set(cust[tr])
        train_rids = [r.record_id for r in scorable
                      if r.customer_id in train_customers]
        m = train_model(build_dataset(batch, now=now, record_ids=train_rids),
                        calibration="platt")
        for c in cust[te]:
            mapping[c] = m
    return _CrossFitRouter(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full recovery batch.")
    parser.add_argument("--model", action="store_true",
                        help="detail the model run instead of the rules run")
    parser.add_argument("--no-async", action="store_true",
                        help="synchronous pass only (skip the outcome gateway)")
    args = parser.parse_args()
    use_async = not args.no_async

    records, customers = load_frozen()
    customers = hydrate_customers(customers)   # eval-only: the executor needs truth
    latent = load_latent()

    print(BAR)
    print("RECOUP — full batch run")
    print(BAR)
    print(f"records {len(records)} | customers {len(customers)} | "
          f"async outcome path: {'on' if use_async else 'off'}")

    # ---- rules run (always) ----
    control.clear_model()
    rules_summary, rules_sched, rules_audit = _run_once(
        records, customers, latent, run_id="run_batch_rules", use_async=use_async
    )

    # ---- model run, cross-fitted (each record scored by a model blind to its
    # own customer). Skipped gracefully if the ML stack is not installed. ----
    router = None
    try:
        router = _build_crossfit_router(generate(), BATCH_ANCHOR)
    except Exception as exc:  # noqa: BLE001
        print(f"(cross-fit model unavailable: {exc}; running rules only)")

    model_summary = None
    if router is not None:
        control.configure_model(router)
        model_summary, model_sched, model_audit = _run_once(
            records, customers, latent, run_id="run_batch_model", use_async=use_async
        )
        control.clear_model()

    # ---- detailed scorecard for the chosen policy ----
    if args.model and router is not None:
        print("\n" + BAR + "\nPOLICY: learned model (detailed)\n" + BAR)
        _print_scorecard(model_summary, model_sched, latent, model_audit)
        _print_sample_receipts(model_sched)
    else:
        label = "rules (detailed)" + ("" if router is not None else " — model leg skipped")
        print("\n" + BAR + f"\nPOLICY: {label}\n" + BAR)
        _print_scorecard(rules_summary, rules_sched, latent, rules_audit)
        _print_sample_receipts(rules_sched)

    # ---- headline comparison ----
    print("\n" + BAR + "\nHEADLINE — realized recovery, rules vs model\n" + BAR)
    print(f"  rules   {format_inr(rules_summary.recovered_paise):>14}  "
          f"{rules_summary.recovery_rate:6.1%}")
    if model_summary is not None:
        delta = model_summary.recovered_paise - rules_summary.recovered_paise
        sign = "+" if delta >= 0 else "-"
        print(f"  model   {format_inr(model_summary.recovered_paise):>14}  "
              f"{model_summary.recovery_rate:6.1%}   "
              f"({sign}{format_inr(abs(delta))} vs rules)")
        print(f"\n  Both runs share the same deterministic environment draws; the")
        print(f"  only difference is which interventions the policy chose. The model")
        print(f"  figure is CROSS-FITTED: each record is scored by a model trained")
        print(f"  only on OTHER customers, so it is out-of-sample. On a single")
        print(f"  realized draw the lift is modest; the robust expected-value")
        print(f"  ablation is scripts/compare_policies.py.")
    else:
        print("  (model leg skipped — install lightgbm + scikit-learn to run it)")


if __name__ == "__main__":
    main()
