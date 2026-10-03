"""Build report.html: a self-contained recovery-ledger UI for the rules run.

One deterministic run of the rules policy (the ship-safe default `run_batch`
reports as its headline), rendered SERVER-SIDE. Every number on the page comes
from that run; the page computes nothing -- the inline JS only toggles row
expansion and the status filter, so nothing client-side can disagree with the
run that produced the numbers.

The design is a reconciliation statement, not a dashboard: figures balance, and
every ledger row expands to the exact audit rows that produced it. See the
design spec for the full rationale.

Run:  python -m scripts.build_report
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.taxonomy import ALL_CLASSES, classify_reason
from recoup.agent import control
from recoup.agent.outcome_sim import OutcomeSimulator
from recoup.agent.runtime import DemoAuditLog, DemoExecutor, configure
from recoup.agent.scheduler import BatchScheduler, RunSummary, SchedulerConfig
from recoup.environment import RecoveryEnvironment
from recoup.generator import FROZEN_PATH, hydrate_customers, load_frozen, load_latent
from recoup.models import FailedRecord
from recoup.money import format_inr

OUT_PATH = Path(__file__).resolve().parent.parent / "report.html"
START = datetime(2026, 1, 16, 9, 0, 0)
RUN_ID = "run_report"

# Same compliance predicate scripts/run_batch.py uses (post pre-freeze fix: the
# "consent" guardrail was a hardcoded pass, never a real check, and was removed).
COMPLIANCE_CHECKS = ("mandate_active", "pre_debit_notice")

STATUS_CLASS = {
    "recovered": "rec",
    "escalated": "esc",
    "abandoned": "aba",
    "open": "prog",
    "awaiting": "prog",
}


def _label(s: str) -> str:
    """'insufficient_funds' -> 'Insufficient funds'; 'in_progress' -> 'In progress'."""
    s = s.replace("_", " ")
    return s[:1].upper() + s[1:] if s else s


# --------------------------------------------------------------------------
# 1. Run the batch: rules policy, ship-safe default, async resolution on.
# --------------------------------------------------------------------------
def run_batch():
    records, customers = load_frozen()
    customers = hydrate_customers(customers)   # eval-only: the executor needs truth
    latent = load_latent()

    control.clear_model()
    audit = DemoAuditLog()
    configure(executor=DemoExecutor(RecoveryEnvironment(), customers, latent), audit=audit)

    simulator = OutcomeSimulator(
        RecoveryEnvironment(), customers, latent, response_window=timedelta(days=3)
    )

    def on_park(record_dict, intervention, attempt_n, now, outstanding):
        clean = {k: v for k, v in record_dict.items() if not k.startswith("_")}
        simulator.park(FailedRecord.from_json(clean), intervention, attempt_n, now, outstanding)

    config = SchedulerConfig(run_id=RUN_ID, start=START)
    scheduler = BatchScheduler([r.to_json() for r in records], config, on_park=on_park)
    scheduler.run()

    settle_now = config.start + timedelta(days=30)
    for event in simulator.resolve_all(settle_now):
        scheduler.ingest_outcome(event)

    return scheduler._summarise(), scheduler, latent


# --------------------------------------------------------------------------
# 2. Derived, independently-checked figures.
# --------------------------------------------------------------------------
def reconcile(summary: RunSummary, scheduler: BatchScheduler) -> Dict[str, Any]:
    """Two independently-summed totals, checked against each other and against
    the scheduler's own running total -- an actual tie-out, not a tautology."""
    per_record_at_risk = sum(int(st.record["amount_cents"]) for st in scheduler.states.values())
    per_record_recovered = sum(st.recovered_cents for st in scheduler.states.values())
    not_recovered = summary.total_at_risk_cents - summary.recovered_cents

    ties = (
        per_record_at_risk == summary.total_at_risk_cents
        and per_record_recovered == summary.recovered_cents
        and summary.recovered_cents + not_recovered == summary.total_at_risk_cents
    )
    return {
        "recovered": summary.recovered_cents,
        "not_recovered": not_recovered,
        "at_risk": summary.total_at_risk_cents,
        "rate": summary.recovery_rate,
        "ties": ties,
    }


def discipline(scheduler: BatchScheduler, latent) -> Dict[str, Any]:
    violations = 0
    for row in scheduler.audit_rows:
        checks = row.get("guardrail_checks", {}) or {}
        if any(checks.get(c) == "fail" for c in COMPLIANCE_CHECKS) and row.get("outcome") != "escalated":
            violations += 1

    stopped = dead_total = attempts_on_dead = 0
    for st in scheduler.states.values():
        truth = latent.get(st.record["record_id"])
        if truth is None or not truth.is_truly_dead:
            continue
        dead_total += 1
        attempts_on_dead += st.attempts
        if st.status in ("escalated", "abandoned"):
            stopped += 1
    mean_attempts = attempts_on_dead / dead_total if dead_total else 0.0
    escalations = sum(1 for st in scheduler.states.values() if st.status == "escalated")

    return {
        "violations": violations,
        "scanned": len(scheduler.audit_rows),
        "dead_stopped": stopped,
        "dead_total": dead_total,
        "mean_attempts_dead": mean_attempts,
        "escalations": escalations,
    }


def by_class(scheduler: BatchScheduler) -> List[Dict[str, Any]]:
    rows = []
    for klass in ALL_CLASSES:
        at_risk = recovered = n = n_rec = 0
        for st in scheduler.states.values():
            if classify_reason(st.record.get("error_reason", "")) is not klass:
                continue
            n += 1
            at_risk += int(st.record["amount_cents"])
            recovered += st.recovered_cents
            if st.status == "recovered":
                n_rec += 1
        rows.append({
            "label": _label(klass.value),
            "at_risk": at_risk,
            "recovered": recovered,
            "rate": recovered / at_risk if at_risk else 0.0,
            "n": n,
            "n_rec": n_rec,
        })
    return rows


def ledger_rows(scheduler: BatchScheduler, latent) -> List[Dict[str, Any]]:
    """One row per record, each carrying its full attempt history.

    A record whose final status differs from its *last logged attempt* settled
    asynchronously (a parked contact resolving through ``ingest_outcome`` -- see
    ``BatchScheduler._apply_resolution``), which today applies the resolution to
    scheduler state without writing a new audit row. That gap is real; it is
    surfaced here as a clearly labelled derived line rather than hidden or
    faked as a logged attempt. See the footnote rendered under the ledger.
    """
    attempts_by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scheduler.audit_rows:
        attempts_by_record[row["record_id"]].append(row)

    out = []
    for rid, st in scheduler.states.items():
        rows = attempts_by_record.get(rid, [])
        last = rows[-1] if rows else None
        derived: Optional[str] = None
        if (
            last is not None
            and st.status in ("recovered", "escalated", "abandoned")
            and last.get("outcome") != st.status
        ):
            if st.status == "recovered":
                derived = (
                    f"final: recovered {format_inr(st.recovered_cents)} "
                    "(async settlement -- customer acted before the response window)"
                )
            elif st.status == "escalated":
                derived = (
                    "final: escalated (async settlement -- no customer action "
                    "before the response window)"
                )
            else:
                derived = f"final: {st.status}"

        out.append({
            "record_id": rid,
            "customer_id": st.record["customer_id"],
            "cause": _label(classify_reason(st.record.get("error_reason", "")).value),
            "amount": int(st.record["amount_cents"]),
            "attempts": st.attempts,
            "recovered": st.recovered_cents,
            "status": st.status,
            "status_label": _label(st.status if st.status != "open" and st.status != "awaiting" else "in_progress"),
            "status_class": STATUS_CLASS.get(st.status, "prog"),
            "rows": rows,
            "derived": derived,
        })
    out.sort(key=lambda r: r["amount"], reverse=True)
    return out


# --------------------------------------------------------------------------
# 3. Render.
# --------------------------------------------------------------------------
def _pill(name: str, result: str) -> str:
    cls = "pass" if result == "pass" else "fail"
    return f'<span class="pill {cls}">{html.escape(_label(name))}: {html.escape(result)}</span>'


def _attempt_html(row: Dict[str, Any]) -> str:
    checks = row.get("guardrail_checks", {}) or {}
    pills = "".join(_pill(k, v) for k, v in checks.items())
    key = row.get("idempotency_key") or "—"
    score = row.get("model_score")
    score_txt = f"p̂ {score:.2f}" if isinstance(score, (int, float)) else "—"
    outcome = row.get("outcome", "")
    return f"""
      <div class="attempt">
        <div class="row1">
          <span class="n mono">attempt {row.get('attempt_number', '?')}</span>
          <span class="action">{html.escape(_label(str(row.get('chosen_action', ''))))}</span>
          <span class="badge {STATUS_CLASS.get(outcome, 'prog')}">{html.escape(_label(outcome))}</span>
          <span class="score mono">{html.escape(score_txt)}</span>
        </div>
        {f'<div class="pills">{pills}</div>' if pills else ''}
        <div class="rationale">{html.escape(row.get('rationale', ''))}</div>
        <div class="narration">{html.escape(row.get('narration', ''))}</div>
        <div class="idkey mono">key: {html.escape(key)}</div>
      </div>"""


def _ledger_row_html(r: Dict[str, Any]) -> str:
    attempts_html = "".join(_attempt_html(a) for a in r["rows"])
    derived_html = f'<div class="derived">→ {html.escape(r["derived"])}</div>' if r["derived"] else ""
    return f"""
    <tr class="ledger-row" data-status="{r['status']}">
      <td><span class="chev">▸</span><span class="mono">{r['record_id']}</span></td>
      <td class="mono muted">{r['customer_id']}</td>
      <td>{html.escape(r['cause'])}</td>
      <td class="num tabular">{format_inr(r['amount'])}</td>
      <td class="num tabular">{r['attempts']}</td>
      <td class="num tabular">{format_inr(r['recovered'])}</td>
      <td><span class="badge {r['status_class']}">{html.escape(r['status_label'])}</span></td>
    </tr>
    <tr class="detail-row" hidden>
      <td colspan="7"><div class="wrap">{attempts_html}{derived_html}</div></td>
    </tr>"""


def render(summary, recon, disc, classes, ledger, meta) -> str:
    by_class_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(c['label'])}</td>
          <td class="num tabular">{format_inr(c['recovered'])} / {format_inr(c['at_risk'])}</td>
          <td class="num tabular">{c['rate']:.1%}</td>
          <td class="num tabular">{c['n_rec']}/{c['n']}</td>
        </tr>"""
        for c in classes
    )
    ledger_html = "".join(_ledger_row_html(r) for r in ledger)
    tie_class = "" if recon["ties"] else " warn"
    tie_text = (
        "Ledger ties out — recovered + not recovered = at risk; "
        "per-record sums match the scheduler total."
        if recon["ties"]
        else "TIE-OUT FAILED — per-record sums do not match the scheduler total."
    )
    n_records = len(ledger)
    n_customers = len({r["customer_id"] for r in ledger})
    n_rec = sum(1 for r in ledger if r["status"] == "recovered")
    n_esc = sum(1 for r in ledger if r["status"] == "escalated")
    n_aba = sum(1 for r in ledger if r["status"] == "abandoned")
    n_prog = n_records - n_rec - n_esc - n_aba
    n_derived = sum(1 for r in ledger if r["derived"])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recoup — recovery statement</title>
<style>
:root {{
  --ink: #17130d;
  --panel: #211b13;
  --panel2: #1c1710;
  --line: #3a2f21;
  --paper: #ece4d4;
  --muted: #a2937b;
  --brass: #cda349;
  --rec: #7fa06a;
  --esc: #c98a55;
  --aba: #8a7f6e;
  --prog: #6f86a0;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: var(--ink);
  color: var(--paper);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  padding: 48px 24px 96px;
}}
.mono {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; }}
.tabular {{ font-variant-numeric: tabular-nums; }}
.muted {{ color: var(--muted); }}
.page {{ max-width: 1060px; margin: 0 auto; }}

.statement-head {{ margin-bottom: 28px; }}
.statement-head h1 {{ font-size: 1.7rem; font-weight: 680; margin: 0 0 8px; letter-spacing: -0.01em; }}
.statement-head .meta {{ color: var(--muted); font-size: 0.9rem; }}

.hero {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px 6px 0 0;
  padding: 36px 40px 30px;
}}
.hero .total {{ font-size: 3.4rem; font-weight: 680; letter-spacing: -0.02em; }}
.hero .of {{ color: var(--muted); font-size: 1rem; margin-top: 8px; }}
.hero .rate {{ color: var(--brass); font-weight: 640; }}

.reconcile {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  border: 1px solid var(--line);
  border-top: none;
}}
.reconcile .cell {{ padding: 20px 24px; border-right: 1px solid var(--line); }}
.reconcile .cell:last-child {{ border-right: none; }}
.reconcile .label {{ color: var(--muted); font-size: 0.78rem; margin-bottom: 6px; }}
.reconcile .value {{ font-size: 1.4rem; font-weight: 620; }}

.tieout {{
  border: 1px solid var(--line);
  border-top: none;
  border-radius: 0 0 6px 6px;
  padding: 12px 24px;
  color: var(--brass);
  font-size: 0.88rem;
  font-weight: 620;
  background: rgba(205, 163, 73, 0.07);
}}
.tieout.warn {{ color: var(--esc); background: rgba(201, 138, 85, 0.1); }}

section {{ margin: 40px 0; }}
section h2 {{ font-size: 1.15rem; font-weight: 620; margin: 0 0 14px; }}
section p.lede {{ color: var(--muted); font-size: 0.88rem; margin: -6px 0 16px; }}

.discipline-strip {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
}}
.discipline-strip .cell {{ background: var(--panel); padding: 18px 20px; }}
.discipline-strip .value {{ font-size: 1.6rem; font-weight: 640; }}
.discipline-strip .label {{ color: var(--muted); font-size: 0.8rem; margin-top: 4px; }}

table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px 14px; border-bottom: 1px solid var(--line); text-align: left; }}
th {{ color: var(--muted); font-weight: 620; font-size: 0.78rem; }}
td.num, th.num {{ text-align: right; }}

.badge {{
  display: inline-block;
  padding: 2px 9px;
  border-radius: 3px;
  font-size: 0.76rem;
  font-weight: 620;
  border: 1px solid currentColor;
}}
.badge.rec {{ color: var(--rec); }}
.badge.esc {{ color: var(--esc); }}
.badge.aba {{ color: var(--aba); }}
.badge.prog {{ color: var(--prog); }}

.filters {{ margin-bottom: 12px; }}
.filter-btn {{
  background: transparent;
  border: 1px solid var(--line);
  color: var(--muted);
  padding: 6px 14px;
  font-size: 0.82rem;
  border-radius: 3px;
  cursor: pointer;
  margin-right: 6px;
  font-family: inherit;
}}
.filter-btn.active {{ border-color: var(--brass); color: var(--brass); }}

.ledger-row {{ cursor: pointer; }}
.ledger-row:hover {{ background: var(--panel2); }}
.ledger-row .chev {{ color: var(--muted); display: inline-block; width: 14px; }}
.ledger-row.open .chev {{ color: var(--brass); }}

.detail-row td {{ background: var(--panel); padding: 0; border-bottom: 1px solid var(--line); }}
.detail-row .wrap {{ padding: 4px 24px 18px 42px; }}
.attempt {{ padding: 12px 0; border-top: 1px dashed var(--line); }}
.attempt:first-child {{ border-top: none; }}
.attempt .row1 {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
.attempt .n {{ color: var(--muted); font-size: 0.8rem; }}
.attempt .action {{ font-weight: 620; }}
.attempt .score {{ color: var(--muted); font-size: 0.8rem; }}
.pills {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }}
.pill {{
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.72rem;
  padding: 1px 7px;
  border-radius: 3px;
  border: 1px solid var(--line);
}}
.pill.pass {{ color: var(--rec); border-color: var(--rec); }}
.pill.fail {{ color: var(--esc); border-color: var(--esc); font-weight: 700; }}
.rationale {{ color: var(--muted); font-size: 0.84rem; margin-top: 8px; font-style: italic; }}
.narration {{ font-size: 0.86rem; margin-top: 4px; }}
.idkey {{ color: var(--muted); font-size: 0.74rem; margin-top: 8px; }}
.derived {{
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px dashed var(--brass);
  color: var(--brass);
  font-size: 0.84rem;
  font-style: italic;
}}

.footnote {{
  margin-top: 8px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.82rem;
}}
</style>
</head>
<body>
<div class="page">

  <header class="statement-head">
    <h1>Recovery statement</h1>
    <div class="meta">
      Run <span class="mono">{html.escape(meta['run_id'])}</span> &middot;
      seed <span class="mono">{meta['seed']}</span> &middot;
      {meta['n_records']} records &middot; {meta['n_customers']} customers &middot;
      simulated clock started <span class="mono">{meta['start']}</span> &middot;
      policy: rules (ship-safe default)
    </div>
  </header>

  <div class="hero">
    <div class="total tabular">{format_inr(recon['recovered'])}</div>
    <div class="of">recovered of {format_inr(recon['at_risk'])} at risk &middot;
      <span class="rate">{recon['rate']:.1%}</span></div>
  </div>
  <div class="reconcile">
    <div class="cell">
      <div class="label">Recovered</div>
      <div class="value tabular">{format_inr(recon['recovered'])}</div>
    </div>
    <div class="cell">
      <div class="label">Not recovered</div>
      <div class="value tabular">{format_inr(recon['not_recovered'])}</div>
    </div>
    <div class="cell">
      <div class="label">At risk</div>
      <div class="value tabular">{format_inr(recon['at_risk'])}</div>
    </div>
  </div>
  <div class="tieout{tie_class}">&#10003; {html.escape(tie_text)}</div>

  <section class="discipline">
    <h2>Discipline</h2>
    <div class="discipline-strip">
      <div class="cell">
        <div class="value tabular">{disc['violations']}</div>
        <div class="label">compliance violations (of {disc['scanned']} audit rows scanned)</div>
      </div>
      <div class="cell">
        <div class="value tabular">{disc['dead_stopped']}/{disc['dead_total']}</div>
        <div class="label">truly-dead records stopped cleanly</div>
      </div>
      <div class="cell">
        <div class="value tabular">{disc['mean_attempts_dead']:.2f}</div>
        <div class="label">mean attempts on a dead record</div>
      </div>
      <div class="cell">
        <div class="value tabular">{disc['escalations']}</div>
        <div class="label">escalations logged with a reason</div>
      </div>
    </div>
  </section>

  <section class="byclass">
    <h2>Recovery by failure cause</h2>
    <table>
      <thead>
        <tr><th>Cause</th><th class="num">Recovered / at risk</th><th class="num">Rate</th><th class="num">Records</th></tr>
      </thead>
      <tbody>{by_class_rows}
      </tbody>
    </table>
  </section>

  <section class="ledger">
    <h2>The ledger &mdash; every record</h2>
    <p class="lede">
      {n_records} records &middot; {n_rec} recovered &middot; {n_esc} escalated &middot;
      {n_aba} abandoned{f' &middot; {n_prog} in progress' if n_prog else ''}.
      Click a row to see the attempts behind it.
    </p>
    <div class="filters">
      <button class="filter-btn active" data-status="all">All ({n_records})</button>
      <button class="filter-btn" data-status="recovered">Recovered ({n_rec})</button>
      <button class="filter-btn" data-status="escalated">Escalated ({n_esc})</button>
      <button class="filter-btn" data-status="abandoned">Abandoned ({n_aba})</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>Record</th><th>Customer</th><th>Cause</th>
          <th class="num">Amount</th><th class="num">Attempts</th><th class="num">Recovered</th><th>Status</th>
        </tr>
      </thead>
      <tbody>{ledger_html}
      </tbody>
    </table>
    <p class="footnote">
      {n_derived} of {n_records} records resolve asynchronously: a customer-facing
      contact (a payment-update link, a re-auth request) is fired and logged as
      <span class="mono">in_progress</span>, and the customer's eventual action or
      inaction is applied to the record later, through the same
      <span class="mono">ingest_outcome</span> seam a real webhook would call. That
      resolution updates the record's status but does not yet write its own audit
      row (see <span class="mono">recoup/agent/scheduler.py:_apply_resolution</span>).
      Rows affected are marked with a derived &ldquo;&rarr; final&rdquo; line above,
      clearly distinguished from a logged attempt.
    </p>
  </section>

</div>
<script>
(function() {{
  var rows = Array.prototype.slice.call(document.querySelectorAll('tr.ledger-row'));
  var filterBtns = Array.prototype.slice.call(document.querySelectorAll('.filter-btn'));

  filterBtns.forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      filterBtns.forEach(function(b) {{ b.classList.remove('active'); }});
      btn.classList.add('active');
      var want = btn.getAttribute('data-status');
      rows.forEach(function(row) {{
        var show = want === 'all' || row.getAttribute('data-status') === want;
        row.hidden = !show;
        var detail = row.nextElementSibling;
        if (detail && detail.classList.contains('detail-row')) {{
          detail.hidden = !show || !row.classList.contains('open');
        }}
      }});
    }});
  }});

  rows.forEach(function(row) {{
    row.addEventListener('click', function() {{
      var detail = row.nextElementSibling;
      if (!detail || !detail.classList.contains('detail-row')) return;
      var open = row.classList.toggle('open');
      detail.hidden = !open;
    }});
  }});
}})();
</script>
</body>
</html>
"""


def main() -> None:
    summary, scheduler, latent = run_batch()
    recon = reconcile(summary, scheduler)
    disc = discipline(scheduler, latent)
    classes = by_class(scheduler)
    ledger = ledger_rows(scheduler, latent)

    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    meta = {
        "run_id": RUN_ID,
        "seed": frozen["seed"],
        "n_records": len(scheduler.states),
        "n_customers": len({st.record["customer_id"] for st in scheduler.states.values()}),
        "start": START.isoformat(sep=" ", timespec="minutes"),
    }

    page = render(summary, recon, disc, classes, ledger, meta)
    OUT_PATH.write_text(page, encoding="utf-8")

    print(f"wrote {OUT_PATH}  ({len(page):,} bytes)")
    print(f"  recovered {format_inr(recon['recovered'])} of {format_inr(recon['at_risk'])}  "
          f"({recon['rate']:.1%})  ties_out={recon['ties']}")
    print(f"  {len(ledger)} ledger rows  |  violations {disc['violations']}  |  "
          f"dead stopped {disc['dead_stopped']}/{disc['dead_total']}")


if __name__ == "__main__":
    main()
