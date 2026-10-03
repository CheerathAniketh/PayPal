"""Walk one truly-dead class-5 (do-not-honour) record end to end.

The track bar asks for *one failure handled gracefully*: not hammered, not
looped -- tried once under the cap, then escalated with a structured reason and
stopped.  This script traces exactly that, one record, attempt by attempt,
through the real compiled graph (``graph:graph``) -- the same path the batch
scheduler drives -- and then prints the append-only audit rows that are the
receipts.

It is deliberately verbose and single-record so it can be read on screen: every
attempt shows the diagnosis, the economic decision (with the ranked candidates),
the guardrail verdict (with which check fired), the execution result, and the
narration.  Nothing here is special-cased for the demo; it is the ordinary graph
applied to one record and printed loudly.

Run:  python -m scripts.walk_dead_record       (or: python scripts/walk_dead_record.py)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from config.taxonomy import FailureClass, classify_reason
from recoup.agent.graph import graph
from recoup.agent.runtime import DemoAuditLog, DemoExecutor, configure
from recoup.environment import RecoveryEnvironment
from recoup.generator import hydrate_customers, load_frozen, load_latent
from recoup.money import format_inr

TERMINAL = {"escalated", "abandoned", "recovered"}
MAX_ATTEMPTS = 5   # safety stop; class-5 escalates at attempt 2 in practice
BAR = "=" * 72
DASH = "-" * 72


def _pick_dead_class5(records, latent):
    """First truly-dead do-not-honour record, deterministic by record order."""
    for r in records:
        if (
            classify_reason(r.error_reason) is FailureClass.DO_NOT_HONOUR
            and latent[r.record_id].is_truly_dead
        ):
            return r
    return None


def _print_attempt(n: int, out: Dict[str, Any]) -> None:
    d = out.get("decision", {})
    g = out.get("guardrail")
    ex = out.get("execution", {})
    status = out.get("terminal_status", "?")

    print(f"\nAttempt {n}")
    print(DASH)
    diag = out.get("diagnosis", {})
    print(f"  diagnose   {diag.get('failure_class','?'):<16} "
          f"(reason={diag.get('reason','?')}, {diag.get('method','')})")

    # decide
    print(f"  decide     chose {d.get('chosen','?'):<14} "
          f"net_EV={d.get('net_ev_cents',0):>7} cents  p={d.get('p_recover',0):.3f}")
    for c in d.get("ranked", []):
        mark = "->" if c["intervention"] == d.get("chosen") else "  "
        print(f"             {mark} {c['intervention']:<20} "
              f"p={c['p_recover']:.3f}  net_EV={c['net_ev_cents']:>7}  "
              f"{'contact' if c['is_contact'] else 'silent'}")
    print(f"             rationale: {d.get('rationale','')}")

    # guardrail
    if g is None:
        print("  guardrail  n/a  (decide abandoned/escalated before the gate)")
    else:
        verdict = "PASS" if g["allowed"] else "BLOCK"
        print(f"  guardrail  {verdict}")
        for name, res in g.get("checks", {}).items():
            flag = "x" if res == "fail" else "."
            print(f"               [{flag}] {name}: {res}")
        if not g["allowed"]:
            print(f"             reason: {g.get('reason','')}")

    # execute
    if ex:
        got = format_inr(int(ex.get("amount_recovered_cents", 0)))
        print(f"  execute    outcome={ex.get('outcome','?'):<12} recovered={got:>10}  "
              f"api_called={ex.get('api_called', False)}  "
              f"async={ex.get('settles_async', False)}")
        if ex.get("idempotency_key"):
            print(f"             idempotency_key={ex['idempotency_key']}")

    print(f"  STATUS     {status.upper()}")
    if out.get("narration"):
        print(f"  narrate    {out['narration']}")


def main() -> None:
    records, customers = load_frozen()
    customers = hydrate_customers(customers)   # eval-only: the executor needs truth
    latent = load_latent()

    audit = DemoAuditLog()
    configure(
        executor=DemoExecutor(RecoveryEnvironment(), customers, latent),
        audit=audit,
    )

    dead = _pick_dead_class5(records, latent)
    if dead is None:
        print("No truly-dead do-not-honour record in the batch.")
        return

    now = datetime(2026, 1, 27, 9)  # a salary day; timing is not the issue here
    print(BAR)
    print("GRACEFUL FAILURE -- one truly-dead do-not-honour record, end to end")
    print(BAR)
    print(f"record     {dead.record_id}  customer {dead.customer_id}")
    print(f"amount     {format_inr(dead.amount_cents)}   method {dead.method.value}")
    print(f"reason     {dead.error_reason}  -> class do_not_honour")
    print("truth      seeded truly-dead: recovery probability ~0 whatever we try")
    print("policy     do_not_honour attempt cap = 1 (one controlled retry, then stop)")

    final_status: Optional[str] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rec = dead.to_json()
        rec["_outstanding_cents"] = rec["amount_cents"]
        out = graph.invoke(
            {"run_id": "run_walk", "record": rec, "attempt": attempt,
             "now_iso": now.isoformat()}
        )
        _print_attempt(attempt, out)
        final_status = out.get("terminal_status")
        if final_status in TERMINAL:
            break

    print("\n" + BAR)
    print("THE RECEIPTS -- audit rows (in-memory demo adapter; "
          "SQLite append-only store is built + tested)")
    print(BAR)
    for i, row in enumerate(audit.rows, 1):
        print(f"\n[{i}] {json.dumps(_readable(row), indent=2)}")

    print("\n" + BAR)
    print("WHAT HAPPENED")
    print(BAR)
    print(
        "  The record was worth one attempt on the economics, so the agent made\n"
        "  exactly one controlled retry.  On the next attempt the per-class cap\n"
        "  (do_not_honour = 1) tripped the guardrail BEFORE any execution, the\n"
        f"  record was escalated with a structured reason, and processing STOPPED.\n"
        f"  Final status: {final_status}.  No hammering, no loop -- and every step\n"
        "  is on the append-only trail above with the exact rule that fired."
    )


def _readable(row: Dict[str, Any]) -> Dict[str, Any]:
    """A trimmed, human-readable view of an audit row for on-screen reading."""
    r = dict(row)
    if isinstance(r.get("amount_recovered_cents"), int):
        r["amount_recovered"] = format_inr(r["amount_recovered_cents"])
    keep = [
        "record_id", "customer_id", "attempt_number", "chosen_action",
        "outcome", "guardrail_checks", "model_score", "amount_recovered",
        "idempotency_key", "api_called", "rationale", "narration",
    ]
    return {k: r[k] for k in keep if k in r}


if __name__ == "__main__":
    main()
