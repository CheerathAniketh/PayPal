"""Build data/ledger.json: the agent's decisions on the frozen batch.

Every row comes from the SIMULATED environment (sampled outcomes, not real
money) and is tagged source="simulated". Live PayPal webhook events are served
separately by /api/webhook-events so the two are never blended.

Run:  python -m scripts.build_ledger
"""
from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config.taxonomy import classify_reason
from recoup.generator import hydrate_customers, load_frozen, load_latent
from scripts.run_batch import _run_once

OUT = Path(__file__).resolve().parent.parent / "data" / "ledger.json"


def main() -> None:
    records, customers = load_frozen()
    narrator = None
    if os.getenv("RECOUP_NARRATOR", "").lower() == "gemini":
        from recoup.agent.gemini_narrator import GeminiNarrator
        from recoup.agent.runtime import configure

        narrator = GeminiNarrator.from_env()
        if narrator is None:
            print("RECOUP_NARRATOR=gemini but GEMINI_API_KEY is not set; using template narration")
        else:
            configure(narrator=narrator)
    customers = hydrate_customers(customers)
    latent = load_latent()
    summary, scheduler, audit = _run_once(
        records, customers, latent, run_id="ledger-default", use_async=True
    )

    by_id = {r.record_id: r for r in records}
    rows = []
    for row in audit.rows:
        out = dict(row)
        rec = by_id.get(row.get("record_id"))
        if rec is not None:
            out["customer_id"] = rec.customer_id
            out["amount_cents"] = rec.amount_cents
            out["error_reason"] = rec.error_reason
            out["failure_class"] = classify_reason(rec.error_reason).value
            out["method"] = getattr(rec.method, "value", str(rec.method))
        out["source"] = "simulated"
        rows.append(out)

    assert hasattr(scheduler, "states"), "_run_once 2nd return is not a BatchScheduler"
    rows_by_rec = {}
    for r in rows:
        rows_by_rec.setdefault(r.get("record_id"), []).append(r)

    records_final = []
    for rid, st in scheduler.states.items():
        rrows = rows_by_rec.get(rid, [])
        rec = by_id.get(rid)
        at_attempt_cents = sum(int(r.get("amount_recovered_cents") or 0) for r in rrows)
        last = rrows[-1] if rrows else {}
        records_final.append({
            "record_id": rid,
            "customer_id": st.record.get("customer_id"),
            "amount_cents": int(st.record["amount_cents"]),
            "error_reason": st.record.get("error_reason"),
            "failure_class": classify_reason(st.record.get("error_reason")).value,
            "method": getattr(rec.method, "value", str(rec.method)) if rec else None,
            "final_status": st.status,
            "recovered_cents": int(st.recovered_cents),
            "outstanding_cents": int(st.outstanding_cents),
            "attempts": int(st.attempts),
            "async_recovered_cents": max(0, int(st.recovered_cents) - at_attempt_cents),
            "had_async_action": any(r.get("settles_async") for r in rrows),
            "last_action": last.get("chosen_action"),
            "last_rationale": last.get("rationale"),
            "source": "simulated",
        })

    summary_dict = (
        dataclasses.asdict(summary) if dataclasses.is_dataclass(summary) else {}
    )
    summary_dict["recovery_rate"] = summary.recovery_rate
    ledger = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "simulated",
        "policy": "rules",
        "summary": summary_dict,
        "rows": rows,
        "records": records_final,
    }
    OUT.write_text(json.dumps(ledger, indent=1, default=str))

    blocked = sum(
        any(v != "pass" for v in (r.get("guardrail_checks") or {}).values())
        for r in rows
    )
    print(summary.render())
    fs = Counter(r["final_status"] for r in records_final)
    tot = sum(r["recovered_cents"] for r in records_final)
    ok = (
        fs["recovered"] == summary.recovered_count
        and fs["escalated"] == summary.escalated_count
        and fs["abandoned"] == summary.abandoned_count
        and tot == summary.recovered_cents
    )
    print(f"records: {len(records_final)} | final status {dict(fs)} | ${tot/100:,.2f} recovered | matches summary: {ok}")
    print()
    print(f"wrote {OUT} | {len(rows)} rows | {blocked} rows with a failed guardrail")
    print("outcomes:", dict(Counter(r.get("outcome") for r in rows)))
    print("row keys:", sorted(rows[0].keys()))
    if narrator is not None:
        print("narrator:", narrator.stats())
    print("sample row:", json.dumps(rows[0], default=str))


if __name__ == "__main__":
    main()
