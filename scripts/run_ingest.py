"""Load the frozen batch into SQLite and print the ingest + detection report.

Run:  python -m scripts.run_ingest
"""

from __future__ import annotations

from recoup.db import Store
from recoup.ingest import ingest
from recoup.money import format_inr


def main() -> None:
    store = Store(":memory:")
    report = ingest(store)

    print("Ingest report")
    print("-" * 50)
    print(f"records          {report.n_records}")
    print(f"customers        {report.n_customers}")
    print(f"total at risk    {format_inr(report.total_at_risk_paise)}")
    print()
    print("detected class histogram:")
    for klass, n in sorted(report.detected_histogram.items()):
        print(f"  {klass:<20} {n}")
    print()
    print(f"unknowns:   {report.unknown_record_ids}")
    print(f"mismatches: {report.detection_mismatches}")
    print()
    print("status histogram after ingest:", store.status_histogram())
    print("CLEAN" if report.is_clean else "NEEDS ATTENTION")
    store.close()


if __name__ == "__main__":
    main()
