"""Ingest the frozen batch into SQLite and run detection.

Detection is the deterministic ``reason -> class`` lookup, nothing more.  The
mismatch check is a **self-consistency test**: if the taxonomy and the generator
ever drift apart, it catches that here rather than at metrics time, when a wrong
number has already reached a slide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config.taxonomy import FailureClass, classify_reason
from recoup.db import Store
from recoup.generator import DATA_DIR, load_frozen, load_latent
from recoup.models import FailedRecord


@dataclass
class IngestReport:
    n_records: int
    n_customers: int
    total_at_risk_cents: int
    detected_histogram: Dict[str, int] = field(default_factory=dict)
    unknown_record_ids: List[str] = field(default_factory=list)
    detection_mismatches: List[Dict[str, str]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.unknown_record_ids and not self.detection_mismatches


def detect(record: FailedRecord) -> FailureClass:
    """The whole of detection: a lookup on the reason PayPal handed us."""
    return classify_reason(record.error_reason)


def ingest(
    store: Store,
    records: Optional[List[FailedRecord]] = None,
    data_dir: Path = DATA_DIR,
    check_against_labels: bool = True,
) -> IngestReport:
    if records is None:
        records, _customers = load_frozen(data_dir)

    histogram: Dict[str, int] = {}
    unknowns: List[str] = []
    for record in records:
        klass = detect(record)
        histogram[klass.value] = histogram.get(klass.value, 0) + 1
        if klass is FailureClass.UNKNOWN:
            # Unmapped signals are not guessed at: they go to a human.
            unknowns.append(record.record_id)
        store.upsert_record(record, detected_class=klass.value)

    mismatches: List[Dict[str, str]] = []
    if check_against_labels:
        try:
            latent = load_latent(data_dir)
        except FileNotFoundError:
            latent = {}
        for record in records:
            truth = latent.get(record.record_id)
            if truth is None:
                continue
            detected = detect(record)
            if detected is not truth.seeded_class:
                mismatches.append(
                    {
                        "record_id": record.record_id,
                        "reason": record.error_reason,
                        "detected": detected.value,
                        "seeded": truth.seeded_class.value,
                    }
                )

    return IngestReport(
        n_records=len(records),
        n_customers=len({r.customer_id for r in records}),
        total_at_risk_cents=sum(r.amount_cents for r in records),
        detected_histogram=histogram,
        unknown_record_ids=unknowns,
        detection_mismatches=mismatches,
    )
