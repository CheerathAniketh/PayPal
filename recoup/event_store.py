"""Append-only outcome-event store with idempotent ingestion.

Correctness is split cleanly:

* the UNIQUE index on the staged key kills *exact duplicates* at write time;
* the monotonic fold (in ``events.py``) handles *distinct-but-stale* events at
  read time.

``ingest_outcome`` is the single entry point for executor intent, simulator
settlement, and (later) real webhooks alike.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from recoup.events import (
    ApiResult,
    FinancialResult,
    OutcomeEvent,
    RecordOutcome,
    Stage,
    fold,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS outcome_events (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key         TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    record_id               TEXT NOT NULL,
    customer_id             TEXT NOT NULL,
    attempt_n               INTEGER NOT NULL,
    stage                   TEXT NOT NULL,
    intervention            TEXT NOT NULL,
    is_contact              INTEGER NOT NULL,
    amount_cents            INTEGER NOT NULL,
    occurred_at             TEXT NOT NULL,
    ingested_at             TEXT NOT NULL,
    api_result              TEXT NOT NULL,
    financial_result        TEXT NOT NULL,
    amount_recovered_cents  INTEGER NOT NULL DEFAULT 0,
    failure_reason          TEXT NOT NULL DEFAULT '',
    source                  TEXT NOT NULL DEFAULT 'simulator',
    raw                     TEXT NOT NULL DEFAULT '{}'
);

-- Enforced at the database, not in memory, so it survives restarts.
CREATE UNIQUE INDEX IF NOT EXISTS outcome_key_unique
    ON outcome_events (idempotency_key);

CREATE INDEX IF NOT EXISTS outcome_by_record ON outcome_events (record_id);
CREATE INDEX IF NOT EXISTS outcome_by_customer ON outcome_events (customer_id);

CREATE TRIGGER IF NOT EXISTS outcome_no_update BEFORE UPDATE ON outcome_events
BEGIN SELECT RAISE(ABORT, 'outcome_events is append-only: UPDATE forbidden'); END;

CREATE TRIGGER IF NOT EXISTS outcome_no_delete BEFORE DELETE ON outcome_events
BEGIN SELECT RAISE(ABORT, 'outcome_events is append-only: DELETE forbidden'); END;
"""


class EventStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def ingest_outcome(self, event: OutcomeEvent) -> bool:
        """Append one event.  Idempotent: an exact duplicate is a no-op.

        Returns True if the event was newly written, False if it was a duplicate
        (its key already existed).  A duplicate returning False -- rather than
        raising -- is what lets a webhook fire twice without the caller caring.
        """
        row = event.to_row()
        try:
            self.conn.execute(
                """
                INSERT INTO outcome_events (
                    idempotency_key, run_id, record_id, customer_id, attempt_n,
                    stage, intervention, is_contact, amount_cents, occurred_at,
                    ingested_at, api_result, financial_result,
                    amount_recovered_cents, failure_reason, source, raw
                ) VALUES (
                    :idempotency_key, :run_id, :record_id, :customer_id,
                    :attempt_n, :stage, :intervention, :is_contact,
                    :amount_cents, :occurred_at, :ingested_at, :api_result,
                    :financial_result, :amount_recovered_cents, :failure_reason,
                    :source, :raw
                )
                """,
                row,
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Exact duplicate key -> the recovered-rupees headline cannot
            # double-count on a retried webhook delivery.
            self.conn.rollback()
            return False

    # ------------------------------------------------------------------
    def events_for_record(self, record_id: str) -> List[OutcomeEvent]:
        rows = self.conn.execute(
            "SELECT * FROM outcome_events WHERE record_id = ? ORDER BY id",
            (record_id,),
        )
        return [self._event_from_row(r) for r in rows]

    def fold_record(self, record_id: str) -> Dict[int, RecordOutcome]:
        return fold(self.events_for_record(record_id))

    def total_recovered_cents(self, run_id: str | None = None) -> int:
        """Recovered money, from RESOLVED events only, deduped by attempt.

        Folds each record so a duplicated resolution cannot be counted twice.
        """
        if run_id is None:
            record_ids = [
                r["record_id"]
                for r in self.conn.execute(
                    "SELECT DISTINCT record_id FROM outcome_events"
                )
            ]
        else:
            record_ids = [
                r["record_id"]
                for r in self.conn.execute(
                    "SELECT DISTINCT record_id FROM outcome_events WHERE run_id = ?",
                    (run_id,),
                )
            ]
        total = 0
        for rid in record_ids:
            for outcome in self.fold_record(rid).values():
                total += outcome.amount_recovered_cents
        return total

    def contact_count(self, customer_id: str, since_iso: str) -> int:
        """Customer-facing contacts with ``occurred_at >= since_iso``.

        Uses ``occurred_at`` (when the customer was contacted), never
        ``ingested_at`` (when we heard about it).  A rolling window is a query,
        evaluated fresh -- never a counter that resets at a boundary.
        """
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT run_id || ':' || record_id || ':' || attempt_n) "
            "AS n FROM outcome_events WHERE customer_id = ? AND is_contact = 1 "
            "AND stage = ? AND occurred_at >= ?",
            (customer_id, Stage.FIRED.value, since_iso),
        ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------
    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> OutcomeEvent:
        return OutcomeEvent(
            run_id=row["run_id"],
            record_id=row["record_id"],
            customer_id=row["customer_id"],
            attempt_n=row["attempt_n"],
            stage=Stage(row["stage"]),
            intervention=row["intervention"],
            is_contact=bool(row["is_contact"]),
            amount_cents=row["amount_cents"],
            occurred_at=row["occurred_at"],
            ingested_at=row["ingested_at"],
            api_result=ApiResult(row["api_result"]),
            financial_result=FinancialResult(row["financial_result"]),
            amount_recovered_cents=row["amount_recovered_cents"],
            failure_reason=row["failure_reason"],
            source=row["source"],
            raw=json.loads(row["raw"]),
        )
