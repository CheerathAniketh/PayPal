"""The store: money at risk, runs, and the receipts.

Three tables.  ``records`` holds money at risk plus its lifecycle status,
``runs`` holds one row per agent run, ``audit_log`` holds the receipts.

Append-only, enforced by triggers
---------------------------------
The audit trail is the *evidence* for every claim in the final report.  Evidence
that could be silently rewritten -- by a buggy node, or by a developer tidying
up a bad run before recording the demo -- is not evidence.  So UPDATE and DELETE
on ``audit_log`` are physically rejected by the database, and both triggers are
tested.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from recoup.idempotency import RecordStatus, assert_transition
from recoup.models import AuditEntry, Customer, ExecutionMode, FailedRecord, RunRecord

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS records (
    record_id                   TEXT PRIMARY KEY,
    customer_id                 TEXT NOT NULL,
    amount_paise                INTEGER NOT NULL,
    recovered_paise             INTEGER NOT NULL DEFAULT 0,
    method                      TEXT NOT NULL,
    error_reason                TEXT NOT NULL,
    error_source                TEXT NOT NULL,
    failed_at                   TEXT NOT NULL,
    prior_retries               INTEGER NOT NULL,
    is_mandate_debit            INTEGER NOT NULL,
    pre_debit_notified          INTEGER NOT NULL,
    subscription_status         TEXT NOT NULL,
    customer_tenure_days        INTEGER NOT NULL,
    customer_avg_payment_paise  INTEGER NOT NULL,
    customer_salary_day         INTEGER NOT NULL,
    customer_prior_failures     INTEGER NOT NULL,
    customer_is_subscriber      INTEGER NOT NULL,
    detected_class              TEXT,
    status                      TEXT NOT NULL DEFAULT 'pending',
    attempts                    INTEGER NOT NULL DEFAULT 0,
    updated_at                  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    seed         INTEGER NOT NULL,
    mode         TEXT NOT NULL,
    batch_size   INTEGER NOT NULL,
    config_json  TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    notes        TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS audit_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL,
    record_id               TEXT NOT NULL,
    customer_id             TEXT NOT NULL DEFAULT '',
    timestamp               TEXT NOT NULL,
    attempt_number          INTEGER NOT NULL,
    chosen_action           TEXT NOT NULL,
    rationale               TEXT NOT NULL,
    guardrail_checks        TEXT NOT NULL,
    model_score             REAL,
    outcome                 TEXT NOT NULL,
    amount_recovered_paise  INTEGER NOT NULL,
    idempotency_key         TEXT NOT NULL,
    execution_mode          TEXT NOT NULL,
    api_called              INTEGER NOT NULL,
    razorpay_entity_id      TEXT,
    was_mocked              INTEGER NOT NULL DEFAULT 0,
    mock_reason             TEXT NOT NULL DEFAULT '',
    api_error               TEXT NOT NULL DEFAULT ''
);

-- The strongest guarantee in the project: a duplicate attempt key raises
-- IntegrityError rather than quietly double-charging. Impossible, not unlikely.
CREATE UNIQUE INDEX IF NOT EXISTS audit_idempotency_unique
    ON audit_log (idempotency_key);

CREATE INDEX IF NOT EXISTS audit_by_run ON audit_log (run_id);
CREATE INDEX IF NOT EXISTS audit_by_record ON audit_log (record_id);

CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE forbidden'); END;

CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: DELETE forbidden'); END;
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def upsert_record(
        self, record: FailedRecord, detected_class: Optional[str] = None
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO records (
                record_id, customer_id, amount_paise, method, error_reason,
                error_source, failed_at, prior_retries, is_mandate_debit,
                pre_debit_notified, subscription_status, customer_tenure_days,
                customer_avg_payment_paise, customer_salary_day,
                customer_prior_failures, customer_is_subscriber, detected_class,
                status, attempts, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(record_id) DO UPDATE SET
                amount_paise = excluded.amount_paise,
                detected_class = excluded.detected_class
            """,
            (
                record.record_id, record.customer_id, record.amount_paise,
                record.method.value, record.error_reason, record.error_source,
                record.failed_at.isoformat(), record.prior_retries,
                int(record.is_mandate_debit), int(record.pre_debit_notified),
                record.subscription_status, record.customer_tenure_days,
                record.customer_avg_payment_paise, record.customer_salary_day,
                record.customer_prior_failures, int(record.customer_is_subscriber),
                detected_class, RecordStatus.PENDING.value, 0, None,
            ),
        )
        self.conn.commit()

    def get_status(self, record_id: str) -> RecordStatus:
        row = self.conn.execute(
            "SELECT status FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise KeyError(record_id)
        return RecordStatus(row["status"])

    def set_status(
        self,
        record_id: str,
        target: RecordStatus,
        *,
        now: Optional[datetime] = None,
    ) -> RecordStatus:
        """Move a record's lifecycle state, refusing illegal transitions.

        Terminal records cannot be reopened: doing so would mean contacting a
        customer the agent already decided to stop contacting.
        """
        current = self.get_status(record_id)
        assert_transition(current, target)
        self.conn.execute(
            "UPDATE records SET status = ?, updated_at = ? WHERE record_id = ?",
            (target.value, (now or datetime.now()).isoformat(), record_id),
        )
        self.conn.commit()
        return target

    def bump_attempts(self, record_id: str) -> int:
        self.conn.execute(
            "UPDATE records SET attempts = attempts + 1 WHERE record_id = ?",
            (record_id,),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT attempts FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        return int(row["attempts"])

    def add_recovered(self, record_id: str, paise: int) -> int:
        self.conn.execute(
            "UPDATE records SET recovered_paise = recovered_paise + ? "
            "WHERE record_id = ?",
            (paise, record_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT recovered_paise FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        return int(row["recovered_paise"])

    def outstanding_paise(self, record_id: str) -> int:
        row = self.conn.execute(
            "SELECT amount_paise - recovered_paise AS outstanding "
            "FROM records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(record_id)
        return max(0, int(row["outstanding"]))

    def open_records(self) -> List[sqlite3.Row]:
        """What is still worth working.  This is what makes a run resumable."""
        placeholders = ",".join("?" for _ in _OPEN_STATUSES)
        return list(
            self.conn.execute(
                f"SELECT * FROM records WHERE status IN ({placeholders}) "
                "ORDER BY amount_paise DESC",
                [s.value for s in _OPEN_STATUSES],
            )
        )

    def status_histogram(self) -> Dict[str, int]:
        return {
            row["status"]: row["n"]
            for row in self.conn.execute(
                "SELECT status, COUNT(*) AS n FROM records GROUP BY status"
            )
        }

    def total_at_risk_paise(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(amount_paise), 0) AS total FROM records"
        ).fetchone()
        return int(row["total"])

    def total_recovered_paise(self, run_id: Optional[str] = None) -> int:
        if run_id is None:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(amount_recovered_paise), 0) AS total "
                "FROM audit_log"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(amount_recovered_paise), 0) AS total "
                "FROM audit_log WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["total"])

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------
    def start_run(self, run: RunRecord) -> str:
        self.conn.execute(
            "INSERT INTO runs (run_id, seed, mode, batch_size, config_json, "
            "started_at, finished_at, notes) VALUES (?,?,?,?,?,?,?,?)",
            (
                run.run_id, run.seed, run.mode.value, run.batch_size,
                run.config_json, run.started_at.isoformat(),
                run.finished_at.isoformat() if run.finished_at else None,
                run.notes,
            ),
        )
        self.conn.commit()
        return run.run_id

    def finish_run(self, run_id: str, finished_at: datetime, notes: str = "") -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, notes = ? WHERE run_id = ?",
            (finished_at.isoformat(), notes, run_id),
        )
        self.conn.commit()

    def runs(self) -> List[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM runs ORDER BY started_at"))

    # ------------------------------------------------------------------
    # audit
    # ------------------------------------------------------------------
    def append_audit(self, entry: AuditEntry) -> int:
        row = entry.to_row()
        cursor = self.conn.execute(
            """
            INSERT INTO audit_log (
                run_id, record_id, customer_id, timestamp, attempt_number,
                chosen_action, rationale, guardrail_checks, model_score,
                outcome, amount_recovered_paise, idempotency_key,
                execution_mode, api_called, razorpay_entity_id, was_mocked,
                mock_reason, api_error
            ) VALUES (
                :run_id, :record_id, :customer_id, :timestamp, :attempt_number,
                :chosen_action, :rationale, :guardrail_checks, :model_score,
                :outcome, :amount_recovered_paise, :idempotency_key,
                :execution_mode, :api_called, :razorpay_entity_id, :was_mocked,
                :mock_reason, :api_error
            )
            """,
            row,
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def audit_for_run(self, run_id: str) -> List[sqlite3.Row]:
        """One run's receipts, cleanly isolated.  Without ``run_id`` on every
        row, re-running the batch would silently double every metric."""
        return list(
            self.conn.execute(
                "SELECT * FROM audit_log WHERE run_id = ? ORDER BY id", (run_id,)
            )
        )

    def audit_for_record(self, record_id: str) -> List[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM audit_log WHERE record_id = ? ORDER BY id",
                (record_id,),
            )
        )

    def outcome_histogram(self, run_id: Optional[str] = None) -> Dict[str, int]:
        if run_id is None:
            rows = self.conn.execute(
                "SELECT outcome, COUNT(*) AS n FROM audit_log GROUP BY outcome"
            )
        else:
            rows = self.conn.execute(
                "SELECT outcome, COUNT(*) AS n FROM audit_log WHERE run_id = ? "
                "GROUP BY outcome",
                (run_id,),
            )
        return {row["outcome"]: row["n"] for row in rows}

    def contact_count(self, customer_id: str, since_iso: str) -> int:
        """Customer-facing contacts in a rolling window.

        A query, never a counter: a counter that increments and resets at a
        window boundary silently reintroduces the reset-burst problem.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE customer_id = ? "
            "AND timestamp >= ? AND chosen_action IN ('update_payment_method', "
            "'re_auth_mandate')",
            (customer_id, since_iso),
        ).fetchone()
        return int(row["n"])


_OPEN_STATUSES = (
    RecordStatus.PENDING,
    RecordStatus.IN_PROGRESS,
    RecordStatus.SCHEDULED,
    RecordStatus.PARTIALLY_RECOVERED,
)


def new_run(
    run_id: str,
    seed: int,
    mode: ExecutionMode,
    batch_size: int,
    config: Dict[str, Any],
    started_at: datetime,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        seed=seed,
        mode=mode,
        batch_size=batch_size,
        config_json=json.dumps(config, default=str, sort_keys=True),
        started_at=started_at,
    )


def load_customers(store: Store) -> Dict[str, Customer]:  # pragma: no cover - helper
    """Reconstruct observable customers from the records table."""
    customers: Dict[str, Customer] = {}
    for row in store.conn.execute(
        "SELECT DISTINCT customer_id, customer_tenure_days, "
        "customer_avg_payment_paise, customer_salary_day, "
        "customer_prior_failures, customer_is_subscriber FROM records"
    ):
        customers[row["customer_id"]] = Customer(
            customer_id=row["customer_id"],
            tenure_days=row["customer_tenure_days"],
            avg_payment_paise=row["customer_avg_payment_paise"],
            salary_day=row["customer_salary_day"],
            is_subscriber=bool(row["customer_is_subscriber"]),
            prior_failures=row["customer_prior_failures"],
        )
    return customers


def iter_rows(rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]
