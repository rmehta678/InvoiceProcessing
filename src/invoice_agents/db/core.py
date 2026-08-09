"""Explicit, versioned SQLite setup and strict preflight verification."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from invoice_agents.errors import DatabaseVerificationError, ErrorCategory

SQLITE_SIGNATURE = b"SQLite format 3\x00"
SEED_ROWS = (
    ("SKU-WIDGET-A", "WidgetA", 15),
    ("SKU-WIDGET-B", "WidgetB", 10),
    ("SKU-GADGET-X", "GadgetX", 5),
    ("SKU-FAKE-ITEM", "FakeItem", 0),
)


class DatabaseKind(StrEnum):
    INVENTORY = "inventory"
    WORKFLOW = "workflow"


# Required version per database kind; preflight rejects anything else, in either
# direction, so an unmigrated or future database never silently processes cases.
SCHEMA_VERSIONS: dict[DatabaseKind, int] = {
    DatabaseKind.INVENTORY: 1,
    DatabaseKind.WORKFLOW: 3,
}

# Named indexes created by the migrations; verification fails when one is missing.
REQUIRED_INDEXES: dict[DatabaseKind, frozenset[str]] = {
    DatabaseKind.INVENTORY: frozenset({"idx_item_aliases_sku"}),
    DatabaseKind.WORKFLOW: frozenset(
        {
            "idx_source_artifacts_hash",
            "idx_cases_invoice_vendor",
            "idx_cases_source_id",
            "idx_payments_case_id",
            "idx_events_case_created",
            "idx_review_requests_case_sequence",
            "idx_cases_execution_lease",
        }
    ),
}

REQUIRED_WORKFLOW_TRIGGERS: dict[str, str] = {
    "trg_final_decisions_no_insert_after_paid": """
        CREATE TRIGGER trg_final_decisions_no_insert_after_paid
        BEFORE INSERT ON final_decisions
        WHEN EXISTS (
            SELECT 1 FROM payments
            WHERE payments.case_id = NEW.case_id AND payments.status = 'PAID'
        )
        BEGIN
            SELECT RAISE(ABORT, 'PAID_FINAL_DECISION_IMMUTABLE');
        END
    """,
    "trg_final_decisions_no_update_after_paid": """
        CREATE TRIGGER trg_final_decisions_no_update_after_paid
        BEFORE UPDATE ON final_decisions
        WHEN EXISTS (
            SELECT 1 FROM payments
            WHERE payments.case_id = OLD.case_id AND payments.status = 'PAID'
        )
        BEGIN
            SELECT RAISE(ABORT, 'PAID_FINAL_DECISION_IMMUTABLE');
        END
    """,
    "trg_final_decisions_no_delete_after_paid": """
        CREATE TRIGGER trg_final_decisions_no_delete_after_paid
        BEFORE DELETE ON final_decisions
        WHEN EXISTS (
            SELECT 1 FROM payments
            WHERE payments.case_id = OLD.case_id AND payments.status = 'PAID'
        )
        BEGIN
            SELECT RAISE(ABORT, 'PAID_FINAL_DECISION_IMMUTABLE');
        END
    """,
    "trg_cases_execution_authority_insert": """
        CREATE TRIGGER trg_cases_execution_authority_insert
        BEFORE INSERT ON cases
        WHEN NOT (
            (NEW.execution_state = 'IDLE' AND NEW.execution_token IS NULL
                AND NEW.lease_expires_at IS NULL
                AND typeof(NEW.execution_generation) = 'integer'
                AND NEW.execution_generation >= 0)
            OR (NEW.execution_state = 'RUNNING' AND NEW.execution_token IS NOT NULL
                AND NEW.execution_token <> '' AND NEW.lease_expires_at IS NOT NULL
                AND substr(NEW.lease_expires_at, 1, 4) <> '0000'
                AND (
                    (length(NEW.lease_expires_at) = 25 AND NEW.lease_expires_at GLOB
                        '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00')
                    OR (length(NEW.lease_expires_at) = 32 AND NEW.lease_expires_at GLOB
                        '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                        AND substr(NEW.lease_expires_at, 21, 6) <> '000000')
                )
                AND datetime(NEW.lease_expires_at) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', NEW.lease_expires_at) =
                    substr(NEW.lease_expires_at, 1, 19)
                AND CAST(substr(NEW.lease_expires_at, 12, 2) AS INTEGER) BETWEEN 0 AND 23
                AND CAST(substr(NEW.lease_expires_at, 15, 2) AS INTEGER) BETWEEN 0 AND 59
                AND CAST(substr(NEW.lease_expires_at, 18, 2) AS INTEGER) BETWEEN 0 AND 59
                AND typeof(NEW.execution_generation) = 'integer'
                AND NEW.execution_generation >= 1)
            OR (NEW.execution_state = 'FINISHED' AND NEW.execution_token IS NOT NULL
                AND NEW.execution_token <> '' AND NEW.lease_expires_at IS NULL
                AND typeof(NEW.execution_generation) = 'integer'
                AND NEW.execution_generation >= 1)
        )
        BEGIN
            SELECT RAISE(ABORT, 'INVALID_EXECUTION_AUTHORITY');
        END
    """,
    "trg_cases_execution_authority_update": """
        CREATE TRIGGER trg_cases_execution_authority_update
        BEFORE UPDATE OF execution_token, execution_generation, execution_state, lease_expires_at
        ON cases
        WHEN NOT (
            (NEW.execution_state = 'IDLE' AND NEW.execution_token IS NULL
                AND NEW.lease_expires_at IS NULL
                AND typeof(NEW.execution_generation) = 'integer'
                AND NEW.execution_generation >= 0)
            OR (NEW.execution_state = 'RUNNING' AND NEW.execution_token IS NOT NULL
                AND NEW.execution_token <> '' AND NEW.lease_expires_at IS NOT NULL
                AND substr(NEW.lease_expires_at, 1, 4) <> '0000'
                AND (
                    (length(NEW.lease_expires_at) = 25 AND NEW.lease_expires_at GLOB
                        '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00')
                    OR (length(NEW.lease_expires_at) = 32 AND NEW.lease_expires_at GLOB
                        '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                        AND substr(NEW.lease_expires_at, 21, 6) <> '000000')
                )
                AND datetime(NEW.lease_expires_at) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', NEW.lease_expires_at) =
                    substr(NEW.lease_expires_at, 1, 19)
                AND CAST(substr(NEW.lease_expires_at, 12, 2) AS INTEGER) BETWEEN 0 AND 23
                AND CAST(substr(NEW.lease_expires_at, 15, 2) AS INTEGER) BETWEEN 0 AND 59
                AND CAST(substr(NEW.lease_expires_at, 18, 2) AS INTEGER) BETWEEN 0 AND 59
                AND typeof(NEW.execution_generation) = 'integer'
                AND NEW.execution_generation >= 1)
            OR (NEW.execution_state = 'FINISHED' AND NEW.execution_token IS NOT NULL
                AND NEW.execution_token <> '' AND NEW.lease_expires_at IS NULL
                AND typeof(NEW.execution_generation) = 'integer'
                AND NEW.execution_generation >= 1)
        )
        BEGIN
            SELECT RAISE(ABORT, 'INVALID_EXECUTION_AUTHORITY');
        END
    """,
}


def _packaged_workflow_trigger_definitions() -> dict[str, str]:
    """Use migration 003 itself as the exact preflight trigger contract."""

    script = (
        files("invoice_agents.db")
        .joinpath("migrations", "workflow", "003_execution_fencing.sql")
        .read_text(encoding="utf-8")
    )
    definitions = {
        match.group(1): match.group(0)
        for match in re.finditer(
            r"CREATE\s+TRIGGER\s+([A-Za-z0-9_]+)\b.*?\bEND\s*;",
            script,
            flags=re.IGNORECASE | re.DOTALL,
        )
    }
    if not definitions:
        raise RuntimeError("workflow migration 003 defines no triggers")
    return definitions


REQUIRED_WORKFLOW_TRIGGERS.update(_packaged_workflow_trigger_definitions())


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def infer_kind(path: Path) -> DatabaseKind:
    return DatabaseKind.WORKFLOW if "workflow" in path.name.lower() else DatabaseKind.INVENTORY


@contextmanager
def connect_database(path: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    """Open SQLite with foreign keys enabled and row dictionaries."""

    if read_only:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5)
    else:
        connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    from invoice_agents.evidence_snapshot import (
        stored_evidence_snapshot_digest,
        stored_unresolved_blocker_count,
    )
    from invoice_agents.payment.identity import payment_identity_key

    connection.create_function(
        "payment_identity_key",
        2,
        payment_identity_key,
        deterministic=True,
    )
    connection.create_function(
        "stored_evidence_snapshot_digest",
        7,
        stored_evidence_snapshot_digest,
        deterministic=True,
    )
    connection.create_function(
        "stored_unresolved_blocker_count",
        2,
        stored_unresolved_blocker_count,
        deterministic=True,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def _migration_resources(kind: DatabaseKind) -> list[Traversable]:
    root = files("invoice_agents.db").joinpath("migrations", kind.value)
    resources = sorted(
        (item for item in root.iterdir() if item.name[:3].isdigit() and item.name.endswith(".sql")),
        key=lambda item: item.name,
    )
    if not resources:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"no packaged migration files found for {kind.value}",
            stop_reason="MIGRATION_NOT_FOUND",
        )
    return resources


def _migration_statements(script: str) -> list[str]:
    """Split SQLite scripts without executing implicit transaction boundaries."""

    uncommented = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("--")
    )
    statements: list[str] = []
    buffer = ""
    for line in uncommented.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                statements.append(statement)
            buffer = ""
    if buffer.strip():
        raise sqlite3.OperationalError("migration contains an incomplete SQL statement")
    return statements


def _normalized_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).casefold()


def _require_reconciled_workflow_authorization(
    connection: sqlite3.Connection,
) -> None:
    """Refuse to invent generation-bound provenance for pre-v3 decisions or payments."""

    final_decision_count = int(
        connection.execute("SELECT COUNT(*) FROM final_decisions").fetchone()[0]
    )
    payment_count = int(connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0])
    if final_decision_count or payment_count:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            "workflow authorization reconciliation is required before migration 003 "
            f"(final_decisions={final_decision_count}, payments={payment_count})",
            stop_reason="AUTHORIZATION_RECONCILIATION_REQUIRED",
            details={
                "final_decision_count": final_decision_count,
                "payment_count": payment_count,
            },
        )


def migrate_database(path: Path, kind: DatabaseKind | None = None) -> list[int]:
    """Apply unapplied migrations; this is never called by normal processing."""

    selected_kind = kind or infer_kind(path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    applied: list[int] = []
    with connect_database(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        existing = {
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_version").fetchall()
        }
        connection.commit()
        for resource in _migration_resources(selected_kind):
            version = int(resource.name.split("_", 1)[0])
            if version in existing:
                continue
            script = resource.read_text(encoding="utf-8")
            statements = _migration_statements(script)
            normalized_statements = {_normalized_sql(statement) for statement in statements}
            disable_foreign_keys = "pragma foreign_keys=off" in normalized_statements
            transaction_controls = {
                "begin",
                "begin transaction",
                "begin immediate",
                "commit",
                "end transaction",
                "pragma foreign_keys=off",
                "pragma foreign_keys=on",
            }
            try:
                if disable_foreign_keys:
                    connection.execute("PRAGMA foreign_keys = OFF")
                connection.execute("BEGIN IMMEDIATE")
                if selected_kind is DatabaseKind.WORKFLOW and version == 3:
                    _require_reconciled_workflow_authorization(connection)
                for statement in statements:
                    if _normalized_sql(statement) in transaction_controls:
                        continue
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (version, utc_now()),
                )
                connection.commit()
            except DatabaseVerificationError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    f"migration {resource.name} failed: {exc}",
                    stop_reason="MIGRATION_FAILED",
                ) from exc
            finally:
                if disable_foreign_keys:
                    connection.execute("PRAGMA foreign_keys = ON")
            applied.append(version)
    return applied


def seed_inventory(path: Path) -> int:
    """Seed only the four authoritative facts supplied in the original README."""

    verify_database(path, DatabaseKind.INVENTORY, require_seed=False)
    with connect_database(path) as connection:
        try:
            connection.executemany(
                "INSERT INTO inventory(sku, item_name, available_stock) VALUES (?, ?, ?) "
                "ON CONFLICT(sku) DO UPDATE SET "
                "item_name=excluded.item_name, available_stock=excluded.available_stock",
                SEED_ROWS,
            )
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                f"inventory seed failed: {exc}",
                stop_reason="SEED_FAILED",
            ) from exc
    return len(SEED_ROWS)


def ensure_databases(inventory_db: Path, workflow_db: Path) -> dict[str, list[int]]:
    """Migrate, seed, and verify both databases; idempotent, so safe on every start.

    Case processing still never repairs a database - this is an explicit setup
    entry point shared by the CLI so one command can bring the system up.
    """

    applied = {
        DatabaseKind.INVENTORY.value: migrate_database(inventory_db, DatabaseKind.INVENTORY),
        DatabaseKind.WORKFLOW.value: migrate_database(workflow_db, DatabaseKind.WORKFLOW),
    }
    seed_inventory(inventory_db)
    verify_database(inventory_db, DatabaseKind.INVENTORY)
    verify_database(workflow_db, DatabaseKind.WORKFLOW)
    return applied


def _assert_sqlite_file(path: Path) -> None:
    if not path.is_file():
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"required database does not exist: {path}",
            stop_reason="DATABASE_MISSING",
        )
    with path.open("rb") as handle:
        if handle.read(len(SQLITE_SIGNATURE)) != SQLITE_SIGNATURE:
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                f"file is not a SQLite database: {path}",
                stop_reason="DATABASE_SIGNATURE_INVALID",
            )


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _workflow_authorization_provenance_counts(
    connection: sqlite3.Connection,
) -> dict[str, int]:
    """Count persisted authorization rows that no longer have a complete relational anchor."""

    valid_anchors_sql = """
        SELECT anchor.case_id, anchor.execution_generation, anchor.evidence_snapshot_digest,
            anchor.review_id, anchor.unresolved_blocker_count, anchor.critique_disposition
        FROM validated_evidence_snapshots anchor
        JOIN cases c ON c.case_id = anchor.case_id
        JOIN extractions e ON e.case_id = anchor.case_id
            AND e.execution_generation = anchor.execution_generation
        JOIN identity_results identity_row ON identity_row.case_id = anchor.case_id
            AND identity_row.execution_generation = anchor.execution_generation
        JOIN comparison_results inventory ON inventory.case_id = anchor.case_id
            AND inventory.execution_generation = anchor.execution_generation
            AND inventory.comparison_type = 'inventory'
        JOIN comparison_results risk ON risk.case_id = anchor.case_id
            AND risk.execution_generation = anchor.execution_generation
            AND risk.comparison_type = 'risk'
        JOIN critique_results critique ON critique.case_id = anchor.case_id
            AND critique.execution_generation = anchor.execution_generation
        WHERE typeof(anchor.execution_generation) = 'integer'
            AND anchor.execution_generation > 0
            AND c.execution_generation = anchor.execution_generation
            AND length(anchor.evidence_snapshot_digest) = 64
            AND anchor.evidence_snapshot_digest NOT GLOB '*[^0-9a-f]*'
            AND e.version = (
                SELECT MAX(latest.version) FROM extractions latest
                WHERE latest.case_id = anchor.case_id
                    AND latest.execution_generation = anchor.execution_generation
            )
            AND identity_row.rowid = (
                SELECT MAX(latest.rowid) FROM identity_results latest
                WHERE latest.case_id = anchor.case_id
                    AND latest.execution_generation = anchor.execution_generation
            )
            AND inventory.rowid = (
                SELECT MAX(latest.rowid) FROM comparison_results latest
                WHERE latest.case_id = anchor.case_id
                    AND latest.execution_generation = anchor.execution_generation
                    AND latest.comparison_type = 'inventory'
            )
            AND risk.rowid = (
                SELECT MAX(latest.rowid) FROM comparison_results latest
                WHERE latest.case_id = anchor.case_id
                    AND latest.execution_generation = anchor.execution_generation
                    AND latest.comparison_type = 'risk'
            )
            AND critique.rowid = (
                SELECT MAX(latest.rowid) FROM critique_results latest
                WHERE latest.case_id = anchor.case_id
                    AND latest.execution_generation = anchor.execution_generation
            )
            AND json_valid(e.payload_json) = 1
            AND json_valid(identity_row.payload_json) = 1
            AND json_valid(inventory.payload_json) = 1
            AND json_valid(risk.payload_json) = 1
            AND json_valid(critique.payload_json) = 1
            AND anchor.evidence_snapshot_digest = CASE
                WHEN json_valid(e.payload_json) = 1
                    AND json_valid(identity_row.payload_json) = 1
                    AND json_valid(inventory.payload_json) = 1
                    AND json_valid(risk.payload_json) = 1
                    AND json_valid(critique.payload_json) = 1
                THEN stored_evidence_snapshot_digest(
                    anchor.case_id,
                    e.payload_json,
                    identity_row.payload_json,
                    identity_row.evaluated_at,
                    inventory.payload_json,
                    risk.payload_json,
                    critique.payload_json
                )
                ELSE NULL
            END
            AND anchor.policy_review_required = CASE
                WHEN json_array_length(json_extract(
                    risk.payload_json, '$.policy_review_reasons'
                )) > 0 THEN 1 ELSE 0 END
            AND anchor.unresolved_blocker_count = CASE
                WHEN json_valid(risk.payload_json) = 1
                    AND (
                        anchor.review_id IS NULL
                        OR EXISTS (
                            SELECT 1 FROM human_decisions h
                            WHERE h.review_id = anchor.review_id
                                AND json_valid(h.payload_json) = 1
                        )
                    )
                THEN stored_unresolved_blocker_count(
                    risk.payload_json,
                    (SELECT h.payload_json FROM human_decisions h
                        WHERE h.review_id = anchor.review_id)
                )
                ELSE NULL
            END
            AND anchor.critique_disposition = json_extract(
                critique.payload_json, '$.recommended_disposition'
            )
            AND (
                (
                    anchor.review_id IS NULL
                    AND anchor.review_snapshot_digest IS NULL
                    AND anchor.policy_review_required = 0
                    AND NOT EXISTS (
                        SELECT 1 FROM review_requests r
                        WHERE r.case_id = anchor.case_id
                            AND r.execution_generation = anchor.execution_generation
                    )
                )
                OR EXISTS (
                    SELECT 1
                    FROM review_requests r
                    JOIN human_decisions h ON h.review_id = r.review_id
                    WHERE r.review_id = anchor.review_id
                        AND r.case_id = anchor.case_id
                        AND r.execution_generation = anchor.execution_generation
                        AND r.sequence = (
                            SELECT MAX(latest.sequence) FROM review_requests latest
                            WHERE latest.case_id = anchor.case_id
                        )
                        AND r.status = 'RESOLVED'
                        AND r.resolved_at = h.decided_at
                        AND r.evidence_snapshot_digest = anchor.review_snapshot_digest
                        AND (
                            (
                                h.decision <> 'ESTABLISH_MAPPING'
                                AND r.evidence_snapshot_digest =
                                    anchor.evidence_snapshot_digest
                            )
                            OR (
                                h.decision = 'ESTABLISH_MAPPING'
                                AND r.execution_generation > 1
                                AND r.evidence_snapshot_digest = (
                                    SELECT stored_evidence_snapshot_digest(
                                        anchor.case_id,
                                        predecessor_extraction.payload_json,
                                        predecessor_identity.payload_json,
                                        predecessor_identity.evaluated_at,
                                        predecessor_inventory.payload_json,
                                        predecessor_risk.payload_json,
                                        predecessor_critique.payload_json
                                    )
                                    FROM extractions predecessor_extraction
                                    JOIN identity_results predecessor_identity
                                        ON predecessor_identity.case_id =
                                            predecessor_extraction.case_id
                                    JOIN comparison_results predecessor_inventory
                                        ON predecessor_inventory.case_id =
                                            predecessor_extraction.case_id
                                        AND predecessor_inventory.comparison_type = 'inventory'
                                    JOIN comparison_results predecessor_risk
                                        ON predecessor_risk.case_id =
                                            predecessor_extraction.case_id
                                        AND predecessor_risk.comparison_type = 'risk'
                                    JOIN critique_results predecessor_critique
                                        ON predecessor_critique.case_id =
                                            predecessor_extraction.case_id
                                    WHERE predecessor_extraction.case_id = anchor.case_id
                                        AND predecessor_extraction.execution_generation =
                                            anchor.execution_generation - 1
                                        AND predecessor_identity.execution_generation =
                                            anchor.execution_generation - 1
                                        AND predecessor_inventory.execution_generation =
                                            anchor.execution_generation - 1
                                        AND predecessor_risk.execution_generation =
                                            anchor.execution_generation - 1
                                        AND predecessor_critique.execution_generation =
                                            anchor.execution_generation - 1
                                        AND predecessor_extraction.version = (
                                            SELECT MAX(latest.version) FROM extractions latest
                                            WHERE latest.case_id = anchor.case_id
                                                AND latest.execution_generation =
                                                    anchor.execution_generation - 1
                                        )
                                        AND predecessor_identity.rowid = (
                                            SELECT MAX(latest.rowid) FROM identity_results latest
                                            WHERE latest.case_id = anchor.case_id
                                                AND latest.execution_generation =
                                                    anchor.execution_generation - 1
                                        )
                                        AND predecessor_inventory.rowid = (
                                            SELECT MAX(latest.rowid) FROM comparison_results latest
                                            WHERE latest.case_id = anchor.case_id
                                                AND latest.execution_generation =
                                                    anchor.execution_generation - 1
                                                AND latest.comparison_type = 'inventory'
                                        )
                                        AND predecessor_risk.rowid = (
                                            SELECT MAX(latest.rowid) FROM comparison_results latest
                                            WHERE latest.case_id = anchor.case_id
                                                AND latest.execution_generation =
                                                    anchor.execution_generation - 1
                                                AND latest.comparison_type = 'risk'
                                        )
                                        AND predecessor_critique.rowid = (
                                            SELECT MAX(latest.rowid) FROM critique_results latest
                                            WHERE latest.case_id = anchor.case_id
                                                AND latest.execution_generation =
                                                    anchor.execution_generation - 1
                                        )
                                )
                            )
                        )
                        AND json_valid(r.payload_json) = 1
                        AND json_valid(h.payload_json) = 1
                        AND json_extract(r.payload_json, '$.review_id') = r.review_id
                        AND json_extract(r.payload_json, '$.case_id') = r.case_id
                        AND json_extract(r.payload_json, '$.sequence') = r.sequence
                        AND json_extract(r.payload_json, '$.status') = r.status
                        AND json_extract(r.payload_json, '$.human_decision.review_id') = h.review_id
                        AND json_extract(r.payload_json, '$.human_decision.reviewer') = h.reviewer
                        AND json_extract(r.payload_json, '$.human_decision.decision') = h.decision
                        AND json_extract(r.payload_json, '$.human_decision.reason') = h.reason
                        AND julianday(json_extract(
                            r.payload_json, '$.human_decision.decided_at'
                        )) = julianday(h.decided_at)
                        AND json_extract(h.payload_json, '$.review_id') = h.review_id
                        AND json_extract(h.payload_json, '$.reviewer') = h.reviewer
                        AND json_extract(h.payload_json, '$.decision') = h.decision
                        AND json_extract(h.payload_json, '$.reason') = h.reason
                        AND julianday(json_extract(h.payload_json, '$.decided_at')) =
                            julianday(h.decided_at)
                        AND json_extract(h.payload_json, '$.mappings') = json_extract(
                            r.payload_json, '$.human_decision.mappings'
                        )
                        AND json_extract(h.payload_json, '$.superseded_case_id') IS json_extract(
                            r.payload_json, '$.human_decision.superseded_case_id'
                        )
                        AND json_extract(h.payload_json, '$.addressed_blocker_ids') = json_extract(
                            r.payload_json, '$.human_decision.addressed_blocker_ids'
                        )
                        AND (SELECT COUNT(*) FROM human_decisions exact
                            WHERE exact.review_id = r.review_id) = 1
                )
            )
    """
    invalid_snapshot_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM validated_evidence_snapshots anchor "
            f"WHERE NOT EXISTS (SELECT 1 FROM ({valid_anchors_sql}) valid "
            "WHERE valid.case_id = anchor.case_id "
            "AND valid.execution_generation = anchor.execution_generation "
            "AND valid.evidence_snapshot_digest = anchor.evidence_snapshot_digest)"
        ).fetchone()[0]
    )
    invalid_final_decision_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM final_decisions f
                WHERE typeof(f.decision_generation) <> 'integer'
                    OR f.decision_generation < 1
                    OR json_valid(f.payload_json) <> 1
                    OR json_extract(
                        CASE WHEN json_valid(f.payload_json) = 1
                            THEN f.payload_json ELSE '{}' END,
                        '$.decision'
                    )
                        NOT IN ('APPROVE', 'REJECT', 'HOLD', 'FAILED')
                    OR NOT (
                        (json_extract(
                            CASE WHEN json_valid(f.payload_json) = 1
                                THEN f.payload_json ELSE '{}' END,
                            '$.decision'
                        ) = 'APPROVE'
                            AND json_extract(
                                CASE WHEN json_valid(f.payload_json) = 1
                                    THEN f.payload_json ELSE '{}' END,
                                '$.payment_eligible'
                            ) = 1)
                        OR (json_extract(
                            CASE WHEN json_valid(f.payload_json) = 1
                                THEN f.payload_json ELSE '{}' END,
                            '$.decision'
                        ) <> 'APPROVE'
                            AND json_extract(
                                CASE WHEN json_valid(f.payload_json) = 1
                                    THEN f.payload_json ELSE '{}' END,
                                '$.payment_eligible'
                            ) = 0)
                )
                OR NOT EXISTS (
                    SELECT 1
                    FROM ("""
            + valid_anchors_sql
            + """
                    ) anchor
                    JOIN cases c ON c.case_id = f.case_id
                    JOIN extractions e ON e.case_id = f.case_id
                        AND e.execution_generation = f.decision_generation
                    WHERE anchor.case_id = f.case_id
                        AND anchor.execution_generation = f.decision_generation
                        AND anchor.evidence_snapshot_digest = f.evidence_snapshot_digest
                        AND anchor.review_id IS f.review_id
                        AND e.version = (
                            SELECT MAX(latest.version) FROM extractions latest
                            WHERE latest.case_id = f.case_id
                                AND latest.execution_generation = f.decision_generation
                        )
                        AND c.source_id = f.source_id
                        AND json_extract(e.payload_json, '$.source.source_id') = f.source_id
                        AND json_extract(e.payload_json, '$.invoice_number.normalized_value')
                            IS f.invoice_number
                        AND json_extract(e.payload_json, '$.vendor.normalized_value') IS f.vendor
                        AND json_extract(e.payload_json, '$.declared_total') IS f.authorized_amount
                        AND json_extract(e.payload_json, '$.currency.normalized_value')
                            IS f.authorized_currency
                        AND f.payment_idempotency_key = payment_identity_key(
                            f.vendor, f.invoice_number
                        )
                        AND (
                            json_extract(f.payload_json, '$.decision') <> 'APPROVE'
                            OR anchor.unresolved_blocker_count = 0
                        )
                        AND (
                            json_extract(f.payload_json, '$.decision') <> 'APPROVE'
                            OR anchor.critique_disposition = 'APPROVE'
                            OR anchor.review_id IS NOT NULL
                        )
                        AND (
                            (
                                anchor.review_id IS NULL
                                AND json_extract(
                                    f.payload_json, '$.human_outcome'
                                ) IS NULL
                            )
                            OR EXISTS (
                                SELECT 1
                                FROM human_decisions h
                                WHERE h.review_id = anchor.review_id
                                    AND json_extract(
                                        f.payload_json, '$.human_outcome.review_id'
                                    ) = h.review_id
                                    AND json_extract(
                                        f.payload_json, '$.human_outcome.reviewer'
                                    ) = h.reviewer
                                    AND json_extract(
                                        f.payload_json, '$.human_outcome.decision'
                                    ) = h.decision
                                    AND json_extract(
                                        f.payload_json, '$.human_outcome.reason'
                                    ) = h.reason
                                    AND julianday(json_extract(
                                        f.payload_json, '$.human_outcome.decided_at'
                                    )) = julianday(h.decided_at)
                                    AND json_extract(
                                        f.payload_json, '$.human_outcome.mappings'
                                    ) = json_extract(h.payload_json, '$.mappings')
                                    AND json_extract(
                                        f.payload_json,
                                        '$.human_outcome.superseded_case_id'
                                    ) IS json_extract(
                                        h.payload_json, '$.superseded_case_id'
                                    )
                                    AND json_extract(
                                        f.payload_json,
                                        '$.human_outcome.addressed_blocker_ids'
                                    ) = json_extract(
                                        h.payload_json, '$.addressed_blocker_ids'
                                    )
                                    AND (
                                        (
                                            h.decision IN (
                                                'APPROVE',
                                                'ESTABLISH_MAPPING',
                                                'SUPERSEDE_REVISION'
                                            )
                                            AND (
                                                json_extract(
                                                    f.payload_json, '$.decision'
                                                ) = 'APPROVE'
                                                OR (
                                                    json_extract(
                                                        f.payload_json, '$.decision'
                                                    ) = 'HOLD'
                                                    AND anchor.unresolved_blocker_count > 0
                                                )
                                            )
                                        )
                                        OR (
                                            h.decision = 'REJECT'
                                            AND json_extract(
                                                f.payload_json, '$.decision'
                                            ) = 'REJECT'
                                        )
                                        OR (
                                            h.decision = 'REQUEST_CORRECTION'
                                            AND json_extract(
                                                f.payload_json, '$.decision'
                                            ) = 'HOLD'
                                        )
                                    )
                            )
                        )
                )
            """
        ).fetchone()[0]
    )
    invalid_payment_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM payments p
            WHERE typeof(p.decision_generation) <> 'integer'
                OR p.decision_generation < 1
                OR p.status NOT IN ('PAID', 'FAILED')
                OR (p.status = 'PAID' AND p.error IS NOT NULL)
                OR (p.status = 'FAILED' AND p.error IS NULL)
                OR NOT EXISTS (
                    SELECT 1
                    FROM final_decisions f
                    JOIN ("""
            + valid_anchors_sql
            + """
                    ) anchor ON anchor.case_id = f.case_id
                        AND anchor.execution_generation = f.decision_generation
                        AND anchor.evidence_snapshot_digest = f.evidence_snapshot_digest
                    JOIN cases c ON c.case_id = f.case_id
                    WHERE f.case_id = p.case_id
                        AND f.decision_generation = p.decision_generation
                        AND f.evidence_snapshot_digest = p.evidence_snapshot_digest
                        AND json_valid(f.payload_json) = 1
                            AND json_extract(
                                CASE WHEN json_valid(f.payload_json) = 1
                                    THEN f.payload_json ELSE '{}' END,
                                '$.decision'
                            ) = 'APPROVE'
                            AND json_extract(
                                CASE WHEN json_valid(f.payload_json) = 1
                                    THEN f.payload_json ELSE '{}' END,
                                '$.payment_eligible'
                            ) = 1
                        AND f.source_id = p.source_id
                        AND f.invoice_number IS p.invoice_number
                        AND f.vendor = p.vendor
                        AND f.authorized_amount = p.amount
                        AND f.authorized_currency = p.currency
                        AND f.payment_idempotency_key = p.idempotency_key
                        AND f.review_id IS p.review_id
                        AND anchor.review_id IS p.review_id
                        AND anchor.unresolved_blocker_count = 0
                        AND (anchor.critique_disposition = 'APPROVE'
                            OR anchor.review_id IS NOT NULL)
                        AND c.source_id = p.source_id
                        AND p.idempotency_key = payment_identity_key(p.vendor, p.invoice_number)
                        AND CAST(p.amount AS NUMERIC) > 0
                )
            """
        ).fetchone()[0]
    )
    return {
        "invalid_snapshot_count": invalid_snapshot_count,
        "invalid_final_decision_count": invalid_final_decision_count,
        "invalid_payment_count": invalid_payment_count,
    }


def verify_database(
    path: Path,
    kind: DatabaseKind | None = None,
    *,
    require_seed: bool = True,
) -> dict[str, object]:
    """Verify signature, integrity, version, required schema, indexes, and seed identity."""

    selected_kind = kind or infer_kind(path)
    resolved = path.resolve()
    _assert_sqlite_file(resolved)
    required: dict[str, set[str]]
    if selected_kind is DatabaseKind.INVENTORY:
        required = {
            "schema_version": {"version", "applied_at"},
            "inventory": {"sku", "item_name", "available_stock"},
            "item_aliases": {
                "alias_normalized",
                "sku",
                "source",
                "approved_by",
                "approved_at",
            },
        }
    else:
        required = {
            "schema_version": {"version", "applied_at"},
            "source_artifacts": {"source_id", "source_hash", "metadata_json"},
            "cases": {
                "case_id",
                "source_id",
                "status",
                "team_state_json",
                "execution_token",
                "execution_generation",
                "execution_state",
                "lease_expires_at",
            },
            "extractions": {"case_id", "payload_json", "execution_generation"},
            "identity_results": {
                "case_id",
                "payload_json",
                "execution_generation",
                "evaluated_at",
            },
            "comparison_results": {
                "case_id",
                "comparison_type",
                "payload_json",
                "execution_generation",
            },
            "critique_results": {"case_id", "payload_json", "execution_generation"},
            "review_requests": {
                "review_id",
                "case_id",
                "sequence",
                "status",
                "payload_json",
                "execution_generation",
                "evidence_snapshot_digest",
            },
            "human_decisions": {"review_id", "reviewer", "decision"},
            "validated_evidence_snapshots": {
                "case_id",
                "execution_generation",
                "evidence_snapshot_digest",
                "policy_review_required",
                "unresolved_blocker_count",
                "critique_disposition",
                "review_id",
                "review_snapshot_digest",
                "validated_at",
            },
            "final_decisions": {
                "case_id",
                "payload_json",
                "decision_generation",
                "evidence_snapshot_digest",
                "source_id",
                "invoice_number",
                "vendor",
                "authorized_amount",
                "authorized_currency",
                "payment_idempotency_key",
                "review_id",
            },
            "payments": {
                "case_id",
                "idempotency_key",
                "status",
                "decision_generation",
                "evidence_snapshot_digest",
                "source_id",
                "invoice_number",
                "review_id",
            },
            "events": {"case_id", "event_type", "payload_json"},
        }
    try:
        with connect_database(resolved, read_only=True) as connection:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    f"SQLite integrity_check returned {integrity}",
                    stop_reason="DATABASE_INTEGRITY_FAILED",
                )
            version_row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_version"
            ).fetchone()
            version = int(version_row["version"] or 0)
            required_version = SCHEMA_VERSIONS[selected_kind]
            if version != required_version:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    f"schema version {version} does not match required {required_version} "
                    f"for {selected_kind.value}",
                    stop_reason="DATABASE_VERSION_MISMATCH",
                )
            for table, expected_columns in required.items():
                actual_columns = _columns(connection, table)
                missing = expected_columns - actual_columns
                if missing:
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        f"table {table} is missing columns: {sorted(missing)}",
                        stop_reason="DATABASE_SCHEMA_MISMATCH",
                    )
            if selected_kind is DatabaseKind.INVENTORY and require_seed:
                actual_rows = {
                    (str(row["sku"]), str(row["item_name"]), int(row["available_stock"]))
                    for row in connection.execute(
                        "SELECT sku, item_name, available_stock FROM inventory"
                    ).fetchall()
                }
                missing_rows = set(SEED_ROWS) - actual_rows
                if missing_rows:
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        f"inventory is missing expected seed identities: {sorted(missing_rows)}",
                        stop_reason="INVENTORY_SEED_MISMATCH",
                    )
            indexes = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            missing_indexes = REQUIRED_INDEXES[selected_kind] - indexes
            if missing_indexes:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    f"required indexes are missing: {sorted(missing_indexes)}",
                    stop_reason="DATABASE_SCHEMA_MISMATCH",
                )
            if selected_kind is DatabaseKind.WORKFLOW:
                trigger_rows = connection.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
                ).fetchall()
                actual_triggers = {
                    str(row["name"]): _normalized_sql(str(row["sql"]))
                    for row in trigger_rows
                    if row["sql"] is not None
                }
                invalid_triggers = {
                    name
                    for name, definition in REQUIRED_WORKFLOW_TRIGGERS.items()
                    if actual_triggers.get(name) != _normalized_sql(definition)
                }
                if invalid_triggers:
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        f"required triggers are missing or changed: {sorted(invalid_triggers)}",
                        stop_reason="DATABASE_SCHEMA_MISMATCH",
                    )
                invalid_authorization = _workflow_authorization_provenance_counts(connection)
                if any(invalid_authorization.values()):
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        "workflow authorization provenance is incomplete or inconsistent "
                        f"({', '.join(f'{key}={value}' for key, value in invalid_authorization.items())})",
                        stop_reason="DATABASE_AUTHORIZATION_PROVENANCE_INVALID",
                        details=invalid_authorization,
                    )
    except DatabaseVerificationError:
        raise
    except sqlite3.Error as exc:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"database verification failed: {exc}",
            stop_reason="DATABASE_VERIFICATION_ERROR",
        ) from exc
    return {
        "path": str(resolved),
        "kind": selected_kind.value,
        "schema_version": version,
        "integrity": "ok",
        "tables": sorted(required),
        "indexes": sorted(indexes),
    }
