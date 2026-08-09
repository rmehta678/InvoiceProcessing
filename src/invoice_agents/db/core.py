"""Explicit, versioned SQLite setup and strict preflight verification."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING

from invoice_agents.errors import DatabaseVerificationError, ErrorCategory

if TYPE_CHECKING:
    from invoice_agents.config import Settings

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


@dataclass(frozen=True, slots=True)
class LegacyAuthorizationReconciliationReceipt:
    reconciliation_id: str
    record_count: int
    table_counts: dict[str, int]
    record_manifest_hash: str
    confirmed_at: str


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
            "idx_legacy_authorization_quarantine_reconciliation",
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
    """Use packaged schema SQL itself as the exact preflight trigger contract."""

    root = files("invoice_agents.db").joinpath("migrations", "workflow")
    scripts = (
        root.joinpath("003_execution_fencing.sql").read_text(encoding="utf-8"),
        root.joinpath("legacy_authorization_archive.sql").read_text(encoding="utf-8"),
    )
    definitions = {
        match.group(1): match.group(0)
        for script in scripts
        for match in re.finditer(
            r"CREATE\s+TRIGGER(?:\s+IF\s+NOT\s+EXISTS)?\s+([A-Za-z0-9_]+)\b.*?\bEND\s*;",
            script,
            flags=re.IGNORECASE | re.DOTALL,
        )
    }
    if not definitions:
        raise RuntimeError("packaged workflow schema defines no triggers")
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


def _legacy_archive_statements() -> list[str]:
    script = (
        files("invoice_agents.db")
        .joinpath("migrations", "workflow", "legacy_authorization_archive.sql")
        .read_text(encoding="utf-8")
    )
    return _migration_statements(script)


def _install_legacy_archive_schema(connection: sqlite3.Connection) -> None:
    for statement in _legacy_archive_statements():
        connection.execute(statement)


def _normalized_sql(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", sql.strip().rstrip(";")).casefold()
    # SQLite does not preserve the optional creation guard consistently in
    # sqlite_master. It has no bearing on the installed object's definition.
    return re.sub(
        r"\b(create (?:index|table|trigger)) if not exists\b",
        r"\1",
        normalized,
    )


def _require_reconciled_workflow_authorization(
    connection: sqlite3.Connection,
) -> None:
    """Refuse to invent generation-bound provenance for pre-v3 decisions or payments."""

    counts = {
        "review_request_count": int(
            connection.execute("SELECT COUNT(*) FROM review_requests").fetchone()[0]
        ),
        "human_decision_count": int(
            connection.execute("SELECT COUNT(*) FROM human_decisions").fetchone()[0]
        ),
        "final_decision_count": int(
            connection.execute("SELECT COUNT(*) FROM final_decisions").fetchone()[0]
        ),
        "payment_count": int(connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0]),
    }
    if any(counts.values()):
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            "workflow authorization reconciliation is required before migration 003 "
            f"({', '.join(f'{key}={value}' for key, value in counts.items())})",
            stop_reason="AUTHORIZATION_RECONCILIATION_REQUIRED",
            details=counts,
        )


def reconcile_legacy_authorization(
    path: Path,
    *,
    reviewer: str,
    reason: str,
    disposition: str,
    confirmed: bool,
) -> LegacyAuthorizationReconciliationReceipt:
    """Permanently quarantine exact legacy authority only after explicit confirmation."""

    from invoice_agents.db.legacy_archive import (
        LEGACY_ACTIVE_TABLE_KEYS,
        LEGACY_NON_AUTHORIZING_DISPOSITION,
        audit_legacy_authorization_archives,
        canonical_legacy_json,
        legacy_manifest_hash,
        legacy_reconciliation_id,
        legacy_record_hash,
    )

    selected_reviewer = reviewer.strip()
    selected_reason = reason.strip()
    if not confirmed:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            "legacy authorization reconciliation requires explicit operator confirmation",
            stop_reason="LEGACY_RECONCILIATION_CONFIRMATION_REQUIRED",
        )
    if (
        not selected_reviewer
        or not selected_reason
        or disposition != LEGACY_NON_AUTHORIZING_DISPOSITION
    ):
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            "legacy reconciliation requires reviewer, reason, and the permanent "
            "non-authorizing disposition",
            stop_reason="LEGACY_RECONCILIATION_INPUT_INVALID",
        )
    resolved = path.resolve()
    _assert_sqlite_file(resolved)
    with connect_database(resolved) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            version_row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_version"
            ).fetchone()
            version = int(version_row["version"] or 0)
            archive_schema_exists = bool(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'legacy_authorization_reconciliations'"
                ).fetchone()
            )
            active_rows = {
                table: [
                    dict(row)
                    for row in connection.execute(
                        f"SELECT * FROM {table} ORDER BY {key}"
                    ).fetchall()
                ]
                for table, key in LEGACY_ACTIVE_TABLE_KEYS.items()
            }
            active_count = sum(len(rows) for rows in active_rows.values())
            if archive_schema_exists:
                metadata_rows = connection.execute(
                    "SELECT reconciliation_id, reviewer, reason, disposition, confirmed_at, "
                    "record_count, record_manifest_hash, table_counts_json "
                    "FROM legacy_authorization_reconciliations ORDER BY reconciliation_id"
                ).fetchall()
                if metadata_rows:
                    if active_count or len(metadata_rows) != 1:
                        raise DatabaseVerificationError(
                            ErrorCategory.DATABASE,
                            "legacy reconciliation archive and active authorization state conflict",
                            stop_reason="LEGACY_RECONCILIATION_STATE_INVALID",
                        )
                    if audit_legacy_authorization_archives(connection):
                        raise DatabaseVerificationError(
                            ErrorCategory.DATABASE,
                            "legacy reconciliation archive failed integrity verification",
                            stop_reason="LEGACY_RECONCILIATION_ARCHIVE_INVALID",
                        )
                    metadata = metadata_rows[0]
                    if (
                        metadata["reviewer"] != selected_reviewer
                        or metadata["reason"] != selected_reason
                        or metadata["disposition"] != disposition
                    ):
                        raise DatabaseVerificationError(
                            ErrorCategory.DATABASE,
                            "legacy reconciliation retry does not match the completed operation",
                            stop_reason="LEGACY_RECONCILIATION_REPLAY_MISMATCH",
                        )
                    table_counts = json.loads(str(metadata["table_counts_json"]))
                    if not isinstance(table_counts, dict):
                        raise DatabaseVerificationError(
                            ErrorCategory.DATABASE,
                            "legacy reconciliation archive metadata is invalid",
                            stop_reason="LEGACY_RECONCILIATION_ARCHIVE_INVALID",
                        )
                    receipt = LegacyAuthorizationReconciliationReceipt(
                        reconciliation_id=str(metadata["reconciliation_id"]),
                        record_count=int(metadata["record_count"]),
                        table_counts={str(key): int(value) for key, value in table_counts.items()},
                        record_manifest_hash=str(metadata["record_manifest_hash"]),
                        confirmed_at=str(metadata["confirmed_at"]),
                    )
                    connection.rollback()
                    return receipt
            if version not in (1, 2):
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "legacy authorization reconciliation applies only before workflow v3",
                    stop_reason="LEGACY_RECONCILIATION_NOT_APPLICABLE",
                )
            if not active_count:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "no active legacy authorization rows require reconciliation",
                    stop_reason="LEGACY_AUTHORIZATION_NOT_FOUND",
                )

            _install_legacy_archive_schema(connection)
            table_counts = {table: len(rows) for table, rows in active_rows.items()}
            archived_at = utc_now()
            archived_records: list[tuple[str, str, str, str]] = []
            for table, key_column in LEGACY_ACTIVE_TABLE_KEYS.items():
                for row in active_rows[table]:
                    key = str(row[key_column])
                    original_json = canonical_legacy_json(row)
                    record_hash = legacy_record_hash(table, key, original_json)
                    archived_records.append((table, key, original_json, record_hash))
            manifest_hash = legacy_manifest_hash(
                [(table, key, record_hash) for table, key, _json, record_hash in archived_records]
            )
            reconciliation_id = legacy_reconciliation_id(
                reviewer=selected_reviewer,
                reason=selected_reason,
                disposition=disposition,
                source_schema_version=version,
                manifest_hash=manifest_hash,
            )
            connection.execute(
                "INSERT INTO legacy_authorization_reconciliations("
                "reconciliation_id, reviewer, reason, disposition, confirmed_at, "
                "source_schema_version, record_count, record_manifest_hash, table_counts_json, "
                "state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED')",
                (
                    reconciliation_id,
                    selected_reviewer,
                    selected_reason,
                    disposition,
                    archived_at,
                    version,
                    len(archived_records),
                    manifest_hash,
                    canonical_legacy_json(table_counts),
                ),
            )
            for table, key, original_json, record_hash in archived_records:
                connection.execute(
                    "INSERT INTO legacy_authorization_quarantine("
                    "archive_id, reconciliation_id, source_table, source_record_key, "
                    "original_row_json, record_hash, authorization_state, archived_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"lqar_{record_hash}",
                        reconciliation_id,
                        table,
                        key,
                        original_json,
                        record_hash,
                        disposition,
                        archived_at,
                    ),
                )
            if audit_legacy_authorization_archives(connection):
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "legacy rows were not archived with exact canonical hashes",
                    stop_reason="LEGACY_RECONCILIATION_ARCHIVE_INVALID",
                )
            for table in ("payments", "final_decisions", "human_decisions", "review_requests"):
                connection.execute(f"DELETE FROM {table}")
            remaining = sum(
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in LEGACY_ACTIVE_TABLE_KEYS
            )
            if remaining:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "legacy active authorization rows remained after verified archival",
                    stop_reason="LEGACY_RECONCILIATION_DELETE_INCOMPLETE",
                )
            connection.commit()
            return LegacyAuthorizationReconciliationReceipt(
                reconciliation_id=reconciliation_id,
                record_count=len(archived_records),
                table_counts=table_counts,
                record_manifest_hash=manifest_hash,
                confirmed_at=archived_at,
            )
        except DatabaseVerificationError:
            connection.rollback()
            raise
        except (sqlite3.Error, TypeError, ValueError) as exc:
            connection.rollback()
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                "legacy authorization reconciliation failed atomically",
                stop_reason="LEGACY_RECONCILIATION_FAILED",
            ) from exc


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
        if (
            selected_kind is DatabaseKind.WORKFLOW
            and SCHEMA_VERSIONS[DatabaseKind.WORKFLOW] in existing
        ):
            try:
                connection.execute("BEGIN IMMEDIATE")
                _install_legacy_archive_schema(connection)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "version-neutral workflow schema installation failed",
                    stop_reason="MIGRATION_FAILED",
                ) from exc
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
                    _install_legacy_archive_schema(connection)
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


def ensure_databases(settings: Settings) -> dict[str, list[int]]:
    """Migrate, seed, and verify both databases; idempotent, so safe on every start.

    Case processing still never repairs a database - this is an explicit setup
    entry point shared by the CLI so one command can bring the system up.
    """

    inventory_db = settings.inventory_db
    workflow_db = settings.workflow_db
    applied = {
        DatabaseKind.INVENTORY.value: migrate_database(inventory_db, DatabaseKind.INVENTORY),
        DatabaseKind.WORKFLOW.value: migrate_database(workflow_db, DatabaseKind.WORKFLOW),
    }
    seed_inventory(inventory_db)
    verify_database(inventory_db, DatabaseKind.INVENTORY)
    verify_database(workflow_db, DatabaseKind.WORKFLOW, settings=settings)
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


def _verify_attached_inventory_context(connection: sqlite3.Connection) -> None:
    """Verify the exact attached inventory snapshot used by workflow authorization."""

    integrity = str(
        connection.execute("PRAGMA authorization_inventory.integrity_check").fetchone()[0]
    )
    if integrity != "ok":
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"attached inventory integrity_check returned {integrity}",
            stop_reason="DATABASE_INTEGRITY_FAILED",
        )
    version_row = connection.execute(
        "SELECT MAX(version) AS version FROM authorization_inventory.schema_version"
    ).fetchone()
    version = int(version_row["version"] or 0)
    if version != SCHEMA_VERSIONS[DatabaseKind.INVENTORY]:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            "attached inventory schema version does not match the required authorization context",
            stop_reason="DATABASE_VERSION_MISMATCH",
        )
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
    for table, expected_columns in required.items():
        actual_columns = {
            str(row["name"])
            for row in connection.execute(f'PRAGMA authorization_inventory.table_info("{table}")')
        }
        missing = expected_columns - actual_columns
        if missing:
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                f"attached inventory table {table} is missing columns: {sorted(missing)}",
                stop_reason="DATABASE_SCHEMA_MISMATCH",
            )
    indexes = {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM authorization_inventory.sqlite_master WHERE type = 'index'"
        )
    }
    missing_indexes = REQUIRED_INDEXES[DatabaseKind.INVENTORY] - indexes
    if missing_indexes:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"attached inventory indexes are missing: {sorted(missing_indexes)}",
            stop_reason="DATABASE_SCHEMA_MISMATCH",
        )


def verify_database(
    path: Path,
    kind: DatabaseKind | None = None,
    *,
    require_seed: bool = True,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Verify signature, integrity, version, required schema, indexes, and seed identity."""

    selected_kind = kind or infer_kind(path)
    resolved = path.resolve()
    _assert_sqlite_file(resolved)
    inventory_resolved: Path | None = None
    if selected_kind is DatabaseKind.WORKFLOW:
        if settings is None:
            raise DatabaseVerificationError(
                ErrorCategory.CONFIGURATION,
                "workflow schema v3 authorization verification requires explicit Settings",
                stop_reason="DATABASE_AUTHORIZATION_CONTEXT_REQUIRED",
            )
        if settings.workflow_db.resolve() != resolved:
            raise DatabaseVerificationError(
                ErrorCategory.CONFIGURATION,
                "workflow verification Settings do not identify the database being audited",
                stop_reason="DATABASE_AUTHORIZATION_CONTEXT_MISMATCH",
            )
        inventory_resolved = settings.inventory_db.resolve()
        _assert_sqlite_file(inventory_resolved)
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
            "legacy_authorization_reconciliations": {
                "reconciliation_id",
                "reviewer",
                "reason",
                "disposition",
                "confirmed_at",
                "source_schema_version",
                "record_count",
                "record_manifest_hash",
                "table_counts_json",
                "state",
            },
            "legacy_authorization_quarantine": {
                "archive_id",
                "reconciliation_id",
                "source_table",
                "source_record_key",
                "original_row_json",
                "record_hash",
                "authorization_state",
                "archived_at",
            },
            "events": {"case_id", "event_type", "payload_json"},
        }
    try:
        with connect_database(resolved, read_only=True) as connection:
            if inventory_resolved is not None:
                inventory_uri = f"{inventory_resolved.as_uri()}?mode=ro"
                connection.execute(
                    "ATTACH DATABASE ? AS authorization_inventory",
                    (inventory_uri,),
                )
                connection.execute("BEGIN")
                # One statement touches both schemas, fixing both read snapshots
                # before any authorization component is enumerated.
                connection.execute(
                    "SELECT (SELECT COUNT(*) FROM main.sqlite_schema), "
                    "(SELECT COUNT(*) FROM authorization_inventory.sqlite_schema)"
                ).fetchone()
                _verify_attached_inventory_context(connection)
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
                from invoice_agents.db.authorization_audit import audit_workflow_authorization

                assert settings is not None
                invalid_authorization = audit_workflow_authorization(
                    connection,
                    settings,
                    inventory_schema="authorization_inventory",
                )
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
