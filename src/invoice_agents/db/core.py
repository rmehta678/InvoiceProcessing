"""Explicit, versioned SQLite setup and strict preflight verification."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import cache
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


@dataclass(frozen=True, slots=True)
class SchemaColumnManifest:
    cid: int
    name: str
    declared_type: str
    not_null: int
    default_sql: str | None
    primary_key_position: int


@dataclass(frozen=True, slots=True)
class SchemaForeignKeyManifest:
    identifier: int
    sequence: int
    referenced_table: str
    from_column: str
    to_column: str | None
    on_update: str
    on_delete: str
    match: str


@dataclass(frozen=True, slots=True)
class SchemaIndexColumnManifest:
    sequence: int
    cid: int
    name: str | None
    descending: int
    collation: str
    key_column: int


@dataclass(frozen=True, slots=True)
class SchemaIndexManifest:
    sequence: int
    name: str
    unique: int
    origin: str
    partial: int
    columns: tuple[SchemaIndexColumnManifest, ...]
    normalized_sql: str | None


@dataclass(frozen=True, slots=True)
class SchemaTriggerManifest:
    name: str
    normalized_sql: str


@dataclass(frozen=True, slots=True)
class TableSchemaManifest:
    name: str
    normalized_sql: str
    columns: tuple[SchemaColumnManifest, ...]
    foreign_keys: tuple[SchemaForeignKeyManifest, ...]
    indexes: tuple[SchemaIndexManifest, ...]
    triggers: tuple[SchemaTriggerManifest, ...]


@dataclass(frozen=True, slots=True)
class SchemaObjectManifest:
    object_type: str
    name: str
    table_name: str
    normalized_sql: str | None


@dataclass(frozen=True, slots=True)
class WorkflowSchemaManifest:
    tables: tuple[TableSchemaManifest, ...]
    objects: tuple[SchemaObjectManifest, ...]


@dataclass(frozen=True, slots=True)
class MigrationHistorySnapshot:
    schema_version_rows: tuple[tuple[object, ...], ...]
    durable_history_rows: tuple[tuple[object, ...], ...] | None


@dataclass(frozen=True, slots=True)
class MigrationPreflight:
    history: tuple[int, ...]
    history_snapshot: MigrationHistorySnapshot
    version_neutral_install_required: bool


@dataclass(frozen=True, slots=True)
class WorkflowVersionNeutralState:
    applies: bool
    durable_history_exists: bool
    archive_install_required: bool

    @property
    def install_required(self) -> bool:
        return self.applies and (self.archive_install_required or not self.durable_history_exists)


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


def _migration_versions(resources: list[Traversable], kind: DatabaseKind) -> tuple[int, ...]:
    versions = tuple(int(resource.name.split("_", 1)[0]) for resource in resources)
    expected = tuple(range(1, len(versions) + 1))
    if versions != expected:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"packaged {kind.value} migration history is not contiguous",
            stop_reason="MIGRATION_HISTORY_INVALID",
        )
    return versions


def _read_migration_history(
    connection: sqlite3.Connection,
    *,
    kind: DatabaseKind,
    packaged_versions: tuple[int, ...],
    packaged_hashes: dict[int, str] | None = None,
    allow_durable_retrofit: bool = False,
) -> tuple[int, ...]:
    """Require legacy set-prefix semantics and durable ordered history from v3 onward."""

    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
        if table is None:
            user_objects = int(
                connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                ).fetchone()[0]
            )
            if user_objects:
                raise ValueError("database has schema objects but no migration history")
            return ()
        rows = connection.execute("SELECT version FROM schema_version").fetchall()
        raw_versions = tuple(row["version"] for row in rows)
        if any(type(version) is not int for version in raw_versions):
            raise ValueError("migration versions have invalid storage types")
        versions = tuple(sorted(raw_versions))
        if len(set(versions)) != len(versions):
            raise ValueError("migration versions are not unique")
        expected = tuple(range(1, len(versions) + 1))
        if (
            versions != expected
            or len(versions) > len(packaged_versions)
            or versions != packaged_versions[: len(versions)]
        ):
            raise ValueError("migration versions are not a packaged contiguous prefix")
        if not versions:
            unexpected_objects = int(
                connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' AND name <> 'schema_version'"
                ).fetchone()[0]
            )
            if unexpected_objects:
                raise ValueError("empty migration history has unexpected schema objects")
        durable_table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'schema_migration_history'"
        ).fetchone()
        if kind is not DatabaseKind.WORKFLOW or not versions or versions[-1] < 3:
            if durable_table is not None:
                raise ValueError("durable migration history exists before workflow v3")
            return versions
        if durable_table is None:
            durable_artifacts = int(
                connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = 'schema_migration_history' "
                    "OR name LIKE 'trg_schema_migration_history_%'"
                ).fetchone()[0]
            )
            if durable_artifacts:
                raise ValueError("partial durable migration history schema is invalid")
            if allow_durable_retrofit and versions == (1, 2, 3):
                return versions
            raise ValueError("durable workflow migration history is missing")
        if packaged_hashes is None:
            raise ValueError("packaged migration hashes are unavailable")
        _verify_durable_migration_history(
            connection,
            versions=versions,
            packaged_hashes=packaged_hashes,
        )
    except (sqlite3.Error, ValueError) as exc:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"{kind.value} migration history cannot be read",
            stop_reason="MIGRATION_HISTORY_INVALID",
        ) from exc
    return versions


def _migration_hashes(resources: list[Traversable]) -> dict[int, str]:
    return {
        int(resource.name.split("_", 1)[0]): hashlib.sha256(resource.read_bytes()).hexdigest()
        for resource in resources
    }


def _migration_history_snapshot(connection: sqlite3.Connection) -> MigrationHistorySnapshot:
    """Capture every migration-history value so a valid-looking rewrite is still a race."""

    schema_version_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    schema_version_rows = (
        tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT version, applied_at FROM schema_version ORDER BY version"
            ).fetchall()
        )
        if schema_version_exists is not None
        else ()
    )
    durable_history_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migration_history'"
    ).fetchone()
    durable_history_rows = (
        tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT ordinal, version, migration_sha256, applied_at "
                "FROM schema_migration_history ORDER BY ordinal"
            ).fetchall()
        )
        if durable_history_exists is not None
        else None
    )
    return MigrationHistorySnapshot(
        schema_version_rows=schema_version_rows,
        durable_history_rows=durable_history_rows,
    )


def _migration_history_schema_contract() -> tuple[str, dict[str, str]]:
    migration = next(
        (
            resource
            for resource in _migration_resources(DatabaseKind.WORKFLOW)
            if int(resource.name.split("_", 1)[0]) == 3
        ),
        None,
    )
    if migration is None:
        raise ValueError("packaged workflow migration 003 is unavailable")
    table_statement: str | None = None
    trigger_statements: dict[str, str] = {}
    for statement in _migration_statements(migration.read_text(encoding="utf-8")):
        table_match = re.match(
            r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(schema_migration_history)\b",
            statement,
            flags=re.IGNORECASE,
        )
        if table_match:
            table_statement = statement
            continue
        trigger_match = re.match(
            r"CREATE\s+TRIGGER(?:\s+IF\s+NOT\s+EXISTS)?\s+"
            r"(trg_schema_migration_history_[A-Za-z0-9_]+)\b",
            statement,
            flags=re.IGNORECASE,
        )
        if trigger_match:
            trigger_statements[trigger_match.group(1)] = statement
    expected_triggers = {
        "trg_schema_migration_history_monotonic_insert",
        "trg_schema_migration_history_immutable_update",
        "trg_schema_migration_history_immutable_delete",
    }
    if table_statement is None or set(trigger_statements) != expected_triggers:
        raise ValueError("packaged durable migration history schema is incomplete")
    return table_statement, trigger_statements


def _is_canonical_utc(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() == UTC.utcoffset(parsed)
        and parsed.isoformat() == value
    )


def _verify_durable_migration_history(
    connection: sqlite3.Connection,
    *,
    versions: tuple[int, ...],
    packaged_hashes: dict[int, str],
) -> None:
    table_statement, trigger_statements = _migration_history_schema_contract()
    actual_table = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'schema_migration_history'"
    ).fetchone()
    if (
        actual_table is None
        or not isinstance(actual_table["sql"], str)
        or _normalized_sql(actual_table["sql"]) != _normalized_sql(table_statement)
    ):
        raise ValueError("durable migration history table schema is invalid")
    actual_triggers = {
        str(row["name"]): str(row["sql"])
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
            "AND tbl_name = 'schema_migration_history'"
        ).fetchall()
        if row["sql"] is not None
    }
    if set(actual_triggers) != set(trigger_statements) or any(
        _normalized_sql(actual_triggers[name]) != _normalized_sql(statement)
        for name, statement in trigger_statements.items()
    ):
        raise ValueError("durable migration history triggers are invalid")
    rows = connection.execute(
        "SELECT ordinal, version, migration_sha256, applied_at "
        "FROM schema_migration_history ORDER BY ordinal"
    ).fetchall()
    if len(rows) != len(versions):
        raise ValueError("durable migration history row count is invalid")
    for expected_ordinal, (version, row) in enumerate(zip(versions, rows, strict=True), start=1):
        if (
            type(row["ordinal"]) is not int
            or row["ordinal"] != expected_ordinal
            or type(row["version"]) is not int
            or row["version"] != version
            or not isinstance(row["migration_sha256"], str)
            or row["migration_sha256"] != packaged_hashes.get(version)
            or not _is_canonical_utc(row["applied_at"])
        ):
            raise ValueError("durable migration history row is invalid")


def _install_durable_migration_history_schema(connection: sqlite3.Connection) -> None:
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migration_history'"
    ).fetchone()
    if table_exists is not None:
        return
    table_statement, trigger_statements = _migration_history_schema_contract()
    connection.execute(table_statement)
    for statement in trigger_statements.values():
        connection.execute(statement)


def _backfill_durable_migration_history(
    connection: sqlite3.Connection,
    *,
    versions: tuple[int, ...],
    packaged_hashes: dict[int, str],
    applied_at: str,
) -> None:
    if connection.execute("SELECT COUNT(*) FROM schema_migration_history").fetchone()[0]:
        raise ValueError("durable migration history is not empty before backfill")
    connection.executemany(
        "INSERT INTO schema_migration_history("
        "ordinal, version, migration_sha256, applied_at) VALUES (?, ?, ?, ?)",
        (
            (ordinal, version, packaged_hashes[version], applied_at)
            for ordinal, version in enumerate(versions, start=1)
        ),
    )


def _inspect_workflow_version_neutral_contract(
    connection: sqlite3.Connection,
    *,
    history: tuple[int, ...],
) -> WorkflowVersionNeutralState:
    """Validate the installed migration prefix before any version-neutral write."""

    if not history or history[-1] < 3:
        return WorkflowVersionNeutralState(
            applies=False,
            durable_history_exists=False,
            archive_install_required=False,
        )
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"SQLite integrity_check returned {integrity}",
            stop_reason="DATABASE_INTEGRITY_FAILED",
        )
    durable_history_exists = bool(
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migration_history'"
        ).fetchone()
    )
    archive_install_required = _verify_schema_manifest(
        connection,
        _expected_workflow_schema_manifest(
            history,
            include_durable_history=durable_history_exists,
        ),
        allow_partial_empty_archive=True,
    )
    return WorkflowVersionNeutralState(
        applies=True,
        durable_history_exists=durable_history_exists,
        archive_install_required=archive_install_required,
    )


def _retrofit_inventory_path(
    workflow_path: Path,
    settings: Settings | None,
) -> Path:
    if settings is None:
        raise DatabaseVerificationError(
            ErrorCategory.CONFIGURATION,
            "legacy workflow v3 retrofit requires explicit Settings authorization context",
            stop_reason="DATABASE_AUTHORIZATION_CONTEXT_REQUIRED",
        )
    if settings.workflow_db.resolve() != workflow_path:
        raise DatabaseVerificationError(
            ErrorCategory.CONFIGURATION,
            "legacy workflow v3 retrofit Settings do not identify the database being migrated",
            stop_reason="DATABASE_AUTHORIZATION_CONTEXT_MISMATCH",
        )
    inventory_path = settings.inventory_db.resolve()
    _assert_sqlite_file(inventory_path)
    _assert_rollback_journal_header(
        inventory_path,
        stop_reason="AUTHORIZATION_INVENTORY_WAL_MODE_UNSUPPORTED",
        database_label="authorization inventory",
    )
    return inventory_path


def _attach_retrofit_inventory(
    connection: sqlite3.Connection,
    inventory_path: Path,
    *,
    read_only: bool,
) -> None:
    target = f"{inventory_path.as_uri()}?mode=ro" if read_only else str(inventory_path)
    connection.execute("ATTACH DATABASE ? AS authorization_inventory", (target,))


def _audit_workflow_retrofit_authorization(
    connection: sqlite3.Connection,
    settings: Settings,
    *,
    archive_install_required: bool,
) -> None:
    _verify_attached_inventory_context(connection)
    from invoice_agents.db.authorization_audit import audit_active_workflow_authorization

    invalid = audit_active_workflow_authorization(
        connection,
        settings,
        inventory_schema="authorization_inventory",
    )
    if not archive_install_required:
        from invoice_agents.db.legacy_archive import audit_legacy_authorization_archives

        invalid["invalid_quarantine_count"] = audit_legacy_authorization_archives(connection)
    if any(invalid.values()):
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            "legacy workflow v3 authorization provenance is incomplete or inconsistent",
            stop_reason="DATABASE_AUTHORIZATION_PROVENANCE_INVALID",
            details=invalid,
        )


def _preflight_existing_migration_history(
    path: Path,
    *,
    kind: DatabaseKind,
    packaged_versions: tuple[int, ...],
    packaged_hashes: dict[int, str],
    settings: Settings | None = None,
) -> MigrationPreflight | None:
    """Validate a present SQLite file without opening any write transaction."""

    if not path.exists() or path.stat().st_size == 0:
        return None
    _assert_sqlite_file(path)
    if kind is DatabaseKind.WORKFLOW:
        _assert_rollback_journal_header(
            path,
            stop_reason="WORKFLOW_WAL_MODE_UNSUPPORTED",
            database_label="workflow",
        )
    with connect_database(path, read_only=True) as connection:
        history = _read_migration_history(
            connection,
            kind=kind,
            packaged_versions=packaged_versions,
            packaged_hashes=packaged_hashes,
            allow_durable_retrofit=True,
        )
        history_snapshot = _migration_history_snapshot(connection)
        version_neutral_state = (
            _inspect_workflow_version_neutral_contract(connection, history=history)
            if kind is DatabaseKind.WORKFLOW
            else WorkflowVersionNeutralState(
                applies=False,
                durable_history_exists=False,
                archive_install_required=False,
            )
        )
        if kind is DatabaseKind.WORKFLOW and version_neutral_state.install_required:
            inventory_path = _retrofit_inventory_path(path, settings)
            _attach_retrofit_inventory(connection, inventory_path, read_only=True)
            connection.execute("BEGIN")
            connection.execute(
                "SELECT (SELECT COUNT(*) FROM main.sqlite_schema), "
                "(SELECT COUNT(*) FROM authorization_inventory.sqlite_schema)"
            ).fetchone()
            locked_history = _read_migration_history(
                connection,
                kind=kind,
                packaged_versions=packaged_versions,
                packaged_hashes=packaged_hashes,
                allow_durable_retrofit=True,
            )
            locked_history_snapshot = _migration_history_snapshot(connection)
            locked_state = _inspect_workflow_version_neutral_contract(
                connection,
                history=locked_history,
            )
            if (
                history != locked_history
                or history_snapshot != locked_history_snapshot
                or not locked_state.install_required
            ):
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "legacy workflow v3 changed during read-only retrofit preflight",
                    stop_reason="DATABASE_SCHEMA_MISMATCH",
                )
            assert settings is not None
            _audit_workflow_retrofit_authorization(
                connection,
                settings,
                archive_install_required=locked_state.archive_install_required,
            )
            history = locked_history
            history_snapshot = locked_history_snapshot
            version_neutral_state = locked_state
        return MigrationPreflight(
            history=history,
            history_snapshot=history_snapshot,
            version_neutral_install_required=version_neutral_state.install_required,
        )


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
    expected_columns = {
        "legacy_authorization_reconciliations": (
            "reconciliation_id",
            "reviewer",
            "reason",
            "disposition",
            "confirmed_at",
            "source_schema_version",
            "record_count",
            "schema_manifest_hash",
            "record_manifest_hash",
            "table_counts_json",
            "state",
        ),
        "legacy_authorization_database_snapshots": (
            "snapshot_id",
            "reconciliation_id",
            "database_image",
            "sha256",
            "size_bytes",
            "captured_at",
            "source_schema_version",
            "page_size",
            "page_count",
            "active_table_counts_json",
        ),
        "legacy_authorization_table_manifests": (
            "manifest_id",
            "reconciliation_id",
            "source_table_order",
            "source_table",
            "source_table_sql",
            "column_manifest",
            "schema_hash",
            "original_row_count",
        ),
        "legacy_authorization_quarantine": (
            "archive_id",
            "reconciliation_id",
            "source_table",
            "source_record_key",
            "source_row_ordinal",
            "source_rowid",
            "original_row_json",
            "typed_row",
            "schema_hash",
            "record_hash",
            "authorization_state",
            "archived_at",
        ),
    }
    installed_columns = {
        table: tuple(
            str(row["name"])
            for row in connection.execute(
                f"PRAGMA table_info({_quote_identifier(table)})"
            ).fetchall()
        )
        for table in expected_columns
    }
    incompatible = {
        table
        for table, columns in installed_columns.items()
        if columns and columns != expected_columns[table]
    }
    missing = {table for table, columns in installed_columns.items() if not columns}
    if incompatible or missing:
        archived_rows = sum(
            int(
                connection.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table)}").fetchone()[0]
            )
            for table, columns in installed_columns.items()
            if columns
        )
        if archived_rows:
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                "an existing legacy authorization archive cannot be upgraded losslessly",
                stop_reason="LEGACY_RECONCILIATION_ARCHIVE_UPGRADE_REQUIRED",
                details={
                    "incompatible_archive_tables": sorted(incompatible),
                    "missing_archive_tables": sorted(missing),
                },
            )
        for table in (
            "legacy_authorization_quarantine",
            "legacy_authorization_table_manifests",
            "legacy_authorization_database_snapshots",
            "legacy_authorization_reconciliations",
        ):
            connection.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table)}")
    for statement in _legacy_archive_statements():
        connection.execute(statement)


def _normalized_sql(sql: str) -> str:
    """Canonicalize SQLite syntax without altering protected token bytes.

    SQLite keywords and unquoted identifiers are case-insensitive, comments and
    token spacing are insignificant, and ``IF NOT EXISTS`` is only an execution
    guard.  String/blob literals and every quoted-identifier spelling are data,
    though, so their exact code points and case remain part of the schema contract.
    """

    tokens: list[tuple[str, str]] = []
    index = 0
    length = len(sql)

    def quoted_token(start: int, delimiter: str) -> int:
        cursor = start + 1
        if delimiter == "[":
            while cursor < length:
                cursor += 1
                if sql[cursor - 1] == "]":
                    break
            return cursor
        while cursor < length:
            if sql[cursor] != delimiter:
                cursor += 1
                continue
            cursor += 1
            if cursor < length and sql[cursor] == delimiter:
                cursor += 1
                continue
            break
        return cursor

    while index < length:
        character = sql[index]
        if character in " \t\r\n\f":
            index += 1
            continue
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if (
            character in "xX"
            and index + 1 < length
            and sql[index + 1] == "'"
            and (index == 0 or not (sql[index - 1].isalnum() or sql[index - 1] in "_$"))
        ):
            end = quoted_token(index + 1, "'")
            tokens.append(("protected", sql[index:end]))
            index = end
            continue
        if character in "'\"`[":
            end = quoted_token(index, character)
            tokens.append(("protected", sql[index:end]))
            index = end
            continue
        if character.isdigit() or (
            character == "." and index + 1 < length and sql[index + 1].isdigit()
        ):
            end = index
            if sql.startswith(("0x", "0X"), index):
                end += 2
                while end < length and (sql[end].isdigit() or sql[end].lower() in "abcdef_"):
                    end += 1
            else:
                while end < length and (sql[end].isdigit() or sql[end] == "_"):
                    end += 1
                if end < length and sql[end] == ".":
                    end += 1
                    while end < length and (sql[end].isdigit() or sql[end] == "_"):
                        end += 1
                if end < length and sql[end] in "eE":
                    exponent = end + 1
                    if exponent < length and sql[exponent] in "+-":
                        exponent += 1
                    digits = exponent
                    while exponent < length and (sql[exponent].isdigit() or sql[exponent] == "_"):
                        exponent += 1
                    if exponent > digits:
                        end = exponent
            tokens.append(("unquoted", _ascii_lower(sql[index:end])))
            index = end
            continue
        if character.isalpha() or character in "_$":
            end = index + 1
            while end < length and (sql[end].isalnum() or sql[end] in "_$"):
                end += 1
            tokens.append(("unquoted", _ascii_lower(sql[index:end])))
            index = end
            continue
        operator = next(
            (
                candidate
                for candidate in ("->>", "||", "->", "<<", ">>", "<=", ">=", "==", "!=", "<>", ":=")
                if sql.startswith(candidate, index)
            ),
            None,
        )
        if operator is not None:
            tokens.append(("punctuation", operator))
            index += len(operator)
            continue
        tokens.append(("punctuation", character))
        index += 1

    while tokens and tokens[-1] == ("punctuation", ";"):
        tokens.pop()

    unquoted = [value if kind == "unquoted" else None for kind, value in tokens]
    try:
        create_at = unquoted.index("create")
    except ValueError:
        create_at = -1
    if create_at >= 0:
        object_at = create_at + 1
        while object_at < len(tokens) and unquoted[object_at] in {
            "temp",
            "temporary",
            "unique",
        }:
            object_at += 1
        if (
            object_at < len(tokens)
            and unquoted[object_at] in {"index", "table", "trigger"}
            and unquoted[object_at + 1 : object_at + 4] == ["if", "not", "exists"]
        ):
            del tokens[object_at + 1 : object_at + 4]

    return json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))


def _ascii_lower(value: str) -> str:
    """Match SQLite's built-in identifier folding without changing Unicode bytes."""

    return "".join(
        chr(ord(character) + 32) if "A" <= character <= "Z" else character for character in value
    )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _table_schema_manifest(
    connection: sqlite3.Connection,
    table: str,
) -> TableSchemaManifest:
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if table_row is None or table_row["sql"] is None:
        raise ValueError(f"required schema manifest table is missing: {table}")
    quoted_table = _quote_identifier(table)
    columns = tuple(
        SchemaColumnManifest(
            cid=int(row["cid"]),
            name=str(row["name"]),
            declared_type=str(row["type"]),
            not_null=int(row["notnull"]),
            default_sql=str(row["dflt_value"]) if row["dflt_value"] is not None else None,
            primary_key_position=int(row["pk"]),
        )
        for row in connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
    )
    foreign_keys = tuple(
        SchemaForeignKeyManifest(
            identifier=int(row["id"]),
            sequence=int(row["seq"]),
            referenced_table=str(row["table"]),
            from_column=str(row["from"]),
            to_column=str(row["to"]) if row["to"] is not None else None,
            on_update=str(row["on_update"]),
            on_delete=str(row["on_delete"]),
            match=str(row["match"]),
        )
        for row in connection.execute(f"PRAGMA foreign_key_list({quoted_table})").fetchall()
    )
    indexes: list[SchemaIndexManifest] = []
    for row in connection.execute(f"PRAGMA index_list({quoted_table})").fetchall():
        index_name = str(row["name"])
        quoted_index = _quote_identifier(index_name)
        index_sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
        index_sql = (
            _normalized_sql(str(index_sql_row["sql"]))
            if index_sql_row is not None and index_sql_row["sql"] is not None
            else None
        )
        index_columns = tuple(
            SchemaIndexColumnManifest(
                sequence=int(column["seqno"]),
                cid=int(column["cid"]),
                name=str(column["name"]) if column["name"] is not None else None,
                descending=int(column["desc"]),
                collation=str(column["coll"]),
                key_column=int(column["key"]),
            )
            for column in connection.execute(f"PRAGMA index_xinfo({quoted_index})").fetchall()
        )
        indexes.append(
            SchemaIndexManifest(
                sequence=int(row["seq"]),
                name=index_name,
                unique=int(row["unique"]),
                origin=str(row["origin"]),
                partial=int(row["partial"]),
                columns=index_columns,
                normalized_sql=index_sql,
            )
        )
    triggers = tuple(
        SchemaTriggerManifest(
            name=str(row["name"]),
            normalized_sql=_normalized_sql(str(row["sql"])),
        )
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND tbl_name = ? "
            "ORDER BY name",
            (table,),
        ).fetchall()
        if row["sql"] is not None
    )
    return TableSchemaManifest(
        name=table,
        normalized_sql=_normalized_sql(str(table_row["sql"])),
        columns=columns,
        foreign_keys=foreign_keys,
        indexes=tuple(indexes),
        triggers=triggers,
    )


def _schema_object_manifest(
    connection: sqlite3.Connection,
) -> tuple[SchemaObjectManifest, ...]:
    """Inventory every persistent SQLite schema object, including autoindexes."""

    return tuple(
        SchemaObjectManifest(
            object_type=str(row["type"]),
            name=str(row["name"]),
            table_name=str(row["tbl_name"]),
            normalized_sql=(_normalized_sql(str(row["sql"])) if row["sql"] is not None else None),
        )
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'index', 'trigger', 'view') "
            "ORDER BY type, name, tbl_name"
        ).fetchall()
    )


@cache
def _expected_workflow_schema_manifest(
    versions: tuple[int, ...] | None = None,
    *,
    include_durable_history: bool = True,
) -> WorkflowSchemaManifest:
    """Build the complete exact contract from the selected packaged migrations."""

    reference = sqlite3.connect(":memory:")
    reference.row_factory = sqlite3.Row
    try:
        reference.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        resources = _migration_resources(DatabaseKind.WORKFLOW)
        selected_versions = (
            versions
            if versions is not None
            else _migration_versions(resources, DatabaseKind.WORKFLOW)
        )
        for resource in resources:
            version = int(resource.name.split("_", 1)[0])
            if version not in selected_versions:
                continue
            reference.executescript(resource.read_text(encoding="utf-8"))
        if not include_durable_history:
            reference.execute("DROP TABLE schema_migration_history")
        reference.executescript(
            files("invoice_agents.db")
            .joinpath("migrations", "workflow", "legacy_authorization_archive.sql")
            .read_text(encoding="utf-8")
        )
        table_names = tuple(
            str(row["name"])
            for row in reference.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        )
        return WorkflowSchemaManifest(
            tables=tuple(_table_schema_manifest(reference, table) for table in table_names),
            objects=_schema_object_manifest(reference),
        )
    finally:
        reference.close()


def _is_legacy_archive_table(name: str) -> bool:
    return name.startswith("legacy_authorization_")


def _is_legacy_archive_object(item: SchemaObjectManifest) -> bool:
    return _is_legacy_archive_table(item.name) or _is_legacy_archive_table(item.table_name)


def _verify_schema_manifest(
    connection: sqlite3.Connection,
    expected: WorkflowSchemaManifest,
    *,
    allow_partial_empty_archive: bool,
) -> bool:
    """Verify one exact schema; return whether optional archive objects are absent."""

    expected_archive_tables = {
        table.name for table in expected.tables if _is_legacy_archive_table(table.name)
    }
    actual_objects = _schema_object_manifest(connection)
    actual_archive_tables = {
        item.name
        for item in actual_objects
        if item.object_type == "table" and item.name in expected_archive_tables
    }
    missing_archive_tables = expected_archive_tables - actual_archive_tables
    compared_tables = tuple(
        table
        for table in expected.tables
        if not _is_legacy_archive_table(table.name) or table.name in actual_archive_tables
    )
    invalid: list[str] = []
    for table_manifest in compared_tables:
        try:
            actual = _table_schema_manifest(connection, table_manifest.name)
        except ValueError:
            invalid.append(table_manifest.name)
            continue
        if actual != table_manifest:
            invalid.append(table_manifest.name)
    if invalid:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            "workflow schema definitions differ from the complete packaged manifest",
            stop_reason="DATABASE_SCHEMA_MISMATCH",
            details={"invalid_schema_definitions": invalid},
        )
    expected_by_identity = {
        (item.object_type, item.name, item.table_name): item
        for item in expected.objects
        if not _is_legacy_archive_object(item) or item.table_name in actual_archive_tables
    }
    actual_by_identity = {
        (item.object_type, item.name, item.table_name): item for item in actual_objects
    }
    missing_objects = sorted(set(expected_by_identity) - set(actual_by_identity))
    unexpected_objects = sorted(set(actual_by_identity) - set(expected_by_identity))
    changed_objects = sorted(
        identity
        for identity in set(expected_by_identity) & set(actual_by_identity)
        if expected_by_identity[identity] != actual_by_identity[identity]
    )
    if missing_archive_tables and not allow_partial_empty_archive:
        missing_objects.extend(("table", table, table) for table in sorted(missing_archive_tables))
    if missing_objects or unexpected_objects or changed_objects:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            "workflow schema definitions differ from the complete packaged manifest",
            stop_reason="DATABASE_SCHEMA_MISMATCH",
            details={
                "invalid_schema_definitions": [],
                "missing_schema_objects": missing_objects,
                "unexpected_schema_objects": unexpected_objects,
                "changed_schema_objects": changed_objects,
            },
        )
    if missing_archive_tables:
        archived_rows = sum(
            int(
                connection.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table)}").fetchone()[0]
            )
            for table in actual_archive_tables
        )
        if archived_rows:
            raise DatabaseVerificationError(
                ErrorCategory.DATABASE,
                "an existing legacy authorization archive cannot be upgraded losslessly",
                stop_reason="LEGACY_RECONCILIATION_ARCHIVE_UPGRADE_REQUIRED",
                details={"missing_archive_tables": sorted(missing_archive_tables)},
            )
    return bool(missing_archive_tables)


def _verify_workflow_schema_manifest(connection: sqlite3.Connection) -> None:
    _verify_schema_manifest(
        connection,
        _expected_workflow_schema_manifest(),
        allow_partial_empty_archive=False,
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
        capture_legacy_authorization_tables,
        export_legacy_database_snapshot,
        legacy_json_safe_row,
        legacy_manifest_hash,
        legacy_reconciliation_id,
        legacy_record_hash,
        legacy_schema_manifest_hash,
        legacy_source_record_key,
        verify_serialized_database_snapshot,
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
    resources = _migration_resources(DatabaseKind.WORKFLOW)
    packaged_versions = _migration_versions(resources, DatabaseKind.WORKFLOW)
    packaged_hashes = _migration_hashes(resources)
    preflight = _preflight_existing_migration_history(
        resolved,
        kind=DatabaseKind.WORKFLOW,
        packaged_versions=packaged_versions,
        packaged_hashes=packaged_hashes,
    )
    assert preflight is not None
    with connect_database(resolved) as connection:
        try:
            current_history = _read_migration_history(
                connection,
                kind=DatabaseKind.WORKFLOW,
                packaged_versions=packaged_versions,
                packaged_hashes=packaged_hashes,
            )
            if current_history != preflight.history:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "workflow migration history changed during reconciliation preflight",
                    stop_reason="MIGRATION_HISTORY_INVALID",
                )
            connection.execute("BEGIN IMMEDIATE")
            version = current_history[-1] if current_history else 0
            archive_schema_exists = bool(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'legacy_authorization_reconciliations'"
                ).fetchone()
            )
            captured_tables = capture_legacy_authorization_tables(connection)
            active_count = sum(len(table.rows) for table in captured_tables)
            if archive_schema_exists:
                _install_legacy_archive_schema(connection)
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

            archived_at = utc_now()
            table_counts = {
                table: int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {_quote_identifier(table)}"
                    ).fetchone()[0]
                )
                for table in LEGACY_ACTIVE_TABLE_KEYS
            }
            snapshot_image = connection.serialize()
            snapshot_facts = verify_serialized_database_snapshot(
                snapshot_image,
                expected_active_table_counts=table_counts,
                expected_schema_version=version,
            )
            captured_counts = {table.source_table: len(table.rows) for table in captured_tables}
            if captured_counts != table_counts:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "legacy active facts changed during exact snapshot capture",
                    stop_reason="LEGACY_RECONCILIATION_STATE_INVALID",
                )
            _install_legacy_archive_schema(connection)
            archived_records: list[
                tuple[str, int, int | None, str, str | None, bytes, str, str]
            ] = []
            for table in captured_tables:
                for row in table.rows:
                    record_hash = legacy_record_hash(
                        table.source_table,
                        table.schema_hash,
                        row.typed_row,
                    )
                    archived_records.append(
                        (
                            table.source_table,
                            row.source_row_ordinal,
                            row.source_rowid,
                            legacy_source_record_key(table, row),
                            legacy_json_safe_row(table, row),
                            row.typed_row,
                            table.schema_hash,
                            record_hash,
                        )
                    )
            schema_manifest_hash = legacy_schema_manifest_hash(captured_tables)
            manifest_hash = legacy_manifest_hash(
                schema_manifest_hash,
                [
                    (table, ordinal, record_hash)
                    for table, ordinal, _rowid, _key, _json, _typed, _schema, record_hash in archived_records
                ],
            )
            reconciliation_id = legacy_reconciliation_id(
                reviewer=selected_reviewer,
                reason=selected_reason,
                disposition=disposition,
                source_schema_version=version,
                manifest_hash=manifest_hash,
                database_snapshot_hash=snapshot_facts.sha256,
            )
            for manifest_table in captured_tables:
                connection.execute(
                    "INSERT INTO legacy_authorization_table_manifests("
                    "manifest_id, reconciliation_id, source_table_order, source_table, "
                    "source_table_sql, column_manifest, schema_hash, original_row_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"ltm_{manifest_table.schema_hash}",
                        reconciliation_id,
                        manifest_table.source_table_order,
                        manifest_table.source_table,
                        sqlite3.Binary(manifest_table.source_table_sql.encode("utf-8")),
                        sqlite3.Binary(manifest_table.column_manifest),
                        manifest_table.schema_hash,
                        len(manifest_table.rows),
                    ),
                )
            for (
                source_table,
                ordinal,
                source_rowid,
                key,
                original_json,
                typed_row,
                schema_hash,
                record_hash,
            ) in archived_records:
                connection.execute(
                    "INSERT INTO legacy_authorization_quarantine("
                    "archive_id, reconciliation_id, source_table, source_record_key, "
                    "source_row_ordinal, source_rowid, original_row_json, typed_row, "
                    "schema_hash, record_hash, authorization_state, archived_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"lqar_{record_hash}",
                        reconciliation_id,
                        source_table,
                        key,
                        ordinal,
                        source_rowid,
                        original_json,
                        sqlite3.Binary(typed_row),
                        schema_hash,
                        record_hash,
                        disposition,
                        archived_at,
                    ),
                )
            connection.execute(
                "INSERT INTO legacy_authorization_reconciliations("
                "reconciliation_id, reviewer, reason, disposition, confirmed_at, "
                "source_schema_version, record_count, schema_manifest_hash, "
                "record_manifest_hash, table_counts_json, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED')",
                (
                    reconciliation_id,
                    selected_reviewer,
                    selected_reason,
                    disposition,
                    archived_at,
                    version,
                    len(archived_records),
                    schema_manifest_hash,
                    manifest_hash,
                    canonical_legacy_json(table_counts),
                ),
            )
            connection.execute(
                "INSERT INTO legacy_authorization_database_snapshots("
                "snapshot_id, reconciliation_id, database_image, sha256, size_bytes, "
                "captured_at, source_schema_version, page_size, page_count, "
                "active_table_counts_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"ldbs_{snapshot_facts.sha256}",
                    reconciliation_id,
                    sqlite3.Binary(snapshot_image),
                    snapshot_facts.sha256,
                    snapshot_facts.size_bytes,
                    archived_at,
                    snapshot_facts.source_schema_version,
                    snapshot_facts.page_size,
                    snapshot_facts.page_count,
                    canonical_legacy_json(snapshot_facts.active_table_counts),
                ),
            )
            readback_image = export_legacy_database_snapshot(connection, reconciliation_id)
            if readback_image != snapshot_image:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "legacy database snapshot readback differs from its source",
                    stop_reason="LEGACY_RECONCILIATION_ARCHIVE_INVALID",
                )
            if audit_legacy_authorization_archives(connection):
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "legacy rows were not archived with exact canonical hashes",
                    stop_reason="LEGACY_RECONCILIATION_ARCHIVE_INVALID",
                )
            for active_table in (
                "payments",
                "final_decisions",
                "human_decisions",
                "review_requests",
            ):
                connection.execute(f"DELETE FROM {active_table}")
            remaining = sum(
                int(connection.execute(f"SELECT COUNT(*) FROM {active_table}").fetchone()[0])
                for active_table in LEGACY_ACTIVE_TABLE_KEYS
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


def migrate_database(
    path: Path,
    kind: DatabaseKind | None = None,
    *,
    settings: Settings | None = None,
) -> list[int]:
    """Apply unapplied migrations; this is never called by normal processing."""

    selected_kind = kind or infer_kind(path)
    path = path.resolve()
    resources = _migration_resources(selected_kind)
    packaged_versions = _migration_versions(resources, selected_kind)
    packaged_hashes = _migration_hashes(resources)
    preflight = _preflight_existing_migration_history(
        path,
        kind=selected_kind,
        packaged_versions=packaged_versions,
        packaged_hashes=packaged_hashes,
        settings=settings,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    applied: list[int] = []
    with connect_database(path) as connection:
        if preflight is None:
            connection.execute(
                "CREATE TABLE schema_version ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            connection.commit()
            history: tuple[int, ...] = ()
        else:
            history = _read_migration_history(
                connection,
                kind=selected_kind,
                packaged_versions=packaged_versions,
                packaged_hashes=packaged_hashes,
                allow_durable_retrofit=True,
            )
            history_snapshot = _migration_history_snapshot(connection)
            if history != preflight.history or history_snapshot != preflight.history_snapshot:
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    f"{selected_kind.value} migration history changed during preflight",
                    stop_reason="MIGRATION_HISTORY_INVALID",
                )
        if (
            selected_kind is DatabaseKind.WORKFLOW
            and preflight is not None
            and preflight.version_neutral_install_required
        ):
            try:
                inventory_path = _retrofit_inventory_path(path, settings)
                _attach_retrofit_inventory(connection, inventory_path, read_only=False)
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "SELECT (SELECT COUNT(*) FROM main.sqlite_schema), "
                    "(SELECT COUNT(*) FROM authorization_inventory.sqlite_schema)"
                ).fetchone()
                locked_history = _read_migration_history(
                    connection,
                    kind=selected_kind,
                    packaged_versions=packaged_versions,
                    packaged_hashes=packaged_hashes,
                    allow_durable_retrofit=True,
                )
                locked_history_snapshot = _migration_history_snapshot(connection)
                if (
                    locked_history != preflight.history
                    or locked_history_snapshot != preflight.history_snapshot
                ):
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        "workflow migration history changed after the retrofit write lock",
                        stop_reason="MIGRATION_HISTORY_INVALID",
                    )
                version_neutral_state = _inspect_workflow_version_neutral_contract(
                    connection,
                    history=locked_history,
                )
                if not version_neutral_state.install_required:
                    raise DatabaseVerificationError(
                        ErrorCategory.DATABASE,
                        "workflow version-neutral schema changed during preflight",
                        stop_reason="DATABASE_SCHEMA_MISMATCH",
                    )
                assert settings is not None
                _audit_workflow_retrofit_authorization(
                    connection,
                    settings,
                    archive_install_required=version_neutral_state.archive_install_required,
                )
                _install_legacy_archive_schema(connection)
                durable_history_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'schema_migration_history'"
                ).fetchone()
                if durable_history_exists is None:
                    _install_durable_migration_history_schema(connection)
                    _backfill_durable_migration_history(
                        connection,
                        versions=locked_history,
                        packaged_hashes=packaged_hashes,
                        applied_at=utc_now(),
                    )
                _verify_schema_manifest(
                    connection,
                    _expected_workflow_schema_manifest(
                        locked_history,
                        include_durable_history=True,
                    ),
                    allow_partial_empty_archive=False,
                )
                _read_migration_history(
                    connection,
                    kind=selected_kind,
                    packaged_versions=packaged_versions,
                    packaged_hashes=packaged_hashes,
                )
                connection.commit()
                history = locked_history
            except DatabaseVerificationError:
                connection.rollback()
                raise
            except (sqlite3.Error, ValueError) as exc:
                connection.rollback()
                raise DatabaseVerificationError(
                    ErrorCategory.DATABASE,
                    "version-neutral workflow schema installation failed",
                    stop_reason="MIGRATION_FAILED",
                ) from exc
        existing = set(history)
        for resource in resources:
            version = int(resource.name.split("_", 1)[0])
            if version in existing:
                continue
            script = resource.read_text(encoding="utf-8")
            statements = _migration_statements(script)
            normalized_statements = {_normalized_sql(statement) for statement in statements}
            disable_foreign_keys = (
                _normalized_sql("PRAGMA foreign_keys=OFF") in normalized_statements
            )
            transaction_controls = {
                _normalized_sql(statement)
                for statement in (
                    "BEGIN",
                    "BEGIN TRANSACTION",
                    "BEGIN IMMEDIATE",
                    "COMMIT",
                    "END TRANSACTION",
                    "PRAGMA foreign_keys=OFF",
                    "PRAGMA foreign_keys=ON",
                )
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
                applied_at = utc_now()
                connection.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (version, applied_at),
                )
                if selected_kind is DatabaseKind.WORKFLOW and version >= 3:
                    if version == 3:
                        _backfill_durable_migration_history(
                            connection,
                            versions=tuple(range(1, version + 1)),
                            packaged_hashes=packaged_hashes,
                            applied_at=applied_at,
                        )
                    else:
                        connection.execute(
                            "INSERT INTO schema_migration_history("
                            "ordinal, version, migration_sha256, applied_at) "
                            "VALUES (?, ?, ?, ?)",
                            (version, version, packaged_hashes[version], applied_at),
                        )
                    _read_migration_history(
                        connection,
                        kind=selected_kind,
                        packaged_versions=packaged_versions,
                        packaged_hashes=packaged_hashes,
                    )
                connection.commit()
            except DatabaseVerificationError:
                connection.rollback()
                raise
            except (sqlite3.Error, ValueError) as exc:
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
        DatabaseKind.WORKFLOW.value: migrate_database(
            workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        ),
    }
    seed_inventory(inventory_db)
    verify_database(inventory_db, DatabaseKind.INVENTORY)
    verify_database(workflow_db, DatabaseKind.WORKFLOW, settings=settings)
    return applied


def _assert_sqlite_file(path: Path) -> bytes:
    if not path.is_file():
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"required database does not exist: {path}",
            stop_reason="DATABASE_MISSING",
        )
    with path.open("rb") as handle:
        header = handle.read(20)
    if len(header) < 20 or header[: len(SQLITE_SIGNATURE)] != SQLITE_SIGNATURE:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"file is not a SQLite database: {path}",
            stop_reason="DATABASE_SIGNATURE_INVALID",
        )
    return header


def _assert_rollback_journal_header(
    path: Path,
    *,
    stop_reason: str,
    database_label: str,
) -> None:
    """Reject WAL from header bytes without opening SQLite or touching its sidecars."""

    header = _assert_sqlite_file(path)
    write_version = header[18]
    read_version = header[19]
    if write_version == 2 or read_version == 2:
        raise DatabaseVerificationError(
            ErrorCategory.DATABASE,
            f"{database_label} database uses WAL file-format header bytes; "
            "rollback-journal mode is required",
            stop_reason=stop_reason,
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
    _assert_rollback_journal_header(
        resolved,
        stop_reason=(
            "WORKFLOW_WAL_MODE_UNSUPPORTED"
            if selected_kind is DatabaseKind.WORKFLOW
            else "INVENTORY_WAL_MODE_UNSUPPORTED"
        ),
        database_label=selected_kind.value,
    )
    migration_resources = _migration_resources(selected_kind)
    packaged_versions = _migration_versions(migration_resources, selected_kind)
    packaged_hashes = _migration_hashes(migration_resources)
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
        _assert_rollback_journal_header(
            inventory_resolved,
            stop_reason="AUTHORIZATION_INVENTORY_WAL_MODE_UNSUPPORTED",
            database_label="authorization inventory",
        )
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
            "schema_migration_history": {
                "ordinal",
                "version",
                "migration_sha256",
                "applied_at",
            },
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
                "schema_manifest_hash",
                "record_manifest_hash",
                "table_counts_json",
                "state",
            },
            "legacy_authorization_database_snapshots": {
                "snapshot_id",
                "reconciliation_id",
                "database_image",
                "sha256",
                "size_bytes",
                "captured_at",
                "source_schema_version",
                "page_size",
                "page_count",
                "active_table_counts_json",
            },
            "legacy_authorization_table_manifests": {
                "manifest_id",
                "reconciliation_id",
                "source_table_order",
                "source_table",
                "source_table_sql",
                "column_manifest",
                "schema_hash",
                "original_row_count",
            },
            "legacy_authorization_quarantine": {
                "archive_id",
                "reconciliation_id",
                "source_table",
                "source_record_key",
                "source_row_ordinal",
                "source_rowid",
                "original_row_json",
                "typed_row",
                "schema_hash",
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
            _read_migration_history(
                connection,
                kind=selected_kind,
                packaged_versions=packaged_versions,
                packaged_hashes=packaged_hashes,
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
                _verify_workflow_schema_manifest(connection)
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
