"""Durable, explicit quarantine for legacy authorization rows before workflow v3."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import invoice_agents.db.core as core_module
from invoice_agents.config import Settings
from invoice_agents.db import cli as database_cli
from invoice_agents.db.core import (
    REQUIRED_WORKFLOW_TRIGGERS,
    DatabaseKind,
    _migration_resources,
    connect_database,
    migrate_database,
    seed_inventory,
    verify_database,
)
from invoice_agents.errors import DatabaseVerificationError

DISPOSITION = "PERMANENTLY_NON_AUTHORIZING"
REVIEWER = "legacy-auditor@example.com"
REASON = "legacy rows have no generation-bound source and policy provenance"
ACTIVE_TABLE_KEYS = {
    "review_requests": "review_id",
    "human_decisions": "decision_id",
    "final_decisions": "decision_id",
    "payments": "payment_id",
}
RECORD_HASH_DOMAIN = b"galatiq.invoice-agents/legacy-authorization-record/v1\x00"


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _record_hash(table: str, key: str, row_json: str) -> str:
    material = RECORD_HASH_DOMAIN + table.encode() + b"\x00" + key.encode() + b"\x00"
    return hashlib.sha256(material + row_json.encode()).hexdigest()


def _build_full_v2_database(tmp_path: Path) -> Path:
    path = tmp_path / "workflow-v2-full-legacy.db"
    at = datetime(2026, 8, 8, 12, 34, 56, tzinfo=UTC).isoformat()
    resources = _migration_resources(DatabaseKind.WORKFLOW)
    with connect_database(path) as connection:
        connection.executescript(resources[0].read_text(encoding="utf-8"))
        connection.executescript(resources[1].read_text(encoding="utf-8"))
        connection.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            ((1, at), (2, at)),
        )
        connection.execute(
            "INSERT INTO source_artifacts(source_id, canonical_path, source_hash, source_format, "
            "size_bytes, modified_at, metadata_json, created_at) VALUES "
            "('legacy_source', '/legacy/exact-invoice.txt', ?, 'txt', 7, ?, ?, ?)",
            (
                "a" * 64,
                at,
                json.dumps({"legacy": "source metadata retained outside authorization archive"}),
                at,
            ),
        )
        connection.execute(
            "INSERT INTO cases(case_id, source_id, invoice_number, vendor, status, started_at, "
            "updated_at) VALUES "
            "('legacy_case', 'legacy_source', 'INV-LEGACY', 'Legacy Vendor', 'NEEDS_HUMAN', ?, ?)",
            (at, at),
        )
        review_payload = json.dumps(
            {"legacy": "exact review payload", "status": "RESOLVED", "opaque": [1, "two"]},
            separators=(",", ":"),
        )
        human_payload = json.dumps(
            {"legacy": "exact human payload", "decision": "APPROVE", "opaque": {"x": True}},
            separators=(",", ":"),
        )
        final_payload = json.dumps(
            {
                "decision": "REJECT",
                "reasons": ["contradictory paid legacy history must never authorize v3"],
                "evidence": [],
                "critic_disposition": "REJECT",
                "human_outcome": None,
                "payment_eligible": False,
            },
            separators=(",", ":"),
        )
        connection.execute(
            "INSERT INTO review_requests(review_id, case_id, sequence, status, payload_json, "
            "created_at, resolved_at) VALUES "
            "('legacy_review', 'legacy_case', 1, 'RESOLVED', ?, ?, ?)",
            (review_payload, at, at),
        )
        connection.execute(
            "INSERT INTO human_decisions(decision_id, review_id, reviewer, decision, reason, "
            "payload_json, decided_at) VALUES "
            "('legacy_human', 'legacy_review', 'old-reviewer@example.com', 'APPROVE', "
            "'legacy ruling', ?, ?)",
            (human_payload, at),
        )
        connection.execute(
            "INSERT INTO final_decisions(decision_id, case_id, payload_json, created_at) VALUES "
            "('legacy_final', 'legacy_case', ?, ?)",
            (final_payload, at),
        )
        connection.execute(
            "INSERT INTO payments(payment_id, case_id, idempotency_key, vendor, amount, currency, "
            "status, error, created_at) VALUES "
            "('legacy_payment', 'legacy_case', 'legacy-key', 'Legacy Vendor', '10.00', 'USD', "
            "'PAID', NULL, ?)",
            (at,),
        )
        connection.commit()
    return path


def _active_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    with connect_database(path, read_only=True) as connection:
        return {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
            for table in ACTIVE_TABLE_KEYS
        }


def _reconcile(path: Path, *, reason: str = REASON, confirmed: bool = True) -> Any:
    assert hasattr(core_module, "reconcile_legacy_authorization")
    return core_module.reconcile_legacy_authorization(
        path,
        reviewer=REVIEWER,
        reason=reason,
        disposition=DISPOSITION,
        confirmed=confirmed,
    )


def test_reconciliation_requires_explicit_confirmation_without_any_mutation(
    tmp_path: Path,
) -> None:
    path = _build_full_v2_database(tmp_path)
    original = _active_rows(path)
    with connect_database(path, read_only=True) as connection:
        schema_before = [
            tuple(row)
            for row in connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            )
        ]

    with pytest.raises(DatabaseVerificationError) as excinfo:
        _reconcile(path, confirmed=False)

    assert excinfo.value.stop_reason == "LEGACY_RECONCILIATION_CONFIRMATION_REQUIRED"
    assert _active_rows(path) == original
    with connect_database(path, read_only=True) as connection:
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            )
        ] == schema_before


def test_reconciliation_cli_requires_every_explicit_disposition_input(
    tmp_path: Path,
) -> None:
    path = _build_full_v2_database(tmp_path)
    before = _active_rows(path)
    arguments = [
        "reconcile-legacy-authorization",
        "--db",
        str(path),
        "--reviewer",
        REVIEWER,
        "--reason",
        REASON,
        "--disposition",
        DISPOSITION,
    ]

    unconfirmed = CliRunner().invoke(database_cli.app, arguments)

    assert unconfirmed.exit_code == 1
    assert isinstance(unconfirmed.exception, DatabaseVerificationError)
    assert unconfirmed.exception.stop_reason == "LEGACY_RECONCILIATION_CONFIRMATION_REQUIRED"
    assert _active_rows(path) == before

    confirmed = CliRunner().invoke(database_cli.app, [*arguments, "--confirm"])

    assert confirmed.exit_code == 0
    assert "reconciliation_id=lrec_" in confirmed.stdout
    assert "record_count=4" in confirmed.stdout
    assert all(not rows for rows in _active_rows(path).values())


def test_reconciliation_archives_exact_rows_and_hashes_before_removing_active_authority(
    tmp_path: Path,
) -> None:
    path = _build_full_v2_database(tmp_path)
    original = _active_rows(path)

    receipt = _reconcile(path)

    assert receipt.record_count == 4
    assert receipt.table_counts == {table: 1 for table in ACTIVE_TABLE_KEYS}
    with connect_database(path, read_only=True) as connection:
        assert {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ACTIVE_TABLE_KEYS
        } == {table: 0 for table in ACTIVE_TABLE_KEYS}
        archive_rows = connection.execute(
            "SELECT reconciliation_id, source_table, source_record_key, original_row_json, "
            "record_hash, authorization_state FROM legacy_authorization_quarantine "
            "ORDER BY source_table"
        ).fetchall()
        metadata = connection.execute(
            "SELECT reviewer, reason, disposition, source_schema_version, record_count, "
            "table_counts_json, state FROM legacy_authorization_reconciliations"
        ).fetchone()
    assert len(archive_rows) == 4
    for archive in archive_rows:
        table = str(archive["source_table"])
        original_row = original[table][0]
        key = str(original_row[ACTIVE_TABLE_KEYS[table]])
        expected_json = _canonical_row(original_row)
        assert archive["reconciliation_id"] == receipt.reconciliation_id
        assert archive["source_record_key"] == key
        assert archive["original_row_json"] == expected_json
        assert archive["record_hash"] == _record_hash(table, key, expected_json)
        assert archive["authorization_state"] == DISPOSITION
    assert tuple(metadata) == (
        REVIEWER,
        REASON,
        DISPOSITION,
        2,
        4,
        _canonical_row({table: 1 for table in ACTIVE_TABLE_KEYS}),
        "COMPLETED",
    )


def test_exact_reconciliation_retry_is_idempotent_and_changed_retry_is_rejected(
    tmp_path: Path,
) -> None:
    path = _build_full_v2_database(tmp_path)
    first = _reconcile(path)

    second = _reconcile(path)

    assert second == first
    with connect_database(path, read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM legacy_authorization_reconciliations"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM legacy_authorization_quarantine").fetchone()[0]
            == 4
        )
    with pytest.raises(DatabaseVerificationError) as excinfo:
        _reconcile(path, reason="a different retry must never reuse the archived disposition")
    assert excinfo.value.stop_reason == "LEGACY_RECONCILIATION_REPLAY_MISMATCH"


def test_reconciliation_rolls_back_archive_and_deletes_on_injected_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _build_full_v2_database(tmp_path)
    original = _active_rows(path)
    real_connect = core_module.connect_database

    class FailingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self.connection, name)

        def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
            if sql.startswith("DELETE FROM payments"):
                raise sqlite3.OperationalError("injected reconciliation crash")
            return self.connection.execute(sql, parameters)

    @contextmanager
    def failing_connect(target: Path, *, read_only: bool = False) -> Iterator[Any]:
        with real_connect(target, read_only=read_only) as connection:
            yield FailingConnection(connection)

    monkeypatch.setattr(core_module, "connect_database", failing_connect)
    with pytest.raises(DatabaseVerificationError) as excinfo:
        _reconcile(path)
    assert excinfo.value.stop_reason == "LEGACY_RECONCILIATION_FAILED"
    assert _active_rows(path) == original
    with real_connect(path, read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'legacy_authorization_%'"
            ).fetchone()[0]
            == 0
        )


def test_reconciled_database_upgrades_without_fabricating_generation_provenance(
    tmp_path: Path,
) -> None:
    path = _build_full_v2_database(tmp_path)
    original = _active_rows(path)
    receipt = _reconcile(path)
    inventory = tmp_path / "inventory.db"
    migrate_database(inventory, DatabaseKind.INVENTORY)
    seed_inventory(inventory)
    settings = Settings(
        workflow_db=path,
        inventory_db=inventory,
        source_archive_dir=tmp_path / "sources",
    )

    assert migrate_database(path, DatabaseKind.WORKFLOW) == [3]
    assert (
        verify_database(
            path,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )["schema_version"]
        == 3
    )
    assert _reconcile(path) == receipt
    with pytest.raises(DatabaseVerificationError) as replay_error:
        _reconcile(path, reason="changed after migration")
    assert replay_error.value.stop_reason == "LEGACY_RECONCILIATION_REPLAY_MISMATCH"

    with connect_database(path, read_only=True) as connection:
        archives = connection.execute(
            "SELECT source_table, original_row_json FROM legacy_authorization_quarantine "
            "WHERE reconciliation_id = ? ORDER BY source_table",
            (receipt.reconciliation_id,),
        ).fetchall()
        assert connection.execute("SELECT COUNT(*) FROM final_decisions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0
    assert len(archives) == 4
    for archive in archives:
        table = str(archive["source_table"])
        assert json.loads(archive["original_row_json"]) == original[table][0]
        assert "execution_generation" not in json.loads(archive["original_row_json"])
        assert "decision_generation" not in json.loads(archive["original_row_json"])


def test_legacy_archives_are_immutable_and_preflight_detects_tampering(
    tmp_path: Path,
) -> None:
    path = _build_full_v2_database(tmp_path)
    _reconcile(path)
    inventory = tmp_path / "inventory.db"
    migrate_database(inventory, DatabaseKind.INVENTORY)
    seed_inventory(inventory)
    settings = Settings(workflow_db=path, inventory_db=inventory)
    migrate_database(path, DatabaseKind.WORKFLOW)

    with connect_database(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="LEGACY_AUTHORIZATION_ARCHIVE_IMMUTABLE"):
            connection.execute(
                "UPDATE legacy_authorization_quarantine SET record_hash = ?",
                ("f" * 64,),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="LEGACY_AUTHORIZATION_ARCHIVE_IMMUTABLE"):
            connection.execute("DELETE FROM legacy_authorization_reconciliations")
        connection.rollback()

        trigger = "trg_legacy_authorization_quarantine_immutable_update"
        connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(
            "UPDATE legacy_authorization_quarantine SET record_hash = ? "
            "WHERE source_table = 'payments'",
            ("f" * 64,),
        )
        connection.execute(REQUIRED_WORKFLOW_TRIGGERS[trigger])
        connection.commit()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(path, DatabaseKind.WORKFLOW, settings=settings)
    assert excinfo.value.stop_reason == "DATABASE_AUTHORIZATION_PROVENANCE_INVALID"
    assert excinfo.value.details["invalid_quarantine_count"] == 1


def test_preflight_detects_reconciliation_metadata_tampering(tmp_path: Path) -> None:
    path = _build_full_v2_database(tmp_path)
    _reconcile(path)
    inventory = tmp_path / "inventory.db"
    migrate_database(inventory, DatabaseKind.INVENTORY)
    seed_inventory(inventory)
    settings = Settings(workflow_db=path, inventory_db=inventory)
    migrate_database(path, DatabaseKind.WORKFLOW)
    trigger = "trg_legacy_authorization_reconciliations_immutable_update"
    with connect_database(path) as connection:
        connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(
            "UPDATE legacy_authorization_reconciliations SET reason = 'forged metadata'"
        )
        connection.execute(REQUIRED_WORKFLOW_TRIGGERS[trigger])
        connection.commit()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(path, DatabaseKind.WORKFLOW, settings=settings)

    assert excinfo.value.stop_reason == "DATABASE_AUTHORIZATION_PROVENANCE_INVALID"
    assert excinfo.value.details["invalid_quarantine_count"] == 1


def test_explicit_migrate_retrofits_version_neutral_archive_schema_on_existing_v3(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing-workflow-v3.db"
    inventory = tmp_path / "inventory.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    migrate_database(inventory, DatabaseKind.INVENTORY)
    seed_inventory(inventory)
    settings = Settings(workflow_db=path, inventory_db=inventory)
    with connect_database(path) as connection:
        connection.execute("DROP TABLE legacy_authorization_quarantine")
        connection.execute("DROP TABLE legacy_authorization_reconciliations")
        connection.commit()

    assert migrate_database(path, DatabaseKind.WORKFLOW) == []
    assert verify_database(path, DatabaseKind.WORKFLOW, settings=settings)["schema_version"] == 3
