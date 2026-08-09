"""Durable, explicit quarantine for legacy authorization rows before workflow v3."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import invoice_agents.db.core as core_module
import invoice_agents.db.legacy_archive as legacy_archive
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


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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


def _build_opaque_v2_database(tmp_path: Path) -> Path:
    path = _build_full_v2_database(tmp_path)
    schemas = {
        "review_requests": (
            "CREATE TABLE review_requests ("
            "payload_json BLOB, review_id TEXT, case_id TEXT, sequence INTEGER, "
            "status TEXT, created_at REAL, resolved_at TEXT)"
        ),
        "human_decisions": (
            "CREATE TABLE human_decisions ("
            "decision_id TEXT, review_id TEXT, reviewer TEXT, decision TEXT, reason BLOB, "
            "payload_json TEXT, decided_at REAL)"
        ),
        "final_decisions": (
            "CREATE TABLE final_decisions ("
            "created_at INTEGER, payload_json BLOB, case_id TEXT, decision_id TEXT)"
        ),
        "payments": (
            "CREATE TABLE payments ("
            "payment_id TEXT, case_id TEXT, idempotency_key BLOB, vendor TEXT, amount REAL, "
            "currency TEXT, status TEXT, error BLOB, created_at INTEGER)"
        ),
    }
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in ("payments", "final_decisions", "human_decisions", "review_requests"):
            connection.execute(f"DROP TABLE {table}")
            connection.execute(schemas[table])
        connection.execute(
            "INSERT INTO review_requests(rowid, payload_json, review_id, case_id, sequence, "
            "status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                7,
                sqlite3.Binary(b"\x00\xffopaque-review"),
                "review_opaque",
                "case_opaque",
                1,
                "RESOLVED",
                0.1,
                None,
            ),
        )
        connection.execute(
            "INSERT INTO review_requests(rowid, payload_json, review_id, case_id, sequence, "
            "status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                3,
                sqlite3.Binary(b"earlier-rowid"),
                "review_earlier",
                "case_opaque",
                2,
                "PENDING",
                -0.0,
                "snowman \u2603\x00tail",
            ),
        )
        connection.execute(
            "INSERT INTO human_decisions(rowid, decision_id, review_id, reviewer, decision, "
            "reason, payload_json, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                11,
                "human_opaque",
                "review_opaque",
                "R\u00e9viewer \u2603",
                "APPROVE",
                sqlite3.Binary(b"\x00\xfeopaque-reason"),
                "text\x00with-nul",
                -0.0,
            ),
        )
        connection.execute(
            "INSERT INTO final_decisions(rowid, created_at, payload_json, case_id, decision_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                13,
                9223372036854775807,
                sqlite3.Binary(b"\x80\x00final"),
                "case_opaque",
                "final_opaque",
            ),
        )
        connection.execute(
            "INSERT INTO payments(rowid, payment_id, case_id, idempotency_key, vendor, amount, "
            "currency, status, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                17,
                "payment_opaque",
                "case_opaque",
                sqlite3.Binary(b"\x00idem\xff"),
                "Vendor \u2603",
                1.0 / 10.0,
                "USD",
                "FAILED",
                sqlite3.Binary(b"\x00\xfffailure"),
                -7,
            ),
        )
        connection.commit()
    return path


def _storage_value(value: object) -> tuple[str, object]:
    if value is None:
        return ("null", None)
    if isinstance(value, bytes):
        return ("blob", value)
    if isinstance(value, str):
        return ("text", value.encode("utf-8"))
    if isinstance(value, int):
        return ("integer", str(value))
    if isinstance(value, float):
        return ("real", struct.pack(">d", value))
    raise AssertionError(f"unexpected SQLite value: {value!r}")


def _exact_table_snapshot(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[str, tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    sql = str(
        connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()[0]
    )
    columns = tuple(tuple(row) for row in connection.execute(f'PRAGMA table_info("{table}")'))
    rows = tuple(
        (int(row[0]), *(_storage_value(value) for value in row[1:]))
        for row in connection.execute(f'SELECT rowid, * FROM "{table}" ORDER BY rowid')
    )
    return sql, columns, rows


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


def test_reconciliation_rejects_sparse_migration_history_before_archive_install(
    tmp_path: Path,
) -> None:
    path = _build_full_v2_database(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM schema_version WHERE version = 2")
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (3, ?)",
            (datetime(2026, 8, 8, 12, 35, tzinfo=UTC).isoformat(),),
        )
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        _reconcile(path)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert path.read_bytes() == before


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
    assert "error_code=LEGACY_RECONCILIATION_CONFIRMATION_REQUIRED" in unconfirmed.stderr
    assert "Traceback" not in unconfirmed.stderr
    assert _active_rows(path) == before

    confirmed = CliRunner().invoke(database_cli.app, [*arguments, "--confirm"])

    assert confirmed.exit_code == 0
    assert "reconciliation_id=lrec_" in confirmed.stdout
    assert "record_count=4" in confirmed.stdout
    assert all(not rows for rows in _active_rows(path).values())


def test_database_cli_subprocess_reports_only_stable_code_for_secret_bearing_driver_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow-secret-bearing-driver-error.db"
    secret = "sk-proj-task8-must-never-reach-operator-output"
    with sqlite3.connect(path) as connection:
        connection.create_function(secret, 0, lambda: 1, deterministic=True)
        connection.execute(
            "CREATE TABLE schema_version ("
            "marker INTEGER, "
            f'version INTEGER GENERATED ALWAYS AS ("{secret}"()) VIRTUAL, '
            "applied_at TEXT)"
        )
        connection.execute("INSERT INTO schema_version(marker, applied_at) VALUES (1, 'legacy')")
        connection.commit()

    completed = subprocess.run(
        [
            str(Path(__file__).parents[2] / ".venv" / "bin" / "invoice-agents"),
            "db",
            "migrate",
            "--db",
            str(path),
            "--kind",
            "workflow",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 1
    assert "error_code=MIGRATION_HISTORY_INVALID" in output
    assert secret not in output
    assert "Traceback" not in output
    assert "sqlite3" not in output
    assert "DatabaseVerificationError" not in output


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
            "typed_row, schema_hash, record_hash, authorization_state "
            "FROM legacy_authorization_quarantine "
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
        assert archive["record_hash"] == legacy_archive.legacy_record_hash(
            table,
            str(archive["schema_hash"]),
            bytes(archive["typed_row"]),
        )
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


def test_reconciliation_losslessly_archives_and_restores_ordered_typed_sqlite_rows(
    tmp_path: Path,
) -> None:
    path = _build_opaque_v2_database(tmp_path)
    with sqlite3.connect(path) as connection:
        original = {table: _exact_table_snapshot(connection, table) for table in ACTIVE_TABLE_KEYS}

    receipt = _reconcile(path)

    assert receipt.record_count == 5
    assert receipt.table_counts == {
        "review_requests": 2,
        "human_decisions": 1,
        "final_decisions": 1,
        "payments": 1,
    }
    assert hasattr(legacy_archive, "decode_legacy_authorization_archive")
    assert hasattr(legacy_archive, "restore_legacy_authorization_archive")
    restored_path = tmp_path / "restored-opaque.db"
    with (
        connect_database(path, read_only=True) as archive_connection,
        sqlite3.connect(restored_path) as restored_connection,
    ):
        decoded = legacy_archive.decode_legacy_authorization_archive(
            archive_connection, receipt.reconciliation_id
        )
        storage_classes = {
            cell.storage_class for table in decoded for row in table.rows for cell in row.cells
        }
        assert storage_classes == {"NULL", "INTEGER", "REAL", "TEXT", "BLOB"}
        review = next(table for table in decoded if table.source_table == "review_requests")
        assert [row.source_rowid for row in review.rows] == [3, 7]
        legacy_archive.restore_legacy_authorization_archive(
            archive_connection,
            restored_connection,
            receipt.reconciliation_id,
        )
        restored_connection.commit()
        restored = {
            table: _exact_table_snapshot(restored_connection, table) for table in ACTIVE_TABLE_KEYS
        }
    assert restored == original


def test_serialized_snapshot_restores_the_entire_exact_pre_reconciliation_database(
    tmp_path: Path,
) -> None:
    path = _build_opaque_v2_database(tmp_path)
    invalid_text = b"\x80\xfflegacy-text\x00tail"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE review_requests SET status = CAST(? AS TEXT) WHERE review_id = ?",
            (sqlite3.Binary(invalid_text), "review_opaque"),
        )
        connection.execute(
            "CREATE INDEX idx_legacy_review_status_exact ON review_requests(status, review_id)"
        )
        connection.execute(
            "CREATE TRIGGER trg_legacy_review_insert_exact BEFORE INSERT ON review_requests "
            "BEGIN SELECT RAISE(ABORT, 'LEGACY_INSERT_BLOCKED'); END"
        )
        connection.execute(
            "CREATE TABLE forensic_shadowed_rowid(rowid TEXT, _rowid_ TEXT, oid TEXT, payload BLOB)"
        )
        connection.execute(
            "INSERT INTO forensic_shadowed_rowid(rowid, _rowid_, oid, payload) "
            "VALUES ('visible-rowid', 'visible-alt', 'visible-oid', ?)",
            (sqlite3.Binary(b"\x00shadowed\xff"),),
        )
        connection.execute(
            "CREATE TABLE forensic_without_rowid("
            "identity BLOB PRIMARY KEY, payload TEXT) WITHOUT ROWID"
        )
        connection.execute(
            "INSERT INTO forensic_without_rowid(identity, payload) VALUES (?, ?)",
            (sqlite3.Binary(b"\xffidentity"), "forensic payload"),
        )
        connection.commit()
        expected_image = connection.serialize()
    expected_hash = hashlib.sha256(expected_image).hexdigest()

    receipt = _reconcile(path)

    assert hasattr(legacy_archive, "export_legacy_database_snapshot")
    assert hasattr(legacy_archive, "restore_legacy_database_snapshot")
    with connect_database(path, read_only=True) as archive_connection:
        snapshot = archive_connection.execute(
            "SELECT database_image, sha256, size_bytes FROM "
            "legacy_authorization_database_snapshots WHERE reconciliation_id = ?",
            (receipt.reconciliation_id,),
        ).fetchone()
        assert bytes(snapshot["database_image"]) == expected_image
        assert snapshot["sha256"] == expected_hash
        assert snapshot["size_bytes"] == len(expected_image)
        assert (
            legacy_archive.export_legacy_database_snapshot(
                archive_connection,
                receipt.reconciliation_id,
            )
            == expected_image
        )
        restored = sqlite3.connect(":memory:")
        try:
            legacy_archive.restore_legacy_database_snapshot(
                archive_connection,
                restored,
                receipt.reconciliation_id,
            )
            assert restored.serialize() == expected_image
            assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            restored.text_factory = bytes
            restored_status = restored.execute(
                "SELECT CAST(status AS BLOB) FROM review_requests WHERE review_id = ?",
                ("review_opaque",),
            ).fetchone()[0]
            assert restored_status == invalid_text
            objects = {
                (row[0].decode(), row[1].decode())
                for row in restored.execute(
                    "SELECT type, name FROM sqlite_master WHERE name LIKE '%exact' "
                    "OR name LIKE 'forensic_%'"
                )
            }
            assert objects == {
                ("index", "idx_legacy_review_status_exact"),
                ("trigger", "trg_legacy_review_insert_exact"),
                ("table", "forensic_shadowed_rowid"),
                ("table", "forensic_without_rowid"),
            }
            assert restored.execute(
                "SELECT rowid, _rowid_, oid, payload FROM forensic_shadowed_rowid"
            ).fetchone() == (
                b"visible-rowid",
                b"visible-alt",
                b"visible-oid",
                b"\x00shadowed\xff",
            )
        finally:
            restored.close()


def test_snapshot_readback_is_deserialized_and_verified_before_active_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _build_full_v2_database(tmp_path)
    original = _active_rows(path)
    real_verify = legacy_archive.verify_serialized_database_snapshot
    calls = 0

    def fail_second_verification(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected readback verification failure")
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(
        legacy_archive,
        "verify_serialized_database_snapshot",
        fail_second_verification,
    )

    with pytest.raises(DatabaseVerificationError) as excinfo:
        _reconcile(path)

    assert excinfo.value.stop_reason == "LEGACY_RECONCILIATION_FAILED"
    assert calls == 2
    assert _active_rows(path) == original
    with connect_database(path, read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'legacy_authorization_%'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("target", ["schema", "typed_row", "database_snapshot"])
def test_typed_archive_schema_and_rows_are_hash_bound_and_tamper_detected(
    tmp_path: Path,
    target: str,
) -> None:
    path = _build_opaque_v2_database(tmp_path)
    _reconcile(path)
    with connect_database(path) as connection:
        if target == "schema":
            trigger = "trg_legacy_authorization_table_manifests_immutable_update"
            connection.execute(f"DROP TRIGGER {trigger}")
            connection.execute(
                "UPDATE legacy_authorization_table_manifests SET source_table_sql = ? "
                "WHERE source_table = 'review_requests'",
                (sqlite3.Binary(b"CREATE TABLE forged(value)"),),
            )
        elif target == "typed_row":
            trigger = "trg_legacy_authorization_quarantine_immutable_update"
            connection.execute(f"DROP TRIGGER {trigger}")
            connection.execute(
                "UPDATE legacy_authorization_quarantine SET typed_row = ? "
                "WHERE source_table = 'payments'",
                (sqlite3.Binary(b"forged typed row"),),
            )
        else:
            trigger = "trg_legacy_authorization_database_snapshots_immutable_update"
            connection.execute(f"DROP TRIGGER {trigger}")
            connection.execute(
                "UPDATE legacy_authorization_database_snapshots SET database_image = ?",
                (sqlite3.Binary(b"forged database image"),),
            )
        connection.commit()

    with connect_database(path, read_only=True) as connection:
        assert legacy_archive.audit_legacy_authorization_archives(connection) == 1


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


def test_populated_row_only_archive_cannot_be_retrofitted_without_original_database_image(
    tmp_path: Path,
) -> None:
    path = _build_full_v2_database(tmp_path)
    _reconcile(path)
    with connect_database(path) as connection:
        connection.execute("DROP TABLE legacy_authorization_database_snapshots")
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as reconcile_error:
        _reconcile(path)
    assert reconcile_error.value.stop_reason == "LEGACY_RECONCILIATION_ARCHIVE_UPGRADE_REQUIRED"
    assert path.read_bytes() == before

    with pytest.raises(DatabaseVerificationError) as migrate_error:
        migrate_database(path, DatabaseKind.WORKFLOW)
    assert migrate_error.value.stop_reason == "LEGACY_RECONCILIATION_ARCHIVE_UPGRADE_REQUIRED"
    assert path.read_bytes() == before
