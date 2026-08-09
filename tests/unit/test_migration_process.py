"""Dedicated migration-process protocol and lifecycle regressions."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

import pytest

import invoice_agents.db.core as core_module
import invoice_agents.db.migration_process as migration_process
import invoice_agents.db.migration_worker as migration_worker
from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind, migrate_database
from invoice_agents.errors import DatabaseVerificationError, ErrorCategory


def _build_large_v2_workflow(path: Path) -> None:
    resources = core_module._migration_resources(DatabaseKind.WORKFLOW)
    with sqlite3.connect(path) as connection:
        connection.executescript(resources[0].read_text(encoding="utf-8"))
        connection.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (1, ?)",
            ("2026-08-09T12:00:00+00:00",),
        )
        connection.commit()
        connection.executescript(resources[1].read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (2, ?)",
            ("2026-08-09T12:00:00+00:00",),
        )
        connection.execute(
            "WITH RECURSIVE sequence(value) AS ("
            "VALUES(1) UNION ALL SELECT value + 1 FROM sequence WHERE value < 1000000"
            ") INSERT INTO cases(case_id, status, started_at, updated_at) "
            "SELECT printf('case_%09d', value), 'INCOMPLETE', ?, ? FROM sequence",
            ("2026-08-09T12:00:00+00:00", "2026-08-09T12:00:00+00:00"),
        )
        connection.commit()


def _rival_create_table(path: Path) -> str:
    script = """
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1], timeout=0.1)
try:
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("CREATE TABLE rival_committed_update(value INTEGER)")
    connection.commit()
    print("COMMITTED")
except sqlite3.OperationalError as exc:
    print(f"LOCKED:{exc}")
finally:
    connection.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip()


def test_public_migration_does_not_execute_migration_internals_in_caller_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "spawned-inventory.db"

    def reject_caller_process_resources(_kind: DatabaseKind) -> list[object]:
        raise AssertionError("public migration executed private work in the caller process")

    monkeypatch.setattr(core_module, "_migration_resources", reject_caller_process_resources)

    assert migrate_database(target, DatabaseKind.INVENTORY) == [1]
    assert target.is_file()


def test_parent_descriptor_close_cannot_release_spawned_worker_transaction_lock(
    tmp_path: Path,
) -> None:
    target = tmp_path / "process-owned-lock.db"
    _build_large_v2_workflow(target)
    applied: list[list[int]] = []
    failures: list[BaseException] = []

    def migrate() -> None:
        try:
            applied.append(migrate_database(target, DatabaseKind.WORKFLOW))
        except BaseException as exc:
            failures.append(exc)

    migration_thread = threading.Thread(target=migrate, name="public-migration-caller")
    migration_thread.start()
    deadline = time.monotonic() + 15
    transaction_journal: Path | None = None
    while time.monotonic() < deadline and migration_thread.is_alive():
        journals = list(tmp_path.glob(".invoice-db-maintenance-*/source-*.db-journal"))
        if journals:
            transaction_journal = journals[0]
            break
        time.sleep(0.001)
    assert transaction_journal is not None
    assert migration_thread.is_alive()

    unrelated_descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    os.close(unrelated_descriptor)
    first_writer = _rival_create_table(target)

    migration_thread.join(timeout=20)
    assert not migration_thread.is_alive()
    assert failures == []
    assert applied == [[3]]
    assert first_writer.startswith("LOCKED:database is locked")
    assert _rival_create_table(target) == "COMMITTED"
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 3
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'table' AND name = 'rival_committed_update'"
            ).fetchone()[0]
            == 1
        )
    assert not list(tmp_path.glob(".invoice-db-maintenance-*"))


def test_worker_rejects_malformed_secret_bearing_request_without_stderr_or_echo() -> None:
    secret = "sk-proj-malformed-worker-request-secret"

    completed = subprocess.run(
        [sys.executable, "-m", "invoice_agents.db.migration_worker"],
        input=f"not-json:{secret}\n",
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert secret not in completed.stdout
    response = json.loads(completed.stdout)
    assert response == {
        "ok": False,
        "error": {
            "category": "DATABASE",
            "message": "database migration worker protocol was invalid",
            "stop_reason": "MIGRATION_WORKER_PROTOCOL_INVALID",
            "details": None,
        },
    }


def test_worker_round_trips_expected_error_contract_with_json_safe_redaction() -> None:
    secret = "sk-proj-worker-error-detail-secret"
    original = DatabaseVerificationError(
        ErrorCategory.DATABASE,
        "schema audit failed",
        stop_reason="DATABASE_SCHEMA_MISMATCH",
        details={
            "missing_schema_objects": [("table", "payments", "payments")],
            "api_key": secret,
        },
    )
    response = migration_worker._safe_expected_failure(original)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process._decode_response(migration_process.encode_worker_response(response))

    assert excinfo.value.category is original.category
    assert excinfo.value.message == original.message
    assert excinfo.value.stop_reason == original.stop_reason
    assert excinfo.value.details == {
        "missing_schema_objects": [["table", "payments", "payments"]],
        "api_key": "[REDACTED]",
    }
    assert secret not in str(response)


def test_worker_crash_has_stable_error_without_raw_child_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-child-crash-output-secret"
    command = [
        sys.executable,
        "-c",
        f"import os,sys; sys.stderr.write({secret!r}); os._exit(17)",
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process.run_migration_in_subprocess(
            tmp_path / "crash.db",
            DatabaseKind.INVENTORY,
            settings=None,
        )

    assert excinfo.value.stop_reason == "MIGRATION_WORKER_CRASHED"
    assert secret not in str(excinfo.value)


def test_worker_crash_reaps_descendants_that_keep_the_protocol_pipe_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "worker-tree.pids"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"pathlib.Path({str(pid_path)!r}).write_text(f'{{os.getpid()}} {{child.pid}}'); "
            "os._exit(17)"
        ),
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    monkeypatch.setattr(migration_process, "MIGRATION_WORKER_TIMEOUT_SECONDS", 0.5)
    process_ids: list[int] = []

    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / "crash-tree.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )
        process_ids = [int(value) for value in pid_path.read_text().split()]

        assert excinfo.value.stop_reason == "MIGRATION_WORKER_CRASHED"
        for process_id in process_ids:
            deadline = time.monotonic() + 2
            while True:
                try:
                    os.kill(process_id, 0)
                except ProcessLookupError:
                    break
                assert time.monotonic() < deadline, f"worker descendant {process_id} survived"
                time.sleep(0.01)
    finally:
        for process_id in process_ids:
            with suppress(ProcessLookupError):
                os.kill(process_id, 9)


def test_worker_timeout_reaps_the_spawned_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "worker.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,time; "
            f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        ),
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    monkeypatch.setattr(migration_process, "MIGRATION_WORKER_TIMEOUT_SECONDS", 0.1)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process.run_migration_in_subprocess(
            tmp_path / "timeout.db",
            DatabaseKind.INVENTORY,
            settings=None,
        )

    assert excinfo.value.stop_reason == "MIGRATION_WORKER_TIMEOUT"
    worker_pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_worker_protocol_is_bounded_and_does_not_surface_raw_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-oversized-worker-output-secret"
    pid_path = tmp_path / "oversized-worker.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,time; "
            f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
            f"os.write(1, ({secret!r}.encode() * 2000)); "
            "time.sleep(60)"
        ),
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    monkeypatch.setattr(migration_process, "MIGRATION_WORKER_TIMEOUT_SECONDS", 0.5)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process.run_migration_in_subprocess(
            tmp_path / "oversized.db",
            DatabaseKind.INVENTORY,
            settings=None,
        )

    assert excinfo.value.stop_reason == "MIGRATION_WORKER_PROTOCOL_INVALID"
    assert secret not in str(excinfo.value)
    worker_pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_worker_request_serializes_explicit_settings_without_provider_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib,sys; "
            f"pathlib.Path({str(request_path)!r}).write_text(sys.stdin.read()); "
            'print(\'{"ok":true,"applied":[]}\')'
        ),
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    secret = "sk-proj-migration-channel-secret"
    settings = Settings(
        xai_api_key=secret,
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )

    assert (
        migration_process.run_migration_in_subprocess(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )
        == []
    )

    raw_request = request_path.read_text()
    request = json.loads(raw_request)
    assert secret not in raw_request
    assert request["protocol_version"] == 1
    assert request["path"] == str(settings.workflow_db)
    assert request["kind"] == "workflow"
    assert request["settings"]["xai_api_key"] is None
    assert request["settings"]["inventory_db"] == str(settings.inventory_db)
    assert request["settings"]["workflow_db"] == str(settings.workflow_db)
    assert request["settings"]["source_archive_dir"] == str(settings.source_archive_dir)
