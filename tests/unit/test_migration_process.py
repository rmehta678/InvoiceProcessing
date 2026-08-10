"""Dedicated migration-process protocol and lifecycle regressions."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import invoice_agents.db.core as core_module
import invoice_agents.db.migration_process as migration_process
import invoice_agents.db.migration_worker as migration_worker
from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind, migrate_database
from invoice_agents.errors import DatabaseVerificationError, ErrorCategory

_EXPECTED_WORKER_DOMAIN_MESSAGES = {
    "AUTHORIZATION_INVENTORY_WAL_MODE_UNSUPPORTED": (
        ErrorCategory.DATABASE,
        "authorization inventory database must use DELETE journal mode",
    ),
    "AUTHORIZATION_RECONCILIATION_REQUIRED": (
        ErrorCategory.DATABASE,
        "workflow authorization reconciliation is required before migration 003 "
        "(review_request_count=1, human_decision_count=2, final_decision_count=3, "
        "payment_count=4)",
    ),
    "DATABASE_AUTHORIZATION_CONTEXT_MISMATCH": (
        ErrorCategory.CONFIGURATION,
        "database migration authorization context does not match its target",
    ),
    "DATABASE_AUTHORIZATION_CONTEXT_REQUIRED": (
        ErrorCategory.CONFIGURATION,
        "database migration requires explicit authorization context",
    ),
    "DATABASE_AUTHORIZATION_PROVENANCE_INVALID": (
        ErrorCategory.DATABASE,
        "legacy workflow v3 authorization provenance is incomplete or inconsistent",
    ),
    "DATABASE_CHANGED_DURING_VERIFICATION": (
        ErrorCategory.DATABASE,
        "database source changed during migration verification",
    ),
    "DATABASE_INTEGRITY_FAILED": (
        ErrorCategory.DATABASE,
        "database integrity verification failed",
    ),
    "DATABASE_LOCK_UNAVAILABLE": (
        ErrorCategory.DATABASE,
        "database maintenance lock is unavailable",
    ),
    "DATABASE_MAINTENANCE_BINDING_FAILED": (
        ErrorCategory.DATABASE,
        "database maintenance binding could not be verified",
    ),
    "DATABASE_MAINTENANCE_CLEANUP_FAILED": (
        ErrorCategory.DATABASE,
        "database maintenance cleanup could not be verified",
    ),
    "DATABASE_MISSING": (ErrorCategory.DATABASE, "required database does not exist"),
    "DATABASE_SCHEMA_MISMATCH": (
        ErrorCategory.DATABASE,
        "database schema does not match the migration contract",
    ),
    "DATABASE_SIDECAR_UNSUPPORTED": (
        ErrorCategory.DATABASE,
        "database sidecar files are not supported during migration",
    ),
    "DATABASE_SIGNATURE_INVALID": (
        ErrorCategory.DATABASE,
        "database file signature is invalid",
    ),
    "DATABASE_SYMLINK_UNSUPPORTED": (
        ErrorCategory.DATABASE,
        "database symlink paths are not supported during migration",
    ),
    "DATABASE_VERIFICATION_ERROR": (
        ErrorCategory.DATABASE,
        "database verification failed",
    ),
    "DATABASE_VERSION_MISMATCH": (
        ErrorCategory.DATABASE,
        "database schema version does not match the migration contract",
    ),
    "INVENTORY_WAL_MODE_UNSUPPORTED": (
        ErrorCategory.DATABASE,
        "inventory database must use DELETE journal mode",
    ),
    "LEGACY_RECONCILIATION_ARCHIVE_INVALID": (
        ErrorCategory.DATABASE,
        "legacy reconciliation archive failed integrity verification",
    ),
    "LEGACY_RECONCILIATION_ARCHIVE_UPGRADE_REQUIRED": (
        ErrorCategory.DATABASE,
        "legacy authorization archive requires a lossless upgrade",
    ),
    "LEGACY_RECONCILIATION_DELETE_INCOMPLETE": (
        ErrorCategory.DATABASE,
        "legacy authorization reconciliation did not remove every active row",
    ),
    "LEGACY_RECONCILIATION_FAILED": (
        ErrorCategory.DATABASE,
        "legacy authorization reconciliation failed atomically",
    ),
    "LEGACY_RECONCILIATION_STATE_INVALID": (
        ErrorCategory.DATABASE,
        "legacy authorization reconciliation state is invalid",
    ),
    "MIGRATION_FAILED": (ErrorCategory.DATABASE, "database migration failed"),
    "MIGRATION_HISTORY_INVALID": (
        ErrorCategory.DATABASE,
        "database migration history is invalid",
    ),
    "MIGRATION_NOT_FOUND": (
        ErrorCategory.DATABASE,
        "database migration resources are unavailable",
    ),
    "WORKFLOW_WAL_MODE_UNSUPPORTED": (
        ErrorCategory.DATABASE,
        "workflow database must use DELETE journal mode",
    ),
}


def _assert_processes_gone(process_ids: list[int]) -> None:
    for process_id in process_ids:
        deadline = time.monotonic() + 2
        while True:
            try:
                os.kill(process_id, 0)
            except ProcessLookupError:
                break
            assert time.monotonic() < deadline, f"worker process {process_id} survived"
            time.sleep(0.01)


def _kill_processes(process_ids: list[int]) -> None:
    for process_id in process_ids:
        with suppress(ProcessLookupError):
            os.kill(process_id, 9)


def _release_quarantine_after_external_stop(process_ids: list[int]) -> None:
    """Model an operator stopping an unsignalable Darwin descendant, then retry cleanup."""

    _kill_processes(process_ids)
    deadline = time.monotonic() + 2
    while not migration_process._retry_quarantined_workers():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _install_member_signal_controller(
    monkeypatch: pytest.MonkeyPatch,
    should_deny: Callable[[int], bool],
) -> list[int]:
    """Record and optionally deny the platform's exact descendant signal primitive."""

    signalled_targets: list[int] = []
    if sys.platform.startswith("linux"):
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        assert callable(pidfd_open)
        assert callable(pidfd_send_signal)
        handle_process_ids: dict[int, int] = {}

        def controlled_pidfd_open(process_id: int) -> int:
            handle = int(pidfd_open(process_id))
            handle_process_ids[handle] = process_id
            return handle

        def controlled_pidfd_send_signal(
            handle: int,
            signal_number: int,
            info: object,
            flags: int,
        ) -> None:
            process_id = handle_process_ids[handle]
            signalled_targets.append(process_id)
            if should_deny(signal_number):
                raise PermissionError("injected worker cleanup denial")
            pidfd_send_signal(handle, signal_number, info, flags)

        monkeypatch.setattr(migration_process.os, "pidfd_open", controlled_pidfd_open)
        monkeypatch.setattr(
            migration_process.signal,
            "pidfd_send_signal",
            controlled_pidfd_send_signal,
        )
        return signalled_targets

    real_killpg = os.killpg

    def controlled_killpg(process_group_id: int, signal_number: int) -> None:
        signalled_targets.append(process_group_id)
        if should_deny(signal_number):
            raise PermissionError("injected worker cleanup denial")
        real_killpg(process_group_id, signal_number)

    monkeypatch.setattr(migration_process.os, "killpg", controlled_killpg)
    return signalled_targets


def _exception_graph_text(error: BaseException) -> str:
    nodes: list[str] = ["".join(traceback.format_exception(error)), repr(error.args)]
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        nodes.extend((repr(current.args), repr(current.__cause__), repr(current.__context__)))
        current = current.__cause__ or current.__context__
    return "\n".join(nodes)


def _setpgrp_worker_command(
    marker_path: Path,
    response: str,
    *,
    ignore_term: bool = False,
    escape_group: bool = True,
) -> list[str]:
    child_script = (
        "import os,pathlib,signal,time;"
        + ("os.setpgrp();" if escape_group else "")
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_term else "")
        + f"pathlib.Path({str(marker_path)!r}).write_text("
        "f'{os.getpid()} {os.getpgrp()} {os.getsid(0)}');"
        "time.sleep(60)"
    )
    worker_script = (
        "import pathlib,subprocess,sys,time;"
        f"marker=pathlib.Path({str(marker_path)!r});"
        f"subprocess.Popen([sys.executable,'-c',{child_script!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        "deadline=time.monotonic()+5;"
        "\nwhile not marker.exists():\n"
        "    assert time.monotonic() < deadline\n"
        "    time.sleep(0.001)\n"
        f"print({response!r})"
    )
    return [sys.executable, "-c", worker_script]


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
            "message": "database migration worker returned an invalid bounded response",
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

    assert response == {
        "ok": False,
        "error": {
            "category": "DATABASE",
            "message": "database schema does not match the migration contract",
            "stop_reason": "DATABASE_SCHEMA_MISMATCH",
            "details": None,
        },
    }
    assert excinfo.value.category is original.category
    assert excinfo.value.message == "database schema does not match the migration contract"
    assert excinfo.value.stop_reason == original.stop_reason
    assert excinfo.value.details is None
    assert secret not in str(response)


@pytest.mark.parametrize("stop_reason", sorted(_EXPECTED_WORKER_DOMAIN_MESSAGES))
def test_every_allowlisted_worker_domain_error_has_parent_owned_output(
    stop_reason: str,
) -> None:
    category, expected_message = _EXPECTED_WORKER_DOMAIN_MESSAGES[stop_reason]
    canary = f"canary-child-domain-{stop_reason.lower()}"
    if stop_reason == "AUTHORIZATION_RECONCILIATION_REQUIRED":
        details: dict[str, object] = {
            "review_request_count": 1,
            "human_decision_count": 2,
            "final_decision_count": 3,
            "payment_count": 4,
        }
    elif stop_reason == "DATABASE_AUTHORIZATION_PROVENANCE_INVALID":
        details = {
            "invalid_review_count": 1,
            "invalid_human_decision_count": 2,
            "invalid_snapshot_count": 3,
            "invalid_final_decision_count": 4,
            "invalid_payment_count": 5,
            "invalid_cardinality_count": 6,
            "invalid_quarantine_count": 7,
        }
    else:
        details = {"child_controlled": [canary]}
    original = DatabaseVerificationError(
        category,
        canary,
        stop_reason=stop_reason,
        details=details,
    )

    response = migration_worker._safe_expected_failure(original)
    error_payload = response["error"]
    assert isinstance(error_payload, dict)
    expected_details = (
        details
        if stop_reason
        in {
            "AUTHORIZATION_RECONCILIATION_REQUIRED",
            "DATABASE_AUTHORIZATION_PROVENANCE_INVALID",
        }
        else None
    )
    assert error_payload == {
        "category": category.value,
        "message": expected_message,
        "stop_reason": stop_reason,
        "details": expected_details,
    }
    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process._decode_response(migration_process.encode_worker_response(response))
    assert excinfo.value.category is category
    assert excinfo.value.message == expected_message
    assert excinfo.value.stop_reason == stop_reason
    assert excinfo.value.details == expected_details
    assert canary not in _exception_graph_text(excinfo.value)


def test_public_worker_error_rejects_every_child_controlled_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canaries = {
        "message": "canary-child-message",
        "nested_key": "canary-child-nested-key",
        "nested_value": "canary-child-nested-value",
        "list_value": "canary-child-list-value",
        "code": "CANARY_CHILD_CODE",
        "traceback": "canary-child-traceback",
        "cause": "canary-child-cause",
        "context": "canary-child-context",
        "stdout": "canary-child-stdout",
        "stderr": "canary-child-stderr",
    }
    payloads = [
        {
            "ok": False,
            "error": {
                "category": "DATABASE",
                "message": canaries["message"],
                "stop_reason": "MIGRATION_HISTORY_INVALID",
                "details": {
                    canaries["nested_key"]: {
                        "value": canaries["nested_value"],
                        "items": [canaries["list_value"]],
                    },
                    "traceback": canaries["traceback"],
                    "cause": canaries["cause"],
                    "context": canaries["context"],
                    "stdout": canaries["stdout"],
                    "stderr": canaries["stderr"],
                },
            },
        },
        {
            "ok": False,
            "error": {
                "category": "DATABASE",
                "message": "database migration history is invalid",
                "stop_reason": canaries["code"],
                "details": None,
            },
        },
    ]

    for index, payload in enumerate(payloads):
        encoded = json.dumps(payload)
        command = [
            sys.executable,
            "-c",
            (f"import sys;sys.stderr.write({canaries['stderr']!r});sys.stdout.write({encoded!r})"),
        ]
        monkeypatch.setattr(
            migration_process,
            "_worker_command",
            lambda command=command: command,
        )

        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / f"child-controlled-error-{index}.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )

        error = excinfo.value
        assert error.stop_reason == "MIGRATION_WORKER_PROTOCOL_INVALID"
        assert error.message == "database migration worker returned an invalid bounded response"
        assert error.details is None
        public_surface = _exception_graph_text(error)
        for canary in canaries.values():
            assert canary not in public_surface


@pytest.mark.parametrize(
    ("raw_response", "canary"),
    [
        (
            '{"ok":false,"error":{"category":"DATABASE",'
            '"message":"database migration history is invalid",'
            '"stop_reason":"MIGRATION_HISTORY_INVALID","details":null},'
            '"canary-extra-top-key":"canary-extra-top-value"}',
            "canary-extra-top-value",
        ),
        (
            '{"ok":false,"error":{"category":"DATABASE",'
            '"message":"database migration history is invalid",'
            '"stop_reason":"MIGRATION_HISTORY_INVALID","details":null,'
            '"traceback":"canary-extra-error-key"}}',
            "canary-extra-error-key",
        ),
        (
            '{"ok":false,"error":{"category":"DATABASE",'
            '"message":"canary-duplicate-key",'
            '"message":"database migration history is invalid",'
            '"stop_reason":"MIGRATION_HISTORY_INVALID","details":null}}',
            "canary-duplicate-key",
        ),
        (
            '{"ok":false,"error":{"category":"canary-category",'
            '"message":"database migration history is invalid",'
            '"stop_reason":"MIGRATION_HISTORY_INVALID","details":null}}',
            "canary-category",
        ),
        (
            '{"ok":false,"error":{"category":"DATABASE",'
            '"message":"database migration history is invalid",'
            '"stop_reason":"MIGRATION_HISTORY_INVALID",'
            '"details":"canary-string-details"}}',
            "canary-string-details",
        ),
        (
            '{"ok":false,"error":{"category":"DATABASE",'
            '"message":"database migration history is invalid",'
            '"stop_reason":"MIGRATION_HISTORY_INVALID","details":NaN},'
            '"canary-non-finite":"canary-non-finite-value"}',
            "canary-non-finite-value",
        ),
    ],
    ids=[
        "extra-top-level-key",
        "extra-error-key",
        "duplicate-key",
        "category-mismatch",
        "string-details",
        "non-finite-number",
    ],
)
def test_worker_protocol_ambiguity_becomes_chainless_parent_owned_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_response: str,
    canary: str,
) -> None:
    command = [sys.executable, "-c", f"import sys;sys.stdout.write({raw_response!r})"]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process.run_migration_in_subprocess(
            tmp_path / "ambiguous-worker-response.db",
            DatabaseKind.INVENTORY,
            settings=None,
        )

    error = excinfo.value
    assert error.stop_reason == "MIGRATION_WORKER_PROTOCOL_INVALID"
    assert error.message == "database migration worker returned an invalid bounded response"
    assert error.details is None
    assert error.__cause__ is None
    assert error.__context__ is None
    assert canary not in _exception_graph_text(error)


def test_public_error_copy_discards_a_secret_bearing_exception_chain() -> None:
    secret = "sk-proj-private-worker-exception-chain-secret"
    private_error = migration_process._protocol_error(
        "MIGRATION_WORKER_START_FAILED",
        "database migration worker could not be started",
    )
    private_error.__cause__ = OSError(secret)

    public_error = migration_process._copy_public_error(private_error)

    assert public_error is not private_error
    assert public_error.category is private_error.category
    assert public_error.message == private_error.message
    assert public_error.stop_reason == private_error.stop_reason
    assert public_error.__cause__ is None
    assert public_error.__context__ is None
    assert secret not in _exception_graph_text(public_error)


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


def test_worker_spawn_failure_has_stable_chainless_secret_safe_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-worker-spawn-constructor-secret"

    def fail_spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        raise OSError(secret)

    monkeypatch.setattr(migration_process.subprocess, "Popen", fail_spawn)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process.run_migration_in_subprocess(
            tmp_path / "spawn-failure.db",
            DatabaseKind.INVENTORY,
            settings=None,
        )

    error = excinfo.value
    assert error.stop_reason == "MIGRATION_WORKER_START_FAILED"
    assert error.message == "database migration worker could not be started"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in _exception_graph_text(error)


def test_worker_watcher_constructor_failure_cleans_or_quarantines_reserved_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-worker-watcher-constructor-secret"
    marker_path = tmp_path / "watcher-failure-worker-member.txt"
    monkeypatch.setattr(
        migration_process,
        "_worker_command",
        lambda: _setpgrp_worker_command(marker_path, '{"ok":true,"applied":[]}'),
    )

    def fail_watcher_after_child_creation(_process_id: int) -> Any:
        deadline = time.monotonic() + 5
        while not marker_path.exists():
            assert time.monotonic() < deadline
            time.sleep(0.001)
        raise OSError(secret)

    monkeypatch.setattr(migration_process, "_WorkerExitWatcher", fail_watcher_after_child_creation)
    signalled_targets = _install_member_signal_controller(monkeypatch, lambda _signal: False)
    process_ids: list[int] = []

    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / "watcher-constructor-failure.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )
        child_pid, child_group, child_session = map(int, marker_path.read_text().split())
        process_ids = [child_pid]

        assert child_group == child_pid
        assert child_session != child_group
        if sys.platform == "darwin":
            assert excinfo.value.stop_reason == "MIGRATION_WORKER_CLEANUP_FAILED"
            assert len(migration_process._QUARANTINED_WORKERS) == 1
            retained = next(iter(migration_process._QUARANTINED_WORKERS.values()))
            assert retained.process.returncode is None
            assert child_group not in signalled_targets
        else:
            assert excinfo.value.stop_reason == "MIGRATION_WORKER_START_FAILED"
            _assert_processes_gone(process_ids)
    finally:
        if sys.platform == "darwin" and process_ids:
            _release_quarantine_after_external_stop(process_ids)
        else:
            _kill_processes(process_ids)


def test_worker_watcher_constructor_failure_has_no_secret_bearing_exception_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-worker-watcher-chain-secret"
    command = [sys.executable, "-c", 'print(\'{"ok":true,"applied":[]}\')']
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)

    def fail_watcher(_process_id: int) -> Any:
        raise OSError(secret)

    monkeypatch.setattr(migration_process, "_WorkerExitWatcher", fail_watcher)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process.run_migration_in_subprocess(
            tmp_path / "watcher-chain-failure.db",
            DatabaseKind.INVENTORY,
            settings=None,
        )

    error = excinfo.value
    assert error.stop_reason == "MIGRATION_WORKER_START_FAILED"
    assert error.message == "database migration worker could not be started"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in _exception_graph_text(error)


def test_worker_watcher_initialization_closes_partial_kernel_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-worker-watcher-initialization-secret"

    class FailingKqueue:
        closed = False

        def control(self, *_args: object) -> list[object]:
            raise OSError(secret)

        def close(self) -> None:
            self.closed = True

    kernel_queue = FailingKqueue()
    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    monkeypatch.setattr(migration_process.select, "kqueue", lambda: kernel_queue)
    monkeypatch.setattr(migration_process.select, "kevent", lambda *_args, **_kwargs: object())

    with pytest.raises(OSError, match=secret):
        migration_process._WorkerExitWatcher(12345)

    assert kernel_queue.closed


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
        _assert_processes_gone(process_ids)
    finally:
        _kill_processes(process_ids)


def test_worker_success_reaps_same_session_descendant_with_closed_stdio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "successful-worker-tree.pids"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            f"pathlib.Path({str(pid_path)!r}).write_text(f'{{os.getpid()}} {{child.pid}}'); "
            'print(\'{"ok":true,"applied":[]}\')'
        ),
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    process_ids: list[int] = []

    try:
        assert (
            migration_process.run_migration_in_subprocess(
                tmp_path / "success-tree.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )
            == []
        )
        process_ids = [int(value) for value in pid_path.read_text().split()]
        _assert_processes_gone(process_ids)
    finally:
        _kill_processes(process_ids)


def test_worker_expected_error_reaps_same_session_descendant_with_closed_stdio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "expected-error-worker-tree.pids"
    command = [
        sys.executable,
        "-c",
        (
            "import json,os,pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            f"pathlib.Path({str(pid_path)!r}).write_text(f'{{os.getpid()}} {{child.pid}}'); "
            "print(json.dumps({'ok':False,'error':{'category':'DATABASE',"
            "'message':'database migration history is invalid',"
            "'stop_reason':'MIGRATION_HISTORY_INVALID',"
            "'details':None}}))"
        ),
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    process_ids: list[int] = []

    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / "expected-error-tree.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )
        process_ids = [int(value) for value in pid_path.read_text().split()]

        assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
        _assert_processes_gone(process_ids)
    finally:
        _kill_processes(process_ids)


@pytest.mark.parametrize(
    ("response", "expected_stop_reason"),
    [
        ('{"ok":true,"applied":[]}', None),
        (
            '{"ok":false,"error":{"category":"DATABASE",'
            '"message":"database migration history is invalid",'
            '"stop_reason":"MIGRATION_HISTORY_INVALID","details":null}}',
            "MIGRATION_HISTORY_INVALID",
        ),
    ],
    ids=["success", "expected-domain-error"],
)
def test_worker_reaps_same_session_descendant_that_escapes_leader_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    expected_stop_reason: str | None,
) -> None:
    marker_path = tmp_path / "escaped-worker-member.txt"
    monkeypatch.setattr(
        migration_process,
        "_worker_command",
        lambda: _setpgrp_worker_command(marker_path, response),
    )
    signalled_targets = _install_member_signal_controller(monkeypatch, lambda _signal: False)
    process_ids: list[int] = []

    try:
        if sys.platform == "darwin":
            with pytest.raises(DatabaseVerificationError) as excinfo:
                migration_process.run_migration_in_subprocess(
                    tmp_path / "escaped-darwin.db",
                    DatabaseKind.INVENTORY,
                    settings=None,
                )
            assert excinfo.value.stop_reason == "MIGRATION_WORKER_CLEANUP_FAILED"
        elif expected_stop_reason is None:
            assert (
                migration_process.run_migration_in_subprocess(
                    tmp_path / "escaped-success.db",
                    DatabaseKind.INVENTORY,
                    settings=None,
                )
                == []
            )
        else:
            with pytest.raises(DatabaseVerificationError) as excinfo:
                migration_process.run_migration_in_subprocess(
                    tmp_path / "escaped-domain-error.db",
                    DatabaseKind.INVENTORY,
                    settings=None,
                )
            assert excinfo.value.stop_reason == expected_stop_reason
        child_pid, child_group, child_session = map(int, marker_path.read_text().split())
        process_ids = [child_pid]

        assert child_group == child_pid
        assert child_session != child_group
        if sys.platform == "darwin":
            assert len(migration_process._QUARANTINED_WORKERS) == 1
            assert child_group not in signalled_targets
        else:
            _assert_processes_gone(process_ids)
    finally:
        if sys.platform == "darwin" and process_ids:
            _release_quarantine_after_external_stop(process_ids)
        else:
            _kill_processes(process_ids)


def test_worker_watcher_failure_quarantines_unreaped_session_until_cleanup_is_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-worker-watcher-cleanup-secret"
    marker_path = tmp_path / "watcher-cleanup-failure-member.txt"
    monkeypatch.setattr(
        migration_process,
        "_worker_command",
        lambda: _setpgrp_worker_command(marker_path, '{"ok":true,"applied":[]}'),
    )
    monkeypatch.setattr(migration_process, "_WORKER_SHUTDOWN_SECONDS", 0.02)
    monkeypatch.setattr(migration_process, "_WORKER_QUARANTINE_RETRY_SECONDS", 0.1)
    monkeypatch.setattr(migration_process, "_WORKER_QUARANTINE_RETRY_ATTEMPTS", 1)

    def fail_watcher_after_child_creation(_process_id: int) -> Any:
        deadline = time.monotonic() + 5
        while not marker_path.exists():
            assert time.monotonic() < deadline
            time.sleep(0.001)
        raise OSError(secret)

    monkeypatch.setattr(migration_process, "_WorkerExitWatcher", fail_watcher_after_child_creation)
    deny_group_signals = True
    signalled_targets = _install_member_signal_controller(
        monkeypatch,
        lambda _signal_number: deny_group_signals,
    )
    child_pid = 0
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )

    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / "watcher-cleanup-failure.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )
        child_pid, child_group, _child_session = map(int, marker_path.read_text().split())

        error = excinfo.value
        assert error.stop_reason == "MIGRATION_WORKER_CLEANUP_FAILED"
        assert error.message == "database migration worker session cleanup could not be verified"
        assert error.__cause__ is None
        assert error.__context__ is None
        assert secret not in _exception_graph_text(error)
        assert len(migration_process._QUARANTINED_WORKERS) == 1
        quarantined_worker = next(iter(migration_process._QUARANTINED_WORKERS.values()))
        assert quarantined_worker.process_id != child_pid
        assert quarantined_worker.process.returncode is None
        snapshot = migration_process._worker_session_snapshot(quarantined_worker)
        assert snapshot.leader_state.startswith("Z")
        assert snapshot.members == (migration_process._WorkerSessionMember(child_pid, child_group),)
        assert unrelated.poll() is None
        assert unrelated.pid not in signalled_targets

        deny_group_signals = False
        if sys.platform == "darwin":
            assert not migration_process._retry_quarantined_workers()
            assert signalled_targets == []
            assert quarantined_worker.process.returncode is None
            _release_quarantine_after_external_stop([child_pid])
        else:
            assert migration_process._retry_quarantined_workers()
            assert signalled_targets and set(signalled_targets) == {child_pid}
        assert quarantined_worker.process.returncode is not None
        _assert_processes_gone([child_pid])
        assert unrelated.poll() is None
        deadline = time.monotonic() + 1
        while migration_process._QUARANTINE_RETRY_THREAD_RUNNING:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not [
            thread
            for thread in threading.enumerate()
            if thread.name == "migration-worker-cleanup" and thread.is_alive()
        ]
    finally:
        deny_group_signals = False
        if migration_process._QUARANTINED_WORKERS and child_pid:
            _release_quarantine_after_external_stop([child_pid])
        else:
            _kill_processes([child_pid] if child_pid else [])
        unrelated.kill()
        unrelated.wait()


@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize(
    "worker_response",
    [
        '{"ok":true,"applied":[]}',
        (
            '{"ok":false,"error":{"category":"DATABASE",'
            '"message":"database migration history is invalid",'
            '"stop_reason":"MIGRATION_HISTORY_INVALID","details":null}}'
        ),
    ],
    ids=["success-response", "domain-error-response"],
)
def test_worker_signal_permission_failure_overrides_response_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    denied_signal: int,
    worker_response: str,
) -> None:
    marker_path = tmp_path / "permission-worker-member.txt"
    monkeypatch.setattr(
        migration_process,
        "_worker_command",
        lambda: _setpgrp_worker_command(
            marker_path,
            worker_response,
            ignore_term=denied_signal == signal.SIGKILL,
            escape_group=False,
        ),
    )
    monkeypatch.setattr(migration_process, "_WORKER_SHUTDOWN_SECONDS", 0.05)
    deny_signals = True
    signalled_targets = _install_member_signal_controller(
        monkeypatch,
        lambda signal_number: deny_signals and signal_number == denied_signal,
    )
    child_pid = 0
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )

    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / "permission-cleanup.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )
        child_pid, child_group, _child_session = map(int, marker_path.read_text().split())

        assert excinfo.value.stop_reason == "MIGRATION_WORKER_CLEANUP_FAILED"
        assert unrelated.poll() is None
        expected_target = child_pid if sys.platform.startswith("linux") else child_group
        assert signalled_targets and set(signalled_targets) == {expected_target}

        deny_signals = False
        assert migration_process._retry_quarantined_workers()
        _assert_processes_gone([child_pid])
        assert unrelated.poll() is None
    finally:
        deny_signals = False
        _kill_processes([child_pid] if child_pid else [])
        unrelated.kill()
        unrelated.wait()


@pytest.mark.parametrize(
    "enumeration_fault",
    [
        "process-failure",
        "malformed-output",
        "stderr-output",
        "nonzero-return",
        "empty-output",
        "missing-leader",
        "duplicate-leader",
        "noncanonical-leader",
        "invalid-stat",
    ],
)
def test_worker_enumeration_failure_overrides_success_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enumeration_fault: str,
) -> None:
    command = [sys.executable, "-c", 'print(\'{"ok":true,"applied":[]}\')']
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    monkeypatch.setattr(migration_process, "_WORKER_SHUTDOWN_SECONDS", 0.01)
    real_run = subprocess.run
    real_capture = migration_process._capture_worker_session
    inject_fault = True
    worker_pid = 0

    def capture_worker(process: subprocess.Popen[bytes]) -> Any:
        nonlocal worker_pid
        worker = real_capture(process)
        worker_pid = worker.process_id
        return worker

    monkeypatch.setattr(migration_process, "_capture_worker_session", capture_worker)

    def controlled_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command_value = args[0] if args else None
        is_process_enumeration = (
            isinstance(command_value, list)
            and len(command_value) == 3
            and command_value[:2] == ["/bin/ps", "-axo"]
            and isinstance(command_value[2], str)
            and command_value[2].startswith("pid=,pgid=")
            and command_value[2].endswith("stat=")
        )
        if inject_fault and is_process_enumeration:
            if enumeration_fault == "process-failure":
                raise OSError("injected ps failure")
            if enumeration_fault == "stderr-output":
                return subprocess.CompletedProcess(
                    command_value, 0, stdout="", stderr="ps warning\n"
                )
            if enumeration_fault == "nonzero-return":
                return subprocess.CompletedProcess(command_value, 17, stdout="", stderr="")
            if enumeration_fault == "malformed-output":
                return subprocess.CompletedProcess(
                    command_value, 0, stdout="malformed\n", stderr=""
                )
            if enumeration_fault == "empty-output":
                return subprocess.CompletedProcess(command_value, 0, stdout="", stderr="")
            completed = real_run(*args, **kwargs)
            assert worker_pid > 0
            lines = completed.stdout.splitlines()
            leader_at = next(
                index
                for index, line in enumerate(lines)
                if line.split() and int(line.split()[0]) == worker_pid
            )
            leader_fields = lines[leader_at].split()
            if enumeration_fault == "missing-leader":
                del lines[leader_at]
            elif enumeration_fault == "duplicate-leader":
                lines.insert(leader_at, lines[leader_at])
            elif enumeration_fault == "noncanonical-leader":
                leader_fields[0] = f"0{leader_fields[0]}"
                lines[leader_at] = " ".join(leader_fields)
            else:
                leader_fields[-1] = "NOT_A_PS_STATE"
                lines[leader_at] = " ".join(leader_fields)
            return subprocess.CompletedProcess(
                command_value,
                0,
                stdout="\n".join(lines) + ("\n" if lines else ""),
                stderr="",
            )
        return real_run(*args, **kwargs)

    monkeypatch.setattr(migration_process.subprocess, "run", controlled_run)
    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process.run_migration_in_subprocess(
            tmp_path / "enumeration-cleanup.db",
            DatabaseKind.INVENTORY,
            settings=None,
        )

    assert excinfo.value.stop_reason == "MIGRATION_WORKER_CLEANUP_FAILED"
    inject_fault = False
    assert migration_process._retry_quarantined_workers()


@pytest.mark.parametrize(
    ("platform", "format_field", "leader_state", "member_state"),
    [
        ("darwin", "pid=,pgid=,stat=", "Zs", "S<+"),
        ("darwin", "pid=,pgid=,stat=", "?s", "R+"),
        ("linux", "pid=,pgid=,sid=,stat=", "Zs", "D<l+"),
        ("linux", "pid=,pgid=,sid=,stat=", "Zs", "P"),
        ("linux", "pid=,pgid=,sid=,stat=", "Zs", "K"),
        ("linux", "pid=,pgid=,sid=,stat=", "Zs", "x"),
    ],
)
def test_worker_process_enumeration_accepts_explicit_platform_stat_grammars(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    format_field: str,
    leader_state: str,
    member_state: str,
) -> None:
    worker = cast(
        Any,
        SimpleNamespace(process_id=101, process_group_id=101, session_id=101),
    )
    command = ["/bin/ps", "-axo", format_field]
    if platform == "darwin":
        stdout = f"101 101 {leader_state}\n102 102 {member_state}\n"
    else:
        stdout = f"101 101 101 {leader_state}\n102 102 101 {member_state}\n"
    monkeypatch.setattr(migration_process.sys, "platform", platform)
    monkeypatch.setattr(
        migration_process.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=""),
    )
    monkeypatch.setattr(migration_process.os, "getsid", lambda process_id: 101)
    monkeypatch.setattr(migration_process.os, "getpgid", lambda process_id: process_id)

    members = migration_process._worker_session_members(worker)

    assert members == (migration_process._WorkerSessionMember(102, 102),)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "101 101 101 NOT_A_PS_STATE\n",
        "101 101 101 Z trailing\n",
        "zero 101 101 Z\n",
        "0 101 101 Z\n",
        "-1 101 101 Z\n",
        "0101 101 101 Z\n",
        "101 101 101 Z\n101 101 101 Z\n",
        "101 999 101 Z\n",
        "101 101 999 Z\n",
        "999 999 999 S\n",
    ],
    ids=[
        "empty",
        "invalid-stat",
        "trailing-field",
        "nonnumeric",
        "zero",
        "negative",
        "leading-zero",
        "duplicate",
        "leader-group-mismatch",
        "leader-session-mismatch",
        "missing-leader",
    ],
)
def test_worker_process_enumeration_rejects_uncertain_identity_grammar(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
) -> None:
    worker = cast(
        Any,
        SimpleNamespace(process_id=101, process_group_id=101, session_id=101),
    )
    command = ["/bin/ps", "-axo", "pid=,pgid=,sid=,stat="]
    monkeypatch.setattr(migration_process.sys, "platform", "linux")
    monkeypatch.setattr(
        migration_process.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=""),
    )

    with pytest.raises(migration_process._WorkerCleanupFailure):
        migration_process._worker_session_members(worker)


@pytest.mark.parametrize("identity_fault", ["changed-session", "esrch"])
def test_worker_group_identity_uncertainty_never_signals_the_group(
    monkeypatch: pytest.MonkeyPatch,
    identity_fault: str,
) -> None:
    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    worker = cast(
        Any,
        SimpleNamespace(process_id=101, session_id=101, process_group_id=101),
    )
    members = (migration_process._WorkerSessionMember(202, 202),)
    signals: list[tuple[int, int]] = []
    session_queries = 0

    def controlled_getsid(_process_id: int) -> int:
        nonlocal session_queries
        session_queries += 1
        if identity_fault == "esrch":
            raise ProcessLookupError
        return 101 if session_queries == 1 else 303

    monkeypatch.setattr(migration_process.os, "getsid", controlled_getsid)
    monkeypatch.setattr(migration_process.os, "getpgid", lambda _process_id: 202)
    monkeypatch.setattr(
        migration_process.os,
        "killpg",
        lambda group_id, signal_number: signals.append((group_id, signal_number)),
    )

    with pytest.raises(migration_process._WorkerCleanupFailure):
        migration_process._signal_worker_session_groups(worker, members, signal.SIGKILL)

    assert signals == []


def test_worker_group_without_live_group_leader_is_never_signalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    worker = cast(
        Any,
        SimpleNamespace(process_id=101, session_id=101, process_group_id=101),
    )
    members = (migration_process._WorkerSessionMember(203, 202),)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(migration_process.os, "getsid", lambda _process_id: 101)
    monkeypatch.setattr(migration_process.os, "getpgid", lambda _process_id: 202)
    monkeypatch.setattr(
        migration_process.os,
        "killpg",
        lambda group_id, signal_number: signals.append((group_id, signal_number)),
    )

    with pytest.raises(migration_process._WorkerCleanupFailure):
        migration_process._signal_worker_session_groups(worker, members, signal.SIGKILL)

    assert signals == []


def test_darwin_descendant_group_identity_change_at_action_boundary_never_signals_numeric_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated numeric PGID can be reassigned before killpg performs its action."""

    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    worker = cast(
        Any,
        SimpleNamespace(process_id=101, session_id=101, process_group_id=101),
    )
    members = (
        migration_process._WorkerSessionMember(202, 202),
        migration_process._WorkerSessionMember(203, 202),
        migration_process._WorkerSessionMember(303, 303),
        migration_process._WorkerSessionMember(304, 303),
    )
    final_validation_returned = False
    unrelated_group_signals: list[tuple[int, int]] = []

    def validated_getpgid(_process_id: int) -> int:
        nonlocal final_validation_returned
        final_validation_returned = True
        return 202

    def numeric_group_action(group_id: int, signal_number: int) -> None:
        assert final_validation_returned
        unrelated_group_signals.append((group_id, signal_number))

    monkeypatch.setattr(migration_process.os, "getsid", lambda _process_id: 101)
    monkeypatch.setattr(migration_process.os, "getpgid", validated_getpgid)
    monkeypatch.setattr(migration_process.os, "killpg", numeric_group_action)

    with pytest.raises(migration_process._WorkerCleanupFailure):
        migration_process._signal_worker_session_groups(worker, members, signal.SIGKILL)

    assert unrelated_group_signals == []


def test_darwin_original_group_signal_is_bound_by_unreaped_leader_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = cast(
        Any,
        SimpleNamespace(process_id=101, process_group_id=101, session_id=101),
    )
    members = (
        migration_process._WorkerSessionMember(202, 101),
        migration_process._WorkerSessionMember(203, 101),
    )
    group_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    monkeypatch.setattr(
        migration_process.os,
        "killpg",
        lambda group_id, signal_number: group_signals.append((group_id, signal_number)),
    )

    migration_process._signal_worker_session_groups(worker, members, signal.SIGTERM)

    assert group_signals == [(101, signal.SIGTERM)]


@pytest.mark.parametrize(
    ("process_id", "process_group_id", "session_id"),
    [(101, 102, 101), (101, 101, 102), (101, 102, 103)],
)
def test_darwin_initial_session_identity_mismatch_never_signals_any_group(
    monkeypatch: pytest.MonkeyPatch,
    process_id: int,
    process_group_id: int,
    session_id: int,
) -> None:
    worker = cast(
        Any,
        SimpleNamespace(
            process_id=process_id,
            process_group_id=process_group_id,
            session_id=session_id,
        ),
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    monkeypatch.setattr(
        migration_process.os,
        "killpg",
        lambda group_id, signal_number: signals.append((group_id, signal_number)),
    )

    with pytest.raises(migration_process._WorkerCleanupFailure):
        migration_process._signal_worker_session_groups(
            worker,
            (migration_process._WorkerSessionMember(202, process_group_id),),
            signal.SIGKILL,
        )

    assert signals == []


def test_linux_worker_cleanup_signals_exact_pidfd_handles_without_numeric_group_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = cast(Any, SimpleNamespace(session_id=101, process_group_id=101))
    members = (
        migration_process._WorkerSessionMember(202, 202),
        migration_process._WorkerSessionMember(203, 202),
    )
    opened: list[int] = []
    sent: list[tuple[int, int, object, int]] = []
    closed: list[int] = []
    group_signals: list[tuple[int, int]] = []

    def open_pid_handle(process_id: int) -> int:
        opened.append(process_id)
        return process_id + 1_000

    monkeypatch.setattr(migration_process.sys, "platform", "linux")
    monkeypatch.setattr(migration_process.os, "pidfd_open", open_pid_handle, raising=False)
    monkeypatch.setattr(
        migration_process.signal,
        "pidfd_send_signal",
        lambda handle, signal_number, info, flags: sent.append(
            (handle, signal_number, info, flags)
        ),
        raising=False,
    )
    monkeypatch.setattr(migration_process.os, "getsid", lambda _process_id: 101)
    monkeypatch.setattr(migration_process.os, "getpgid", lambda _process_id: 202)
    monkeypatch.setattr(migration_process.os, "close", closed.append)
    monkeypatch.setattr(
        migration_process.os,
        "killpg",
        lambda group_id, signal_number: group_signals.append((group_id, signal_number)),
    )

    migration_process._signal_worker_session_groups(worker, members, signal.SIGKILL)

    assert opened == [202, 203]
    assert sent == [
        (1_202, signal.SIGKILL, None, 0),
        (1_203, signal.SIGKILL, None, 0),
    ]
    assert closed == [1_202, 1_203]
    assert group_signals == []


def test_linux_pidfd_identity_change_closes_handle_without_signalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = cast(Any, SimpleNamespace(session_id=101, process_group_id=101))
    members = (migration_process._WorkerSessionMember(202, 202),)
    session_ids = iter((101, 303))
    sent: list[int] = []
    closed: list[int] = []
    monkeypatch.setattr(migration_process.sys, "platform", "linux")
    monkeypatch.setattr(
        migration_process.os,
        "pidfd_open",
        lambda _process_id: 1_202,
        raising=False,
    )
    monkeypatch.setattr(
        migration_process.signal,
        "pidfd_send_signal",
        lambda *_args: sent.append(1),
        raising=False,
    )
    monkeypatch.setattr(migration_process.os, "getsid", lambda _process_id: next(session_ids))
    monkeypatch.setattr(migration_process.os, "getpgid", lambda _process_id: 202)
    monkeypatch.setattr(migration_process.os, "close", closed.append)

    with pytest.raises(migration_process._WorkerCleanupFailure):
        migration_process._signal_worker_session_groups(worker, members, signal.SIGTERM)

    assert sent == []
    assert closed == [1_202]


def test_worker_identity_query_permission_failure_is_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = tmp_path / "identity-denial-worker-member.txt"
    monkeypatch.setattr(
        migration_process,
        "_worker_command",
        lambda: _setpgrp_worker_command(
            marker_path,
            '{"ok":true,"applied":[]}',
            escape_group=False,
        ),
    )
    real_getsid = os.getsid
    deny_identity = True
    child_pid = 0

    def deny_member_identity_query(process_id: int) -> int:
        if deny_identity and marker_path.exists():
            member_pid = int(marker_path.read_text().split()[0])
            if process_id == member_pid:
                raise PermissionError("injected identity-query denial")
        return real_getsid(process_id)

    monkeypatch.setattr(migration_process.os, "getsid", deny_member_identity_query)
    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / "identity-cleanup.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )
        child_pid = int(marker_path.read_text().split()[0])

        assert excinfo.value.stop_reason == "MIGRATION_WORKER_CLEANUP_FAILED"
        deny_identity = False
        assert migration_process._retry_quarantined_workers()
        _assert_processes_gone([child_pid])
    finally:
        deny_identity = False
        _kill_processes([child_pid] if child_pid else [])


def test_worker_normal_success_does_not_wait_for_shutdown_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = [sys.executable, "-c", 'print(\'{"ok":true,"applied":[]}\')']
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    started = time.monotonic()

    result = migration_process.run_migration_in_subprocess(
        tmp_path / "ordinary-success.db",
        DatabaseKind.INVENTORY,
        settings=None,
    )

    assert result == []
    assert time.monotonic() - started < 1.0


def test_worker_timeout_reaps_the_spawned_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "timeout-worker-tree.pids"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            f"pathlib.Path({str(pid_path)!r}).write_text(f'{{os.getpid()}} {{child.pid}}'); "
            "time.sleep(60)"
        ),
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    monkeypatch.setattr(migration_process, "MIGRATION_WORKER_TIMEOUT_SECONDS", 0.1)

    process_ids: list[int] = []
    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / "timeout.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )

        assert excinfo.value.stop_reason == "MIGRATION_WORKER_TIMEOUT"
        process_ids = [int(value) for value in pid_path.read_text().split()]
        _assert_processes_gone(process_ids)
    finally:
        _kill_processes(process_ids)


def test_worker_protocol_is_bounded_and_does_not_surface_raw_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-oversized-worker-output-secret"
    pid_path = tmp_path / "oversized-worker-tree.pids"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            f"pathlib.Path({str(pid_path)!r}).write_text(f'{{os.getpid()}} {{child.pid}}'); "
            f"os.write(1, ({secret!r}.encode() * 2000)); "
            "time.sleep(60)"
        ),
    ]
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    monkeypatch.setattr(migration_process, "MIGRATION_WORKER_TIMEOUT_SECONDS", 0.5)

    process_ids: list[int] = []
    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migration_process.run_migration_in_subprocess(
                tmp_path / "oversized.db",
                DatabaseKind.INVENTORY,
                settings=None,
            )

        assert excinfo.value.stop_reason == "MIGRATION_WORKER_PROTOCOL_INVALID"
        assert secret not in str(excinfo.value)
        process_ids = [int(value) for value in pid_path.read_text().split()]
        _assert_processes_gone(process_ids)
    finally:
        _kill_processes(process_ids)


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
