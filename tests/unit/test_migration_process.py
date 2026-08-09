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
from contextlib import suppress
from pathlib import Path

import pytest

import invoice_agents.db.core as core_module
import invoice_agents.db.migration_process as migration_process
import invoice_agents.db.migration_worker as migration_worker
from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind, migrate_database
from invoice_agents.errors import DatabaseVerificationError, ErrorCategory


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


def _setpgrp_worker_command(
    marker_path: Path,
    response: str,
    *,
    ignore_term: bool = False,
) -> list[str]:
    child_script = (
        "import os,pathlib,signal,time;"
        "os.setpgrp();"
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
            "'message':'schema audit failed','stop_reason':'MIGRATION_HISTORY_INVALID',"
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
            '{"ok":false,"error":{"category":"DATABASE","message":"schema audit failed",'
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
    process_ids: list[int] = []

    try:
        if expected_stop_reason is None:
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
        _assert_processes_gone(process_ids)
    finally:
        _kill_processes(process_ids)


@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize(
    "worker_response",
    [
        '{"ok":true,"applied":[]}',
        (
            '{"ok":false,"error":{"category":"DATABASE","message":"schema audit failed",'
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
        ),
    )
    monkeypatch.setattr(migration_process, "_WORKER_SHUTDOWN_SECONDS", 0.05)
    real_killpg = os.killpg
    deny_signals = True
    signalled_groups: list[int] = []
    child_pid = 0
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )

    def controlled_killpg(process_group_id: int, signal_number: int) -> None:
        signalled_groups.append(process_group_id)
        if deny_signals and signal_number == denied_signal:
            raise PermissionError("injected worker cleanup denial")
        real_killpg(process_group_id, signal_number)

    monkeypatch.setattr(migration_process.os, "killpg", controlled_killpg)
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
        assert signalled_groups and set(signalled_groups) == {child_group}

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
    ["process-failure", "malformed-output", "stderr-output", "nonzero-return"],
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
    inject_fault = True

    def controlled_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if inject_fault and args and args[0] == ["/bin/ps", "-axo", "pid=,pgid=,stat="]:
            if enumeration_fault == "process-failure":
                raise OSError("injected ps failure")
            if enumeration_fault == "stderr-output":
                return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="ps warning\n")
            if enumeration_fault == "nonzero-return":
                return subprocess.CompletedProcess(args[0], 17, stdout="", stderr="")
            return subprocess.CompletedProcess(args[0], 0, stdout="malformed\n", stderr="")
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


def test_worker_identity_query_permission_failure_is_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = [sys.executable, "-c", 'print(\'{"ok":true,"applied":[]}\')']
    monkeypatch.setattr(migration_process, "_worker_command", lambda: command)
    real_getsid = os.getsid
    deny_identity = True

    def deny_first_identity_query(process_id: int) -> int:
        nonlocal deny_identity
        if deny_identity:
            deny_identity = False
            raise PermissionError("injected identity-query denial")
        return real_getsid(process_id)

    monkeypatch.setattr(migration_process.os, "getsid", deny_first_identity_query)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migration_process.run_migration_in_subprocess(
            tmp_path / "identity-cleanup.db",
            DatabaseKind.INVENTORY,
            settings=None,
        )

    assert excinfo.value.stop_reason == "MIGRATION_WORKER_CLEANUP_FAILED"


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
