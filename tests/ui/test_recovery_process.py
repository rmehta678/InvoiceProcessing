"""Fail-closed process and ownership contracts for startup recovery."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from invoice_agents import recovery_process, recovery_worker
from invoice_agents.config import Settings
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.isolated_process import (
    IsolatedProcessCleanupError,
    IsolatedProcessResult,
    ProcessCancellation,
)
from invoice_agents.ui import recovery as recovery_module
from invoice_agents.ui.recovery import RecoveryCoordinator

_MINIMAL_STORE_FIELDS = {
    "due_date_tolerance_days",
    "inventory_db",
    "review_threshold_amount",
    "review_threshold_currency",
    "review_threshold_effective_date",
    "sqlite_journal_mode",
    "workflow_db",
}


def _scan_at() -> datetime:
    return datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _run_recovery_worker_main(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    recover: Any,
) -> bytes:
    output = io.BytesIO()
    monkeypatch.setattr(recovery_worker.os, "open", lambda *_args: 41)
    monkeypatch.setattr(recovery_worker.os, "dup2", lambda *_args: None)
    monkeypatch.setattr(recovery_worker.os, "close", lambda *_args: None)
    monkeypatch.setattr(
        recovery_worker.sys,
        "stdin",
        type("Input", (), {"buffer": io.BytesIO(b"request")})(),
    )
    monkeypatch.setattr(
        recovery_worker.sys,
        "stdout",
        type("Output", (), {"buffer": output})(),
    )
    monkeypatch.setattr(
        recovery_worker,
        "decode_recovery_request",
        lambda _request: (settings, _scan_at()),
    )
    monkeypatch.setattr(recovery_worker, "_recover", recover)

    recovery_worker.main()

    return output.getvalue()


def _pid_is_absent(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _bounded_marker_pids(marker: Path) -> tuple[int, int]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            fields = marker.read_text(encoding="ascii").split()
        except (OSError, UnicodeError):
            fields = []
        if len(fields) == 2 and all(
            value.isascii() and value.isdigit() and int(value) > 0 and str(int(value)) == value
            for value in fields
        ):
            return int(fields[0]), int(fields[1])
        time.sleep(0.01)
    raise AssertionError("recovery worker did not publish its bounded PID receipt")


def _assert_processes_absent(process_ids: tuple[int, int]) -> None:
    for process_id in process_ids:
        deadline = time.monotonic() + 2.0
        while not _pid_is_absent(process_id):
            assert time.monotonic() < deadline, f"recovery process {process_id} survived"
            time.sleep(0.01)


def _hung_recovery_command(marker: Path) -> list[str]:
    descendant = "import time; time.sleep(30)"
    worker = "\n".join(
        (
            "import os, subprocess, sys, time",
            "from pathlib import Path",
            f"child = subprocess.Popen([sys.executable, '-I', '-c', {descendant!r}])",
            f"Path({os.fspath(marker)!r}).write_text("
            "f'{os.getpid()} {child.pid}', encoding='ascii')",
            "time.sleep(30)",
        )
    )
    return [os.fspath(Path(sys.executable).resolve(strict=True)), "-I", "-c", worker]


def test_recovery_request_contains_only_strict_store_authority(
    tmp_path: Path,
    settings: Settings,
) -> None:
    provider_canary = "recovery-provider-canary"
    session_canary = "recovery-session-canary-0000000000000000"
    selected = settings.model_copy(
        update={
            "xai_api_key": SecretStr(provider_canary),
            "inventory_db": tmp_path / "inventory.db",
            "workflow_db": tmp_path / "workflow.db",
            "source_archive_dir": tmp_path / "not-recovery-authority",
            "ui_session_secret": SecretStr(session_canary),
        }
    )

    encoded = recovery_process._encode_request(settings=selected, scan_at=_scan_at())
    payload = json.loads(encoded)

    assert set(payload) == {"protocol_version", "scan_at", "store"}
    assert set(payload["store"]) == _MINIMAL_STORE_FIELDS
    assert payload["store"]["inventory_db"] == os.fspath(selected.inventory_db.resolve())
    assert payload["store"]["workflow_db"] == os.fspath(selected.workflow_db.resolve())
    assert provider_canary.encode() not in encoded
    assert session_canary.encode() not in encoded
    assert os.fspath(selected.source_archive_dir).encode() not in encoded

    decoded_settings, decoded_scan_at = recovery_process.decode_recovery_request(encoded)
    assert decoded_settings.inventory_db == selected.inventory_db.resolve()
    assert decoded_settings.workflow_db == selected.workflow_db.resolve()
    assert decoded_settings.xai_api_key is None
    assert decoded_settings.ui_session_secret is None
    assert decoded_scan_at == _scan_at()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: {**payload, "unexpected": True},
        lambda payload: {**payload, "protocol_version": True},
        lambda payload: {**payload, "scan_at": "2026-08-10T12:00:00"},
        lambda payload: {
            **payload,
            "store": {**payload["store"], "workflow_db": "relative.db"},
        },
        lambda payload: {
            **payload,
            "store": {**payload["store"], "xai_api_key": "must-not-cross"},
        },
    ],
)
def test_recovery_request_rejects_noncanonical_or_excess_authority(
    settings: Settings,
    mutation: Any,
) -> None:
    payload = json.loads(recovery_process._encode_request(settings=settings, scan_at=_scan_at()))
    malformed = json.dumps(
        mutation(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    with pytest.raises(ValueError):
        recovery_process.decode_recovery_request(malformed)


def test_recovery_request_rejects_duplicate_protocol_keys(settings: Settings) -> None:
    encoded = recovery_process._encode_request(settings=settings, scan_at=_scan_at())
    malformed = encoded.replace(
        b'{"protocol_version":1,',
        b'{"protocol_version":1,"protocol_version":1,',
    )

    with pytest.raises(ValueError, match="duplicate recovery protocol key"):
        recovery_process.decode_recovery_request(malformed)


def test_recovery_process_uses_only_the_public_isolated_controller(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = ProcessCancellation()
    calls: list[dict[str, object]] = []

    def controlled(**kwargs: object) -> IsolatedProcessResult:
        calls.append(kwargs)
        return IsolatedProcessResult(b'{"ok":true}', None)

    monkeypatch.setattr(recovery_process, "run_isolated_process", controlled)

    outcome = recovery_process.run_recovery_process(
        settings=settings,
        scan_at=_scan_at(),
        timeout_seconds=1.0,
        cancel_requested=cancellation,
    )

    assert outcome == recovery_process.RecoveryProcessOutcome(True, None, None)
    assert len(calls) == 1
    call = calls[0]
    assert set(call) == {
        "cancel_requested",
        "command",
        "env",
        "max_response_bytes",
        "request",
        "timeout_seconds",
    }
    assert call["cancel_requested"] is cancellation
    command = call["command"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1] == "-I"
    assert Path(command[2]).is_absolute()
    assert Path(command[2]).name == "recovery_worker.py"


def test_recovery_worker_command_preserves_virtual_environment_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable).resolve(strict=True))
    monkeypatch.setattr(recovery_process.sys, "executable", os.fspath(launcher))

    command = recovery_process._recovery_worker_command()

    assert command[0] == os.fspath(launcher)
    assert command[1] == "-I"
    assert Path(command[2]).resolve(strict=True) == Path(recovery_worker.__file__).resolve(
        strict=True
    )


def test_recovery_worker_imports_only_its_exact_colocated_package(
    tmp_path: Path,
) -> None:
    """Isolation must not select an unrelated editable checkout of this package."""

    package = tmp_path / "exact-source" / "invoice_agents"
    database_package = package / "db"
    database_package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (database_package / "__init__.py").write_text("", encoding="utf-8")
    (package / "config.py").write_text(
        "class Settings:\n    pass\n",
        encoding="utf-8",
    )
    (package / "errors.py").write_text(
        """from enum import StrEnum

class ErrorCategory(StrEnum):
    ORCHESTRATION = 'ORCHESTRATION'

class InvoiceAgentsError(Exception):
    def __init__(self, category=ErrorCategory.ORCHESTRATION, stop_reason=None):
        self.category = category
        self.stop_reason = stop_reason
""",
        encoding="utf-8",
    )
    (database_package / "store.py").write_text(
        """class WorkflowStore:
    def __init__(self, settings):
        pass

    def recover_expired_executions(self, *, now):
        return []

    def unrecovered_execution_case_ids(self, *, checked_at):
        return []
""",
        encoding="utf-8",
    )
    (package / "recovery_process.py").write_text(
        """import json
from datetime import UTC, datetime
from invoice_agents.config import Settings

RECOVERY_MAX_MESSAGE_BYTES = 65536

def decode_recovery_request(request):
    if request != b'exact-request':
        raise ValueError('wrong request')
    return Settings(), datetime(2026, 8, 11, tzinfo=UTC)

def encode_recovery_response(response):
    return json.dumps(response, separators=(',', ':'), sort_keys=True).encode('ascii')

def is_safe_recovery_stop_reason(value):
    return isinstance(value, str) and value == 'EXECUTION_RECOVERY_FAILED'
""",
        encoding="utf-8",
    )
    worker = package / "recovery_worker.py"
    shutil.copyfile(Path(recovery_worker.__file__).resolve(strict=True), worker)

    completed = subprocess.run(
        [
            os.fspath(Path(sys.executable).resolve(strict=True)),
            "-I",
            "-S",
            os.fspath(worker.resolve(strict=True)),
        ],
        input=b"exact-request",
        capture_output=True,
        env=recovery_process.sanitized_worker_environment(),
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == b'{"ok":true}'
    assert completed.stderr == b""


@pytest.mark.parametrize(
    ("result", "expected_code"),
    [
        (IsolatedProcessResult(None, "cancelled"), "EXECUTION_RECOVERY_WORKER_CANCELLED"),
        (IsolatedProcessResult(None, "timeout"), "EXECUTION_RECOVERY_WORKER_TIMED_OUT"),
        (IsolatedProcessResult(None, "start"), "EXECUTION_RECOVERY_WORKER_CRASHED"),
        (IsolatedProcessResult(None, "crash"), "EXECUTION_RECOVERY_WORKER_CRASHED"),
        (IsolatedProcessResult(None, "protocol"), "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID"),
        (IsolatedProcessResult(None, None), "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID"),
    ],
)
def test_recovery_process_has_an_explicit_supervisor_failure_taxonomy(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    result: IsolatedProcessResult,
    expected_code: str,
) -> None:
    monkeypatch.setattr(recovery_process, "run_isolated_process", lambda **_kwargs: result)

    outcome = recovery_process.run_recovery_process(
        settings=settings,
        scan_at=_scan_at(),
        timeout_seconds=1.0,
    )

    assert outcome == recovery_process.RecoveryProcessOutcome(
        False,
        ErrorCategory.ORCHESTRATION,
        expected_code,
    )


def test_recovery_process_exposes_unproven_cleanup_as_failure(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cleanup_failed(**_kwargs: object) -> IsolatedProcessResult:
        raise IsolatedProcessCleanupError

    monkeypatch.setattr(recovery_process, "run_isolated_process", cleanup_failed)

    outcome = recovery_process.run_recovery_process(
        settings=settings,
        scan_at=_scan_at(),
        timeout_seconds=1.0,
    )

    assert outcome == recovery_process.RecoveryProcessOutcome(
        False,
        ErrorCategory.ORCHESTRATION,
        "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
    )


@pytest.mark.parametrize(
    ("acknowledged", "error_category", "stop_reason"),
    [
        (True, ErrorCategory.ORCHESTRATION, "EXECUTION_RECOVERY_FAILED"),
        (False, None, "EXECUTION_RECOVERY_FAILED"),
        (False, ErrorCategory.ORCHESTRATION, None),
        (False, "ORCHESTRATION", "EXECUTION_RECOVERY_FAILED"),
        (False, ErrorCategory.ORCHESTRATION, "not_safe"),
        (False, ErrorCategory.ORCHESTRATION, "A" * 129),
        (False, ErrorCategory.ORCHESTRATION, "ÉXECUTION_RECOVERY_FAILED"),
        (1, None, None),
    ],
)
def test_recovery_outcome_rejects_contradictory_or_unsafe_state(
    acknowledged: object,
    error_category: object,
    stop_reason: object,
) -> None:
    with pytest.raises(ValueError):
        recovery_process.RecoveryProcessOutcome(  # type: ignore[arg-type]
            acknowledged,
            error_category,
            stop_reason,
        )


@pytest.mark.parametrize(
    ("response", "acknowledged", "error_category", "stop_reason"),
    [
        (b'{"ok":true}', True, None, None),
        (
            b'{"error_category":"ORCHESTRATION","ok":false,'
            b'"stop_reason":"EXECUTION_RECOVERY_FAILED"}',
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_FAILED",
        ),
        (
            b'{"error_category":"DATABASE","ok":false,"stop_reason":"DATABASE_MISSING"}',
            False,
            ErrorCategory.DATABASE,
            "DATABASE_MISSING",
        ),
        (
            b'{"error_category":"PROVIDER","ok":false,"stop_reason":"FUTURE_VALID_FAILURE_42"}',
            False,
            ErrorCategory.PROVIDER,
            "FUTURE_VALID_FAILURE_42",
        ),
    ],
)
def test_recovery_process_accepts_only_canonical_worker_outcomes(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    response: bytes,
    acknowledged: bool,
    error_category: ErrorCategory | None,
    stop_reason: str | None,
) -> None:
    monkeypatch.setattr(
        recovery_process,
        "run_isolated_process",
        lambda **_kwargs: IsolatedProcessResult(response, None),
    )

    assert recovery_process.run_recovery_process(
        settings=settings,
        scan_at=_scan_at(),
        timeout_seconds=1.0,
    ) == recovery_process.RecoveryProcessOutcome(
        acknowledged,
        error_category,
        stop_reason,
    )


def test_recovery_process_preserves_valid_worker_error_category_and_stop_reason(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery_process,
        "run_isolated_process",
        lambda **_kwargs: IsolatedProcessResult(
            b'{"error_category":"DATABASE","ok":false,"stop_reason":"PERSISTED_RESULT_INVALID"}',
            None,
        ),
    )

    outcome = recovery_process.run_recovery_process(
        settings=settings,
        scan_at=_scan_at(),
        timeout_seconds=1.0,
    )

    assert outcome == recovery_process.RecoveryProcessOutcome(
        False,
        ErrorCategory.DATABASE,
        "PERSISTED_RESULT_INVALID",
    )


@pytest.mark.parametrize(
    "reserved_stop_reason",
    [
        "EXECUTION_RECOVERY_WORKER_CANCELLED",
        "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
        "EXECUTION_RECOVERY_WORKER_CRASHED",
        "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID",
        "EXECUTION_RECOVERY_WORKER_TIMED_OUT",
    ],
)
@pytest.mark.parametrize("error_category", ["ORCHESTRATION", "DATABASE"])
def test_recovery_process_rejects_child_forged_controller_outcomes(
    reserved_stop_reason: str,
    error_category: str,
) -> None:
    response = json.dumps(
        {
            "error_category": error_category,
            "ok": False,
            "stop_reason": reserved_stop_reason,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    with pytest.raises(ValueError, match="controller-owned"):
        recovery_process._decode_response(response)


@pytest.mark.parametrize(
    ("category", "stop_reason"),
    [
        (ErrorCategory.DATABASE, "PERSISTED_RESULT_INVALID"),
        (ErrorCategory.ORCHESTRATION, "EXECUTION_RECOVERY_RACE"),
        (ErrorCategory.PROVIDER, "FUTURE_VALID_FAILURE_42"),
    ],
)
def test_recovery_ui_recreates_exact_validated_domain_failure(
    category: ErrorCategory,
    stop_reason: str,
) -> None:
    outcome = recovery_process.RecoveryProcessOutcome(
        False,
        category,
        stop_reason,
    )

    error = recovery_module._outcome_error(outcome)

    assert error.category is category
    assert error.stop_reason == stop_reason
    assert error.message == "the isolated execution recovery scan failed"
    assert error.details is None


@pytest.mark.parametrize(
    "response",
    [
        b"",
        b"not-json",
        b'{"ok":true,"unexpected":null}',
        b'{"error_category":"ORCHESTRATION","ok":true,"stop_reason":"EXECUTION_RECOVERY_FAILED"}',
        b'{"error_category":"NOT_A_CATEGORY","ok":false,"stop_reason":"EXECUTION_RECOVERY_FAILED"}',
        b'{"error_category":"DATABASE","ok":false,"stop_reason":"not_safe"}',
        b'{"error_category":"DATABASE","ok":false}',
        b'{"ok":false,"ok":true}',
        b'{ "ok":true}',
    ],
)
def test_recovery_process_rejects_malformed_or_noncanonical_responses(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    response: bytes,
) -> None:
    monkeypatch.setattr(
        recovery_process,
        "run_isolated_process",
        lambda **_kwargs: IsolatedProcessResult(response, None),
    )

    outcome = recovery_process.run_recovery_process(
        settings=settings,
        scan_at=_scan_at(),
        timeout_seconds=1.0,
    )

    assert outcome == recovery_process.RecoveryProcessOutcome(
        False,
        ErrorCategory.ORCHESTRATION,
        "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID",
    )


def test_recovery_worker_fails_closed_when_eligible_claims_remain(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SilentStore:
        def __init__(self, _settings: Settings) -> None:
            pass

        def recover_expired_executions(self, *, now: datetime) -> list[str]:
            assert now == _scan_at()
            return []

        def unrecovered_execution_case_ids(self, *, checked_at: datetime) -> list[str]:
            assert checked_at == _scan_at()
            return ["case_unrecovered"]

    monkeypatch.setattr(recovery_worker, "WorkflowStore", SilentStore)

    assert recovery_worker._recover(settings, _scan_at()) is False


@pytest.mark.parametrize(
    ("category", "stop_reason", "expected"),
    [
        (
            ErrorCategory.DATABASE,
            "EXECUTION_AUTHORITY_CORRUPT",
            b'{"error_category":"DATABASE","ok":false,"stop_reason":"EXECUTION_AUTHORITY_CORRUPT"}',
        ),
        (
            ErrorCategory.DATABASE,
            "PERSISTED_RESULT_INVALID",
            b'{"error_category":"DATABASE","ok":false,"stop_reason":"PERSISTED_RESULT_INVALID"}',
        ),
        (
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_RACE",
            b'{"error_category":"ORCHESTRATION","ok":false,'
            b'"stop_reason":"EXECUTION_RECOVERY_RACE"}',
        ),
    ],
)
def test_recovery_worker_preserves_only_exact_audit_safe_error_metadata(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    category: ErrorCategory,
    stop_reason: str,
    expected: bytes,
) -> None:
    def fail_with_exact_cause(*_args: object) -> bool:
        raise InvoiceAgentsError(
            category,
            "sensitive recovery failure canary",
            stop_reason=stop_reason,
            details={"path": "/sensitive/recovery/path"},
        )

    encoded = _run_recovery_worker_main(monkeypatch, settings, fail_with_exact_cause)

    assert encoded == expected
    assert b"canary" not in encoded
    assert b"/sensitive/recovery/path" not in encoded


@pytest.mark.parametrize(
    "error",
    [
        InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "missing reason",
            stop_reason=None,
        ),
        InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "unsafe reason",
            stop_reason="unsafe-reason",
        ),
        InvoiceAgentsError(  # type: ignore[arg-type]
            "DATABASE",
            "non-enum category",
            stop_reason="PERSISTED_RESULT_INVALID",
        ),
    ],
)
def test_recovery_worker_marks_invalid_error_metadata_as_protocol_invalid(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    error: InvoiceAgentsError,
) -> None:
    def fail_with_invalid_metadata(*_args: object) -> bool:
        raise error

    encoded = _run_recovery_worker_main(monkeypatch, settings, fail_with_invalid_metadata)

    assert encoded == (
        b'{"error_category":"ORCHESTRATION","ok":false,'
        b'"stop_reason":"EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID"}'
    )


def test_recovery_worker_does_not_retry_response_encoding(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = io.BytesIO()
    encode_calls = 0

    def fail_encoding(_response: dict[str, object]) -> bytes:
        nonlocal encode_calls
        encode_calls += 1
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(recovery_worker.os, "open", lambda *_args: 41)
    monkeypatch.setattr(recovery_worker.os, "dup2", lambda *_args: None)
    monkeypatch.setattr(recovery_worker.os, "close", lambda *_args: None)
    monkeypatch.setattr(
        recovery_worker.sys,
        "stdin",
        type("Input", (), {"buffer": io.BytesIO(b"request")})(),
    )
    monkeypatch.setattr(
        recovery_worker.sys,
        "stdout",
        type("Output", (), {"buffer": output})(),
    )
    monkeypatch.setattr(
        recovery_worker,
        "decode_recovery_request",
        lambda _request: (settings, _scan_at()),
    )
    monkeypatch.setattr(recovery_worker, "_recover", lambda *_args: True)
    monkeypatch.setattr(recovery_worker, "encode_recovery_response", fail_encoding)

    with pytest.raises(RuntimeError, match="encoding failed"):
        recovery_worker.main()

    assert encode_calls == 1
    assert output.getvalue() == b""


def test_recovery_process_timeout_returns_only_after_descendant_cleanup(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "timeout.pids"
    monkeypatch.setattr(
        recovery_process,
        "_recovery_worker_command",
        lambda: _hung_recovery_command(marker),
    )

    outcome = recovery_process.run_recovery_process(
        settings=settings,
        scan_at=_scan_at(),
        timeout_seconds=0.1,
    )

    assert outcome == recovery_process.RecoveryProcessOutcome(
        False,
        ErrorCategory.ORCHESTRATION,
        "EXECUTION_RECOVERY_WORKER_TIMED_OUT",
    )
    _assert_processes_absent(_bounded_marker_pids(marker))


@pytest.mark.asyncio
async def test_concurrent_coordinators_admit_each_scan_only_after_prior_cleanup(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    calls_lock = threading.Lock()
    calls = 0
    active = 0
    maximum_active = 0

    def supervised_scan(**_kwargs: object) -> recovery_process.RecoveryProcessOutcome:
        nonlocal active, calls, maximum_active
        with calls_lock:
            calls += 1
            call = calls
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if call == 1:
                first_entered.set()
                assert release_first.wait(timeout=2)
            return recovery_process.RecoveryProcessOutcome(True, None, None)
        finally:
            with calls_lock:
                active -= 1

    monkeypatch.setattr(recovery_module, "run_recovery_process", supervised_scan)
    first = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    second = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    first_start = asyncio.create_task(first.start())
    second_start: asyncio.Task[None] | None = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(first_entered.wait), timeout=1)
        second_start = asyncio.create_task(second.start())
        await asyncio.sleep(0.02)
        assert calls == 1
        assert not second_start.done()

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first_start, second_start), timeout=1)
        assert calls == 2
        assert maximum_active == 1
    finally:
        release_first.set()
        for task in (first_start, second_start):
            if task is not None and not task.done():
                with suppress(BaseException):
                    await task
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_start_cancellation_keeps_reservation_until_cleanup_proof(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    cancel_seen = threading.Event()
    cleanup_proven = threading.Event()
    calls = 0

    def controlled_scan(
        *,
        cancel_requested: ProcessCancellation,
        **_kwargs: object,
    ) -> recovery_process.RecoveryProcessOutcome:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            deadline = time.monotonic() + 2
            while not cancel_requested.is_set():
                assert time.monotonic() < deadline
                time.sleep(0.005)
            cancel_seen.set()
            assert cleanup_proven.wait(timeout=2)
            return recovery_process.RecoveryProcessOutcome(
                False,
                ErrorCategory.ORCHESTRATION,
                "EXECUTION_RECOVERY_WORKER_CANCELLED",
            )
        return recovery_process.RecoveryProcessOutcome(True, None, None)

    monkeypatch.setattr(recovery_module, "run_recovery_process", controlled_scan)
    interrupted = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    blocked_waiter = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    successor = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    interrupted_start = asyncio.create_task(interrupted.start())
    blocked_start: asyncio.Task[None] | None = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1)
        interrupted_start.cancel()
        assert await asyncio.wait_for(asyncio.to_thread(cancel_seen.wait), timeout=1)
        await asyncio.sleep(0)
        assert not interrupted_start.done()

        blocked_start = asyncio.create_task(blocked_waiter.start())
        await asyncio.sleep(0.02)
        assert calls == 1
        assert not blocked_start.done()

        cleanup_proven.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(interrupted_start, timeout=1)
        with pytest.raises(InvoiceAgentsError) as blocked_error:
            await asyncio.wait_for(blocked_start, timeout=1)
        assert blocked_error.value.stop_reason == "EXECUTION_RECOVERY_WORKER_CANCELLED"

        await asyncio.wait_for(successor.start(), timeout=1)
        assert calls == 2
    finally:
        cleanup_proven.set()
        if blocked_start is not None and not blocked_start.done():
            with suppress(BaseException):
                await blocked_start
        if successor.state not in {"created", "stopped"}:
            await successor.close()


@pytest.mark.asyncio
async def test_cleanup_ambiguity_poison_blocks_all_later_admission(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery_module, "_RECOVERY_OWNERSHIP_POISONED", False)
    monkeypatch.setattr(recovery_module, "_SCAN_RESERVATION", None)
    calls = 0

    def ambiguous_cleanup(**_kwargs: object) -> recovery_process.RecoveryProcessOutcome:
        nonlocal calls
        calls += 1
        return recovery_process.RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
        )

    monkeypatch.setattr(recovery_module, "run_recovery_process", ambiguous_cleanup)
    first = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    with pytest.raises(InvoiceAgentsError) as first_error:
        await first.start()
    assert first_error.value.stop_reason == "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED"
    assert first.health().ownership_poisoned is True

    successor = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    with pytest.raises(InvoiceAgentsError) as successor_error:
        await successor.start()
    assert successor_error.value.stop_reason == "EXECUTION_RECOVERY_OWNERSHIP_UNRESOLVED"
    assert calls == 1


@pytest.mark.asyncio
async def test_mismatched_cleanup_category_cannot_poison_controller_ownership(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery_module, "_RECOVERY_OWNERSHIP_POISONED", False)
    monkeypatch.setattr(recovery_module, "_SCAN_RESERVATION", None)

    def mismatched_cleanup(**_kwargs: object) -> recovery_process.RecoveryProcessOutcome:
        return recovery_process.RecoveryProcessOutcome(
            False,
            ErrorCategory.DATABASE,
            "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
        )

    monkeypatch.setattr(recovery_module, "run_recovery_process", mismatched_cleanup)
    coordinator = RecoveryCoordinator(settings, scan_interval_seconds=3_600)

    with pytest.raises(InvoiceAgentsError) as error:
        await coordinator.start()

    assert error.value.category is ErrorCategory.DATABASE
    assert error.value.stop_reason == "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED"
    assert coordinator.health().ownership_poisoned is False


@pytest.mark.asyncio
async def test_mismatched_cancelled_category_cannot_claim_shutdown_control(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery_module, "_RECOVERY_OWNERSHIP_POISONED", False)
    monkeypatch.setattr(recovery_module, "_SCAN_RESERVATION", None)

    def mismatched_cancel(**_kwargs: object) -> recovery_process.RecoveryProcessOutcome:
        return recovery_process.RecoveryProcessOutcome(
            False,
            ErrorCategory.DATABASE,
            "EXECUTION_RECOVERY_WORKER_CANCELLED",
        )

    monkeypatch.setattr(recovery_module, "run_recovery_process", mismatched_cancel)
    coordinator = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    coordinator._stop_requested.set()

    with pytest.raises(InvoiceAgentsError) as error:
        await coordinator._recover_once()

    assert error.value.category is ErrorCategory.DATABASE
    assert error.value.stop_reason == "EXECUTION_RECOVERY_WORKER_CANCELLED"


@pytest.mark.asyncio
async def test_mismatched_cleanup_category_cannot_claim_shared_cleanup_control(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    reservation = object()
    outcome = recovery_process.RecoveryProcessOutcome(
        False,
        ErrorCategory.DATABASE,
        "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
    )

    monkeypatch.setattr(
        coordinator,
        "_reserve_scan",
        lambda _scan_at: (reservation, False),
    )

    async def shared_scan(_reservation: object) -> recovery_process.RecoveryProcessOutcome:
        return outcome

    monkeypatch.setattr(coordinator, "_wait_for_shared_scan", shared_scan)

    with pytest.raises(InvoiceAgentsError) as error:
        await coordinator._recover_once()

    assert error.value.category is ErrorCategory.DATABASE
    assert error.value.stop_reason == "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED"


@pytest.mark.asyncio
async def test_mismatched_cancelled_category_cannot_replace_caller_cancellation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class JoinedThread:
        def join(self) -> None:
            return None

        def is_alive(self) -> bool:
            return False

    outcome: Future[recovery_process.RecoveryProcessOutcome] = Future()
    reservation = recovery_module._ScanReservation(
        signature=(settings.workflow_db.resolve(), settings.inventory_db.resolve()),
        scan_at=_scan_at(),
        cancellation=ProcessCancellation(),
        worker=outcome,
        thread=JoinedThread(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(recovery_module, "_RECOVERY_OWNERSHIP_POISONED", False)
    monkeypatch.setattr(recovery_module, "_SCAN_RESERVATION", reservation)
    coordinator = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    coordinator._set_active_reservation(reservation)
    waiting = asyncio.create_task(coordinator._wait_for_owned_scan(reservation))
    await asyncio.sleep(0)

    waiting.cancel()
    await asyncio.sleep(0)
    assert reservation.cancellation.is_set()
    outcome.set_result(
        recovery_process.RecoveryProcessOutcome(
            False,
            ErrorCategory.DATABASE,
            "EXECUTION_RECOVERY_WORKER_CANCELLED",
        )
    )

    with pytest.raises(InvoiceAgentsError) as error:
        await waiting

    assert error.value.category is ErrorCategory.DATABASE
    assert error.value.stop_reason == "EXECUTION_RECOVERY_WORKER_CANCELLED"


@pytest.mark.asyncio
async def test_invalid_controller_outcome_cannot_publish_recovery_success(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery_module, "_RECOVERY_OWNERSHIP_POISONED", False)
    monkeypatch.setattr(recovery_module, "_SCAN_RESERVATION", None)
    monkeypatch.setattr(
        recovery_module,
        "run_recovery_process",
        lambda **_kwargs: object(),
    )
    coordinator = RecoveryCoordinator(settings, scan_interval_seconds=3_600)

    with pytest.raises(TypeError, match="invalid outcome type"):
        await coordinator.start()

    assert coordinator.completed_scans == 0
    assert coordinator.health().ownership_poisoned is True


def test_thread_start_error_retains_possibly_started_reservation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StartControlError(BaseException):
        pass

    class StartsThenRaisesThread(threading.Thread):
        def start(self) -> None:
            super().start()
            raise StartControlError

    def cancelled_scan(
        *,
        cancel_requested: ProcessCancellation,
        **_kwargs: object,
    ) -> recovery_process.RecoveryProcessOutcome:
        assert cancel_requested.wait(timeout=1)
        return recovery_process.RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_CANCELLED",
        )

    monkeypatch.setattr(recovery_module, "_RECOVERY_OWNERSHIP_POISONED", False)
    monkeypatch.setattr(recovery_module, "_SCAN_RESERVATION", None)
    monkeypatch.setattr(recovery_module, "Thread", StartsThenRaisesThread)
    monkeypatch.setattr(recovery_module, "run_recovery_process", cancelled_scan)
    coordinator = RecoveryCoordinator(settings, scan_interval_seconds=3_600)

    with pytest.raises(StartControlError):
        coordinator._reserve_scan(_scan_at())

    reservation = recovery_module._SCAN_RESERVATION
    assert reservation is not None
    assert reservation.cancellation.is_set()
    assert recovery_module._RECOVERY_OWNERSHIP_POISONED is True
    assert reservation.thread is not None
    reservation.thread.join(timeout=1)
    assert not reservation.thread.is_alive()


def test_retirement_join_ambiguity_poison_retains_reservation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AmbiguousThread:
        def join(self) -> None:
            raise RuntimeError("join ownership is ambiguous")

        def is_alive(self) -> bool:
            return False

    outcome: Future[recovery_process.RecoveryProcessOutcome] = Future()
    outcome.set_result(recovery_process.RecoveryProcessOutcome(True, None, None))
    reservation = recovery_module._ScanReservation(
        signature=(settings.workflow_db.resolve(), settings.inventory_db.resolve()),
        scan_at=_scan_at(),
        cancellation=ProcessCancellation(),
        worker=outcome,
        thread=AmbiguousThread(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(recovery_module, "_RECOVERY_OWNERSHIP_POISONED", False)
    monkeypatch.setattr(recovery_module, "_SCAN_RESERVATION", reservation)

    with pytest.raises(InvoiceAgentsError) as error:
        RecoveryCoordinator._retire_completed_reservation(reservation)

    assert error.value.stop_reason == "EXECUTION_RECOVERY_OWNERSHIP_UNRESOLVED"
    assert recovery_module._RECOVERY_OWNERSHIP_POISONED is True
    assert recovery_module._SCAN_RESERVATION is reservation


@pytest.mark.asyncio
async def test_poisoned_ownership_cannot_increment_success_counter(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery_module, "_RECOVERY_OWNERSHIP_POISONED", True)
    coordinator = RecoveryCoordinator(settings, scan_interval_seconds=3_600)

    with pytest.raises(InvoiceAgentsError) as error:
        await coordinator._publish_successful_scan(_scan_at())

    assert error.value.stop_reason == "EXECUTION_RECOVERY_OWNERSHIP_UNRESOLVED"
    assert coordinator.completed_scans == 0


@pytest.mark.asyncio
async def test_late_worker_failure_precedes_shutdown_cancellation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_entered = threading.Event()
    cancel_seen = threading.Event()
    release_failure = threading.Event()
    calls = 0

    def late_failure(
        *,
        cancel_requested: ProcessCancellation,
        **_kwargs: object,
    ) -> recovery_process.RecoveryProcessOutcome:
        nonlocal calls
        calls += 1
        if calls == 1:
            return recovery_process.RecoveryProcessOutcome(True, None, None)
        runtime_entered.set()
        deadline = time.monotonic() + 2
        while not cancel_requested.is_set():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        cancel_seen.set()
        assert release_failure.wait(timeout=2)
        return recovery_process.RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_FAILED",
        )

    monkeypatch.setattr(recovery_module, "run_recovery_process", late_failure)
    coordinator = RecoveryCoordinator(settings, scan_interval_seconds=3_600)
    await coordinator.start()
    coordinator.request_scan()
    assert await asyncio.wait_for(asyncio.to_thread(runtime_entered.wait), timeout=1)
    close_task = asyncio.create_task(coordinator.close())
    try:
        assert await asyncio.wait_for(asyncio.to_thread(cancel_seen.wait), timeout=1)
        close_task.cancel()
        release_failure.set()

        with pytest.raises(InvoiceAgentsError) as error:
            await asyncio.wait_for(close_task, timeout=1)
        assert error.value.stop_reason == "EXECUTION_RECOVERY_FAILED"
        assert coordinator.state == "failed"
    finally:
        release_failure.set()
