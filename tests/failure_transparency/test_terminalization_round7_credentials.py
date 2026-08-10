"""Focused RED reproducers for the seventh Task 9 credential review."""

from __future__ import annotations

import asyncio
import errno
import os
import socket
import sys
import threading
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from invoice_agents import isolated_process, lifecycle_process, orchestration
from invoice_agents.config import Settings
from invoice_agents.db.store import ExecutionClaim


@pytest.fixture(autouse=True)
def _forbid_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every round-7 test fails immediately at an unstubbed provider seam."""

    def forbidden(_settings: Settings) -> object:
        raise AssertionError("round7 test reached an unstubbed provider boundary")

    monkeypatch.setattr(orchestration, "create_model_client", forbidden)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", "import os; os._exit(87)"],
    )


def _lifecycle_inputs(
    tmp_path: Path,
    credential: str,
) -> tuple[Settings, datetime, ExecutionClaim]:
    settings = Settings(
        xai_api_key=SecretStr(credential),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_round7_credential_cleanup",
        f"exec_{'7' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )
    return settings, started_at, claim


def _stable_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_mode, status.st_rdev


def _original_descriptor_is_gone(
    descriptor: int,
    original: os.stat_result,
) -> bool:
    try:
        current = os.fstat(descriptor)
    except OSError as exc:
        return exc.errno == errno.EBADF
    return _stable_identity(current) != _stable_identity(original)


def _recover_original_datagram(
    descriptor: int,
    original: os.stat_result,
) -> bytes:
    """Read only while the number still names the captured credential socket."""

    try:
        current = os.fstat(descriptor)
    except OSError:
        return b""
    if _stable_identity(current) != _stable_identity(original):
        return b""
    with socket.socket(fileno=os.dup(descriptor)) as probe:
        probe.settimeout(0.25)
        return probe.recv(lifecycle_process.LIFECYCLE_MAX_CREDENTIAL_BYTES + 1)


def _assert_no_external_secret_surface(
    *,
    canary: bytes,
    invocation_surfaces: list[bytes],
    retained_buffers: list[bytearray],
    spawned_processes: list[object],
    tmp_path: Path,
) -> None:
    assert invocation_surfaces
    assert all(canary not in surface for surface in invocation_surfaces)
    assert retained_buffers and all(not any(buffer) for buffer in retained_buffers)
    assert spawned_processes
    assert all(process.poll() is not None for process in spawned_processes)  # type: ignore[attr-defined]
    assert all(canary not in repr(process.args).encode("utf-8") for process in spawned_processes)  # type: ignore[attr-defined]
    assert all(process.stderr is None for process in spawned_processes)  # type: ignore[attr-defined]
    assert all(
        process.stdout is None or process.stdout.closed  # type: ignore[attr-defined]
        for process in spawned_processes
    )
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert canary not in artifact.read_bytes()


def test_round7_persistent_close_failures_retire_the_credential_socket_before_reporting_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two failed close primitives cannot leave a readable credential identity."""

    canary = b"round7-persistent-close-credential-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    stable_descriptors: list[int] = []
    socket_close_calls = 0
    real_os_close = os.close
    real_dup = os.dup
    real_socket_close = socket.close
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session

    def observed_run(**kwargs: Any) -> object:
        descriptor = kwargs["pass_fds"][0]
        target_descriptor.append(descriptor)
        original_identity.append(os.fstat(descriptor))
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def observed_dup(descriptor: int) -> int:
        duplicate = real_dup(descriptor)
        if target_descriptor and descriptor == target_descriptor[0]:
            stable_descriptors.append(duplicate)
        return duplicate

    def fail_target_os_close(descriptor: int) -> None:
        if target_descriptor and descriptor == target_descriptor[0]:
            raise OSError(errno.EIO, "round7 persistent os.close failure")
        real_os_close(descriptor)

    def fail_target_socket_close(descriptor: int) -> None:
        nonlocal socket_close_calls
        if target_descriptor and descriptor == target_descriptor[0]:
            socket_close_calls += 1
            raise OSError(errno.EIO, "round7 persistent socket.close failure")
        real_socket_close(descriptor)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process.os, "dup", observed_dup)
    monkeypatch.setattr(isolated_process.os, "close", fail_target_os_close)
    monkeypatch.setattr(isolated_process.socket, "close", fail_target_socket_close)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert target_descriptor and original_identity
    descriptor = target_descriptor[0]
    try:
        observed = (
            outcome.error_code,
            _original_descriptor_is_gone(descriptor, original_identity[0]),
            _recover_original_datagram(descriptor, original_identity[0]),
        )
        assert observed == (
            "LIFECYCLE_WORKER_CLEANUP_FAILED",
            True,
            b"",
        )
        assert socket_close_calls == 0
        assert stable_descriptors
        for stable_descriptor in stable_descriptors:
            with pytest.raises(OSError) as stable_error:
                os.fstat(stable_descriptor)
            assert stable_error.value.errno == errno.EBADF
        _assert_no_external_secret_surface(
            canary=canary,
            invocation_surfaces=invocation_surfaces,
            retained_buffers=retained_buffers,
            spawned_processes=spawned_processes,
            tmp_path=tmp_path,
        )
    finally:
        with suppress(OSError):
            real_os_close(descriptor)


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round7_process_control_after_real_reap_cannot_skip_credential_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """A cleanup interruption is secondary when credential cleanup already failed."""

    canary = b"round7-cleanup-keyboard-interrupt-credential-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    reaped_before_interrupt: list[bool] = []
    real_os_close = os.close
    real_socket_close = socket.close
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_stop = isolated_process._stop_worker

    def observed_run(**kwargs: Any) -> object:
        descriptor = kwargs["pass_fds"][0]
        target_descriptor.append(descriptor)
        original_identity.append(os.fstat(descriptor))
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def fail_target_os_close(descriptor: int) -> None:
        if target_descriptor and descriptor == target_descriptor[0]:
            raise OSError(errno.EIO, "round7 persistent os.close failure")
        real_os_close(descriptor)

    def fail_target_socket_close(descriptor: int) -> None:
        if target_descriptor and descriptor == target_descriptor[0]:
            raise OSError(errno.EIO, "round7 persistent socket.close failure")
        real_socket_close(descriptor)

    def reap_then_interrupt(worker: object) -> object:
        result = real_stop(worker)  # type: ignore[arg-type]
        assert result is None
        reaped_before_interrupt.append(worker.process.poll() is not None)  # type: ignore[attr-defined]
        raise control_type("round7 injected cleanup process control")

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_stop_worker", reap_then_interrupt)
    monkeypatch.setattr(isolated_process.os, "close", fail_target_os_close)
    monkeypatch.setattr(isolated_process.socket, "close", fail_target_socket_close)

    outcome: lifecycle_process.LifecycleProcessOutcome | None = None
    escaped: BaseException | None = None
    try:
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )
    except BaseException as exc:  # process control is part of the asserted boundary
        escaped = exc

    assert target_descriptor and original_identity
    descriptor = target_descriptor[0]
    try:
        observed = (
            escaped,
            None if outcome is None else outcome.error_code,
            reaped_before_interrupt,
            _original_descriptor_is_gone(descriptor, original_identity[0]),
            _recover_original_datagram(descriptor, original_identity[0]),
        )
        assert observed == (
            None,
            "LIFECYCLE_WORKER_CLEANUP_FAILED",
            [True],
            True,
            b"",
        )
        _assert_no_external_secret_surface(
            canary=canary,
            invocation_surfaces=invocation_surfaces,
            retained_buffers=retained_buffers,
            spawned_processes=spawned_processes,
            tmp_path=tmp_path,
        )
    finally:
        with suppress(OSError):
            real_os_close(descriptor)


def _worker_command(worker_mode: str, tmp_path: Path) -> list[str]:
    if worker_mode == "start":
        return [os.fspath(tmp_path / "missing-round7-worker")]
    tail = {
        "success": (
            "b=bytearray(os.read(d,16385)); os.close(d); "
            "b[:]=bytes(len(b)); sys.stdout.buffer.write(b'{\"ok\":true}')"
        ),
        "crash": "os._exit(17)",
        "timeout": "time.sleep(30)",
        "cancel": "time.sleep(30)",
    }[worker_mode]
    return [
        sys.executable,
        "-c",
        "import json,os,sys,time; "
        "p=json.loads(sys.stdin.buffer.read()); d=p['credential_fd']; " + tail,
    ]


def _assert_descriptor_absent(descriptor: int) -> None:
    with pytest.raises(OSError) as absent_error:
        os.fstat(descriptor)
    assert absent_error.value.errno == errno.EBADF


@pytest.mark.parametrize("worker_mode", ["success", "crash", "timeout", "cancel", "start"])
def test_round7_persistent_close_fault_precedes_every_terminal_path_without_a_secret_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_mode: str,
) -> None:
    """Every terminal path retires secret identities and retains an uncertain close."""

    canary = f"round7-{worker_mode}-persistent-close-canary".encode()
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    stable_descriptors: list[int] = []
    retained_buffers: list[bytearray] = []
    drained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    socket_close_calls = 0
    real_os_close = os.close
    real_socket_close = socket.close
    real_dup = os.dup
    real_recv_into = socket.socket.recv_into
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_uncertain = isolated_process._uncertain_worker_session

    def observed_run(**kwargs: Any) -> object:
        descriptor = kwargs["pass_fds"][0]
        target_descriptor.append(descriptor)
        original_identity.append(os.fstat(descriptor))
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_dup(descriptor: int) -> int:
        duplicate = real_dup(descriptor)
        if target_descriptor and descriptor == target_descriptor[0]:
            stable_descriptors.append(duplicate)
        return duplicate

    def observed_recv_into(
        transport: socket.socket,
        buffer: bytearray,
        size: int,
        flags: int,
    ) -> int:
        drained_buffers.append(buffer)
        return real_recv_into(transport, buffer, size, flags)

    def remember_process(process: object) -> None:
        if process not in spawned_processes:
            spawned_processes.append(process)

    def observed_capture(process: object) -> object:
        remember_process(process)
        return real_capture(process)  # type: ignore[arg-type]

    def observed_uncertain(process: object) -> object:
        remember_process(process)
        return real_uncertain(process)  # type: ignore[arg-type]

    def fail_target_os_close(descriptor: int) -> None:
        if target_descriptor and descriptor == target_descriptor[0]:
            raise OSError(errno.EIO, "round7 persistent terminal-path close failure")
        real_os_close(descriptor)

    def fail_target_socket_close(descriptor: int) -> None:
        nonlocal socket_close_calls
        if target_descriptor and descriptor == target_descriptor[0]:
            socket_close_calls += 1
            raise OSError(errno.EIO, "round7 forbidden raw socket.close retry")
        real_socket_close(descriptor)

    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _worker_command(worker_mode, tmp_path),
    )
    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_uncertain_worker_session", observed_uncertain)
    monkeypatch.setattr(isolated_process.os, "dup", observed_dup)
    monkeypatch.setattr(isolated_process.os, "close", fail_target_os_close)
    monkeypatch.setattr(isolated_process.socket, "close", fail_target_socket_close)
    monkeypatch.setattr(isolated_process.socket.socket, "recv_into", observed_recv_into)
    cancel_requested = threading.Event()
    if worker_mode == "cancel":
        cancel_requested.set()

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=0.15 if worker_mode in {"timeout", "cancel"} else 1.0,
        cancel_requested=cancel_requested,
    )

    assert target_descriptor and original_identity and stable_descriptors
    descriptor = target_descriptor[0]
    try:
        assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
        assert _original_descriptor_is_gone(descriptor, original_identity[0])
        assert _recover_original_datagram(descriptor, original_identity[0]) == b""
        for stable_descriptor in stable_descriptors:
            _assert_descriptor_absent(stable_descriptor)
        with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
            retained_references = {
                reference.descriptor
                for owner in isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.values()
                for reference in owner.retained_references.values()
            }
        assert descriptor in retained_references
        assert socket_close_calls == 0
        assert drained_buffers and all(not any(buffer) for buffer in drained_buffers)
        assert retained_buffers and all(not any(buffer) for buffer in retained_buffers)
        assert all(canary not in surface for surface in invocation_surfaces)
        if worker_mode == "start":
            assert spawned_processes == []
        else:
            _assert_no_external_secret_surface(
                canary=canary,
                invocation_surfaces=invocation_surfaces,
                retained_buffers=retained_buffers,
                spawned_processes=spawned_processes,
                tmp_path=tmp_path,
            )
        for artifact in tmp_path.rglob("*"):
            if artifact.is_file():
                assert canary not in artifact.read_bytes()
    finally:
        with suppress(OSError):
            real_os_close(descriptor)


@pytest.mark.parametrize("fault_profile", ["fstat", "drain", "drain_and_shutdown"])
def test_round7_post_reap_faults_destroy_the_credential_identity_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_profile: str,
) -> None:
    """Inspection and invalidation faults cannot strand a queued secret."""

    canary = f"round7-{fault_profile}-credential-canary".encode()
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    stable_descriptors: list[int] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    reaped_before_fault: list[bool] = []
    fault_active = False
    fault_calls = 0
    real_fstat = os.fstat
    real_dup = os.dup
    real_recv_into = socket.socket.recv_into
    real_shutdown = socket.socket.shutdown
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_stop = isolated_process._stop_worker

    def observed_run(**kwargs: Any) -> object:
        descriptor = kwargs["pass_fds"][0]
        target_descriptor.append(descriptor)
        original_identity.append(real_fstat(descriptor))
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def observed_dup(descriptor: int) -> int:
        duplicate = real_dup(descriptor)
        if target_descriptor and descriptor == target_descriptor[0]:
            stable_descriptors.append(duplicate)
        return duplicate

    def stop_then_activate_fault(worker: object) -> object:
        nonlocal fault_active
        result = real_stop(worker)  # type: ignore[arg-type]
        reaped_before_fault.append(worker.process.poll() is not None)  # type: ignore[attr-defined]
        fault_active = True
        return result

    def faulting_fstat(descriptor: int) -> os.stat_result:
        nonlocal fault_calls
        if (
            fault_active
            and fault_profile == "fstat"
            and target_descriptor
            and descriptor == target_descriptor[0]
        ):
            fault_calls += 1
            raise OSError(errno.EIO, "round7 injected fstat failure")
        return real_fstat(descriptor)

    def is_credential_socket(transport: socket.socket) -> bool:
        if not original_identity:
            return False
        try:
            return _stable_identity(real_fstat(transport.fileno())) == _stable_identity(
                original_identity[0]
            )
        except OSError:
            return False

    def faulting_recv_into(
        transport: socket.socket,
        buffer: bytearray,
        size: int,
        flags: int,
    ) -> int:
        nonlocal fault_calls
        if (
            fault_active
            and fault_profile in {"drain", "drain_and_shutdown"}
            and is_credential_socket(transport)
        ):
            fault_calls += 1
            raise OSError(errno.EIO, "round7 injected credential drain failure")
        return real_recv_into(transport, buffer, size, flags)

    def faulting_shutdown(transport: socket.socket, how: int) -> None:
        nonlocal fault_calls
        if (
            fault_active
            and fault_profile == "drain_and_shutdown"
            and is_credential_socket(transport)
        ):
            fault_calls += 1
            raise OSError(errno.EIO, "round7 injected credential shutdown failure")
        real_shutdown(transport, how)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_stop_worker", stop_then_activate_fault)
    monkeypatch.setattr(isolated_process.os, "dup", observed_dup)
    monkeypatch.setattr(isolated_process.os, "fstat", faulting_fstat)
    monkeypatch.setattr(isolated_process.socket.socket, "recv_into", faulting_recv_into)
    monkeypatch.setattr(isolated_process.socket.socket, "shutdown", faulting_shutdown)

    try:
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )
    finally:
        fault_active = False

    assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
    assert fault_calls >= 1
    assert reaped_before_fault == [True]
    assert target_descriptor and original_identity and stable_descriptors
    _assert_descriptor_absent(target_descriptor[0])
    for stable_descriptor in stable_descriptors:
        _assert_descriptor_absent(stable_descriptor)
    _assert_no_external_secret_surface(
        canary=canary,
        invocation_surfaces=invocation_surfaces,
        retained_buffers=retained_buffers,
        spawned_processes=spawned_processes,
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round7_clean_process_control_propagates_only_after_credential_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """Clean containment preserves process control instead of converting its flow."""

    canary = f"round7-clean-{control_type.__name__}-credential-canary".encode()
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    stable_descriptors: list[int] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    reaped_before_control: list[bool] = []
    real_dup = os.dup
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_stop = isolated_process._stop_worker

    def observed_run(**kwargs: Any) -> object:
        descriptor = kwargs["pass_fds"][0]
        target_descriptor.append(descriptor)
        original_identity.append(os.fstat(descriptor))
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_dup(descriptor: int) -> int:
        duplicate = real_dup(descriptor)
        if target_descriptor and descriptor == target_descriptor[0]:
            stable_descriptors.append(duplicate)
        return duplicate

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def reap_then_control(worker: object) -> object:
        result = real_stop(worker)  # type: ignore[arg-type]
        assert result is None
        reaped_before_control.append(worker.process.poll() is not None)  # type: ignore[attr-defined]
        raise control_type("round7 clean containment process control")

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_stop_worker", reap_then_control)
    monkeypatch.setattr(isolated_process.os, "dup", observed_dup)

    with pytest.raises(control_type, match="round7 clean containment process control"):
        lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )

    assert reaped_before_control == [True]
    assert target_descriptor and original_identity and stable_descriptors
    _assert_descriptor_absent(target_descriptor[0])
    for stable_descriptor in stable_descriptors:
        _assert_descriptor_absent(stable_descriptor)
    _assert_no_external_secret_surface(
        canary=canary,
        invocation_surfaces=invocation_surfaces,
        retained_buffers=retained_buffers,
        spawned_processes=spawned_processes,
        tmp_path=tmp_path,
    )


def test_round7_unprovable_os_containment_retains_explicit_ownership_and_never_returns_green(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every destruction primitive fails, ownership remains explicit and blocking."""

    canary = b"round7-unprovable-os-containment-credential-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    reaped_before_fault: list[bool] = []
    fault_active = False
    real_fstat = os.fstat
    real_close = os.close
    real_recv_into = socket.socket.recv_into
    real_shutdown = socket.socket.shutdown
    real_pipe = os.pipe
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_stop = isolated_process._stop_worker

    def observed_run(**kwargs: Any) -> object:
        descriptor = kwargs["pass_fds"][0]
        target_descriptor.append(descriptor)
        original_identity.append(real_fstat(descriptor))
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def stop_then_activate_fault(worker: object) -> object:
        nonlocal fault_active
        result = real_stop(worker)  # type: ignore[arg-type]
        reaped_before_fault.append(worker.process.poll() is not None)  # type: ignore[attr-defined]
        fault_active = True
        return result

    def is_credential_socket(transport: socket.socket) -> bool:
        if not original_identity:
            return False
        try:
            return _stable_identity(real_fstat(transport.fileno())) == _stable_identity(
                original_identity[0]
            )
        except OSError:
            return False

    def faulting_recv_into(
        transport: socket.socket,
        buffer: bytearray,
        size: int,
        flags: int,
    ) -> int:
        if fault_active and is_credential_socket(transport):
            raise OSError(errno.EIO, "round7 unprovable credential drain failure")
        return real_recv_into(transport, buffer, size, flags)

    def faulting_shutdown(transport: socket.socket, how: int) -> None:
        if fault_active and is_credential_socket(transport):
            raise OSError(errno.EIO, "round7 unprovable credential shutdown failure")
        real_shutdown(transport, how)

    def faulting_pipe() -> tuple[int, int]:
        if fault_active:
            raise OSError(errno.EMFILE, "round7 unprovable atomic replacement failure")
        return real_pipe()

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_stop_worker", stop_then_activate_fault)
    monkeypatch.setattr(isolated_process.socket.socket, "recv_into", faulting_recv_into)
    monkeypatch.setattr(isolated_process.socket.socket, "shutdown", faulting_shutdown)
    monkeypatch.setattr(isolated_process.os, "pipe", faulting_pipe)

    retained_ownership: list[object] = []
    try:
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )
        assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
        assert reaped_before_fault == [True]
        assert target_descriptor and original_identity
        with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
            retained_ownership = list(isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.values())
        assert retained_ownership
        assert any(
            owner.descriptor == target_descriptor[0]  # type: ignore[attr-defined]
            and owner.stable_descriptor is not None  # type: ignore[attr-defined]
            for owner in retained_ownership
        )
        _assert_no_external_secret_surface(
            canary=canary,
            invocation_surfaces=invocation_surfaces,
            retained_buffers=retained_buffers,
            spawned_processes=spawned_processes,
            tmp_path=tmp_path,
        )
    finally:
        fault_active = False
        for owner in retained_ownership:
            references = {
                owner.descriptor,  # type: ignore[attr-defined]
                owner.stable_descriptor,  # type: ignore[attr-defined]
            }
            for descriptor in references:
                if descriptor is not None:
                    with suppress(OSError):
                        real_close(descriptor)
        with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
            isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.clear()


def test_round7_unknown_identity_after_raw_reuse_never_mutates_the_unrelated_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inspection failure is not permission to overwrite a reused FD number."""

    canary = b"round7-unknown-identity-credential-canary"
    unrelated_payload = b"round7-unrelated-reused-descriptor"
    unrelated_path = tmp_path / "unrelated-reused-descriptor"
    unrelated_path.write_bytes(unrelated_payload)
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    stable_descriptors: list[int] = []
    replacement_identity: list[os.stat_result] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    reaped_before_reuse: list[bool] = []
    fault_active = False
    real_fstat = os.fstat
    real_close = os.close
    real_dup = os.dup
    real_dup2 = os.dup2
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_stop = isolated_process._stop_worker

    def observed_run(**kwargs: Any) -> object:
        descriptor = kwargs["pass_fds"][0]
        target_descriptor.append(descriptor)
        original_identity.append(real_fstat(descriptor))
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def observed_dup(descriptor: int) -> int:
        duplicate = real_dup(descriptor)
        if target_descriptor and descriptor == target_descriptor[0]:
            stable_descriptors.append(duplicate)
        return duplicate

    def stop_then_reuse_raw_number(worker: object) -> object:
        nonlocal fault_active
        result = real_stop(worker)  # type: ignore[arg-type]
        reaped_before_reuse.append(worker.process.poll() is not None)  # type: ignore[attr-defined]
        descriptor = target_descriptor[0]
        real_close(descriptor)
        replacement = os.open(unrelated_path, os.O_RDONLY)
        if replacement != descriptor:
            real_dup2(replacement, descriptor, inheritable=False)
            real_close(replacement)
        replacement_identity.append(real_fstat(descriptor))
        fault_active = True
        return result

    def faulting_fstat(descriptor: int) -> os.stat_result:
        if fault_active and target_descriptor and descriptor == target_descriptor[0]:
            raise OSError(errno.EIO, "round7 unknown reused descriptor identity")
        return real_fstat(descriptor)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_stop_worker", stop_then_reuse_raw_number)
    monkeypatch.setattr(isolated_process.os, "dup", observed_dup)
    monkeypatch.setattr(isolated_process.os, "fstat", faulting_fstat)

    retained_ownership: list[object] = []
    try:
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )
        fault_active = False
        assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
        assert reaped_before_reuse == [True]
        assert target_descriptor and original_identity and replacement_identity
        descriptor = target_descriptor[0]
        assert real_fstat(descriptor) == replacement_identity[0]
        assert os.read(descriptor, len(unrelated_payload) + 1) == unrelated_payload
        for stable_descriptor in stable_descriptors:
            _assert_descriptor_absent(stable_descriptor)
        with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
            retained_ownership = list(isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.values())
        assert any(
            owner.descriptor == descriptor  # type: ignore[attr-defined]
            for owner in retained_ownership
        )
        _assert_no_external_secret_surface(
            canary=canary,
            invocation_surfaces=invocation_surfaces,
            retained_buffers=retained_buffers,
            spawned_processes=spawned_processes,
            tmp_path=tmp_path,
        )
    finally:
        fault_active = False
        if target_descriptor:
            with suppress(OSError):
                real_close(target_descriptor[0])
        for owner in retained_ownership:
            stable_descriptor = owner.stable_descriptor  # type: ignore[attr-defined]
            if stable_descriptor is not None:
                with suppress(OSError):
                    real_close(stable_descriptor)
        with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
            isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.clear()


def test_round7_real_drain_with_atomic_replacement_failure_retains_every_open_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drained bytes do not erase ownership of descriptors that remain open."""

    canary = b"round7-drained-but-open-credential-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    stable_descriptors: list[int] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    reaped_before_fault: list[bool] = []
    fault_active = False
    real_fstat = os.fstat
    real_close = os.close
    real_dup = os.dup
    real_pipe = os.pipe
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_stop = isolated_process._stop_worker

    def observed_run(**kwargs: Any) -> object:
        target_descriptor.extend(kwargs["pass_fds"])
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def observed_dup(descriptor: int) -> int:
        duplicate = real_dup(descriptor)
        if target_descriptor and descriptor == target_descriptor[0]:
            stable_descriptors.append(duplicate)
        return duplicate

    def stop_then_activate_fault(worker: object) -> object:
        nonlocal fault_active
        result = real_stop(worker)  # type: ignore[arg-type]
        reaped_before_fault.append(worker.process.poll() is not None)  # type: ignore[attr-defined]
        fault_active = True
        return result

    def faulting_pipe() -> tuple[int, int]:
        if fault_active:
            raise OSError(errno.EMFILE, "round7 replacement unavailable after real drain")
        return real_pipe()

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_stop_worker", stop_then_activate_fault)
    monkeypatch.setattr(isolated_process.os, "dup", observed_dup)
    monkeypatch.setattr(isolated_process.os, "pipe", faulting_pipe)

    retained_ownership: list[object] = []
    try:
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )
        fault_active = False
        assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
        assert reaped_before_fault == [True]
        assert target_descriptor and stable_descriptors
        real_fstat(target_descriptor[0])
        for stable_descriptor in stable_descriptors:
            real_fstat(stable_descriptor)
        with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
            retained_ownership = list(isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.values())
        assert any(
            owner.descriptor == target_descriptor[0]  # type: ignore[attr-defined]
            and owner.stable_descriptor in stable_descriptors  # type: ignore[attr-defined]
            for owner in retained_ownership
        )
        _assert_no_external_secret_surface(
            canary=canary,
            invocation_surfaces=invocation_surfaces,
            retained_buffers=retained_buffers,
            spawned_processes=spawned_processes,
            tmp_path=tmp_path,
        )
    finally:
        fault_active = False
        references = set(target_descriptor + stable_descriptors)
        for descriptor in references:
            with suppress(OSError):
                real_close(descriptor)
        with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
            isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.clear()


@pytest.mark.parametrize(
    ("control_point", "control_type"),
    [
        ("recv", KeyboardInterrupt),
        ("dup2", SystemExit),
        ("close", asyncio.CancelledError),
    ],
)
def test_round7_cleanup_process_control_is_reraised_after_independently_proven_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_point: str,
    control_type: type[BaseException],
) -> None:
    """Control survives only when every interrupted cleanup action is proven."""

    canary = f"round7-{control_point}-control-credential-canary".encode()
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    stable_descriptors: list[int] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    invocation_surfaces: list[bytes] = []
    reaped_before_control: list[bool] = []
    fault_active = False
    injected = False
    real_fstat = os.fstat
    real_close = os.close
    real_dup = os.dup
    real_dup2 = os.dup2
    real_recv_into = socket.socket.recv_into
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_stop = isolated_process._stop_worker

    def observed_run(**kwargs: Any) -> object:
        descriptor = kwargs["pass_fds"][0]
        target_descriptor.append(descriptor)
        original_identity.append(real_fstat(descriptor))
        invocation_surfaces.extend(
            repr(kwargs[field]).encode("utf-8") for field in ("command", "request", "env")
        )
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def observed_dup(descriptor: int) -> int:
        duplicate = real_dup(descriptor)
        if target_descriptor and descriptor == target_descriptor[0]:
            stable_descriptors.append(duplicate)
        return duplicate

    def stop_then_activate_control(worker: object) -> object:
        nonlocal fault_active
        result = real_stop(worker)  # type: ignore[arg-type]
        reaped_before_control.append(worker.process.poll() is not None)  # type: ignore[attr-defined]
        fault_active = True
        return result

    def is_credential_socket(transport: socket.socket) -> bool:
        if not original_identity:
            return False
        try:
            return _stable_identity(real_fstat(transport.fileno())) == _stable_identity(
                original_identity[0]
            )
        except OSError:
            return False

    def control_after_recv(
        transport: socket.socket,
        buffer: bytearray,
        size: int,
        flags: int,
    ) -> int:
        nonlocal injected
        received = real_recv_into(transport, buffer, size, flags)
        if (
            fault_active
            and control_point == "recv"
            and not injected
            and is_credential_socket(transport)
        ):
            injected = True
            raise control_type("round7 process control after real recv")
        return received

    def control_after_dup2(old: int, new: int, *, inheritable: bool = True) -> None:
        nonlocal injected
        real_dup2(old, new, inheritable=inheritable)
        if (
            fault_active
            and control_point == "dup2"
            and not injected
            and target_descriptor
            and new == target_descriptor[0]
        ):
            injected = True
            raise control_type("round7 process control after real dup2")

    def control_after_close(descriptor: int) -> None:
        nonlocal injected
        real_close(descriptor)
        if (
            fault_active
            and control_point == "close"
            and not injected
            and target_descriptor
            and descriptor == target_descriptor[0]
        ):
            injected = True
            raise control_type("round7 process control after real close")

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_stop_worker", stop_then_activate_control)
    monkeypatch.setattr(isolated_process.os, "dup", observed_dup)
    monkeypatch.setattr(isolated_process.os, "dup2", control_after_dup2)
    monkeypatch.setattr(isolated_process.os, "close", control_after_close)
    monkeypatch.setattr(isolated_process.socket.socket, "recv_into", control_after_recv)

    outcome: lifecycle_process.LifecycleProcessOutcome | None = None
    escaped: BaseException | None = None
    try:
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )
    except BaseException as exc:
        escaped = exc

    assert injected
    assert reaped_before_control == [True]
    assert target_descriptor and original_identity and stable_descriptors
    descriptor = target_descriptor[0]
    try:
        if control_point == "recv":
            assert outcome is None
            assert isinstance(escaped, control_type)
            _assert_descriptor_absent(descriptor)
            with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
                assert isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP == {}
        else:
            assert escaped is None
            assert outcome is not None
            assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
            assert _original_descriptor_is_gone(descriptor, original_identity[0])
            assert _recover_original_datagram(descriptor, original_identity[0]) == b""
            with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
                retained_references = {
                    reference.descriptor
                    for owner in isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.values()
                    for reference in owner.retained_references.values()
                }
            assert descriptor in retained_references
        for stable_descriptor in stable_descriptors:
            _assert_descriptor_absent(stable_descriptor)
        _assert_no_external_secret_surface(
            canary=canary,
            invocation_surfaces=invocation_surfaces,
            retained_buffers=retained_buffers,
            spawned_processes=spawned_processes,
            tmp_path=tmp_path,
        )
    finally:
        with suppress(OSError):
            real_close(descriptor)
