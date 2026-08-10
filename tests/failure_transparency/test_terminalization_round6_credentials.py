"""Focused RED reproducers for the sixth Task 9 credential review."""

from __future__ import annotations

import errno
import os
import socket
import sys
import threading
from collections.abc import Callable
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
    """Every round-6 test fails immediately at an unstubbed provider seam."""

    def forbidden(_settings: Settings) -> object:
        raise AssertionError("round6 test reached an unstubbed provider boundary")

    monkeypatch.setattr(orchestration, "create_model_client", forbidden)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", "import os; os._exit(86)"],
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
        "case_round6_credential_cleanup",
        f"exec_{'6' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )
    return settings, started_at, claim


def _recover_queued_datagram(
    descriptor: int,
    original: os.stat_result,
) -> tuple[bool, bytes]:
    try:
        current = os.fstat(descriptor)
    except OSError:
        return False, b""
    if _stable_identity(current) != _stable_identity(original):
        return False, b""
    with socket.socket(fileno=os.dup(descriptor)) as probe:
        probe.settimeout(0.25)
        return True, probe.recv(lifecycle_process.LIFECYCLE_MAX_CREDENTIAL_BYTES + 1)


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


def _close_if_original_or_retained_descriptor(
    descriptor: int,
    original: os.stat_result,
    close: Callable[[int], None],
) -> None:
    try:
        current = os.fstat(descriptor)
    except OSError:
        return
    current_identity = isolated_process._descriptor_identity(descriptor)
    with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
        retained_identities = {
            identity
            for owner in isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.values()
            for reference in owner.retained_references.values()
            if reference.descriptor == descriptor
            for identity in reference.expected_identities
        }
    if (
        _stable_identity(current) == _stable_identity(original)
        or current_identity in retained_identities
    ):
        with suppress(OSError):
            close(descriptor)


def test_round6_parent_credential_close_failure_is_cleanup_failure_and_secret_is_unrecoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed parent close cannot become an ordinary crash with a readable secret."""

    canary = b"round6-parent-close-failure-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary.decode())
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    injected = False
    real_close = os.close
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session

    def observed_run(**kwargs: Any) -> object:
        target_descriptor.extend(kwargs["pass_fds"])
        original_identity.append(os.fstat(kwargs["pass_fds"][0]))
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def observed_capture(process: object) -> object:
        spawned_processes.append(process)
        return real_capture(process)  # type: ignore[arg-type]

    def fail_exact_parent_close(descriptor: int) -> None:
        nonlocal injected
        if target_descriptor and descriptor == target_descriptor[0] and not injected:
            injected = True
            raise OSError(errno.EIO, "round6 injected credential close failure")
        real_close(descriptor)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process.os, "close", fail_exact_parent_close)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )
    assert target_descriptor and original_identity and injected
    descriptor = target_descriptor[0]
    try:
        descriptor_open, recovered = _recover_queued_datagram(
            descriptor,
            original_identity[0],
        )
        observed = (
            outcome.error_code,
            descriptor_open,
            recovered,
            bool(retained_buffers) and all(not any(buffer) for buffer in retained_buffers),
            bool(spawned_processes)
            and all(process.poll() is not None for process in spawned_processes),  # type: ignore[attr-defined]
        )
        assert observed == (
            "LIFECYCLE_WORKER_CLEANUP_FAILED",
            False,
            b"",
            True,
            True,
        )
    finally:
        _close_if_original_or_retained_descriptor(
            descriptor,
            original_identity[0],
            real_close,
        )


def _worker_command(worker_mode: str, tmp_path: Path) -> list[str]:
    if worker_mode == "start":
        return [os.fspath(tmp_path / "missing-round6-worker")]
    tail = {
        "success": (
            "b=bytearray(os.read(d,16385)); os.close(d); "
            "b[:]=bytes(len(b)); sys.stdout.buffer.write(b'{\"ok\":true}')"
        ),
        "crash": "os._exit(9)",
        "timeout": "time.sleep(30)",
        "cancel": "time.sleep(30)",
    }[worker_mode]
    return [
        sys.executable,
        "-c",
        "import json,os,sys,time; "
        "p=json.loads(sys.stdin.buffer.read()); d=p['credential_fd']; " + tail,
    ]


@pytest.mark.parametrize("failure_profile", ["transient", "persistent"])
@pytest.mark.parametrize("worker_mode", ["success", "crash", "timeout", "cancel", "start"])
def test_round6_close_failure_precedes_every_lifecycle_terminal_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_mode: str,
    failure_profile: str,
) -> None:
    """Every worker outcome is secondary to an unclean parent credential close."""

    canary = f"round6-{failure_profile}-{worker_mode}-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    command = _worker_command(worker_mode, tmp_path)
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    retained_buffers: list[bytearray] = []
    spawned_processes: list[object] = []
    injected_failures = 0
    real_close = os.close
    real_run = isolated_process.run_isolated_process
    real_send = lifecycle_process._send_private_credential
    real_capture = isolated_process._capture_worker_session
    real_uncertain = isolated_process._uncertain_worker_session

    def observed_run(**kwargs: Any) -> object:
        target_descriptor.extend(kwargs["pass_fds"])
        original_identity.append(os.fstat(kwargs["pass_fds"][0]))
        return real_run(**kwargs)

    def observed_send(writer: socket.socket, credential: bytearray) -> None:
        retained_buffers.append(credential)
        real_send(writer, credential)

    def remember_process(process: object) -> None:
        if process not in spawned_processes:
            spawned_processes.append(process)

    def observed_capture(process: object) -> object:
        remember_process(process)
        return real_capture(process)  # type: ignore[arg-type]

    def observed_uncertain(process: object) -> object:
        remember_process(process)
        return real_uncertain(process)  # type: ignore[arg-type]

    def fail_parent_close(descriptor: int) -> None:
        nonlocal injected_failures
        if (
            target_descriptor
            and descriptor == target_descriptor[0]
            and (failure_profile == "persistent" or injected_failures == 0)
        ):
            injected_failures += 1
            raise OSError(errno.EIO, "round6 injected credential close failure")
        real_close(descriptor)

    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", lambda: command)
    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(lifecycle_process, "_send_private_credential", observed_send)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(isolated_process, "_uncertain_worker_session", observed_uncertain)
    monkeypatch.setattr(isolated_process.os, "close", fail_parent_close)
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

    assert target_descriptor and original_identity and injected_failures >= 1
    descriptor = target_descriptor[0]
    try:
        assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
        assert _original_descriptor_is_gone(descriptor, original_identity[0])
        assert retained_buffers and all(not any(buffer) for buffer in retained_buffers)
        if worker_mode == "start":
            assert spawned_processes == []
        else:
            assert spawned_processes
            assert all(process.poll() is not None for process in spawned_processes)  # type: ignore[attr-defined]
        for artifact in tmp_path.rglob("*"):
            if artifact.is_file():
                assert canary.encode() not in artifact.read_bytes()
    finally:
        _close_if_original_or_retained_descriptor(
            descriptor,
            original_identity[0],
            real_close,
        )


@pytest.mark.parametrize("descriptor_remains_open", [False, True])
def test_round6_ebadf_only_proves_cleanup_when_descriptor_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptor_remains_open: bool,
) -> None:
    """EBADF is accepted only when fstat independently proves nonexistence."""

    canary = f"round6-ebadf-{descriptor_remains_open}-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    target_descriptor: list[int] = []
    original_identity: list[os.stat_result] = []
    real_close = os.close
    real_run = isolated_process.run_isolated_process
    injected = False

    def observed_run(**kwargs: Any) -> object:
        target_descriptor.extend(kwargs["pass_fds"])
        original_identity.append(os.fstat(kwargs["pass_fds"][0]))
        return real_run(**kwargs)

    def ebadf_close(descriptor: int) -> None:
        nonlocal injected
        if target_descriptor and descriptor == target_descriptor[0] and not injected:
            injected = True
            if not descriptor_remains_open:
                real_close(descriptor)
            raise OSError(errno.EBADF, "round6 injected EBADF")
        real_close(descriptor)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(isolated_process.os, "close", ebadf_close)
    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert target_descriptor and original_identity and injected
    descriptor = target_descriptor[0]
    try:
        assert outcome.error_code == (
            "LIFECYCLE_WORKER_CLEANUP_FAILED"
            if descriptor_remains_open
            else "LIFECYCLE_WORKER_CRASHED"
        )
        assert _original_descriptor_is_gone(descriptor, original_identity[0])
    finally:
        _close_if_original_or_retained_descriptor(
            descriptor,
            original_identity[0],
            real_close,
        )


def test_round6_eintr_reuse_never_closes_the_replacement_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An indeterminate close cannot be retried against a reused descriptor number."""

    canary = "round6-eintr-reuse-credential-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    replacement_path = tmp_path / "unrelated-replacement"
    replacement_path.write_bytes(b"round6-unrelated-descriptor")
    target_descriptor: list[int] = []
    replacement_identity: list[os.stat_result] = []
    real_close = os.close
    real_run = isolated_process.run_isolated_process
    injected = False
    target_close_calls = 0

    def observed_run(**kwargs: Any) -> object:
        target_descriptor.extend(kwargs["pass_fds"])
        return real_run(**kwargs)

    def close_then_reuse_and_interrupt(descriptor: int) -> None:
        nonlocal injected, target_close_calls
        if target_descriptor and descriptor == target_descriptor[0]:
            target_close_calls += 1
            if not injected:
                injected = True
                real_close(descriptor)
                replacement = os.open(replacement_path, os.O_RDONLY)
                if replacement != descriptor:
                    os.dup2(replacement, descriptor, inheritable=False)
                    real_close(replacement)
                replacement_identity.append(os.fstat(descriptor))
                raise OSError(errno.EINTR, "round6 injected EINTR after descriptor reuse")
        real_close(descriptor)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(isolated_process.os, "close", close_then_reuse_and_interrupt)
    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert target_descriptor and injected and replacement_identity
    descriptor = target_descriptor[0]
    try:
        assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
        assert target_close_calls == 1
        assert os.fstat(descriptor) == replacement_identity[0]
        assert os.read(descriptor, 64) == b"round6-unrelated-descriptor"
    finally:
        real_close(descriptor)
