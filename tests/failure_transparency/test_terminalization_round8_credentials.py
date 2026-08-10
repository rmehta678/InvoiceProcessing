"""Focused RED reproducers for the eighth Task 9 credential review."""

from __future__ import annotations

import asyncio
import errno
import os
import socket
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from invoice_agents import isolated_process
from invoice_agents.db import migration_process


def _credential_descriptor(canary: bytes) -> tuple[int, os.stat_result]:
    reader, writer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        writer.sendall(canary)
        descriptor = reader.detach()
        return descriptor, os.fstat(descriptor)
    finally:
        writer.close()


def _stable_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_mode, status.st_rdev


def _descriptor_no_longer_names(
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
    if _descriptor_no_longer_names(descriptor, original):
        return b""
    with socket.socket(fileno=os.dup(descriptor)) as probe:
        probe.settimeout(0.25)
        return probe.recv(16_385)


def _worker_is_contained(worker: object) -> bool:
    if worker.cleaned:  # type: ignore[attr-defined]
        return True
    with migration_process._QUARANTINED_WORKERS_LOCK:
        return (
            migration_process._QUARANTINED_WORKERS.get(  # type: ignore[attr-defined]
                worker.process_id  # type: ignore[attr-defined]
            )
            is worker
        )


def _follow_up_command() -> list[str]:
    return [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'round8-follow-up')",
    ]


def _run_follow_up() -> isolated_process.IsolatedProcessResult:
    return isolated_process.run_isolated_process(
        command=_follow_up_command(),
        request=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=1_024,
    )


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
@pytest.mark.parametrize("fault_profile", ["transient", "persistent"])
def test_round8_stop_control_never_forgets_a_live_unquarantined_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
    fault_profile: str,
) -> None:
    """A stop interruption is reraised only after reap or durable quarantine."""

    canary = f"round8-stop-{fault_profile}-{control_type.__name__}-canary".encode()
    descriptor, original = _credential_descriptor(canary)
    captured_workers: list[object] = []
    stop_calls = 0
    live_at_control: list[bool] = []
    escaped: BaseException | None = None
    contained_before_test_cleanup = False
    canary_after_return = canary
    real_capture = isolated_process._capture_worker_session
    real_cleanup = isolated_process._cleanup_worker_session
    real_stop = isolated_process._stop_worker
    real_close = os.close

    def observed_capture(process: object) -> object:
        worker = real_capture(process)  # type: ignore[arg-type]
        captured_workers.append(worker)
        return worker

    def control_before_stop(worker: object) -> object:
        nonlocal stop_calls
        stop_calls += 1
        if fault_profile == "persistent" or stop_calls == 1:
            live_at_control.append(worker.process.poll() is None)  # type: ignore[attr-defined]
            raise control_type("round8 control before worker reap")
        return real_stop(worker)  # type: ignore[arg-type]

    def persistent_independent_reap_failure(worker: object) -> None:
        if fault_profile == "persistent":
            raise OSError(errno.EIO, "round8 persistent independent reap failure")
        real_cleanup(worker)  # type: ignore[arg-type]

    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)
    monkeypatch.setattr(
        isolated_process,
        "_cleanup_worker_session",
        persistent_independent_reap_failure,
    )
    monkeypatch.setattr(isolated_process, "_stop_worker", control_before_stop)

    try:
        try:
            isolated_process.run_isolated_process(
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
                request=b"{}",
                timeout_seconds=0.05,
                max_response_bytes=1_024,
                pass_fds=(descriptor,),
            )
        except BaseException as exc:
            escaped = exc
        assert captured_workers
        contained_before_test_cleanup = all(
            _worker_is_contained(worker) for worker in captured_workers
        )
        canary_after_return = _recover_original_datagram(descriptor, original)
    finally:
        for worker in captured_workers:
            with suppress(BaseException):
                real_stop(worker)  # type: ignore[arg-type]
        if not _descriptor_no_longer_names(descriptor, original):
            with suppress(OSError):
                real_close(descriptor)

    assert live_at_control and all(live_at_control)
    assert isinstance(escaped, control_type)
    assert contained_before_test_cleanup
    assert canary_after_return == b""


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round8_capture_control_survives_independently_proven_retirement(
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """Descriptor-capture control is not collapsed into a cleanup error."""

    canary = f"round8-capture-{control_type.__name__}-canary".encode()
    descriptor, original = _credential_descriptor(canary)
    stable_descriptors: list[int] = []
    injected = False
    escaped: BaseException | None = None
    real_set_inheritable = os.set_inheritable
    real_close = os.close

    def control_after_inheritable(candidate: int, inheritable: bool) -> None:
        nonlocal injected
        real_set_inheritable(candidate, inheritable)
        if not injected:
            injected = True
            stable_descriptors.append(candidate)
            raise control_type("round8 control after stable alias capture")

    monkeypatch.setattr(isolated_process.os, "set_inheritable", control_after_inheritable)

    try:
        try:
            isolated_process.run_isolated_process(
                command=_follow_up_command(),
                request=b"{}",
                timeout_seconds=1.0,
                max_response_bytes=1_024,
                pass_fds=(descriptor,),
            )
        except BaseException as exc:
            escaped = exc
        assert injected and stable_descriptors
        assert _descriptor_no_longer_names(descriptor, original)
        assert _recover_original_datagram(descriptor, original) == b""
        for stable_descriptor in stable_descriptors:
            with pytest.raises(OSError) as absent_error:
                os.fstat(stable_descriptor)
            assert absent_error.value.errno == errno.EBADF
    finally:
        if not _descriptor_no_longer_names(descriptor, original):
            with suppress(OSError):
                real_close(descriptor)
        for stable_descriptor in stable_descriptors:
            with suppress(OSError):
                real_close(stable_descriptor)

    assert isinstance(escaped, control_type)


def test_round8_uncertain_close_never_retries_a_reused_raw_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No check/action retry may close an unrelated FD after close is uncertain."""

    canary = b"round8-close-reuse-credential-canary"
    unrelated_payload = b"round8-unrelated-descriptor"
    unrelated_path = tmp_path / "round8-unrelated-descriptor"
    unrelated_path.write_bytes(unrelated_payload)
    descriptor, original = _credential_descriptor(canary)
    close_became_uncertain = False
    adversarial_ranges: list[tuple[int, int]] = []
    post_uncertainty_raw_actions: list[str] = []
    unrelated_identities: list[os.stat_result] = []
    unrelated_was_mutated = False
    real_close = os.close
    real_closerange = os.closerange
    real_dup2 = os.dup2
    real_fstat = os.fstat

    def install_unrelated_reuse() -> None:
        real_close(descriptor)
        replacement = os.open(unrelated_path, os.O_RDONLY)
        if replacement != descriptor:
            real_dup2(replacement, descriptor, inheritable=False)
            real_close(replacement)
        unrelated_identities.append(real_fstat(descriptor))

    def interrupted_replacement_close(candidate: int) -> None:
        nonlocal close_became_uncertain
        if candidate == descriptor:
            if close_became_uncertain:
                post_uncertainty_raw_actions.append("close")
                install_unrelated_reuse()
            else:
                current = real_fstat(candidate)
                if _stable_identity(current) != _stable_identity(original):
                    close_became_uncertain = True
                    raise OSError(errno.EINTR, "round8 uncertain close")
        real_close(candidate)

    def overwrite_after_uncertain_close(
        old: int,
        new: int,
        *,
        inheritable: bool = True,
    ) -> None:
        if close_became_uncertain and new == descriptor:
            post_uncertainty_raw_actions.append("dup2")
            install_unrelated_reuse()
        real_dup2(old, new, inheritable=inheritable)

    def reuse_between_check_and_action(first: int, last: int) -> None:
        if close_became_uncertain and first <= descriptor < last:
            adversarial_ranges.append((first, last))
            post_uncertainty_raw_actions.append("closerange")
            install_unrelated_reuse()
        real_closerange(first, last)

    monkeypatch.setattr(isolated_process.os, "close", interrupted_replacement_close)
    monkeypatch.setattr(isolated_process.os, "closerange", reuse_between_check_and_action)
    monkeypatch.setattr(isolated_process.os, "dup2", overwrite_after_uncertain_close)

    try:
        with pytest.raises(isolated_process.IsolatedProcessCleanupError):
            isolated_process.run_isolated_process(
                command=[os.fspath(tmp_path / "missing-round8-worker")],
                request=b"{}",
                timeout_seconds=1.0,
                max_response_bytes=1_024,
                pass_fds=(descriptor,),
            )
        if unrelated_identities:
            try:
                unrelated_was_mutated = _stable_identity(
                    real_fstat(descriptor)
                ) != _stable_identity(unrelated_identities[0])
            except OSError:
                unrelated_was_mutated = True
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        monkeypatch.setattr(isolated_process.os, "closerange", real_closerange)
        monkeypatch.setattr(isolated_process.os, "dup2", real_dup2)
        with suppress(OSError):
            real_close(descriptor)
        with suppress(isolated_process.IsolatedProcessCleanupError):
            _run_follow_up()

    assert close_became_uncertain
    assert adversarial_ranges == []
    assert post_uncertainty_raw_actions == []
    assert not unrelated_was_mutated


def test_round8_retained_ownership_reconciles_without_private_registry_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-ups stay bounded/blocking, then self-heal from proven absence."""

    canary = b"round8-production-reconciliation-canary"
    descriptor, original = _credential_descriptor(canary)
    real_pipe = os.pipe
    real_close = os.close
    real_fstat = os.fstat

    def unavailable_replacement() -> tuple[int, int]:
        raise OSError(errno.EMFILE, "round8 replacement unavailable")

    monkeypatch.setattr(isolated_process.os, "pipe", unavailable_replacement)
    with pytest.raises(isolated_process.IsolatedProcessCleanupError):
        isolated_process.run_isolated_process(
            command=[os.fspath(tmp_path / "missing-round8-worker")],
            request=b"{}",
            timeout_seconds=1.0,
            max_response_bytes=1_024,
            pass_fds=(descriptor,),
        )
    monkeypatch.setattr(isolated_process.os, "pipe", real_pipe)

    with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
        retained = tuple(isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP.values())
        retained_keys = tuple(isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP)
    assert len(retained) == 1
    stable_reference = retained[0].stable_descriptor
    assert stable_reference is not None

    references = {
        reference
        for owner in retained
        for reference in (owner.descriptor, owner.stable_descriptor)
        if reference is not None
    }
    reconciliation_fstat_calls = 0

    def observed_reconciliation_fstat(reference: int) -> os.stat_result:
        nonlocal reconciliation_fstat_calls
        if reference in references:
            reconciliation_fstat_calls += 1
        return real_fstat(reference)

    monkeypatch.setattr(isolated_process.os, "fstat", observed_reconciliation_fstat)

    for _attempt in range(3):
        calls_before_attempt = reconciliation_fstat_calls
        with pytest.raises(isolated_process.IsolatedProcessCleanupError):
            _run_follow_up()
        with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
            assert tuple(isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP) == retained_keys
        calls_this_attempt = reconciliation_fstat_calls - calls_before_attempt
        assert 1 <= calls_this_attempt <= len(references)
    with pytest.raises(OSError) as retired_alias_error:
        real_fstat(stable_reference)
    assert retired_alias_error.value.errno == errno.EBADF
    assert _stable_identity(real_fstat(descriptor)) == _stable_identity(original)
    for reference in references:
        with suppress(OSError):
            real_close(reference)

    follow_up: isolated_process.IsolatedProcessResult | None = None
    try:
        follow_up = _run_follow_up()
    finally:
        for reference in references:
            with suppress(OSError):
                real_close(reference)

    assert follow_up == isolated_process.IsolatedProcessResult(b"round8-follow-up", None)
    assert _recover_original_datagram(descriptor, original) == b""
    with isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
        assert isolated_process._UNPROVEN_DESCRIPTOR_OWNERSHIP == {}
