"""Shared bounded process/session controller for case workers."""

from __future__ import annotations

import errno
import os
import selectors
import socket
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Literal

from invoice_agents.db.migration_process import (
    _capture_worker_session,
    _retry_quarantined_workers,
    _stop_worker,
    _uncertain_worker_session,
)
from invoice_agents.worker_environment import (
    sanitized_worker_environment as _sanitized_worker_environment,
)

_POLL_SECONDS = 0.02


def sanitized_worker_environment() -> dict[str, str]:
    """Compatibility export for worker callers using the shared controller."""

    return _sanitized_worker_environment()


@dataclass(frozen=True, slots=True)
class _DescriptorIdentity:
    """Stable-enough identity for refusing to close a reused descriptor number."""

    device: int
    inode: int
    file_type: int
    device_type: int


def _descriptor_identity(descriptor: int) -> _DescriptorIdentity:
    status = os.fstat(descriptor)
    return _DescriptorIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        file_type=stat.S_IFMT(status.st_mode),
        device_type=status.st_rdev,
    )


def _capture_owned_fds(descriptors: set[int]) -> dict[int, _DescriptorIdentity | None]:
    """Capture transferred FD identities without treating an absent FD as owned."""

    owned: dict[int, _DescriptorIdentity | None] = {}
    for descriptor in descriptors:
        try:
            owned[descriptor] = _descriptor_identity(descriptor)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                # Identity inspection failed, so retain conservative ownership.
                owned[descriptor] = None
    return owned


def _close_owned_fds(owned: dict[int, _DescriptorIdentity | None]) -> bool:
    """Attempt each transferred close once and retain every unproven ownership.

    A failed ``close(2)`` is never retried by descriptor number: EINTR and
    deferred I/O errors can mean that the number was already released and
    reused.  ``fstat`` proves either nonexistence or a changed identity before
    ownership is discarded.  Confirmed EBADF/nonexistence is already clean;
    every other close error remains a fail-closed lifecycle outcome.
    """

    close_failed = False
    for descriptor, expected_identity in tuple(owned.items()):
        try:
            os.close(descriptor)
        except OSError as close_error:
            current_identity: _DescriptorIdentity | None
            try:
                current_identity = _descriptor_identity(descriptor)
            except OSError as inspect_error:
                if inspect_error.errno == errno.EBADF:
                    owned.pop(descriptor, None)
                    current_identity = None
                else:
                    current_identity = expected_identity
            else:
                if expected_identity is not None and current_identity != expected_identity:
                    # The original FD is gone and this number now belongs elsewhere.
                    owned.pop(descriptor, None)
            if close_error.errno != errno.EBADF or current_identity is not None:
                close_failed = True
        else:
            owned.pop(descriptor, None)
    return close_failed


def _destroy_still_owned_fds(owned: dict[int, _DescriptorIdentity | None]) -> bool:
    """Destroy retained descriptors after no child can consume their contents.

    This is containment, not a successful retry: callers preserve the original
    close-failure flag and report cleanup failure even when containment proves
    the secret-bearing descriptor gone.  Socket FDs use the socket close
    primitive so an injected or persistently failing ``os.close`` cannot leave
    a queued credential recoverable.
    """

    for descriptor, expected_identity in tuple(owned.items()):
        try:
            current_identity = _descriptor_identity(descriptor)
        except OSError as exc:
            if exc.errno == errno.EBADF:
                owned.pop(descriptor, None)
            continue
        if expected_identity is not None and current_identity != expected_identity:
            # A failed close released the original before this number was reused.
            owned.pop(descriptor, None)
            continue
        try:
            if current_identity.file_type == stat.S_IFSOCK:
                socket.close(descriptor)
            else:
                os.closerange(descriptor, descriptor + 1)
        except OSError:
            pass
        try:
            final_identity = _descriptor_identity(descriptor)
        except OSError as exc:
            if exc.errno == errno.EBADF:
                owned.pop(descriptor, None)
            continue
        if expected_identity is not None and final_identity != expected_identity:
            owned.pop(descriptor, None)
    return bool(owned)


@dataclass(frozen=True, slots=True)
class IsolatedProcessResult:
    """Sanitized bytes observed only after the worker session is proven empty."""

    response: bytes | None
    failure: Literal["start", "timeout", "cancelled", "protocol", "crash"] | None


class IsolatedProcessCleanupError(Exception):
    """The Task 8 session primitive could not prove cleanup."""


def _read_response(
    worker: object,
    *,
    timeout_seconds: float,
    max_response_bytes: int,
    cancel_requested: threading.Event | None,
) -> tuple[bytes | None, Literal["timeout", "cancelled", "protocol"] | None]:
    process = worker.process  # type: ignore[attr-defined]
    stdout = process.stdout
    watcher = worker.exit_watcher  # type: ignore[attr-defined]
    if stdout is None or watcher is None:
        return None, "protocol"
    descriptor = stdout.fileno()
    os.set_blocking(descriptor, False)
    deadline = time.monotonic() + timeout_seconds
    response = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            if cancel_requested is not None and cancel_requested.is_set():
                return None, "cancelled"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "timeout"
            if not selector.select(min(remaining, _POLL_SECONDS)):
                if watcher.wait(0):
                    break
                continue
            try:
                chunk = os.read(
                    descriptor,
                    min(8_192, max_response_bytes + 1 - len(response)),
                )
            except OSError:
                return None, "protocol"
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > max_response_bytes:
                return None, "protocol"
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, "timeout"
    if cancel_requested is not None and cancel_requested.is_set():
        return None, "cancelled"
    if not watcher.wait(remaining):
        return None, "timeout"
    return bytes(response), None


def run_isolated_process(
    *,
    command: list[str],
    request: bytes,
    timeout_seconds: float,
    max_response_bytes: int,
    cancel_requested: threading.Event | None = None,
    pass_fds: tuple[int, ...] = (),
    env: dict[str, str] | None = None,
) -> IsolatedProcessResult:
    """Spawn one fresh session, then stop/reap all members before returning.

    Every descriptor in ``pass_fds`` transfers to this controller.  The parent
    copy is closed immediately after the spawn attempt on every path.
    """

    descriptor_numbers = {
        descriptor for descriptor in pass_fds if type(descriptor) is int and descriptor >= 3
    }
    owned_descriptors = _capture_owned_fds(descriptor_numbers)
    descriptors_are_valid = all(
        type(descriptor) is int and descriptor >= 3 for descriptor in pass_fds
    ) and len(set(pass_fds)) == len(pass_fds)
    if (
        type(timeout_seconds) is not float
        or timeout_seconds <= 0
        or type(max_response_bytes) is not int
        or max_response_bytes <= 0
        or not request
        or type(command) is not list
        or not command
        or any(type(argument) is not str or not argument for argument in command)
        or not descriptors_are_valid
    ):
        close_failed = _close_owned_fds(owned_descriptors)
        containment_failed = _destroy_still_owned_fds(owned_descriptors)
        if close_failed or containment_failed:
            raise IsolatedProcessCleanupError from None
        raise ValueError("invalid isolated worker controller input")
    try:
        if not _retry_quarantined_workers():
            raise IsolatedProcessCleanupError from None
    except BaseException:
        close_failed = _close_owned_fds(owned_descriptors)
        containment_failed = _destroy_still_owned_fds(owned_descriptors)
        if close_failed or containment_failed:
            raise IsolatedProcessCleanupError from None
        raise
    process: subprocess.Popen[bytes] | None = None
    worker: object | None = None
    start_failed = False
    descriptor_close_attempted = False
    descriptor_cleanup_failed = False
    try:
        with tempfile.TemporaryFile() as request_stream:
            request_stream.write(request)
            request_stream.seek(0)
            process = subprocess.Popen(
                command,
                stdin=request_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                pass_fds=pass_fds,
                env=env,
            )
        descriptor_close_attempted = True
        descriptor_cleanup_failed = _close_owned_fds(owned_descriptors)
        worker = _capture_worker_session(process)
    except Exception:
        start_failed = True
        if process is not None:
            worker = _uncertain_worker_session(process)
    finally:
        if not descriptor_close_attempted:
            descriptor_cleanup_failed = _close_owned_fds(owned_descriptors)
    if worker is None:
        containment_failed = _destroy_still_owned_fds(owned_descriptors)
        if descriptor_cleanup_failed or containment_failed:
            raise IsolatedProcessCleanupError from None
        raise IsolatedProcessCleanupError from None
    start_failed = (
        start_failed
        or not worker.watcher_initialized  # type: ignore[attr-defined]
        or not worker.identity_verified  # type: ignore[attr-defined]
    )
    response: bytes | None = None
    response_failure: Literal["timeout", "cancelled", "protocol"] | None = None
    if not start_failed:
        response, response_failure = _read_response(
            worker,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            cancel_requested=cancel_requested,
        )
    cleanup_error = _stop_worker(worker)  # type: ignore[arg-type]
    containment_failed = _destroy_still_owned_fds(owned_descriptors)
    if cleanup_error is not None or descriptor_cleanup_failed or containment_failed:
        raise IsolatedProcessCleanupError from None
    if start_failed:
        return IsolatedProcessResult(None, "start")
    if response_failure is not None:
        return IsolatedProcessResult(None, response_failure)
    if process is None or process.returncode != 0:
        return IsolatedProcessResult(None, "crash")
    return IsolatedProcessResult(response, None)
