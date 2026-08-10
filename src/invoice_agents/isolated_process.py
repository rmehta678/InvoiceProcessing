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
from contextlib import suppress
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


@dataclass(slots=True)
class _OwnedDescriptor:
    """One transferred descriptor plus a stable parent-only alias."""

    descriptor: int
    identity: _DescriptorIdentity | None
    stable_descriptor: int | None
    stable_identity: _DescriptorIdentity | None


@dataclass(slots=True)
class _CleanupOutcome:
    """Cleanup proof plus the first process-control exception to preserve."""

    failed: bool = False
    control_error: BaseException | None = None

    def capture_control(self, error: BaseException) -> None:
        if self.control_error is None:
            self.control_error = error

    def merge(self, other: _CleanupOutcome) -> None:
        self.failed = self.failed or other.failed
        if self.control_error is None:
            self.control_error = other.control_error


_UNPROVEN_DESCRIPTOR_OWNERSHIP: dict[int, _OwnedDescriptor] = {}
_UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK = threading.Lock()


def _descriptor_identity(descriptor: int) -> _DescriptorIdentity:
    status = os.fstat(descriptor)
    return _DescriptorIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        file_type=stat.S_IFMT(status.st_mode),
        device_type=status.st_rdev,
    )


def _unproven_descriptor_ownership_exists() -> bool:
    with _UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
        return bool(_UNPROVEN_DESCRIPTOR_OWNERSHIP)


def _retain_unproven_descriptor_ownership(owned: _OwnedDescriptor) -> None:
    with _UNPROVEN_DESCRIPTOR_OWNERSHIP_LOCK:
        _UNPROVEN_DESCRIPTOR_OWNERSHIP[id(owned)] = owned


def _capture_owned_fds(descriptors: set[int]) -> tuple[dict[int, _OwnedDescriptor], bool]:
    """Capture identities and stable aliases before any descriptor can be uncertain."""

    owned: dict[int, _OwnedDescriptor] = {}
    capture_failed = False
    for descriptor in descriptors:
        identity: _DescriptorIdentity | None = None
        try:
            identity = _descriptor_identity(descriptor)
        except BaseException as exc:
            if isinstance(exc, OSError) and exc.errno == errno.EBADF:
                continue
            capture_failed = True
        stable_descriptor: int | None = None
        stable_identity: _DescriptorIdentity | None = None
        try:
            stable_descriptor = os.dup(descriptor)
            os.set_inheritable(stable_descriptor, False)
        except BaseException:
            capture_failed = True
        if stable_descriptor is not None:
            try:
                stable_identity = _descriptor_identity(stable_descriptor)
            except BaseException:
                capture_failed = True
            if identity is None:
                identity = stable_identity
            elif stable_identity is not None and stable_identity != identity:
                capture_failed = True
        owned[descriptor] = _OwnedDescriptor(
            descriptor=descriptor,
            identity=identity,
            stable_descriptor=stable_descriptor,
            stable_identity=stable_identity,
        )
    return owned, capture_failed


def _zero_buffer(buffer: bytearray) -> None:
    for index in range(len(buffer)):
        buffer[index] = 0


def _drain_socket_descriptor(descriptor: int) -> tuple[bool, _CleanupOutcome]:
    """Consume and erase queued datagrams through an identity-bound alias."""

    transport: socket.socket | None = None
    buffer = bytearray(8_192)
    drained = False
    outcome = _CleanupOutcome()
    try:
        transport = socket.socket(fileno=descriptor)
        for _attempt in range(64):
            try:
                received = transport.recv_into(buffer, len(buffer), socket.MSG_DONTWAIT)
            except BlockingIOError:
                drained = True
                break
            except OSError as exc:
                if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                    drained = True
                    break
                if exc.errno == errno.ECONNRESET:
                    continue
                outcome.failed = True
                break
            except BaseException as exc:
                outcome.capture_control(exc)
                continue
            finally:
                _zero_buffer(buffer)
            if received == 0:
                drained = True
                break
        else:
            outcome.failed = True
        if not drained:
            try:
                transport.shutdown(socket.SHUT_RD)
            except Exception:
                outcome.failed = True
            except BaseException as exc:
                outcome.capture_control(exc)
            else:
                drained = True
    except Exception:
        outcome.failed = True
    except BaseException as exc:
        outcome.capture_control(exc)
    finally:
        _zero_buffer(buffer)
        if transport is not None:
            try:
                detached = transport.detach()
            except Exception:
                outcome.failed = True
            except BaseException as exc:
                outcome.capture_control(exc)
            else:
                if detached != descriptor:
                    outcome.failed = True
    return drained, outcome


def _descriptor_is_socket(
    owned: _OwnedDescriptor,
) -> tuple[bool | None, _CleanupOutcome]:
    outcome = _CleanupOutcome()
    identity = owned.stable_identity or owned.identity
    if identity is not None:
        return identity.file_type == stat.S_IFSOCK, outcome
    descriptor = owned.stable_descriptor or owned.descriptor
    transport: socket.socket | None = None
    try:
        transport = socket.socket(fileno=descriptor)
        transport.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
    except OSError as exc:
        if exc.errno in {errno.ENOTSOCK, errno.EBADF}:
            return False, outcome
        outcome.failed = True
        return None, outcome
    except BaseException as exc:
        outcome.capture_control(exc)
        return None, outcome
    finally:
        if transport is not None:
            with suppress(BaseException):
                transport.detach()
    return True, outcome


def _classify_after_failed_fstat(
    descriptor: int,
    expected_identity: _DescriptorIdentity | None,
    outcome: _CleanupOutcome,
) -> Literal["match", "unknown"]:
    """Use a disposable alias to confirm a match without mutating the raw FD."""

    if expected_identity is None:
        return "unknown"
    probe: int | None = None
    probe_identity: _DescriptorIdentity | None = None
    probe_retired = False
    try:
        probe = os.dup(descriptor)
        os.set_inheritable(probe, False)
        probe_identity = _descriptor_identity(probe)
    except Exception:
        outcome.failed = True
    except BaseException as exc:
        outcome.capture_control(exc)
    finally:
        if probe is not None:
            try:
                os.closerange(probe, probe + 1)
            except Exception:
                outcome.failed = True
            except BaseException as exc:
                outcome.capture_control(exc)
            try:
                remaining_identity = _descriptor_identity(probe)
            except OSError as exc:
                probe_retired = exc.errno == errno.EBADF
                if not probe_retired:
                    outcome.failed = True
            except BaseException as exc:
                outcome.capture_control(exc)
                probe_retired = False
            else:
                probe_retired = remaining_identity != probe_identity
            if not probe_retired:
                outcome.failed = True
                _retain_unproven_descriptor_ownership(
                    _OwnedDescriptor(
                        descriptor=probe,
                        identity=probe_identity,
                        stable_descriptor=None,
                        stable_identity=None,
                    )
                )
    if probe_identity == expected_identity and probe_retired:
        return "match"
    return "unknown"


def _descriptor_state(
    descriptor: int,
    expected_identity: _DescriptorIdentity | None,
    outcome: _CleanupOutcome,
) -> Literal["match", "absent", "mismatch", "unknown"]:
    """Classify one raw number without treating inspection failure as ownership."""

    try:
        current_identity = _descriptor_identity(descriptor)
    except OSError as exc:
        if exc.errno == errno.EBADF:
            return "absent"
        outcome.failed = True
        return _classify_after_failed_fstat(descriptor, expected_identity, outcome)
    except BaseException as exc:
        outcome.capture_control(exc)
        return "unknown"
    if expected_identity is None:
        outcome.failed = True
        return "unknown"
    if current_identity != expected_identity:
        return "mismatch"
    return "match"


def _reconcile_replacement(
    descriptor: int,
    original_identity: _DescriptorIdentity | None,
    replacement_identity: _DescriptorIdentity | None,
    outcome: _CleanupOutcome,
) -> Literal["replaced", "retired", "unresolved"]:
    """Prove what an interrupted atomic replacement actually changed."""

    if replacement_identity is not None:
        replacement_state = _descriptor_state(
            descriptor,
            replacement_identity,
            outcome,
        )
        if replacement_state == "match":
            return "replaced"
        if replacement_state == "absent":
            return "retired"
        if replacement_state == "unknown":
            return "unresolved"
    original_state = _descriptor_state(descriptor, original_identity, outcome)
    if original_state in {"absent", "mismatch"}:
        return "retired"
    return "unresolved"


def _close_replaced_descriptor(
    descriptor: int,
    replacement_identity: _DescriptorIdentity | None,
) -> tuple[bool, _CleanupOutcome]:
    """Close a harmless replacement once without retrying an uncertain number."""

    outcome = _CleanupOutcome()
    close_was_failure = False
    try:
        os.close(descriptor)
    except Exception as close_error:
        state = _descriptor_state(
            descriptor,
            replacement_identity,
            outcome,
        )
        confirmed_absent = (
            isinstance(close_error, OSError)
            and close_error.errno == errno.EBADF
            and state == "absent"
        )
        if confirmed_absent:
            return True, outcome
        close_was_failure = True
        outcome.failed = True
    except BaseException as exc:
        outcome.capture_control(exc)
        state = _descriptor_state(descriptor, replacement_identity, outcome)
    else:
        return True, outcome
    if state == "match" and replacement_identity is not None:
        try:
            # This number is proven to name our harmless replacement, never the
            # original socket.  Contain that nonsecret leak without invoking the
            # uncertain close primitive again or touching a changed identity.
            os.closerange(descriptor, descriptor + 1)
        except Exception:
            outcome.failed = True
        except BaseException as exc:
            outcome.capture_control(exc)
        final_state = _descriptor_state(descriptor, replacement_identity, outcome)
        if final_state not in {"absent", "mismatch"}:
            outcome.failed = True
        return final_state in {"absent", "mismatch"}, outcome
    if state == "unknown":
        outcome.failed = True
        return False, outcome
    if close_was_failure:
        outcome.failed = True
    return True, outcome


def _retire_owned_fds(
    owned: dict[int, _OwnedDescriptor],
    *,
    child_session_empty: bool,
) -> _CleanupOutcome:
    """Erase queued secrets, atomically replace identities, then close once.

    No uncertain close is attempted until every still-owned secret-bearing raw
    descriptor has first been replaced by a unique harmless identity.  A close
    failure is contained only while that harmless identity still matches; a
    changed or reused number is never touched again.
    """

    outcome = _CleanupOutcome()
    if not owned:
        return outcome
    invalidation_proven: dict[int, bool] = {}
    socket_owned: dict[int, bool] = {}
    for descriptor, descriptor_owner in owned.items():
        is_socket, socket_outcome = _descriptor_is_socket(descriptor_owner)
        outcome.merge(socket_outcome)
        if is_socket is None:
            is_socket = True
        socket_owned[descriptor] = is_socket
        if not is_socket:
            invalidation_proven[descriptor] = True
            continue
        invalidation_proven[descriptor] = False
        drain_candidates = (
            (
                descriptor_owner.stable_descriptor,
                descriptor_owner.stable_identity,
            ),
            (descriptor_owner.descriptor, descriptor_owner.identity),
        )
        inspected: set[int] = set()
        for drain_descriptor, drain_identity in drain_candidates:
            if drain_descriptor is None or drain_descriptor in inspected:
                continue
            inspected.add(drain_descriptor)
            state = _descriptor_state(drain_descriptor, drain_identity, outcome)
            if state != "match":
                continue
            drained, drain_outcome = _drain_socket_descriptor(drain_descriptor)
            outcome.merge(drain_outcome)
            invalidation_proven[descriptor] = drained
            break

    harmless_descriptor: int | None = None
    harmless_identity: _DescriptorIdentity | None = None
    harmless_writer: int | None = None
    harmless_writer_identity: _DescriptorIdentity | None = None
    try:
        harmless_descriptor, harmless_writer = os.pipe()
        harmless_identity = _descriptor_identity(harmless_descriptor)
        harmless_writer_identity = _descriptor_identity(harmless_writer)
    except Exception:
        outcome.failed = True
    except BaseException as exc:
        outcome.capture_control(exc)

    replaced: dict[int, int] = {}
    secret_references_retired: dict[int, bool] = {}
    for descriptor, descriptor_owner in owned.items():
        references = (
            (descriptor_owner.descriptor, descriptor_owner.identity),
            (descriptor_owner.stable_descriptor, descriptor_owner.stable_identity),
        )
        all_references_retired = True
        for reference, expected_identity in references:
            if reference is None:
                continue
            state = _descriptor_state(reference, expected_identity, outcome)
            if state in {"absent", "mismatch"}:
                continue
            if state == "unknown" or harmless_descriptor is None:
                all_references_retired = False
                continue
            try:
                os.dup2(harmless_descriptor, reference, inheritable=False)
            except Exception:
                outcome.failed = True
                replacement_state = _reconcile_replacement(
                    reference,
                    expected_identity,
                    harmless_identity,
                    outcome,
                )
                if replacement_state == "replaced":
                    replaced[reference] = descriptor
                elif replacement_state == "unresolved":
                    all_references_retired = False
            except BaseException as exc:
                outcome.capture_control(exc)
                replacement_state = _reconcile_replacement(
                    reference,
                    expected_identity,
                    harmless_identity,
                    outcome,
                )
                if replacement_state == "replaced":
                    replaced[reference] = descriptor
                elif replacement_state == "unresolved":
                    outcome.failed = True
                    all_references_retired = False
            else:
                replaced[reference] = descriptor
        secret_references_retired[descriptor] = all_references_retired

    for reference, owner_descriptor in replaced.items():
        retired, close_outcome = _close_replaced_descriptor(reference, harmless_identity)
        outcome.merge(close_outcome)
        if not retired:
            secret_references_retired[owner_descriptor] = False
    if harmless_descriptor is not None and harmless_descriptor not in replaced:
        retired, close_outcome = _close_replaced_descriptor(
            harmless_descriptor,
            harmless_identity,
        )
        outcome.merge(close_outcome)
        if not retired:
            outcome.failed = True
    if harmless_writer is not None:
        retired, close_outcome = _close_replaced_descriptor(
            harmless_writer,
            harmless_writer_identity,
        )
        outcome.merge(close_outcome)
        if not retired:
            outcome.failed = True

    for descriptor, is_socket in socket_owned.items():
        references_retired = secret_references_retired.get(descriptor, False)
        secret_destroyed = (
            (not is_socket)
            or invalidation_proven.get(
                descriptor,
                False,
            )
            or (child_session_empty and references_retired)
        )
        if not references_retired or not secret_destroyed:
            outcome.failed = True
            _retain_unproven_descriptor_ownership(owned[descriptor])
    owned.clear()
    return outcome


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
    retains an identity-bound alias until the worker session is empty, erases
    queued socket data, and replaces the transferred identities before close.
    """

    descriptor_numbers = {
        descriptor for descriptor in pass_fds if type(descriptor) is int and descriptor >= 3
    }
    owned_descriptors, descriptor_capture_failed = _capture_owned_fds(descriptor_numbers)
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
        descriptor_cleanup = _retire_owned_fds(
            owned_descriptors,
            child_session_empty=True,
        )
        if descriptor_capture_failed or descriptor_cleanup.failed:
            raise IsolatedProcessCleanupError from None
        if descriptor_cleanup.control_error is not None:
            raise descriptor_cleanup.control_error
        raise ValueError("invalid isolated worker controller input")
    if descriptor_capture_failed:
        _retire_owned_fds(owned_descriptors, child_session_empty=True)
        raise IsolatedProcessCleanupError from None
    try:
        if not _retry_quarantined_workers() or _unproven_descriptor_ownership_exists():
            raise IsolatedProcessCleanupError from None
    except BaseException:
        descriptor_cleanup = _retire_owned_fds(
            owned_descriptors,
            child_session_empty=True,
        )
        if descriptor_cleanup.failed:
            raise IsolatedProcessCleanupError from None
        raise
    process: subprocess.Popen[bytes] | None = None
    worker: object | None = None
    start_failed = False
    primary_error: BaseException | None = None
    cleanup_control_error: BaseException | None = None
    cleanup_error: object | None = None
    response: bytes | None = None
    response_failure: Literal["timeout", "cancelled", "protocol"] | None = None
    try:
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
        except Exception:
            start_failed = True
        except BaseException as exc:
            primary_error = exc
        if process is not None:
            if start_failed or primary_error is not None:
                try:
                    worker = _uncertain_worker_session(process)
                except BaseException as exc:
                    if primary_error is None:
                        primary_error = exc
            else:
                try:
                    worker = _capture_worker_session(process)
                except Exception:
                    start_failed = True
                    try:
                        worker = _uncertain_worker_session(process)
                    except BaseException as exc:
                        primary_error = exc
                except BaseException as exc:
                    primary_error = exc
                    with suppress(BaseException):
                        worker = _uncertain_worker_session(process)
        if worker is not None:
            start_failed = (
                start_failed
                or not worker.watcher_initialized  # type: ignore[attr-defined]
                or not worker.identity_verified  # type: ignore[attr-defined]
            )
            if primary_error is None and not start_failed:
                try:
                    response, response_failure = _read_response(
                        worker,
                        timeout_seconds=timeout_seconds,
                        max_response_bytes=max_response_bytes,
                        cancel_requested=cancel_requested,
                    )
                except BaseException as exc:
                    primary_error = exc
    finally:
        if worker is not None:
            try:
                cleanup_error = _stop_worker(worker)  # type: ignore[arg-type]
            except BaseException as exc:
                cleanup_control_error = exc
        child_session_empty = (process is None) or bool(
            worker is not None and worker.cleaned  # type: ignore[attr-defined]
        )
        descriptor_cleanup = _retire_owned_fds(
            owned_descriptors,
            child_session_empty=child_session_empty,
        )
    worker_cleanup_failed = cleanup_error is not None or (
        process is not None and not child_session_empty
    )
    if descriptor_cleanup.failed or worker_cleanup_failed:
        raise IsolatedProcessCleanupError from None
    if primary_error is not None:
        raise primary_error
    if cleanup_control_error is not None:
        raise cleanup_control_error
    if descriptor_cleanup.control_error is not None:
        raise descriptor_cleanup.control_error
    if worker is None:
        raise IsolatedProcessCleanupError from None
    if start_failed:
        return IsolatedProcessResult(None, "start")
    if response_failure is not None:
        return IsolatedProcessResult(None, response_failure)
    if process is None or process.returncode != 0:
        return IsolatedProcessResult(None, "crash")
    return IsolatedProcessResult(response, None)
