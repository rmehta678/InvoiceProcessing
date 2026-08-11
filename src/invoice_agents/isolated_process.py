"""Shared bounded process/session controller for case workers."""

from __future__ import annotations

import os
import selectors
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal, cast

from invoice_agents.db.migration_process import (
    _capture_worker_session,
    _cleanup_cooperative_worker_session,
    _uncertain_worker_session,
    _worker_resource_cleanup_is_poisoned,
    _WorkerSession,
)
from invoice_agents.worker_environment import (
    sanitized_worker_environment as _sanitized_worker_environment,
)

_CREDENTIAL_FRAME_HEADER_BYTES = 4
_POLL_SECONDS = 0.02
_MAX_PRIVATE_PIPE_ENDPOINTS = 128
_MAX_SUPERVISOR_STATUS_BYTES = 64
_MAX_SUPERVISOR_OWNER_BYTES = 64
_SUPERVISOR_CLEANUP_SECONDS = 2.0
_START_CONTROL = b"S"
_ABORT_CONTROL = b"A"


class IsolatedProcessCleanupError(Exception):
    """The worker session or one-way private transport was not cleanly retired."""


class _StartupWaitInterrupted(Exception):
    """An internal startup wait was interrupted before worker authorization."""


_PRIVATE_PIPE_STATE_LOCK = threading.Lock()
_ACTIVE_PRIVATE_PIPE_ENDPOINTS = 0
_PRIVATE_PIPE_STATE_POISONED = False


class ProcessCancellation:
    """Thread-safe cancellation state for isolated process admission."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False

    def set(self) -> None:
        with self._lock:
            self._requested = True

    def is_set(self) -> bool:
        with self._lock:
            return self._requested

    def _deliver_if_not_cancelled(self, delivery: Callable[[], None]) -> bool:
        """Order cancellation wholly before or after one nonblocking write attempt."""

        with self._lock:
            if self._requested:
                return False
            delivery()
            return True


def sanitized_worker_environment() -> dict[str, str]:
    """Compatibility export for worker callers using the shared controller."""

    return _sanitized_worker_environment()


@dataclass(slots=True)
class PrivatePipeEndpoint:
    """Own exactly one directional anonymous-pipe descriptor.

    Ownership is retired before ``close(2)`` is attempted.  A failed close is
    visible to the caller, but the raw number is never probed or retried after
    another thread could have reused it.
    """

    _descriptor: int | None
    readable: bool
    writable: bool
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _managed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._descriptor is None:
            raise ValueError("private pipe endpoint requires an open descriptor")

    def fileno(self) -> int:
        descriptor = self._descriptor
        if descriptor is None:
            raise ValueError("private pipe endpoint is closed")
        return descriptor

    @property
    def closed(self) -> bool:
        return self._descriptor is None

    def close(self) -> None:
        with self._lock:
            descriptor = self._descriptor
            if descriptor is None:
                return
            self._descriptor = None
            _release_private_endpoint(self)
            try:
                os.close(descriptor)
            except BaseException:
                _poison_private_endpoint_state()
                raise

    def detach(self) -> int:
        """Transfer ownership for focused child-reader tests."""

        with self._lock:
            descriptor = self._descriptor
            if descriptor is None:
                raise ValueError("private pipe endpoint is closed")
            self._descriptor = None
            _release_private_endpoint(self)
            return descriptor

    def __enter__(self) -> PrivatePipeEndpoint:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(slots=True)
class PrivatePipeInput:
    """One bounded secret payload and its strictly directional transport."""

    reader: PrivatePipeEndpoint
    writer: PrivatePipeEndpoint
    payload: bytearray = field(repr=False)
    max_payload_bytes: int


@dataclass(slots=True)
class _TransportOutcome:
    """Credential-resource cleanup proof plus one preserved control exception."""

    failed: bool = False
    control_error: BaseException | None = None

    def capture_control(self, error: BaseException) -> None:
        if self.control_error is None:
            self.control_error = error


@dataclass(slots=True)
class _WorkerCleanupOutcome:
    """Independent proof that the exact worker session is empty and reaped."""

    contained: bool = False
    reaped: bool = False
    failed: bool = False
    control_error: BaseException | None = None

    def capture_control(self, error: BaseException) -> None:
        if self.control_error is None:
            self.control_error = error


@dataclass(slots=True)
class _SpawnedProcess:
    """PID-backed child handle published by the startup owner."""

    pid: int
    stdout: BinaryIO | None
    returncode: int | None = None
    args: tuple[str, ...] = ("isolated-worker",)
    _native_handle: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._native_handle is not None:
            self.returncode = self._native_handle.poll()
            return self.returncode
        try:
            observed_id, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            # Another waiter consumed the only exact status. Treat that loss of
            # evidence as failure rather than manufacturing a successful exit.
            self.returncode = 1
        else:
            if observed_id == self.pid:
                self.returncode = int(os.waitstatus_to_exitcode(status))
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self._native_handle is not None:
            self.returncode = self._native_handle.wait(timeout=timeout)
            return self.returncode
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.poll()
            if result is not None:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, cast(float, timeout))
            time.sleep(_POLL_SECONDS)


@dataclass(slots=True)
class _EndpointReader:
    """File-like response reader retaining endpoint cleanup semantics."""

    endpoint: PrivatePipeEndpoint

    @property
    def closed(self) -> bool:
        return self.endpoint.closed

    def fileno(self) -> int:
        return self.endpoint.fileno()

    def close(self) -> None:
        self.endpoint.close()


def _after_worker_spawn(_process: _SpawnedProcess) -> None:
    """Fault-injection seam reached only after PID ownership is published."""


def _supervisor_command(
    *,
    status_descriptor: int,
    control_descriptor: int,
    owner_descriptor: int,
    lifetime_descriptor: int,
    command: list[str],
) -> list[str]:
    """Build an isolated argv for the trusted local supervisor script."""

    return [
        os.fspath(Path(sys.executable).resolve(strict=True)),
        "-I",
        os.fspath(Path(__file__).with_name("spawn_supervisor.py").resolve(strict=True)),
        "--status-fd",
        str(status_descriptor),
        "--control-fd",
        str(control_descriptor),
        "--owner-fd",
        str(owner_descriptor),
        "--lifetime-fd",
        str(lifetime_descriptor),
        "--",
        *command,
    ]


def _declare_worker_descriptors(
    supervisor_command: list[str],
    worker_descriptors: tuple[int, ...],
) -> list[str]:
    """Declare every worker-only inherited FD in the trusted protocol argv."""

    separator = supervisor_command.index("--")
    declaration = [
        value for descriptor in worker_descriptors for value in ("--worker-fd", str(descriptor))
    ]
    return [*supervisor_command[:separator], *declaration, *supervisor_command[separator:]]


def _initialize_reserved_process(
    process: subprocess.Popen[bytes],
    *args: object,
    **kwargs: object,
) -> None:
    """Initialize one already-retained ``Popen`` object in place."""

    subprocess.Popen.__init__(process, *args, **kwargs)  # type: ignore[call-overload]


def _reserved_process_has_native_child(process: subprocess.Popen[bytes]) -> bool:
    """Return whether initialization yielded a live, unreaped native child handle."""

    process_id = getattr(process, "pid", None)
    return (
        getattr(process, "_child_created", False) is True
        and type(process_id) is int
        and process_id > 0
        and getattr(process, "returncode", None) is None
    )


def _launch_supervisor(
    owner: _StartupOwner,
    *,
    command: list[str],
    request_stream: BinaryIO,
    response_writer: PrivatePipeEndpoint,
    status_writer: PrivatePipeEndpoint,
    control_reader: PrivatePipeEndpoint,
    owner_writer: PrivatePipeEndpoint,
    lifetime_writer: PrivatePipeEndpoint,
    pass_fds: tuple[int, ...],
    env: dict[str, str] | None,
) -> None:
    """Reserve, initialize, and publish exactly one native supervisor handle."""

    status_descriptor = status_writer.fileno()
    control_descriptor = control_reader.fileno()
    owner_descriptor = owner_writer.fileno()
    lifetime_descriptor = lifetime_writer.fileno()
    inherited = (
        *pass_fds,
        status_descriptor,
        control_descriptor,
        owner_descriptor,
        lifetime_descriptor,
    )
    process = cast("subprocess.Popen[bytes]", subprocess.Popen.__new__(subprocess.Popen))
    process._child_created = False  # type: ignore[attr-defined]
    owner.reserve_native_process(process)
    initialization_completed = False
    classification_completed = False
    native_child = False
    launch_error: BaseException | None = None
    try:
        supervisor_command = _supervisor_command(
            status_descriptor=status_descriptor,
            control_descriptor=control_descriptor,
            owner_descriptor=owner_descriptor,
            lifetime_descriptor=lifetime_descriptor,
            command=command,
        )
        supervisor_command = _declare_worker_descriptors(supervisor_command, pass_fds)
        _initialize_reserved_process(
            process,
            supervisor_command,
            stdin=request_stream,
            stdout=response_writer.fileno(),
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            pass_fds=inherited,
            env=env,
        )
        initialization_completed = True
        native_child = _reserved_process_has_native_child(process)
        classification_completed = True
        if not native_child:
            raise RuntimeError("isolated supervisor initialization produced no native child")
    except BaseException as exc:
        launch_error = exc
        if initialization_completed and not classification_completed:
            # A completed Popen initializer is positive native-child ownership
            # evidence even when classification itself is interrupted.
            native_child = True
        elif not classification_completed:
            while True:
                try:
                    native_child = _reserved_process_has_native_child(process)
                    classification_completed = True
                    break
                except BaseException:
                    try:
                        time.sleep(_POLL_SECONDS)
                    except BaseException:
                        continue
    if not native_child:
        process._child_created = False  # type: ignore[attr-defined]
        owner.release_native_process(process)
        owner.spawn_creation_failed = True
        if launch_error is not None:
            raise launch_error
        raise RuntimeError("isolated supervisor initialization produced no native child")
    owner.native_child_owned = True
    owner.retain_native_cleanup_session(process)
    try:
        owner.publish_native_handle(process, command)
    finally:
        # Reaping belongs exclusively to the PID-backed worker session.  The
        # disposable Popen wrapper must neither warn nor enter subprocess's
        # competing global reaper after this ownership transfer.
        process._child_created = False  # type: ignore[attr-defined]
    if launch_error is not None:
        raise launch_error


def _startup_wait_interrupted(
    cancel_requested: threading.Event | ProcessCancellation | None,
    abort_requested: threading.Event,
) -> bool:
    return abort_requested.is_set() or (cancel_requested is not None and cancel_requested.is_set())


def _read_supervisor_status_line(
    reader: PrivatePipeEndpoint,
    *,
    deadline: float,
    cancel_requested: threading.Event | ProcessCancellation | None,
    abort_requested: threading.Event,
) -> bytes:
    """Read one exact bounded status line while remaining cancellation-wakeable."""

    descriptor = reader.fileno()
    os.set_blocking(descriptor, False)
    status = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            if _startup_wait_interrupted(cancel_requested, abort_requested):
                raise _StartupWaitInterrupted
            remaining = deadline - time.monotonic()
            wait_seconds = 0.0 if remaining <= 0 else min(remaining, _POLL_SECONDS)
            if not selector.select(wait_seconds):
                if remaining <= 0:
                    raise TimeoutError("isolated supervisor status exceeded its deadline")
                continue
            try:
                chunk = os.read(descriptor, 1)
            except BlockingIOError:
                continue
            if not chunk:
                raise ValueError("isolated supervisor status ended before a complete line")
            status.extend(chunk)
            if chunk == b"\n":
                return bytes(status)
            if len(status) >= _MAX_SUPERVISOR_STATUS_BYTES:
                raise ValueError("invalid isolated supervisor status")


def _status_process_id(status: bytes, *, prefix: bytes) -> int:
    if not status.startswith(prefix) or not status.endswith(b"\n"):
        raise ValueError("invalid isolated supervisor status")
    encoded = status[len(prefix) : -1]
    try:
        process_id = int(encoded)
    except ValueError:
        raise ValueError("invalid isolated supervisor status") from None
    if (
        not encoded
        or not encoded.isascii()
        or not encoded.isdigit()
        or process_id <= 0
        or str(process_id).encode("ascii") != encoded
    ):
        raise ValueError("invalid isolated supervisor status")
    return process_id


def _read_owner_publication(reader: PrivatePipeEndpoint, *, deadline: float) -> int:
    """Require one canonical OWNER frame followed by publication-channel EOF."""

    descriptor = reader.fileno()
    os.set_blocking(descriptor, False)
    payload = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("isolated supervisor ownership publication did not arrive")
            if not selector.select(min(remaining, _POLL_SECONDS)):
                continue
            try:
                chunk = os.read(
                    descriptor,
                    _MAX_SUPERVISOR_OWNER_BYTES + 1 - len(payload),
                )
            except BlockingIOError:
                continue
            if not chunk:
                if payload.count(b"\n") != 1 or not payload.endswith(b"\n"):
                    raise ValueError("invalid isolated supervisor ownership publication")
                return _status_process_id(bytes(payload), prefix=b"OWNER ")
            payload.extend(chunk)
            if len(payload) > _MAX_SUPERVISOR_OWNER_BYTES:
                raise ValueError("invalid isolated supervisor ownership publication")
            if payload.count(b"\n") > 1 or (b"\n" in payload and not payload.endswith(b"\n")):
                raise ValueError("invalid isolated supervisor ownership publication")


def _read_ready_while_lifetime_open(
    status_reader: PrivatePipeEndpoint,
    lifetime_reader: PrivatePipeEndpoint,
    *,
    deadline: float,
    cancel_requested: threading.Event | ProcessCancellation | None,
    abort_requested: threading.Event,
) -> bytes:
    """Read READY while continuously rejecting lifetime data or EOF."""

    status_descriptor = status_reader.fileno()
    lifetime_descriptor = lifetime_reader.fileno()
    os.set_blocking(status_descriptor, False)
    os.set_blocking(lifetime_descriptor, False)
    status = bytearray()

    def inspect_lifetime() -> None:
        try:
            payload = os.read(lifetime_descriptor, 1)
        except BlockingIOError:
            return
        if payload:
            raise ValueError("isolated supervisor lifetime channel carried data")
        raise EOFError("isolated supervisor lifetime ended before START")

    # Establish that the exact sentinel is open before accepting any READY
    # bytes, then monitor the same descriptor for the entire wait.
    inspect_lifetime()
    with selectors.DefaultSelector() as selector:
        selector.register(status_descriptor, selectors.EVENT_READ)
        selector.register(lifetime_descriptor, selectors.EVENT_READ)
        while True:
            if _startup_wait_interrupted(cancel_requested, abort_requested):
                raise _StartupWaitInterrupted
            remaining = deadline - time.monotonic()
            wait_seconds = 0.0 if remaining <= 0 else min(remaining, _POLL_SECONDS)
            events = selector.select(wait_seconds)
            if not events:
                if remaining <= 0:
                    raise TimeoutError("isolated supervisor status exceeded its deadline")
                continue
            if any(key.fd == lifetime_descriptor for key, _mask in events):
                inspect_lifetime()
            if not any(key.fd == status_descriptor for key, _mask in events):
                continue
            try:
                chunk = os.read(status_descriptor, 1)
            except BlockingIOError:
                continue
            if not chunk:
                raise ValueError("isolated supervisor status ended before a complete line")
            status.extend(chunk)
            if chunk == b"\n":
                # A lifetime write racing with the final READY byte must lose
                # admission even if the selector returned only the status FD.
                inspect_lifetime()
                return bytes(status)
            if len(status) >= _MAX_SUPERVISOR_STATUS_BYTES:
                raise ValueError("invalid isolated supervisor status")


def _wait_for_lifetime_eof(
    reader: PrivatePipeEndpoint,
    *,
    deadline: float,
    discard_rejected_data: bool = False,
) -> None:
    """Prove the supervisor released its sentinel on an independent deadline."""

    descriptor = reader.fileno()
    os.set_blocking(descriptor, False)
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("isolated supervisor lifetime did not end")
            if not selector.select(min(remaining, _POLL_SECONDS)):
                continue
            try:
                payload = os.read(descriptor, 1)
            except BlockingIOError:
                continue
            if payload:
                if discard_rejected_data:
                    continue
                raise ValueError("isolated supervisor lifetime channel carried data")
            return


@dataclass(slots=True)
class _StartupOwner:
    """Independently publish and capture a spawn before caller control resumes."""

    process: _SpawnedProcess | None = None
    worker: _WorkerSession | None = None
    native_process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    native_process_id: int | None = None
    native_child_owned: bool = False
    owner_process_id: int | None = None
    ready_process_id: int | None = None
    response_reader: PrivatePipeEndpoint | None = None
    response_writer: PrivatePipeEndpoint | None = None
    status_reader: PrivatePipeEndpoint | None = None
    status_writer: PrivatePipeEndpoint | None = None
    control_reader: PrivatePipeEndpoint | None = None
    control_writer: PrivatePipeEndpoint | None = None
    owner_reader: PrivatePipeEndpoint | None = None
    owner_writer: PrivatePipeEndpoint | None = None
    lifetime_reader: PrivatePipeEndpoint | None = None
    lifetime_writer: PrivatePipeEndpoint | None = None
    start_failed: bool = False
    response_failure: Literal["cancelled", "timeout"] | None = None
    primary_error: BaseException | None = None
    spawn_creation_failed: bool = False
    ownership_failed: bool = False
    lifetime_failed: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    abort_requested: threading.Event = field(default_factory=threading.Event)

    def reserve_native_process(self, process: subprocess.Popen[bytes]) -> None:
        """Retain the exact object before its initializer can create a child."""

        if self.native_process is not None:
            raise RuntimeError("isolated supervisor native handle was reserved more than once")
        self.native_process = process

    def release_native_process(self, process: subprocess.Popen[bytes] | None = None) -> None:
        """Release a blank reservation or a retained handle after cleanup completes."""

        if process is not None and self.native_process is not process:
            raise RuntimeError("isolated supervisor native handle reservation changed")
        self.native_process = None
        self.native_child_owned = False

    def retain_native_cleanup_session(self, process: subprocess.Popen[bytes]) -> None:
        """Publish cleanup ownership before constructing the protocol-facing handle."""

        if self.native_process is not process or not self.native_child_owned:
            raise RuntimeError("isolated supervisor native cleanup authority changed")
        if self.worker is None:
            self.worker = _uncertain_worker_session(process)

    def publish_native_handle(
        self,
        process: subprocess.Popen[bytes],
        command: list[str],
    ) -> None:
        """Publish a PID-backed process before a wrapping caller regains control."""

        if self.process is not None or self.response_reader is None:
            raise RuntimeError("isolated supervisor process was published more than once")
        if self.native_process is not process:
            raise RuntimeError("isolated supervisor native handle publication changed")
        if self.native_process_id is not None:
            raise RuntimeError("isolated supervisor native PID was published more than once")
        self.native_process_id = process.pid
        published_process = _SpawnedProcess(
            process.pid,
            cast(BinaryIO, _EndpointReader(self.response_reader)),
            args=tuple(command),
        )
        reserved_worker = _WorkerSession(
            process=cast("subprocess.Popen[bytes]", published_process),
            process_id=process.pid,
            process_group_id=process.pid,
            session_id=process.pid,
            exit_watcher=None,
            watcher_initialized=False,
            identity_verified=False,
        )
        self.process = published_process
        self.worker = reserved_worker

    def _adopt_process_id(self, process_id: int, command: list[str]) -> None:
        """Retained compatibility seam that can only corroborate native authority."""

        del command
        if (
            self.process is None
            or self.native_process_id is None
            or self.process.pid != self.native_process_id
        ):
            self.ownership_failed = True
        elif self.native_process_id != process_id:
            self.start_failed = True

    def _record_error(self, error: BaseException) -> None:
        if isinstance(error, Exception):
            self.start_failed = True
        elif self.primary_error is None:
            self.primary_error = error

    def _adopt_transport_control(self, transport: _TransportOutcome) -> None:
        """Make a captured startup control primary before any secret delivery."""

        if self.primary_error is None and transport.control_error is not None:
            self.primary_error = transport.control_error

    def _capture_process(self) -> None:
        if self.process is None or self.worker is None:
            return
        try:
            captured = _capture_worker_session(
                cast("subprocess.Popen[bytes]", self.process),
                self.worker,
            )
        except BaseException as exc:
            self._record_error(exc)
        else:
            self.worker = captured
        self.start_failed = (
            self.start_failed
            or not self.worker.watcher_initialized
            or not self.worker.identity_verified
        )

    def _write_control(self, payload: bytes, *, required: bool) -> bool:
        writer = self.control_writer
        if writer is None or writer.closed:
            if required:
                self.start_failed = True
            return False
        try:
            written = os.write(writer.fileno(), payload)
        except Exception:
            if required:
                self.start_failed = True
            return False
        except BaseException as exc:
            if self.primary_error is None:
                self.primary_error = exc
            return False
        if written != len(payload):
            if required:
                self.start_failed = True
            return False
        return True

    def _close_control_writer(self, transport: _TransportOutcome) -> None:
        if self.control_writer is not None:
            _close_endpoint(self.control_writer, transport)
            self._adopt_transport_control(transport)

    def _abort_before_start(self, transport: _TransportOutcome) -> None:
        self._write_control(_ABORT_CONTROL, required=False)
        self._close_control_writer(transport)
        if self.lifetime_reader is None:
            transport.failed = True
            return
        cleanup_deadline = time.monotonic() + _SUPERVISOR_CLEANUP_SECONDS
        try:
            _wait_for_lifetime_eof(
                self.lifetime_reader,
                deadline=cleanup_deadline,
                discard_rejected_data=self.lifetime_failed,
            )
        except BaseException as exc:
            if isinstance(exc, Exception):
                transport.failed = True
            elif self.primary_error is None:
                self.primary_error = exc

    def _run(
        self,
        *,
        command: list[str],
        request_stream: BinaryIO,
        pass_fds: tuple[int, ...],
        private_input: PrivatePipeInput | None,
        cancel_requested: threading.Event | ProcessCancellation | None,
        env: dict[str, str] | None,
        deadline: float,
        transport: _TransportOutcome,
    ) -> None:
        start_signal_sent = False
        try:
            if cancel_requested is not None and cancel_requested.is_set():
                self.response_failure = "cancelled"
            self.response_reader, self.response_writer = _owned_pipe_channel(
                nonblocking_writer=False
            )
            self.status_reader, self.status_writer = _owned_pipe_channel(nonblocking_writer=False)
            self.control_reader, self.control_writer = _owned_pipe_channel(nonblocking_writer=False)
            self.owner_reader, self.owner_writer = _owned_pipe_channel(nonblocking_writer=False)
            self.lifetime_reader, self.lifetime_writer = _owned_pipe_channel(
                nonblocking_writer=False
            )
            try:
                _launch_supervisor(
                    self,
                    command=command,
                    request_stream=request_stream,
                    response_writer=self.response_writer,
                    status_writer=self.status_writer,
                    control_reader=self.control_reader,
                    owner_writer=self.owner_writer,
                    lifetime_writer=self.lifetime_writer,
                    pass_fds=pass_fds,
                    env=env,
                )
            except BaseException as exc:
                self._record_error(exc)
            finally:
                _close_endpoint(self.response_writer, transport)
                _close_endpoint(self.status_writer, transport)
                _close_endpoint(self.control_reader, transport)
                _close_endpoint(self.owner_writer, transport)
                _close_endpoint(self.lifetime_writer, transport)
                self._adopt_transport_control(transport)

            native_authority = (
                self.native_process is not None
                and self.process is not None
                and self.native_process_id is not None
                and self.process.pid == self.native_process_id
            )
            if self.spawn_creation_failed:
                pass
            elif not native_authority:
                # Protocol bytes cannot create cleanup authority.  Without the
                # exact object retained before initialization, cleanup is
                # unprovable even if OWNER or READY advertises a plausible PID.
                self.ownership_failed = True
                self.start_failed = True
                transport.failed = True
            else:
                self._capture_process()
                if self.owner_reader is None:
                    self.ownership_failed = True
                    self.start_failed = True
                else:
                    owner_deadline = time.monotonic() + _SUPERVISOR_CLEANUP_SECONDS
                    try:
                        owner_process_id = _read_owner_publication(
                            self.owner_reader,
                            deadline=owner_deadline,
                        )
                    except BaseException as exc:
                        self.ownership_failed = True
                        self.start_failed = True
                        if not isinstance(exc, Exception) and self.primary_error is None:
                            self.primary_error = exc
                    else:
                        self.owner_process_id = owner_process_id
                        if owner_process_id != self.native_process_id:
                            self.start_failed = True
                if self.lifetime_reader is None:
                    self.lifetime_failed = True
                    self.start_failed = True

            if cancel_requested is not None and cancel_requested.is_set():
                self.response_failure = "cancelled"

            can_read_ready = (
                native_authority
                and self.status_reader is not None
                and self.lifetime_reader is not None
                and self.primary_error is None
                and not transport.failed
                and self.response_failure is None
            )
            if can_read_ready:
                try:
                    ready = _read_ready_while_lifetime_open(
                        self.status_reader,
                        self.lifetime_reader,
                        deadline=deadline,
                        cancel_requested=cancel_requested,
                        abort_requested=self.abort_requested,
                    )
                    published_id = _status_process_id(ready, prefix=b"READY ")
                except _StartupWaitInterrupted:
                    if cancel_requested is not None and cancel_requested.is_set():
                        self.response_failure = "cancelled"
                    else:
                        self.start_failed = True
                except TimeoutError:
                    if self.process is None:
                        self.start_failed = True
                    elif self.response_failure is None:
                        self.response_failure = "timeout"
                except BaseException as exc:
                    self.lifetime_failed = True
                    self._record_error(exc)
                else:
                    self.ready_process_id = published_id
                    if (
                        self.native_process_id is None
                        or self.owner_process_id is None
                        or published_id != self.owner_process_id
                        or published_id != self.native_process_id
                    ):
                        self.start_failed = True

            if private_input is not None:
                try:
                    _release_parent_reader(private_input.reader)
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        transport.failed = True
                    elif self.primary_error is None:
                        self.primary_error = exc

            can_start = (
                self.process is not None
                and self.worker is not None
                and self.primary_error is None
                and not self.start_failed
                and not transport.failed
                and self.response_failure is None
                and self.status_reader is not None
                and self.native_process_id is not None
                and self.owner_process_id == self.native_process_id
                and self.ready_process_id == self.owner_process_id
            )
            if can_start:
                try:
                    _after_worker_spawn(cast("_SpawnedProcess", self.process))
                except BaseException as exc:
                    self._record_error(exc)

            can_start = (
                can_start
                and self.primary_error is None
                and not self.start_failed
                and not transport.failed
            )
            if can_start and self._write_control(_START_CONTROL, required=True):
                start_signal_sent = True
                try:
                    control_status = _read_supervisor_status_line(
                        self.status_reader,
                        deadline=deadline,
                        cancel_requested=cancel_requested,
                        abort_requested=self.abort_requested,
                    )
                    if control_status != b"CONTROL S\n":
                        raise ValueError("invalid isolated supervisor control acknowledgement")
                    self._close_control_writer(transport)
                    if transport.failed or self.primary_error is not None:
                        raise IsolatedProcessCleanupError
                    started = _read_supervisor_status_line(
                        self.status_reader,
                        deadline=deadline,
                        cancel_requested=cancel_requested,
                        abort_requested=self.abort_requested,
                    )
                    _status_process_id(started, prefix=b"STARTED ")
                except _StartupWaitInterrupted:
                    if cancel_requested is not None and cancel_requested.is_set():
                        self.response_failure = "cancelled"
                    else:
                        self.start_failed = True
                except TimeoutError:
                    if self.response_failure is None:
                        self.response_failure = "timeout"
                except BaseException as exc:
                    self._record_error(exc)
            else:
                self.start_failed = self.start_failed or self.response_failure is None

            if not start_signal_sent:
                self._abort_before_start(transport)
            else:
                self._close_control_writer(transport)

            if private_input is not None:
                if (
                    self.process is not None
                    and self.primary_error is None
                    and not self.start_failed
                    and not transport.failed
                ):
                    if cancel_requested is not None and cancel_requested.is_set():
                        self.response_failure = "cancelled"
                    elif time.monotonic() >= deadline:
                        self.response_failure = "timeout"
                    else:
                        try:
                            delivered = send_private_frame(
                                private_input.writer,
                                private_input.payload,
                                max_payload_bytes=private_input.max_payload_bytes,
                                deadline=deadline,
                                cancel_requested=cast(
                                    "ProcessCancellation | None", cancel_requested
                                ),
                            )
                            if not delivered:
                                self.response_failure = "cancelled"
                        except BaseException as exc:
                            if isinstance(exc, Exception):
                                transport.failed = True
                            elif self.primary_error is None:
                                self.primary_error = exc
                _zero_buffer(private_input.payload)
                try:
                    _release_parent_writer(private_input.writer)
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        transport.failed = True
                    elif self.primary_error is None:
                        self.primary_error = exc
        except BaseException as exc:
            self._record_error(exc)
        finally:
            while (
                self.native_child_owned
                and self.native_process is not None
                and self.worker is None
            ):
                try:
                    self.retain_native_cleanup_session(self.native_process)
                except BaseException as exc:
                    self._record_error(exc)
                    try:
                        time.sleep(_POLL_SECONDS)
                    except BaseException as sleep_error:
                        self._record_error(sleep_error)
            if self.response_writer is not None:
                _close_endpoint(self.response_writer, transport)
            if self.status_writer is not None:
                _close_endpoint(self.status_writer, transport)
            if self.status_reader is not None:
                _close_endpoint(self.status_reader, transport)
            if self.control_reader is not None:
                _close_endpoint(self.control_reader, transport)
            if self.control_writer is not None:
                _close_endpoint(self.control_writer, transport)
            if self.owner_writer is not None:
                _close_endpoint(self.owner_writer, transport)
            if self.lifetime_writer is not None:
                _close_endpoint(self.lifetime_writer, transport)
            self.finished.set()

    def spawn(
        self,
        *,
        command: list[str],
        request_stream: BinaryIO,
        pass_fds: tuple[int, ...],
        private_input: PrivatePipeInput | None,
        cancel_requested: threading.Event | ProcessCancellation | None,
        env: dict[str, str] | None,
        deadline: float,
        transport: _TransportOutcome,
    ) -> _SpawnedProcess | None:
        owner_thread = threading.Thread(
            target=self._run,
            kwargs={
                "command": command,
                "request_stream": request_stream,
                "pass_fds": pass_fds,
                "private_input": private_input,
                "cancel_requested": cancel_requested,
                "env": env,
                "deadline": deadline,
                "transport": transport,
            },
            name="isolated-spawn-owner",
            daemon=False,
        )
        caller_control: BaseException | None = None
        try:
            owner_thread.start()
        except BaseException as exc:
            caller_control = exc
            self.abort_requested.set()
            if owner_thread.ident is None:
                self._run(
                    command=command,
                    request_stream=request_stream,
                    pass_fds=pass_fds,
                    private_input=private_input,
                    cancel_requested=cancel_requested,
                    env=env,
                    deadline=deadline,
                    transport=transport,
                )
        while owner_thread.is_alive():
            try:
                owner_thread.join(_POLL_SECONDS)
            except BaseException as exc:
                if caller_control is None:
                    caller_control = exc
                self.abort_requested.set()
        if not self.finished.is_set():
            transport.failed = True
        if self.primary_error is None:
            self.primary_error = caller_control
        return self.process


def _reserve_private_endpoint_capacity(count: int) -> None:
    global _ACTIVE_PRIVATE_PIPE_ENDPOINTS

    with _PRIVATE_PIPE_STATE_LOCK:
        if (
            _PRIVATE_PIPE_STATE_POISONED
            or _ACTIVE_PRIVATE_PIPE_ENDPOINTS + count > _MAX_PRIVATE_PIPE_ENDPOINTS
        ):
            raise IsolatedProcessCleanupError
        _ACTIVE_PRIVATE_PIPE_ENDPOINTS += count


def _poison_private_endpoint_state() -> None:
    """Fail closed after an ambiguous raw-descriptor close result."""

    global _PRIVATE_PIPE_STATE_POISONED

    with _PRIVATE_PIPE_STATE_LOCK:
        _PRIVATE_PIPE_STATE_POISONED = True


def _release_private_endpoint_capacity(count: int) -> None:
    global _ACTIVE_PRIVATE_PIPE_ENDPOINTS

    with _PRIVATE_PIPE_STATE_LOCK:
        _ACTIVE_PRIVATE_PIPE_ENDPOINTS -= count


def _release_private_endpoint(endpoint: PrivatePipeEndpoint) -> None:
    global _ACTIVE_PRIVATE_PIPE_ENDPOINTS

    with _PRIVATE_PIPE_STATE_LOCK:
        if not endpoint._managed:
            return
        endpoint._managed = False
        _ACTIVE_PRIVATE_PIPE_ENDPOINTS -= 1


def _retire_unmanaged_private_descriptors(*descriptors: int) -> BaseException | None:
    """Attempt every raw close once and fail closed after any ambiguity."""

    first_error: BaseException | None = None
    for descriptor in descriptors:
        if descriptor < 0:
            continue
        try:
            os.close(descriptor)
        except BaseException as exc:
            _poison_private_endpoint_state()
            if first_error is None:
                first_error = exc
    return first_error


def _owned_pipe_channel(
    *,
    nonblocking_writer: bool,
) -> tuple[PrivatePipeEndpoint, PrivatePipeEndpoint]:
    """Create one managed directional pipe without retaining failed raw numbers."""

    _reserve_private_endpoint_capacity(2)
    reader_descriptor = -1
    writer_descriptor = -1
    try:
        reader_descriptor, writer_descriptor = os.pipe()
        reader = PrivatePipeEndpoint(reader_descriptor, readable=True, writable=False)
        writer = PrivatePipeEndpoint(writer_descriptor, readable=False, writable=True)
    except BaseException as construction_error:
        cleanup_error = _retire_unmanaged_private_descriptors(
            reader_descriptor,
            writer_descriptor,
        )
        _release_private_endpoint_capacity(2)
        if cleanup_error is not None:
            raise cleanup_error from construction_error
        raise
    reader._managed = True
    writer._managed = True
    try:
        os.set_blocking(writer_descriptor, not nonblocking_writer)
    except BaseException:
        with suppress(BaseException):
            reader.close()
        with suppress(BaseException):
            writer.close()
        raise
    return reader, writer


def private_pipe_channel() -> tuple[PrivatePipeEndpoint, PrivatePipeEndpoint]:
    """Create one anonymous read-only/write-only credential pipe pair."""

    return _owned_pipe_channel(nonblocking_writer=True)


def _private_pipe_duplicate(
    descriptor: int,
    *,
    readable: bool,
    writable: bool,
) -> PrivatePipeEndpoint:
    """Create one separately owned inheritance alias for a spawn file action."""

    _reserve_private_endpoint_capacity(1)
    duplicate = -1
    try:
        duplicate = os.dup(descriptor)
        endpoint = PrivatePipeEndpoint(duplicate, readable=readable, writable=writable)
    except BaseException as construction_error:
        cleanup_error = _retire_unmanaged_private_descriptors(duplicate)
        _release_private_endpoint_capacity(1)
        if cleanup_error is not None:
            raise cleanup_error from construction_error
        raise
    endpoint._managed = True
    return endpoint


def _zero_buffer(buffer: bytearray) -> None:
    for index in range(len(buffer)):
        buffer[index] = 0


def _close_endpoint(
    endpoint: PrivatePipeEndpoint,
    outcome: _TransportOutcome,
) -> None:
    if endpoint.closed:
        return
    try:
        endpoint.close()
    except Exception:
        outcome.failed = True
    except BaseException as exc:
        outcome.capture_control(exc)


def _release_parent_reader(reader: PrivatePipeEndpoint) -> None:
    """Fault-visible pre-write release seam; cleanup owns an unattempted endpoint."""

    reader.close()


def _release_parent_writer(writer: PrivatePipeEndpoint) -> None:
    """Fault-visible post-write release seam; cleanup owns an unattempted endpoint."""

    writer.close()


def _write_buffer(
    descriptor: int,
    buffer: bytearray,
    *,
    deadline: float,
    selector: selectors.BaseSelector,
    cancel_requested: ProcessCancellation | None = None,
) -> bool:
    view = memoryview(buffer)
    offset = 0
    try:
        while offset < len(view):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("private pipe write exceeded its deadline")
            written: int | None
            try:
                if cancel_requested is None:
                    written = os.write(descriptor, view[offset:])
                else:
                    written = None

                    def write_once(write_offset: int = offset) -> None:
                        nonlocal written
                        written = os.write(descriptor, view[write_offset:])

                    if not cancel_requested._deliver_if_not_cancelled(write_once):
                        return False
                    if written is None:
                        raise RuntimeError("private pipe write produced no result")
            except BlockingIOError:
                if not selector.select(min(remaining, _POLL_SECONDS)):
                    continue
                continue
            if written <= 0 or written > len(view) - offset:
                raise OSError("private pipe write was incomplete")
            offset += written
        return True
    finally:
        view.release()


def send_private_frame(
    writer: PrivatePipeEndpoint,
    credential: bytearray,
    *,
    max_payload_bytes: int,
    deadline: float,
    cancel_requested: ProcessCancellation | None = None,
) -> bool:
    """Write one exact bounded frame without creating a readable parent endpoint."""

    if (
        writer.closed
        or writer.readable
        or not writer.writable
        or not credential
        or len(credential) > max_payload_bytes
    ):
        raise ValueError("invalid private pipe frame")
    header = bytearray(_CREDENTIAL_FRAME_HEADER_BYTES)
    struct.pack_into("!I", header, 0, len(credential))
    try:
        descriptor = writer.fileno()

        def deliver() -> None:
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_WRITE)
                _write_buffer(descriptor, header, deadline=deadline, selector=selector)
                _write_buffer(descriptor, credential, deadline=deadline, selector=selector)

        if cancel_requested is None:
            deliver()
            return True
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_WRITE)
            if not _write_buffer(
                descriptor,
                header,
                deadline=deadline,
                selector=selector,
                cancel_requested=cancel_requested,
            ):
                return False
            return _write_buffer(
                descriptor,
                credential,
                deadline=deadline,
                selector=selector,
                cancel_requested=cancel_requested,
            )
    finally:
        _zero_buffer(header)


def _valid_private_input(
    private_input: PrivatePipeInput | None,
    pass_fds: tuple[int, ...],
) -> bool:
    if private_input is None:
        return not pass_fds
    try:
        reader_descriptor = private_input.reader.fileno()
        writer_descriptor = private_input.writer.fileno()
    except ValueError:
        return False
    return (
        private_input.reader.readable
        and not private_input.reader.writable
        and not private_input.writer.readable
        and private_input.writer.writable
        and reader_descriptor >= 3
        and writer_descriptor >= 3
        and reader_descriptor != writer_descriptor
        and pass_fds == (reader_descriptor,)
        and type(private_input.max_payload_bytes) is int
        and private_input.max_payload_bytes > 0
        and type(private_input.payload) is bytearray
        and 0 < len(private_input.payload) <= private_input.max_payload_bytes
    )


def _retire_private_input(
    private_input: PrivatePipeInput | None,
    outcome: _TransportOutcome,
) -> None:
    if private_input is None:
        return
    _zero_buffer(private_input.payload)
    _close_endpoint(private_input.reader, outcome)
    _close_endpoint(private_input.writer, outcome)


@dataclass(frozen=True, slots=True)
class IsolatedProcessResult:
    """Sanitized bytes observed only after the worker session is proven empty."""

    response: bytes | None
    failure: Literal["start", "timeout", "cancelled", "protocol", "crash"] | None


def _run_worker_cleanup_phase(
    worker: _WorkerSession,
    outcome: _WorkerCleanupOutcome,
) -> None:
    """Synchronously empty and reap the exact reserved worker session."""

    try:
        _cleanup_cooperative_worker_session(worker)
    except Exception:
        outcome.failed = True
    except BaseException as exc:
        outcome.capture_control(exc)
    outcome.contained = worker.cleaned
    outcome.reaped = worker.cleaned
    if not worker.cleaned:
        outcome.failed = True


def _stop_worker_synchronously(worker: _WorkerSession) -> _WorkerCleanupOutcome:
    """Join exact session cleanup even when the calling thread is interrupted."""

    outcome = _WorkerCleanupOutcome()
    finished = threading.Event()

    def cleanup_target() -> None:
        try:
            _run_worker_cleanup_phase(worker, outcome)
        except Exception:
            outcome.failed = True
        except BaseException as exc:
            outcome.capture_control(exc)
        finally:
            outcome.contained = worker.cleaned
            outcome.reaped = worker.cleaned
            if not outcome.contained:
                outcome.failed = True
            finished.set()

    cleanup_thread = threading.Thread(
        target=cleanup_target,
        name="isolated-worker-containment",
        daemon=False,
    )
    wait_control: BaseException | None = None
    try:
        cleanup_thread.start()
    except Exception:
        if cleanup_thread.ident is None:
            cleanup_target()
    except BaseException as exc:
        wait_control = exc
        if cleanup_thread.ident is None:
            cleanup_target()
    while cleanup_thread.is_alive():
        try:
            cleanup_thread.join(_POLL_SECONDS)
        except BaseException as exc:
            if wait_control is None:
                wait_control = exc
    if not finished.is_set():
        outcome.failed = True
    if outcome.control_error is None:
        outcome.control_error = wait_control
    outcome.contained = worker.cleaned
    outcome.reaped = worker.cleaned
    if not outcome.contained:
        outcome.failed = True
    return outcome


def _read_response(
    worker: _WorkerSession,
    *,
    deadline: float,
    max_response_bytes: int,
    cancel_requested: threading.Event | ProcessCancellation | None,
) -> tuple[bytes | None, Literal["timeout", "cancelled", "protocol"] | None]:
    process = worker.process
    stdout = process.stdout
    watcher = worker.exit_watcher
    if stdout is None or watcher is None:
        return None, "protocol"
    descriptor = stdout.fileno()
    os.set_blocking(descriptor, False)
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
            except BlockingIOError:
                continue
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
    cancel_requested: threading.Event | ProcessCancellation | None = None,
    pass_fds: tuple[int, ...] = (),
    private_input: PrivatePipeInput | None = None,
    env: dict[str, str] | None = None,
) -> IsolatedProcessResult:
    """Spawn one fresh session and retire it before returning.

    When ``private_input`` is present, the child inherits only its read-only
    pipe endpoint.  The parent closes that endpoint after spawn and writes a
    single bounded frame only after the close and worker capture both succeed.
    """

    transport = _TransportOutcome()
    inputs_are_valid = (
        type(timeout_seconds) is float
        and timeout_seconds > 0
        and type(max_response_bytes) is int
        and max_response_bytes > 0
        and bool(request)
        and type(command) is list
        and bool(command)
        and all(type(argument) is str and bool(argument) for argument in command)
        and all(type(descriptor) is int and descriptor >= 3 for descriptor in pass_fds)
        and len(set(pass_fds)) == len(pass_fds)
        and _valid_private_input(private_input, pass_fds)
        and (
            private_input is None
            or cancel_requested is None
            or isinstance(cancel_requested, ProcessCancellation)
        )
    )
    if not inputs_are_valid:
        _retire_private_input(private_input, transport)
        if transport.failed:
            raise IsolatedProcessCleanupError from None
        if transport.control_error is not None:
            raise transport.control_error
        raise ValueError("invalid isolated worker controller input")

    if _worker_resource_cleanup_is_poisoned():
        _retire_private_input(private_input, transport)
        raise IsolatedProcessCleanupError from None

    if cancel_requested is not None and cancel_requested.is_set():
        _retire_private_input(private_input, transport)
        if transport.failed:
            raise IsolatedProcessCleanupError from None
        if transport.control_error is not None:
            raise transport.control_error
        return IsolatedProcessResult(None, "cancelled")

    deadline = time.monotonic() + timeout_seconds
    process: _SpawnedProcess | None = None
    worker: _WorkerSession | None = None
    start_failed = False
    primary_error: BaseException | None = None
    worker_cleanup = _WorkerCleanupOutcome(contained=True, reaped=True)
    response: bytes | None = None
    response_failure: Literal["timeout", "cancelled", "protocol"] | None = None
    startup_owner = _StartupOwner()
    try:
        try:
            with tempfile.TemporaryFile() as request_stream:
                request_stream.write(request)
                request_stream.seek(0)
                process = startup_owner.spawn(
                    command=command,
                    request_stream=request_stream,
                    pass_fds=pass_fds,
                    private_input=private_input,
                    cancel_requested=cancel_requested,
                    env=env,
                    deadline=deadline,
                    transport=transport,
                )
        except Exception:
            start_failed = True
        except BaseException as exc:
            primary_error = exc
        if process is None:
            process = startup_owner.process
        worker = startup_owner.worker
        start_failed = start_failed or startup_owner.start_failed
        if primary_error is None:
            primary_error = startup_owner.primary_error
        response_failure = startup_owner.response_failure

        if response_failure is None and time.monotonic() >= deadline and process is not None:
            response_failure = "timeout"

        if (
            process is not None
            and worker is not None
            and primary_error is None
            and not start_failed
            and not transport.failed
            and response_failure is None
        ):
            try:
                response, response_failure = _read_response(
                    worker,
                    deadline=deadline,
                    max_response_bytes=max_response_bytes,
                    cancel_requested=cancel_requested,
                )
            except BaseException as exc:
                primary_error = exc
    finally:
        if worker is not None:
            try:
                worker_cleanup = _stop_worker_synchronously(worker)
            finally:
                startup_owner.release_native_process()
        else:
            worker_cleanup.contained = process is None
            worker_cleanup.reaped = process is None
            worker_cleanup.failed = process is not None
            startup_owner.release_native_process()
        if startup_owner.response_reader is not None:
            _close_endpoint(startup_owner.response_reader, transport)
        if startup_owner.response_writer is not None:
            _close_endpoint(startup_owner.response_writer, transport)
        if startup_owner.status_reader is not None:
            _close_endpoint(startup_owner.status_reader, transport)
        if startup_owner.status_writer is not None:
            _close_endpoint(startup_owner.status_writer, transport)
        if startup_owner.control_reader is not None:
            _close_endpoint(startup_owner.control_reader, transport)
        if startup_owner.control_writer is not None:
            _close_endpoint(startup_owner.control_writer, transport)
        if startup_owner.owner_reader is not None:
            _close_endpoint(startup_owner.owner_reader, transport)
        if startup_owner.owner_writer is not None:
            _close_endpoint(startup_owner.owner_writer, transport)
        if startup_owner.lifetime_reader is not None:
            _close_endpoint(startup_owner.lifetime_reader, transport)
        if startup_owner.lifetime_writer is not None:
            _close_endpoint(startup_owner.lifetime_writer, transport)
        _retire_private_input(private_input, transport)

    if transport.failed or worker_cleanup.failed or not worker_cleanup.contained:
        raise IsolatedProcessCleanupError from None
    if primary_error is not None:
        raise primary_error
    if worker_cleanup.control_error is not None:
        raise worker_cleanup.control_error
    if transport.control_error is not None:
        raise transport.control_error
    if worker is None:
        if start_failed:
            return IsolatedProcessResult(None, "start")
        raise IsolatedProcessCleanupError from None
    if response_failure is not None:
        return IsolatedProcessResult(None, response_failure)
    if start_failed:
        return IsolatedProcessResult(None, "start")
    if process is None or process.returncode != 0:
        return IsolatedProcessResult(None, "crash")
    return IsolatedProcessResult(response, None)
