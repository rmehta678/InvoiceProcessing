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
from typing import BinaryIO, Literal, cast

from invoice_agents.db.migration_process import (
    _QUARANTINED_WORKERS,
    _QUARANTINED_WORKERS_LOCK,
    _capture_worker_session,
    _cleanup_worker_session,
    _quarantine_worker,
    _remove_quarantined_worker,
    _retry_quarantined_workers,
    _stop_worker,
    _WorkerSession,
)
from invoice_agents.worker_environment import (
    sanitized_worker_environment as _sanitized_worker_environment,
)

_CREDENTIAL_FRAME_HEADER_BYTES = 4
_POLL_SECONDS = 0.02
_MAX_PRIVATE_PIPE_ENDPOINTS = 128


class IsolatedProcessCleanupError(Exception):
    """The worker session or one-way private transport was not cleanly retired."""


_PRIVATE_PIPE_STATE_LOCK = threading.Lock()
_ACTIVE_PRIVATE_PIPE_ENDPOINTS = 0


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
        """Order cancellation wholly before or after one credential delivery."""

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
            os.close(descriptor)

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
    """Independent proof that a worker is reaped or durably quarantined."""

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


def _supervisor_command(publication_descriptor: int, command: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "invoice_agents.spawn_supervisor",
        str(publication_descriptor),
        *command,
    ]


def _launch_supervisor(
    owner: _StartupOwner,
    *,
    command: list[str],
    request_stream: BinaryIO,
    response_writer: PrivatePipeEndpoint,
    publication_writer: PrivatePipeEndpoint,
    pass_fds: tuple[int, ...],
    env: dict[str, str] | None,
) -> None:
    """Launch through the close-all supervisor and publish its native handle immediately."""

    publication_descriptor = publication_writer.fileno()
    inherited = (*pass_fds, publication_descriptor)
    process = subprocess.Popen(
        _supervisor_command(publication_descriptor, command),
        stdin=request_stream,
        stdout=response_writer.fileno(),
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        pass_fds=inherited,
        env=env,
    )
    owner.publish_native_handle(process, command)


def _read_spawn_publication(
    reader: PrivatePipeEndpoint,
    *,
    deadline: float,
) -> int:
    """Read the supervisor's exact PID publication through one bounded pipe."""

    descriptor = reader.fileno()
    os.set_blocking(descriptor, False)
    published = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            wait_seconds = 0.0 if remaining <= 0 else min(remaining, _POLL_SECONDS)
            if not selector.select(wait_seconds):
                if remaining <= 0:
                    raise TimeoutError("isolated supervisor did not publish its process identity")
                continue
            chunk = os.read(descriptor, 32 - len(published))
            if not chunk:
                break
            published.extend(chunk)
            if len(published) >= 32:
                raise ValueError("invalid isolated supervisor process publication")
    try:
        encoded = bytes(published)
        if not encoded.endswith(b"\n") or not encoded[:-1].isascii() or not encoded[:-1].isdigit():
            raise ValueError("invalid isolated supervisor process publication")
        process_id = int(encoded[:-1])
    finally:
        _zero_buffer(published)
    if process_id <= 0 or str(process_id).encode("ascii") != encoded[:-1]:
        raise ValueError("invalid isolated supervisor process publication")
    return process_id


@dataclass(slots=True)
class _StartupOwner:
    """Independently publish and capture a spawn before caller control resumes."""

    process: _SpawnedProcess | None = None
    worker: _WorkerSession | None = None
    response_reader: PrivatePipeEndpoint | None = None
    response_writer: PrivatePipeEndpoint | None = None
    publication_reader: PrivatePipeEndpoint | None = None
    publication_writer: PrivatePipeEndpoint | None = None
    start_failed: bool = False
    response_failure: Literal["cancelled", "timeout"] | None = None
    primary_error: BaseException | None = None
    finished: threading.Event = field(default_factory=threading.Event)

    def publish_native_handle(
        self,
        process: subprocess.Popen[bytes],
        command: list[str],
    ) -> None:
        """Publish the PID and handle before a wrapping caller regains control."""

        if self.process is not None or self.response_reader is None:
            raise RuntimeError("isolated supervisor process was published more than once")
        self.process = _SpawnedProcess(
            process.pid,
            cast(BinaryIO, _EndpointReader(self.response_reader)),
            args=tuple(command),
            _native_handle=process,
        )

    def _record_error(self, error: BaseException) -> None:
        if isinstance(error, Exception):
            self.start_failed = True
        elif self.primary_error is None:
            self.primary_error = error

    def _adopt_transport_control(self, transport: _TransportOutcome) -> None:
        """Make a captured startup control primary before any secret delivery."""

        if self.primary_error is None and transport.control_error is not None:
            self.primary_error = transport.control_error

    def _run(
        self,
        *,
        command: list[str],
        request_stream: BinaryIO,
        pass_fds: tuple[int, ...],
        private_input: PrivatePipeInput | None,
        cancel_requested: ProcessCancellation | None,
        env: dict[str, str] | None,
        deadline: float,
        transport: _TransportOutcome,
    ) -> None:
        try:
            if cancel_requested is not None and cancel_requested.is_set():
                self.response_failure = "cancelled"
            self.response_reader, self.response_writer = _owned_pipe_channel(
                nonblocking_writer=False
            )
            self.publication_reader, self.publication_writer = _owned_pipe_channel(
                nonblocking_writer=False
            )
            try:
                _launch_supervisor(
                    self,
                    command=command,
                    request_stream=request_stream,
                    response_writer=self.response_writer,
                    publication_writer=self.publication_writer,
                    pass_fds=pass_fds,
                    env=env,
                )
            except BaseException as exc:
                self._record_error(exc)
            finally:
                _close_endpoint(self.response_writer, transport)
                _close_endpoint(self.publication_writer, transport)
                self._adopt_transport_control(transport)

            try:
                published_id = _read_spawn_publication(
                    self.publication_reader,
                    deadline=deadline,
                )
            except TimeoutError:
                if self.process is None:
                    self.start_failed = True
                elif self.response_failure is None:
                    self.response_failure = "timeout"
            except BaseException as exc:
                self._record_error(exc)
            else:
                if self.process is None:
                    self.process = _SpawnedProcess(
                        published_id,
                        cast(BinaryIO, _EndpointReader(self.response_reader)),
                        args=tuple(command),
                    )
                elif self.process.pid != published_id:
                    self.start_failed = True

            if self.process is not None:
                self.worker = _fallback_worker_session(self.process)
                if self.primary_error is None and not self.start_failed and not transport.failed:
                    try:
                        self.worker = _capture_worker_session(
                            cast("subprocess.Popen[bytes]", self.process)
                        )
                    except BaseException as exc:
                        self._record_error(exc)
                if self.worker is not None:
                    self.start_failed = (
                        self.start_failed
                        or not self.worker.watcher_initialized
                        or not self.worker.identity_verified
                    )
                if (
                    self.primary_error is None
                    and not self.start_failed
                    and not transport.failed
                    and self.response_failure is None
                ):
                    try:
                        _after_worker_spawn(self.process)
                    except BaseException as exc:
                        self._record_error(exc)

            if private_input is not None:
                try:
                    _release_parent_reader(private_input.reader)
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        transport.failed = True
                    elif self.primary_error is None:
                        self.primary_error = exc

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
                                cancel_requested=cancel_requested,
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
            if self.response_writer is not None:
                _close_endpoint(self.response_writer, transport)
            if self.publication_writer is not None:
                _close_endpoint(self.publication_writer, transport)
            if self.publication_reader is not None:
                _close_endpoint(self.publication_reader, transport)
            self.finished.set()

    def spawn(
        self,
        *,
        command: list[str],
        request_stream: BinaryIO,
        pass_fds: tuple[int, ...],
        private_input: PrivatePipeInput | None,
        cancel_requested: ProcessCancellation | None,
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
            daemon=True,
        )
        caller_control: BaseException | None = None
        try:
            owner_thread.start()
        except BaseException as exc:
            caller_control = exc
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
        while not self.finished.is_set():
            try:
                self.finished.wait(_POLL_SECONDS)
            except BaseException as exc:
                if caller_control is None:
                    caller_control = exc
        if self.primary_error is None:
            self.primary_error = caller_control
        return self.process


def _reserve_private_endpoint_capacity(count: int) -> None:
    global _ACTIVE_PRIVATE_PIPE_ENDPOINTS

    with _PRIVATE_PIPE_STATE_LOCK:
        if _ACTIVE_PRIVATE_PIPE_ENDPOINTS + count > _MAX_PRIVATE_PIPE_ENDPOINTS:
            raise IsolatedProcessCleanupError
        _ACTIVE_PRIVATE_PIPE_ENDPOINTS += count


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
    except BaseException:
        if reader_descriptor >= 0:
            with suppress(BaseException):
                os.close(reader_descriptor)
        if writer_descriptor >= 0:
            with suppress(BaseException):
                os.close(writer_descriptor)
        _release_private_endpoint_capacity(2)
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
    except BaseException:
        if duplicate >= 0:
            with suppress(BaseException):
                os.close(duplicate)
        _release_private_endpoint_capacity(1)
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
) -> None:
    view = memoryview(buffer)
    offset = 0
    try:
        while offset < len(view):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("private pipe write exceeded its deadline")
            try:
                written = os.write(descriptor, view[offset:])
            except BlockingIOError:
                if not selector.select(min(remaining, _POLL_SECONDS)):
                    continue
                continue
            if written <= 0 or written > len(view) - offset:
                raise OSError("private pipe write was incomplete")
            offset += written
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
        return cancel_requested._deliver_if_not_cancelled(deliver)
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


def _fallback_worker_session(
    process: subprocess.Popen[bytes] | _SpawnedProcess,
) -> _WorkerSession:
    """Track the reserved start-new-session identity without a fallible syscall."""

    return _WorkerSession(
        process=cast("subprocess.Popen[bytes]", process),
        process_id=process.pid,
        process_group_id=process.pid,
        session_id=process.pid,
        exit_watcher=None,
        watcher_initialized=False,
        identity_verified=False,
    )


def _worker_containment_is_proven(worker: _WorkerSession) -> tuple[bool, bool]:
    if worker.cleaned:
        return True, True
    with _QUARANTINED_WORKERS_LOCK:
        quarantined = _QUARANTINED_WORKERS.get(worker.process_id) is worker
    return quarantined, False


def _run_worker_cleanup_phase(
    worker: _WorkerSession,
    outcome: _WorkerCleanupOutcome,
) -> None:
    """Stop through the public seam, then independently reap or quarantine."""

    stop_failed = False
    try:
        cleanup_error = _stop_worker(worker)
    except Exception:
        stop_failed = True
    except BaseException as exc:
        outcome.capture_control(exc)
    else:
        stop_failed = cleanup_error is not None

    contained, _reaped = _worker_containment_is_proven(worker)
    if not contained:
        try:
            _cleanup_worker_session(worker)
        except Exception:
            stop_failed = True
        except BaseException as exc:
            outcome.capture_control(exc)
        else:
            _remove_quarantined_worker(worker)

    contained, _reaped = _worker_containment_is_proven(worker)
    if not contained:
        try:
            _quarantine_worker(worker)
        except Exception:
            stop_failed = True
        except BaseException as exc:
            outcome.capture_control(exc)

    outcome.contained, outcome.reaped = _worker_containment_is_proven(worker)
    outcome.failed = not outcome.contained or (stop_failed and outcome.control_error is None)


def _stop_or_quarantine_worker(worker: _WorkerSession) -> _WorkerCleanupOutcome:
    """Run containment independently and wait through caller cancellation."""

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
            contained, _reaped = _worker_containment_is_proven(worker)
            if not contained:
                try:
                    _quarantine_worker(worker)
                except Exception:
                    outcome.failed = True
                except BaseException as exc:
                    outcome.capture_control(exc)
            outcome.contained, outcome.reaped = _worker_containment_is_proven(worker)
            if not outcome.contained:
                outcome.failed = True
            finished.set()

    cleanup_thread = threading.Thread(
        target=cleanup_target,
        name="isolated-worker-containment",
        daemon=True,
    )
    wait_control: BaseException | None = None
    try:
        cleanup_thread.start()
    except Exception:
        cleanup_target()
    except BaseException as exc:
        wait_control = exc
        if cleanup_thread.ident is None:
            cleanup_target()
    while not finished.is_set():
        try:
            finished.wait(_POLL_SECONDS)
        except BaseException as exc:
            if wait_control is None:
                wait_control = exc
    if outcome.control_error is None:
        outcome.control_error = wait_control
    outcome.contained, outcome.reaped = _worker_containment_is_proven(worker)
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

    try:
        workers_clear = _retry_quarantined_workers()
    except Exception:
        workers_clear = False
    except BaseException as exc:
        _retire_private_input(private_input, transport)
        if transport.failed:
            raise IsolatedProcessCleanupError from None
        raise exc
    if not workers_clear:
        _retire_private_input(private_input, transport)
        raise IsolatedProcessCleanupError from None

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
                    cancel_requested=cast("ProcessCancellation | None", cancel_requested),
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

        if process is not None:
            if worker is None:
                worker = _fallback_worker_session(process)

            if (
                primary_error is None
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
        if worker is None and process is not None:
            worker = _fallback_worker_session(process)
        if worker is not None:
            worker_cleanup = _stop_or_quarantine_worker(worker)
        else:
            worker_cleanup.contained = process is None
            worker_cleanup.reaped = process is None
            worker_cleanup.failed = process is not None
        if startup_owner.response_reader is not None:
            _close_endpoint(startup_owner.response_reader, transport)
        if startup_owner.response_writer is not None:
            _close_endpoint(startup_owner.response_writer, transport)
        if startup_owner.publication_reader is not None:
            _close_endpoint(startup_owner.publication_reader, transport)
        if startup_owner.publication_writer is not None:
            _close_endpoint(startup_owner.publication_writer, transport)
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
