"""Shared bounded process/session controller for case workers."""

from __future__ import annotations

import os
import selectors
import struct
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Literal

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


def sanitized_worker_environment() -> dict[str, str]:
    """Compatibility export for worker callers using the shared controller."""

    return _sanitized_worker_environment()


@dataclass(slots=True)
class PrivatePipeEndpoint:
    """Own exactly one directional anonymous-pipe descriptor.

    The descriptor number is forgotten before ``close(2)`` is attempted.  An
    interrupted or failed close is therefore never retried against a number
    that another thread could have reused.
    """

    _descriptor: int | None
    readable: bool
    writable: bool

    def fileno(self) -> int:
        descriptor = self._descriptor
        if descriptor is None:
            raise ValueError("private pipe endpoint is closed")
        return descriptor

    @property
    def closed(self) -> bool:
        return self._descriptor is None

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        os.close(descriptor)

    def detach(self) -> int:
        """Transfer ownership for focused child-reader tests."""

        descriptor = self.fileno()
        self._descriptor = None
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


def private_pipe_channel() -> tuple[PrivatePipeEndpoint, PrivatePipeEndpoint]:
    """Create one anonymous read-only/write-only pipe pair."""

    reader_descriptor, writer_descriptor = os.pipe()
    reader = PrivatePipeEndpoint(reader_descriptor, readable=True, writable=False)
    writer = PrivatePipeEndpoint(writer_descriptor, readable=False, writable=True)
    try:
        os.set_blocking(writer_descriptor, False)
    except BaseException:
        with suppress(BaseException):
            reader.close()
        with suppress(BaseException):
            writer.close()
        raise
    return reader, writer


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
) -> None:
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
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_WRITE)
            _write_buffer(descriptor, header, deadline=deadline, selector=selector)
            _write_buffer(descriptor, credential, deadline=deadline, selector=selector)
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


class IsolatedProcessCleanupError(Exception):
    """The worker session or one-way private transport was not cleanly retired."""


def _fallback_worker_session(process: subprocess.Popen[bytes]) -> _WorkerSession:
    """Track the reserved start-new-session identity without a fallible syscall."""

    return _WorkerSession(
        process=process,
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
    cancel_requested: threading.Event | None,
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
    cancel_requested: threading.Event | None = None,
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
    process: subprocess.Popen[bytes] | None = None
    worker: _WorkerSession | None = None
    start_failed = False
    primary_error: BaseException | None = None
    worker_cleanup = _WorkerCleanupOutcome(contained=True, reaped=True)
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
            # This assignment is the immediate, non-fallible ownership record for
            # cleanup if close/capture/control flow is interrupted next.
            worker = _fallback_worker_session(process)

            if private_input is not None:
                try:
                    _release_parent_reader(private_input.reader)
                except Exception:
                    transport.failed = True
                except BaseException as exc:
                    primary_error = exc

            if primary_error is None and not start_failed and not transport.failed:
                try:
                    worker = _capture_worker_session(process)
                except Exception:
                    start_failed = True
                except BaseException as exc:
                    primary_error = exc

            if worker is not None:
                start_failed = (
                    start_failed or not worker.watcher_initialized or not worker.identity_verified
                )

            if (
                primary_error is None
                and not start_failed
                and not transport.failed
                and private_input is not None
            ):
                if cancel_requested is not None and cancel_requested.is_set():
                    response_failure = "cancelled"
                elif time.monotonic() >= deadline:
                    response_failure = "timeout"
                else:
                    try:
                        send_private_frame(
                            private_input.writer,
                            private_input.payload,
                            max_payload_bytes=private_input.max_payload_bytes,
                            deadline=deadline,
                        )
                    except Exception:
                        transport.failed = True
                    except BaseException as exc:
                        primary_error = exc
                    finally:
                        _zero_buffer(private_input.payload)
                        try:
                            _release_parent_writer(private_input.writer)
                        except Exception:
                            transport.failed = True
                        except BaseException as exc:
                            if primary_error is None:
                                primary_error = exc

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
    if start_failed:
        return IsolatedProcessResult(None, "start")
    if response_failure is not None:
        return IsolatedProcessResult(None, response_failure)
    if process is None or process.returncode != 0:
        return IsolatedProcessResult(None, "crash")
    return IsolatedProcessResult(response, None)
