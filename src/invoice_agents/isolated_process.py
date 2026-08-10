"""Shared bounded process/session controller for case workers."""

from __future__ import annotations

import os
import re
import selectors
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

_POLL_SECONDS = 0.02
_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?i)(?:api[_-]?key|authorization|credential|secret|token|cookie)"
)


def sanitized_worker_environment() -> dict[str, str]:
    """Copy only non-credential ambient variables into a case worker."""

    return {
        key: value
        for key, value in os.environ.items()
        if _SENSITIVE_ENVIRONMENT_NAME.search(key) is None
    }


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
    """Spawn one fresh session, then stop/reap all members before returning."""

    if (
        type(timeout_seconds) is not float
        or timeout_seconds <= 0
        or type(max_response_bytes) is not int
        or max_response_bytes <= 0
        or not request
        or type(command) is not list
        or not command
        or any(type(argument) is not str or not argument for argument in command)
        or any(type(descriptor) is not int or descriptor < 0 for descriptor in pass_fds)
    ):
        raise ValueError("invalid isolated worker controller input")
    if not _retry_quarantined_workers():
        raise IsolatedProcessCleanupError from None
    process: subprocess.Popen[bytes] | None = None
    worker: object | None = None
    start_failed = False
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
        worker = _capture_worker_session(process)
    except Exception:
        start_failed = True
        if process is not None:
            worker = _uncertain_worker_session(process)
    if worker is None:
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
    if cleanup_error is not None:
        raise IsolatedProcessCleanupError from None
    if start_failed:
        return IsolatedProcessResult(None, "start")
    if response_failure is not None:
        return IsolatedProcessResult(None, response_failure)
    if process is None or process.returncode != 0:
        return IsolatedProcessResult(None, "crash")
    return IsolatedProcessResult(response, None)
