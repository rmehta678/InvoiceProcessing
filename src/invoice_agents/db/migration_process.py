"""Bounded subprocess boundary for descriptor-isolated SQLite migrations."""

from __future__ import annotations

import json
import math
import os
import select
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind
from invoice_agents.errors import DatabaseVerificationError, ErrorCategory

MIGRATION_WORKER_PROTOCOL_VERSION = 1
MIGRATION_WORKER_MAX_MESSAGE_BYTES = 65_536
MIGRATION_WORKER_TIMEOUT_SECONDS = 120.0
_WORKER_SHUTDOWN_SECONDS = 2.0
_WORKER_POLL_SECONDS = 0.05
_SERIALIZED_SETTINGS_FIELDS = frozenset(
    {
        "xai_api_key",
        "inventory_db",
        "workflow_db",
        "sqlite_journal_mode",
        "source_archive_dir",
        "source_max_bytes",
        "pdf_max_pages",
        "pdf_parse_timeout_seconds",
        "pdf_worker_cpu_seconds",
        "pdf_worker_memory_bytes",
        "pdf_worker_result_max_bytes",
        "review_threshold_amount",
        "review_threshold_currency",
        "review_threshold_effective_date",
        "due_date_tolerance_days",
        "review_age_amber_hours",
        "max_messages",
        "model_timeout_seconds",
        "transient_retries",
        "case_concurrency",
        "log_level",
    }
)


class _WorkerExitWatcher:
    """Observe a child exit without reaping its identity before group cleanup."""

    def __init__(self, process_id: int) -> None:
        self._process_id = process_id
        self._exited = False
        self._kqueue: Any | None = None
        self._pidfd: int | None = None
        if sys.platform == "darwin" and hasattr(select, "kqueue"):
            self._kqueue = select.kqueue()
            event = select.kevent(
                process_id,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                fflags=select.KQ_NOTE_EXIT,
            )
            self._kqueue.control([event], 0, 0)
            return
        pidfd_open = getattr(os, "pidfd_open", None)
        if sys.platform.startswith("linux") and pidfd_open is not None:
            self._pidfd = int(pidfd_open(process_id))
            return
        raise OSError("platform cannot observe a worker exit without reaping it")

    def wait(self, timeout: float) -> bool:
        if self._exited:
            return True
        bounded_timeout = max(0.0, timeout)
        if self._kqueue is not None:
            self._exited = bool(self._kqueue.control(None, 1, bounded_timeout))
        elif self._pidfd is not None:
            readable, _, _ = select.select([self._pidfd], [], [], bounded_timeout)
            self._exited = bool(readable)
        return self._exited

    def close(self) -> None:
        if self._kqueue is not None:
            self._kqueue.close()
            self._kqueue = None
        if self._pidfd is not None:
            os.close(self._pidfd)
            self._pidfd = None


@dataclass(slots=True)
class _WorkerSession:
    """A dedicated child session whose leader remains unreaped during cleanup."""

    process: subprocess.Popen[bytes]
    process_id: int
    process_group_id: int
    session_id: int
    exit_watcher: _WorkerExitWatcher


def _worker_command() -> list[str]:
    return [sys.executable, "-m", "invoice_agents.db.migration_worker"]


def _serialize_settings(settings: Settings | None) -> dict[str, object] | None:
    if settings is None:
        return None
    return {
        "xai_api_key": None,
        "inventory_db": os.fspath(settings.inventory_db),
        "workflow_db": os.fspath(settings.workflow_db),
        "sqlite_journal_mode": settings.sqlite_journal_mode,
        "source_archive_dir": os.fspath(settings.source_archive_dir),
        "source_max_bytes": settings.source_max_bytes,
        "pdf_max_pages": settings.pdf_max_pages,
        "pdf_parse_timeout_seconds": settings.pdf_parse_timeout_seconds,
        "pdf_worker_cpu_seconds": settings.pdf_worker_cpu_seconds,
        "pdf_worker_memory_bytes": settings.pdf_worker_memory_bytes,
        "pdf_worker_result_max_bytes": settings.pdf_worker_result_max_bytes,
        "review_threshold_amount": str(settings.review_threshold_amount),
        "review_threshold_currency": settings.review_threshold_currency,
        "review_threshold_effective_date": settings.review_threshold_effective_date.isoformat(),
        "due_date_tolerance_days": settings.due_date_tolerance_days,
        "review_age_amber_hours": settings.review_age_amber_hours,
        "max_messages": settings.max_messages,
        "model_timeout_seconds": settings.model_timeout_seconds,
        "transient_retries": settings.transient_retries,
        "case_concurrency": settings.case_concurrency,
        "log_level": settings.log_level,
    }


def _protocol_error(stop_reason: str, message: str) -> DatabaseVerificationError:
    return DatabaseVerificationError(
        ErrorCategory.DATABASE,
        message,
        stop_reason=stop_reason,
    )


def _encode_request(
    path: Path,
    kind: DatabaseKind,
    settings: Settings | None,
) -> bytes:
    payload = {
        "protocol_version": MIGRATION_WORKER_PROTOCOL_VERSION,
        "path": os.fspath(path),
        "kind": kind.value,
        "settings": _serialize_settings(settings),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    if len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker request exceeded its bounded protocol",
        )
    return encoded


def _safe_details(value: object, *, depth: int = 0) -> object:
    if depth > 6:
        raise ValueError("worker details are too deeply nested")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("worker detail number is not finite")
        return value
    if isinstance(value, str):
        if len(value) > 4_096:
            raise ValueError("worker detail string is too large")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise ValueError("worker detail list is too large")
        return [_safe_details(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 256:
            raise ValueError("worker detail mapping is too large")
        safe: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise ValueError("worker detail key is invalid")
            lowered = key.casefold().replace("-", "_")
            if any(token in lowered for token in ("secret", "token", "password", "api_key")):
                safe[key] = "[REDACTED]"
            else:
                safe[key] = _safe_details(item, depth=depth + 1)
        return safe
    raise ValueError("worker details contain an unsupported value")


def _decode_response(encoded: bytes) -> list[int]:
    if not encoded or len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        ) from exc
    if not isinstance(payload, dict) or type(payload.get("ok")) is not bool:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    if payload["ok"]:
        if set(payload) != {"ok", "applied"}:
            raise _protocol_error(
                "MIGRATION_WORKER_PROTOCOL_INVALID",
                "database migration worker returned an invalid bounded response",
            )
        applied = payload["applied"]
        if (
            not isinstance(applied, list)
            or len(applied) > 1_000
            or any(type(version) is not int or version < 1 for version in applied)
        ):
            raise _protocol_error(
                "MIGRATION_WORKER_PROTOCOL_INVALID",
                "database migration worker returned an invalid bounded response",
            )
        return applied
    if set(payload) != {"ok", "error"} or not isinstance(payload["error"], dict):
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    error = payload["error"]
    if set(error) != {"category", "message", "stop_reason", "details"}:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    category = error["category"]
    message = error["message"]
    stop_reason = error["stop_reason"]
    if (
        not isinstance(category, str)
        or not isinstance(message, str)
        or not 1 <= len(message) <= 4_096
        or not isinstance(stop_reason, str)
        or not 1 <= len(stop_reason) <= 256
    ):
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    try:
        error_category = ErrorCategory(category)
    except ValueError as exc:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        ) from exc
    try:
        details = _safe_details(error["details"])
    except ValueError as exc:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        ) from exc
    return_error = DatabaseVerificationError(
        error_category,
        message,
        stop_reason=stop_reason,
        details=details if isinstance(details, dict) else None,
    )
    raise return_error


def _capture_worker_session(process: subprocess.Popen[bytes]) -> _WorkerSession:
    """Capture the start_new_session identity before any operation can reap it."""

    process_id = process.pid
    exit_watcher = _WorkerExitWatcher(process_id)
    try:
        process_group_id = os.getpgid(process_id)
        session_id = os.getsid(process_id)
    except ProcessLookupError:
        # Popen(start_new_session=True) completed setsid before exec. A fast child may
        # already be an unreaped zombie on Darwin, where getpgid/getsid return ESRCH.
        process_group_id = process_id
        session_id = process_id
    if process_group_id != process_id or session_id != process_id:
        exit_watcher.close()
        raise OSError("migration worker did not enter its dedicated session")
    return _WorkerSession(
        process=process,
        process_id=process_id,
        process_group_id=process_group_id,
        session_id=session_id,
        exit_watcher=exit_watcher,
    )


def _signal_worker_group(worker: _WorkerSession, signal_number: int) -> None:
    """Signal only the captured group while its unreaped leader prevents ID reuse."""

    # Darwin reports EPERM rather than ESRCH when the group contains only its
    # already-exited, unreaped leader. In either case there is no live target.
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(worker.process_group_id, signal_number)


def _worker_group_members(worker: _WorkerSession) -> tuple[int, ...] | None:
    """List still-present same-session members other than the unreaped leader."""

    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,pgid=,stat="],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    members: list[int] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            process_id = int(fields[0])
            process_group_id = int(fields[1])
        except ValueError:
            continue
        if process_id == worker.process_id or process_group_id != worker.process_group_id:
            continue
        try:
            if (
                os.getpgid(process_id) != worker.process_group_id
                or os.getsid(process_id) != worker.session_id
            ):
                continue
        except OSError:
            continue
        members.append(process_id)
    return tuple(members)


def _wait_for_worker_group_empty(worker: _WorkerSession, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        members = _worker_group_members(worker)
        if members == ():
            return True
        if members is None or time.monotonic() >= deadline:
            return False
        time.sleep(min(_WORKER_POLL_SECONDS, max(0.0, deadline - time.monotonic())))


def _stop_worker(worker: _WorkerSession) -> None:
    """Terminate the dedicated group and reap its leader on every return path."""

    process = worker.process
    try:
        if not worker.exit_watcher.wait(0):
            _signal_worker_group(worker, signal.SIGTERM)
            if not worker.exit_watcher.wait(_WORKER_SHUTDOWN_SECONDS):
                _signal_worker_group(worker, signal.SIGKILL)
                worker.exit_watcher.wait(_WORKER_SHUTDOWN_SECONDS)

        # The leader is still unreaped here. Its PID therefore cannot be reused as
        # an unrelated process group while these group-directed signals are sent.
        _signal_worker_group(worker, signal.SIGTERM)
        if not _wait_for_worker_group_empty(worker, _WORKER_SHUTDOWN_SECONDS):
            _signal_worker_group(worker, signal.SIGKILL)
            _wait_for_worker_group_empty(worker, _WORKER_SHUTDOWN_SECONDS)

        try:
            process.wait(timeout=_WORKER_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        worker.exit_watcher.close()


def _stop_unwatched_worker(process: subprocess.Popen[bytes]) -> None:
    """Best-effort cleanup when exit-watcher creation itself fails."""

    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_WORKER_SHUTDOWN_SECONDS)
    if process.returncode is None:
        process.kill()
        process.wait()


def _read_bounded_worker_response(worker: _WorkerSession) -> bytes:
    """Read at most one capped response while enforcing the whole-worker deadline."""

    process = worker.process
    stdout = process.stdout
    if stdout is None:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    descriptor = stdout.fileno()
    os.set_blocking(descriptor, False)
    deadline = time.monotonic() + MIGRATION_WORKER_TIMEOUT_SECONDS
    response = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _protocol_error(
                    "MIGRATION_WORKER_TIMEOUT",
                    "database migration worker exceeded its execution deadline",
                )
            if not selector.select(min(remaining, _WORKER_POLL_SECONDS)):
                if worker.exit_watcher.wait(0):
                    raise _protocol_error(
                        "MIGRATION_WORKER_CRASHED",
                        "database migration worker exited before returning a result",
                    )
                continue
            try:
                chunk = os.read(
                    descriptor,
                    min(8_192, MIGRATION_WORKER_MAX_MESSAGE_BYTES + 1 - len(response)),
                )
            except BlockingIOError:
                continue
            except OSError as exc:
                raise _protocol_error(
                    "MIGRATION_WORKER_PROTOCOL_INVALID",
                    "database migration worker returned an invalid bounded response",
                ) from exc
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
                raise _protocol_error(
                    "MIGRATION_WORKER_PROTOCOL_INVALID",
                    "database migration worker returned an invalid bounded response",
                )
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not worker.exit_watcher.wait(remaining):
        raise _protocol_error(
            "MIGRATION_WORKER_TIMEOUT",
            "database migration worker exceeded its execution deadline",
        )
    return bytes(response)


def run_migration_in_subprocess(
    path: Path,
    kind: DatabaseKind,
    *,
    settings: Settings | None,
) -> list[int]:
    """Run the complete migration in a fresh interpreter owning every SQLite descriptor."""

    if settings is not None:
        settings.assert_delete_journal_mode()
    request = _encode_request(path, kind, settings)
    process: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.TemporaryFile() as request_stream:
            request_stream.write(request)
            request_stream.seek(0)
            process = subprocess.Popen(
                _worker_command(),
                stdin=request_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        worker = _capture_worker_session(process)
    except OSError as exc:
        if process is not None:
            _stop_unwatched_worker(process)
        raise _protocol_error(
            "MIGRATION_WORKER_START_FAILED",
            "database migration worker could not be started",
        ) from exc
    try:
        stdout = _read_bounded_worker_response(worker)
    finally:
        _stop_worker(worker)
        if process.stdout is not None:
            process.stdout.close()
    if process.returncode != 0:
        raise _protocol_error(
            "MIGRATION_WORKER_CRASHED",
            "database migration worker exited before returning a result",
        )
    return _decode_response(stdout)


def decode_worker_request(encoded: bytes) -> tuple[Path, DatabaseKind, Settings | None]:
    """Validate the complete child request without using ambient configuration values."""

    if not encoded or len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise ValueError("invalid worker request size")
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or set(payload) != {
        "protocol_version",
        "path",
        "kind",
        "settings",
    }:
        raise ValueError("invalid worker request shape")
    if payload["protocol_version"] != MIGRATION_WORKER_PROTOCOL_VERSION:
        raise ValueError("invalid worker protocol version")
    if not isinstance(payload["path"], str) or not 1 <= len(payload["path"]) <= 16_384:
        raise ValueError("invalid worker path")
    if not isinstance(payload["kind"], str):
        raise ValueError("invalid worker database kind")
    kind = DatabaseKind(payload["kind"])
    settings_payload = payload["settings"]
    if settings_payload is None:
        settings = None
    elif isinstance(settings_payload, dict):
        if set(settings_payload) != _SERIALIZED_SETTINGS_FIELDS:
            raise ValueError("invalid worker settings shape")
        settings = Settings(**settings_payload)
    else:
        raise ValueError("invalid worker settings")
    return Path(payload["path"]), kind, settings


def encode_worker_response(payload: dict[str, Any]) -> bytes:
    """Encode one already-sanitized child response under the channel cap."""

    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    if len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise ValueError("worker response exceeded protocol bound")
    return encoded
