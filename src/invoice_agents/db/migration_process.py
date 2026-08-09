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
import threading
import time
from dataclasses import dataclass, field
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
_WORKER_QUARANTINE_RETRY_ATTEMPTS = 3
_WORKER_QUARANTINE_RETRY_SECONDS = 0.1
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
            kernel_queue = select.kqueue()
            self._kqueue = kernel_queue
            try:
                event = select.kevent(
                    process_id,
                    filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                    fflags=select.KQ_NOTE_EXIT,
                )
                kernel_queue.control([event], 0, 0)
            except BaseException:
                kernel_queue.close()
                self._kqueue = None
                raise
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
    exit_watcher: _WorkerExitWatcher | None
    watcher_initialized: bool = True
    identity_verified: bool = True
    cleanup_lock: threading.Lock = field(default_factory=threading.Lock)
    cleaned: bool = False


@dataclass(frozen=True, slots=True)
class _WorkerSessionMember:
    process_id: int
    process_group_id: int


@dataclass(frozen=True, slots=True)
class _WorkerSessionSnapshot:
    leader_state: str
    members: tuple[_WorkerSessionMember, ...]


class _WorkerCleanupFailure(Exception):
    """Private marker whose platform details must never cross the boundary."""


_QUARANTINED_WORKERS: dict[int, _WorkerSession] = {}
_QUARANTINED_WORKERS_LOCK = threading.Lock()
_QUARANTINE_RETRY_THREAD_RUNNING = False


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
    exit_watcher: _WorkerExitWatcher | None
    watcher_initialized = True
    try:
        exit_watcher = _WorkerExitWatcher(process_id)
    except Exception:
        # The Popen contract already reserved PID == PGID == SID. Keep that
        # unreaped identity and use strict ps snapshots for cleanup; the raw
        # construction failure is deliberately not retained.
        exit_watcher = None
        watcher_initialized = False
    identity_verified = True
    try:
        process_group_id = os.getpgid(process_id)
        session_id = os.getsid(process_id)
    except ProcessLookupError:
        # Popen(start_new_session=True) completed setsid before exec. A fast child may
        # already be an unreaped zombie on Darwin, where getpgid/getsid return ESRCH.
        process_group_id = process_id
        session_id = process_id
    except OSError:
        # Popen completed start_new_session=True before returning. Retain that
        # reserved identity for strict cleanup, but never accept the worker result.
        process_group_id = process_id
        session_id = process_id
        identity_verified = False
    if process_group_id != process_id or session_id != process_id:
        if exit_watcher is not None:
            exit_watcher.close()
        exit_watcher = None
        watcher_initialized = False
        identity_verified = False
        process_group_id = process_id
        session_id = process_id
    return _WorkerSession(
        process=process,
        process_id=process_id,
        process_group_id=process_group_id,
        session_id=session_id,
        exit_watcher=exit_watcher,
        watcher_initialized=watcher_initialized,
        identity_verified=identity_verified,
    )


def _uncertain_worker_session(process: subprocess.Popen[bytes]) -> _WorkerSession:
    """Retain the start_new_session identity when capture itself could not finish."""

    return _WorkerSession(
        process=process,
        process_id=process.pid,
        process_group_id=process.pid,
        session_id=process.pid,
        exit_watcher=None,
        watcher_initialized=False,
        identity_verified=False,
    )


def _cleanup_protocol_error() -> DatabaseVerificationError:
    return _protocol_error(
        "MIGRATION_WORKER_CLEANUP_FAILED",
        "database migration worker session cleanup could not be verified",
    )


def _canonical_process_identifier(value: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise _WorkerCleanupFailure
    parsed = int(value)
    if parsed <= 0 or str(parsed) != value:
        raise _WorkerCleanupFailure
    return parsed


def _process_status_command() -> list[str]:
    if sys.platform == "darwin":
        # Darwin's documented `sess` field is an audit/login session identifier
        # (and is zero in non-login launch contexts), not the POSIX SID returned
        # by getsid(2). Query SID through the kernel below instead.
        return ["/bin/ps", "-axo", "pid=,pgid=,stat="]
    if sys.platform.startswith("linux"):
        return ["/bin/ps", "-axo", "pid=,pgid=,sid=,stat="]
    raise _WorkerCleanupFailure


def _valid_process_state(value: str) -> bool:
    """Validate the documented Darwin or procps STAT alphabet exactly."""

    if not value or not value.isascii():
        return False
    if sys.platform == "darwin":
        # Darwin ps can report `?` for the short post-fork/pre-exec window;
        # it is a single explicit transient state, not a wildcard.
        primary_states = frozenset("?IRSTUZ")
        modifiers = frozenset("+<>AELNSsVWX")
    elif sys.platform.startswith("linux"):
        # Include every current and documented historical /proc state that
        # procps can expose, while rejecting arbitrary state bytes.
        primary_states = frozenset("DIKPRStTWXxZ")
        modifiers = frozenset("<NLsl+")
    else:
        return False
    suffix = value[1:]
    return (
        value[0] in primary_states
        and all(character in modifiers for character in suffix)
        and len(set(suffix)) == len(suffix)
    )


def _worker_session_snapshot(worker: _WorkerSession) -> _WorkerSessionSnapshot:
    """Strictly prove the reserved leader row and enumerate its session members."""

    command = _process_status_command()
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _WorkerCleanupFailure from exc
    if (
        completed.args != command
        or type(completed.returncode) is not int
        or completed.returncode != 0
        or type(completed.stdout) is not str
        or type(completed.stderr) is not str
        or completed.stderr
    ):
        raise _WorkerCleanupFailure
    if not completed.stdout:
        raise _WorkerCleanupFailure
    seen: set[int] = set()
    leader_state: str | None = None
    members: list[_WorkerSessionMember] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            raise _WorkerCleanupFailure
        fields = line.split()
        expected_fields = 3 if sys.platform == "darwin" else 4
        if len(fields) != expected_fields:
            raise _WorkerCleanupFailure
        state = fields[-1]
        if not _valid_process_state(state):
            raise _WorkerCleanupFailure
        process_id = _canonical_process_identifier(fields[0])
        listed_group_id = _canonical_process_identifier(fields[1])
        listed_session_id = (
            _canonical_process_identifier(fields[2]) if expected_fields == 4 else None
        )
        if process_id in seen:
            raise _WorkerCleanupFailure
        seen.add(process_id)
        if process_id == worker.process_id:
            if (
                listed_group_id != worker.process_group_id
                or (listed_session_id is not None and listed_session_id != worker.session_id)
                or leader_state is not None
            ):
                raise _WorkerCleanupFailure
            leader_state = state
            continue
        if listed_session_id is not None and listed_session_id != worker.session_id:
            continue
        try:
            first_session_id = os.getsid(process_id)
            process_group_id = os.getpgid(process_id)
            current_session_id = os.getsid(process_id)
        except ProcessLookupError:
            # getsid/getpgid ESRCH is kernel confirmation that this exact ps row
            # disappeared. It authorizes no signal; the next complete snapshot
            # must independently prove the reserved session empty.
            continue
        except OSError as exc:
            raise _WorkerCleanupFailure from exc
        if (
            first_session_id != worker.session_id
            or current_session_id != worker.session_id
            or process_group_id != listed_group_id
        ):
            if listed_session_id is None and (
                first_session_id != worker.session_id and current_session_id != worker.session_id
            ):
                continue
            raise _WorkerCleanupFailure
        members.append(_WorkerSessionMember(process_id, listed_group_id))
    if leader_state is None:
        raise _WorkerCleanupFailure
    return _WorkerSessionSnapshot(
        leader_state=leader_state,
        members=tuple(sorted(members, key=lambda member: member.process_id)),
    )


def _worker_session_members(worker: _WorkerSession) -> tuple[_WorkerSessionMember, ...]:
    """Return members only after the strict leader/session snapshot is proven."""

    return _worker_session_snapshot(worker).members


def _watcher_observed_exit(worker: _WorkerSession) -> bool | None:
    watcher = worker.exit_watcher
    if watcher is None:
        return None
    try:
        return watcher.wait(0)
    except OSError as exc:
        raise _WorkerCleanupFailure from exc


def _leader_exit_is_proven(
    worker: _WorkerSession,
    snapshot: _WorkerSessionSnapshot,
) -> bool:
    zombie = snapshot.leader_state.startswith("Z")
    observed = _watcher_observed_exit(worker)
    if observed is not None and observed != zombie:
        raise _WorkerCleanupFailure
    return zombie


def _signal_worker_leader(
    worker: _WorkerSession,
    snapshot: _WorkerSessionSnapshot,
    signal_number: int,
) -> None:
    """Signal the unreaped, therefore non-reusable, leader PID when still running."""

    if _leader_exit_is_proven(worker, snapshot):
        return
    try:
        os.kill(worker.process_id, signal_number)
    except OSError as exc:
        raise _WorkerCleanupFailure from exc


def _signal_worker_session_groups(
    worker: _WorkerSession,
    members: tuple[_WorkerSessionMember, ...],
    signal_number: int,
) -> None:
    """Signal each group only after an immediate reserved-session identity check."""

    if sys.platform.startswith("linux"):
        _signal_worker_session_members_with_pidfds(worker, members, signal_number)
        return
    groups: dict[int, list[int]] = {}
    for member in members:
        groups.setdefault(member.process_group_id, []).append(member.process_id)
    for process_group_id in sorted(groups):
        if process_group_id == worker.process_group_id:
            # The unreaped session leader has PID == PGID and reserves this
            # group identity until the final wait below.
            verified = True
        else:
            # A descendant-created group is signalled only while its own live
            # group leader still anchors PGID inside the reserved SID.
            anchors = [
                process_id
                for process_id in groups[process_group_id]
                if process_id == process_group_id
            ]
            if len(anchors) != 1:
                raise _WorkerCleanupFailure
            process_id = anchors[0]
            try:
                first_session_id = os.getsid(process_id)
                current_group_id = os.getpgid(process_id)
                current_session_id = os.getsid(process_id)
            except ProcessLookupError:
                # The group anchor disappeared before signalling. Do not signal
                # its numeric PGID; a complete re-enumeration decides what remains.
                continue
            except OSError as exc:
                raise _WorkerCleanupFailure from exc
            verified = (
                first_session_id == worker.session_id
                and current_session_id == worker.session_id
                and current_group_id == process_group_id
            )
        if not verified:
            raise _WorkerCleanupFailure
        try:
            os.killpg(process_group_id, signal_number)
        except ProcessLookupError:
            continue
        except OSError as exc:
            raise _WorkerCleanupFailure from exc


def _signal_worker_session_members_with_pidfds(
    worker: _WorkerSession,
    members: tuple[_WorkerSessionMember, ...],
    signal_number: int,
) -> None:
    """On Linux, signal exact processes through stable kernel PID handles."""

    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        raise _WorkerCleanupFailure
    for member in members:
        try:
            process_handle = int(pidfd_open(member.process_id))
        except ProcessLookupError:
            continue
        except OSError as exc:
            raise _WorkerCleanupFailure from exc
        try:
            try:
                first_session_id = os.getsid(member.process_id)
                current_group_id = os.getpgid(member.process_id)
                current_session_id = os.getsid(member.process_id)
            except ProcessLookupError:
                continue
            except OSError as exc:
                raise _WorkerCleanupFailure from exc
            if (
                first_session_id != worker.session_id
                or current_session_id != worker.session_id
                or current_group_id != member.process_group_id
            ):
                raise _WorkerCleanupFailure
            try:
                pidfd_send_signal(process_handle, signal_number, None, 0)
            except ProcessLookupError:
                continue
            except OSError as exc:
                raise _WorkerCleanupFailure from exc
        finally:
            try:
                os.close(process_handle)
            except OSError as exc:
                raise _WorkerCleanupFailure from exc


def _run_worker_cleanup_phase(
    worker: _WorkerSession,
    signal_number: int,
    timeout: float,
) -> bool:
    """Signal and re-enumerate until two consecutive session-empty snapshots."""

    deadline = time.monotonic() + max(0.0, timeout)
    empty_snapshots = 0
    while True:
        snapshot = _worker_session_snapshot(worker)
        leader_exited = _leader_exit_is_proven(worker, snapshot)
        if leader_exited and not snapshot.members:
            empty_snapshots += 1
            if empty_snapshots == 2:
                return True
            continue
        empty_snapshots = 0
        _signal_worker_leader(worker, snapshot, signal_number)
        _signal_worker_session_groups(worker, snapshot.members, signal_number)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        wait_seconds = min(_WORKER_POLL_SECONDS, remaining)
        watcher = worker.exit_watcher
        if leader_exited or watcher is None:
            time.sleep(wait_seconds)
        else:
            try:
                watcher.wait(wait_seconds)
            except OSError as exc:
                raise _WorkerCleanupFailure from exc


def _cleanup_worker_session(worker: _WorkerSession) -> None:
    """Empty the reserved session before releasing its leader PID/SID identity."""

    with worker.cleanup_lock:
        if worker.cleaned:
            return
        session_empty = _run_worker_cleanup_phase(
            worker,
            signal.SIGTERM,
            _WORKER_SHUTDOWN_SECONDS,
        )
        if not session_empty:
            session_empty = _run_worker_cleanup_phase(
                worker,
                signal.SIGKILL,
                _WORKER_SHUTDOWN_SECONDS,
            )
        final_snapshot = _worker_session_snapshot(worker)
        if (
            not session_empty
            or final_snapshot.members
            or not _leader_exit_is_proven(worker, final_snapshot)
        ):
            raise _WorkerCleanupFailure
        watcher = worker.exit_watcher
        if watcher is not None:
            try:
                watcher.close()
            except OSError as exc:
                raise _WorkerCleanupFailure from exc
        stdout = worker.process.stdout
        if stdout is not None and not stdout.closed:
            try:
                stdout.close()
            except OSError as exc:
                raise _WorkerCleanupFailure from exc
        try:
            worker.process.wait(timeout=max(_WORKER_POLL_SECONDS, _WORKER_SHUTDOWN_SECONDS))
        except (OSError, subprocess.SubprocessError) as exc:
            raise _WorkerCleanupFailure from exc
        worker.cleaned = True


def _remove_quarantined_worker(worker: _WorkerSession) -> None:
    with _QUARANTINED_WORKERS_LOCK:
        if _QUARANTINED_WORKERS.get(worker.process_id) is worker:
            del _QUARANTINED_WORKERS[worker.process_id]


def _retry_quarantined_workers() -> bool:
    """Boundedly retry retained sessions without ever signaling a released SID."""

    with _QUARANTINED_WORKERS_LOCK:
        workers = tuple(_QUARANTINED_WORKERS.values())
    for worker in workers:
        try:
            _cleanup_worker_session(worker)
        except Exception:
            continue
        _remove_quarantined_worker(worker)
    with _QUARANTINED_WORKERS_LOCK:
        return not _QUARANTINED_WORKERS


def _background_quarantine_retry() -> None:
    global _QUARANTINE_RETRY_THREAD_RUNNING

    try:
        for _attempt in range(_WORKER_QUARANTINE_RETRY_ATTEMPTS):
            time.sleep(_WORKER_QUARANTINE_RETRY_SECONDS)
            if _retry_quarantined_workers():
                return
    finally:
        with _QUARANTINED_WORKERS_LOCK:
            _QUARANTINE_RETRY_THREAD_RUNNING = False


def _quarantine_worker(worker: _WorkerSession) -> None:
    global _QUARANTINE_RETRY_THREAD_RUNNING

    start_retry_thread = False
    with _QUARANTINED_WORKERS_LOCK:
        _QUARANTINED_WORKERS[worker.process_id] = worker
        if not _QUARANTINE_RETRY_THREAD_RUNNING:
            _QUARANTINE_RETRY_THREAD_RUNNING = True
            start_retry_thread = True
    if start_retry_thread:
        try:
            threading.Thread(
                target=_background_quarantine_retry,
                name="migration-worker-cleanup",
                daemon=True,
            ).start()
        except RuntimeError:
            with _QUARANTINED_WORKERS_LOCK:
                _QUARANTINE_RETRY_THREAD_RUNNING = False


def _stop_worker(worker: _WorkerSession) -> DatabaseVerificationError | None:
    """Strictly clean or retain the worker and return a chainless public error."""

    cleanup_failed = False
    try:
        _cleanup_worker_session(worker)
    except Exception:
        _quarantine_worker(worker)
        cleanup_failed = True
    if cleanup_failed:
        return _cleanup_protocol_error()
    _remove_quarantined_worker(worker)
    return None


def _read_bounded_worker_response(worker: _WorkerSession) -> bytes:
    """Read at most one capped response while enforcing the whole-worker deadline."""

    process = worker.process
    exit_watcher = worker.exit_watcher
    if exit_watcher is None:
        raise _protocol_error(
            "MIGRATION_WORKER_START_FAILED",
            "database migration worker could not be started",
        )
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
                if exit_watcher.wait(0):
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
    if remaining <= 0 or not exit_watcher.wait(remaining):
        raise _protocol_error(
            "MIGRATION_WORKER_TIMEOUT",
            "database migration worker exceeded its execution deadline",
        )
    return bytes(response)


def _run_migration_in_subprocess(
    path: Path,
    kind: DatabaseKind,
    *,
    settings: Settings | None,
) -> list[int]:
    """Run the complete migration in a fresh interpreter owning every SQLite descriptor."""

    if settings is not None:
        settings.assert_delete_journal_mode()
    if not _retry_quarantined_workers():
        raise _cleanup_protocol_error()
    request = _encode_request(path, kind, settings)
    process: subprocess.Popen[bytes] | None = None
    worker: _WorkerSession | None = None
    start_failed = False
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
    except Exception:
        start_failed = True
        if process is not None:
            worker = _uncertain_worker_session(process)
    if worker is None:
        raise _protocol_error(
            "MIGRATION_WORKER_START_FAILED",
            "database migration worker could not be started",
        )
    start_failed = start_failed or not worker.watcher_initialized or not worker.identity_verified
    if start_failed:
        cleanup_error = _stop_worker(worker)
        if cleanup_error is not None:
            raise cleanup_error
        raise _protocol_error(
            "MIGRATION_WORKER_START_FAILED",
            "database migration worker could not be started",
        )
    try:
        stdout = _read_bounded_worker_response(worker)
    finally:
        cleanup_error = _stop_worker(worker)
        if cleanup_error is not None:
            raise cleanup_error
    if worker.process.returncode != 0:
        raise _protocol_error(
            "MIGRATION_WORKER_CRASHED",
            "database migration worker exited before returning a result",
        )
    return _decode_response(stdout)


def _copy_public_error(error: DatabaseVerificationError) -> DatabaseVerificationError:
    """Copy only the stable domain payload; exception implementation details stay private."""

    return DatabaseVerificationError(
        error.category,
        error.message,
        case_id=error.case_id,
        stop_reason=error.stop_reason,
        provider_request_id=error.provider_request_id,
        details=error.details,
    )


def run_migration_in_subprocess(
    path: Path,
    kind: DatabaseKind,
    *,
    settings: Settings | None,
) -> list[int]:
    """Run migration with a stable, chainless public exception boundary."""

    public_error: DatabaseVerificationError | None = None
    result: list[int] | None = None
    try:
        result = _run_migration_in_subprocess(path, kind, settings=settings)
    except DatabaseVerificationError as exc:
        public_error = _copy_public_error(exc)
    except Exception:
        public_error = _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    if public_error is not None:
        raise public_error
    if result is None:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    return result


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
