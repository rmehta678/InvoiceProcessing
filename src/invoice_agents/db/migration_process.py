"""Bounded subprocess boundary for descriptor-isolated SQLite migrations."""

from __future__ import annotations

import json
import os
import re
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
from typing import Any, cast

from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind
from invoice_agents.errors import DatabaseVerificationError, ErrorCategory
from invoice_agents.wire_settings import decode_wire_settings, serialize_wire_settings
from invoice_agents.worker_environment import sanitized_worker_environment

MIGRATION_WORKER_PROTOCOL_VERSION = 1
MIGRATION_WORKER_MAX_MESSAGE_BYTES = 65_536
MIGRATION_WORKER_TIMEOUT_SECONDS = 120.0
_WORKER_SHUTDOWN_SECONDS = 2.0
_WORKER_POLL_SECONDS = 0.05
_PROCESS_TABLE_MAX_BYTES = 1_048_576
_WORKER_RESOURCE_CLEANUP_POISONED = threading.Event()


def _worker_resource_cleanup_is_poisoned() -> bool:
    """Return whether an ambiguous descriptor close permanently blocks admission."""

    return _WORKER_RESOURCE_CLEANUP_POISONED.is_set()


def _poison_worker_resource_cleanup() -> None:
    """Permanently fail closed when owned worker cleanup cannot be proven."""

    _WORKER_RESOURCE_CLEANUP_POISONED.set()


@dataclass(frozen=True, slots=True)
class _WorkerErrorContract:
    category: ErrorCategory
    message: str
    count_detail_fields: tuple[str, ...] = ()


_AUTHORIZATION_RECONCILIATION_COUNT_FIELDS = (
    "review_request_count",
    "human_decision_count",
    "final_decision_count",
    "payment_count",
)
_AUTHORIZATION_AUDIT_COUNT_FIELDS = (
    "invalid_review_count",
    "invalid_human_decision_count",
    "invalid_snapshot_count",
    "invalid_final_decision_count",
    "invalid_payment_count",
    "invalid_cardinality_count",
    "invalid_quarantine_count",
)

_WORKER_DOMAIN_ERROR_CONTRACTS: dict[str, _WorkerErrorContract] = {
    "AUTHORIZATION_INVENTORY_WAL_MODE_UNSUPPORTED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "authorization inventory database must use DELETE journal mode",
    ),
    "AUTHORIZATION_RECONCILIATION_REQUIRED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "workflow authorization reconciliation is required before migration 003",
        _AUTHORIZATION_RECONCILIATION_COUNT_FIELDS,
    ),
    "DATABASE_AUTHORIZATION_CONTEXT_MISMATCH": _WorkerErrorContract(
        ErrorCategory.CONFIGURATION,
        "database migration authorization context does not match its target",
    ),
    "DATABASE_AUTHORIZATION_CONTEXT_REQUIRED": _WorkerErrorContract(
        ErrorCategory.CONFIGURATION,
        "database migration requires explicit authorization context",
    ),
    "DATABASE_AUTHORIZATION_PROVENANCE_INVALID": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "legacy workflow v3 authorization provenance is incomplete or inconsistent",
        _AUTHORIZATION_AUDIT_COUNT_FIELDS,
    ),
    "DATABASE_CHANGED_DURING_VERIFICATION": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database source changed during migration verification",
    ),
    "DATABASE_INTEGRITY_FAILED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database integrity verification failed",
    ),
    "DATABASE_LOCK_UNAVAILABLE": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database maintenance lock is unavailable",
    ),
    "DATABASE_MAINTENANCE_BINDING_FAILED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database maintenance binding could not be verified",
    ),
    "DATABASE_MAINTENANCE_CLEANUP_FAILED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database maintenance cleanup could not be verified",
    ),
    "DATABASE_MISSING": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "required database does not exist",
    ),
    "DATABASE_SCHEMA_MISMATCH": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database schema does not match the migration contract",
    ),
    "DATABASE_SIDECAR_UNSUPPORTED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database sidecar files are not supported during migration",
    ),
    "DATABASE_SIGNATURE_INVALID": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database file signature is invalid",
    ),
    "DATABASE_SYMLINK_UNSUPPORTED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database symlink paths are not supported during migration",
    ),
    "DATABASE_VERIFICATION_ERROR": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database verification failed",
    ),
    "DATABASE_VERSION_MISMATCH": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database schema version does not match the migration contract",
    ),
    "INVENTORY_WAL_MODE_UNSUPPORTED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "inventory database must use DELETE journal mode",
    ),
    "LEGACY_RECONCILIATION_ARCHIVE_INVALID": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "legacy reconciliation archive failed integrity verification",
    ),
    "LEGACY_RECONCILIATION_ARCHIVE_UPGRADE_REQUIRED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "legacy authorization archive requires a lossless upgrade",
    ),
    "LEGACY_RECONCILIATION_DELETE_INCOMPLETE": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "legacy authorization reconciliation did not remove every active row",
    ),
    "LEGACY_RECONCILIATION_FAILED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "legacy authorization reconciliation failed atomically",
    ),
    "LEGACY_RECONCILIATION_STATE_INVALID": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "legacy authorization reconciliation state is invalid",
    ),
    "MIGRATION_FAILED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database migration failed",
    ),
    "MIGRATION_HISTORY_INVALID": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database migration history is invalid",
    ),
    "MIGRATION_NOT_FOUND": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database migration resources are unavailable",
    ),
    "WORKFLOW_WAL_MODE_UNSUPPORTED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "workflow database must use DELETE journal mode",
    ),
}

_PARENT_WORKER_ERROR_CONTRACTS: dict[str, _WorkerErrorContract] = {
    "MIGRATION_WORKER_PROTOCOL_INVALID": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database migration worker returned an invalid bounded response",
    ),
    "MIGRATION_WORKER_START_FAILED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database migration worker could not be started",
    ),
    "MIGRATION_WORKER_CLEANUP_FAILED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database migration worker session cleanup could not be verified",
    ),
    "MIGRATION_WORKER_TIMEOUT": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database migration worker exceeded its execution deadline",
    ),
    "MIGRATION_WORKER_CRASHED": _WorkerErrorContract(
        ErrorCategory.DATABASE,
        "database migration worker exited before returning a result",
    ),
}


@dataclass(frozen=True, slots=True)
class _WorkerProcessBinding:
    """Immutable process identity accepted by one exit watcher."""

    process_id: int


class _WorkerExitWatcher:
    """Observe a child exit without reaping its identity before group cleanup."""

    def __init__(self, process_id: int) -> None:
        self._binding = _WorkerProcessBinding(process_id)
        self._exited = False
        self._kqueue: Any | None = None
        self._pidfd: int | None = None
        self._close_error: BaseException | None = None
        self._initialize_platform_handle()

    def _initialize_platform_handle(self) -> None:
        process_id = self._binding.process_id
        if sys.platform == "darwin" and hasattr(select, "kqueue"):
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                {signal.SIGINT, signal.SIGTERM},
            )
            try:
                kernel_queue = select.kqueue()
                self._kqueue = kernel_queue
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            try:
                event = select.kevent(
                    process_id,
                    filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                    fflags=select.KQ_NOTE_EXIT,
                )
                kernel_queue.control([event], 0, 0)
            except BaseException as initialization_error:
                try:
                    self.close()
                except BaseException as close_error:
                    # close() made the ambiguity sticky before raising. Preserve
                    # the earlier initialization error; later cleanup will still
                    # fail closed on the retained watcher state.
                    raise initialization_error from close_error
                raise initialization_error
            return
        pidfd_open = getattr(os, "pidfd_open", None)
        if sys.platform.startswith("linux") and pidfd_open is not None:
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                {signal.SIGINT, signal.SIGTERM},
            )
            try:
                process_handle = int(pidfd_open(process_id))
                self._pidfd = process_handle
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            return
        raise OSError("platform cannot observe a worker exit without reaping it")

    def wait(self, timeout: float) -> bool:
        if self._exited:
            return True
        bounded_timeout = max(0.0, timeout)
        if self._kqueue is not None:
            events = self._kqueue.control(None, 1, bounded_timeout)
            if not events:
                return False
            if type(events) is not list or len(events) != 1:
                raise OSError("invalid worker exit event")
            event = events[0]
            try:
                invalid = (
                    event.ident != self._binding.process_id
                    or event.filter != select.KQ_FILTER_PROC
                    or event.fflags != select.KQ_NOTE_EXIT
                    or event.flags & select.KQ_EV_ERROR
                )
            except (AttributeError, TypeError):
                raise OSError("invalid worker exit event") from None
            if invalid:
                raise OSError("invalid worker exit event")
            self._exited = True
        elif self._pidfd is not None:
            readable, _, _ = select.select([self._pidfd], [], [], bounded_timeout)
            self._exited = bool(readable)
        return self._exited

    def close(self) -> None:
        if self._close_error is not None:
            raise self._close_error
        first_error: BaseException | None = None
        kernel_queue = self._kqueue
        self._kqueue = None
        if kernel_queue is not None:
            try:
                kernel_queue.close()
            except BaseException as exc:
                first_error = exc
        process_handle = self._pidfd
        self._pidfd = None
        if process_handle is not None:
            try:
                os.close(process_handle)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            self._close_error = first_error
            raise first_error


def _reserve_worker_exit_watcher(process_id: int) -> _WorkerExitWatcher:
    """Publish watcher ownership before any platform handle operation can interrupt."""

    watcher = _WorkerExitWatcher.__new__(_WorkerExitWatcher)
    watcher._binding = _WorkerProcessBinding(process_id)
    watcher._exited = False
    watcher._kqueue = None
    watcher._pidfd = None
    watcher._close_error = None
    return watcher


def _initialize_reserved_worker_exit_watcher(watcher: _WorkerExitWatcher) -> None:
    """Initialize one watcher whose exact ownership is already retained."""

    watcher._initialize_platform_handle()


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


@dataclass(slots=True)
class _WorkerMemberProcessHandle:
    """Prepublished ownership for one temporary Linux member pidfd."""

    descriptor: int | None = None
    close_error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _WorkerSessionSnapshot:
    leader_state: str
    members: tuple[_WorkerSessionMember, ...]


class _WorkerCleanupFailure(Exception):
    """Private marker whose platform details must never cross the boundary."""


def _worker_command() -> list[str]:
    return [sys.executable, "-m", "invoice_agents.db.migration_worker"]


def _serialize_settings(settings: Settings | None) -> dict[str, object] | None:
    if settings is None:
        return None
    return serialize_wire_settings(settings)


def _protocol_error(stop_reason: str, _message: str) -> DatabaseVerificationError:
    contract = _PARENT_WORKER_ERROR_CONTRACTS.get(stop_reason)
    if contract is None:
        contract = _PARENT_WORKER_ERROR_CONTRACTS["MIGRATION_WORKER_PROTOCOL_INVALID"]
        stop_reason = "MIGRATION_WORKER_PROTOCOL_INVALID"
    return DatabaseVerificationError(
        contract.category,
        contract.message,
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


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate worker response key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite worker response number")


def _canonical_count_details(
    details: object,
    fields: tuple[str, ...],
) -> dict[str, int] | None:
    if type(details) is not dict or set(details) != set(fields):
        return None
    canonical: dict[str, int] = {}
    for detail_field in fields:
        value = details[detail_field]
        if type(value) is not int or value < 0 or value > 2**63 - 1:
            return None
        canonical[detail_field] = value
    return canonical


def _canonical_domain_message(
    stop_reason: str,
    contract: _WorkerErrorContract,
    details: dict[str, int] | None,
) -> str:
    if stop_reason != "AUTHORIZATION_RECONCILIATION_REQUIRED":
        return contract.message
    assert details is not None
    rendered = ", ".join(f"{field}={details[field]}" for field in contract.count_detail_fields)
    return f"{contract.message} ({rendered})"


def _canonical_worker_domain_error(
    *,
    category: object,
    stop_reason: object,
    details: object,
    discard_uncontracted_details: bool,
) -> dict[str, object] | None:
    if type(stop_reason) is not str:
        return None
    contract = _WORKER_DOMAIN_ERROR_CONTRACTS.get(stop_reason)
    if contract is None:
        return None
    category_value = category.value if isinstance(category, ErrorCategory) else category
    if type(category_value) is not str or category_value != contract.category.value:
        return None
    canonical_details: dict[str, int] | None
    if contract.count_detail_fields:
        canonical_details = _canonical_count_details(details, contract.count_detail_fields)
        if canonical_details is None:
            return None
    else:
        if not discard_uncontracted_details and details is not None:
            return None
        canonical_details = None
    return {
        "category": contract.category.value,
        "message": _canonical_domain_message(stop_reason, contract, canonical_details),
        "stop_reason": stop_reason,
        "details": canonical_details,
    }


def _encode_expected_worker_failure(
    *,
    category: object,
    stop_reason: object,
    details: object,
) -> dict[str, object] | None:
    """Build only a parent-defined domain payload from a trusted in-worker exception."""

    return _canonical_worker_domain_error(
        category=category,
        stop_reason=stop_reason,
        details=details,
        discard_uncontracted_details=True,
    )


def _protocol_failure_payload() -> dict[str, object]:
    contract = _PARENT_WORKER_ERROR_CONTRACTS["MIGRATION_WORKER_PROTOCOL_INVALID"]
    return {
        "category": contract.category.value,
        "message": contract.message,
        "stop_reason": "MIGRATION_WORKER_PROTOCOL_INVALID",
        "details": None,
    }


def _decode_response(encoded: bytes) -> list[int]:
    if not encoded or len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        ) from None
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
    canonical = _canonical_worker_domain_error(
        category=error["category"],
        stop_reason=error["stop_reason"],
        details=error["details"],
        discard_uncontracted_details=False,
    )
    if canonical is None or error != canonical:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    raise DatabaseVerificationError(
        ErrorCategory(str(canonical["category"])),
        str(canonical["message"]),
        stop_reason=str(canonical["stop_reason"]),
        details=canonical["details"] if isinstance(canonical["details"], dict) else None,
    )


def _capture_worker_session(
    process: subprocess.Popen[bytes],
    retained_worker: _WorkerSession,
) -> _WorkerSession:
    """Mutate one prepublished session while retaining every acquired handle."""

    process_id = process.pid
    if (
        retained_worker.process is not process
        or retained_worker.process_id != process_id
        or retained_worker.process_group_id != process_id
        or retained_worker.session_id != process_id
        or retained_worker.exit_watcher is not None
        or retained_worker.watcher_initialized
        or retained_worker.identity_verified
        or retained_worker.cleaned
    ):
        raise RuntimeError("worker capture did not receive its exact reserved session")

    exit_watcher = _reserve_worker_exit_watcher(process_id)
    retained_worker.exit_watcher = exit_watcher
    try:
        _initialize_reserved_worker_exit_watcher(exit_watcher)
    except Exception:
        # The Popen contract already reserved PID == PGID == SID. Keep that
        # unreaped identity and the prepublished partial watcher for cleanup.
        retained_worker.watcher_initialized = False
    else:
        retained_worker.watcher_initialized = True
    try:
        process_group_id = os.getpgid(process_id)
        session_id = os.getsid(process_id)
    except ProcessLookupError:
        # Popen(start_new_session=True) completed setsid before exec. A fast child may
        # already be an unreaped zombie on Darwin, where getpgid/getsid return ESRCH.
        process_group_id = process_id
        session_id = process_id
        retained_worker.identity_verified = True
    except OSError:
        # Popen completed start_new_session=True before returning. Retain that
        # reserved identity for strict cleanup, but never accept the worker result.
        process_group_id = process_id
        session_id = process_id
        retained_worker.identity_verified = False
    else:
        retained_worker.identity_verified = True
    retained_worker.process_group_id = process_group_id
    retained_worker.session_id = session_id
    if process_group_id != process_id or session_id != process_id:
        retained_worker.watcher_initialized = False
        retained_worker.identity_verified = False
        retained_worker.process_group_id = process_id
        retained_worker.session_id = process_id
        if retained_worker.exit_watcher is not None:
            exit_watcher.close()
            retained_worker.exit_watcher = None
    return retained_worker


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


def _initialize_reserved_migration_process(
    process: subprocess.Popen[bytes],
    *args: object,
    **kwargs: object,
) -> None:
    """Initialize one already-retained migration worker handle in place."""

    subprocess.Popen.__init__(process, *args, **kwargs)  # type: ignore[call-overload]


def _reserved_process_has_native_child(process: subprocess.Popen[bytes]) -> bool:
    """Return whether initialization created an owned, unreaped native child."""

    process_id = getattr(process, "pid", None)
    return (
        getattr(process, "_child_created", False) is True
        and type(process_id) is int
        and process_id > 0
        and getattr(process, "returncode", None) is None
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


def _canonical_process_table_rows(
    stdout: bytes,
    stderr: bytes,
) -> tuple[tuple[int, int, int | None, str], ...]:
    """Parse one complete, bounded, platform-canonical process table."""

    if stderr:
        raise _WorkerCleanupFailure
    if (
        not stdout
        or len(stdout) > _PROCESS_TABLE_MAX_BYTES
        or not stdout.endswith(b"\n")
        or b"\r" in stdout
    ):
        raise _WorkerCleanupFailure
    if sys.platform == "darwin":
        row_pattern = re.compile(rb" *([1-9][0-9]*) +([1-9][0-9]*) +([^\x00-\x20\x7f-\xff]+)( *)")
    elif sys.platform.startswith("linux"):
        row_pattern = re.compile(
            rb" *([1-9][0-9]*) +([1-9][0-9]*) +([1-9][0-9]*) +"
            rb"([^\x00-\x20\x7f-\xff]+)( *)"
        )
    else:
        raise _WorkerCleanupFailure

    rows: list[tuple[int, int, int | None, str]] = []
    seen: set[int] = set()
    for raw_line in stdout.splitlines():
        matched = row_pattern.fullmatch(raw_line)
        if matched is None:
            raise _WorkerCleanupFailure
        fields = matched.groups()
        if sys.platform == "darwin":
            raw_process_id, raw_group_id, raw_state, raw_padding = fields
            raw_session_id: bytes | None = None
        else:
            (
                raw_process_id,
                raw_group_id,
                raw_session_id,
                raw_state,
                raw_padding,
            ) = fields
        try:
            state = raw_state.decode("ascii")
        except UnicodeDecodeError as exc:
            raise _WorkerCleanupFailure from exc
        if sys.platform == "darwin":
            valid_state = (
                re.fullmatch(
                    r"[?IRSTUZ](?:>|W)?(?:<|N)?(?:A|S)?X?E?V?L?s?\+?",
                    state,
                )
                is not None
            )
            expected_padding = max(0, 4 - len(state))
        else:
            valid_state = (
                re.fullmatch(
                    r"[DIKPRStTWXxZ](?:<|N)?L?s?l?\+?",
                    state,
                )
                is not None
            )
            expected_padding = 0
        if not valid_state or len(raw_padding) != expected_padding:
            raise _WorkerCleanupFailure
        process_id = _canonical_process_identifier(raw_process_id.decode("ascii"))
        group_id = _canonical_process_identifier(raw_group_id.decode("ascii"))
        session_id = (
            _canonical_process_identifier(raw_session_id.decode("ascii"))
            if raw_session_id is not None
            else None
        )
        if process_id in seen:
            raise _WorkerCleanupFailure
        seen.add(process_id)
        rows.append((process_id, group_id, session_id, state))
    return tuple(sorted(rows))


def _worker_session_snapshot(worker: _WorkerSession) -> _WorkerSessionSnapshot:
    """Strictly prove the reserved leader row and enumerate its session members."""

    command = _process_status_command()
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            check=True,
            capture_output=True,
            env={**os.environ, "LANG": "C", "LC_ALL": "C"},
            text=False,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _WorkerCleanupFailure from exc
    if (
        completed.args != command
        or type(completed.returncode) is not int
        or completed.returncode != 0
        or type(completed.stdout) is not bytes
        or type(completed.stderr) is not bytes
    ):
        raise _WorkerCleanupFailure
    rows = _canonical_process_table_rows(completed.stdout, completed.stderr)
    leader_state: str | None = None
    members: list[_WorkerSessionMember] = []
    for process_id, listed_group_id, listed_session_id, state in rows:
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
    # NOTE_EXIT and a zombie process-table row are independent positive exit
    # proofs. Do not query a possibly faulted watcher after the process table
    # already proved the unreaped leader is a zombie.
    if snapshot.leader_state.startswith("Z"):
        return True
    return _watcher_observed_exit(worker) is True


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
    """Signal only groups whose identities are bound for the exact kernel action."""

    if sys.platform.startswith("linux"):
        _signal_worker_session_members_with_pidfds(worker, members, signal_number)
        return
    if sys.platform != "darwin":
        raise _WorkerCleanupFailure
    if worker.process_id != worker.process_group_id or worker.process_id != worker.session_id:
        raise _WorkerCleanupFailure
    groups: dict[int, list[int]] = {}
    for member in members:
        groups.setdefault(member.process_group_id, []).append(member.process_id)
    for process_group_id in sorted(groups):
        if process_group_id != worker.process_group_id:
            # Darwin has no supported public pidfd-like action. A descendant-created
            # numeric PGID can be released and reused after any getsid/getpgid check,
            # so retain the unreaped session and report cleanup failure instead.
            raise _WorkerCleanupFailure
        try:
            # Popen(start_new_session=True) established PID == PGID == SID. The
            # unreaped leader keeps this original PGID reserved through killpg.
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
        owned_handle = _WorkerMemberProcessHandle()
        first_error: BaseException | None = None
        disappeared = False
        try:
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                {signal.SIGINT, signal.SIGTERM},
            )
            acquisition_error: BaseException | None = None
            restoration_error: BaseException | None = None
            try:
                try:
                    acquired_handle = int(pidfd_open(member.process_id))
                    if acquired_handle < 0:
                        raise OSError("invalid member pidfd")
                    owned_handle.descriptor = acquired_handle
                except ProcessLookupError:
                    disappeared = True
                except BaseException as exc:
                    acquisition_error = exc
            finally:
                try:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                except BaseException as exc:
                    restoration_error = exc
            if acquisition_error is not None:
                first_error = acquisition_error
            elif restoration_error is not None:
                first_error = restoration_error

            if first_error is None and not disappeared:
                published_handle = owned_handle.descriptor
                if published_handle is None:
                    raise _WorkerCleanupFailure
                try:
                    first_session_id = os.getsid(member.process_id)
                    current_group_id = os.getpgid(member.process_id)
                    current_session_id = os.getsid(member.process_id)
                except ProcessLookupError:
                    disappeared = True
                if not disappeared and (
                    first_session_id != worker.session_id
                    or current_session_id != worker.session_id
                    or current_group_id != member.process_group_id
                ):
                    raise _WorkerCleanupFailure
                if not disappeared:
                    pidfd_send_signal(published_handle, signal_number, None, 0)
        except ProcessLookupError:
            disappeared = True
        except OSError as exc:
            first_error = _WorkerCleanupFailure()
            first_error.__cause__ = exc
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        finally:
            retired_handle = owned_handle.descriptor
            owned_handle.descriptor = None
            if retired_handle is not None:
                try:
                    os.close(retired_handle)
                except BaseException as exc:
                    owned_handle.close_error = exc
                    _WORKER_RESOURCE_CLEANUP_POISONED.set()
                    if first_error is None:
                        first_error = _WorkerCleanupFailure()
                        first_error.__cause__ = exc
        if first_error is not None:
            raise first_error
        if disappeared:
            continue


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


def _finalize_empty_worker_session(
    worker: _WorkerSession,
    *,
    session_empty: bool,
) -> None:
    """Recheck, close, and exactly reap a session already driven to extinction."""

    first_error: BaseException | None = None

    def preserve_first(error: BaseException) -> None:
        nonlocal first_error
        if first_error is None:
            first_error = error

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
        except BaseException as exc:
            _WORKER_RESOURCE_CLEANUP_POISONED.set()
            preserve_first(exc)
    try:
        stdout = getattr(worker.process, "stdout", None)
        if stdout is not None and not stdout.closed:
            stdout.close()
    except BaseException as exc:
        _WORKER_RESOURCE_CLEANUP_POISONED.set()
        preserve_first(exc)
    wait_completed = False
    try:
        returncode = worker.process.wait(
            timeout=max(_WORKER_POLL_SECONDS, _WORKER_SHUTDOWN_SECONDS)
        )
        wait_completed = (
            type(returncode) is int and getattr(worker.process, "returncode", None) == returncode
        )
        if not wait_completed:
            preserve_first(_WorkerCleanupFailure())
    except BaseException as exc:
        preserve_first(exc)
    if wait_completed:
        worker.cleaned = True
    if first_error is not None:
        raise first_error


def _cleanup_worker_session(worker: _WorkerSession) -> None:
    """Retain ownership until the reserved session is empty and exactly reaped."""

    first_error: BaseException | None = None

    def preserve_first(error: BaseException) -> None:
        nonlocal first_error
        if first_error is None:
            first_error = error

    with worker.cleanup_lock:
        while not worker.cleaned:
            try:
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
                _finalize_empty_worker_session(worker, session_empty=session_empty)
            except BaseException as exc:
                preserve_first(exc)
                if not worker.cleaned:
                    try:
                        time.sleep(_WORKER_POLL_SECONDS)
                    except BaseException as sleep_error:
                        preserve_first(sleep_error)
        if first_error is not None:
            raise first_error


def _cleanup_cooperative_worker_session(worker: _WorkerSession) -> None:
    """Wait for a trusted owner to reap escaped authority without killing it."""

    first_error: BaseException | None = None

    def preserve_first(error: BaseException) -> None:
        nonlocal first_error
        if first_error is None:
            first_error = error

    with worker.cleanup_lock:
        while not worker.cleaned:
            try:
                empty_snapshots = 0
                while empty_snapshots < 2:
                    snapshot = _worker_session_snapshot(worker)
                    leader_exited = _leader_exit_is_proven(worker, snapshot)
                    if leader_exited and not snapshot.members:
                        empty_snapshots += 1
                        continue
                    empty_snapshots = 0
                    if not leader_exited:
                        _signal_worker_leader(worker, snapshot, signal.SIGTERM)
                    elif sys.platform.startswith("linux"):
                        _signal_worker_session_members_with_pidfds(
                            worker,
                            snapshot.members,
                            signal.SIGKILL,
                        )
                    elif sys.platform != "darwin":
                        raise _WorkerCleanupFailure
                    watcher = worker.exit_watcher
                    if not leader_exited and watcher is not None:
                        watcher.wait(_WORKER_POLL_SECONDS)
                    else:
                        time.sleep(_WORKER_POLL_SECONDS)
                _finalize_empty_worker_session(worker, session_empty=True)
            except BaseException as exc:
                preserve_first(exc)
                try:
                    time.sleep(_WORKER_POLL_SECONDS)
                except BaseException as sleep_error:
                    preserve_first(sleep_error)
        if first_error is not None:
            raise first_error


def _stop_worker(worker: _WorkerSession) -> DatabaseVerificationError | None:
    """Strictly clean the worker before returning a chainless public error."""

    try:
        _cleanup_worker_session(worker)
    except Exception:
        return _cleanup_protocol_error()
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
    if _worker_resource_cleanup_is_poisoned():
        raise _cleanup_protocol_error()
    request = _encode_request(path, kind, settings)
    process: subprocess.Popen[bytes] | None = None
    worker: _WorkerSession | None = None
    launch_error: BaseException | None = None
    initialization_completed = False
    classification_completed = False
    native_child = False
    try:
        process = cast(
            "subprocess.Popen[bytes]",
            subprocess.Popen.__new__(subprocess.Popen),
        )
        process._child_created = False  # type: ignore[attr-defined]
        with tempfile.TemporaryFile() as request_stream:
            request_stream.write(request)
            request_stream.seek(0)
            _initialize_reserved_migration_process(
                process,
                _worker_command(),
                stdin=request_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=sanitized_worker_environment(),
            )
        initialization_completed = True
        native_child = _reserved_process_has_native_child(process)
        classification_completed = True
        if not native_child:
            raise RuntimeError("migration worker initialization produced no native child")
        worker = _uncertain_worker_session(process)
        worker = _capture_worker_session(process, worker)
    except BaseException as exc:
        launch_error = exc
        if process is not None:
            if initialization_completed and not classification_completed:
                # A completed Popen initializer is itself positive native-child
                # ownership evidence. Classification was the interrupted step.
                native_child = True
            elif not classification_completed:
                while True:
                    try:
                        native_child = _reserved_process_has_native_child(process)
                        classification_completed = True
                        break
                    except BaseException:
                        try:
                            time.sleep(_WORKER_POLL_SECONDS)
                        except BaseException:
                            continue
            if native_child:
                if worker is None:
                    worker = _uncertain_worker_session(process)
            else:
                process._child_created = False  # type: ignore[attr-defined]
                process = None
    if worker is None:
        if launch_error is not None and not isinstance(launch_error, Exception):
            raise launch_error
        raise _protocol_error(
            "MIGRATION_WORKER_START_FAILED",
            "database migration worker could not be started",
        )
    start_failed = (
        launch_error is not None or not worker.watcher_initialized or not worker.identity_verified
    )
    if start_failed:
        cleanup_control: BaseException | None = None
        try:
            cleanup_error = _stop_worker(worker)
        except BaseException as exc:
            cleanup_error = None
            cleanup_control = exc
        if launch_error is not None and not isinstance(launch_error, Exception):
            raise launch_error
        if cleanup_control is not None:
            raise cleanup_control
        if cleanup_error is not None:
            raise cleanup_error
        raise _protocol_error(
            "MIGRATION_WORKER_START_FAILED",
            "database migration worker could not be started",
        )
    response_error: BaseException | None = None
    stdout: bytes | None = None
    try:
        stdout = _read_bounded_worker_response(worker)
    except BaseException as exc:
        response_error = exc
    cleanup_control = None
    try:
        cleanup_error = _stop_worker(worker)
    except BaseException as exc:
        cleanup_error = None
        cleanup_control = exc
    if response_error is not None and not isinstance(response_error, Exception):
        raise response_error
    if cleanup_control is not None:
        raise cleanup_control
    if cleanup_error is not None:
        raise cleanup_error
    if response_error is not None:
        raise response_error
    if worker.process.returncode != 0:
        raise _protocol_error(
            "MIGRATION_WORKER_CRASHED",
            "database migration worker exited before returning a result",
        )
    assert stdout is not None
    return _decode_response(stdout)


def _copy_public_error(error: DatabaseVerificationError) -> DatabaseVerificationError:
    """Reconstruct only an exact parent-owned error contract without a private chain."""

    stop_reason = error.stop_reason
    if type(stop_reason) is str:
        parent_contract = _PARENT_WORKER_ERROR_CONTRACTS.get(stop_reason)
        if parent_contract is not None and (
            error.category is parent_contract.category
            and error.message == parent_contract.message
            and error.case_id is None
            and error.provider_request_id is None
            and error.details is None
        ):
            return DatabaseVerificationError(
                parent_contract.category,
                parent_contract.message,
                stop_reason=stop_reason,
            )
        canonical = _canonical_worker_domain_error(
            category=error.category,
            stop_reason=stop_reason,
            details=error.details,
            discard_uncontracted_details=False,
        )
        if (
            canonical is not None
            and error.message == canonical["message"]
            and error.case_id is None
            and error.provider_request_id is None
        ):
            details = canonical["details"]
            return DatabaseVerificationError(
                ErrorCategory(str(canonical["category"])),
                str(canonical["message"]),
                stop_reason=str(canonical["stop_reason"]),
                details=dict(details) if isinstance(details, dict) else None,
            )
    return _protocol_error(
        "MIGRATION_WORKER_PROTOCOL_INVALID",
        "database migration worker returned an invalid bounded response",
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
    payload = json.loads(
        encoded,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if type(payload) is not dict or set(payload) != {
        "protocol_version",
        "path",
        "kind",
        "settings",
    }:
        raise ValueError("invalid worker request shape")
    if (
        type(payload["protocol_version"]) is not int
        or payload["protocol_version"] != MIGRATION_WORKER_PROTOCOL_VERSION
    ):
        raise ValueError("invalid worker protocol version")
    if not isinstance(payload["path"], str) or not 1 <= len(payload["path"]) <= 16_384:
        raise ValueError("invalid worker path")
    if not isinstance(payload["kind"], str):
        raise ValueError("invalid worker database kind")
    kind = DatabaseKind(payload["kind"])
    settings_payload = payload["settings"]
    if settings_payload is None:
        settings = None
    elif type(settings_payload) is dict:
        settings = decode_wire_settings(settings_payload)
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
