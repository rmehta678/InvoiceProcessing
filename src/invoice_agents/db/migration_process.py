"""Bounded subprocess boundary for descriptor-isolated SQLite migrations."""

from __future__ import annotations

import json
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
from invoice_agents.wire_settings import decode_wire_settings, serialize_wire_settings
from invoice_agents.worker_environment import sanitized_worker_environment

MIGRATION_WORKER_PROTOCOL_VERSION = 1
MIGRATION_WORKER_MAX_MESSAGE_BYTES = 65_536
MIGRATION_WORKER_TIMEOUT_SECONDS = 120.0
_WORKER_SHUTDOWN_SECONDS = 2.0
_WORKER_POLL_SECONDS = 0.05
_WORKER_QUARANTINE_RETRY_ATTEMPTS = 3
_WORKER_QUARANTINE_RETRY_SECONDS = 0.1


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
    # NOTE_EXIT and a zombie process-table row are independent positive exit
    # proofs. Their visibility is not atomic, so a negative observation from
    # either source cannot invalidate a positive observation from the other.
    return zombie or observed is True


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
                env=sanitized_worker_environment(),
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
