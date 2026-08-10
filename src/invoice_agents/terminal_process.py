"""Terminable subprocess boundary for terminal case persistence."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from invoice_agents.config import Settings
from invoice_agents.db.migration_process import (
    _capture_worker_session,
    _cleanup_worker_session,
    _poison_worker_resource_cleanup,
    _reserved_process_has_native_child,
    _stop_worker,
    _uncertain_worker_session,
    _worker_resource_cleanup_is_poisoned,
    _WorkerSession,
)
from invoice_agents.db.store import ExecutionClaim, validate_execution_claim
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult
from invoice_agents.observability.audit import sanitize_case_result
from invoice_agents.wire_settings import decode_wire_settings, serialize_wire_settings
from invoice_agents.worker_environment import sanitized_worker_environment

TERMINAL_WORKER_PROTOCOL_VERSION = 2
TERMINAL_WORKER_MAX_MESSAGE_BYTES = 2_097_152
_TERMINAL_WORKER_POLL_SECONDS = 0.02
TerminalMode = Literal[
    "cancel_unstarted",
    "finish",
    "inspect_claim",
    "publish_cancel_recovery",
    "update",
]
TerminalEvidenceState = Literal["DURABLE_DATABASE_RESULT", "RECOVERABLE_RUNNING"]
_TERMINAL_ERROR_CODES = frozenset(
    {
        "EVIDENCE_AUTHORITY_MISSING",
        "EXECUTION_AUTHORITY_CORRUPT",
        "PERSISTED_RESULT_INVALID",
        "STALE_EXECUTION_CLAIM",
        "TERMINAL_DURABILITY_UNRESOLVED",
        "TERMINAL_RECOVERY_ARTIFACT_FAILED",
        "TERMINAL_WORKER_FAILED",
        "TERMINAL_WORKER_TIMEOUT",
    }
)
_RECOVERY_WORKER_ERROR_CODES = frozenset(
    {
        "EVIDENCE_AUTHORITY_MISSING",
        "EXECUTION_AUTHORITY_CORRUPT",
        "PERSISTED_RESULT_INVALID",
        "STALE_EXECUTION_CLAIM",
        "TERMINAL_DURABILITY_UNRESOLVED",
        "TERMINAL_WORKER_FAILED",
        "TERMINAL_WORKER_TIMEOUT",
    }
)


@dataclass(frozen=True, slots=True)
class TerminalProcessOutcome:
    """One parent-owned worker outcome after its whole session is proven empty."""

    result: CaseResult | None
    error_code: str | None
    evidence_state: TerminalEvidenceState | None = None
    evidence_result: CaseResult | None = None


@dataclass(slots=True)
class _TerminalCleanupEnvelope:
    """Pre-spawn ownership retained until cleanup or sticky poison is proven."""

    process: subprocess.Popen[bytes]
    worker: _WorkerSession
    native_child: bool = False
    ownership_observed: bool = False
    poison_proven: bool = False
    cleanup_proven: bool = False


def _terminal_worker_command() -> list[str]:
    return [sys.executable, "-m", "invoice_agents.terminal_worker"]


def _initialize_reserved_terminal_process(
    process: subprocess.Popen[bytes],
    *args: object,
    **kwargs: object,
) -> None:
    """Initialize one already-retained terminal worker handle in place."""

    subprocess.Popen.__init__(process, *args, **kwargs)  # type: ignore[call-overload]


def _reserve_terminal_cleanup_session(
    process: subprocess.Popen[bytes],
) -> _WorkerSession:
    """Prepublish exact native-child cleanup ownership before capture can fail."""

    process_id = process.pid
    return _WorkerSession(
        process=process,
        process_id=process_id,
        process_group_id=process_id,
        session_id=process_id,
        exit_watcher=None,
        watcher_initialized=False,
        identity_verified=False,
    )


def _bind_terminal_cleanup_session(
    worker: _WorkerSession,
    process: subprocess.Popen[bytes],
) -> None:
    """Bind one pre-spawn cleanup reservation to its published native identity."""

    process_id = process.pid
    worker.process = process
    worker.process_id = process_id
    worker.process_group_id = process_id
    worker.session_id = process_id


def _terminal_cleanup_binding_is_exact(
    worker: _WorkerSession,
    process: subprocess.Popen[bytes],
) -> bool:
    process_id = getattr(process, "pid", None)
    return (
        type(process_id) is int
        and process_id > 0
        and worker.process is process
        and worker.process_id == process_id
        and worker.process_group_id == process_id
        and worker.session_id == process_id
    )


def _stop_terminal_worker_owned(
    worker: _WorkerSession,
) -> tuple[bool, BaseException | None]:
    """Keep cleaning after control until extinction or sticky poison is proven."""

    cleanup_failed = False
    cleanup_control: BaseException | None = None
    try:
        cleanup_failed = _stop_worker(worker) is not None
    except BaseException as exc:
        cleanup_control = exc
    if not worker.cleaned and not _worker_resource_cleanup_is_poisoned():
        try:
            cleanup_failed = _stop_worker(worker) is not None or cleanup_failed
        except BaseException as exc:
            cleanup_control = _select_classification_error(cleanup_control, exc)
    if not worker.cleaned and not _worker_resource_cleanup_is_poisoned():
        try:
            _cleanup_worker_session(worker)
        except BaseException as exc:
            cleanup_control = _select_classification_error(cleanup_control, exc)
    if not worker.cleaned and not _worker_resource_cleanup_is_poisoned():
        _poison_worker_resource_cleanup()
    if not worker.cleaned:
        cleanup_failed = True
    return cleanup_failed, cleanup_control


def _publish_cleanup_poison_until_proven() -> BaseException | None:
    """Publish sticky poison despite control, or terminate before unsafe return."""

    retained_control: BaseException | None = None

    def preserve(error: BaseException) -> None:
        nonlocal retained_control
        if retained_control is None or (
            isinstance(retained_control, Exception)
            and not isinstance(error, Exception)
        ):
            retained_control = error

    for _attempt in range(8):
        try:
            if _worker_resource_cleanup_is_poisoned():
                return retained_control
        except BaseException as exc:
            preserve(exc)
        try:
            _poison_worker_resource_cleanup()
        except BaseException as exc:
            preserve(exc)
        try:
            if _worker_resource_cleanup_is_poisoned():
                return retained_control
        except BaseException as exc:
            preserve(exc)
    # The process cannot safely admit more work or expose a catchable failure
    # without either child-extinction or sticky poison proof.
    os._exit(70)


def _finalize_terminal_cleanup_envelope(
    envelope: _TerminalCleanupEnvelope,
) -> tuple[bool, BaseException | None]:
    """Contain every finalizer failure until owned cleanup is proven or poisoned."""

    retained_control: BaseException | None = None

    def preserve(error: BaseException) -> None:
        nonlocal retained_control
        if retained_control is None or (
            isinstance(retained_control, Exception)
            and not isinstance(error, Exception)
        ):
            retained_control = error

    process = envelope.process
    worker = envelope.worker
    process_id: object = None
    raw_native_child = False
    raw_inspection_completed = False
    try:
        process_id = getattr(process, "pid", None)
        raw_native_child = type(process_id) is int and process_id > 0
    except BaseException as exc:
        preserve(exc)
    else:
        raw_inspection_completed = True
    direct_binding_exact = (
        type(process_id) is int
        and process_id > 0
        and worker.process is process
        and worker.process_id == process_id
        and worker.process_group_id == process_id
        and worker.session_id == process_id
    )
    possible_native_child = (
        envelope.native_child
        or raw_native_child
        or direct_binding_exact
        or not raw_inspection_completed
    )
    if not possible_native_child:
        envelope.cleanup_proven = True
        return False, retained_control
    envelope.ownership_observed = True

    binding_exact = direct_binding_exact
    try:
        binding_exact = _terminal_cleanup_binding_is_exact(worker, process)
    except BaseException as exc:
        preserve(exc)
    if not binding_exact:
        try:
            _bind_terminal_cleanup_session(worker, process)
        except BaseException as exc:
            preserve(exc)
        try:
            process_id = getattr(process, "pid", None)
        except BaseException as exc:
            preserve(exc)
            process_id = None
        binding_exact = (
            type(process_id) is int
            and process_id > 0
            and worker.process is process
            and worker.process_id == process_id
            and worker.process_group_id == process_id
            and worker.session_id == process_id
        )
    if not binding_exact:
        poison_control = _publish_cleanup_poison_until_proven()
        if poison_control is not None:
            preserve(poison_control)
        envelope.poison_proven = True
        envelope.cleanup_proven = True
        return True, retained_control

    cleanup_failed = False
    try:
        cleanup_failed, cleanup_control = _stop_terminal_worker_owned(worker)
    except BaseException as exc:
        preserve(exc)
    else:
        if cleanup_control is not None:
            preserve(cleanup_control)
    if not worker.cleaned:
        try:
            _cleanup_worker_session(worker)
        except BaseException as exc:
            preserve(exc)
    if not worker.cleaned:
        cleanup_failed = True
        poison_control = _publish_cleanup_poison_until_proven()
        if poison_control is not None:
            preserve(poison_control)
        envelope.poison_proven = True
    envelope.cleanup_proven = worker.cleaned or envelope.poison_proven
    return cleanup_failed, retained_control


def _select_classification_error(
    prior: BaseException | None,
    later: BaseException,
) -> BaseException:
    """Preserve first failure unless later process control has higher precedence."""

    if prior is None or (
        isinstance(prior, Exception) and not isinstance(later, Exception)
    ):
        return later
    return prior


def _classify_reserved_terminal_process_until(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
) -> tuple[bool | None, BaseException | None]:
    """Bound native-child classification while retaining every control failure."""

    classification_error: BaseException | None = None
    while True:
        try:
            return _reserved_process_has_native_child(process), classification_error
        except BaseException as exc:
            classification_error = _select_classification_error(classification_error, exc)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, classification_error
        try:
            time.sleep(min(_TERMINAL_WORKER_POLL_SECONDS, remaining))
        except BaseException as exc:
            classification_error = _select_classification_error(classification_error, exc)


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate terminal worker key")
        payload[key] = value
    return payload


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite terminal worker number")


def _claim_payload(claim: ExecutionClaim) -> dict[str, object]:
    exact = validate_execution_claim(claim)
    return {
        "case_id": exact.case_id,
        "token": exact.token,
        "generation": exact.generation,
        "expires_at": exact.expires_at.isoformat(),
    }


def _encode_request(
    *,
    mode: TerminalMode,
    settings: Settings,
    claim: ExecutionClaim,
    started_at: datetime | None = None,
    result: CaseResult | None = None,
    worker_error_code: str | None = None,
) -> bytes:
    exact = validate_execution_claim(claim)
    if mode in {"cancel_unstarted", "publish_cancel_recovery"}:
        if (
            type(started_at) is not datetime
            or started_at.utcoffset() != timedelta(0)
            or started_at.astimezone(UTC).isoformat()
            != datetime.fromisoformat(started_at.isoformat()).astimezone(UTC).isoformat()
            or result is not None
        ):
            raise ValueError("cancel worker requires one canonical UTC start time")
        if mode == "cancel_unstarted" and worker_error_code is not None:
            raise ValueError("cancel worker cannot receive a prior worker error")
        if mode == "publish_cancel_recovery" and worker_error_code not in (
            _RECOVERY_WORKER_ERROR_CODES
        ):
            raise ValueError("recovery worker requires one stable prior worker error")
    elif mode == "inspect_claim":
        if started_at is not None or result is not None or worker_error_code is not None:
            raise ValueError("inspection worker accepts only exact claim authority")
    elif (
        result is None
        or started_at is not None
        or result.case_id != exact.case_id
        or worker_error_code is not None
    ):
        raise ValueError("finish worker requires one claim-bound case result")
    if result is not None:
        result = sanitize_case_result(result)
    payload = {
        "protocol_version": TERMINAL_WORKER_PROTOCOL_VERSION,
        "mode": mode,
        "settings": serialize_wire_settings(settings),
        "claim": _claim_payload(exact),
        "started_at": started_at.astimezone(UTC).isoformat() if started_at is not None else None,
        "result": result.model_dump(mode="json") if result is not None else None,
        "worker_error_code": worker_error_code,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > TERMINAL_WORKER_MAX_MESSAGE_BYTES:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "terminal persistence request exceeds its bounded protocol",
            case_id=exact.case_id,
            stop_reason="TERMINAL_WORKER_PROTOCOL_INVALID",
        ) from None
    return encoded


def decode_terminal_request(
    encoded: bytes,
) -> tuple[
    TerminalMode,
    Settings,
    ExecutionClaim,
    datetime | None,
    CaseResult | None,
    str | None,
]:
    """Strict child-side request decoder with no ambient settings fallback."""

    if not encoded or len(encoded) > TERMINAL_WORKER_MAX_MESSAGE_BYTES:
        raise ValueError("invalid terminal worker request size")
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if type(payload) is not dict or set(payload) != {
        "protocol_version",
        "mode",
        "settings",
        "claim",
        "started_at",
        "result",
        "worker_error_code",
    }:
        raise ValueError("invalid terminal worker request shape")
    if (
        type(payload["protocol_version"]) is not int
        or payload["protocol_version"] != TERMINAL_WORKER_PROTOCOL_VERSION
    ):
        raise ValueError("invalid terminal worker protocol version")
    mode = payload["mode"]
    if type(mode) is not str or mode not in {
        "cancel_unstarted",
        "finish",
        "inspect_claim",
        "publish_cancel_recovery",
        "update",
    }:
        raise ValueError("invalid terminal worker mode")
    mode = cast("TerminalMode", mode)
    settings_payload = payload["settings"]
    if type(settings_payload) is not dict:
        raise ValueError("invalid terminal worker settings")
    settings = decode_wire_settings(settings_payload)
    raw_claim = payload["claim"]
    if type(raw_claim) is not dict or set(raw_claim) != {
        "case_id",
        "token",
        "generation",
        "expires_at",
    }:
        raise ValueError("invalid terminal worker claim")
    if type(raw_claim["expires_at"]) is not str:
        raise ValueError("invalid terminal worker claim expiry")
    expires_at = datetime.fromisoformat(raw_claim["expires_at"])
    claim = validate_execution_claim(
        ExecutionClaim(
            case_id=raw_claim["case_id"],
            token=raw_claim["token"],
            generation=raw_claim["generation"],
            expires_at=expires_at,
        )
    )
    raw_started = payload["started_at"]
    raw_result = payload["result"]
    worker_error_code = payload["worker_error_code"]
    if mode in {"cancel_unstarted", "publish_cancel_recovery"}:
        if type(raw_started) is not str or raw_result is not None:
            raise ValueError("invalid cancel worker payload")
        started_at = datetime.fromisoformat(raw_started)
        if started_at.tzinfo is not UTC or started_at.isoformat() != raw_started:
            raise ValueError("invalid cancel worker start time")
        result = None
        if mode == "cancel_unstarted" and worker_error_code is not None:
            raise ValueError("invalid cancel worker error")
        if mode == "publish_cancel_recovery" and (
            type(worker_error_code) is not str
            or worker_error_code not in _RECOVERY_WORKER_ERROR_CODES
        ):
            raise ValueError("invalid recovery worker error")
    elif mode == "inspect_claim":
        if raw_started is not None or raw_result is not None or worker_error_code is not None:
            raise ValueError("invalid inspection worker payload")
        started_at = None
        result = None
    else:
        if (
            raw_started is not None
            or type(raw_result) is not dict
            or worker_error_code is not None
        ):
            raise ValueError("invalid finish worker payload")
        started_at = None
        # Pydantic's strict Python mode rejects JSON datetime/enum strings even
        # though they are their canonical wire representation.  Re-validate the
        # already shape-checked object through strict JSON mode so transport
        # scalars remain non-coercible while canonical model JSON is accepted.
        result = sanitize_case_result(
            CaseResult.model_validate_json(
                json.dumps(raw_result, ensure_ascii=True, separators=(",", ":")),
                strict=True,
            )
        )
        if result.case_id != claim.case_id:
            raise ValueError("terminal worker result is not claim-bound")
    return mode, settings, claim, started_at, result, worker_error_code


def encode_terminal_response(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > TERMINAL_WORKER_MAX_MESSAGE_BYTES:
        raise ValueError("terminal worker response exceeds its bound")
    return encoded


def _remaining_response_time(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    return remaining


def _read_response(worker: Any, *, deadline: float) -> bytes:
    _remaining_response_time(deadline)
    stdout = worker.process.stdout
    watcher = worker.exit_watcher
    if stdout is None or watcher is None:
        raise TimeoutError
    descriptor = stdout.fileno()
    _remaining_response_time(deadline)
    os.set_blocking(descriptor, False)
    _remaining_response_time(deadline)
    response = bytearray()
    with selectors.DefaultSelector() as selector:
        _remaining_response_time(deadline)
        selector.register(descriptor, selectors.EVENT_READ)
        _remaining_response_time(deadline)
        while True:
            remaining = _remaining_response_time(deadline)
            selected = selector.select(
                min(remaining, _TERMINAL_WORKER_POLL_SECONDS)
            )
            _remaining_response_time(deadline)
            if not selected:
                watcher_exited = watcher.wait(0)
                _remaining_response_time(deadline)
                if watcher_exited:
                    break
                continue
            chunk = os.read(
                descriptor,
                min(8_192, TERMINAL_WORKER_MAX_MESSAGE_BYTES + 1 - len(response)),
            )
            _remaining_response_time(deadline)
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > TERMINAL_WORKER_MAX_MESSAGE_BYTES:
                raise ValueError("oversize terminal worker response")
    remaining = _remaining_response_time(deadline)
    watcher_exited = watcher.wait(remaining)
    _remaining_response_time(deadline)
    if not watcher_exited:
        raise TimeoutError
    return bytes(response)


def _read_response_until(worker: Any, *, deadline: float) -> bytes:
    """Read only within the launch-to-response operation deadline."""

    _remaining_response_time(deadline)
    return _read_response(worker, deadline=deadline)


def _decode_response(
    encoded: bytes,
    *,
    expected_claim: ExecutionClaim,
) -> TerminalProcessOutcome:
    if not encoded or len(encoded) > TERMINAL_WORKER_MAX_MESSAGE_BYTES:
        raise ValueError("invalid terminal worker response size")
    expected_claim = validate_execution_claim(expected_claim)
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if (
        type(payload) is not dict
        or set(payload)
        != {"ok", "claim", "result", "error_code", "evidence_state", "evidence_result"}
        or type(payload.get("ok")) is not bool
    ):
        raise ValueError("invalid terminal worker response shape")
    raw_claim = payload["claim"]
    if type(raw_claim) is not dict or set(raw_claim) != {
        "case_id",
        "token",
        "generation",
        "expires_at",
    }:
        raise ValueError("invalid terminal worker response claim")
    raw_case_id = raw_claim["case_id"]
    raw_token = raw_claim["token"]
    raw_generation = raw_claim["generation"]
    raw_expires_at = raw_claim["expires_at"]
    if (
        type(raw_case_id) is not str
        or type(raw_token) is not str
        or type(raw_generation) is not int
        or type(raw_expires_at) is not str
    ):
        raise ValueError("invalid terminal worker response claim")
    try:
        echoed_claim = validate_execution_claim(
            ExecutionClaim(
                case_id=raw_case_id,
                token=raw_token,
                generation=raw_generation,
                expires_at=datetime.fromisoformat(raw_expires_at),
            )
        )
    except (InvoiceAgentsError, ValueError, TypeError):
        raise ValueError("invalid terminal worker response claim") from None
    if echoed_claim.expires_at.isoformat() != raw_expires_at or echoed_claim != expected_claim:
        raise ValueError("terminal worker response claim does not match request")
    raw_result = payload["result"]
    raw_error_code = payload["error_code"]
    if payload["ok"]:
        if raw_error_code is not None or (
            raw_result is not None and type(raw_result) is not dict
        ):
            raise ValueError("invalid terminal worker success response")
    elif raw_result is not None or (
        type(raw_error_code) is not str or raw_error_code not in _TERMINAL_ERROR_CODES
    ):
        raise ValueError("invalid terminal worker failure response")
    result = (
        CaseResult.model_validate_json(
            json.dumps(
                raw_result,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            strict=True,
        )
        if type(raw_result) is dict
        else None
    )
    raw_evidence_state = payload["evidence_state"]
    raw_evidence_result = payload["evidence_result"]
    if raw_evidence_state is None:
        if raw_evidence_result is not None:
            raise ValueError("invalid absent terminal evidence")
        evidence_state = None
        evidence_result = None
    elif raw_evidence_state == "RECOVERABLE_RUNNING":
        if raw_evidence_result is not None:
            raise ValueError("invalid recoverable terminal evidence")
        evidence_state = cast("TerminalEvidenceState", raw_evidence_state)
        evidence_result = None
    elif raw_evidence_state == "DURABLE_DATABASE_RESULT":
        if type(raw_evidence_result) is not dict:
            raise ValueError("invalid durable terminal evidence")
        evidence_state = cast("TerminalEvidenceState", raw_evidence_state)
        evidence_result = CaseResult.model_validate_json(
            json.dumps(
                raw_evidence_result,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            strict=True,
        )
    else:
        raise ValueError("invalid terminal evidence state")
    if result is not None and evidence_result is not None and (
        result.case_id != evidence_result.case_id
    ):
        raise ValueError("terminal result and evidence identities conflict")
    return TerminalProcessOutcome(
        result=result,
        error_code=cast("str | None", raw_error_code),
        evidence_state=evidence_state,
        evidence_result=evidence_result,
    )


def _run_terminal_process_owned_state(
    *,
    envelope: _TerminalCleanupEnvelope,
    request: bytes,
    operation_deadline: float,
    exact: ExecutionClaim,
) -> TerminalProcessOutcome:
    """Execute all post-reservation state under the caller's cleanup envelope."""

    process = envelope.process
    worker: _WorkerSession | None = envelope.worker
    launch_error: BaseException | None = None
    launch_failure_timed_out = False
    initialization_completed = False
    classification_completed = False
    native_child = False
    try:
        with tempfile.TemporaryFile() as request_stream:
            request_stream.write(request)
            request_stream.seek(0)
            command = _terminal_worker_command()
            environment = sanitized_worker_environment()
            if time.monotonic() >= operation_deadline:
                return TerminalProcessOutcome(None, "TERMINAL_WORKER_TIMEOUT")
            _initialize_reserved_terminal_process(
                process,
                command,
                stdin=request_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=environment,
            )
        initialization_completed = True
        assert worker is not None
        _bind_terminal_cleanup_session(worker, process)
        native_child = _reserved_process_has_native_child(process)
        envelope.native_child = native_child
        envelope.ownership_observed = native_child
        classification_completed = True
        if not native_child:
            raise RuntimeError("terminal worker initialization produced no native child")
        worker = _reserve_terminal_cleanup_session(process)
        envelope.worker = worker
        retained_worker = _uncertain_worker_session(process)
        worker = retained_worker
        envelope.worker = worker
        worker = _capture_worker_session(process, worker)
        envelope.worker = worker
    except BaseException as exc:
        launch_error = exc
        if initialization_completed and not classification_completed:
            native_child = True
        elif not classification_completed:
            classified, classification_error = _classify_reserved_terminal_process_until(
                process,
                deadline=operation_deadline,
            )
            if classification_error is not None:
                launch_error = _select_classification_error(
                    launch_error,
                    classification_error,
                )
            if classified is not None:
                process_id = getattr(process, "pid", None)
                native_child = classified or (
                    type(process_id) is int and process_id > 0
                )
                classification_completed = True
            else:
                process_id = getattr(process, "pid", None)
                native_child = type(process_id) is int and process_id > 0
        envelope.native_child = envelope.native_child or native_child
        envelope.ownership_observed = envelope.ownership_observed or native_child
        if native_child:
            assert worker is not None
            _bind_terminal_cleanup_session(worker, process)
            envelope.worker = worker
        else:
            process._child_created = False  # type: ignore[attr-defined]
            worker = None
        if isinstance(launch_error, Exception):
            launch_failure_timed_out = time.monotonic() >= operation_deadline
    if worker is None:
        if launch_error is not None and not isinstance(launch_error, Exception):
            raise launch_error
        if launch_failure_timed_out:
            return TerminalProcessOutcome(None, "TERMINAL_WORKER_TIMEOUT")
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "terminal helper could not be started",
            case_id=exact.case_id,
            stop_reason="TERMINAL_WORKER_START_FAILED",
        ) from None

    start_failed = (
        launch_error is not None or not worker.watcher_initialized or not worker.identity_verified
    )
    response: bytes | None = None
    response_failure: Literal["timeout", "protocol", "crash"] | None = None
    response_error: BaseException | None = None
    if not start_failed:
        try:
            response = _read_response_until(worker, deadline=operation_deadline)
        except TimeoutError:
            response_failure = "timeout"
        except (OSError, UnicodeError, ValueError):
            response_failure = (
                "timeout" if time.monotonic() >= operation_deadline else "protocol"
            )
        except BaseException as exc:
            if isinstance(exc, Exception) and time.monotonic() >= operation_deadline:
                response_failure = "timeout"
            else:
                response_error = exc

    cleanup_failed, cleanup_control = _stop_terminal_worker_owned(worker)
    envelope.cleanup_proven = worker.cleaned or _worker_resource_cleanup_is_poisoned()
    if launch_error is not None and not isinstance(launch_error, Exception):
        raise launch_error
    if response_error is not None and not isinstance(response_error, Exception):
        raise response_error
    if cleanup_control is not None:
        raise cleanup_control
    if cleanup_failed:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "terminal helper session cleanup could not be verified",
            case_id=exact.case_id,
            stop_reason="TERMINAL_WORKER_CLEANUP_FAILED",
        ) from None
    if response_error is not None:
        raise response_error
    if start_failed:
        if launch_failure_timed_out:
            return TerminalProcessOutcome(None, "TERMINAL_WORKER_TIMEOUT")
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED")
    if response_failure == "timeout":
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_TIMEOUT")
    if response_failure is not None or process.returncode != 0:
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED")
    try:
        assert response is not None
        outcome = _decode_response(response, expected_claim=exact)
        if any(
            candidate is not None and candidate.case_id != exact.case_id
            for candidate in (outcome.result, outcome.evidence_result)
        ):
            raise ValueError("terminal worker response is not claim-bound")
        return outcome
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED")


def run_terminal_process(
    *,
    mode: TerminalMode,
    settings: Settings,
    claim: ExecutionClaim,
    timeout_seconds: float,
    started_at: datetime | None = None,
    result: CaseResult | None = None,
    worker_error_code: str | None = None,
) -> TerminalProcessOutcome:
    """Run terminal persistence, then prove the entire helper session is empty."""

    exact = validate_execution_claim(claim)
    if type(timeout_seconds) is not float or timeout_seconds <= 0:
        raise ValueError("terminal worker timeout must be a positive float")
    operation_deadline = time.monotonic() + timeout_seconds
    if _worker_resource_cleanup_is_poisoned():
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "terminal helper session cleanup could not be verified",
            case_id=exact.case_id,
            stop_reason="TERMINAL_WORKER_CLEANUP_FAILED",
        ) from None
    if time.monotonic() >= operation_deadline:
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_TIMEOUT")
    try:
        request = _encode_request(
            mode=mode,
            settings=settings,
            claim=exact,
            started_at=started_at,
            result=result,
            worker_error_code=worker_error_code,
        )
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        if time.monotonic() >= operation_deadline:
            return TerminalProcessOutcome(None, "TERMINAL_WORKER_TIMEOUT")
        raise
    if time.monotonic() >= operation_deadline:
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_TIMEOUT")
    process = cast(
        "subprocess.Popen[bytes]",
        subprocess.Popen.__new__(subprocess.Popen),
    )
    process._child_created = False  # type: ignore[attr-defined]
    worker = _WorkerSession(
        process=process,
        process_id=0,
        process_group_id=0,
        session_id=0,
        exit_watcher=None,
        watcher_initialized=False,
        identity_verified=False,
    )
    envelope = _TerminalCleanupEnvelope(process=process, worker=worker)
    try:
        return _run_terminal_process_owned_state(
            envelope=envelope,
            request=request,
            operation_deadline=operation_deadline,
            exact=exact,
        )
    finally:
        active_error = sys.exception()
        retained_cleanup_control: BaseException | None = None
        cleanup_failed = False

        def preserve_cleanup_control(error: BaseException) -> None:
            nonlocal retained_cleanup_control
            if retained_cleanup_control is None or (
                isinstance(retained_cleanup_control, Exception)
                and not isinstance(error, Exception)
            ):
                retained_cleanup_control = error

        for _proof_attempt in range(4):
            proof_visible = False
            if envelope.cleanup_proven:
                if not envelope.ownership_observed or envelope.worker.cleaned:
                    proof_visible = True
                elif envelope.poison_proven:
                    try:
                        proof_visible = _worker_resource_cleanup_is_poisoned()
                    except BaseException as exc:
                        preserve_cleanup_control(exc)
            if proof_visible:
                break
            envelope.cleanup_proven = False
            envelope.poison_proven = False
            finalized_failed, cleanup_control = _finalize_terminal_cleanup_envelope(envelope)
            cleanup_failed = cleanup_failed or finalized_failed
            if cleanup_control is not None:
                preserve_cleanup_control(cleanup_control)
        else:
            os._exit(70)

        active_is_process_control = active_error is not None and not isinstance(
            active_error,
            Exception,
        )
        cleanup_is_process_control = retained_cleanup_control is not None and not isinstance(
            retained_cleanup_control,
            Exception,
        )
        if cleanup_is_process_control and not active_is_process_control:
            assert retained_cleanup_control is not None
            raise retained_cleanup_control
        if cleanup_failed and not active_is_process_control:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "terminal helper session cleanup could not be verified",
                case_id=exact.case_id,
                stop_reason="TERMINAL_WORKER_CLEANUP_FAILED",
            ) from None
        if active_error is None and retained_cleanup_control is not None:
            raise retained_cleanup_control
