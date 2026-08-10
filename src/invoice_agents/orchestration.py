"""Case lifecycle, streamed AutoGen execution, status mapping, persistence, and resume."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any, Literal, cast
from uuid import uuid4

import openai
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    HandoffMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from invoice_agents.agents.team import AgentCaseContext, build_team, create_model_client
from invoice_agents.config import XAI_BASE_URL, XAI_MODEL, Settings
from invoice_agents.db.core import DatabaseKind, verify_database
from invoice_agents.db.store import (
    CaseExecutionSnapshot,
    ExecutionClaim,
    ResultArtifactBinding,
    WorkflowStore,
    parse_canonical_utc,
    validate_execution_claim,
)
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import (
    CaseResult,
    CaseStatus,
    DecisionKind,
    ErrorRecord,
    HumanDecision,
    HumanDecisionKind,
    IdentityCandidate,
    PaymentStatus,
    ToolStatus,
    UsageSummary,
)
from invoice_agents.observability.audit import (
    AuditRecorder,
    bind_audit_recorder,
    normalize_autogen_event_payload,
    redact,
    safe_provider_request_id,
    sanitize_case_result,
    sanitize_text,
)
from invoice_agents.source_store import snapshot_source
from invoice_agents.terminal_process import TerminalProcessOutcome, run_terminal_process
from invoice_agents.tools.comparison import (
    InventoryReader,
    apply_mapping_evidence,
    build_risk_assessment,
    compare_inventory_evidence,
    compute_invoice_totals,
)
from invoice_agents.tools.evidence import extract_invoice_evidence

# AutoGen's MaxMessageTermination phrasing, matched exactly so an upgrade that changes
# it fails the pinned contract test instead of silently misclassifying stops.
MAX_MESSAGES_STOP_PHRASE = "maximum number of messages"
EXECUTION_LEASE_SECONDS = 21_600
EXECUTION_RENEWAL_INTERVAL_SECONDS = 30.0
CLIENT_CLOSE_TIMEOUT_SECONDS = 5.0
DURABILITY_DEADLINE_SECONDS = 5.0
TERMINAL_WORKER_CLEANUP_GRACE_SECONDS = 5.0
RECOVERY_ARTIFACT_FORMAT: Literal[2] = 2
RECOVERY_ARTIFACT_MAX_BYTES = 1_048_576


def is_max_messages_stop(stop_reason: str) -> bool:
    """True only for AutoGen's max-message termination phrasing."""

    return MAX_MESSAGES_STOP_PHRASE in stop_reason.lower()


def _now() -> datetime:
    return datetime.now(UTC)


def validate_case_concurrency(value: object, configured_default: object) -> int:
    """Select only an exact positive integer; ``None`` alone requests the default."""

    selected = configured_default if value is None else value
    if type(selected) is not int or not 1 <= selected <= 8:
        raise ValueError("concurrency must be an exact integer from 1 through 8")
    return selected


async def _run_with_lease_heartbeat[ResultT](
    operation: Awaitable[ResultT],
    *,
    renew: Callable[[ExecutionClaim, int], ExecutionClaim],
    claim: ExecutionClaim,
    replace_claim: Callable[[ExecutionClaim], None],
    lease_seconds: int,
    renewal_interval_seconds: float,
) -> ResultT:
    """Run one lifecycle while renewal failures cancel and replace its result."""

    claim = validate_execution_claim(claim)
    if renewal_interval_seconds <= 0:
        raise ValueError("renewal_interval_seconds must be positive")
    stopped = asyncio.Event()
    current_claim = claim

    async def capture_heartbeat() -> BaseException | None:
        nonlocal current_claim
        try:
            while True:
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=renewal_interval_seconds)
                    return None
                except TimeoutError:
                    renewed = renew(current_claim, lease_seconds)
                    renewed = validate_execution_claim(
                        renewed,
                        expected_case_id=current_claim.case_id,
                    )
                    if (
                        renewed.case_id != current_claim.case_id
                        or renewed.token != current_claim.token
                        or renewed.generation != current_claim.generation
                        or renewed.expires_at <= current_claim.expires_at
                    ):
                        raise InvoiceAgentsError(
                            ErrorCategory.ORCHESTRATION,
                            "execution lease renewal returned contradictory authority",
                            case_id=current_claim.case_id,
                            stop_reason="STALE_EXECUTION_CLAIM",
                        ) from None
                    replace_claim(renewed)
                    current_claim = renewed
        except BaseException as exc:
            return exc

    async def capture_operation() -> tuple[ResultT | None, BaseException | None]:
        try:
            return await operation, None
        except BaseException as exc:
            return None, exc

    operation_task = asyncio.create_task(capture_operation())
    heartbeat_task = asyncio.create_task(capture_heartbeat())
    waiters: set[asyncio.Future[Any]] = set()
    waiters.add(operation_task)
    waiters.add(heartbeat_task)
    try:
        done, _pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        if heartbeat_task in done and (failure := heartbeat_task.result()) is not None:
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise failure
        if heartbeat_task in done and operation_task not in done:
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "execution lease heartbeat stopped before the lifecycle completed",
                case_id=claim.case_id,
                stop_reason="EXECUTION_HEARTBEAT_STOPPED",
            )
        stopped.set()
        await heartbeat_task
        operation_result, operation_failure = operation_task.result()
        if operation_failure is not None:
            raise operation_failure
        return cast(ResultT, operation_result)
    finally:
        stopped.set()
        if not operation_task.done():
            operation_task.cancel()
        if not heartbeat_task.done():
            heartbeat_task.cancel()
        await asyncio.gather(operation_task, heartbeat_task, return_exceptions=True)


def preflight(settings: Settings) -> None:
    """Require credentials and both compatible on-disk databases before a case starts."""

    settings.provider_key()
    verify_database(settings.inventory_db, DatabaseKind.INVENTORY)
    verify_database(settings.workflow_db, DatabaseKind.WORKFLOW, settings=settings)


def _safe_provider_request_id(exc: BaseException) -> str | None:
    return safe_provider_request_id(getattr(exc, "request_id", None))


def _error_record(exc: BaseException, case_id: str | None = None) -> ErrorRecord:
    if isinstance(exc, InvoiceAgentsError):
        return ErrorRecord(
            category=exc.category,
            message=sanitize_text(str(exc.message)),
            case_id=exc.case_id or case_id,
            stop_reason=(sanitize_text(exc.stop_reason) if exc.stop_reason is not None else None),
            provider_request_id=safe_provider_request_id(exc.provider_request_id),
            details=cast(dict[str, Any], redact(exc.details or {})),
        )
    if isinstance(exc, openai.APIResponseValidationError):
        # The provider answered with a payload the SDK could not validate; this is a
        # provider contract failure, never a value to repair or retry into success.
        category = ErrorCategory.PROVIDER
        stop_reason = "PROVIDER_RESPONSE_INVALID"
        message = "provider response failed schema validation"
    elif isinstance(exc, openai.AuthenticationError):
        category = ErrorCategory.AUTHENTICATION
        stop_reason = "PROVIDER_AUTHENTICATION_FAILED"
        message = "provider authentication failed"
    elif isinstance(exc, openai.RateLimitError):
        category = ErrorCategory.RATE_LIMIT
        stop_reason = "PROVIDER_RATE_LIMIT_EXHAUSTED"
        message = "provider rate limit was exhausted"
    elif isinstance(exc, (openai.APITimeoutError, TimeoutError, asyncio.TimeoutError)):
        category = ErrorCategory.TIMEOUT
        stop_reason = "PROVIDER_TIMEOUT"
        message = "provider request exceeded the configured timeout"
    elif isinstance(exc, (openai.APIConnectionError, openai.APIStatusError)):
        category = ErrorCategory.PROVIDER
        stop_reason = "PROVIDER_REQUEST_FAILED"
        message = "provider request failed"
    elif isinstance(exc, sqlite3.Error):
        category = ErrorCategory.DATABASE
        stop_reason = "DATABASE_ERROR"
        message = "workflow database operation failed"
    elif isinstance(exc, json.JSONDecodeError):
        category = ErrorCategory.SCHEMA
        stop_reason = "RESPONSE_DECODE_FAILED"
        message = "response JSON decoding failed"
    elif isinstance(exc, ValidationError):
        category = ErrorCategory.SCHEMA
        stop_reason = "SCHEMA_VALIDATION_FAILED"
        message = "response schema validation failed"
    else:
        category = ErrorCategory.ORCHESTRATION
        stop_reason = "UNEXPECTED_RUNTIME_ERROR"
        message = "unexpected runtime failure"
    return ErrorRecord(
        category=category,
        message=message,
        case_id=case_id,
        stop_reason=stop_reason,
        provider_request_id=_safe_provider_request_id(exc),
        details={"exception_type": type(exc).__name__},
    )


def _failed_result(
    case_id: str,
    source_id: str | None,
    started_at: datetime,
    exc: BaseException,
) -> CaseResult:
    error = _error_record(exc, case_id)
    return CaseResult(
        case_id=case_id,
        source_id=source_id,
        status=CaseStatus.FAILED,
        stop_reason=error.stop_reason or "FAILED",
        errors=[error],
        started_at=started_at,
        finished_at=_now(),
    )


def _cancelled_result(
    case_id: str,
    source_id: str | None,
    started_at: datetime,
    previous: CaseResult | None = None,
) -> CaseResult:
    """Build the one explicit cancellation result while retaining durable evidence."""

    cancelled = ErrorRecord(
        category=ErrorCategory.CANCELLED,
        message="case execution was cancelled",
        case_id=case_id,
        stop_reason="CANCELLED",
    )
    if previous is None:
        return CaseResult(
            case_id=case_id,
            source_id=source_id,
            status=CaseStatus.INCOMPLETE,
            stop_reason="CANCELLED",
            errors=[cancelled],
            started_at=started_at,
            finished_at=_now(),
        )
    errors = list(previous.errors)
    if not any(error.stop_reason == "CANCELLED" for error in errors):
        errors.append(cancelled)
    return previous.model_copy(
        update={
            "status": CaseStatus.INCOMPLETE,
            "stop_reason": "CANCELLED",
            "errors": errors,
            "finished_at": _now(),
        },
        deep=True,
    )


@dataclass(slots=True)
class _ClaimedExecution:
    store: WorkflowStore
    claim: ExecutionClaim
    case_id: str
    started_at: datetime
    source_id: str | None = None
    audit: AuditRecorder | None = None
    client: Any | None = None
    context: AgentCaseContext | None = None
    usage: UsageSummary | None = None
    clock: float = 0.0


def _secondary_error(
    exc: BaseException,
    *,
    case_id: str,
    category: ErrorCategory,
    stop_reason: str,
    message: str,
) -> ErrorRecord:
    cause = _error_record(exc, case_id)
    return ErrorRecord(
        category=category,
        message=message,
        case_id=case_id,
        stop_reason=stop_reason,
        provider_request_id=cause.provider_request_id,
        details={
            "cause_category": cause.category,
            "exception_type": type(exc).__name__,
        },
    )


def _append_error(result: CaseResult, error: ErrorRecord) -> None:
    result.errors = [*result.errors, error]


_ARTIFACT_RESULT_ERROR_STOPS = frozenset(
    {
        "RESULT_ARTIFACT_WRITE_CANCELLED",
        "RESULT_ARTIFACT_WRITE_FAILED",
        "RESULT_ARTIFACT_DURABILITY_UNRESOLVED",
    }
)


def _append_error_before_artifact(result: CaseResult, error: ErrorRecord) -> None:
    """Keep terminal boundary error ordering stable after publication is deferred."""

    insertion = len(result.errors)
    while (
        insertion > 0 and result.errors[insertion - 1].stop_reason in _ARTIFACT_RESULT_ERROR_STOPS
    ):
        insertion -= 1
    result.errors = [*result.errors[:insertion], error, *result.errors[insertion:]]


def _artifact_durability_unresolved_result(result: CaseResult) -> CaseResult:
    stop_reason = "RESULT_ARTIFACT_DURABILITY_UNRESOLVED"
    unresolved = result.model_copy(
        update={
            "status": CaseStatus.INCOMPLETE,
            "stop_reason": stop_reason,
            "finished_at": _now(),
        },
        deep=True,
    )
    if not unresolved.errors or unresolved.errors[-1].stop_reason != stop_reason:
        _append_error(
            unresolved,
            ErrorRecord(
                category=ErrorCategory.ORCHESTRATION,
                message="result artifact rollback durability could not be proven",
                case_id=result.case_id,
                stop_reason=stop_reason,
            ),
        )
    return unresolved


def _cancellation_may_define_outcome(control_exception: BaseException | None) -> bool:
    return control_exception is None or isinstance(control_exception, asyncio.CancelledError)


def _preserve_prior_result_evidence(result: CaseResult, previous: CaseResult | None) -> CaseResult:
    if previous is None:
        return result
    return result.model_copy(
        update={
            "final_decision": result.final_decision or previous.final_decision,
            "review_request": result.review_request or previous.review_request,
            "payment": result.payment or previous.payment,
            "errors": [*previous.errors, *result.errors],
        },
        deep=True,
    )


class _RecoveryEnvelope(BaseModel):
    """Strict claim-bound recovery evidence; no ambient authority is serialized."""

    model_config = ConfigDict(extra="forbid", strict=True)

    recovery_format: Literal[2]
    case_id: str = Field(min_length=1)
    execution_token: str = Field(min_length=1)
    execution_generation: int = Field(ge=1)
    lease_expires_at: str = Field(min_length=1)
    case_result: CaseResult
    terminal_persistence_error: ErrorRecord


@dataclass(frozen=True, slots=True)
class _RecoveryAuthority:
    store: WorkflowStore
    claim: ExecutionClaim


_RECOVERY_AUTHORITY: ContextVar[_RecoveryAuthority] = ContextVar("task9_recovery_authority")


class _ExactClaimEvidenceState(StrEnum):
    """The only two trustworthy states for one already-inspected exact claim."""

    DURABLE_DATABASE_RESULT = "DURABLE_DATABASE_RESULT"
    RECOVERABLE_RUNNING = "RECOVERABLE_RUNNING"


@dataclass(frozen=True, slots=True)
class _ExactClaimEvidence:
    state: _ExactClaimEvidenceState
    result: CaseResult | None


_RECOVERY_PERSISTENCE_STOPS = frozenset(
    {
        "TERMINAL_PERSISTENCE_FAILED",
        "TERMINAL_DURABILITY_TIMEOUT",
        "TERMINAL_RESULT_UPDATE_CANCELLED",
    }
)


def _canonical_recovery_persistence_error(
    case_id: str,
    stop_reason: str,
) -> ErrorRecord:
    """Reconstruct an exact parent-owned recovery error from one stable stop code."""

    if stop_reason == "TERMINAL_PERSISTENCE_FAILED":
        category = ErrorCategory.DATABASE
        message = "terminal database write failed"
    elif stop_reason == "TERMINAL_DURABILITY_TIMEOUT":
        category = ErrorCategory.TIMEOUT
        message = "terminal durability work exceeded its monotonic deadline"
    elif stop_reason == "TERMINAL_RESULT_UPDATE_CANCELLED":
        category = ErrorCategory.CANCELLED
        message = "terminal database result update was cancelled"
    else:
        raise ValueError("unrecognized terminal recovery persistence stop")
    return ErrorRecord(
        category=category,
        message=message,
        case_id=case_id,
        stop_reason=stop_reason,
        provider_request_id=None,
        details={},
    )


def _recovery_only_result(result: CaseResult, persistence_error: ErrorRecord) -> CaseResult:
    """Make failed persistence explicit without laundering a successful terminal result."""

    canonical_error = _canonical_recovery_persistence_error(
        result.case_id,
        persistence_error.stop_reason or "",
    )
    retained_errors = [
        error for error in result.errors if error.stop_reason not in _RECOVERY_PERSISTENCE_STOPS
    ]
    return result.model_copy(
        update={
            "status": CaseStatus.INCOMPLETE,
            "stop_reason": canonical_error.stop_reason,
            "errors": [*retained_errors, canonical_error],
        },
        deep=True,
    )


def _canonical_recovery_bytes(envelope: _RecoveryEnvelope) -> bytes:
    return json.dumps(
        envelope.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _recovery_semantics_are_exact(envelope: _RecoveryEnvelope) -> bool:
    result = envelope.case_result
    error = envelope.terminal_persistence_error
    try:
        canonical_error = _canonical_recovery_persistence_error(
            envelope.case_id,
            error.stop_reason or "",
        )
    except ValueError:
        return False
    return (
        error == canonical_error
        and result.status is CaseStatus.INCOMPLETE
        and result.stop_reason == canonical_error.stop_reason
        and result.finished_at >= result.started_at
        and bool(result.errors)
        and result.errors[-1] == canonical_error
        and sum(item.stop_reason in _RECOVERY_PERSISTENCE_STOPS for item in result.errors) == 1
    )


def _inspect_exact_claim_evidence(
    store: WorkflowStore,
    claim: ExecutionClaim,
) -> _ExactClaimEvidence:
    """Classify only an exact case/token/generation/lease tuple; contradictions fail closed."""

    claim = validate_execution_claim(claim)
    snapshot = store.load_case_execution_snapshot(claim.case_id)
    exact_identity = (
        snapshot is not None
        and type(snapshot.execution_token) is str
        and snapshot.execution_token == claim.token
        and type(snapshot.execution_generation) is int
        and snapshot.execution_generation == claim.generation
    )
    if not exact_identity:
        raise _unresolved_durability_error(claim.case_id) from None
    assert snapshot is not None
    if (
        snapshot.execution_state == "FINISHED"
        and snapshot.lease_expires_at is None
        and type(snapshot.result) is CaseResult
    ):
        return _ExactClaimEvidence(
            _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT,
            store.merge_relational_case_evidence(snapshot.result),
        )
    if (
        snapshot.execution_state == "RUNNING"
        and type(snapshot.lease_expires_at) is datetime
        and snapshot.lease_expires_at == claim.expires_at
    ):
        return _ExactClaimEvidence(_ExactClaimEvidenceState.RECOVERABLE_RUNNING, None)
    raise _unresolved_durability_error(claim.case_id) from None


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate recovery key")
        payload[key] = value
    return payload


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite recovery number")


def _recovery_result_bound_to_authority(
    result: CaseResult,
    authority: _RecoveryAuthority,
) -> CaseResult:
    """Validate current DB authority and return a source-bound recovery aggregate."""

    claim = validate_execution_claim(authority.claim, expected_case_id=result.case_id)
    evidence = _inspect_exact_claim_evidence(authority.store, claim)
    if evidence.state is not _ExactClaimEvidenceState.RECOVERABLE_RUNNING:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "recovery artifacts require the exact still-running claim",
            case_id=result.case_id,
            stop_reason="TERMINAL_DURABILITY_UNRESOLVED",
        ) from None
    source_id = authority.store.load_recovery_case_source_id(claim)
    if result.source_id not in {None, source_id}:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "recovery result identity conflicts with authoritative source evidence",
            case_id=result.case_id,
            stop_reason="PERSISTED_RESULT_INVALID",
        ) from None
    source_bound = result.model_copy(update={"source_id": source_id}, deep=True)
    return authority.store.merge_relational_case_evidence(source_bound)


def _write_recovery_artifact(result: CaseResult, terminal_persistence_error: ErrorRecord) -> Path:
    authority = _RECOVERY_AUTHORITY.get()
    claim = validate_execution_claim(authority.claim, expected_case_id=result.case_id)
    if terminal_persistence_error.case_id != result.case_id:
        raise ValueError("recovery persistence error is not case-bound")
    terminal_persistence_error = _canonical_recovery_persistence_error(
        result.case_id,
        terminal_persistence_error.stop_reason or "",
    )
    result = _recovery_only_result(result, terminal_persistence_error)
    result = _recovery_result_bound_to_authority(result, authority)
    result = sanitize_case_result(result)
    output_dir = Path("artifacts/results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{result.case_id}.recovery.json"
    envelope = _RecoveryEnvelope(
        recovery_format=RECOVERY_ARTIFACT_FORMAT,
        case_id=result.case_id,
        execution_token=claim.token,
        execution_generation=claim.generation,
        lease_expires_at=claim.expires_at.isoformat(),
        case_result=result,
        terminal_persistence_error=terminal_persistence_error,
    )
    payload = _canonical_recovery_bytes(envelope)
    if len(payload) > RECOVERY_ARTIFACT_MAX_BYTES:
        raise ValueError("recovery artifact exceeds its bounded envelope")
    _atomic_publish(target, payload)
    return target


def _recovery_artifact_or_raise(
    result: CaseResult,
    terminal_persistence_error: ErrorRecord,
    *,
    store: WorkflowStore,
    claim: ExecutionClaim,
) -> None:
    """Publish recovery evidence once or raise one stable, chainless failure."""

    artifact_exc: BaseException | None = None
    token = _RECOVERY_AUTHORITY.set(_RecoveryAuthority(store, claim))
    try:
        _write_recovery_artifact(result, terminal_persistence_error)
    except BaseException as exc:
        artifact_exc = exc
    finally:
        _RECOVERY_AUTHORITY.reset(token)
    if artifact_exc is None:
        return
    artifact_error = _secondary_error(
        artifact_exc,
        case_id=result.case_id,
        category=ErrorCategory.ORCHESTRATION,
        stop_reason="TERMINAL_RECOVERY_ARTIFACT_FAILED",
        message="atomic terminal recovery artifact publication failed",
    )
    raise InvoiceAgentsError(
        ErrorCategory.ORCHESTRATION,
        artifact_error.message,
        case_id=result.case_id,
        stop_reason=artifact_error.stop_reason,
        details={
            "terminal_persistence_stop_reason": terminal_persistence_error.stop_reason,
            "artifact_exception_type": type(artifact_exc).__name__,
            "artifact_publication_stop_reason": (
                artifact_exc.stop_reason if isinstance(artifact_exc, InvoiceAgentsError) else None
            ),
        },
    ) from None


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


@dataclass(frozen=True, slots=True)
class _CleanupOutcome:
    error: ErrorRecord | None = None
    control_exception: BaseException | None = None


async def _close_claimed_client(execution: _ClaimedExecution) -> _CleanupOutcome:
    client = execution.client
    if client is None:
        return _CleanupOutcome()

    async def capture_close() -> BaseException | None:
        try:
            await client.close()
        except BaseException as exc:
            return exc
        return None

    deadline = monotonic() + CLIENT_CLOSE_TIMEOUT_SECONDS
    close_task = asyncio.create_task(capture_close())
    boundary_control: BaseException | None = None
    try:
        close_failure = await asyncio.wait_for(
            asyncio.shield(close_task),
            timeout=max(0.0, deadline - monotonic()),
        )
    except asyncio.CancelledError as exc:
        boundary_control = exc
        if close_task.done():
            close_failure = close_task.result()
        else:
            try:
                close_failure = await _await_task_despite_cancellation(
                    close_task,
                    deadline=deadline,
                    case_id=execution.case_id,
                )
            except InvoiceAgentsError:
                close_task.cancel()
                close_task.add_done_callback(_consume_task_result)
                return _CleanupOutcome(
                    error=ErrorRecord(
                        category=ErrorCategory.TIMEOUT,
                        message="client cleanup exceeded its bounded timeout",
                        case_id=execution.case_id,
                        stop_reason="CLIENT_CLOSE_TIMEOUT",
                    ),
                    control_exception=boundary_control,
                )
            except BaseException as drain_error:
                close_failure = drain_error
        if close_failure is None:
            return _CleanupOutcome(
                error=ErrorRecord(
                    category=ErrorCategory.CANCELLED,
                    message="client cleanup was cancelled",
                    case_id=execution.case_id,
                    stop_reason="CLIENT_CLOSE_CANCELLED",
                ),
                control_exception=boundary_control,
            )
    except TimeoutError:
        close_task.cancel()
        close_task.add_done_callback(_consume_task_result)
        return _CleanupOutcome(
            error=ErrorRecord(
                category=ErrorCategory.TIMEOUT,
                message="client cleanup exceeded its bounded timeout",
                case_id=execution.case_id,
                stop_reason="CLIENT_CLOSE_TIMEOUT",
            ),
        )
    if isinstance(close_failure, asyncio.CancelledError):
        return _CleanupOutcome(
            error=ErrorRecord(
                category=ErrorCategory.CANCELLED,
                message="client cleanup was cancelled",
                case_id=execution.case_id,
                stop_reason="CLIENT_CLOSE_CANCELLED",
            ),
            control_exception=boundary_control or close_failure,
        )
    if close_failure is not None:
        return _CleanupOutcome(
            error=_secondary_error(
                close_failure,
                case_id=execution.case_id,
                category=ErrorCategory.ORCHESTRATION,
                stop_reason="CLIENT_CLOSE_FAILED",
                message="client cleanup failed",
            ),
            control_exception=(
                boundary_control
                if boundary_control is not None
                else close_failure if not isinstance(close_failure, Exception) else None
            ),
        )
    return _CleanupOutcome()


@dataclass(slots=True)
class _PersistenceOutcome:
    result: CaseResult
    persisted: bool
    persistence_error: ErrorRecord | None
    control_exception: BaseException | None


@dataclass(frozen=True, slots=True)
class _TerminalWriteBoundaryOutcome:
    """One helper result plus the exact claim evidence read under its deadline."""

    process_outcome: TerminalProcessOutcome
    evidence: _ExactClaimEvidence
    control_exception: BaseException | None


def _select_terminal_control_exception(
    prior: BaseException | None,
    later: BaseException | None,
) -> BaseException | None:
    """Retain explicit durability loss, then process control, then first failure."""

    for candidate in (prior, later):
        if (
            isinstance(candidate, InvoiceAgentsError)
            and candidate.stop_reason in _DURABILITY_PRECEDENCE_STOPS
        ):
            return candidate

    if prior is not None and not isinstance(prior, (Exception, asyncio.CancelledError)):
        return prior
    if later is not None and not isinstance(later, (Exception, asyncio.CancelledError)):
        return later
    return prior if prior is not None else later


def _terminal_outcome_evidence(
    outcome: TerminalProcessOutcome,
) -> _ExactClaimEvidence | None:
    """Translate only one strict helper evidence state into parent-owned evidence."""

    state = getattr(outcome, "evidence_state", None)
    result = getattr(outcome, "evidence_result", None)
    if state == _ExactClaimEvidenceState.RECOVERABLE_RUNNING.value and result is None:
        return _ExactClaimEvidence(_ExactClaimEvidenceState.RECOVERABLE_RUNNING, None)
    if (
        state == _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT.value
        and type(result) is CaseResult
    ):
        return _ExactClaimEvidence(_ExactClaimEvidenceState.DURABLE_DATABASE_RESULT, result)
    return None


def _terminal_durability_timeout(case_id: str) -> InvoiceAgentsError:
    return InvoiceAgentsError(
        ErrorCategory.TIMEOUT,
        "terminal durability work exceeded its monotonic deadline",
        case_id=case_id,
        stop_reason="TERMINAL_DURABILITY_TIMEOUT",
    )


async def _run_terminal_mode_owned(
    *,
    mode: Literal[
        "cancel_unstarted",
        "finish",
        "inspect_claim",
        "publish_cancel_recovery",
        "update",
    ],
    settings: Settings,
    claim: ExecutionClaim,
    deadline: float,
    started_at: datetime | None = None,
    result: CaseResult | None = None,
    worker_error_code: str | None = None,
    prior_cancellation: asyncio.CancelledError | None = None,
) -> tuple[TerminalProcessOutcome, BaseException | None]:
    """Run one self-bounded helper and drain its thread through caller cancellation."""

    remaining = deadline - monotonic()
    if remaining <= 0:
        raise _terminal_durability_timeout(claim.case_id) from None
    timeout_seconds = float(min(DURABILITY_DEADLINE_SECONDS, remaining))

    def invoke() -> tuple[TerminalProcessOutcome, BaseException | None]:
        try:
            return (
                run_terminal_process(
                    mode=mode,
                    settings=settings,
                    claim=claim,
                    timeout_seconds=timeout_seconds,
                    started_at=started_at,
                    result=result,
                    worker_error_code=worker_error_code,
                ),
                None,
            )
        except InvoiceAgentsError as exc:
            if exc.stop_reason in _DURABILITY_PRECEDENCE_STOPS:
                raise
            return TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED"), exc
        except BaseException as exc:
            return TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED"), exc

    worker = asyncio.create_task(
        asyncio.to_thread(invoke),
        name=f"invoice-terminal-{mode}-{claim.case_id}",
    )
    cancellation = prior_cancellation
    while True:
        try:
            outcome, process_control = await asyncio.shield(worker)
            return outcome, _select_terminal_control_exception(cancellation, process_control)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            if worker.done():
                outcome, process_control = worker.result()
                return outcome, _select_terminal_control_exception(
                    cancellation,
                    process_control,
                )


def _persist_terminal_result(
    execution: _ClaimedExecution,
    result: CaseResult,
    control_exception: BaseException | None,
) -> _PersistenceOutcome:
    """Persist once; a cancellation at this boundary gets one CANCELLED retry."""

    result = sanitize_case_result(result)
    try:
        execution.store.finish_case(result, execution.claim)
        return _PersistenceOutcome(result, True, None, control_exception)
    except asyncio.CancelledError as exc:
        terminal_result = (
            _cancelled_result(
                execution.case_id,
                execution.source_id,
                execution.started_at,
                result,
            )
            if _cancellation_may_define_outcome(control_exception)
            else result
        )
        try:
            execution.store.finish_case(terminal_result, execution.claim)
            return _PersistenceOutcome(terminal_result, True, None, control_exception or exc)
        except InvoiceAgentsError as stale_exc:
            if stale_exc.stop_reason != "STALE_EXECUTION_CLAIM":
                terminal_exc: BaseException = stale_exc
            else:
                try:
                    execution.store.update_finished_case_result(terminal_result, execution.claim)
                    return _PersistenceOutcome(
                        terminal_result, True, None, control_exception or exc
                    )
                except BaseException as update_exc:
                    terminal_exc = update_exc
        except BaseException as retry_exc:
            terminal_exc = retry_exc
        persistence_error = _secondary_error(
            terminal_exc,
            case_id=execution.case_id,
            category=ErrorCategory.DATABASE,
            stop_reason="TERMINAL_PERSISTENCE_FAILED",
            message="terminal database write failed",
        )
        _append_error(terminal_result, persistence_error)
        return _PersistenceOutcome(
            terminal_result,
            False,
            persistence_error,
            control_exception or exc,
        )
    except BaseException as exc:
        persistence_error = _secondary_error(
            exc,
            case_id=execution.case_id,
            category=ErrorCategory.DATABASE,
            stop_reason="TERMINAL_PERSISTENCE_FAILED",
            message="terminal database write failed",
        )
        _append_error(result, persistence_error)
        return _PersistenceOutcome(
            result,
            False,
            persistence_error,
            (control_exception if isinstance(exc, Exception) else control_exception or exc),
        )


async def _terminal_process_write(
    execution: _ClaimedExecution,
    result: CaseResult,
    *,
    mode: Literal["finish", "update"],
    durability_deadline: float | None = None,
) -> _TerminalWriteBoundaryOutcome:
    """Run one terminal write and return evidence from only reaped helper sessions."""

    result = sanitize_case_result(result)
    settings = execution.store._snapshot_settings()
    if durability_deadline is None:
        durability_deadline = (
            monotonic() + DURABILITY_DEADLINE_SECONDS + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
        )
    outcome, boundary_control = await _run_terminal_mode_owned(
        mode=mode,
        settings=settings,
        claim=execution.claim,
        deadline=durability_deadline,
        result=result,
    )
    evidence = _terminal_outcome_evidence(outcome)
    if evidence is None:
        if monotonic() >= durability_deadline:
            raise _terminal_durability_timeout(execution.case_id) from None
        inspection, inspection_control = await _run_terminal_mode_owned(
            mode="inspect_claim",
            settings=settings,
            claim=execution.claim,
            deadline=durability_deadline,
            prior_cancellation=(
                boundary_control if isinstance(boundary_control, asyncio.CancelledError) else None
            ),
        )
        boundary_control = _select_terminal_control_exception(
            boundary_control,
            inspection_control,
        )
        evidence = _terminal_outcome_evidence(inspection)
        if evidence is None:
            if outcome.error_code == "TERMINAL_WORKER_TIMEOUT" or (
                inspection.error_code == "TERMINAL_WORKER_TIMEOUT"
            ):
                raise _terminal_durability_timeout(execution.case_id) from None
            raise _unresolved_durability_error(execution.case_id) from None
    return _TerminalWriteBoundaryOutcome(
        process_outcome=outcome,
        evidence=evidence,
        control_exception=boundary_control,
    )


def _terminal_process_error(
    outcome: object,
    *,
    case_id: str,
    update: bool,
) -> ErrorRecord:
    error_code = getattr(outcome, "error_code", None)
    if error_code == "TERMINAL_WORKER_TIMEOUT":
        return ErrorRecord(
            category=ErrorCategory.TIMEOUT,
            message="terminal database write exceeded its bounded helper deadline",
            case_id=case_id,
            stop_reason="TERMINAL_DURABILITY_TIMEOUT",
        )
    return ErrorRecord(
        category=ErrorCategory.DATABASE,
        message=(
            "terminal database result update failed" if update else "terminal database write failed"
        ),
        case_id=case_id,
        stop_reason="TERMINAL_PERSISTENCE_FAILED",
        details={"worker_error_code": error_code or "TERMINAL_WORKER_FAILED"},
    )


async def _persist_terminal_result_safely(
    execution: _ClaimedExecution,
    result: CaseResult,
    control_exception: BaseException | None,
) -> _PersistenceOutcome:
    """Persist in a terminable helper; cancellation may require one fenced update."""

    durability_deadline = (
        monotonic() + DURABILITY_DEADLINE_SECONDS + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
    )
    write = await _terminal_process_write(
        execution,
        result,
        mode="finish",
        durability_deadline=durability_deadline,
    )
    outcome = write.process_outcome
    evidence = write.evidence
    boundary_control = write.control_exception
    stored = (
        evidence.result
        if evidence.state is _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT
        else None
    )
    control_exception = _select_terminal_control_exception(
        control_exception,
        boundary_control,
    )
    if stored is not None:
        if boundary_control is not None and _cancellation_may_define_outcome(control_exception):
            cancelled = _cancelled_result(
                execution.case_id,
                stored.source_id,
                execution.started_at,
                stored,
            )
            return await _refresh_terminal_evidence_safely(
                execution,
                cancelled,
                persisted=True,
                persistence_error=None,
                control_exception=control_exception,
                durability_deadline=durability_deadline,
            )
        return _PersistenceOutcome(stored, True, None, control_exception)
    if isinstance(boundary_control, asyncio.CancelledError) and _cancellation_may_define_outcome(
        control_exception
    ):
        cancelled = _cancelled_result(
            execution.case_id,
            result.source_id,
            execution.started_at,
            result,
        )
        retry_write = await _terminal_process_write(
            execution,
            cancelled,
            mode="finish",
            durability_deadline=durability_deadline,
        )
        retry_outcome = retry_write.process_outcome
        retry_evidence = retry_write.evidence
        retry_control = retry_write.control_exception
        retry_stored = (
            retry_evidence.result
            if retry_evidence.state is _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT
            else None
        )
        control_exception = _select_terminal_control_exception(
            control_exception,
            retry_control,
        )
        if retry_stored is not None:
            return _PersistenceOutcome(
                retry_stored,
                True,
                None,
                control_exception,
            )
        outcome = retry_outcome
        evidence = retry_evidence
        result = cancelled
    if evidence.state is not _ExactClaimEvidenceState.RECOVERABLE_RUNNING:
        raise _unresolved_durability_error(execution.case_id) from None
    persistence_error = _terminal_process_error(
        outcome,
        case_id=execution.case_id,
        update=False,
    )
    _append_error(result, persistence_error)
    return _PersistenceOutcome(
        result,
        False,
        persistence_error,
        control_exception,
    )


async def _refresh_terminal_evidence_safely(
    execution: _ClaimedExecution,
    result: CaseResult,
    *,
    persisted: bool,
    persistence_error: ErrorRecord | None,
    control_exception: BaseException | None,
    durability_deadline: float | None = None,
) -> _PersistenceOutcome:
    if not persisted:
        assert persistence_error is not None
        return _PersistenceOutcome(result, False, persistence_error, control_exception)
    prior_control = control_exception
    if durability_deadline is None:
        durability_deadline = (
            monotonic() + DURABILITY_DEADLINE_SECONDS + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
        )
    write = await _terminal_process_write(
        execution,
        result,
        mode="update",
        durability_deadline=durability_deadline,
    )
    outcome = write.process_outcome
    boundary_control = write.control_exception
    stored = (
        write.evidence.result
        if write.evidence.state is _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT
        else None
    )
    control_exception = _select_terminal_control_exception(
        control_exception,
        boundary_control,
    )
    if stored == result:
        return _PersistenceOutcome(result, True, persistence_error, control_exception)
    if stored is not None:
        # An update failure cannot turn an already exact FINISHED row back into
        # recoverable RUNNING authority.  Keep the sole durable database result;
        # a recovery frame here would be a conflicting second result channel.
        return _PersistenceOutcome(stored, True, persistence_error, control_exception)
    if isinstance(boundary_control, asyncio.CancelledError):
        if _cancellation_may_define_outcome(prior_control):
            result = _cancelled_result(
                execution.case_id,
                result.source_id,
                execution.started_at,
                result,
            )
        update_error = ErrorRecord(
            category=ErrorCategory.CANCELLED,
            message="terminal database result update was cancelled",
            case_id=execution.case_id,
            stop_reason="TERMINAL_RESULT_UPDATE_CANCELLED",
        )
    else:
        update_error = _terminal_process_error(
            outcome,
            case_id=execution.case_id,
            update=True,
        )
    _append_error(result, update_error)
    return _PersistenceOutcome(result, False, update_error, control_exception)


def _refresh_terminal_evidence(
    execution: _ClaimedExecution,
    result: CaseResult,
    *,
    persisted: bool,
    persistence_error: ErrorRecord | None,
    control_exception: BaseException | None,
) -> _PersistenceOutcome:
    if not persisted:
        assert persistence_error is not None
        return _PersistenceOutcome(
            result,
            False,
            persistence_error,
            control_exception,
        )
    try:
        execution.store.update_finished_case_result(result, execution.claim)
        return _PersistenceOutcome(result, True, persistence_error, control_exception)
    except asyncio.CancelledError as exc:
        cancellation_may_define_outcome = _cancellation_may_define_outcome(control_exception)
        terminal_result = (
            _cancelled_result(
                execution.case_id,
                execution.source_id,
                execution.started_at,
                result,
            )
            if cancellation_may_define_outcome
            else result
        )
        update_error = ErrorRecord(
            category=ErrorCategory.CANCELLED,
            message="terminal database result update was cancelled",
            case_id=execution.case_id,
            stop_reason="TERMINAL_RESULT_UPDATE_CANCELLED",
        )
        _append_error(terminal_result, update_error)
        return _PersistenceOutcome(
            terminal_result,
            False,
            update_error,
            control_exception or exc,
        )
    except BaseException as exc:
        update_error = _secondary_error(
            exc,
            case_id=execution.case_id,
            category=ErrorCategory.DATABASE,
            stop_reason="TERMINAL_PERSISTENCE_FAILED",
            message="terminal database result update failed",
        )
        _append_error(result, update_error)
        return _PersistenceOutcome(
            result,
            False,
            update_error,
            (control_exception if isinstance(exc, Exception) else control_exception or exc),
        )


async def _execute_claimed_case(
    case_id: str,
    started_at: datetime,
    store: WorkflowStore,
    claim: ExecutionClaim,
    lifecycle: Callable[[_ClaimedExecution], Awaitable[CaseResult]],
    *,
    finished_event_type: str,
    terminal_writes_in_process: bool = False,
) -> CaseResult:
    """Run setup through terminal persistence under one claimed outer boundary."""

    async def persist_terminal(
        execution: _ClaimedExecution,
        result: CaseResult,
        control_exception: BaseException | None,
    ) -> _PersistenceOutcome:
        if terminal_writes_in_process:
            return _persist_terminal_result(execution, result, control_exception)
        return await _persist_terminal_result_safely(execution, result, control_exception)

    async def refresh_terminal(
        execution: _ClaimedExecution,
        result: CaseResult,
        *,
        persisted: bool,
        persistence_error: ErrorRecord | None,
        control_exception: BaseException | None,
    ) -> _PersistenceOutcome:
        if terminal_writes_in_process:
            return _refresh_terminal_evidence(
                execution,
                result,
                persisted=persisted,
                persistence_error=persistence_error,
                control_exception=control_exception,
            )
        return await _refresh_terminal_evidence_safely(
            execution,
            result,
            persisted=persisted,
            persistence_error=persistence_error,
            control_exception=control_exception,
        )

    execution = _ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
        usage=UsageSummary(),
        clock=monotonic(),
    )
    control_exception: BaseException | None = None
    cancelled = False
    previous_result: CaseResult | None = None
    try:
        execution.source_id = store.load_authoritative_case_source_id(claim)
        previous_result = store.load_result(case_id)
        result = await lifecycle(execution)
    except asyncio.CancelledError as exc:
        cancelled = True
        control_exception = exc
        result = _cancelled_result(
            case_id,
            execution.source_id,
            execution.started_at,
            previous_result,
        )
        if execution.usage is not None:
            result.usage = execution.usage
    except Exception as exc:
        result = _failed_result(case_id, execution.source_id, execution.started_at, exc)
        result = _preserve_prior_result_evidence(result, previous_result)
        if execution.usage is not None:
            result.usage = execution.usage
    except BaseException as exc:
        control_exception = exc
        result = _failed_result(case_id, execution.source_id, execution.started_at, exc)
        result = _preserve_prior_result_evidence(result, previous_result)
        if execution.usage is not None:
            result.usage = execution.usage

    try:
        result = store.merge_relational_case_evidence(result)
    except asyncio.CancelledError as exc:
        cancellation_may_define_outcome = _cancellation_may_define_outcome(control_exception)
        control_exception = control_exception or exc
        if cancellation_may_define_outcome:
            result = _cancelled_result(
                case_id,
                execution.source_id,
                execution.started_at,
                result,
            )
            cancelled = True
        _append_error(
            result,
            ErrorRecord(
                category=ErrorCategory.CANCELLED,
                message="durable terminal evidence reconciliation was cancelled",
                case_id=case_id,
                stop_reason="TERMINAL_EVIDENCE_RECONCILIATION_CANCELLED",
            ),
        )
    except BaseException as exc:
        _append_error(
            result,
            _secondary_error(
                exc,
                case_id=case_id,
                category=ErrorCategory.DATABASE,
                stop_reason="TERMINAL_EVIDENCE_RECONCILIATION_FAILED",
                message="durable terminal evidence reconciliation failed",
            ),
        )
        if result.status in {CaseStatus.SUCCEEDED, CaseStatus.NEEDS_HUMAN}:
            result = result.model_copy(
                update={
                    "status": CaseStatus.INCOMPLETE,
                    "stop_reason": "TERMINAL_EVIDENCE_RECONCILIATION_FAILED",
                    "finished_at": _now(),
                },
                deep=True,
            )
        if not isinstance(exc, Exception):
            control_exception = control_exception or exc

    if execution.usage is not None:
        elapsed = int((monotonic() - execution.clock) * 1000)
        execution.usage.latency_ms = max(execution.usage.latency_ms, elapsed)
        result.usage = execution.usage

    persisted = False
    terminal_write_attempted = False
    persistence_error: ErrorRecord | None = None
    # Cancellation becomes durable before cleanup, then is re-raised after bounded cleanup.
    if cancelled:
        persistence = await persist_terminal(
            execution,
            result,
            control_exception,
        )
        terminal_write_attempted = True
        result = persistence.result
        persisted = persistence.persisted
        persistence_error = persistence.persistence_error
        control_exception = persistence.control_exception

    cleanup = await _close_claimed_client(execution)
    if cleanup.control_exception is not None:
        cancellation_may_define_outcome = _cancellation_may_define_outcome(control_exception)
        control_exception = control_exception or cleanup.control_exception
        if (
            isinstance(cleanup.control_exception, asyncio.CancelledError)
            and cancellation_may_define_outcome
        ):
            result = _cancelled_result(
                case_id,
                execution.source_id,
                execution.started_at,
                result,
            )
            cancelled = True
    if cleanup.error is not None:
        _append_error(result, cleanup.error)

    try:
        result.usage.retries = store.count_events(case_id, "provider.retry")
    except asyncio.CancelledError as exc:
        cancellation_may_define_outcome = _cancellation_may_define_outcome(control_exception)
        control_exception = control_exception or exc
        if cancellation_may_define_outcome:
            result = _cancelled_result(case_id, execution.source_id, execution.started_at, result)
            cancelled = True
        _append_error(
            result,
            ErrorRecord(
                category=ErrorCategory.CANCELLED,
                message="retry accounting was cancelled",
                case_id=case_id,
                stop_reason="RETRY_COUNT_CANCELLED",
            ),
        )
    except BaseException as exc:
        _append_error(
            result,
            _secondary_error(
                exc,
                case_id=case_id,
                category=ErrorCategory.DATABASE,
                stop_reason="RETRY_COUNT_FAILED",
                message="persisted retry accounting failed",
            ),
        )
        if not isinstance(exc, Exception):
            control_exception = control_exception or exc

    if persisted:
        refresh = await refresh_terminal(
            execution,
            result,
            persisted=persisted,
            persistence_error=persistence_error,
            control_exception=control_exception,
        )
        result = refresh.result
        persisted = refresh.persisted
        persistence_error = refresh.persistence_error
        control_exception = refresh.control_exception
    elif not terminal_write_attempted:
        persistence = await persist_terminal(
            execution,
            result,
            control_exception,
        )
        terminal_write_attempted = True
        result = persistence.result
        persisted = persistence.persisted
        persistence_error = persistence.persistence_error
        control_exception = persistence.control_exception

    artifact_failure: BaseException | None = None
    artifact_current = False
    if persisted:
        try:
            _write_bound_result(execution, result)
            artifact_current = True
        except asyncio.CancelledError as exc:
            cancellation_may_define_outcome = _cancellation_may_define_outcome(control_exception)
            control_exception = control_exception or exc
            if cancellation_may_define_outcome:
                result = _cancelled_result(
                    case_id, execution.source_id, execution.started_at, result
                )
                cancelled = True
            _append_error(
                result,
                ErrorRecord(
                    category=ErrorCategory.CANCELLED,
                    message="result artifact publication was cancelled",
                    case_id=case_id,
                    stop_reason="RESULT_ARTIFACT_WRITE_CANCELLED",
                ),
            )
            artifact_failure = exc
        except InvoiceAgentsError as exc:
            if exc.stop_reason in {
                "ARTIFACT_PUBLICATION_CLEANUP_UNRESOLVED",
                "ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED",
                "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED",
            }:
                result = _artifact_durability_unresolved_result(result)
            else:
                _append_error(
                    result,
                    _secondary_error(
                        exc,
                        case_id=case_id,
                        category=ErrorCategory.ORCHESTRATION,
                        stop_reason="RESULT_ARTIFACT_WRITE_FAILED",
                        message="atomic result artifact publication failed",
                    ),
                )
            artifact_failure = exc
        except BaseException as exc:
            _append_error(
                result,
                _secondary_error(
                    exc,
                    case_id=case_id,
                    category=ErrorCategory.ORCHESTRATION,
                    stop_reason="RESULT_ARTIFACT_WRITE_FAILED",
                    message="atomic result artifact publication failed",
                ),
            )
            if not isinstance(exc, Exception):
                control_exception = control_exception or exc
            artifact_failure = exc

        if artifact_failure is not None:
            refresh = await refresh_terminal(
                execution,
                result,
                persisted=persisted,
                persistence_error=persistence_error,
                control_exception=control_exception,
            )
            result = refresh.result
            persisted = refresh.persisted
            persistence_error = refresh.persistence_error
            control_exception = refresh.control_exception

        if (
            persisted
            and artifact_failure is not None
            and getattr(artifact_failure, _PUBLICATION_ROLLBACK_PROVEN, False) is True
        ):
            try:
                _write_bound_result(execution, result)
                artifact_current = True
            except BaseException as retry_exc:
                if not isinstance(retry_exc, Exception):
                    control_exception = control_exception or retry_exc
                result = _artifact_durability_unresolved_result(result)
                refresh = await refresh_terminal(
                    execution,
                    result,
                    persisted=persisted,
                    persistence_error=persistence_error,
                    control_exception=control_exception,
                )
                result = refresh.result
                persisted = refresh.persisted
                persistence_error = refresh.persistence_error
                control_exception = refresh.control_exception

    audit_failure: BaseException | None = None
    if execution.audit is not None:
        try:
            execution.audit.record(
                finished_event_type,
                result.model_dump(mode="json"),
                source_id=execution.source_id,
            )
        except asyncio.CancelledError as exc:
            audit_failure = exc
            cancellation_may_define_outcome = _cancellation_may_define_outcome(control_exception)
            control_exception = control_exception or exc
            if cancellation_may_define_outcome:
                result = _cancelled_result(
                    case_id, execution.source_id, execution.started_at, result
                )
                cancelled = True
            _append_error_before_artifact(
                result,
                ErrorRecord(
                    category=ErrorCategory.CANCELLED,
                    message="final audit write was cancelled",
                    case_id=case_id,
                    stop_reason="FINAL_AUDIT_WRITE_CANCELLED",
                ),
            )
            refresh = await refresh_terminal(
                execution,
                result,
                persisted=persisted,
                persistence_error=persistence_error,
                control_exception=control_exception,
            )
            result = refresh.result
            persisted = refresh.persisted
            persistence_error = refresh.persistence_error
            control_exception = refresh.control_exception
        except BaseException as exc:
            audit_failure = exc
            _append_error_before_artifact(
                result,
                _secondary_error(
                    exc,
                    case_id=case_id,
                    category=ErrorCategory.DATABASE,
                    stop_reason="FINAL_AUDIT_WRITE_FAILED",
                    message="final audit write failed",
                ),
            )
            if not isinstance(exc, Exception):
                control_exception = control_exception or exc
            refresh = await refresh_terminal(
                execution,
                result,
                persisted=persisted,
                persistence_error=persistence_error,
                control_exception=control_exception,
            )
            result = refresh.result
            persisted = refresh.persisted
            persistence_error = refresh.persistence_error
            control_exception = refresh.control_exception
    if audit_failure is not None and persisted and artifact_current:
        try:
            _write_bound_result(execution, result)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                control_exception = control_exception or exc
            result = _artifact_durability_unresolved_result(result)
            refresh = await refresh_terminal(
                execution,
                result,
                persisted=persisted,
                persistence_error=persistence_error,
                control_exception=control_exception,
            )
            result = refresh.result
            persisted = refresh.persisted
            persistence_error = refresh.persistence_error
            control_exception = refresh.control_exception
    if not persisted:
        assert persistence_error is not None
        _recovery_artifact_or_raise(
            result,
            persistence_error,
            store=execution.store,
            claim=execution.claim,
        )

    if control_exception is not None:
        raise control_exception
    return result


@dataclass(frozen=True, slots=True)
class _PreparationFailure:
    result: CaseResult
    recovery_published: bool


def _terminalize_preparation_failure(
    *,
    store: WorkflowStore,
    claim: ExecutionClaim,
    case_id: str,
    source_id: str | None,
    started_at: datetime,
    failure: BaseException,
) -> _PreparationFailure:
    control_exception = failure if not isinstance(failure, Exception) else None
    result = (
        _cancelled_result(case_id, source_id, started_at)
        if isinstance(failure, asyncio.CancelledError)
        else _failed_result(case_id, source_id, started_at, failure)
    )
    execution = _ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
        source_id=source_id,
        usage=UsageSummary(),
        clock=monotonic(),
    )
    persistence = _persist_terminal_result(execution, result, control_exception)
    result = persistence.result
    if not persistence.persisted:
        assert persistence.persistence_error is not None
        _recovery_artifact_or_raise(
            result,
            persistence.persistence_error,
            store=store,
            claim=claim,
        )
    if persistence.control_exception is not None:
        raise persistence.control_exception
    return _PreparationFailure(result=result, recovery_published=not persistence.persisted)


def _prepare_case(
    path: Path,
    settings: Settings,
    *,
    retain_execution_claim: bool,
    case_id: str | None = None,
    started_at: datetime | None = None,
    preparation_token: str | None = None,
    run_token: str | None = None,
) -> tuple[str, datetime] | tuple[str, datetime, ExecutionClaim] | _PreparationFailure:
    """Create immutable source/case evidence before any model request.

    Batch preparation is sequential so every later identity agent can see all submitted
    representations and revisions even though independent Swarms run concurrently.
    """

    parent_assigned = any(
        value is not None for value in (case_id, started_at, preparation_token, run_token)
    )
    if parent_assigned:
        if (
            type(case_id) is not str
            or not case_id.startswith("case_")
            or type(started_at) is not datetime
            or started_at.tzinfo is not UTC
            or type(preparation_token) is not str
            or type(run_token) is not str
            or preparation_token == run_token
            or not retain_execution_claim
        ):
            raise ValueError("invalid parent-assigned preparation authority")
    else:
        case_id = f"case_{uuid4().hex}"
        started_at = _now()
    assert case_id is not None and started_at is not None
    source_id: str | None = None
    case_created = False
    claim: ExecutionClaim | None = None
    store: WorkflowStore | None = None
    failure: BaseException | None = None
    try:
        source = snapshot_source(path, settings.source_archive_dir, settings.source_max_bytes)
        source_id = source.source_id
        store = WorkflowStore(settings)
        store.register_source(source)
        store.create_case(case_id, source, started_at)
        case_created = True
        claim = store.claim_case_execution(
            case_id,
            frozenset({CaseStatus.INCOMPLETE}),
            EXECUTION_LEASE_SECONDS,
            requested_token=preparation_token,
        )
        invoice = extract_invoice_evidence(source)
        store.save_extraction(case_id, invoice, claim)
        AuditRecorder(settings.workflow_db, case_id).record(
            "case.prepared",
            {
                "source": source.model_dump(mode="json"),
                "extraction_version": 1,
                "note": "pre-model extraction enables complete batch identity visibility",
            },
            source_id=source.source_id,
        )
        if retain_execution_claim:
            run_claim = store.handoff_case_execution(
                claim,
                EXECUTION_LEASE_SECONDS,
                requested_token=run_token,
            )
            return case_id, started_at, run_claim
        store.release_case_execution(claim)
        return case_id, started_at
    except BaseException as exc:
        failure = exc
    assert failure is not None
    if case_created and store is not None:
        if claim is None:
            try:
                claim = store.claim_case_execution(
                    case_id,
                    frozenset({CaseStatus.INCOMPLETE}),
                    EXECUTION_LEASE_SECONDS,
                    requested_token=preparation_token,
                )
            except BaseException as claim_failure:
                raise _unresolved_durability_error(case_id, claim_failure) from None
        return _terminalize_preparation_failure(
            store=store,
            claim=claim,
            case_id=case_id,
            source_id=source_id,
            started_at=started_at,
            failure=failure,
        )
    result = _failed_result(case_id, source_id, started_at, failure)
    if not isinstance(failure, Exception):
        raise failure
    return _PreparationFailure(result=result, recovery_published=False)


def prepare_case(path: Path, settings: Settings) -> tuple[str, datetime] | CaseResult:
    prepared = _prepare_case(path, settings, retain_execution_claim=False)
    return (
        prepared.result
        if isinstance(prepared, _PreparationFailure)
        else cast(tuple[str, datetime], prepared)
    )


def prepare_claimed_case(
    path: Path, settings: Settings
) -> tuple[str, datetime, ExecutionClaim] | _PreparationFailure:
    """Prepare a batch case while retaining its exact publication disposition."""

    return cast(
        tuple[str, datetime, ExecutionClaim] | _PreparationFailure,
        _prepare_case(path, settings, retain_execution_claim=True),
    )


def _event_semantic_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _event_semantic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_event_semantic_value(item) for item in value]
    return value


def _event_payload(event: Any) -> dict[str, Any]:
    """Build an allowlisted semantic event without retaining the raw model dump."""

    event_type = f"autogen.{type(event).__name__}"
    payload: dict[str, Any] = {
        "type": type(event).__name__,
        "source": getattr(event, "source", None),
    }
    model_usage = getattr(event, "models_usage", None)
    if model_usage is not None:
        payload["models_usage"] = {
            "prompt_tokens": getattr(model_usage, "prompt_tokens", None),
            "completion_tokens": getattr(model_usage, "completion_tokens", None),
        }
    if isinstance(event, ToolCallRequestEvent):
        payload["content"] = [
            {
                "id": getattr(call, "id", None),
                "name": getattr(call, "name", None),
                "arguments": getattr(call, "arguments", None),
            }
            for call in event.content
        ]
    elif isinstance(event, ToolCallExecutionEvent):
        payload["content"] = [
            {
                "call_id": getattr(execution, "call_id", None),
                "name": getattr(execution, "name", None),
                "content": getattr(execution, "content", None),
                "is_error": getattr(execution, "is_error", None),
            }
            for execution in event.content
        ]
    elif hasattr(event, "content"):
        payload["content"] = _event_semantic_value(event.content)
    if isinstance(event, HandoffMessage):
        payload["target"] = getattr(event, "target", None)
    return normalize_autogen_event_payload(event_type, payload)


def _validated_tool_call_ids(values: list[object], case_id: str) -> list[str]:
    validated: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise InvoiceAgentsError(
                ErrorCategory.SCHEMA,
                "AutoGen tool event omitted its correlation ID",
                case_id=case_id,
                stop_reason="TOOL_CALL_ID_MISSING",
            ) from None
        if (
            len(value) > 256
            or value != value.strip()
            or not value.isprintable()
            or sanitize_text(value) != value
        ):
            raise InvoiceAgentsError(
                ErrorCategory.SCHEMA,
                "AutoGen tool event returned an invalid correlation ID",
                case_id=case_id,
                stop_reason="TOOL_CALL_ID_INVALID",
            ) from None
        validated.append(value)
    return validated


def _record_stream_event(
    event: Any,
    context: AgentCaseContext,
    usage: UsageSummary,
) -> None:
    """Account for one AutoGen event and persist normalized per-call audit rows."""

    event_payload = _event_payload(event)
    model_usage = getattr(event, "models_usage", None)
    if model_usage is not None:
        usage.prompt_tokens += int(model_usage.prompt_tokens)
        usage.completion_tokens += int(model_usage.completion_tokens)
        usage.model_calls += 1
    if isinstance(event, ToolCallRequestEvent):
        usage.tool_calls += len(event.content)

    tool_call_ids: list[str] | None = None
    if isinstance(event, ToolCallRequestEvent):
        tool_call_ids = _validated_tool_call_ids(
            [getattr(call, "id", None) for call in event.content], context.case_id
        )
    elif isinstance(event, ToolCallExecutionEvent):
        tool_call_ids = _validated_tool_call_ids(
            [getattr(execution, "call_id", None) for execution in event.content],
            context.case_id,
        )

    content_payloads: list[Any] | None = None
    if tool_call_ids is not None:
        raw_content = event_payload.get("content")
        if not isinstance(raw_content, list) or len(raw_content) != len(tool_call_ids):
            raise InvoiceAgentsError(
                ErrorCategory.SCHEMA,
                "AutoGen tool event payload did not match its correlation items",
                case_id=context.case_id,
                stop_reason="TOOL_EVENT_PAYLOAD_INVALID",
            ) from None
        content_payloads = raw_content

    if isinstance(event, ToolCallExecutionEvent):
        assert content_payloads is not None
        for execution, content_payload in zip(event.content, content_payloads, strict=True):
            if execution.is_error:
                safe_item = content_payload if isinstance(content_payload, Mapping) else {}
                safe_name = safe_item.get("name", "unknown_tool")
                safe_content = safe_item.get("content", "tool execution failed")
                context.tool_failures.append(
                    sanitize_text(f"{safe_name}({execution.call_id}): {safe_content}")
                )

    source_id = context.invoice().source.source_id
    metadata = getattr(event, "metadata", {}) or {}
    provider_request_id = metadata.get("request_id") if isinstance(metadata, Mapping) else None
    record_kwargs = {
        "source_id": source_id,
        "agent_name": event_payload.get("source"),
        "provider_request_id": provider_request_id,
    }
    event_type = f"autogen.{type(event).__name__}"
    if tool_call_ids is None:
        context.audit.record(event_type, event_payload, **record_kwargs)
        return

    assert content_payloads is not None
    for tool_call_id, content_payload in zip(tool_call_ids, content_payloads, strict=True):
        context.audit.record(
            event_type,
            {**event_payload, "content": [content_payload]},
            tool_call_id=tool_call_id,
            **record_kwargs,
        )


async def _stream_team(
    team: Any,
    context: AgentCaseContext,
    *,
    task: str | TextMessage | HandoffMessage,
    usage: UsageSummary,
) -> TaskResult:
    final_task_result: TaskResult | None = None
    async for event in team.run_stream(task=task):
        if isinstance(event, TaskResult):
            final_task_result = event
            continue
        _record_stream_event(event, context, usage)
    if final_task_result is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "AutoGen stream ended without TaskResult",
            case_id=context.case_id,
            stop_reason="TASK_RESULT_MISSING",
        )
    return final_task_result


def _result_from_stop(
    context: AgentCaseContext,
    task_result: TaskResult,
    started_at: datetime,
    usage: UsageSummary,
) -> CaseResult:
    stop_reason = task_result.stop_reason or "AUTOGEN_STOP_REASON_MISSING"
    review = context.store.load_current_review(context.claim)
    final = context.store.load_current_final_decision(context.claim)
    errors = [
        ErrorRecord(
            category=ErrorCategory.TOOL,
            message=sanitize_text(failure),
            case_id=context.case_id,
            stop_reason="TOOL_EXECUTION_FAILED",
        )
        for failure in context.tool_failures
    ]
    if review is not None and review.status == "PENDING":
        status = CaseStatus.NEEDS_HUMAN
        normalized_stop = "HUMAN_REVIEW_REQUESTED"
    elif is_max_messages_stop(stop_reason):
        status = CaseStatus.INCOMPLETE
        normalized_stop = "MAX_MESSAGES_EXHAUSTED"
    elif errors:
        status = CaseStatus.FAILED
        normalized_stop = "TOOL_EXECUTION_FAILED"
    elif final is None:
        status = CaseStatus.FAILED
        normalized_stop = "FINAL_DECISION_MISSING"
        errors.append(
            ErrorRecord(
                category=ErrorCategory.SCHEMA,
                message="team stopped without a persisted schema-valid final decision",
                case_id=context.case_id,
                stop_reason=normalized_stop,
            )
        )
    elif final.decision is DecisionKind.FAILED:
        status = CaseStatus.FAILED
        normalized_stop = "FINAL_DECISION_FAILED"
    elif final.decision is DecisionKind.APPROVE:
        payment = context.payment_result
        if payment is None:
            status = CaseStatus.FAILED
            normalized_stop = "APPROVED_PAYMENT_RESULT_MISSING"
            errors.append(
                ErrorRecord(
                    category=ErrorCategory.PAYMENT,
                    message="approved case stopped without a payment result",
                    case_id=context.case_id,
                    stop_reason=normalized_stop,
                )
            )
        elif payment.status in {PaymentStatus.PAID, PaymentStatus.DUPLICATE}:
            status = CaseStatus.SUCCEEDED
            normalized_stop = "APPROVED_PAYMENT_RECORDED"
        else:
            status = CaseStatus.FAILED
            normalized_stop = "PAYMENT_FAILED"
            errors.append(
                ErrorRecord(
                    category=ErrorCategory.PAYMENT,
                    message=sanitize_text(payment.error or f"payment status is {payment.status}"),
                    case_id=context.case_id,
                    stop_reason=normalized_stop,
                )
            )
    else:
        status = CaseStatus.SUCCEEDED
        normalized_stop = f"DECISION_{final.decision}"
    return CaseResult(
        case_id=context.case_id,
        source_id=context.invoice().source.source_id,
        status=status,
        stop_reason=normalized_stop,
        final_decision=final,
        review_request=review,
        payment=context.payment_result,
        errors=errors,
        usage=usage,
        started_at=started_at,
        finished_at=_now(),
    )


@dataclass(frozen=True, slots=True)
class _PublicationReceipt:
    artifact_sha256: str
    artifact_device: int
    artifact_inode: int
    artifact_file_type: int
    artifact_size_bytes: int


@dataclass(slots=True)
class _PublicationLockEntry:
    lock: Any
    users: int = 0


_PUBLICATION_LOCKS_GUARD = threading.Lock()
_PUBLICATION_LOCKS: dict[str, _PublicationLockEntry] = {}


@contextmanager
def _serialized_publication(target: Path) -> Any:
    key = os.fspath(target.parent.absolute() / target.name)
    with _PUBLICATION_LOCKS_GUARD:
        entry = _PUBLICATION_LOCKS.get(key)
        if entry is None:
            entry = _PublicationLockEntry(threading.Lock())
            _PUBLICATION_LOCKS[key] = entry
        entry.users += 1
    entry.lock.acquire()
    try:
        yield
    finally:
        entry.lock.release()
        with _PUBLICATION_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0 and _PUBLICATION_LOCKS.get(key) is entry:
                del _PUBLICATION_LOCKS[key]


def _result_artifact_target(result: CaseResult) -> Path:
    output_dir = Path.cwd() / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{result.case_id}.json"


def _write_result(result: CaseResult) -> Path:
    result = sanitize_case_result(result)
    target = _result_artifact_target(result)
    _atomic_publish(target, result.model_dump_json(indent=2).encode("utf-8"))
    return target


def _write_result_for_generation(
    store: WorkflowStore,
    result: CaseResult,
    execution_generation: int,
) -> Path:
    result = sanitize_case_result(result)
    target = _result_artifact_target(result)
    payload = result.model_dump_json(indent=2).encode("utf-8")

    def bind_published_candidate(receipt: _PublicationReceipt) -> BaseException | None:
        binding = ResultArtifactBinding(
            case_id=result.case_id,
            execution_generation=execution_generation,
            artifact_sha256=receipt.artifact_sha256,
            artifact_device=receipt.artifact_device,
            artifact_inode=receipt.artifact_inode,
            artifact_file_type=receipt.artifact_file_type,
            artifact_size_bytes=receipt.artifact_size_bytes,
        )
        try:
            store.save_result_artifact_binding(binding, result)
        except BaseException as failure:
            try:
                observed_result, observed_generation, observed_binding = (
                    store.load_result_with_artifact_binding(result.case_id)
                )
            except BaseException as readback_failure:
                for unresolved in (failure, readback_failure):
                    if _is_process_control(unresolved):
                        setattr(unresolved, _PUBLICATION_PRESERVE_EVIDENCE, True)
                        raise _clear_exception_chain(unresolved) from None
                unresolved_error = _publication_failure(
                    "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED"
                )
                setattr(unresolved_error, _PUBLICATION_PRESERVE_EVIDENCE, True)
                raise unresolved_error from None
            exact_parent = (
                observed_result == result
                and observed_generation == execution_generation
            )
            if exact_parent and observed_binding == binding:
                return failure if _is_process_control(failure) else None
            if exact_parent and observed_binding is None:
                raise _clear_exception_chain(failure) from None
            if _is_process_control(failure):
                setattr(failure, _PUBLICATION_PRESERVE_EVIDENCE, True)
                raise _clear_exception_chain(failure) from None
            unresolved_error = _publication_failure(
                "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED"
            )
            setattr(unresolved_error, _PUBLICATION_PRESERVE_EVIDENCE, True)
            raise unresolved_error from None
        return None

    with _serialized_publication(target):
        _atomic_publish_locked(
            target,
            payload,
            bind_published_candidate=bind_published_candidate,
        )
    return target


def _write_bound_result(execution: _ClaimedExecution, result: CaseResult) -> Path:
    return _write_result_for_generation(
        execution.store,
        result,
        execution.claim.generation,
    )


_PUBLICATION_ROLLBACK_PROVEN = "_invoice_agents_publication_rollback_proven"
_PUBLICATION_PRESERVE_EVIDENCE = "_invoice_agents_publication_preserve_evidence"


def _is_process_control(error: BaseException) -> bool:
    return not isinstance(error, Exception)


def _clear_exception_chain(error: BaseException) -> BaseException:
    error.__cause__ = None
    error.__context__ = None
    return error


def _publication_failure(stop_reason: str) -> InvoiceAgentsError:
    messages = {
        "ARTIFACT_PUBLICATION_CLEANUP_UNRESOLVED": (
            "atomic artifact publication cleanup could not be proven complete"
        ),
        "ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED": (
            "atomic artifact rollback durability could not be proven"
        ),
        "ARTIFACT_PUBLICATION_TARGET_UNSAFE": (
            "atomic artifact publication target is not a regular file"
        ),
        "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED": (
            "atomic artifact publication namespace identity could not be proven"
        ),
        "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED": (
            "result-artifact binding durability could not be proven"
        ),
    }
    return InvoiceAgentsError(
        ErrorCategory.ORCHESTRATION,
        messages[stop_reason],
        stop_reason=stop_reason,
    )


def _retire_and_close(descriptor: int) -> BaseException | None:
    """Close one owned descriptor once; callers must retire ownership first."""

    try:
        os.close(descriptor)
    except BaseException as exc:
        return exc
    return None


def _atomic_publish(target: Path, payload: bytes) -> _PublicationReceipt:
    with _serialized_publication(target):
        return _atomic_publish_locked(target, payload)


def _atomic_publish_locked(
    target: Path,
    payload: bytes,
    *,
    bind_published_candidate: (
        Callable[[_PublicationReceipt], BaseException | None] | None
    ) = None,
) -> _PublicationReceipt:
    """Publish bytes durably or restore the exact prior namespace durably.

    Raw descriptor ownership is transferred away before every close call.  A
    close that reports failure may already have released and recycled the
    numeric descriptor, so retrying or probing that number would be unsafe.
    """

    temporary = target.with_name(f"{target.name}.tmp")
    rollback = target.with_name(f".{target.name}.rollback-{uuid4().hex}")
    file_descriptor: int | None = None
    directory_descriptors: list[int] = []
    temporary_present = False
    rollback_present = False
    prior_present = False
    candidate_replaced = False
    rollback_applied = False
    candidate_identity: tuple[int, int, int, int] | None = None
    prior_identity: tuple[int, int, int, int] | None = None
    published_identity: tuple[int, int, int, int] | None = None
    publication_directory_identity: tuple[int, int] | None = None
    binding_proven = False
    preserve_publication_evidence = False
    deferred_control: BaseException | None = None
    primary_failure: BaseException | None = None
    cleanup_failures: list[BaseException] = []
    rollback_failures: list[BaseException] = []
    controls: list[BaseException] = []

    def capture(error: BaseException, *, cleanup: bool = False, rollback: bool = False) -> None:
        nonlocal primary_failure
        if _is_process_control(error):
            controls.append(error)
        if rollback:
            rollback_failures.append(error)
        elif cleanup:
            cleanup_failures.append(error)
        elif primary_failure is None:
            primary_failure = error

    def lstat_entry(name: str) -> os.stat_result | None:
        try:
            return os.lstat(name, dir_fd=directory_descriptors[0])
        except FileNotFoundError as exc:
            if exc.errno == errno.ENOENT:
                return None
            raise _publication_failure("ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED") from None
        except OSError:
            raise _publication_failure("ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED") from None

    def identity_tuple(identity: os.stat_result) -> tuple[int, int, int, int]:
        return (
            identity.st_dev,
            identity.st_ino,
            stat.S_IFMT(identity.st_mode),
            identity.st_size,
        )

    def acquire_publication_directory() -> int:
        try:
            descriptor = os.open(
                target.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            directory_descriptors.append(descriptor)
            opened = os.fstat(descriptor)
        except OSError:
            raise _publication_failure(
                "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
            ) from None
        if not stat.S_ISDIR(opened.st_mode):
            raise _publication_failure(
                "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
            ) from None
        opened_identity = (opened.st_dev, opened.st_ino)
        if (
            publication_directory_identity is None
            or opened_identity != publication_directory_identity
        ):
            raise _publication_failure(
                "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
            ) from None
        return descriptor

    try:
        try:
            classified_directory = os.lstat(target.parent)
            if not stat.S_ISDIR(classified_directory.st_mode):
                raise _publication_failure(
                    "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
                ) from None
            publication_directory_identity = (
                classified_directory.st_dev,
                classified_directory.st_ino,
            )
            # Acquire every namespace capability before creating the
            # candidate.  A directory-open failure therefore cannot strand a
            # temporary entry, and a later candidate-close failure can still
            # be contained through an already-owned directory descriptor.
            for _ in range(2):
                acquire_publication_directory()
        except OSError:
            capture(
                _publication_failure("ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED")
            )
        except BaseException as exc:
            capture(exc)

        if primary_failure is None and not controls:
            try:
                file_descriptor = os.open(
                    temporary.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_descriptors[0],
                )
                temporary_present = True
                offset = 0
                while offset < len(payload):
                    written = os.write(file_descriptor, payload[offset:])
                    if written <= 0:
                        raise OSError("atomic artifact write made no forward progress")
                    offset += written
                os.fsync(file_descriptor)
                candidate_identity = identity_tuple(os.fstat(file_descriptor))
                if (
                    candidate_identity[2] != stat.S_IFREG
                    or candidate_identity[3] != len(payload)
                ):
                    raise _publication_failure(
                        "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
                    ) from None
            except BaseException as exc:
                capture(exc)

        if file_descriptor is not None:
            owned_descriptor = file_descriptor
            file_descriptor = None
            close_failure = _retire_and_close(owned_descriptor)
            if close_failure is not None:
                capture(close_failure, cleanup=True)

        if primary_failure is None and not cleanup_failures and not controls:
            try:
                classified = lstat_entry(target.name)
                prior_present = classified is not None
                if prior_present:
                    assert classified is not None
                    if not stat.S_ISREG(classified.st_mode):
                        raise _publication_failure("ARTIFACT_PUBLICATION_TARGET_UNSAFE") from None
                    prior_identity = identity_tuple(classified)
                    os.link(
                        target.name,
                        rollback.name,
                        src_dir_fd=directory_descriptors[0],
                        dst_dir_fd=directory_descriptors[0],
                        follow_symlinks=False,
                    )
                    rollback_present = True
                    rollback_identity = lstat_entry(rollback.name)
                    current_identity = lstat_entry(target.name)
                    if (
                        rollback_identity is None
                        or current_identity is None
                        or identity_tuple(rollback_identity) != prior_identity
                        or identity_tuple(current_identity) != prior_identity
                    ):
                        os.unlink(rollback.name, dir_fd=directory_descriptors[0])
                        rollback_present = False
                        raise _publication_failure(
                            "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
                        ) from None
                os.replace(
                    temporary.name,
                    target.name,
                    src_dir_fd=directory_descriptors[0],
                    dst_dir_fd=directory_descriptors[0],
                )
                temporary_present = False
                candidate_replaced = True
                replaced_identity = lstat_entry(target.name)
                if (
                    candidate_identity is None
                    or replaced_identity is None
                    or identity_tuple(replaced_identity) != candidate_identity
                ):
                    raise _publication_failure(
                        "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
                    ) from None
                published_identity = candidate_identity
                os.fsync(directory_descriptors[0])
                # Reopen the still-named final directory after the candidate
                # is durable.  The callback may commit the database binding,
                # so an early capability cannot authorize a renamed directory.
                acquire_publication_directory()
                if bind_published_candidate is not None:
                    receipt = _PublicationReceipt(
                        artifact_sha256=hashlib.sha256(payload).hexdigest(),
                        artifact_device=candidate_identity[0],
                        artifact_inode=candidate_identity[1],
                        artifact_file_type=candidate_identity[2],
                        artifact_size_bytes=candidate_identity[3],
                    )
                    deferred_control = bind_published_candidate(receipt)
                    if deferred_control is not None and not _is_process_control(
                        deferred_control
                    ):
                        raise TypeError("binding callback returned a non-control failure")
                    binding_proven = True
                    preserve_publication_evidence = True
            except BaseException as exc:
                preserve_publication_evidence = (
                    binding_proven
                    or getattr(exc, _PUBLICATION_PRESERVE_EVIDENCE, False) is True
                )
                remaining_temporary = lstat_entry(temporary.name)
                if temporary_present and remaining_temporary is None:
                    temporary_present = False
                    current_identity = lstat_entry(target.name)
                    candidate_replaced = (
                        candidate_identity is not None
                        and current_identity is not None
                        and identity_tuple(current_identity) == candidate_identity
                    )
                    if candidate_replaced:
                        published_identity = candidate_identity
                capture(exc)

        # The first directory close is part of publication.  Keep independent
        # pre-opened descriptors and the rollback link until it succeeds.
        if candidate_replaced and primary_failure is None and not cleanup_failures and not controls:
            owned_descriptor = directory_descriptors.pop(0)
            close_failure = _retire_and_close(owned_descriptor)
            if close_failure is not None:
                capture(close_failure, cleanup=True)

        publication_failed = primary_failure is not None or cleanup_failures or controls
        if candidate_replaced and publication_failed and not preserve_publication_evidence:
            try:
                current_identity = lstat_entry(target.name)
                if (
                    candidate_identity is None
                    or current_identity is None
                    or identity_tuple(current_identity) != candidate_identity
                ):
                    raise _publication_failure(
                        "ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED"
                    ) from None
                if prior_present:
                    os.replace(
                        rollback.name,
                        target.name,
                        src_dir_fd=directory_descriptors[0],
                        dst_dir_fd=directory_descriptors[0],
                    )
                    rollback_present = False
                else:
                    os.unlink(target.name, dir_fd=directory_descriptors[0])
                rollback_applied = True
            except BaseException as exc:
                current_identity = lstat_entry(target.name)
                remaining_rollback = lstat_entry(rollback.name) if prior_present else None
                if (
                    prior_present
                    and prior_identity is not None
                    and current_identity is not None
                    and identity_tuple(current_identity) == prior_identity
                    and remaining_rollback is None
                ):
                    rollback_present = False
                    rollback_applied = True
                elif not prior_present and current_identity is None:
                    rollback_applied = True
                capture(exc, rollback=True)

            if rollback_applied:
                try:
                    os.fsync(directory_descriptors[0])
                except BaseException as exc:
                    capture(exc, rollback=True)
            else:
                try:
                    os.fsync(directory_descriptors[0])
                except BaseException as exc:
                    capture(exc, rollback=True)
                rollback_failures.append(
                    _publication_failure("ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED")
                )

        if candidate_replaced and publication_failed and preserve_publication_evidence:
            try:
                os.fsync(directory_descriptors[0])
            except BaseException as exc:
                capture(exc, rollback=True)

        if not publication_failed and candidate_replaced and rollback_present:
            try:
                os.unlink(rollback.name, dir_fd=directory_descriptors[0])
                rollback_present = False
                os.fsync(directory_descriptors[0])
            except BaseException as exc:
                capture(exc, cleanup=True)

        # Namespace cleanup uses an owned directory capability before raw
        # descriptor ownership is retired.
        if temporary_present:
            try:
                os.unlink(temporary.name, dir_fd=directory_descriptors[0])
                temporary_present = False
            except FileNotFoundError as exc:
                if exc.errno == errno.ENOENT:
                    temporary_present = False
                else:
                    capture(exc, cleanup=True)
            except BaseException as exc:
                capture(exc, cleanup=True)
        if rollback_present and not candidate_replaced:
            try:
                os.unlink(rollback.name, dir_fd=directory_descriptors[0])
                rollback_present = False
            except FileNotFoundError as exc:
                if exc.errno == errno.ENOENT:
                    rollback_present = False
                else:
                    capture(exc, cleanup=True)
            except BaseException as exc:
                capture(exc, cleanup=True)
        # Cleanup is exhaustive.  Each descriptor is removed from ownership
        # before close, and every remaining resource is attempted once.
        while directory_descriptors:
            owned_descriptor = directory_descriptors.pop(0)
            close_failure = _retire_and_close(owned_descriptor)
            if close_failure is not None:
                capture(close_failure, cleanup=True)
    finally:
        # Defensive containment for failures in the cleanup implementation
        # itself.  Ownership is still retired before close and never retried.
        if file_descriptor is not None:
            owned_descriptor = file_descriptor
            file_descriptor = None
            close_failure = _retire_and_close(owned_descriptor)
            if close_failure is not None:
                capture(close_failure, cleanup=True)
        while directory_descriptors:
            owned_descriptor = directory_descriptors.pop(0)
            close_failure = _retire_and_close(owned_descriptor)
            if close_failure is not None:
                capture(close_failure, cleanup=True)

    if deferred_control is not None:
        raise _clear_exception_chain(deferred_control) from None
    if controls:
        control = _clear_exception_chain(controls[0])
        if candidate_replaced and rollback_applied and not rollback_failures:
            setattr(control, _PUBLICATION_ROLLBACK_PROVEN, True)
        raise control from None
    if rollback_failures:
        error = _publication_failure("ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED")
        raise _clear_exception_chain(error) from None
    if cleanup_failures:
        error = _publication_failure("ARTIFACT_PUBLICATION_CLEANUP_UNRESOLVED")
        raise _clear_exception_chain(error) from None
    if primary_failure is not None:
        failure = _clear_exception_chain(primary_failure)
        if candidate_replaced and rollback_applied:
            setattr(failure, _PUBLICATION_ROLLBACK_PROVEN, True)
        raise failure from None
    if published_identity is None:
        raise _publication_failure("ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED") from None
    return _PublicationReceipt(
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_device=published_identity[0],
        artifact_inode=published_identity[1],
        artifact_file_type=published_identity[2],
        artifact_size_bytes=published_identity[3],
    )


async def _run_prepared_case_in_process(
    case_id: str,
    started_at: datetime,
    settings: Settings,
    *,
    claim: ExecutionClaim | None = None,
    terminal_writes_in_process: bool = False,
) -> CaseResult:
    """Child-side fresh Swarm lifecycle; public callers use the process boundary."""

    if claim is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "an exact execution claim is required to run a prepared case",
            case_id=case_id,
            stop_reason="EXECUTION_CLAIM_MISSING",
        ) from None
    claim = validate_execution_claim(claim, expected_case_id=case_id)
    store = WorkflowStore(settings)
    store.require_current_execution_claim(claim)

    async def execute_lifecycle(execution: _ClaimedExecution) -> CaseResult:
        invoice = store.promote_predecessor_extraction(execution.claim)
        execution.source_id = invoice.source.source_id
        audit = AuditRecorder(settings.workflow_db, case_id)
        execution.audit = audit
        audit.record(
            "provider.configuration",
            {
                "model": XAI_MODEL,
                "base_url": XAI_BASE_URL,
                "parallel_tool_calls": False,
                "reasoning_effort": "high",
                "configured_transient_retries": settings.transient_retries,
                "include_name_in_message": False,
                "add_name_prefixes": True,
                "zdr_response_header_status": (
                    "NOT_EXPOSED_BY_AUTOGEN_OPENAI_CLIENT; no ZDR claim recorded"
                ),
            },
            source_id=invoice.source.source_id,
        )
        context = AgentCaseContext(
            case_id=case_id,
            settings=settings,
            store=store,
            audit=audit,
            claim=execution.claim,
        )
        execution.context = context
        usage = cast(UsageSummary, execution.usage)
        client = create_model_client(settings)
        execution.client = client
        team = build_team(context, client)

        with bind_audit_recorder(audit):
            task_result = await _stream_team(
                team,
                context,
                task=(
                    f"Process case {case_id} end to end. Use only recorded evidence and tools. "
                    "Any failure or ambiguity must remain visible."
                ),
                usage=usage,
            )
        # State is saved only after run_stream has yielded its TaskResult and stopped.
        state = await team.save_state()
        store.save_team_state(case_id, dict(state), execution.claim)
        usage.latency_ms = int((monotonic() - execution.clock) * 1000)
        return _result_from_stop(context, task_result, started_at, usage)

    async def heartbeat_lifecycle(execution: _ClaimedExecution) -> CaseResult:
        def replace_claim(renewed: ExecutionClaim) -> None:
            renewed = validate_execution_claim(renewed, expected_case_id=case_id)
            execution.claim = renewed
            if execution.context is not None:
                execution.context.claim = renewed

        return await _run_with_lease_heartbeat(
            execute_lifecycle(execution),
            renew=store.renew_case_execution,
            claim=execution.claim,
            replace_claim=replace_claim,
            lease_seconds=EXECUTION_LEASE_SECONDS,
            renewal_interval_seconds=EXECUTION_RENEWAL_INTERVAL_SECONDS,
        )

    return await _execute_claimed_case(
        case_id,
        started_at,
        store,
        claim,
        heartbeat_lifecycle,
        finished_event_type="case.finished",
        terminal_writes_in_process=terminal_writes_in_process,
    )


@dataclass(frozen=True, slots=True)
class _LifecycleBoundaryOutcome:
    acknowledged: bool
    error_code: str | None


def _lifecycle_boundary_error(case_id: str, code: str) -> InvoiceAgentsError:
    contracts: dict[str, tuple[ErrorCategory, str]] = {
        "LIFECYCLE_FAILED": (
            ErrorCategory.ORCHESTRATION,
            "isolated case lifecycle failed",
        ),
        "LIFECYCLE_WORKER_CRASHED": (
            ErrorCategory.ORCHESTRATION,
            "isolated case lifecycle worker exited unexpectedly",
        ),
        "LIFECYCLE_WORKER_PROTOCOL_INVALID": (
            ErrorCategory.ORCHESTRATION,
            "isolated case lifecycle worker returned an invalid acknowledgement",
        ),
        "LIFECYCLE_WORKER_TIMED_OUT": (
            ErrorCategory.TIMEOUT,
            "isolated case lifecycle exceeded its bounded deadline",
        ),
    }
    category, message = contracts.get(
        code,
        (ErrorCategory.ORCHESTRATION, "isolated case lifecycle failed"),
    )
    return InvoiceAgentsError(
        category,
        message,
        case_id=case_id,
        stop_reason=code,
    )


async def _invoke_lifecycle_boundary(
    *,
    mode: Literal["process", "resume"],
    settings: Settings,
    claim: ExecutionClaim,
    started_at: datetime,
) -> tuple[_LifecycleBoundaryOutcome, asyncio.CancelledError | None]:
    """Drain a cancellation-aware process controller without retaining raw failures."""

    from invoice_agents.isolated_process import ProcessCancellation

    cancel_requested = ProcessCancellation()

    def invoke() -> _LifecycleBoundaryOutcome:
        try:
            from invoice_agents import lifecycle_process

            outcome = lifecycle_process.run_lifecycle_process(
                mode=mode,
                settings=settings,
                claim=claim,
                started_at=started_at,
                timeout_seconds=float(EXECUTION_LEASE_SECONDS),
                cancel_requested=cancel_requested,
            )
            return _LifecycleBoundaryOutcome(
                acknowledged=outcome.acknowledged,
                error_code=outcome.error_code,
            )
        except BaseException:
            return _LifecycleBoundaryOutcome(False, "LIFECYCLE_WORKER_CRASHED")

    controller = asyncio.create_task(
        asyncio.to_thread(invoke),
        name=f"invoice-lifecycle-controller-{claim.case_id}",
    )
    cancellation: asyncio.CancelledError | None = None
    try:
        outcome = await asyncio.shield(controller)
    except asyncio.CancelledError as exc:
        cancellation = exc
        cancel_requested.set()
        try:
            outcome = await _await_task_despite_cancellation(
                controller,
                deadline=(
                    monotonic()
                    + DURABILITY_DEADLINE_SECONDS
                    + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                ),
                case_id=claim.case_id,
            )
        except BaseException:
            outcome = _LifecycleBoundaryOutcome(
                False,
                "LIFECYCLE_WORKER_CLEANUP_FAILED",
            )
    return outcome, cancellation


def _same_running_claim(
    issued: ExecutionClaim,
    snapshot: CaseExecutionSnapshot,
) -> ExecutionClaim | None:
    if (
        snapshot.execution_state == "RUNNING"
        and snapshot.execution_token == issued.token
        and snapshot.execution_generation == issued.generation
        and snapshot.lease_expires_at is not None
    ):
        return ExecutionClaim(
            issued.case_id,
            issued.token,
            issued.generation,
            snapshot.lease_expires_at,
        )
    return None


async def _publish_parent_terminal_result(
    execution: _ClaimedExecution,
    persistence: _PersistenceOutcome,
) -> CaseResult:
    """Publish the parent-owned result or its sole claim-bound recovery frame."""

    result = persistence.result
    persisted = persistence.persisted
    persistence_error = persistence.persistence_error
    control_exception = persistence.control_exception
    if not persisted:
        assert persistence_error is not None
        _recovery_artifact_or_raise(
            result,
            persistence_error,
            store=execution.store,
            claim=execution.claim,
        )
    else:
        artifact_failure: BaseException | None = None
        try:
            _write_bound_result(execution, result)
        except asyncio.CancelledError as exc:
            cancellation_may_define_outcome = _cancellation_may_define_outcome(control_exception)
            control_exception = control_exception or exc
            if cancellation_may_define_outcome:
                result = _cancelled_result(
                    execution.case_id,
                    execution.source_id,
                    execution.started_at,
                    result,
                )
            _append_error(
                result,
                ErrorRecord(
                    category=ErrorCategory.CANCELLED,
                    message="result artifact publication was cancelled",
                    case_id=execution.case_id,
                    stop_reason="RESULT_ARTIFACT_WRITE_CANCELLED",
                ),
            )
            artifact_failure = exc
        except InvoiceAgentsError as exc:
            if exc.stop_reason in {
                "ARTIFACT_PUBLICATION_CLEANUP_UNRESOLVED",
                "ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED",
                "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED",
            }:
                result = _artifact_durability_unresolved_result(result)
            else:
                _append_error(
                    result,
                    _secondary_error(
                        exc,
                        case_id=execution.case_id,
                        category=ErrorCategory.ORCHESTRATION,
                        stop_reason="RESULT_ARTIFACT_WRITE_FAILED",
                        message="atomic result artifact publication failed",
                    ),
                )
            artifact_failure = exc
        except BaseException as exc:
            _append_error(
                result,
                _secondary_error(
                    exc,
                    case_id=execution.case_id,
                    category=ErrorCategory.ORCHESTRATION,
                    stop_reason="RESULT_ARTIFACT_WRITE_FAILED",
                    message="atomic result artifact publication failed",
                ),
            )
            if not isinstance(exc, Exception):
                control_exception = control_exception or exc
            artifact_failure = exc
        if artifact_failure is not None:
            refreshed = await _refresh_terminal_evidence_safely(
                execution,
                result,
                persisted=True,
                persistence_error=persistence_error,
                control_exception=control_exception,
            )
            result = refreshed.result
            persisted = refreshed.persisted
            control_exception = refreshed.control_exception
            if not persisted:
                raise _unresolved_durability_error(execution.case_id) from None
        if (
            artifact_failure is not None
            and getattr(artifact_failure, _PUBLICATION_ROLLBACK_PROVEN, False) is True
        ):
            try:
                _write_bound_result(execution, result)
            except BaseException as retry_exc:
                if not isinstance(retry_exc, Exception):
                    control_exception = control_exception or retry_exc
                result = _artifact_durability_unresolved_result(result)
                refreshed = await _refresh_terminal_evidence_safely(
                    execution,
                    result,
                    persisted=True,
                    persistence_error=persistence_error,
                    control_exception=control_exception,
                )
                result = refreshed.result
                persisted = refreshed.persisted
                control_exception = refreshed.control_exception
                if not persisted:
                    raise _unresolved_durability_error(execution.case_id) from None
    if control_exception is not None:
        raise control_exception
    return result


async def _terminalize_lifecycle_boundary_failure(
    *,
    store: WorkflowStore,
    claim: ExecutionClaim,
    started_at: datetime,
    failure: BaseException,
    load_previous: bool = True,
) -> CaseResult:
    """Publish one parent-owned terminal result after the worker session is empty."""

    try:
        source_id = store.load_authoritative_case_source_id(claim)
        previous = store.load_result(claim.case_id) if load_previous else None
    except BaseException as exc:
        raise _unresolved_durability_error(claim.case_id, exc) from None
    result = (
        _cancelled_result(claim.case_id, source_id, started_at, previous)
        if isinstance(failure, asyncio.CancelledError)
        else _preserve_prior_result_evidence(
            _failed_result(claim.case_id, source_id, started_at, failure),
            previous,
        )
    )
    try:
        result = store.merge_relational_case_evidence(result)
    except BaseException as exc:
        _append_error(
            result,
            _secondary_error(
                exc,
                case_id=claim.case_id,
                category=ErrorCategory.DATABASE,
                stop_reason="TERMINAL_EVIDENCE_RECONCILIATION_FAILED",
                message="durable terminal evidence reconciliation failed",
            ),
        )
    execution = _ClaimedExecution(
        store=store,
        claim=claim,
        case_id=claim.case_id,
        started_at=started_at,
        source_id=source_id,
        usage=result.usage,
        clock=monotonic(),
    )
    persistence = await _persist_terminal_result_safely(
        execution,
        result,
        failure if not isinstance(failure, Exception) else None,
    )
    return await _publish_parent_terminal_result(execution, persistence)


async def _cancel_exact_finished_lifecycle(
    *,
    store: WorkflowStore,
    claim: ExecutionClaim,
    started_at: datetime,
    result: CaseResult,
    cancellation: asyncio.CancelledError,
) -> CaseResult:
    cancelled = _cancelled_result(
        claim.case_id,
        result.source_id,
        started_at,
        result,
    )
    execution = _ClaimedExecution(
        store=store,
        claim=claim,
        case_id=claim.case_id,
        started_at=started_at,
        source_id=result.source_id,
        usage=cancelled.usage,
        clock=monotonic(),
    )
    persistence = await _refresh_terminal_evidence_safely(
        execution,
        cancelled,
        persisted=True,
        persistence_error=None,
        control_exception=cancellation,
    )
    return await _publish_parent_terminal_result(execution, persistence)


async def _reconcile_lifecycle_boundary(
    *,
    case_id: str,
    started_at: datetime,
    settings: Settings,
    issued_claim: ExecutionClaim,
    outcome: _LifecycleBoundaryOutcome,
    cancellation: asyncio.CancelledError | None,
) -> CaseResult:
    """Trust only an exact post-reap database snapshot, never worker output."""

    store = WorkflowStore(settings)
    try:
        snapshot = store.load_case_execution_snapshot(case_id)
    except BaseException as exc:
        raise _unresolved_durability_error(case_id, exc) from None
    if snapshot is None:
        raise _unresolved_durability_error(case_id) from None
    exact_identity = (
        snapshot.execution_token == issued_claim.token
        and snapshot.execution_generation == issued_claim.generation
    )
    if not exact_identity:
        raise _unresolved_durability_error(case_id) from None
    if outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED":
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "isolated lifecycle worker session cleanup could not be proven",
            case_id=case_id,
            stop_reason="LIFECYCLE_WORKER_CLEANUP_FAILED",
        ) from None
    if snapshot.execution_state == "FINISHED":
        if snapshot.lease_expires_at is not None or snapshot.result is None:
            raise _unresolved_durability_error(case_id) from None
        result = snapshot.result
        if cancellation is not None:
            result = await _cancel_exact_finished_lifecycle(
                store=store,
                claim=issued_claim,
                started_at=started_at,
                result=result,
                cancellation=cancellation,
            )
            raise cancellation
        execution = _ClaimedExecution(
            store=store,
            claim=issued_claim,
            case_id=case_id,
            started_at=started_at,
            source_id=result.source_id,
            usage=result.usage,
            clock=monotonic(),
        )
        return await _publish_parent_terminal_result(
            execution,
            _PersistenceOutcome(result, True, None, None),
        )
    current_claim = _same_running_claim(issued_claim, snapshot)
    if current_claim is None:
        raise _unresolved_durability_error(case_id) from None
    recovery_result = _load_valid_recovery_result(case_id, store, current_claim)
    if recovery_result is not None:
        if cancellation is not None:
            raise cancellation
        return recovery_result
    if cancellation is not None:
        await _terminalize_lifecycle_boundary_failure(
            store=store,
            claim=current_claim,
            started_at=started_at,
            failure=cancellation,
        )
        raise cancellation
    code = outcome.error_code or "LIFECYCLE_WORKER_PROTOCOL_INVALID"
    return await _terminalize_lifecycle_boundary_failure(
        store=store,
        claim=current_claim,
        started_at=started_at,
        failure=_lifecycle_boundary_error(case_id, code),
    )


async def run_prepared_case(
    case_id: str,
    started_at: datetime,
    settings: Settings,
    *,
    claim: ExecutionClaim | None = None,
) -> CaseResult:
    """Run all provider/model/team work in one terminable fresh interpreter."""

    if claim is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "an exact execution claim is required to run a prepared case",
            case_id=case_id,
            stop_reason="EXECUTION_CLAIM_MISSING",
        ) from None
    claim = validate_execution_claim(claim, expected_case_id=case_id)
    store = WorkflowStore(settings)
    store.require_current_execution_claim(claim)
    try:
        authoritative_started_at = store.load_authoritative_case_started_at(claim)
    except BaseException as exc:
        raise _unresolved_durability_error(case_id, exc) from None
    supplied_offset = started_at.utcoffset() if type(started_at) is datetime else None
    if (
        type(started_at) is not datetime
        or started_at.tzinfo is None
        or supplied_offset != timedelta(0)
        or started_at != authoritative_started_at
    ):
        failure = InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "prepared case start does not match its authoritative row",
            case_id=case_id,
            stop_reason="PERSISTED_RESULT_INVALID",
        )
        return await _terminalize_lifecycle_boundary_failure(
            store=store,
            claim=claim,
            started_at=authoritative_started_at,
            failure=failure,
            load_previous=False,
        )
    started_at = authoritative_started_at
    try:
        settings.provider_key()
    except BaseException as failure:
        result = await _terminalize_lifecycle_boundary_failure(
            store=store,
            claim=claim,
            started_at=started_at,
            failure=failure,
        )
        if not isinstance(failure, Exception):
            raise failure
        return result
    outcome, cancellation = await _invoke_lifecycle_boundary(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
    )
    return await _reconcile_lifecycle_boundary(
        case_id=case_id,
        started_at=started_at,
        settings=settings,
        issued_claim=claim,
        outcome=outcome,
        cancellation=cancellation,
    )


def _prepare_invoice(
    path: Path,
    settings: Settings,
    *,
    retain_execution_claim: bool,
) -> tuple[str, datetime] | tuple[str, datetime, ExecutionClaim] | CaseResult:
    """Preflight and prepare one source; failures return an already-written terminal result.

    This is the shared pre-model seam for the CLI and the web console: the returned
    case ID exists before any model call, so a caller can watch the run live while
    `run_prepared_case` executes.
    """

    started_at = _now()
    try:
        preflight(settings)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # The printed artifacts/results path must exist for every terminal outcome,
        # including failures that happen before a case or model call exists.
        result = _failed_result(f"case_{uuid4().hex}", None, started_at, exc)
        _write_result(result)
        return result
    prepared = _prepare_case(
        path,
        settings,
        retain_execution_claim=retain_execution_claim,
    )
    if isinstance(prepared, _PreparationFailure):
        if not prepared.recovery_published:
            _write_result(prepared.result)
        return prepared.result
    return prepared


def prepare_invoice(path: Path, settings: Settings) -> tuple[str, datetime] | CaseResult:
    return cast(
        tuple[str, datetime] | CaseResult,
        _prepare_invoice(path, settings, retain_execution_claim=False),
    )


def prepare_claimed_invoice(
    path: Path, settings: Settings
) -> tuple[str, datetime, ExecutionClaim] | CaseResult:
    return cast(
        tuple[str, datetime, ExecutionClaim] | CaseResult,
        _prepare_invoice(path, settings, retain_execution_claim=True),
    )


@dataclass(frozen=True, slots=True)
class _PreparationBoundaryOutcome:
    acknowledged: bool
    error_code: str | None


def _preparation_boundary_error(case_id: str, code: str) -> InvoiceAgentsError:
    contracts: dict[str, tuple[ErrorCategory, str]] = {
        "PREPARATION_FAILED": (
            ErrorCategory.ORCHESTRATION,
            "isolated claimed preparation failed",
        ),
        "PREPARATION_WORKER_CRASHED": (
            ErrorCategory.ORCHESTRATION,
            "isolated claimed preparation worker exited unexpectedly",
        ),
        "PREPARATION_WORKER_PROTOCOL_INVALID": (
            ErrorCategory.ORCHESTRATION,
            "isolated claimed preparation worker returned an invalid acknowledgement",
        ),
        "PREPARATION_WORKER_TIMED_OUT": (
            ErrorCategory.TIMEOUT,
            "isolated claimed preparation exceeded its bounded deadline",
        ),
    }
    category, message = contracts.get(
        code,
        (ErrorCategory.ORCHESTRATION, "isolated claimed preparation failed"),
    )
    return InvoiceAgentsError(
        category,
        message,
        case_id=case_id,
        stop_reason=code,
    )


async def _invoke_preparation_boundary(
    *,
    path: Path,
    settings: Settings,
    case_id: str,
    started_at: datetime,
    preparation_token: str,
    run_token: str,
) -> tuple[_PreparationBoundaryOutcome, asyncio.CancelledError | None]:
    cancel_requested = threading.Event()

    def invoke() -> _PreparationBoundaryOutcome:
        try:
            from invoice_agents import preparation_process

            outcome = preparation_process.run_preparation_process(
                path=path,
                settings=settings,
                case_id=case_id,
                started_at=started_at,
                preparation_token=preparation_token,
                run_token=run_token,
                timeout_seconds=300.0,
                cancel_requested=cancel_requested,
            )
            return _PreparationBoundaryOutcome(
                acknowledged=outcome.acknowledged,
                error_code=outcome.error_code,
            )
        except BaseException:
            return _PreparationBoundaryOutcome(False, "PREPARATION_WORKER_CRASHED")

    controller = asyncio.create_task(
        asyncio.to_thread(invoke),
        name=f"invoice-preparation-controller-{case_id}",
    )
    cancellation: asyncio.CancelledError | None = None
    try:
        outcome = await asyncio.shield(controller)
    except asyncio.CancelledError as exc:
        cancellation = exc
        cancel_requested.set()
        try:
            outcome = await _await_task_despite_cancellation(
                controller,
                deadline=(
                    monotonic()
                    + DURABILITY_DEADLINE_SECONDS
                    + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                ),
                case_id=case_id,
            )
        except BaseException:
            outcome = _PreparationBoundaryOutcome(
                False,
                "PREPARATION_WORKER_CLEANUP_FAILED",
            )
    return outcome, cancellation


async def _reconcile_preparation_boundary(
    *,
    case_id: str,
    started_at: datetime,
    settings: Settings,
    preparation_token: str,
    run_token: str,
    outcome: _PreparationBoundaryOutcome,
    cancellation: asyncio.CancelledError | None,
) -> tuple[str, datetime, ExecutionClaim] | CaseResult:
    store = WorkflowStore(settings)
    try:
        snapshot = store.load_case_execution_snapshot(case_id)
    except BaseException as exc:
        raise _unresolved_durability_error(case_id, exc) from None
    if outcome.error_code == "PREPARATION_WORKER_CLEANUP_FAILED":
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "isolated preparation worker session cleanup could not be proven",
            case_id=case_id,
            stop_reason="PREPARATION_WORKER_CLEANUP_FAILED",
        ) from None
    if snapshot is None:
        if cancellation is not None:
            raise cancellation
        code = outcome.error_code or "PREPARATION_WORKER_PROTOCOL_INVALID"
        result = _failed_result(
            case_id,
            None,
            started_at,
            _preparation_boundary_error(case_id, code),
        )
        _write_result(result)
        return result
    if snapshot.started_at != started_at:
        raise _unresolved_durability_error(case_id) from None

    known_finished = (
        snapshot.execution_state == "FINISHED"
        and snapshot.lease_expires_at is None
        and snapshot.result is not None
        and (
            (snapshot.execution_token == preparation_token and snapshot.execution_generation == 1)
            or (snapshot.execution_token == run_token and snapshot.execution_generation == 2)
        )
    )
    if known_finished:
        assert snapshot.result is not None
        _write_result_for_generation(
            store,
            snapshot.result,
            snapshot.execution_generation,
        )
        if cancellation is not None:
            raise cancellation
        return snapshot.result

    if (
        snapshot.execution_state == "RUNNING"
        and snapshot.execution_token == run_token
        and snapshot.execution_generation == 2
        and snapshot.lease_expires_at is not None
    ):
        run_claim = ExecutionClaim(case_id, run_token, 2, snapshot.lease_expires_at)
        if cancellation is not None:
            await _terminalize_lifecycle_boundary_failure(
                store=store,
                claim=run_claim,
                started_at=started_at,
                failure=cancellation,
            )
            raise cancellation
        return case_id, started_at, run_claim

    preparation_claim: ExecutionClaim | None = None
    if (
        snapshot.execution_state == "RUNNING"
        and snapshot.execution_token == preparation_token
        and snapshot.execution_generation == 1
        and snapshot.lease_expires_at is not None
    ):
        preparation_claim = ExecutionClaim(
            case_id,
            preparation_token,
            1,
            snapshot.lease_expires_at,
        )
        recovery_result = _load_valid_recovery_result(
            case_id,
            store,
            preparation_claim,
        )
        if recovery_result is not None:
            if cancellation is not None:
                raise cancellation
            return recovery_result
    elif (
        snapshot.execution_state == "IDLE"
        and snapshot.execution_token is None
        and snapshot.execution_generation == 0
        and snapshot.lease_expires_at is None
    ):
        try:
            preparation_claim = store.claim_case_execution(
                case_id,
                frozenset({CaseStatus.INCOMPLETE}),
                EXECUTION_LEASE_SECONDS,
                requested_token=preparation_token,
            )
        except BaseException as exc:
            raise _unresolved_durability_error(case_id, exc) from None
    if preparation_claim is None:
        raise _unresolved_durability_error(case_id) from None
    failure: BaseException = (
        cancellation
        if cancellation is not None
        else _preparation_boundary_error(
            case_id,
            outcome.error_code or "PREPARATION_WORKER_PROTOCOL_INVALID",
        )
    )
    result = await _terminalize_lifecycle_boundary_failure(
        store=store,
        claim=preparation_claim,
        started_at=started_at,
        failure=failure,
    )
    if cancellation is not None:
        raise cancellation
    return result


async def prepare_claimed_invoice_async(
    path: Path,
    settings: Settings,
) -> tuple[str, datetime, ExecutionClaim] | CaseResult:
    """Prepare and hand off one claim through a fully terminable child session."""

    started_at = _now()
    try:
        preflight(settings)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        result = _failed_result(f"case_{uuid4().hex}", None, started_at, exc)
        _write_result(result)
        return result
    case_id = f"case_{uuid4().hex}"
    preparation_token = f"exec_{uuid4().hex}"
    run_token = f"exec_{uuid4().hex}"
    outcome, cancellation = await _invoke_preparation_boundary(
        path=path,
        settings=settings,
        case_id=case_id,
        started_at=started_at,
        preparation_token=preparation_token,
        run_token=run_token,
    )
    return await _reconcile_preparation_boundary(
        case_id=case_id,
        started_at=started_at,
        settings=settings,
        preparation_token=preparation_token,
        run_token=run_token,
        outcome=outcome,
        cancellation=cancellation,
    )


async def process_invoice(path: Path, settings: Settings) -> CaseResult:
    """Preflight, prepare, and process one source artifact."""

    prepared = await prepare_claimed_invoice_async(path, settings)
    if isinstance(prepared, CaseResult):
        return prepared
    return await run_prepared_case(
        prepared[0],
        prepared[1],
        settings,
        claim=prepared[2],
    )


async def _await_task_despite_cancellation[ResultT](
    task: asyncio.Task[ResultT],
    *,
    deadline: float,
    case_id: str | None = None,
) -> ResultT:
    """Drain through repeated cancellation, but never beyond one monotonic deadline."""

    loop = asyncio.get_running_loop()
    initial_remaining = max(0.0, deadline - monotonic())
    loop_deadline = loop.time() + initial_remaining
    while not task.done():
        remaining = min(deadline - monotonic(), loop_deadline - loop.time())
        if remaining <= 0:
            task.add_done_callback(_consume_task_result)
            raise InvoiceAgentsError(
                ErrorCategory.TIMEOUT,
                "terminal durability work exceeded its monotonic deadline",
                case_id=case_id,
                stop_reason="TERMINAL_DURABILITY_TIMEOUT",
            ) from None
        try:
            done, _pending = await asyncio.wait(
                {task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            continue
        if not done:
            task.add_done_callback(_consume_task_result)
            raise InvoiceAgentsError(
                ErrorCategory.TIMEOUT,
                "terminal durability work exceeded its monotonic deadline",
                case_id=case_id,
                stop_reason="TERMINAL_DURABILITY_TIMEOUT",
            ) from None
    return task.result()


def _load_valid_recovery_result(
    case_id: str,
    store: WorkflowStore,
    claim: ExecutionClaim,
) -> CaseResult | None:
    """Return one bounded strict recovery result bound to current exact authority."""

    try:
        claim = validate_execution_claim(claim, expected_case_id=case_id)
    except InvoiceAgentsError:
        return None
    target = Path("artifacts/results").resolve() / f"{case_id}.recovery.json"
    try:
        raw = target.read_bytes()
        if not raw or len(raw) > RECOVERY_ARTIFACT_MAX_BYTES:
            return None
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        envelope = _RecoveryEnvelope.model_validate_json(raw, strict=True)
        if type(parsed) is not dict or raw != _canonical_recovery_bytes(envelope):
            return None
        expires_at = parse_canonical_utc(envelope.lease_expires_at)
        if expires_at is None or expires_at.tzinfo is not UTC:
            return None
        evidence = _inspect_exact_claim_evidence(store, claim)
        source_id = store.load_recovery_case_source_id(claim)
        merged = store.merge_relational_case_evidence(envelope.case_result)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, InvoiceAgentsError, ValueError):
        return None
    valid = (
        envelope.recovery_format == RECOVERY_ARTIFACT_FORMAT
        and envelope.case_id == case_id
        and envelope.execution_token == claim.token
        and envelope.execution_generation == claim.generation
        and expires_at == claim.expires_at
        and envelope.lease_expires_at == claim.expires_at.isoformat()
        and envelope.case_result.case_id == case_id
        and envelope.case_result.source_id == source_id
        and merged == envelope.case_result
        and _recovery_semantics_are_exact(envelope)
        and evidence.state is _ExactClaimEvidenceState.RECOVERABLE_RUNNING
    )
    return envelope.case_result if valid else None


def _recovery_artifact_is_valid(
    case_id: str,
    store: WorkflowStore,
    claim: ExecutionClaim,
) -> bool:
    """Validate one bounded strict envelope against current DB and exact claim authority."""

    return _load_valid_recovery_result(case_id, store, claim) is not None


def _claim_has_durable_terminal_evidence(
    case_id: str,
    settings: Settings,
    claim: ExecutionClaim,
) -> bool:
    store = WorkflowStore(settings)
    claim = validate_execution_claim(claim, expected_case_id=case_id)
    evidence = _inspect_exact_claim_evidence(store, claim)
    if evidence.state is _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT:
        return True
    return _recovery_artifact_is_valid(case_id, store, claim)


def _unresolved_durability_error(
    case_id: str, exc: BaseException | None = None
) -> InvoiceAgentsError:
    return InvoiceAgentsError(
        ErrorCategory.ORCHESTRATION,
        "no trustworthy terminal database result or recovery artifact exists",
        case_id=case_id,
        stop_reason="TERMINAL_DURABILITY_UNRESOLVED",
        details={"inspection_exception_type": type(exc).__name__ if exc is not None else None},
    )


async def _inspect_claim_durability(
    claims: list[tuple[str, datetime, ExecutionClaim]],
    settings: Settings,
) -> dict[str, BaseException | None]:
    """Bound every read-only durability inspection and return one result per claim."""

    async def inspect(
        case_id: str,
        claim: ExecutionClaim,
    ) -> tuple[str, BaseException | None]:
        task = asyncio.create_task(
            asyncio.to_thread(
                _claim_has_durable_terminal_evidence,
                case_id,
                settings,
                claim,
            ),
            name=f"invoice-durability-inspect-{case_id}",
        )
        try:
            durable = await _await_task_despite_cancellation(
                task,
                deadline=monotonic() + DURABILITY_DEADLINE_SECONDS,
                case_id=case_id,
            )
        except InvoiceAgentsError as exc:
            return case_id, exc
        except BaseException as exc:
            return case_id, _unresolved_durability_error(case_id, exc)
        return case_id, None if durable else _unresolved_durability_error(case_id)

    inspection_tasks = [
        asyncio.create_task(inspect(case_id, claim), name=f"invoice-durability-{case_id}")
        for case_id, _started, claim in claims
    ]

    async def drain() -> list[tuple[str, BaseException | None]]:
        return await asyncio.gather(*inspection_tasks)

    drain_task = asyncio.create_task(drain(), name="invoice-durability-inspection-drain")
    inspected = await _await_task_despite_cancellation(
        drain_task,
        deadline=monotonic() + (2 * DURABILITY_DEADLINE_SECONDS),
    )
    return dict(inspected)


_DURABILITY_PRECEDENCE_STOPS = frozenset(
    {
        "TERMINAL_RECOVERY_ARTIFACT_FAILED",
        "TERMINAL_DURABILITY_TIMEOUT",
        "TERMINAL_DURABILITY_UNRESOLVED",
    }
)


def _select_drained_failure(
    primary: BaseException,
    outcomes: list[object],
    durability: dict[str, BaseException | None],
) -> BaseException:
    """Choose durability loss before ordinary/process cancellation, independent of order."""

    failures = [item for item in outcomes if isinstance(item, BaseException)]
    for failure in failures:
        if (
            isinstance(failure, InvoiceAgentsError)
            and failure.stop_reason in _DURABILITY_PRECEDENCE_STOPS
        ):
            return failure
    for durability_failure in durability.values():
        if durability_failure is not None:
            return durability_failure
    for failure in failures:
        if not isinstance(failure, (Exception, asyncio.CancelledError)):
            return failure
    if isinstance(primary, asyncio.CancelledError):
        for failure in failures:
            if not isinstance(failure, asyncio.CancelledError):
                return failure
    return primary


async def _terminalize_unstarted_claim(
    case_id: str,
    started_at: datetime,
    settings: Settings,
    claim: ExecutionClaim,
) -> None:
    """Persist cancellation or recovery entirely inside owned helper sessions."""

    claim = validate_execution_claim(claim, expected_case_id=case_id)
    durability_deadline = (
        monotonic() + DURABILITY_DEADLINE_SECONDS + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
    )
    outcome, boundary_control = await _run_terminal_mode_owned(
        mode="cancel_unstarted",
        settings=settings,
        claim=claim,
        deadline=durability_deadline,
        started_at=started_at,
    )
    evidence = _terminal_outcome_evidence(outcome)
    if evidence is None:
        if monotonic() >= durability_deadline:
            raise _terminal_durability_timeout(case_id) from None
        inspection, inspection_control = await _run_terminal_mode_owned(
            mode="inspect_claim",
            settings=settings,
            claim=claim,
            deadline=durability_deadline,
            prior_cancellation=(
                boundary_control if isinstance(boundary_control, asyncio.CancelledError) else None
            ),
        )
        boundary_control = _select_terminal_control_exception(
            boundary_control,
            inspection_control,
        )
        evidence = _terminal_outcome_evidence(inspection)
        if evidence is None:
            if outcome.error_code == "TERMINAL_WORKER_TIMEOUT" or (
                inspection.error_code == "TERMINAL_WORKER_TIMEOUT"
            ):
                raise _terminal_durability_timeout(case_id) from None
            raise _unresolved_durability_error(case_id) from None
    if evidence.state is _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT:
        if boundary_control is not None:
            raise boundary_control
        return
    if outcome.result is not None or outcome.error_code is None:
        raise _unresolved_durability_error(case_id) from None
    if monotonic() >= durability_deadline:
        raise _terminal_durability_timeout(case_id) from None
    recovery, recovery_control = await _run_terminal_mode_owned(
        mode="publish_cancel_recovery",
        settings=settings,
        claim=claim,
        deadline=durability_deadline,
        started_at=started_at,
        worker_error_code=outcome.error_code,
        prior_cancellation=(
            boundary_control if isinstance(boundary_control, asyncio.CancelledError) else None
        ),
    )
    boundary_control = _select_terminal_control_exception(
        boundary_control,
        recovery_control,
    )
    recovery_evidence = _terminal_outcome_evidence(recovery)
    recovery_is_durable = (
        recovery_evidence is not None
        and recovery_evidence.state is _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT
    )
    recovery_was_published = (
        recovery.error_code is None
        and type(recovery.result) is CaseResult
        and recovery_evidence is not None
        and recovery_evidence.state is _ExactClaimEvidenceState.RECOVERABLE_RUNNING
    )
    if not recovery_is_durable and not recovery_was_published:
        if recovery.error_code == "TERMINAL_WORKER_TIMEOUT":
            raise _terminal_durability_timeout(case_id) from None
        if recovery.error_code == "TERMINAL_RECOVERY_ARTIFACT_FAILED":
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "atomic terminal recovery artifact publication failed",
                case_id=case_id,
                stop_reason="TERMINAL_RECOVERY_ARTIFACT_FAILED",
            ) from None
        raise _unresolved_durability_error(case_id) from None
    if outcome.error_code == "TERMINAL_WORKER_TIMEOUT":
        raise _terminal_durability_timeout(case_id) from None
    if boundary_control is not None:
        raise boundary_control


async def _durably_cancel_unstarted_claim(
    case_id: str,
    started_at: datetime,
    settings: Settings,
    claim: ExecutionClaim,
) -> None:
    await _terminalize_unstarted_claim(case_id, started_at, settings, claim)


async def _durably_cancel_unstarted_claims(
    claims: list[tuple[str, datetime, ExecutionClaim]],
    settings: Settings,
) -> list[BaseException | None]:
    """Attempt terminal cancellation for every handed-off claim before returning."""

    tasks = [
        asyncio.create_task(
            _terminalize_unstarted_claim(case_id, started_at, settings, claim),
            name=f"invoice-cancel-unstarted-{case_id}",
        )
        for case_id, started_at, claim in claims
    ]

    async def drain() -> list[BaseException | None]:
        return await asyncio.gather(*tasks, return_exceptions=True)

    drain_task = asyncio.create_task(drain(), name="invoice-unstarted-claims-drain")
    return await _await_task_despite_cancellation(
        drain_task,
        deadline=(
            monotonic() + (3 * DURABILITY_DEADLINE_SECONDS) + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
        ),
    )


async def process_batch(
    paths: list[Path], settings: Settings, concurrency: int | None = None
) -> list[CaseResult]:
    """Prepare all identities, then run independent fresh teams with bounded concurrency."""

    selected_concurrency = validate_case_concurrency(concurrency, settings.case_concurrency)
    started_at = _now()
    try:
        preflight(settings)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failed = [_failed_result(f"case_{uuid4().hex}", None, started_at, exc) for _ in paths]
        for result in failed:
            _write_result(result)
        return failed
    prepared: list[tuple[str, datetime, ExecutionClaim]] = []
    results: list[CaseResult] = []
    selected_failure: BaseException | None = None
    try:
        for path in paths:
            item = await prepare_claimed_invoice_async(path, settings)
            if isinstance(item, CaseResult):
                results.append(item)
            else:
                prepared.append(item)
    except BaseException as primary_failure:
        preparation_outcomes = await _durably_cancel_unstarted_claims(prepared, settings)
        durability = await _inspect_claim_durability(prepared, settings)
        selected_failure = _select_drained_failure(
            primary_failure,
            list(preparation_outcomes),
            durability,
        )
    if selected_failure is not None:
        selected_failure.__cause__ = None
        selected_failure.__context__ = None
        raise selected_failure from None
    semaphore = asyncio.Semaphore(selected_concurrency)

    async def bounded(item: tuple[str, datetime, ExecutionClaim]) -> CaseResult:
        try:
            async with semaphore:
                return await run_prepared_case(item[0], item[1], settings, claim=item[2])
        except asyncio.CancelledError:

            async def ensure_durability() -> None:
                durability = await _inspect_claim_durability([item], settings)
                if durability[item[0]] is not None:
                    await _durably_cancel_unstarted_claim(item[0], item[1], settings, item[2])

            durability_task = asyncio.create_task(
                ensure_durability(), name=f"invoice-batch-durability-{item[0]}"
            )
            await _await_task_despite_cancellation(
                durability_task,
                deadline=(
                    monotonic()
                    + (2 * DURABILITY_DEADLINE_SECONDS)
                    + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                ),
                case_id=item[0],
            )
            raise

    tasks = [
        asyncio.create_task(bounded(item), name=f"invoice-batch-{item[0]}") for item in prepared
    ]
    selected_failure = None
    try:
        results.extend(await asyncio.gather(*tasks))
    except BaseException as primary_failure:
        for task in tasks:
            if not task.done():
                task.cancel()

        async def drain() -> list[CaseResult | BaseException]:
            return await asyncio.gather(*tasks, return_exceptions=True)

        drain_task = asyncio.create_task(drain(), name="invoice-batch-cancellation-drain")
        task_outcomes = await _await_task_despite_cancellation(
            drain_task,
            deadline=(
                monotonic()
                + (3 * DURABILITY_DEADLINE_SECONDS)
                + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
            ),
        )
        durability = await _inspect_claim_durability(prepared, settings)
        unowned_cancelled_claims = [
            item
            for item, outcome in zip(prepared, task_outcomes, strict=True)
            if isinstance(outcome, asyncio.CancelledError) and durability[item[0]] is not None
        ]
        direct_outcomes = await _durably_cancel_unstarted_claims(unowned_cancelled_claims, settings)
        if unowned_cancelled_claims:
            durability = await _inspect_claim_durability(prepared, settings)
        selected_failure = _select_drained_failure(
            primary_failure,
            [*task_outcomes, *direct_outcomes],
            durability,
        )
    if selected_failure is not None:
        selected_failure.__cause__ = None
        selected_failure.__context__ = None
        raise selected_failure from None
    return results


def _recompute_after_mapping(
    case_id: str,
    human: HumanDecision,
    settings: Settings,
    store: WorkflowStore,
    audit: AuditRecorder,
    claim: ExecutionClaim,
) -> None:
    """Re-derive inventory, mapping, financial, and risk evidence deterministically.

    Runs after an ESTABLISH_MAPPING decision and before the team resumes, with no
    model calls: the newly approved alias must flow through the same comparison and
    policy code as the original pass, and the audit trail must show the recompute
    between the human decision and any final decision (remediation G6).
    """

    invoice = store.load_current_extraction(claim)
    reader = InventoryReader(settings.inventory_db)
    mappings, comparisons, unresolved = compare_inventory_evidence(invoice, reader)
    errors = [
        result.error
        for result in unresolved.values()
        if result.status is ToolStatus.ERROR and result.error
    ]
    if errors:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "; ".join(errors),
            case_id=case_id,
            stop_reason="INVENTORY_QUERY_FAILED",
        )
    inventory_payload = {
        "comparisons": [comparison.model_dump(mode="json") for comparison in comparisons],
        "unresolved_candidates": {
            item: result.model_dump(mode="json") for item, result in unresolved.items()
        },
    }
    comparison_id = store.save_comparison(case_id, "inventory", inventory_payload, claim)
    enriched = apply_mapping_evidence(invoice, mappings, unresolved)
    extraction_id = store.save_extraction(case_id, enriched, claim)
    financial = compute_invoice_totals(enriched)
    identity = [
        IdentityCandidate.model_validate(item) for item in store.load_current_identity(claim)
    ]
    risk = build_risk_assessment(enriched, comparisons, identity, financial, settings)
    risk_id = store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    audit.record(
        "recompute.after_human_mapping",
        {
            "review_id": human.review_id,
            "human_decision": human.decision,
            "inventory_comparison_id": comparison_id,
            "extraction_id": extraction_id,
            "risk_comparison_id": risk_id,
            "policy_review_reasons": risk.policy_review_reasons,
        },
        source_id=enriched.source.source_id,
        review_id=human.review_id,
        db_evidence_id=risk_id,
    )


def claim_resumable_case(case_id: str, settings: Settings) -> ExecutionClaim:
    """Acquire the exact resume authority that callers must pass unchanged."""

    try:
        return WorkflowStore(settings).claim_case_execution(
            case_id,
            frozenset({CaseStatus.NEEDS_HUMAN}),
            EXECUTION_LEASE_SECONDS,
        )
    except InvoiceAgentsError as exc:
        if exc.stop_reason == "CASE_STATUS_NOT_CLAIMABLE":
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                f"case {case_id} is not waiting for human review",
                case_id=case_id,
                stop_reason="CASE_NOT_RESUMABLE",
            ) from None
        raise


async def _resume_case_in_process(
    case_id: str,
    settings: Settings,
    *,
    claim: ExecutionClaim | None = None,
    terminal_writes_in_process: bool = False,
) -> CaseResult:
    """Child-side resume lifecycle; public callers use the process boundary."""

    if claim is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "an exact execution claim is required to resume a case",
            case_id=case_id,
            stop_reason="EXECUTION_CLAIM_MISSING",
        ) from None
    claim = validate_execution_claim(claim, expected_case_id=case_id)
    store = WorkflowStore(settings)
    store.require_current_execution_claim(claim)

    async def resume_lifecycle(execution: _ClaimedExecution) -> CaseResult:
        preflight(settings)
        previous = store.load_result(case_id)
        if previous is None:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                f"case {case_id} has no persisted result to resume",
                case_id=case_id,
                stop_reason="CASE_RESULT_MISSING",
            )
        execution.started_at = previous.started_at
        execution.usage = previous.usage.model_copy(deep=True)
        if previous.status is not CaseStatus.NEEDS_HUMAN:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                f"case {case_id} is not waiting for human review",
                case_id=case_id,
                stop_reason="CASE_NOT_RESUMABLE",
            )
        review = store.load_case_review(case_id)
        if review is None or review.status != "RESOLVED" or review.human_decision is None:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "a persisted human decision is required before resume",
                case_id=case_id,
                stop_reason="HUMAN_DECISION_MISSING",
            )
        state = store.load_team_state(case_id)
        if state is None:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "stopped AutoGen team state is missing",
                case_id=case_id,
                stop_reason="TEAM_STATE_MISSING",
            )
        store.adopt_latest_evidence(execution.claim)
        invoice = store.load_current_extraction(execution.claim)
        execution.source_id = invoice.source.source_id
        audit = AuditRecorder(settings.workflow_db, case_id)
        execution.audit = audit
        human = review.human_decision
        if human.decision is HumanDecisionKind.ESTABLISH_MAPPING:
            _recompute_after_mapping(case_id, human, settings, store, audit, execution.claim)
            invoice = store.load_current_extraction(execution.claim)
            execution.source_id = invoice.source.source_id
        context = AgentCaseContext(case_id, settings, store, audit, execution.claim)
        execution.context = context
        client = create_model_client(settings)
        execution.client = client
        team = build_team(context, client)
        usage = execution.usage

        async def execute_resume_team() -> CaseResult:
            await team.load_state(state)
            message = HandoffMessage(
                source="human_reviewer",
                target="approval_agent",
                content=json.dumps(
                    {
                        "review_id": human.review_id,
                        "reviewer": human.reviewer,
                        "decision": human.decision,
                        "reason": human.reason,
                        "mappings": [mapping.model_dump(mode="json") for mapping in human.mappings],
                        "superseded_case_id": human.superseded_case_id,
                    },
                    default=str,
                ),
            )
            with bind_audit_recorder(audit):
                task_result = await _stream_team(team, context, task=message, usage=usage)
            new_state = await team.save_state()
            store.save_team_state(case_id, dict(new_state), execution.claim)
            usage.latency_ms += int((monotonic() - execution.clock) * 1000)
            return _result_from_stop(context, task_result, previous.started_at, usage)

        def replace_claim(renewed: ExecutionClaim) -> None:
            renewed = validate_execution_claim(renewed, expected_case_id=case_id)
            execution.claim = renewed
            if execution.context is not None:
                execution.context.claim = renewed

        return await _run_with_lease_heartbeat(
            execute_resume_team(),
            renew=store.renew_case_execution,
            claim=execution.claim,
            replace_claim=replace_claim,
            lease_seconds=EXECUTION_LEASE_SECONDS,
            renewal_interval_seconds=EXECUTION_RENEWAL_INTERVAL_SECONDS,
        )

    return await _execute_claimed_case(
        case_id,
        _now(),
        store,
        claim,
        resume_lifecycle,
        finished_event_type="case.resumed_finished",
        terminal_writes_in_process=terminal_writes_in_process,
    )


async def resume_case(
    case_id: str,
    settings: Settings,
    *,
    claim: ExecutionClaim | None = None,
) -> CaseResult:
    """Resume provider/model/team work in one terminable fresh interpreter."""

    if claim is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "an exact execution claim is required to resume a case",
            case_id=case_id,
            stop_reason="EXECUTION_CLAIM_MISSING",
        ) from None
    claim = validate_execution_claim(claim, expected_case_id=case_id)
    store = WorkflowStore(settings)
    store.require_current_execution_claim(claim)
    try:
        started_at = store.load_authoritative_case_started_at(claim)
    except BaseException as exc:
        raise _unresolved_durability_error(case_id, exc) from None
    try:
        previous = store.load_result(case_id)
        if previous is not None and previous.started_at != started_at:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case result start does not match its authoritative row",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
    except BaseException as failure:
        result = await _terminalize_lifecycle_boundary_failure(
            store=store,
            claim=claim,
            started_at=started_at,
            failure=failure,
            load_previous=False,
        )
        if not isinstance(failure, Exception):
            raise failure
        return result
    try:
        settings.provider_key()
    except BaseException as failure:
        result = await _terminalize_lifecycle_boundary_failure(
            store=store,
            claim=claim,
            started_at=started_at,
            failure=failure,
        )
        if not isinstance(failure, Exception):
            raise failure
        return result
    outcome, cancellation = await _invoke_lifecycle_boundary(
        mode="resume",
        settings=settings,
        claim=claim,
        started_at=started_at,
    )
    return await _reconcile_lifecycle_boundary(
        case_id=case_id,
        started_at=started_at,
        settings=settings,
        issued_claim=claim,
        outcome=outcome,
        cancellation=cancellation,
    )
