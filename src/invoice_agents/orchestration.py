"""Case lifecycle, streamed AutoGen execution, status mapping, persistence, and resume."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, cast
from uuid import uuid4

import openai
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    HandoffMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from pydantic import ValidationError

from invoice_agents.agents.team import AgentCaseContext, build_team, create_model_client
from invoice_agents.config import XAI_BASE_URL, XAI_MODEL, Settings
from invoice_agents.db.core import DatabaseKind, verify_database
from invoice_agents.db.store import (
    ExecutionClaim,
    WorkflowStore,
    execution_claim_expiry_iso,
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
from invoice_agents.observability.audit import AuditRecorder, bind_audit_recorder
from invoice_agents.source_store import snapshot_source
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


def is_max_messages_stop(stop_reason: str) -> bool:
    """True only for AutoGen's max-message termination phrasing."""

    return MAX_MESSAGES_STOP_PHRASE in stop_reason.lower()


def _now() -> datetime:
    return datetime.now(UTC)


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
                    if (
                        renewed.case_id != current_claim.case_id
                        or renewed.token != current_claim.token
                        or renewed.generation != current_claim.generation
                        or execution_claim_expiry_iso(renewed) is None
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


_EXTERNAL_SECRET = re.compile(
    r"(?i)(?:bearer\s+[A-Za-z0-9._~+/=-]+|(?:sk(?:-proj)?|xai)-[A-Za-z0-9._~+/=-]+)"
)
_SENSITIVE_DETAIL_KEY = re.compile(r"(?i)(?:authorization|api[_-]?key|secret|token|cookie)")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _sanitize_error_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _SENSITIVE_DETAIL_KEY.search(str(key))
                else _sanitize_error_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_error_value(item) for item in value]
    if isinstance(value, str):
        return _EXTERNAL_SECRET.sub("[REDACTED]", value)
    return value


def _safe_provider_request_id(exc: BaseException) -> str | None:
    value = getattr(exc, "request_id", None)
    return value if isinstance(value, str) and _SAFE_REQUEST_ID.fullmatch(value) else None


def _error_record(exc: BaseException, case_id: str | None = None) -> ErrorRecord:
    if isinstance(exc, InvoiceAgentsError):
        return ErrorRecord(
            category=exc.category,
            message=str(_sanitize_error_value(exc.message)),
            case_id=exc.case_id or case_id,
            stop_reason=exc.stop_reason,
            provider_request_id=(
                exc.provider_request_id
                if isinstance(exc.provider_request_id, str)
                and _SAFE_REQUEST_ID.fullmatch(exc.provider_request_id)
                else None
            ),
            details=cast(dict[str, Any], _sanitize_error_value(exc.details or {})),
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


def _write_recovery_artifact(result: CaseResult, terminal_persistence_error: ErrorRecord) -> Path:
    output_dir = Path("artifacts/results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{result.case_id}.recovery.json"
    payload = json.dumps(
        {
            "recovery_format": 1,
            "case_result": result.model_dump(mode="json"),
            "terminal_persistence_error": terminal_persistence_error.model_dump(mode="json"),
        },
        default=str,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_publish(target, payload)
    return target


def _recovery_artifact_or_raise(
    result: CaseResult, terminal_persistence_error: ErrorRecord
) -> None:
    """Publish recovery evidence once or raise one stable, chainless failure."""

    artifact_exc: BaseException | None = None
    try:
        _write_recovery_artifact(result, terminal_persistence_error)
    except BaseException as exc:
        artifact_exc = exc
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

    close_task = asyncio.create_task(capture_close())
    try:
        close_failure = await asyncio.wait_for(
            asyncio.shield(close_task),
            timeout=CLIENT_CLOSE_TIMEOUT_SECONDS,
        )
        if close_failure is not None:
            raise close_failure
    except asyncio.CancelledError as exc:
        close_task.cancel()
        close_task.add_done_callback(_consume_task_result)
        return _CleanupOutcome(
            error=ErrorRecord(
                category=ErrorCategory.CANCELLED,
                message="client cleanup was cancelled",
                case_id=execution.case_id,
                stop_reason="CLIENT_CLOSE_CANCELLED",
            ),
            control_exception=exc,
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
            )
        )
    except BaseException as exc:
        return _CleanupOutcome(
            error=_secondary_error(
                exc,
                case_id=execution.case_id,
                category=ErrorCategory.ORCHESTRATION,
                stop_reason="CLIENT_CLOSE_FAILED",
                message="client cleanup failed",
            ),
            control_exception=exc if not isinstance(exc, Exception) else None,
        )
    return _CleanupOutcome()


@dataclass(slots=True)
class _PersistenceOutcome:
    result: CaseResult
    persisted: bool
    persistence_error: ErrorRecord | None
    control_exception: BaseException | None


def _persist_terminal_result(
    execution: _ClaimedExecution,
    result: CaseResult,
    control_exception: BaseException | None,
) -> _PersistenceOutcome:
    """Persist once; a cancellation at this boundary gets one CANCELLED retry."""

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
) -> CaseResult:
    """Run setup through terminal persistence under one claimed outer boundary."""

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
        execution.source_id = store.load_case_source_id(case_id)
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
        persistence = _persist_terminal_result(execution, result, control_exception)
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
        refresh = _refresh_terminal_evidence(
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
        persistence = _persist_terminal_result(execution, result, control_exception)
        terminal_write_attempted = True
        result = persistence.result
        persisted = persistence.persisted
        persistence_error = persistence.persistence_error
        control_exception = persistence.control_exception

    if execution.audit is not None:
        try:
            execution.audit.record(
                finished_event_type,
                result.model_dump(mode="json"),
                source_id=execution.source_id,
            )
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
                    message="final audit write was cancelled",
                    case_id=case_id,
                    stop_reason="FINAL_AUDIT_WRITE_CANCELLED",
                ),
            )
            refresh = _refresh_terminal_evidence(
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
            _append_error(
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
            refresh = _refresh_terminal_evidence(
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

    if persisted:
        try:
            _write_result(result)
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
            refresh = _refresh_terminal_evidence(
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
            refresh = _refresh_terminal_evidence(
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
        _recovery_artifact_or_raise(result, persistence_error)

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
        _recovery_artifact_or_raise(result, persistence.persistence_error)
    if persistence.control_exception is not None:
        raise persistence.control_exception
    return _PreparationFailure(result=result, recovery_published=not persistence.persisted)


def _prepare_case(
    path: Path,
    settings: Settings,
    *,
    retain_execution_claim: bool,
) -> tuple[str, datetime] | tuple[str, datetime, ExecutionClaim] | _PreparationFailure:
    """Create immutable source/case evidence before any model request.

    Batch preparation is sequential so every later identity agent can see all submitted
    representations and revisions even though independent Swarms run concurrently.
    """

    started_at = _now()
    case_id = f"case_{uuid4().hex}"
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
            run_claim = store.handoff_case_execution(claim, EXECUTION_LEASE_SECONDS)
            return case_id, started_at, run_claim
        store.release_case_execution(claim)
        return case_id, started_at
    except BaseException as exc:
        failure = exc
    assert failure is not None
    if case_created and store is not None and claim is not None:
        return _terminalize_preparation_failure(
            store=store,
            claim=claim,
            case_id=case_id,
            source_id=source_id,
            started_at=started_at,
            failure=failure,
        )
    result = _failed_result(case_id, source_id, started_at, failure)
    if case_created:
        persistence_error = _secondary_error(
            failure,
            case_id=case_id,
            category=ErrorCategory.DATABASE,
            stop_reason="TERMINAL_PERSISTENCE_FAILED",
            message="preparation failed before execution authority was acquired",
        )
        _append_error(result, persistence_error)
        _recovery_artifact_or_raise(result, persistence_error)
        if not isinstance(failure, Exception):
            raise failure
        return _PreparationFailure(result=result, recovery_published=True)
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


def _event_payload(event: Any) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        return cast(dict[str, Any], event.model_dump(mode="json"))
    return {"type": type(event).__name__, "value": str(event)}


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
        model_usage = getattr(event, "models_usage", None)
        if model_usage is not None:
            usage.prompt_tokens += int(model_usage.prompt_tokens)
            usage.completion_tokens += int(model_usage.completion_tokens)
            usage.model_calls += 1
        if isinstance(event, ToolCallRequestEvent):
            usage.tool_calls += len(event.content)
        if isinstance(event, ToolCallExecutionEvent):
            for execution in event.content:
                if execution.is_error:
                    context.tool_failures.append(
                        f"{execution.name}({execution.call_id}): {execution.content}"
                    )
        context.audit.record(
            f"autogen.{type(event).__name__}",
            _event_payload(event),
            source_id=context.invoice().source.source_id,
            agent_name=getattr(event, "source", None),
            provider_request_id=(getattr(event, "metadata", {}) or {}).get("request_id"),
        )
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
            message=failure,
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
                    message=payment.error or f"payment status is {payment.status}",
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


def _write_result(result: CaseResult) -> Path:
    output_dir = Path("artifacts/results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{result.case_id}.json"
    _atomic_publish(target, result.model_dump_json(indent=2).encode("utf-8"))
    return target


def _atomic_publish(target: Path, payload: bytes) -> None:
    """Publish complete bytes with file and directory durability before return."""

    temporary = target.with_name(f"{target.name}.tmp")
    file_descriptor: int | None = None
    directory_descriptor: int | None = None
    created_temporary = False
    try:
        file_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created_temporary = True
        offset = 0
        while offset < len(payload):
            written = os.write(file_descriptor, payload[offset:])
            if written <= 0:
                raise OSError("atomic artifact write made no forward progress")
            offset += written
        os.fsync(file_descriptor)
        os.close(file_descriptor)
        file_descriptor = None
        os.replace(temporary, target)
        created_temporary = False
        directory_descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(directory_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if created_temporary:
            temporary.unlink(missing_ok=True)


async def run_prepared_case(
    case_id: str,
    started_at: datetime,
    settings: Settings,
    *,
    claim: ExecutionClaim | None = None,
) -> CaseResult:
    """Run one fresh Swarm and convert every terminal path to an explicit case status."""

    if claim is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "an exact execution claim is required to run a prepared case",
            case_id=case_id,
            stop_reason="EXECUTION_CLAIM_MISSING",
        ) from None
    if claim.case_id != case_id:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "supplied execution claim belongs to a different case",
            case_id=case_id,
            stop_reason="STALE_EXECUTION_CLAIM",
        ) from None
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


async def process_invoice(path: Path, settings: Settings) -> CaseResult:
    """Preflight, prepare, and process one source artifact."""

    prepared = prepare_claimed_invoice(path, settings)
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


def _recovery_artifact_is_valid(case_id: str) -> bool:
    target = Path("artifacts/results").resolve() / f"{case_id}.recovery.json"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        result = CaseResult.model_validate_json(json.dumps(payload["case_result"]), strict=True)
        persistence_error = ErrorRecord.model_validate_json(
            json.dumps(payload["terminal_persistence_error"]), strict=True
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return (
        payload.get("recovery_format") == 1
        and result.case_id == case_id
        and persistence_error.case_id == case_id
        and persistence_error.stop_reason
        in {"TERMINAL_PERSISTENCE_FAILED", "TERMINAL_DURABILITY_TIMEOUT"}
    )


def _claim_has_durable_terminal_evidence(case_id: str, settings: Settings) -> bool:
    store = WorkflowStore(settings)
    snapshot = store.load_case_execution_snapshot(case_id)
    if (
        snapshot is not None
        and snapshot.execution_state == "FINISHED"
        and snapshot.result is not None
    ):
        store.merge_relational_case_evidence(snapshot.result)
        return True
    return _recovery_artifact_is_valid(case_id)


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

    async def inspect(case_id: str) -> tuple[str, BaseException | None]:
        task = asyncio.create_task(
            asyncio.to_thread(_claim_has_durable_terminal_evidence, case_id, settings),
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
        asyncio.create_task(inspect(case_id), name=f"invoice-durability-{case_id}")
        for case_id, _started, _claim in claims
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
    """Persist cancellation in a bounded worker with one recovery publisher."""

    fallback = _cancelled_result(case_id, None, started_at)

    @dataclass(frozen=True, slots=True)
    class PersistenceOutcome:
        result: CaseResult
        failure: BaseException | None = None
        control_exception: BaseException | None = None

    def persist() -> PersistenceOutcome:
        store = WorkflowStore(settings)
        result = fallback
        control_exception: BaseException | None = None
        try:
            store.require_current_execution_claim(claim)
            source_id = store.load_case_source_id(case_id)
            previous = store.load_result(case_id)
            result = _cancelled_result(case_id, source_id, started_at, previous)
            try:
                result = store.merge_relational_case_evidence(result)
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
                if not isinstance(exc, Exception):
                    control_exception = exc
            store.finish_case(result, claim)
        except BaseException as exc:
            return PersistenceOutcome(
                result=result,
                failure=exc,
                control_exception=control_exception,
            )
        return PersistenceOutcome(result=result, control_exception=control_exception)

    async def publish_recovery(result: CaseResult, error: ErrorRecord) -> None:
        publication = asyncio.create_task(
            asyncio.to_thread(_recovery_artifact_or_raise, result, error),
            name=f"invoice-cancel-recovery-{case_id}",
        )
        await _await_task_despite_cancellation(
            publication,
            deadline=monotonic() + DURABILITY_DEADLINE_SECONDS,
            case_id=case_id,
        )

    worker = asyncio.create_task(
        asyncio.to_thread(persist),
        name=f"invoice-cancel-persistence-{case_id}",
    )
    try:
        outcome = await _await_task_despite_cancellation(
            worker,
            deadline=monotonic() + DURABILITY_DEADLINE_SECONDS,
            case_id=case_id,
        )
    except InvoiceAgentsError as exc:
        if exc.stop_reason != "TERMINAL_DURABILITY_TIMEOUT":
            raise
        timeout_error = ErrorRecord(
            category=ErrorCategory.TIMEOUT,
            message="terminal cancellation persistence exceeded its monotonic deadline",
            case_id=case_id,
            stop_reason="TERMINAL_DURABILITY_TIMEOUT",
        )
        _append_error(fallback, timeout_error)
        await publish_recovery(fallback, timeout_error)
        raise exc from None

    if outcome.failure is not None:
        persistence_error = _secondary_error(
            outcome.failure,
            case_id=case_id,
            category=ErrorCategory.DATABASE,
            stop_reason="TERMINAL_PERSISTENCE_FAILED",
            message="terminal database write failed",
        )
        _append_error(outcome.result, persistence_error)
        await publish_recovery(outcome.result, persistence_error)
        if not isinstance(outcome.failure, Exception):
            raise outcome.failure
        if outcome.control_exception is not None:
            raise outcome.control_exception
        return
    if outcome.control_exception is not None:
        raise outcome.control_exception


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
        deadline=monotonic() + (3 * DURABILITY_DEADLINE_SECONDS),
    )


async def process_batch(
    paths: list[Path], settings: Settings, concurrency: int | None = None
) -> list[CaseResult]:
    """Prepare all identities, then run independent fresh teams with bounded concurrency."""

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
            item = prepare_claimed_case(path, settings)
            if isinstance(item, _PreparationFailure):
                if not item.recovery_published:
                    _write_result(item.result)
                results.append(item.result)
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
    semaphore = asyncio.Semaphore(concurrency or settings.case_concurrency)

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
                deadline=monotonic() + (2 * DURABILITY_DEADLINE_SECONDS),
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
            deadline=monotonic() + (3 * DURABILITY_DEADLINE_SECONDS),
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


async def resume_case(
    case_id: str,
    settings: Settings,
    *,
    claim: ExecutionClaim | None = None,
) -> CaseResult:
    """Load a stopped Swarm only after an attributable human decision and resume it."""

    if claim is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "an exact execution claim is required to resume a case",
            case_id=case_id,
            stop_reason="EXECUTION_CLAIM_MISSING",
        ) from None
    if claim.case_id != case_id:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "supplied execution claim belongs to a different case",
            case_id=case_id,
            stop_reason="STALE_EXECUTION_CLAIM",
        ) from None
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
    )
