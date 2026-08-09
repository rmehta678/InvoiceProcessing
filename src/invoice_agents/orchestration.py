"""Case lifecycle, streamed AutoGen execution, status mapping, persistence, and resume."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
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
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
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
    lease_seconds: int,
    renewal_interval_seconds: float,
) -> ResultT:
    """Run one lifecycle while renewal failures cancel and replace its result."""

    if renewal_interval_seconds <= 0:
        raise ValueError("renewal_interval_seconds must be positive")
    stopped = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(stopped.wait(), timeout=renewal_interval_seconds)
                return
            except TimeoutError:
                await asyncio.to_thread(renew, claim, lease_seconds)

    operation_task: asyncio.Future[ResultT] = asyncio.ensure_future(operation)
    heartbeat_task: asyncio.Task[None] = asyncio.create_task(heartbeat())
    waiters: set[asyncio.Future[Any]] = set()
    waiters.add(operation_task)
    waiters.add(heartbeat_task)
    try:
        done, _pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        if heartbeat_task in done and (failure := heartbeat_task.exception()) is not None:
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
        return operation_task.result()
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


def _error_record(exc: BaseException, case_id: str | None = None) -> ErrorRecord:
    if isinstance(exc, InvoiceAgentsError):
        return ErrorRecord(
            category=exc.category,
            message=exc.message,
            case_id=exc.case_id or case_id,
            stop_reason=exc.stop_reason,
            provider_request_id=exc.provider_request_id,
            details=exc.details or {},
        )
    if isinstance(exc, openai.APIResponseValidationError):
        # The provider answered with a payload the SDK could not validate; this is a
        # provider contract failure, never a value to repair or retry into success.
        category = ErrorCategory.PROVIDER
        stop_reason = "PROVIDER_RESPONSE_INVALID"
    elif isinstance(exc, openai.AuthenticationError):
        category = ErrorCategory.AUTHENTICATION
        stop_reason = "PROVIDER_AUTHENTICATION_FAILED"
    elif isinstance(exc, openai.RateLimitError):
        category = ErrorCategory.RATE_LIMIT
        stop_reason = "PROVIDER_RATE_LIMIT_EXHAUSTED"
    elif isinstance(exc, (openai.APITimeoutError, TimeoutError, asyncio.TimeoutError)):
        category = ErrorCategory.TIMEOUT
        stop_reason = "PROVIDER_TIMEOUT"
    elif isinstance(exc, (openai.APIConnectionError, openai.APIStatusError)):
        category = ErrorCategory.PROVIDER
        stop_reason = "PROVIDER_REQUEST_FAILED"
    elif isinstance(exc, sqlite3.Error):
        category = ErrorCategory.DATABASE
        stop_reason = "DATABASE_ERROR"
    elif isinstance(exc, json.JSONDecodeError):
        category = ErrorCategory.SCHEMA
        stop_reason = "RESPONSE_DECODE_FAILED"
    elif isinstance(exc, ValidationError):
        category = ErrorCategory.SCHEMA
        stop_reason = "SCHEMA_VALIDATION_FAILED"
    else:
        category = ErrorCategory.ORCHESTRATION
        stop_reason = "UNEXPECTED_RUNTIME_ERROR"
    return ErrorRecord(
        category=category,
        message=str(exc),
        case_id=case_id,
        stop_reason=stop_reason,
        provider_request_id=getattr(exc, "request_id", None),
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


def prepare_case(path: Path, settings: Settings) -> tuple[str, datetime] | CaseResult:
    """Create immutable source/case evidence before any model request.

    Batch preparation is sequential so every later identity agent can see all submitted
    representations and revisions even though independent Swarms run concurrently.
    """

    started_at = _now()
    case_id = f"case_{uuid4().hex}"
    source_id: str | None = None
    case_created = False
    claim: ExecutionClaim | None = None
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
        store.release_case_execution(claim)
        return case_id, started_at
    except BaseException as exc:
        result = _failed_result(case_id, source_id, started_at, exc)
        if source_id is not None and case_created:
            try:
                failed_store = WorkflowStore(settings)
                if claim is None:
                    claim = failed_store.claim_case_execution(
                        case_id,
                        frozenset({CaseStatus.INCOMPLETE}),
                        EXECUTION_LEASE_SECONDS,
                    )
                failed_store.finish_case(result, claim)
            except BaseException as persistence_exc:
                # Preserve both the original case failure and the secondary audit-write
                # failure; neither may disappear behind the other.
                result.errors.append(_error_record(persistence_exc, case_id))
        return result


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
    target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return target


async def run_prepared_case(case_id: str, started_at: datetime, settings: Settings) -> CaseResult:
    """Run one fresh Swarm and convert every terminal path to an explicit case status."""

    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        EXECUTION_LEASE_SECONDS,
    )
    invoice = store.promote_predecessor_extraction(claim)
    audit = AuditRecorder(settings.workflow_db, case_id)
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
        claim=claim,
    )
    usage = UsageSummary()
    clock = monotonic()
    client = create_model_client(settings)
    team = build_team(context, client)

    async def execute_lifecycle() -> CaseResult:
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
        store.save_team_state(case_id, dict(state), claim)
        usage.latency_ms = int((monotonic() - clock) * 1000)
        return _result_from_stop(context, task_result, started_at, usage)

    try:
        result = await _run_with_lease_heartbeat(
            execute_lifecycle(),
            renew=store.renew_case_execution,
            claim=claim,
            lease_seconds=EXECUTION_LEASE_SECONDS,
            renewal_interval_seconds=EXECUTION_RENEWAL_INTERVAL_SECONDS,
        )
    except BaseException as exc:
        usage.latency_ms = int((monotonic() - clock) * 1000)
        result = _failed_result(case_id, invoice.source.source_id, started_at, exc)
        result.usage = usage
        audit.record(
            "case.failed",
            result.model_dump(mode="json"),
            source_id=invoice.source.source_id,
            provider_request_id=result.errors[0].provider_request_id if result.errors else None,
        )
    finally:
        await client.close()
    # Retries are counted from persisted provider.retry audit events so the summary
    # reflects what actually happened, not an in-memory counter that can be lost.
    result.usage.retries = store.count_events(case_id, "provider.retry")
    store.finish_case(result, claim)
    _write_result(result)
    audit.record(
        "case.finished", result.model_dump(mode="json"), source_id=invoice.source.source_id
    )
    return result


def prepare_invoice(path: Path, settings: Settings) -> tuple[str, datetime] | CaseResult:
    """Preflight and prepare one source; failures return an already-written terminal result.

    This is the shared pre-model seam for the CLI and the web console: the returned
    case ID exists before any model call, so a caller can watch the run live while
    `run_prepared_case` executes.
    """

    started_at = _now()
    try:
        preflight(settings)
    except BaseException as exc:
        # The printed artifacts/results path must exist for every terminal outcome,
        # including failures that happen before a case or model call exists.
        result = _failed_result(f"case_{uuid4().hex}", None, started_at, exc)
        _write_result(result)
        return result
    prepared = prepare_case(path, settings)
    if isinstance(prepared, CaseResult):
        _write_result(prepared)
        return prepared
    return prepared


async def process_invoice(path: Path, settings: Settings) -> CaseResult:
    """Preflight, prepare, and process one source artifact."""

    prepared = prepare_invoice(path, settings)
    if isinstance(prepared, CaseResult):
        return prepared
    return await run_prepared_case(prepared[0], prepared[1], settings)


async def process_batch(
    paths: list[Path], settings: Settings, concurrency: int | None = None
) -> list[CaseResult]:
    """Prepare all identities, then run independent fresh teams with bounded concurrency."""

    started_at = _now()
    try:
        preflight(settings)
    except BaseException as exc:
        failed = [_failed_result(f"case_{uuid4().hex}", None, started_at, exc) for _ in paths]
        for result in failed:
            _write_result(result)
        return failed
    prepared: list[tuple[str, datetime]] = []
    results: list[CaseResult] = []
    for path in paths:
        item = prepare_case(path, settings)
        if isinstance(item, CaseResult):
            _write_result(item)
            results.append(item)
        else:
            prepared.append(item)
    semaphore = asyncio.Semaphore(concurrency or settings.case_concurrency)

    async def bounded(item: tuple[str, datetime]) -> CaseResult:
        async with semaphore:
            return await run_prepared_case(item[0], item[1], settings)

    results.extend(await asyncio.gather(*(bounded(item) for item in prepared)))
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


async def resume_case(case_id: str, settings: Settings) -> CaseResult:
    """Load a stopped Swarm only after an attributable human decision and resume it."""

    preflight(settings)
    store = WorkflowStore(settings)
    try:
        claim = store.claim_case_execution(
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
            ) from exc
        raise
    previous = store.load_result(case_id)
    if previous is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            f"case {case_id} has no persisted result to resume",
            case_id=case_id,
            stop_reason="CASE_RESULT_MISSING",
        )
    if previous.status is not CaseStatus.NEEDS_HUMAN:
        store.finish_case(previous, claim)
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            f"case {case_id} is not waiting for human review",
            case_id=case_id,
            stop_reason="CASE_NOT_RESUMABLE",
        )
    review = store.load_case_review(case_id)
    if review is None or review.status != "RESOLVED" or review.human_decision is None:
        store.finish_case(previous, claim)
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "a persisted human decision is required before resume",
            case_id=case_id,
            stop_reason="HUMAN_DECISION_MISSING",
        )
    state = store.load_team_state(case_id)
    if state is None:
        store.finish_case(previous, claim)
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "stopped AutoGen team state is missing",
            case_id=case_id,
            stop_reason="TEAM_STATE_MISSING",
        )
    store.adopt_latest_evidence(claim)
    invoice = store.load_current_extraction(claim)
    audit = AuditRecorder(settings.workflow_db, case_id)
    human = review.human_decision
    if human.decision is HumanDecisionKind.ESTABLISH_MAPPING:
        _recompute_after_mapping(case_id, human, settings, store, audit, claim)
        invoice = store.load_current_extraction(claim)
    context = AgentCaseContext(case_id, settings, store, audit, claim)
    client = create_model_client(settings)
    team = build_team(context, client)
    usage = previous.usage.model_copy(deep=True)
    clock = monotonic()

    async def execute_resume_lifecycle() -> CaseResult:
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
        store.save_team_state(case_id, dict(new_state), claim)
        usage.latency_ms += int((monotonic() - clock) * 1000)
        return _result_from_stop(context, task_result, previous.started_at, usage)

    try:
        result = await _run_with_lease_heartbeat(
            execute_resume_lifecycle(),
            renew=store.renew_case_execution,
            claim=claim,
            lease_seconds=EXECUTION_LEASE_SECONDS,
            renewal_interval_seconds=EXECUTION_RENEWAL_INTERVAL_SECONDS,
        )
    except BaseException as exc:
        usage.latency_ms += int((monotonic() - clock) * 1000)
        result = _failed_result(case_id, invoice.source.source_id, previous.started_at, exc)
        result.usage = usage
    finally:
        await client.close()
    result.usage.retries = store.count_events(case_id, "provider.retry")
    store.finish_case(result, claim)
    _write_result(result)
    audit.record(
        "case.resumed_finished", result.model_dump(mode="json"), source_id=invoice.source.source_id
    )
    return result
