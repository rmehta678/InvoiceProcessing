"""Server-sent event tail over persisted case events.

The stream is only a window onto the events table: rows are read with a rowid
cursor at ~1s cadence and forwarded verbatim (summarized for display, payload
untouched). Terminal state is emitted from the stored case row, never inferred.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sse_starlette import ServerSentEvent

from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult, ErrorRecord
from invoice_agents.ui.queries import EventRow, events_after
from invoice_agents.ui.runs import RunRegistry

POLL_SECONDS = 1.0
HEARTBEAT_SECONDS = 15.0


def _tool_names(content: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                calls.append(
                    {
                        "id": item.get("id") or item.get("call_id"),
                        "name": item.get("name"),
                        "is_error": item.get("is_error"),
                    }
                )
    return calls


def summarize_event(row: EventRow) -> dict[str, Any]:
    """Extract display fields for the live timeline from one stored event."""

    try:
        payload = json.loads(row.payload_json)
    except json.JSONDecodeError:
        payload = None
    summary: dict[str, Any] = {
        "seq": row.seq,
        "event_type": row.event_type,
        "agent": row.agent_name,
        "created_at": row.created_at,
    }
    if not isinstance(payload, dict):
        return summary
    if row.event_type == "autogen.HandoffMessage":
        summary["handoff"] = {"source": payload.get("source"), "target": payload.get("target")}
    elif row.event_type == "autogen.ToolCallRequestEvent":
        summary["tool_calls"] = _tool_names(payload.get("content"))
    elif row.event_type == "autogen.ToolCallExecutionEvent":
        summary["tool_results"] = _tool_names(payload.get("content"))
    elif row.event_type == "provider.retry":
        summary["message"] = payload.get("message")
    elif row.event_type in {"case.finished", "case.resumed_finished", "case.failed"}:
        summary["status"] = payload.get("status")
        summary["stop_reason"] = payload.get("stop_reason")
    return summary


def terminal_payload(
    workflow_db: Path, case_id: str, registry: RunRegistry
) -> dict[str, Any] | None:
    """Return storage-derived terminal state or None for one valid active lease."""

    store = WorkflowStore(workflow_db)
    try:
        snapshot = store.load_case_execution_snapshot(case_id)
    except InvoiceAgentsError as exc:
        return _recovery_failure_payload(
            case_id,
            exc.stop_reason
            if exc.stop_reason in {"PERSISTED_RESULT_INVALID", "EXECUTION_AUTHORITY_CORRUPT"}
            else "EXECUTION_RECOVERY_FAILED",
        )
    except Exception:
        return _recovery_failure_payload(case_id, "EXECUTION_RECOVERY_FAILED")
    if snapshot is None:
        return {"case_id": case_id, "missing": True}
    if snapshot.has_valid_lease:
        return None
    if snapshot.execution_state == "FINISHED":
        if snapshot.result is None:
            return _recovery_failure_payload(case_id, "PERSISTED_RESULT_INVALID")
        return _stored_terminal_payload(case_id, snapshot.result, registry)
    try:
        store.recover_expired_executions(case_id=case_id)
        snapshot = store.load_case_execution_snapshot(case_id)
    except InvoiceAgentsError as exc:
        return _recovery_failure_payload(
            case_id,
            exc.stop_reason
            if exc.stop_reason in {"PERSISTED_RESULT_INVALID", "EXECUTION_AUTHORITY_CORRUPT"}
            else "EXECUTION_RECOVERY_FAILED",
        )
    except Exception:
        return _recovery_failure_payload(case_id, "EXECUTION_RECOVERY_FAILED")
    if snapshot is None:
        return _recovery_failure_payload(case_id, "EXECUTION_RECOVERY_FAILED")
    if snapshot.has_valid_lease:
        return None
    if snapshot.execution_state != "FINISHED" or snapshot.result is None:
        return _recovery_failure_payload(case_id, "EXECUTION_RECOVERY_FAILED")
    return _stored_terminal_payload(case_id, snapshot.result, registry)


def _stored_terminal_payload(
    case_id: str,
    result: CaseResult,
    registry: RunRegistry,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": str(result.status),
        "stop_reason": result.stop_reason,
        "run_error": (
            "background execution ended with an exception; persisted result is authoritative"
            if registry.run_error(case_id) is not None
            else None
        ),
    }


def _recovery_failure_payload(case_id: str, stop_reason: str) -> dict[str, Any]:
    error = ErrorRecord(
        category=ErrorCategory.DATABASE,
        message="persisted execution state could not be trusted or recovered",
        case_id=case_id,
        stop_reason=stop_reason,
    )
    return {
        "case_id": case_id,
        "status": "INCOMPLETE",
        "stop_reason": stop_reason,
        "run_error": None,
        "recovery_error": error.model_dump(mode="json"),
    }


async def case_event_stream(
    workflow_db: Path,
    case_id: str,
    registry: RunRegistry,
    after_seq: int = 0,
) -> AsyncIterator[ServerSentEvent]:
    """Yield stored events as they appear, then one terminal event, then stop."""

    last_seq = after_seq
    loop = asyncio.get_running_loop()
    last_heartbeat = loop.time()
    while True:
        rows = await asyncio.to_thread(events_after, workflow_db, case_id, last_seq)
        for row in rows:
            last_seq = row.seq
            yield ServerSentEvent(
                event="case-event",
                id=str(row.seq),
                data=json.dumps(summarize_event(row), ensure_ascii=False, default=str),
            )
        terminal = await asyncio.to_thread(terminal_payload, workflow_db, case_id, registry)
        if terminal is not None:
            yield ServerSentEvent(
                event="terminal",
                data=json.dumps(terminal, ensure_ascii=False, default=str),
            )
            return
        observed_at = loop.time()
        since_heartbeat = observed_at - last_heartbeat
        if since_heartbeat >= HEARTBEAT_SECONDS:
            yield ServerSentEvent(comment="heartbeat")
            last_heartbeat = loop.time()
            since_heartbeat = 0.0
        await asyncio.sleep(min(POLL_SECONDS, max(0.0, HEARTBEAT_SECONDS - since_heartbeat)))
