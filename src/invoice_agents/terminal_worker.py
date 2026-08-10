"""Fresh-interpreter entry point for terminal case persistence."""

from __future__ import annotations

import os
import sys
from typing import Any

from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.terminal_process import (
    TERMINAL_WORKER_MAX_MESSAGE_BYTES,
    decode_terminal_request,
    encode_terminal_response,
)

_EXPOSED_ERROR_CODES = frozenset(
    {
        "EVIDENCE_AUTHORITY_MISSING",
        "EXECUTION_AUTHORITY_CORRUPT",
        "PERSISTED_RESULT_INVALID",
        "STALE_EXECUTION_CLAIM",
    }
)


def _failure(code: str = "TERMINAL_WORKER_FAILED") -> dict[str, Any]:
    return {"ok": False, "error_code": code}


def main() -> None:
    """Read one bounded request and emit one traceback-free bounded response."""

    stderr_descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(stderr_descriptor, sys.stderr.fileno())
    os.close(stderr_descriptor)
    request = sys.stdin.buffer.read(TERMINAL_WORKER_MAX_MESSAGE_BYTES + 1)
    try:
        mode, settings, claim, started_at, result = decode_terminal_request(request)
        from invoice_agents.db.store import WorkflowStore

        store = WorkflowStore(settings)
        if mode == "cancel_unstarted":
            assert started_at is not None and result is None
            from invoice_agents.orchestration import _cancelled_result

            store.require_current_execution_claim(claim)
            source_id = store.load_authoritative_case_source_id(claim)
            previous = store.load_result(claim.case_id)
            result = _cancelled_result(claim.case_id, source_id, started_at, previous)
            result = store.merge_relational_case_evidence(result)
            store.finish_case(result, claim)
        elif mode == "finish":
            assert result is not None and started_at is None
            store.finish_case(result, claim)
        else:
            assert result is not None and started_at is None
            store.update_finished_case_result(result, claim)
        response: dict[str, Any] = {"ok": True, "result": result.model_dump(mode="json")}
    except InvoiceAgentsError as exc:
        response = _failure(
            exc.stop_reason if exc.stop_reason in _EXPOSED_ERROR_CODES else "TERMINAL_WORKER_FAILED"
        )
    except BaseException:
        response = _failure()
    try:
        encoded = encode_terminal_response(response)
    except BaseException:
        encoded = encode_terminal_response(_failure())
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
