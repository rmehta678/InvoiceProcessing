"""Fresh-interpreter entry point for terminal case persistence."""

from __future__ import annotations

import os
import sys
from typing import Any

from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.terminal_process import (
    TERMINAL_WORKER_MAX_MESSAGE_BYTES,
    _claim_payload,
    decode_terminal_request,
    encode_terminal_response,
)

_EXPOSED_ERROR_CODES = frozenset(
    {
        "EVIDENCE_AUTHORITY_MISSING",
        "EXECUTION_AUTHORITY_CORRUPT",
        "PERSISTED_RESULT_INVALID",
        "STALE_EXECUTION_CLAIM",
        "TERMINAL_DURABILITY_UNRESOLVED",
        "TERMINAL_RECOVERY_ARTIFACT_FAILED",
    }
)


def _failure(
    code: str = "TERMINAL_WORKER_FAILED",
    *,
    claim: Any | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "claim": _claim_payload(claim) if claim is not None else None,
        "result": None,
        "error_code": code,
        "evidence_state": None,
        "evidence_result": None,
    }


def _response(
    *,
    result: Any,
    error_code: str | None,
    evidence: Any,
    claim: Any,
) -> dict[str, Any]:
    evidence_state = evidence.state
    evidence_result = evidence.result
    return {
        "ok": error_code is None,
        "claim": _claim_payload(claim),
        "result": (
            result.model_dump(mode="json") if error_code is None and result is not None else None
        ),
        "error_code": error_code,
        "evidence_state": evidence_state.value,
        "evidence_result": (
            evidence_result.model_dump(mode="json") if evidence_result is not None else None
        ),
    }


def main() -> None:
    """Read one bounded request and emit one traceback-free bounded response."""

    stderr_descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(stderr_descriptor, sys.stderr.fileno())
    os.close(stderr_descriptor)
    request = sys.stdin.buffer.read(TERMINAL_WORKER_MAX_MESSAGE_BYTES + 1)
    response_claim: Any | None = None
    try:
        mode, settings, claim, started_at, result, worker_error_code = (
            decode_terminal_request(request)
        )
        response_claim = claim
        from invoice_agents.db.store import WorkflowStore
        from invoice_agents.orchestration import (
            _cancelled_result,
            _canonical_recovery_persistence_error,
            _ExactClaimEvidenceState,
            _inspect_exact_claim_evidence,
            _recovery_artifact_or_raise,
            _recovery_only_result,
        )

        store = WorkflowStore(settings)
        operation_result = result
        operation_error: str | None = None
        if mode == "publish_cancel_recovery":
            if started_at is None or worker_error_code is None:
                raise ValueError("invalid recovery worker request")
            evidence = _inspect_exact_claim_evidence(store, claim)
            if evidence.state is _ExactClaimEvidenceState.DURABLE_DATABASE_RESULT:
                operation_result = evidence.result
            else:
                source_id = store.load_authoritative_case_source_id(claim)
                previous = store.load_result(claim.case_id)
                operation_result = _cancelled_result(
                    claim.case_id,
                    source_id,
                    started_at,
                    previous,
                )
                operation_result = store.merge_relational_case_evidence(operation_result)
                persistence_stop = (
                    "TERMINAL_DURABILITY_TIMEOUT"
                    if worker_error_code == "TERMINAL_WORKER_TIMEOUT"
                    else "TERMINAL_PERSISTENCE_FAILED"
                )
                persistence_error = _canonical_recovery_persistence_error(
                    claim.case_id,
                    persistence_stop,
                )
                operation_result = _recovery_only_result(
                    operation_result,
                    persistence_error,
                )
                _recovery_artifact_or_raise(
                    operation_result,
                    persistence_error,
                    store=store,
                    claim=claim,
                )
        else:
            try:
                if mode == "cancel_unstarted":
                    if started_at is None or result is not None:
                        raise ValueError("invalid cancel worker request")
                    store.require_current_execution_claim(claim)
                    source_id = store.load_authoritative_case_source_id(claim)
                    previous = store.load_result(claim.case_id)
                    operation_result = _cancelled_result(
                        claim.case_id,
                        source_id,
                        started_at,
                        previous,
                    )
                    operation_result = store.merge_relational_case_evidence(operation_result)
                    store.finish_case(operation_result, claim)
                elif mode == "finish":
                    if result is None or started_at is not None:
                        raise ValueError("invalid finish worker request")
                    store.finish_case(result, claim)
                elif mode == "update":
                    if result is None or started_at is not None:
                        raise ValueError("invalid update worker request")
                    store.update_finished_case_result(result, claim)
                elif mode != "inspect_claim":
                    raise ValueError("invalid terminal worker mode")
            except InvoiceAgentsError as exc:
                operation_result = None
                operation_error = (
                    exc.stop_reason
                    if exc.stop_reason in _EXPOSED_ERROR_CODES
                    else "TERMINAL_WORKER_FAILED"
                )
            except Exception:
                operation_result = None
                operation_error = "TERMINAL_WORKER_FAILED"
            evidence = _inspect_exact_claim_evidence(store, claim)
        response = _response(
            result=operation_result,
            error_code=operation_error,
            evidence=evidence,
            claim=claim,
        )
    except InvoiceAgentsError as exc:
        response = _failure(
            exc.stop_reason
            if exc.stop_reason in _EXPOSED_ERROR_CODES
            else "TERMINAL_WORKER_FAILED",
            claim=response_claim,
        )
    except BaseException:
        response = _failure(claim=response_claim)
    try:
        encoded = encode_terminal_response(response)
    except BaseException:
        encoded = encode_terminal_response(_failure(claim=response_claim))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
