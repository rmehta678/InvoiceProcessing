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
from typing import Any, Literal

from invoice_agents.config import Settings
from invoice_agents.db.migration_process import (
    _capture_worker_session,
    _retry_quarantined_workers,
    _serialize_settings,
    _stop_worker,
    _uncertain_worker_session,
)
from invoice_agents.db.store import ExecutionClaim, validate_execution_claim
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult

TERMINAL_WORKER_PROTOCOL_VERSION = 1
TERMINAL_WORKER_MAX_MESSAGE_BYTES = 2_097_152
_TERMINAL_WORKER_POLL_SECONDS = 0.02
_TERMINAL_ERROR_CODES = frozenset(
    {
        "EVIDENCE_AUTHORITY_MISSING",
        "EXECUTION_AUTHORITY_CORRUPT",
        "PERSISTED_RESULT_INVALID",
        "STALE_EXECUTION_CLAIM",
        "TERMINAL_WORKER_FAILED",
    }
)


@dataclass(frozen=True, slots=True)
class TerminalProcessOutcome:
    """One parent-owned worker outcome after its whole session is proven empty."""

    result: CaseResult | None
    error_code: str | None


def _terminal_worker_command() -> list[str]:
    return [sys.executable, "-m", "invoice_agents.terminal_worker"]


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
    mode: Literal["cancel_unstarted", "finish", "update"],
    settings: Settings,
    claim: ExecutionClaim,
    started_at: datetime | None = None,
    result: CaseResult | None = None,
) -> bytes:
    exact = validate_execution_claim(claim)
    if mode == "cancel_unstarted":
        if (
            type(started_at) is not datetime
            or started_at.utcoffset() != timedelta(0)
            or started_at.astimezone(UTC).isoformat()
            != datetime.fromisoformat(started_at.isoformat()).astimezone(UTC).isoformat()
            or result is not None
        ):
            raise ValueError("cancel worker requires one canonical UTC start time")
    elif result is None or started_at is not None or result.case_id != exact.case_id:
        raise ValueError("finish worker requires one claim-bound case result")
    payload = {
        "protocol_version": TERMINAL_WORKER_PROTOCOL_VERSION,
        "mode": mode,
        "settings": _serialize_settings(settings),
        "claim": _claim_payload(exact),
        "started_at": started_at.astimezone(UTC).isoformat() if started_at is not None else None,
        "result": result.model_dump(mode="json") if result is not None else None,
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
) -> tuple[str, Settings, ExecutionClaim, datetime | None, CaseResult | None]:
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
    }:
        raise ValueError("invalid terminal worker request shape")
    if type(payload["protocol_version"]) is not int or payload["protocol_version"] != 1:
        raise ValueError("invalid terminal worker protocol version")
    mode = payload["mode"]
    if type(mode) is not str or mode not in {"cancel_unstarted", "finish", "update"}:
        raise ValueError("invalid terminal worker mode")
    settings_payload = payload["settings"]
    if type(settings_payload) is not dict:
        raise ValueError("invalid terminal worker settings")
    settings = Settings(**settings_payload)
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
    if mode == "cancel_unstarted":
        if type(raw_started) is not str or raw_result is not None:
            raise ValueError("invalid cancel worker payload")
        started_at = datetime.fromisoformat(raw_started)
        if started_at.tzinfo is not UTC or started_at.isoformat() != raw_started:
            raise ValueError("invalid cancel worker start time")
        result = None
    else:
        if raw_started is not None or type(raw_result) is not dict:
            raise ValueError("invalid finish worker payload")
        started_at = None
        # Pydantic's strict Python mode rejects JSON datetime/enum strings even
        # though they are their canonical wire representation.  Re-validate the
        # already shape-checked object through strict JSON mode so transport
        # scalars remain non-coercible while canonical model JSON is accepted.
        result = CaseResult.model_validate_json(
            json.dumps(raw_result, ensure_ascii=True, separators=(",", ":")),
            strict=True,
        )
        if result.case_id != claim.case_id:
            raise ValueError("terminal worker result is not claim-bound")
    return mode, settings, claim, started_at, result


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


def _read_response(worker: Any, timeout_seconds: float) -> bytes:
    stdout = worker.process.stdout
    watcher = worker.exit_watcher
    if stdout is None or watcher is None:
        raise TimeoutError
    descriptor = stdout.fileno()
    os.set_blocking(descriptor, False)
    deadline = time.monotonic() + timeout_seconds
    response = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            if not selector.select(min(remaining, _TERMINAL_WORKER_POLL_SECONDS)):
                if watcher.wait(0):
                    break
                continue
            chunk = os.read(
                descriptor,
                min(8_192, TERMINAL_WORKER_MAX_MESSAGE_BYTES + 1 - len(response)),
            )
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > TERMINAL_WORKER_MAX_MESSAGE_BYTES:
                raise ValueError("oversize terminal worker response")
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not watcher.wait(remaining):
        raise TimeoutError
    return bytes(response)


def _decode_response(encoded: bytes) -> TerminalProcessOutcome:
    if not encoded or len(encoded) > TERMINAL_WORKER_MAX_MESSAGE_BYTES:
        raise ValueError("invalid terminal worker response size")
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if type(payload) is not dict or type(payload.get("ok")) is not bool:
        raise ValueError("invalid terminal worker response shape")
    if payload["ok"]:
        if set(payload) != {"ok", "result"} or type(payload["result"]) is not dict:
            raise ValueError("invalid terminal worker success response")
        return TerminalProcessOutcome(
            result=CaseResult.model_validate_json(
                json.dumps(
                    payload["result"],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                strict=True,
            ),
            error_code=None,
        )
    if set(payload) != {"ok", "error_code"}:
        raise ValueError("invalid terminal worker failure response")
    code = payload["error_code"]
    if type(code) is not str or code not in _TERMINAL_ERROR_CODES:
        raise ValueError("invalid terminal worker failure code")
    return TerminalProcessOutcome(result=None, error_code=code)


def run_terminal_process(
    *,
    mode: Literal["cancel_unstarted", "finish", "update"],
    settings: Settings,
    claim: ExecutionClaim,
    timeout_seconds: float,
    started_at: datetime | None = None,
    result: CaseResult | None = None,
) -> TerminalProcessOutcome:
    """Run terminal persistence, then prove the entire helper session is empty."""

    exact = validate_execution_claim(claim)
    if type(timeout_seconds) is not float or timeout_seconds <= 0:
        raise ValueError("terminal worker timeout must be a positive float")
    if not _retry_quarantined_workers():
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "terminal helper session cleanup is unresolved",
            case_id=exact.case_id,
            stop_reason="TERMINAL_WORKER_CLEANUP_FAILED",
        ) from None
    request = _encode_request(
        mode=mode,
        settings=settings,
        claim=exact,
        started_at=started_at,
        result=result,
    )
    process: subprocess.Popen[bytes] | None = None
    worker: Any | None = None
    start_failed = False
    try:
        with tempfile.TemporaryFile() as request_stream:
            request_stream.write(request)
            request_stream.seek(0)
            process = subprocess.Popen(
                _terminal_worker_command(),
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
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "terminal helper could not be started",
            case_id=exact.case_id,
            stop_reason="TERMINAL_WORKER_START_FAILED",
        ) from None
    start_failed = start_failed or not worker.watcher_initialized or not worker.identity_verified
    response: bytes | None = None
    response_failure: Literal["timeout", "protocol", "crash"] | None = None
    if not start_failed:
        try:
            response = _read_response(worker, timeout_seconds)
        except TimeoutError:
            response_failure = "timeout"
        except (OSError, UnicodeError, ValueError):
            response_failure = "protocol"
    cleanup_error = _stop_worker(worker)
    if cleanup_error is not None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "terminal helper session cleanup could not be verified",
            case_id=exact.case_id,
            stop_reason="TERMINAL_WORKER_CLEANUP_FAILED",
        ) from None
    if start_failed:
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED")
    if response_failure == "timeout":
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_TIMEOUT")
    if response_failure is not None or process is None or process.returncode != 0:
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED")
    try:
        assert response is not None
        return _decode_response(response)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED")
