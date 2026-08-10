"""Terminable provider/model/team lifecycle process with private credential transport."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from invoice_agents.config import Settings
from invoice_agents.db.store import ExecutionClaim, validate_execution_claim
from invoice_agents.isolated_process import (
    IsolatedProcessCleanupError,
    PrivatePipeEndpoint,
    PrivatePipeInput,
    ProcessCancellation,
    private_pipe_channel,
    run_isolated_process,
    send_private_frame,
)
from invoice_agents.wire_settings import decode_wire_settings, serialize_wire_settings
from invoice_agents.worker_environment import sanitized_worker_environment

LIFECYCLE_PROTOCOL_VERSION = 1
LIFECYCLE_MAX_MESSAGE_BYTES = 2_097_152
LIFECYCLE_MAX_CREDENTIAL_BYTES = 16_384
_LIFECYCLE_ERROR_CODES = frozenset(
    {
        "LIFECYCLE_FAILED",
        "LIFECYCLE_WORKER_CANCELLED",
        "LIFECYCLE_WORKER_CLEANUP_FAILED",
        "LIFECYCLE_WORKER_CRASHED",
        "LIFECYCLE_WORKER_PROTOCOL_INVALID",
        "LIFECYCLE_WORKER_TIMED_OUT",
    }
)


@dataclass(frozen=True, slots=True)
class LifecycleProcessOutcome:
    """One allowlisted hint observed after the whole lifecycle session was reaped."""

    acknowledged: bool
    error_code: str | None


def _lifecycle_worker_command() -> list[str]:
    return [sys.executable, "-m", "invoice_agents.lifecycle_worker"]


def _private_credential_channel() -> tuple[PrivatePipeEndpoint, PrivatePipeEndpoint]:
    """Create one anonymous, strictly directional credential pipe."""

    return private_pipe_channel()


def _send_private_credential(
    writer: PrivatePipeEndpoint,
    credential: bytearray,
) -> None:
    """Compatibility seam for focused frame-reader tests."""

    send_private_frame(
        writer,
        credential,
        max_payload_bytes=LIFECYCLE_MAX_CREDENTIAL_BYTES,
        deadline=time.monotonic() + 1.0,
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate lifecycle protocol key")
        payload[key] = value
    return payload


def _reject_constant(_value: str) -> object:
    raise ValueError("non-finite lifecycle protocol number")


def _redacted_settings(settings: Settings) -> dict[str, object]:
    payload = serialize_wire_settings(settings)
    if payload.get("xai_api_key") is not None:
        raise ValueError("lifecycle settings serialization is not redacted")
    payload = dict(payload)
    payload["inventory_db"] = os.fspath(settings.inventory_db.resolve())
    payload["workflow_db"] = os.fspath(settings.workflow_db.resolve())
    payload["source_archive_dir"] = os.fspath(settings.source_archive_dir.resolve())
    return payload


def _claim_payload(claim: ExecutionClaim) -> dict[str, object]:
    claim = validate_execution_claim(claim)
    return {
        "case_id": claim.case_id,
        "expires_at": claim.expires_at.isoformat(),
        "generation": claim.generation,
        "token": claim.token,
    }


def _canonical_start(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise ValueError("lifecycle start time must be canonical UTC")
    encoded = value.isoformat()
    if datetime.fromisoformat(encoded) != value:
        raise ValueError("lifecycle start time must be canonical UTC")
    return encoded


def _encode_request(
    *,
    mode: Literal["process", "resume"],
    settings: Settings,
    claim: ExecutionClaim,
    started_at: datetime,
    credential_fd: int,
) -> tuple[bytes, bytearray]:
    claim = validate_execution_claim(claim)
    if type(mode) is not str or mode not in {"process", "resume"}:
        raise ValueError("invalid lifecycle mode")
    if type(credential_fd) is not int or credential_fd < 0:
        raise ValueError("invalid lifecycle credential descriptor")
    payload = {
        "claim": _claim_payload(claim),
        "credential_fd": credential_fd,
        "mode": mode,
        "protocol_version": LIFECYCLE_PROTOCOL_VERSION,
        "settings": _redacted_settings(settings),
        "started_at": _canonical_start(started_at),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > LIFECYCLE_MAX_MESSAGE_BYTES:
        raise ValueError("lifecycle request exceeds its bound")
    credential = bytearray(settings.provider_key().encode("utf-8"))
    if not credential or len(credential) > LIFECYCLE_MAX_CREDENTIAL_BYTES:
        for index in range(len(credential)):
            credential[index] = 0
        raise ValueError("provider credential exceeds its private transport bound")
    return encoded, credential


def decode_lifecycle_request(
    encoded: bytes,
) -> tuple[str, Settings, ExecutionClaim, datetime, int]:
    """Decode strict nonsecret lifecycle metadata; the credential remains on its pipe."""

    if not encoded or len(encoded) > LIFECYCLE_MAX_MESSAGE_BYTES:
        raise ValueError("invalid lifecycle request size")
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if type(payload) is not dict or set(payload) != {
        "claim",
        "credential_fd",
        "mode",
        "protocol_version",
        "settings",
        "started_at",
    }:
        raise ValueError("invalid lifecycle request shape")
    if type(payload["protocol_version"]) is not int or payload["protocol_version"] != 1:
        raise ValueError("invalid lifecycle protocol version")
    mode = payload["mode"]
    raw_claim = payload["claim"]
    raw_settings = payload["settings"]
    raw_started_at = payload["started_at"]
    credential_fd = payload["credential_fd"]
    if (
        type(mode) is not str
        or mode not in {"process", "resume"}
        or type(raw_claim) is not dict
        or set(raw_claim) != {"case_id", "expires_at", "generation", "token"}
        or type(raw_claim["expires_at"]) is not str
        or type(raw_settings) is not dict
        or raw_settings.get("xai_api_key") is not None
        or type(raw_started_at) is not str
        or type(credential_fd) is not int
        or credential_fd < 3
    ):
        raise ValueError("invalid lifecycle request values")
    expires_at = datetime.fromisoformat(raw_claim["expires_at"])
    claim = validate_execution_claim(
        ExecutionClaim(
            case_id=raw_claim["case_id"],
            token=raw_claim["token"],
            generation=raw_claim["generation"],
            expires_at=expires_at,
        )
    )
    started_at = datetime.fromisoformat(raw_started_at)
    if (
        expires_at.tzinfo is not UTC
        or expires_at.isoformat() != raw_claim["expires_at"]
        or started_at.tzinfo is not UTC
        or started_at.isoformat() != raw_started_at
    ):
        raise ValueError("invalid lifecycle timestamp")
    settings = decode_wire_settings(raw_settings)
    if settings.xai_api_key is not None:
        raise ValueError("lifecycle JSON contains provider credentials")
    return mode, settings, claim, started_at, credential_fd


def encode_lifecycle_response(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > LIFECYCLE_MAX_MESSAGE_BYTES:
        raise ValueError("lifecycle response exceeds its bound")
    return encoded


def _decode_response(encoded: bytes) -> LifecycleProcessOutcome:
    if not encoded or len(encoded) > LIFECYCLE_MAX_MESSAGE_BYTES:
        raise ValueError("invalid lifecycle response size")
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if type(payload) is not dict or type(payload.get("ok")) is not bool:
        raise ValueError("invalid lifecycle response shape")
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if canonical != encoded:
        raise ValueError("noncanonical lifecycle response")
    if payload["ok"]:
        if set(payload) != {"ok"}:
            raise ValueError("invalid lifecycle success response")
        return LifecycleProcessOutcome(True, None)
    if set(payload) != {"error_code", "ok"}:
        raise ValueError("invalid lifecycle failure response")
    code = payload["error_code"]
    if type(code) is not str or code not in _LIFECYCLE_ERROR_CODES:
        raise ValueError("invalid lifecycle failure code")
    return LifecycleProcessOutcome(False, code)


def run_lifecycle_process(
    *,
    mode: Literal["process", "resume"],
    settings: Settings,
    claim: ExecutionClaim,
    started_at: datetime,
    timeout_seconds: float,
    cancel_requested: ProcessCancellation | None = None,
) -> LifecycleProcessOutcome:
    """Run provider/team work and prove all local descendants stopped before return."""

    credential = bytearray()
    credential_reader, credential_writer = _private_credential_channel()
    with credential_reader, credential_writer:
        try:
            request, credential = _encode_request(
                mode=mode,
                settings=settings,
                claim=claim,
                started_at=started_at,
                credential_fd=credential_reader.fileno(),
            )
            try:
                outcome = run_isolated_process(
                    command=_lifecycle_worker_command(),
                    request=request,
                    timeout_seconds=timeout_seconds,
                    max_response_bytes=LIFECYCLE_MAX_MESSAGE_BYTES,
                    cancel_requested=cancel_requested,
                    pass_fds=(credential_reader.fileno(),),
                    private_input=PrivatePipeInput(
                        reader=credential_reader,
                        writer=credential_writer,
                        payload=credential,
                        max_payload_bytes=LIFECYCLE_MAX_CREDENTIAL_BYTES,
                    ),
                    env=sanitized_worker_environment(),
                )
            except IsolatedProcessCleanupError:
                return LifecycleProcessOutcome(False, "LIFECYCLE_WORKER_CLEANUP_FAILED")
        finally:
            for index in range(len(credential)):
                credential[index] = 0
    if outcome.failure == "cancelled":
        return LifecycleProcessOutcome(False, "LIFECYCLE_WORKER_CANCELLED")
    if outcome.failure == "timeout":
        return LifecycleProcessOutcome(False, "LIFECYCLE_WORKER_TIMED_OUT")
    if outcome.failure in {"start", "crash"}:
        return LifecycleProcessOutcome(False, "LIFECYCLE_WORKER_CRASHED")
    if outcome.failure == "protocol" or outcome.response is None:
        return LifecycleProcessOutcome(False, "LIFECYCLE_WORKER_PROTOCOL_INVALID")
    try:
        return _decode_response(outcome.response)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return LifecycleProcessOutcome(False, "LIFECYCLE_WORKER_PROTOCOL_INVALID")
