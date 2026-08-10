"""Terminable process boundary for source preparation and claim handoff."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from invoice_agents.config import Settings
from invoice_agents.isolated_process import (
    IsolatedProcessCleanupError,
    run_isolated_process,
    sanitized_worker_environment,
)
from invoice_agents.wire_settings import decode_wire_settings, serialize_wire_settings

PREPARATION_PROTOCOL_VERSION = 1
PREPARATION_MAX_MESSAGE_BYTES = 1_048_576
_EXECUTION_TOKEN = re.compile(r"^exec_[0-9a-f]{32}$")
_PREPARATION_ERROR_CODES = frozenset(
    {
        "PREPARATION_FAILED",
        "PREPARATION_WORKER_CANCELLED",
        "PREPARATION_WORKER_CLEANUP_FAILED",
        "PREPARATION_WORKER_CRASHED",
        "PREPARATION_WORKER_PROTOCOL_INVALID",
        "PREPARATION_WORKER_TIMED_OUT",
    }
)


@dataclass(frozen=True, slots=True)
class PreparationProcessOutcome:
    """One allowlisted hint observed after the entire worker session was reaped."""

    acknowledged: bool
    error_code: str | None


def _preparation_worker_command() -> list[str]:
    return [sys.executable, "-m", "invoice_agents.preparation_worker"]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate preparation protocol key")
        payload[key] = value
    return payload


def _reject_constant(_value: str) -> object:
    raise ValueError("non-finite preparation protocol number")


def _canonical_utc_text(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise ValueError("preparation start time must be canonical UTC")
    encoded = value.isoformat()
    if datetime.fromisoformat(encoded) != value:
        raise ValueError("preparation start time must be canonical UTC")
    return encoded


def _canonical_token(value: str) -> str:
    if type(value) is not str or _EXECUTION_TOKEN.fullmatch(value) is None:
        raise ValueError("preparation token must be canonical")
    return value


def _redacted_settings(settings: Settings) -> dict[str, object]:
    payload = serialize_wire_settings(settings)
    if payload.get("xai_api_key") is not None:
        raise ValueError("preparation settings serialization is not redacted")
    payload = dict(payload)
    payload["inventory_db"] = os.fspath(settings.inventory_db.resolve())
    payload["workflow_db"] = os.fspath(settings.workflow_db.resolve())
    payload["source_archive_dir"] = os.fspath(settings.source_archive_dir.resolve())
    return payload


def _encode_request(
    *,
    path: Path,
    settings: Settings,
    case_id: str,
    started_at: datetime,
    preparation_token: str,
    run_token: str,
) -> bytes:
    if (
        not isinstance(path, Path)
        or type(case_id) is not str
        or not case_id.startswith("case_")
        or not case_id.strip() == case_id
    ):
        raise ValueError("invalid preparation identity")
    preparation_token = _canonical_token(preparation_token)
    run_token = _canonical_token(run_token)
    if preparation_token == run_token:
        raise ValueError("preparation and run tokens must differ")
    payload = {
        "case_id": case_id,
        "path": os.fspath(path.resolve()),
        "preparation_token": preparation_token,
        "protocol_version": PREPARATION_PROTOCOL_VERSION,
        "run_token": run_token,
        "settings": _redacted_settings(settings),
        "started_at": _canonical_utc_text(started_at),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > PREPARATION_MAX_MESSAGE_BYTES:
        raise ValueError("preparation request exceeds its bound")
    return encoded


def decode_preparation_request(
    encoded: bytes,
) -> tuple[Path, Settings, str, datetime, str, str]:
    """Decode one exact redacted preparation request in the child."""

    if not encoded or len(encoded) > PREPARATION_MAX_MESSAGE_BYTES:
        raise ValueError("invalid preparation request size")
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if type(payload) is not dict or set(payload) != {
        "case_id",
        "path",
        "preparation_token",
        "protocol_version",
        "run_token",
        "settings",
        "started_at",
    }:
        raise ValueError("invalid preparation request shape")
    if type(payload["protocol_version"]) is not int or payload["protocol_version"] != 1:
        raise ValueError("invalid preparation protocol version")
    if (
        type(payload["case_id"]) is not str
        or not payload["case_id"].startswith("case_")
        or payload["case_id"].strip() != payload["case_id"]
        or type(payload["path"]) is not str
        or not Path(payload["path"]).is_absolute()
        or type(payload["settings"]) is not dict
        or payload["settings"].get("xai_api_key") is not None
        or type(payload["started_at"]) is not str
    ):
        raise ValueError("invalid preparation request values")
    started_at = datetime.fromisoformat(payload["started_at"])
    if started_at.tzinfo is not UTC or started_at.isoformat() != payload["started_at"]:
        raise ValueError("invalid preparation start time")
    preparation_token = _canonical_token(payload["preparation_token"])
    run_token = _canonical_token(payload["run_token"])
    if preparation_token == run_token:
        raise ValueError("preparation tokens collide")
    settings = decode_wire_settings(payload["settings"])
    if settings.xai_api_key is not None:
        raise ValueError("preparation worker received provider credentials")
    return (
        Path(payload["path"]),
        settings,
        payload["case_id"],
        started_at,
        preparation_token,
        run_token,
    )


def encode_preparation_response(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > PREPARATION_MAX_MESSAGE_BYTES:
        raise ValueError("preparation response exceeds its bound")
    return encoded


def _decode_response(encoded: bytes) -> PreparationProcessOutcome:
    if not encoded or len(encoded) > PREPARATION_MAX_MESSAGE_BYTES:
        raise ValueError("invalid preparation response size")
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if type(payload) is not dict or type(payload.get("ok")) is not bool:
        raise ValueError("invalid preparation response shape")
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if canonical != encoded:
        raise ValueError("noncanonical preparation response")
    if payload["ok"]:
        if set(payload) != {"ok"}:
            raise ValueError("invalid preparation success response")
        return PreparationProcessOutcome(True, None)
    if set(payload) != {"error_code", "ok"}:
        raise ValueError("invalid preparation failure response")
    code = payload["error_code"]
    if type(code) is not str or code not in _PREPARATION_ERROR_CODES:
        raise ValueError("invalid preparation failure code")
    return PreparationProcessOutcome(False, code)


def run_preparation_process(
    *,
    path: Path,
    settings: Settings,
    case_id: str,
    started_at: datetime,
    preparation_token: str,
    run_token: str,
    timeout_seconds: float,
    cancel_requested: threading.Event | None = None,
) -> PreparationProcessOutcome:
    """Run preparation, then return only after Task 8 proves its session empty."""

    request = _encode_request(
        path=path,
        settings=settings,
        case_id=case_id,
        started_at=started_at,
        preparation_token=preparation_token,
        run_token=run_token,
    )
    try:
        outcome = run_isolated_process(
            command=_preparation_worker_command(),
            request=request,
            timeout_seconds=timeout_seconds,
            max_response_bytes=PREPARATION_MAX_MESSAGE_BYTES,
            cancel_requested=cancel_requested,
            env=sanitized_worker_environment(),
        )
    except IsolatedProcessCleanupError:
        return PreparationProcessOutcome(False, "PREPARATION_WORKER_CLEANUP_FAILED")
    if outcome.failure == "cancelled":
        return PreparationProcessOutcome(False, "PREPARATION_WORKER_CANCELLED")
    if outcome.failure == "timeout":
        return PreparationProcessOutcome(False, "PREPARATION_WORKER_TIMED_OUT")
    if outcome.failure in {"start", "crash"}:
        return PreparationProcessOutcome(False, "PREPARATION_WORKER_CRASHED")
    if outcome.failure == "protocol" or outcome.response is None:
        return PreparationProcessOutcome(False, "PREPARATION_WORKER_PROTOCOL_INVALID")
    try:
        return _decode_response(outcome.response)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return PreparationProcessOutcome(False, "PREPARATION_WORKER_PROTOCOL_INVALID")
