"""Fail-closed isolated process boundary for expired execution recovery."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from invoice_agents.config import Settings
from invoice_agents.errors import ErrorCategory
from invoice_agents.isolated_process import (
    IsolatedProcessCleanupError,
    ProcessCancellation,
    run_isolated_process,
    sanitized_worker_environment,
)

RECOVERY_PROTOCOL_VERSION = 1
RECOVERY_MAX_MESSAGE_BYTES = 65_536
RECOVERY_CONTROLLER_STOP_REASONS = frozenset(
    {
        "EXECUTION_RECOVERY_WORKER_CANCELLED",
        "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
        "EXECUTION_RECOVERY_WORKER_CRASHED",
        "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID",
        "EXECUTION_RECOVERY_WORKER_TIMED_OUT",
    }
)
_SAFE_STOP_REASON = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_STORE_FIELDS = frozenset(
    {
        "due_date_tolerance_days",
        "inventory_db",
        "review_threshold_amount",
        "review_threshold_currency",
        "review_threshold_effective_date",
        "sqlite_journal_mode",
        "workflow_db",
    }
)


@dataclass(frozen=True, slots=True)
class RecoveryProcessOutcome:
    """One validated result observed only after isolated cleanup is proven."""

    acknowledged: bool
    error_category: ErrorCategory | None
    stop_reason: str | None

    def __post_init__(self) -> None:
        if type(self.acknowledged) is not bool:
            raise ValueError("recovery acknowledgement must be an exact boolean")
        if self.acknowledged:
            if self.error_category is not None or self.stop_reason is not None:
                raise ValueError("acknowledged recovery cannot carry error metadata")
            return
        if type(self.error_category) is not ErrorCategory:
            raise ValueError("failed recovery must carry one exact error category")
        if not is_safe_recovery_stop_reason(self.stop_reason):
            raise ValueError("failed recovery must carry one safe stop reason")


def is_safe_recovery_stop_reason(value: object) -> bool:
    """Return whether a stop reason is safe for the bounded child protocol."""

    return type(value) is str and _SAFE_STOP_REASON.fullmatch(value) is not None


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate recovery protocol key")
        payload[key] = value
    return payload


def _reject_constant(_value: str) -> object:
    raise ValueError("non-finite recovery protocol number")


def _canonical_scan_at(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise ValueError("recovery scan time must be canonical UTC")
    encoded = value.isoformat()
    if datetime.fromisoformat(encoded) != value:
        raise ValueError("recovery scan time must be canonical UTC")
    return encoded


def _canonical_database_path(value: Path) -> str:
    if type(value) is not Path:
        value = Path(value)
    resolved = value.resolve()
    if not resolved.is_absolute():
        raise ValueError("recovery database path must be absolute")
    return os.fspath(resolved)


def _store_payload(settings: Settings) -> dict[str, object]:
    return {
        "due_date_tolerance_days": settings.due_date_tolerance_days,
        "inventory_db": _canonical_database_path(settings.inventory_db),
        "review_threshold_amount": str(settings.review_threshold_amount),
        "review_threshold_currency": settings.review_threshold_currency,
        "review_threshold_effective_date": (settings.review_threshold_effective_date.isoformat()),
        "sqlite_journal_mode": settings.sqlite_journal_mode,
        "workflow_db": _canonical_database_path(settings.workflow_db),
    }


def _encode_request(*, settings: Settings, scan_at: datetime) -> bytes:
    payload = {
        "protocol_version": RECOVERY_PROTOCOL_VERSION,
        "scan_at": _canonical_scan_at(scan_at),
        "store": _store_payload(settings),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > RECOVERY_MAX_MESSAGE_BYTES:
        raise ValueError("recovery request exceeds its bound")
    return encoded


def _decode_store(payload: object) -> Settings:
    if type(payload) is not dict or set(payload) != _STORE_FIELDS:
        raise ValueError("invalid recovery store settings shape")
    due_date_tolerance_days = payload["due_date_tolerance_days"]
    inventory_db = payload["inventory_db"]
    workflow_db = payload["workflow_db"]
    journal_mode = payload["sqlite_journal_mode"]
    threshold_amount = payload["review_threshold_amount"]
    threshold_currency = payload["review_threshold_currency"]
    threshold_date = payload["review_threshold_effective_date"]
    if (
        type(due_date_tolerance_days) is not int
        or type(inventory_db) is not str
        or type(workflow_db) is not str
        or type(journal_mode) is not str
        or type(threshold_amount) is not str
        or type(threshold_currency) is not str
        or type(threshold_date) is not str
    ):
        raise ValueError("invalid recovery store settings values")
    paths: dict[str, Path] = {}
    for field, raw_path in (
        ("inventory_db", inventory_db),
        ("workflow_db", workflow_db),
    ):
        path = Path(raw_path)
        if not path.is_absolute() or os.fspath(path.resolve()) != raw_path:
            raise ValueError("recovery database path is not canonical and absolute")
        paths[field] = path
    try:
        amount = Decimal(threshold_amount)
        effective_date = date.fromisoformat(threshold_date)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid recovery business policy settings") from exc
    if (
        not amount.is_finite()
        or str(amount) != threshold_amount
        or effective_date.isoformat() != threshold_date
    ):
        raise ValueError("noncanonical recovery business policy settings")
    native = {
        "due_date_tolerance_days": due_date_tolerance_days,
        "inventory_db": paths["inventory_db"],
        "review_threshold_amount": amount,
        "review_threshold_currency": threshold_currency,
        "review_threshold_effective_date": effective_date,
        "sqlite_journal_mode": journal_mode,
        "workflow_db": paths["workflow_db"],
        "xai_api_key": None,
        "ui_session_secret": None,
    }
    settings = Settings.model_validate(native, strict=True)
    if settings.xai_api_key is not None or settings.ui_session_secret is not None:
        raise ValueError("recovery worker settings acquired excess authority")
    if _store_payload(settings) != payload:
        raise ValueError("recovery settings changed during strict construction")
    return settings


def decode_recovery_request(encoded: bytes) -> tuple[Settings, datetime]:
    """Decode one exact least-authority recovery request in the child."""

    if type(encoded) is not bytes or not encoded or len(encoded) > RECOVERY_MAX_MESSAGE_BYTES:
        raise ValueError("invalid recovery request size")
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if type(payload) is not dict or set(payload) != {
        "protocol_version",
        "scan_at",
        "store",
    }:
        raise ValueError("invalid recovery request shape")
    if (
        type(payload["protocol_version"]) is not int
        or payload["protocol_version"] != RECOVERY_PROTOCOL_VERSION
        or type(payload["scan_at"]) is not str
    ):
        raise ValueError("invalid recovery request values")
    scan_at = datetime.fromisoformat(payload["scan_at"])
    if scan_at.tzinfo is not UTC or scan_at.isoformat() != payload["scan_at"]:
        raise ValueError("invalid recovery scan time")
    settings = _decode_store(payload["store"])
    canonical = _encode_request(settings=settings, scan_at=scan_at)
    if canonical != encoded:
        raise ValueError("noncanonical recovery request")
    return settings, scan_at


def encode_recovery_response(payload: dict[str, object]) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > RECOVERY_MAX_MESSAGE_BYTES:
        raise ValueError("recovery response exceeds its bound")
    return encoded


def _decode_response(encoded: bytes) -> RecoveryProcessOutcome:
    if type(encoded) is not bytes or not encoded or len(encoded) > RECOVERY_MAX_MESSAGE_BYTES:
        raise ValueError("invalid recovery response size")
    payload = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if canonical != encoded:
        raise ValueError("noncanonical recovery response")
    if type(payload) is not dict or type(payload.get("ok")) is not bool:
        raise ValueError("invalid recovery response shape")
    if payload == {"ok": True}:
        return RecoveryProcessOutcome(True, None, None)
    if (
        set(payload) == {"error_category", "ok", "stop_reason"}
        and payload["ok"] is False
        and type(payload["error_category"]) is str
        and is_safe_recovery_stop_reason(payload["stop_reason"])
    ):
        try:
            error_category = ErrorCategory(payload["error_category"])
        except ValueError as exc:
            raise ValueError("invalid recovery response error category") from exc
        if payload["stop_reason"] in RECOVERY_CONTROLLER_STOP_REASONS:
            raise ValueError("child response claimed a controller-owned recovery outcome")
        return RecoveryProcessOutcome(
            False,
            error_category,
            payload["stop_reason"],
        )
    raise ValueError("invalid recovery response value")


def _recovery_worker_command() -> list[str]:
    if type(sys.executable) is not str or not sys.executable:
        raise RuntimeError("recovery worker interpreter path is unavailable")
    executable_path = Path(sys.executable)
    if (
        not executable_path.is_absolute()
        or not executable_path.is_file()
        or not os.access(executable_path, os.X_OK)
    ):
        raise RuntimeError("recovery worker interpreter path is invalid")
    # Preserve the exact launcher identity.  Resolving a virtual-environment
    # symlink selects its base interpreter and silently discards the venv's
    # dependency environment under ``-I``.
    executable = os.fspath(executable_path)
    worker = os.fspath(Path(__file__).with_name("recovery_worker.py").resolve(strict=True))
    return [executable, "-I", worker]


def run_recovery_process(
    *,
    settings: Settings,
    scan_at: datetime,
    timeout_seconds: float,
    cancel_requested: ProcessCancellation | None = None,
) -> RecoveryProcessOutcome:
    """Recover expired executions and return only after process cleanup is proven."""

    request = _encode_request(settings=settings, scan_at=scan_at)
    try:
        outcome = run_isolated_process(
            command=_recovery_worker_command(),
            request=request,
            timeout_seconds=timeout_seconds,
            max_response_bytes=RECOVERY_MAX_MESSAGE_BYTES,
            cancel_requested=cancel_requested,
            env=sanitized_worker_environment(),
        )
    except IsolatedProcessCleanupError:
        return RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
        )
    if outcome.failure == "cancelled":
        return RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_CANCELLED",
        )
    if outcome.failure == "timeout":
        return RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_TIMED_OUT",
        )
    if outcome.failure in {"start", "crash"}:
        return RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_CRASHED",
        )
    if outcome.failure == "protocol" or outcome.response is None:
        return RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID",
        )
    if outcome.failure is not None:
        return RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID",
        )
    try:
        return _decode_response(outcome.response)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return RecoveryProcessOutcome(
            False,
            ErrorCategory.ORCHESTRATION,
            "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID",
        )
