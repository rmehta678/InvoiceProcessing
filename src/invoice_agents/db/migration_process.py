"""Bounded subprocess boundary for descriptor-isolated SQLite migrations."""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind
from invoice_agents.errors import DatabaseVerificationError, ErrorCategory

MIGRATION_WORKER_PROTOCOL_VERSION = 1
MIGRATION_WORKER_MAX_MESSAGE_BYTES = 65_536
MIGRATION_WORKER_TIMEOUT_SECONDS = 120.0
_WORKER_SHUTDOWN_SECONDS = 2.0
_WORKER_POLL_SECONDS = 0.05
_SERIALIZED_SETTINGS_FIELDS = frozenset(
    {
        "xai_api_key",
        "inventory_db",
        "workflow_db",
        "sqlite_journal_mode",
        "source_archive_dir",
        "source_max_bytes",
        "pdf_max_pages",
        "pdf_parse_timeout_seconds",
        "pdf_worker_cpu_seconds",
        "pdf_worker_memory_bytes",
        "pdf_worker_result_max_bytes",
        "review_threshold_amount",
        "review_threshold_currency",
        "review_threshold_effective_date",
        "due_date_tolerance_days",
        "review_age_amber_hours",
        "max_messages",
        "model_timeout_seconds",
        "transient_retries",
        "case_concurrency",
        "log_level",
    }
)


def _worker_command() -> list[str]:
    return [sys.executable, "-m", "invoice_agents.db.migration_worker"]


def _serialize_settings(settings: Settings | None) -> dict[str, object] | None:
    if settings is None:
        return None
    return {
        "xai_api_key": None,
        "inventory_db": os.fspath(settings.inventory_db),
        "workflow_db": os.fspath(settings.workflow_db),
        "sqlite_journal_mode": settings.sqlite_journal_mode,
        "source_archive_dir": os.fspath(settings.source_archive_dir),
        "source_max_bytes": settings.source_max_bytes,
        "pdf_max_pages": settings.pdf_max_pages,
        "pdf_parse_timeout_seconds": settings.pdf_parse_timeout_seconds,
        "pdf_worker_cpu_seconds": settings.pdf_worker_cpu_seconds,
        "pdf_worker_memory_bytes": settings.pdf_worker_memory_bytes,
        "pdf_worker_result_max_bytes": settings.pdf_worker_result_max_bytes,
        "review_threshold_amount": str(settings.review_threshold_amount),
        "review_threshold_currency": settings.review_threshold_currency,
        "review_threshold_effective_date": settings.review_threshold_effective_date.isoformat(),
        "due_date_tolerance_days": settings.due_date_tolerance_days,
        "review_age_amber_hours": settings.review_age_amber_hours,
        "max_messages": settings.max_messages,
        "model_timeout_seconds": settings.model_timeout_seconds,
        "transient_retries": settings.transient_retries,
        "case_concurrency": settings.case_concurrency,
        "log_level": settings.log_level,
    }


def _protocol_error(stop_reason: str, message: str) -> DatabaseVerificationError:
    return DatabaseVerificationError(
        ErrorCategory.DATABASE,
        message,
        stop_reason=stop_reason,
    )


def _encode_request(
    path: Path,
    kind: DatabaseKind,
    settings: Settings | None,
) -> bytes:
    payload = {
        "protocol_version": MIGRATION_WORKER_PROTOCOL_VERSION,
        "path": os.fspath(path),
        "kind": kind.value,
        "settings": _serialize_settings(settings),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    if len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker request exceeded its bounded protocol",
        )
    return encoded


def _safe_details(value: object, *, depth: int = 0) -> object:
    if depth > 6:
        raise ValueError("worker details are too deeply nested")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("worker detail number is not finite")
        return value
    if isinstance(value, str):
        if len(value) > 4_096:
            raise ValueError("worker detail string is too large")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise ValueError("worker detail list is too large")
        return [_safe_details(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 256:
            raise ValueError("worker detail mapping is too large")
        safe: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise ValueError("worker detail key is invalid")
            lowered = key.casefold().replace("-", "_")
            if any(token in lowered for token in ("secret", "token", "password", "api_key")):
                safe[key] = "[REDACTED]"
            else:
                safe[key] = _safe_details(item, depth=depth + 1)
        return safe
    raise ValueError("worker details contain an unsupported value")


def _decode_response(encoded: bytes) -> list[int]:
    if not encoded or len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        ) from exc
    if not isinstance(payload, dict) or type(payload.get("ok")) is not bool:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    if payload["ok"]:
        if set(payload) != {"ok", "applied"}:
            raise _protocol_error(
                "MIGRATION_WORKER_PROTOCOL_INVALID",
                "database migration worker returned an invalid bounded response",
            )
        applied = payload["applied"]
        if (
            not isinstance(applied, list)
            or len(applied) > 1_000
            or any(type(version) is not int or version < 1 for version in applied)
        ):
            raise _protocol_error(
                "MIGRATION_WORKER_PROTOCOL_INVALID",
                "database migration worker returned an invalid bounded response",
            )
        return applied
    if set(payload) != {"ok", "error"} or not isinstance(payload["error"], dict):
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    error = payload["error"]
    if set(error) != {"category", "message", "stop_reason", "details"}:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    category = error["category"]
    message = error["message"]
    stop_reason = error["stop_reason"]
    if (
        not isinstance(category, str)
        or not isinstance(message, str)
        or not 1 <= len(message) <= 4_096
        or not isinstance(stop_reason, str)
        or not 1 <= len(stop_reason) <= 256
    ):
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    try:
        error_category = ErrorCategory(category)
    except ValueError as exc:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        ) from exc
    try:
        details = _safe_details(error["details"])
    except ValueError as exc:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        ) from exc
    return_error = DatabaseVerificationError(
        error_category,
        message,
        stop_reason=stop_reason,
        details=details if isinstance(details, dict) else None,
    )
    raise return_error


def _stop_worker(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_WORKER_SHUTDOWN_SECONDS)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.wait()


def _read_bounded_worker_response(process: subprocess.Popen[bytes]) -> bytes:
    """Read at most one capped response while enforcing the whole-worker deadline."""

    stdout = process.stdout
    if stdout is None:
        raise _protocol_error(
            "MIGRATION_WORKER_PROTOCOL_INVALID",
            "database migration worker returned an invalid bounded response",
        )
    descriptor = stdout.fileno()
    os.set_blocking(descriptor, False)
    deadline = time.monotonic() + MIGRATION_WORKER_TIMEOUT_SECONDS
    response = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _protocol_error(
                    "MIGRATION_WORKER_TIMEOUT",
                    "database migration worker exceeded its execution deadline",
                )
            if not selector.select(min(remaining, _WORKER_POLL_SECONDS)):
                if process.poll() is not None:
                    raise _protocol_error(
                        "MIGRATION_WORKER_CRASHED",
                        "database migration worker exited before returning a result",
                    )
                continue
            try:
                chunk = os.read(
                    descriptor,
                    min(8_192, MIGRATION_WORKER_MAX_MESSAGE_BYTES + 1 - len(response)),
                )
            except BlockingIOError:
                continue
            except OSError as exc:
                raise _protocol_error(
                    "MIGRATION_WORKER_PROTOCOL_INVALID",
                    "database migration worker returned an invalid bounded response",
                ) from exc
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
                raise _protocol_error(
                    "MIGRATION_WORKER_PROTOCOL_INVALID",
                    "database migration worker returned an invalid bounded response",
                )
    remaining = deadline - time.monotonic()
    if process.poll() is None:
        if remaining <= 0:
            raise _protocol_error(
                "MIGRATION_WORKER_TIMEOUT",
                "database migration worker exceeded its execution deadline",
            )
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise _protocol_error(
                "MIGRATION_WORKER_TIMEOUT",
                "database migration worker exceeded its execution deadline",
            ) from exc
    return bytes(response)


def run_migration_in_subprocess(
    path: Path,
    kind: DatabaseKind,
    *,
    settings: Settings | None,
) -> list[int]:
    """Run the complete migration in a fresh interpreter owning every SQLite descriptor."""

    if settings is not None:
        settings.assert_delete_journal_mode()
    request = _encode_request(path, kind, settings)
    try:
        with tempfile.TemporaryFile() as request_stream:
            request_stream.write(request)
            request_stream.seek(0)
            process = subprocess.Popen(
                _worker_command(),
                stdin=request_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except OSError as exc:
        raise _protocol_error(
            "MIGRATION_WORKER_START_FAILED",
            "database migration worker could not be started",
        ) from exc
    try:
        stdout = _read_bounded_worker_response(process)
        if process.returncode != 0:
            raise _protocol_error(
                "MIGRATION_WORKER_CRASHED",
                "database migration worker exited before returning a result",
            )
        return _decode_response(stdout)
    except BaseException:
        _stop_worker(process)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()


def decode_worker_request(encoded: bytes) -> tuple[Path, DatabaseKind, Settings | None]:
    """Validate the complete child request without using ambient configuration values."""

    if not encoded or len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise ValueError("invalid worker request size")
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or set(payload) != {
        "protocol_version",
        "path",
        "kind",
        "settings",
    }:
        raise ValueError("invalid worker request shape")
    if payload["protocol_version"] != MIGRATION_WORKER_PROTOCOL_VERSION:
        raise ValueError("invalid worker protocol version")
    if not isinstance(payload["path"], str) or not 1 <= len(payload["path"]) <= 16_384:
        raise ValueError("invalid worker path")
    if not isinstance(payload["kind"], str):
        raise ValueError("invalid worker database kind")
    kind = DatabaseKind(payload["kind"])
    settings_payload = payload["settings"]
    if settings_payload is None:
        settings = None
    elif isinstance(settings_payload, dict):
        if set(settings_payload) != _SERIALIZED_SETTINGS_FIELDS:
            raise ValueError("invalid worker settings shape")
        settings = Settings(**settings_payload)
    else:
        raise ValueError("invalid worker settings")
    return Path(payload["path"]), kind, settings


def encode_worker_response(payload: dict[str, Any]) -> bytes:
    """Encode one already-sanitized child response under the channel cap."""

    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    if len(encoded) > MIGRATION_WORKER_MAX_MESSAGE_BYTES:
        raise ValueError("worker response exceeded protocol bound")
    return encoded
