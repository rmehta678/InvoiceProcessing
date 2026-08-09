"""Fresh-interpreter entry point for descriptor-isolated database migration."""

from __future__ import annotations

import os
import sys
from typing import Any

from invoice_agents.db.migration_process import (
    MIGRATION_WORKER_MAX_MESSAGE_BYTES,
    decode_worker_request,
    encode_worker_response,
)
from invoice_agents.errors import InvoiceAgentsError


def _protocol_failure() -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "category": "DATABASE",
            "message": "database migration worker protocol was invalid",
            "stop_reason": "MIGRATION_WORKER_PROTOCOL_INVALID",
            "details": None,
        },
    }


def _safe_expected_failure(exc: InvoiceAgentsError) -> dict[str, Any]:
    from invoice_agents.db.migration_process import _safe_details

    try:
        details = _safe_details(exc.details)
        message = exc.message if 1 <= len(exc.message) <= 4_096 else "database migration failed"
        stop_reason = (
            exc.stop_reason
            if exc.stop_reason is not None and 1 <= len(exc.stop_reason) <= 256
            else "MIGRATION_FAILED"
        )
    except (TypeError, ValueError):
        return _protocol_failure()
    return {
        "ok": False,
        "error": {
            "category": exc.category.value,
            "message": message,
            "stop_reason": stop_reason,
            "details": details,
        },
    }


def main() -> None:
    """Read one bounded request and emit exactly one bounded, traceback-free response."""

    stderr_descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(stderr_descriptor, sys.stderr.fileno())
    os.close(stderr_descriptor)
    request = sys.stdin.buffer.read(MIGRATION_WORKER_MAX_MESSAGE_BYTES + 1)
    try:
        path, kind, settings = decode_worker_request(request)
        from invoice_agents.db.core import _migrate_database_in_process

        response: dict[str, Any] = {
            "ok": True,
            "applied": _migrate_database_in_process(path, kind, settings=settings),
        }
    except InvoiceAgentsError as exc:
        response = _safe_expected_failure(exc)
    except BaseException:
        response = _protocol_failure()
    try:
        encoded = encode_worker_response(response)
    except BaseException:
        encoded = encode_worker_response(_protocol_failure())
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
