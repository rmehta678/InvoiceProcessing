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
    from invoice_agents.db.migration_process import _protocol_failure_payload

    return {
        "ok": False,
        "error": _protocol_failure_payload(),
    }


def _safe_expected_failure(exc: InvoiceAgentsError) -> dict[str, Any]:
    from invoice_agents.db.migration_process import _encode_expected_worker_failure

    error = _encode_expected_worker_failure(
        category=exc.category,
        stop_reason=exc.stop_reason,
        details=exc.details,
    )
    if error is None:
        return _protocol_failure()
    return {
        "ok": False,
        "error": error,
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
