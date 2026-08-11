"""Fresh isolated-interpreter entry point for expired execution recovery."""

from __future__ import annotations

import os
import sys
from datetime import datetime

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.recovery_process import (
    RECOVERY_MAX_MESSAGE_BYTES,
    decode_recovery_request,
    encode_recovery_response,
)


def _failure() -> dict[str, object]:
    return {"error_code": "EXECUTION_RECOVERY_FAILED", "ok": False}


def _recover(settings: Settings, scan_at: datetime) -> bool:
    store = WorkflowStore(settings)
    store.recover_expired_executions(now=scan_at)
    return not store.unrecovered_execution_case_ids(checked_at=scan_at)


def main() -> None:
    """Run one recovery pass and emit only a fixed, traceback-free result."""

    stderr_descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(stderr_descriptor, sys.stderr.fileno())
    finally:
        os.close(stderr_descriptor)
    request = sys.stdin.buffer.read(RECOVERY_MAX_MESSAGE_BYTES + 1)
    try:
        settings, scan_at = decode_recovery_request(request)
        response: dict[str, object] = (
            {"ok": True} if _recover(settings, scan_at) else _failure()
        )
    except BaseException:
        response = _failure()
    encoded = encode_recovery_response(response)
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
