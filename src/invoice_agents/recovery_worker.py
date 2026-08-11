"""Fresh isolated-interpreter entry point for expired execution recovery."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# ``-I`` intentionally excludes the script directory and ignores PYTHONPATH.
# Bind imports to the exact package tree containing the trusted worker path so
# an unrelated editable checkout cannot supply a different recovery protocol.
_SOURCE_ROOT = Path(__file__).resolve(strict=True).parents[1]
sys.path.insert(0, os.fspath(_SOURCE_ROOT))

from invoice_agents.config import Settings  # noqa: E402
from invoice_agents.db.store import WorkflowStore  # noqa: E402
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError  # noqa: E402
from invoice_agents.recovery_process import (  # noqa: E402
    RECOVERY_MAX_MESSAGE_BYTES,
    decode_recovery_request,
    encode_recovery_response,
    is_safe_recovery_stop_reason,
)


def _failure(
    error_category: object = ErrorCategory.ORCHESTRATION,
    stop_reason: object = "EXECUTION_RECOVERY_FAILED",
) -> dict[str, object]:
    if type(error_category) is not ErrorCategory or not is_safe_recovery_stop_reason(stop_reason):
        error_category = ErrorCategory.ORCHESTRATION
        stop_reason = "EXECUTION_RECOVERY_WORKER_PROTOCOL_INVALID"
    return {
        "error_category": error_category.value,
        "ok": False,
        "stop_reason": stop_reason,
    }


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
        response: dict[str, object] = {"ok": True} if _recover(settings, scan_at) else _failure()
    except InvoiceAgentsError as exc:
        response = _failure(exc.category, exc.stop_reason)
    except BaseException:
        response = _failure()
    encoded = encode_recovery_response(response)
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
