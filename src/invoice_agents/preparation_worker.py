"""Fresh-interpreter entry point for claimed source preparation."""

from __future__ import annotations

import os
import sys
from typing import Any

from invoice_agents.preparation_process import (
    PREPARATION_MAX_MESSAGE_BYTES,
    decode_preparation_request,
    encode_preparation_response,
)


def _failure() -> dict[str, Any]:
    return {"error_code": "PREPARATION_FAILED", "ok": False}


def main() -> None:
    """Run one redacted request and emit one traceback-free acknowledgement."""

    stderr_descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(stderr_descriptor, sys.stderr.fileno())
    os.close(stderr_descriptor)
    request = sys.stdin.buffer.read(PREPARATION_MAX_MESSAGE_BYTES + 1)
    try:
        (
            path,
            settings,
            case_id,
            started_at,
            preparation_token,
            run_token,
        ) = decode_preparation_request(request)
        from invoice_agents.orchestration import _PreparationFailure, _prepare_case

        prepared = _prepare_case(
            path,
            settings,
            retain_execution_claim=True,
            case_id=case_id,
            started_at=started_at,
            preparation_token=preparation_token,
            run_token=run_token,
        )
        response: dict[str, Any] = (
            _failure() if isinstance(prepared, _PreparationFailure) else {"ok": True}
        )
    except BaseException:
        response = _failure()
    try:
        encoded = encode_preparation_response(response)
    except BaseException:
        encoded = encode_preparation_response(_failure())
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
