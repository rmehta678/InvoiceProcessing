"""Disposable descriptor-ownership worker for one result artifact read."""

from __future__ import annotations

import sys

from invoice_agents.ui.routes import (
    _RESULT_ARTIFACT_INVALID,
    _RESULT_ARTIFACT_MISSING,
    _RESULT_ARTIFACT_OK,
    RESULT_ARTIFACT_WORKER_MAX_REQUEST_BYTES,
    _decode_result_artifact_worker_request,
    _read_bounded_regular_file,
    _ResultArtifactMissing,
)


def main() -> int:
    encoded = sys.stdin.buffer.read(RESULT_ARTIFACT_WORKER_MAX_REQUEST_BYTES + 1)
    try:
        target = _decode_result_artifact_worker_request(encoded)
        raw = _read_bounded_regular_file(target)
    except _ResultArtifactMissing:
        response = _RESULT_ARTIFACT_MISSING
    except BaseException:
        response = _RESULT_ARTIFACT_INVALID
    else:
        response = _RESULT_ARTIFACT_OK + raw
    try:
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
