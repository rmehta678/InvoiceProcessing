"""Disposable descriptor-ownership worker for one result artifact read."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_WORKER_PATH = Path(__file__).resolve(strict=True)
_PACKAGE_ROOT = _WORKER_PATH.parents[2]
_COLOCATED_ROUTES = _PACKAGE_ROOT / "invoice_agents" / "ui" / "routes.py"
if _COLOCATED_ROUTES.resolve(strict=True) != _WORKER_PATH.with_name("routes.py"):
    raise RuntimeError("result-artifact worker source binding is invalid")
sys.path.insert(0, os.fspath(_PACKAGE_ROOT))

from invoice_agents.ui.routes import (  # noqa: E402
    _RESULT_ARTIFACT_INVALID,
    _RESULT_ARTIFACT_MISSING,
    _RESULT_ARTIFACT_OK,
    RESULT_ARTIFACT_WORKER_MAX_REQUEST_BYTES,
    _decode_result_artifact_worker_request,
    _read_bounded_regular_file,
    _ResultArtifactMissing,
)


def main() -> int:
    try:
        stderr_descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(stderr_descriptor, sys.stderr.fileno())
        finally:
            os.close(stderr_descriptor)
    except BaseException:
        return 1
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
