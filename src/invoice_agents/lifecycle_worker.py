"""Fresh-interpreter entry point for the complete provider/team lifecycle."""

from __future__ import annotations

import asyncio
import fcntl
import os
import stat
import struct
import sys
from typing import Any

from pydantic import SecretStr

from invoice_agents.lifecycle_process import (
    LIFECYCLE_MAX_CREDENTIAL_BYTES,
    LIFECYCLE_MAX_MESSAGE_BYTES,
    decode_lifecycle_request,
    encode_lifecycle_response,
)


def _failure() -> dict[str, Any]:
    return {"error_code": "LIFECYCLE_FAILED", "ok": False}


def _disable_core_dumps() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        return


def _zero_buffer(buffer: bytearray) -> None:
    for index in range(len(buffer)):
        buffer[index] = 0


def _read_exact(descriptor: int, buffer: bytearray) -> None:
    view = memoryview(buffer)
    offset = 0
    try:
        while offset < len(view):
            received = os.readv(descriptor, [view[offset:]])
            if received <= 0 or received > len(view) - offset:
                raise ValueError("incomplete private credential frame")
            offset += received
    finally:
        view.release()


def _read_private_credential(descriptor: int) -> bytearray:
    """Read one exact bounded frame and close the only readable endpoint."""

    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("invalid private credential descriptor")
    header = bytearray(4)
    credential = bytearray()
    completed = False
    close_error: BaseException | None = None
    try:
        flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        if flags & os.O_ACCMODE != os.O_RDONLY or not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
            raise ValueError("invalid private credential transport")
        _read_exact(descriptor, header)
        (credential_size,) = struct.unpack_from("!I", header)
        if credential_size < 1 or credential_size > LIFECYCLE_MAX_CREDENTIAL_BYTES:
            raise ValueError("invalid private credential size")
        credential = bytearray(credential_size)
        _read_exact(descriptor, credential)
        completed = True
    finally:
        try:
            os.close(descriptor)
        except BaseException as exc:
            close_error = exc
        _zero_buffer(header)
        if not completed or close_error is not None:
            _zero_buffer(credential)
    if close_error is not None:
        raise close_error
    return credential


def main() -> None:
    """Read one private-key request and emit no provider-controlled data."""

    _disable_core_dumps()
    stderr_descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(stderr_descriptor, sys.stderr.fileno())
    os.close(stderr_descriptor)
    request = sys.stdin.buffer.read(LIFECYCLE_MAX_MESSAGE_BYTES + 1)
    credential = bytearray()
    try:
        mode, settings, claim, started_at, credential_fd = decode_lifecycle_request(request)
        credential = _read_private_credential(credential_fd)
        provider_key = credential.decode("utf-8", errors="strict")
        if not provider_key.strip():
            raise ValueError("private credential is empty")
        settings = settings.model_copy(update={"xai_api_key": SecretStr(provider_key)})
        provider_key = ""
        for index in range(len(credential)):
            credential[index] = 0

        from invoice_agents.orchestration import (
            _resume_case_in_process,
            _run_prepared_case_in_process,
        )

        if mode == "process":
            asyncio.run(
                _run_prepared_case_in_process(
                    claim.case_id,
                    started_at,
                    settings,
                    claim=claim,
                    terminal_writes_in_process=True,
                )
            )
        else:
            asyncio.run(
                _resume_case_in_process(
                    claim.case_id,
                    settings,
                    claim=claim,
                    terminal_writes_in_process=True,
                )
            )
        response: dict[str, Any] = {"ok": True}
    except BaseException:
        response = _failure()
    finally:
        for index in range(len(credential)):
            credential[index] = 0
    try:
        encoded = encode_lifecycle_response(response)
    except BaseException:
        encoded = encode_lifecycle_response(_failure())
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
