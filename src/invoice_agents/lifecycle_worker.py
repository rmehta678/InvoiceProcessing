"""Fresh-interpreter entry point for the complete provider/team lifecycle."""

from __future__ import annotations

import asyncio
import os
import socket
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


def _read_private_credential(descriptor: int) -> bytearray:
    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("invalid private credential descriptor")
    credential = bytearray(LIFECYCLE_MAX_CREDENTIAL_BYTES + 1)
    transport: socket.socket | None = None
    try:
        transport = socket.socket(fileno=descriptor)
        if (
            transport.family != socket.AF_UNIX
            or transport.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_DGRAM
        ):
            raise ValueError("invalid private credential transport")
        received = transport.recv_into(credential, LIFECYCLE_MAX_CREDENTIAL_BYTES + 1)
    finally:
        if transport is None:
            os.close(descriptor)
        else:
            transport.close()
    if received < 1 or received > LIFECYCLE_MAX_CREDENTIAL_BYTES:
        for index in range(len(credential)):
            credential[index] = 0
        raise ValueError("invalid private credential size")
    del credential[received:]
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
