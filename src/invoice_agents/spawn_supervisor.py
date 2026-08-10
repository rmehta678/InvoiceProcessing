"""Minimal exec supervisor for isolated worker process publication."""

from __future__ import annotations

import os
import signal
import sys


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0 or written > len(payload) - offset:
            raise OSError("isolated supervisor process publication was incomplete")
        offset += written


def main() -> None:
    """Publish the exact session leader, normalize its signal mask, then exec."""

    if len(sys.argv) < 3:
        raise SystemExit(127)
    raw_descriptor = sys.argv[1]
    if not raw_descriptor.isascii() or not raw_descriptor.isdigit():
        raise SystemExit(127)
    publication_descriptor = int(raw_descriptor)
    if publication_descriptor < 3 or str(publication_descriptor) != raw_descriptor:
        raise SystemExit(127)
    command = sys.argv[2:]
    if not command or any(not argument for argument in command):
        raise SystemExit(127)

    signal.pthread_sigmask(signal.SIG_SETMASK, set())
    try:
        _write_all(publication_descriptor, f"{os.getpid()}\n".encode("ascii"))
    finally:
        os.close(publication_descriptor)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
