"""Clean-exec worker bootstrap for the trusted process supervisor."""

from __future__ import annotations

import os
import signal
import sys

_START = b"S"
_BOOTSTRAP_READY = b"R"
_PROTOCOL_FAILURE = 127


def _canonical_positive_integer(value: str, *, label: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"invalid worker bootstrap {label}")
    parsed = int(value)
    if parsed < 1 or str(parsed) != value:
        raise ValueError(f"invalid worker bootstrap {label}")
    return parsed


def _canonical_descriptor(value: str) -> int:
    descriptor = _canonical_positive_integer(value, label="descriptor")
    if descriptor < 3:
        raise ValueError("invalid worker bootstrap descriptor")
    return descriptor


def _parse_arguments() -> tuple[int, int, int, tuple[int, ...], list[str]]:
    expected = ("--ready-fd", "--start-fd", "--supervisor-sid")
    if len(sys.argv) < 9 or (sys.argv[1], sys.argv[3], sys.argv[5]) != expected:
        raise ValueError("invalid worker bootstrap arguments")
    ready_descriptor = _canonical_descriptor(sys.argv[2])
    start_descriptor = _canonical_descriptor(sys.argv[4])
    supervisor_session_id = _canonical_positive_integer(
        sys.argv[6],
        label="supervisor session",
    )
    worker_descriptors: list[int] = []
    argument_index = 7
    while argument_index < len(sys.argv) and sys.argv[argument_index] == "--worker-fd":
        if argument_index + 1 >= len(sys.argv):
            raise ValueError("invalid worker bootstrap arguments")
        worker_descriptors.append(_canonical_descriptor(sys.argv[argument_index + 1]))
        argument_index += 2
    if argument_index >= len(sys.argv) or sys.argv[argument_index] != "--":
        raise ValueError("invalid worker bootstrap arguments")
    all_descriptors = (ready_descriptor, start_descriptor, *worker_descriptors)
    if len(set(all_descriptors)) != len(all_descriptors):
        raise ValueError("worker bootstrap descriptors must be role separated")
    command = sys.argv[argument_index + 1 :]
    if not command or any(not argument for argument in command):
        raise ValueError("invalid worker bootstrap command")
    return (
        ready_descriptor,
        start_descriptor,
        supervisor_session_id,
        tuple(worker_descriptors),
        command,
    )


def _write_all_once(descriptor: int, payload: bytes) -> None:
    if os.write(descriptor, payload) != len(payload):
        raise OSError("worker bootstrap publication was incomplete")


def _run() -> int:
    (
        ready_descriptor,
        start_descriptor,
        supervisor_session_id,
        _worker_descriptors,
        command,
    ) = _parse_arguments()
    signal.pthread_sigmask(signal.SIG_SETMASK, set())
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    process_id = os.getpid()
    if os.getpgrp() != process_id or os.getsid(0) != supervisor_session_id:
        raise RuntimeError("worker bootstrap process identity was not isolated")

    _write_all_once(ready_descriptor, _BOOTSTRAP_READY)
    os.close(ready_descriptor)

    admission = os.read(start_descriptor, 1)
    trailing = os.read(start_descriptor, 1)
    os.close(start_descriptor)
    if admission != _START or trailing:
        raise ValueError("invalid worker bootstrap admission")
    os.execvpe(command[0], command, os.environ)
    return _PROTOCOL_FAILURE


def main() -> None:
    """Run one sanitized bootstrap attempt without exposing protocol details."""

    try:
        return_code = _run()
    except BaseException:
        return_code = _PROTOCOL_FAILURE
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
