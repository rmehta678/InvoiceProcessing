"""Independent RED contracts for the Task 9 trusted-supervisor protocol."""

from __future__ import annotations

import asyncio
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pytest

from invoice_agents import isolated_process, spawn_supervisor
from invoice_agents.db import migration_process

_PROTOCOL_WAIT_SECONDS = 2.0
_IMPORT_ISOLATION_LIFECYCLE_SECONDS = 5.0 * _PROTOCOL_WAIT_SECONDS
_TRUSTED_SUPERVISOR_SUCCESS_SECONDS = 9.0 * _PROTOCOL_WAIT_SECONDS


def _install_synthetic_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[..., list[str]],
) -> None:
    monkeypatch.setattr(isolated_process, "_supervisor_command", builder)
    monkeypatch.setattr(
        isolated_process,
        "_declare_worker_descriptors",
        lambda command, _worker_descriptors: command,
    )


def _remaining_success_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("positive lifecycle exceeded its phase-derived deadline")
    return remaining


@dataclass(frozen=True, slots=True)
class _ProtocolObservation:
    session_identity: tuple[int, int, int] | None
    owner_line: bytes
    status_lines: tuple[bytes, ...]
    worker_receipts: tuple[dict[str, object], ...]
    worker_marker_before_control_eof: bool
    credential_receipt_before_control_eof: bytes | None
    lifetime_open_before_control_eof: bool
    credential_receipt_after_start: bytes | None
    worker_private_reader_closed_before_worker_exit: bool
    lifetime_eof_after_exit: bool
    supervisor_returncode: int | None
    supervisor_stderr: bytes
    supervisor_not_running_before_test_cleanup: bool
    supervisor_reaped_before_test_cleanup: bool
    group_after_public_return: tuple[tuple[int, str], ...] | None
    forced_test_cleanup: bool


@dataclass(frozen=True, slots=True)
class _RejectedControlObservation:
    session_identity: tuple[int, int, int] | None
    owner_line: bytes
    ready_line: bytes
    control_ack_line: bytes | None
    worker_marker_at_control_ack: bool | None
    credential_receipt_at_control_ack: bytes | None
    lifetime_open_at_control_ack: bool | None
    worker_marker_exists: bool
    credential_receipt: bytes | None
    lifetime_eof_after_exit: bool
    supervisor_returncode: int | None
    supervisor_stderr: bytes
    supervisor_not_running_before_test_cleanup: bool
    supervisor_reaped_before_test_cleanup: bool
    group_after_public_return: tuple[tuple[int, str], ...] | None
    forced_test_cleanup: bool


@dataclass(frozen=True, slots=True)
class _PublicationControlObservation:
    owner_receipt: dict[str, object] | None
    action_receipt: dict[str, object] | None
    control_receipt: dict[str, object]
    lifetime_receipt: dict[str, object]
    raised_is_injected: bool
    returned_before_lifetime_release: bool
    controller_joined_after_lifetime: bool
    supervisor_not_running_before_test_cleanup: bool
    supervisor_reaped_before_test_cleanup: bool
    group_after_public_return: tuple[tuple[int, str], ...] | None
    worker_marker_exists: bool
    supervisor_returncode_before_test_cleanup: int | None
    forced_test_cleanup: bool


def _read_line(descriptor: int, *, timeout: float = _PROTOCOL_WAIT_SECONDS) -> bytes:
    """Read one newline-terminated receipt without using a timing sleep as a barrier."""

    deadline = time.monotonic() + timeout
    payload = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(max(0.0, remaining)):
                raise TimeoutError("bounded protocol receipt was not published")
            chunk = os.read(descriptor, 1)
            if not chunk:
                raise EOFError("protocol endpoint reached EOF before a complete receipt")
            payload.extend(chunk)
            if chunk == b"\n":
                return bytes(payload)
            if len(payload) > 4_096:
                raise ValueError("protocol receipt exceeded its independent test bound")


def _read_line_or_error(
    descriptor: int,
    *,
    timeout: float = _PROTOCOL_WAIT_SECONDS,
) -> bytes:
    """Preserve a failed observation so process cleanup can happen before assertions."""

    try:
        return _read_line(descriptor, timeout=timeout)
    except (EOFError, OSError, TimeoutError, ValueError) as exc:
        return f"TEST_READ_ERROR:{type(exc).__name__}:{exc}\n".encode("ascii")


def _endpoint_is_readable(descriptor: int) -> bool:
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        return bool(selector.select(0.0))


def _endpoint_reached_eof(descriptor: int) -> bool:
    if not _endpoint_is_readable(descriptor):
        return False
    try:
        return os.read(descriptor, 1) == b""
    except OSError:
        return False


def _read_json_receipt(
    descriptor: int,
    *,
    timeout: float = _PROTOCOL_WAIT_SECONDS,
) -> dict[str, object]:
    encoded = _read_line_or_error(descriptor, timeout=timeout)
    try:
        value = json.loads(encoded)
    except (UnicodeError, json.JSONDecodeError):
        return {"test_read_error": encoded.decode("ascii", errors="backslashreplace")}
    if type(value) is not dict:
        return {"test_read_error": "worker receipt was not a JSON object"}
    return value


def _wait_for_path(path: Path, *, timeout: float = _PROTOCOL_WAIT_SECONDS) -> None:
    """Wait only for an explicit child-written receipt, never for elapsed-time correctness."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.005)
    raise TimeoutError(f"bounded child receipt was not published: {path}")


def _close_descriptor(descriptor: int) -> None:
    with suppress(OSError):
        os.close(descriptor)


def _process_is_not_running(process_id: int) -> bool:
    completed = subprocess.run(
        ["/bin/ps", "-o", "stat=", "-p", str(process_id)],
        check=False,
        capture_output=True,
        text=True,
        timeout=_PROTOCOL_WAIT_SECONDS,
    )
    if completed.returncode not in {0, 1}:
        return False
    state = completed.stdout.strip()
    if completed.returncode == 1:
        if state:
            raise AssertionError("ps reported an absent PID with a process state")
        return True
    return False


def _session_identity(process_id: int) -> tuple[int, int, int]:
    return process_id, os.getpgid(process_id), os.getsid(process_id)


def _exact_child_was_reaped(process_id: int) -> bool:
    try:
        os.waitpid(process_id, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


def _process_group_snapshot(process_group_id: int) -> tuple[tuple[int, str], ...]:
    """Take one parseable test-cleanup snapshot without trusting leader liveness."""

    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,stat="],
        check=True,
        capture_output=True,
        text=True,
        timeout=_PROTOCOL_WAIT_SECONDS,
    )
    members: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        raw_process_id, raw_group_id, state = fields
        if (
            raw_process_id.isdigit()
            and raw_group_id.isdigit()
            and int(raw_group_id) == process_group_id
        ):
            members.append((int(raw_process_id), state))
    return tuple(sorted(members))


def _cleanup_test_process_group(
    process_group_id: int,
    handle: subprocess.Popen[bytes] | None,
) -> tuple[tuple[int, str], ...]:
    """Snapshot, then unconditionally kill the known test group before reaping its leader."""

    snapshot = _process_group_snapshot(process_group_id)
    with suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGKILL)
    if handle is not None:
        with suppress(subprocess.TimeoutExpired, ChildProcessError):
            handle.wait(timeout=_PROTOCOL_WAIT_SECONDS)
    return snapshot


def _trusted_supervisor_argv(
    *,
    status_descriptor: int,
    control_descriptor: int,
    owner_descriptor: int,
    lifetime_descriptor: int,
    worker_descriptors: tuple[int, ...] = (),
    worker_command: list[str],
) -> list[str]:
    """Literal expected argv, independently constructed from the protocol contract."""

    return [
        os.fspath(Path(sys.executable).resolve()),
        "-I",
        os.fspath(Path(spawn_supervisor.__file__).resolve()),
        "--status-fd",
        str(status_descriptor),
        "--control-fd",
        str(control_descriptor),
        "--owner-fd",
        str(owner_descriptor),
        "--lifetime-fd",
        str(lifetime_descriptor),
        *[value for descriptor in worker_descriptors for value in ("--worker-fd", str(descriptor))],
        "--",
        *worker_command,
    ]


def _worker_with_noninheritance_receipts(
    marker: Path,
    release: Path,
    status_descriptor: int,
    control_descriptor: int,
    owner_descriptor: int,
    lifetime_descriptor: int,
    credential_descriptor: int,
    credential_receipt: Path,
) -> list[str]:
    protocol_identities: dict[int, tuple[int, int, int, int]] = {}
    for descriptor in (
        status_descriptor,
        control_descriptor,
        owner_descriptor,
        lifetime_descriptor,
    ):
        descriptor_status = os.fstat(descriptor)
        protocol_identities[descriptor] = (
            descriptor_status.st_dev,
            descriptor_status.st_ino,
            descriptor_status.st_mode & 0o170000,
            descriptor_status.st_rdev,
        )
    credential_status = os.fstat(credential_descriptor)
    credential_identity = (
        credential_status.st_dev,
        credential_status.st_ino,
        credential_status.st_mode & 0o170000,
        credential_status.st_rdev,
    )
    descendant_code = "\n".join(
        (
            "import json, os, sys",
            "protocol_identities = {int(fd): tuple(value) for fd, value in json.loads(sys.argv[1]).items()}",
            "credential = int(sys.argv[2])",
            "credential_identity = tuple(json.loads(sys.argv[3]))",
            "def same_identity(descriptor, identity):",
            "    try: status = os.fstat(descriptor)",
            "    except OSError: return False",
            "    return (status.st_dev, status.st_ino, status.st_mode & 0o170000, status.st_rdev) == identity",
            "def same_credential(descriptor):",
            "    return same_identity(descriptor, credential_identity)",
            "protocol_inherited = [fd for fd, identity in protocol_identities.items() if same_identity(fd, identity)]",
            "print(json.dumps({'role':'descendant','protocol_inherited':protocol_inherited,'credential_inherited':same_credential(credential)}), flush=True)",
        )
    )
    worker_code = "\n".join(
        (
            "import json, os, subprocess, sys, time",
            "from pathlib import Path",
            "marker = Path(sys.argv[1])",
            "lifetime_descriptor = int(sys.argv[2])",
            "credential_descriptor = int(sys.argv[4])",
            "protocol_identities = {int(fd): tuple(value) for fd, value in json.loads(sys.argv[7]).items()}",
            "def same_identity(descriptor, identity):",
            "    try: status = os.fstat(descriptor)",
            "    except OSError: return False",
            "    return (status.st_dev, status.st_ino, status.st_mode & 0o170000, status.st_rdev) == identity",
            "credential = os.read(credential_descriptor, 4096)",
            "os.close(credential_descriptor)",
            "Path(sys.argv[5]).write_bytes(credential)",
            "marker.write_text(str(os.getpid()), encoding='ascii')",
            "release = Path(sys.argv[6])",
            "while not release.exists(): time.sleep(0.005)",
            "protocol_inherited = [fd for fd, identity in protocol_identities.items() if same_identity(fd, identity)]",
            "print(json.dumps({'role':'worker','lifetime_inherited':lifetime_descriptor in protocol_inherited,'protocol_inherited':protocol_inherited,'credential_closed':True}), flush=True)",
            "subprocess.run(",
            "    [sys.executable, '-I', '-c', sys.argv[3], sys.argv[7], sys.argv[4], sys.argv[8]],",
            "    check=True,",
            "    close_fds=False,",
            ")",
        )
    )
    return [
        os.fspath(Path(sys.executable).resolve()),
        "-I",
        "-c",
        worker_code,
        os.fspath(marker),
        str(lifetime_descriptor),
        descendant_code,
        str(credential_descriptor),
        os.fspath(credential_receipt),
        os.fspath(release),
        json.dumps(protocol_identities, sort_keys=True),
        json.dumps(credential_identity),
    ]


def test_round12_supervisor_command_requires_four_role_separated_descriptors() -> None:
    """Status, control, OWNER publication, and lifetime cannot share endpoints."""

    worker_command = [os.fspath(Path(sys.executable).resolve()), "-I", "-c", "pass"]

    actual = isolated_process._supervisor_command(
        status_descriptor=41,
        control_descriptor=43,
        owner_descriptor=47,
        lifetime_descriptor=53,
        command=worker_command,
    )

    expected = _trusted_supervisor_argv(
        status_descriptor=41,
        control_descriptor=43,
        owner_descriptor=47,
        lifetime_descriptor=53,
        worker_command=worker_command,
    )
    assert actual == expected
    assert len({41, 43, 47, 53}) == 4


def test_round13_launch_declares_every_worker_only_descriptor_before_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trusted parent receives the exact worker-only FD set it retires after spawn."""

    reader, writer = isolated_process.private_pipe_channel()
    worker_descriptor = reader.fileno()
    credential = bytearray(b"round13-declared-worker-fd")
    captured_commands: list[list[str]] = []
    captured_inherited: list[tuple[int, ...]] = []

    def capture_without_spawn(
        _process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        raw_command = args[0] if args else kwargs["args"]
        assert type(raw_command) is list
        captured_commands.append(raw_command)
        captured_inherited.append(tuple(kwargs["pass_fds"]))
        raise OSError("round13 stop after argv capture")

    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        capture_without_spawn,
        raising=False,
    )
    try:
        result = isolated_process.run_isolated_process(
            command=[sys.executable, "-I", "-c", "pass"],
            request=b"{}",
            timeout_seconds=1.0,
            max_response_bytes=32,
            pass_fds=(worker_descriptor,),
            private_input=isolated_process.PrivatePipeInput(
                reader=reader,
                writer=writer,
                payload=credential,
                max_payload_bytes=len(credential),
            ),
            env=isolated_process.sanitized_worker_environment(),
        )
    finally:
        if not reader.closed:
            reader.close()
        if not writer.closed:
            writer.close()

    assert result == isolated_process.IsolatedProcessResult(None, "start")
    assert len(captured_commands) == 1
    command = captured_commands[0]
    expected_entrypoint = Path(spawn_supervisor.__file__).resolve(strict=True)
    assert Path(command[0]).resolve(strict=True) == Path(sys.executable).resolve(strict=True)
    assert command[1] == "-I"
    assert Path(command[2]).is_absolute()
    assert Path(command[2]).resolve(strict=True) == expected_entrypoint
    assert "-m" not in command[:3]
    separator = command.index("--")
    assert command[separator - 2 : separator] == ["--worker-fd", str(worker_descriptor)]
    assert command[:separator].count("--worker-fd") == 1
    assert captured_inherited and captured_inherited[0].count(worker_descriptor) == 1
    assert credential == bytearray(len(credential))


def test_round12_trusted_supervisor_waits_for_exact_start_and_drops_lifetime_fd(
    tmp_path: Path,
) -> None:
    """A child acknowledgement precedes EOF-authorized START and lifetime transfer."""

    status_reader, status_writer = os.pipe()
    control_reader, control_writer = os.pipe()
    owner_reader, owner_writer = os.pipe()
    lifetime_reader, lifetime_writer = os.pipe()
    credential_reader, credential_writer = os.pipe()
    marker = tmp_path / "worker-started"
    release_worker = tmp_path / "release-worker"
    credential_receipt = tmp_path / "worker-credential-received"
    canary = b"round12-exact-control-handshake-canary"
    worker_command = _worker_with_noninheritance_receipts(
        marker,
        release_worker,
        status_writer,
        control_reader,
        owner_writer,
        lifetime_writer,
        credential_reader,
        credential_receipt,
    )
    command = _trusted_supervisor_argv(
        status_descriptor=status_writer,
        control_descriptor=control_reader,
        owner_descriptor=owner_writer,
        lifetime_descriptor=lifetime_writer,
        worker_descriptors=(credential_reader,),
        worker_command=worker_command,
    )
    process: subprocess.Popen[bytes] | None = None
    owner_line = b"TEST_READ_ERROR:supervisor-not-started\n"
    status_lines: list[bytes] = []
    receipts: list[dict[str, object]] = []
    marker_before_control_eof = True
    credential_before_control_eof: bytes | None = b"test-not-observed"
    lifetime_open_before_control_eof = False
    credential_after_start: bytes | None = None
    worker_private_reader_closed_before_exit = False
    lifetime_eof_after_exit = False
    returncode: int | None = None
    stderr = b""
    session_identity: tuple[int, int, int] | None = None
    supervisor_not_running = False
    supervisor_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    forced_cleanup = False
    success_deadline = time.monotonic() + _TRUSTED_SUPERVISOR_SUCCESS_SECONDS
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
            pass_fds=(
                status_writer,
                control_reader,
                owner_writer,
                lifetime_writer,
                credential_reader,
            ),
        )
        session_identity = _session_identity(process.pid)
        for descriptor in (
            status_writer,
            control_reader,
            owner_writer,
            lifetime_writer,
            credential_reader,
        ):
            _close_descriptor(descriptor)
        owner_line = _read_line_or_error(
            owner_reader,
            timeout=_remaining_success_seconds(success_deadline),
        )
        status_lines.append(
            _read_line_or_error(
                status_reader,
                timeout=_remaining_success_seconds(success_deadline),
            )
        )
        with suppress(BrokenPipeError):
            os.write(credential_writer, canary)
        with suppress(BrokenPipeError):
            os.write(control_writer, b"S")
        status_lines.append(
            _read_line_or_error(
                status_reader,
                timeout=_remaining_success_seconds(success_deadline),
            )
        )
        marker_before_control_eof = marker.exists()
        credential_before_control_eof = (
            credential_receipt.read_bytes() if credential_receipt.exists() else None
        )
        lifetime_open_before_control_eof = not _endpoint_is_readable(lifetime_reader)
        _close_descriptor(control_writer)
        status_lines.append(
            _read_line_or_error(
                status_reader,
                timeout=_remaining_success_seconds(success_deadline),
            )
        )
        _wait_for_path(
            marker,
            timeout=_remaining_success_seconds(success_deadline),
        )
        try:
            os.write(credential_writer, b"after-worker-close")
        except BrokenPipeError:
            worker_private_reader_closed_before_exit = True
        _close_descriptor(credential_writer)
        release_worker.write_text("release", encoding="ascii")
        if process.stdout is not None:
            receipts.append(
                _read_json_receipt(
                    process.stdout.fileno(),
                    timeout=_remaining_success_seconds(success_deadline),
                )
            )
            receipts.append(
                _read_json_receipt(
                    process.stdout.fileno(),
                    timeout=_remaining_success_seconds(success_deadline),
                )
            )
        credential_after_start = (
            credential_receipt.read_bytes() if credential_receipt.exists() else None
        )
        try:
            returncode = process.wait(timeout=_remaining_success_seconds(success_deadline))
        except subprocess.TimeoutExpired:
            forced_cleanup = True
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=_PROTOCOL_WAIT_SECONDS)
        lifetime_eof_after_exit = _read_line_or_error(
            lifetime_reader,
            timeout=_remaining_success_seconds(success_deadline),
        ).startswith(b"TEST_READ_ERROR:EOFError:")
        if process.stderr is not None:
            stderr = process.stderr.read()
        supervisor_not_running = _process_is_not_running(process.pid)
        supervisor_reaped = _exact_child_was_reaped(process.pid)
        group_after_return = _process_group_snapshot(process.pid)
    finally:
        release_worker.write_text("release", encoding="ascii")
        if process is not None:
            _cleanup_test_process_group(process.pid, process)
        for descriptor in (
            status_reader,
            status_writer,
            control_reader,
            control_writer,
            owner_reader,
            owner_writer,
            lifetime_reader,
            lifetime_writer,
            credential_reader,
            credential_writer,
        ):
            _close_descriptor(descriptor)
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    observation = _ProtocolObservation(
        session_identity=session_identity,
        owner_line=owner_line,
        status_lines=tuple(status_lines),
        worker_receipts=tuple(receipts),
        worker_marker_before_control_eof=marker_before_control_eof,
        credential_receipt_before_control_eof=credential_before_control_eof,
        lifetime_open_before_control_eof=lifetime_open_before_control_eof,
        credential_receipt_after_start=credential_after_start,
        worker_private_reader_closed_before_worker_exit=(worker_private_reader_closed_before_exit),
        lifetime_eof_after_exit=lifetime_eof_after_exit,
        supervisor_returncode=returncode,
        supervisor_stderr=stderr,
        supervisor_not_running_before_test_cleanup=supervisor_not_running,
        supervisor_reaped_before_test_cleanup=supervisor_reaped,
        group_after_public_return=group_after_return,
        forced_test_cleanup=forced_cleanup,
    )
    assert observation.owner_line == f"OWNER {process.pid}\n".encode("ascii")
    assert observation.status_lines[0] == f"READY {process.pid}\n".encode("ascii")
    assert observation.status_lines[1] == b"CONTROL S\n"
    assert not observation.worker_marker_before_control_eof
    assert observation.credential_receipt_before_control_eof is None
    assert observation.lifetime_open_before_control_eof
    assert observation.status_lines[2].startswith(b"STARTED ")
    assert observation.worker_receipts == (
        {
            "role": "worker",
            "lifetime_inherited": False,
            "protocol_inherited": [],
            "credential_closed": True,
        },
        {
            "role": "descendant",
            "protocol_inherited": [],
            "credential_inherited": False,
        },
    )
    assert observation.credential_receipt_after_start == canary
    assert observation.worker_private_reader_closed_before_worker_exit
    assert observation.lifetime_eof_after_exit
    assert observation.supervisor_returncode == 0
    assert observation.supervisor_stderr == b""
    assert observation.session_identity == (process.pid, process.pid, process.pid)
    assert observation.supervisor_not_running_before_test_cleanup
    assert observation.supervisor_reaped_before_test_cleanup
    assert observation.group_after_public_return == ()
    assert not observation.forced_test_cleanup


def _run_rejected_control_protocol(tmp_path: Path, token: bytes) -> _RejectedControlObservation:
    """Exercise one whole control stream and preserve observations before cleanup."""

    status_reader, status_writer = os.pipe()
    control_reader, control_writer = os.pipe()
    owner_reader, owner_writer = os.pipe()
    lifetime_reader, lifetime_writer = os.pipe()
    credential_reader, credential_writer = os.pipe()
    marker = tmp_path / "invalid-control-worker-started"
    credential_receipt = tmp_path / "invalid-control-credential-received"
    canary = b"round12-invalid-control-credential-canary"
    worker_code = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')",
            "credential = os.read(int(sys.argv[3]), 4096)",
            "Path(sys.argv[2]).write_bytes(credential)",
        )
    )
    worker_command = [
        os.fspath(Path(sys.executable).resolve()),
        "-I",
        "-c",
        worker_code,
        os.fspath(marker),
        os.fspath(credential_receipt),
        str(credential_reader),
    ]
    command = _trusted_supervisor_argv(
        status_descriptor=status_writer,
        control_descriptor=control_reader,
        owner_descriptor=owner_writer,
        lifetime_descriptor=lifetime_writer,
        worker_descriptors=(credential_reader,),
        worker_command=worker_command,
    )
    process: subprocess.Popen[bytes] | None = None
    owner_line = b"TEST_READ_ERROR:supervisor-not-started\n"
    ready_line = b"TEST_READ_ERROR:supervisor-not-started\n"
    control_ack_line: bytes | None = None
    worker_marker_at_control_ack: bool | None = None
    credential_receipt_at_control_ack: bytes | None = None
    lifetime_open_at_control_ack: bool | None = None
    returncode: int | None = None
    stderr = b""
    session_identity: tuple[int, int, int] | None = None
    supervisor_not_running = False
    supervisor_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    forced_cleanup = False
    lifetime_eof = False
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
            pass_fds=(
                status_writer,
                control_reader,
                owner_writer,
                lifetime_writer,
                credential_reader,
            ),
        )
        session_identity = _session_identity(process.pid)
        for descriptor in (
            status_writer,
            control_reader,
            owner_writer,
            lifetime_writer,
            credential_reader,
        ):
            _close_descriptor(descriptor)
        owner_line = _read_line_or_error(owner_reader)
        ready_line = _read_line_or_error(status_reader)
        with suppress(BrokenPipeError):
            os.write(credential_writer, canary)
        _close_descriptor(credential_writer)
        if token.startswith(b"S"):
            with suppress(BrokenPipeError):
                os.write(control_writer, b"S")
            control_ack_line = _read_line_or_error(status_reader)
            worker_marker_at_control_ack = marker.exists()
            credential_receipt_at_control_ack = (
                credential_receipt.read_bytes() if credential_receipt.exists() else None
            )
            lifetime_open_at_control_ack = not _endpoint_is_readable(lifetime_reader)
            if len(token) > 1:
                with suppress(BrokenPipeError):
                    os.write(control_writer, token[1:])
        elif token:
            with suppress(BrokenPipeError):
                os.write(control_writer, token)
        _close_descriptor(control_writer)
        try:
            returncode = process.wait(timeout=_PROTOCOL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            forced_cleanup = True
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=_PROTOCOL_WAIT_SECONDS)
        lifetime_eof = _endpoint_reached_eof(lifetime_reader)
        if process.stderr is not None:
            stderr = process.stderr.read()
        supervisor_not_running = _process_is_not_running(process.pid)
        supervisor_reaped = _exact_child_was_reaped(process.pid)
        group_after_return = _process_group_snapshot(process.pid)
    finally:
        if process is not None:
            _cleanup_test_process_group(process.pid, process)
        for descriptor in (
            status_reader,
            status_writer,
            control_reader,
            control_writer,
            owner_reader,
            owner_writer,
            lifetime_reader,
            lifetime_writer,
            credential_reader,
            credential_writer,
        ):
            _close_descriptor(descriptor)
        if process is not None and process.stderr is not None:
            process.stderr.close()

    return _RejectedControlObservation(
        session_identity=session_identity,
        owner_line=owner_line,
        ready_line=ready_line,
        control_ack_line=control_ack_line,
        worker_marker_at_control_ack=worker_marker_at_control_ack,
        credential_receipt_at_control_ack=credential_receipt_at_control_ack,
        lifetime_open_at_control_ack=lifetime_open_at_control_ack,
        worker_marker_exists=marker.exists(),
        credential_receipt=(
            credential_receipt.read_bytes() if credential_receipt.exists() else None
        ),
        lifetime_eof_after_exit=lifetime_eof,
        supervisor_returncode=returncode,
        supervisor_stderr=stderr,
        supervisor_not_running_before_test_cleanup=supervisor_not_running,
        supervisor_reaped_before_test_cleanup=supervisor_reaped,
        group_after_public_return=group_after_return,
        forced_test_cleanup=forced_cleanup,
    )


@pytest.mark.parametrize("token", [b"", b"A"], ids=["pre-start-eof", "exact-abort"])
def test_round12_abort_or_pre_start_eof_never_admits_worker_or_credential(
    tmp_path: Path,
    token: bytes,
) -> None:
    """EOF and exact ABORT are clean pre-START retirement, never implicit START."""

    observation = _run_rejected_control_protocol(tmp_path, token)

    assert observation.ready_line.startswith(b"READY ")
    assert observation.owner_line == b"OWNER " + observation.ready_line[len(b"READY ") :]
    owner_process_id = int(observation.owner_line.removeprefix(b"OWNER ").rstrip(b"\n"))
    assert observation.session_identity == (
        owner_process_id,
        owner_process_id,
        owner_process_id,
    )
    assert observation.control_ack_line is None
    assert not observation.worker_marker_exists
    assert observation.credential_receipt is None
    assert observation.lifetime_eof_after_exit
    assert observation.supervisor_returncode == 0
    assert observation.supervisor_stderr == b""
    assert observation.supervisor_not_running_before_test_cleanup
    assert observation.supervisor_reaped_before_test_cleanup
    assert observation.group_after_public_return == ()
    assert not observation.forced_test_cleanup


@pytest.mark.parametrize(
    "token",
    [b"X", b"SS", b"S\n", b"SA", b"A\n"],
    ids=["wrong-token", "multi-byte", "trailing-newline", "start-with-trailer", "abort-trailer"],
)
def test_round12_invalid_control_grammar_fails_without_worker_or_credential(
    tmp_path: Path,
    token: bytes,
) -> None:
    """Only one exact byte is authorized; prefixes and trailing data cannot be accepted."""

    observation = _run_rejected_control_protocol(tmp_path, token)

    assert observation.ready_line.startswith(b"READY ")
    assert observation.owner_line == b"OWNER " + observation.ready_line[len(b"READY ") :]
    owner_process_id = int(observation.owner_line.removeprefix(b"OWNER ").rstrip(b"\n"))
    assert observation.session_identity == (
        owner_process_id,
        owner_process_id,
        owner_process_id,
    )
    if token.startswith(b"S"):
        assert observation.control_ack_line == b"CONTROL S\n"
        assert observation.worker_marker_at_control_ack is False
        assert observation.credential_receipt_at_control_ack is None
        assert observation.lifetime_open_at_control_ack is True
    else:
        assert observation.control_ack_line is None
    assert not observation.worker_marker_exists
    assert observation.credential_receipt is None
    assert observation.lifetime_eof_after_exit
    assert observation.supervisor_returncode not in {None, 0}
    assert observation.supervisor_stderr == b""
    assert observation.supervisor_not_running_before_test_cleanup
    assert observation.supervisor_reaped_before_test_cleanup
    assert observation.group_after_public_return == ()
    assert not observation.forced_test_cleanup


@pytest.mark.parametrize("shadow_kind", ["cwd", "pythonpath"])
def test_round12_supervisor_launch_cannot_execute_import_shadows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shadow_kind: str,
) -> None:
    """A resolved script under isolated Python ignores both cwd and PYTHONPATH packages."""

    working_directory = tmp_path / "working"
    import_directory = tmp_path / "imports"
    working_directory.mkdir()
    import_directory.mkdir()
    shadow_root = working_directory if shadow_kind == "cwd" else import_directory
    shadow_package = shadow_root / "invoice_agents"
    shadow_package.mkdir()
    (shadow_package / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / f"{shadow_kind}-shadow-executed"
    (shadow_package / "spawn_supervisor.py").write_text(
        "\n".join(
            (
                "from pathlib import Path",
                f"Path({os.fspath(marker)!r}).write_text('executed', encoding='ascii')",
                "raise SystemExit(97)",
            )
        ),
        encoding="utf-8",
    )
    environment = isolated_process.sanitized_worker_environment()
    if shadow_kind == "pythonpath":
        environment["PYTHONPATH"] = os.fspath(import_directory)
    captured_commands: list[list[str]] = []
    captured_supervisors: list[int] = []
    captured_sessions: list[tuple[int, int, int]] = []
    real_popen_initializer = subprocess.Popen.__init__

    def capture_command(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        raw_command = args[0] if args else kwargs["args"]
        real_popen_initializer(process, *args, **kwargs)  # type: ignore[arg-type]
        if type(raw_command) is list and any(
            "spawn_supervisor" in str(argument) for argument in raw_command
        ):
            captured_commands.append(raw_command)
            captured_supervisors.append(process.pid)
            captured_sessions.append(_session_identity(process.pid))

    monkeypatch.chdir(working_directory)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        capture_command,
        raising=False,
    )
    result = isolated_process.run_isolated_process(
        command=[
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            "import sys; sys.stdout.buffer.write(b'ok')",
        ],
        request=b"{}",
        # This positive lifecycle proves five bounded phases: launch+OWNER,
        # READY, CONTROL, STARTED, and worker response. Expiry is still failure.
        timeout_seconds=_IMPORT_ISOLATION_LIFECYCLE_SECONDS,
        max_response_bytes=32,
        env=environment,
    )

    expected_entrypoint = Path(spawn_supervisor.__file__).resolve(strict=True)
    assert result == isolated_process.IsolatedProcessResult(b"ok", None)
    assert not marker.exists()
    assert len(captured_commands) == 1
    actual = captured_commands[0]
    assert Path(actual[0]).resolve(strict=True) == Path(sys.executable).resolve(strict=True)
    assert actual[1] == "-I"
    assert Path(actual[2]).is_absolute()
    assert Path(actual[2]).resolve(strict=True) == expected_entrypoint
    assert "-m" not in actual[:3]
    assert len(captured_supervisors) == 1
    supervisor_id = captured_supervisors[0]
    assert captured_sessions == [(supervisor_id, supervisor_id, supervisor_id)]
    assert _process_is_not_running(supervisor_id)
    assert _exact_child_was_reaped(supervisor_id)
    assert _process_group_snapshot(supervisor_id) == ()


def test_round12_pre_set_cancellation_returns_without_any_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation already ordered before admission must not create a supervisor."""

    calls = 0

    def forbidden_initializer(
        _process: subprocess.Popen[bytes],
        *_args: object,
        **_kwargs: object,
    ) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("pre-set cancellation crossed the zero-spawn boundary")

    cancellation = isolated_process.ProcessCancellation()
    cancellation.set()
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        forbidden_initializer,
        raising=False,
    )

    result = isolated_process.run_isolated_process(
        command=[os.fspath(Path(sys.executable).resolve()), "-I", "-c", "pass"],
        request=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=32,
        cancel_requested=cancellation,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert calls == 0
    assert result == isolated_process.IsolatedProcessResult(None, "cancelled")


def test_round12_real_pre_spawn_exec_failure_is_classified_as_contained_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real exec lookup failure before PID creation is START, never crash or cancellation."""

    missing_entrypoint = tmp_path / "trusted-supervisor-does-not-exist"
    initializer_calls = 0
    real_popen_initializer = subprocess.Popen.__init__

    def missing_supervisor(
        *,
        status_descriptor: int,
        control_descriptor: int,
        owner_descriptor: int,
        lifetime_descriptor: int,
        command: list[str],
    ) -> list[str]:
        del status_descriptor, control_descriptor, owner_descriptor, lifetime_descriptor, command
        return [os.fspath(missing_entrypoint)]

    def observed_initializer(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal initializer_calls
        initializer_calls += 1
        real_popen_initializer(process, *args, **kwargs)  # type: ignore[arg-type]

    _install_synthetic_supervisor(monkeypatch, missing_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        observed_initializer,
        raising=False,
    )

    result = isolated_process.run_isolated_process(
        command=[os.fspath(Path(sys.executable).resolve()), "-I", "-c", "pass"],
        request=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=32,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert initializer_calls == 1
    assert result == isolated_process.IsolatedProcessResult(None, "start")


def _publication_fault_supervisor_source(
    *,
    profile: str,
    receipt_descriptor: int,
    publication_release_descriptor: int,
    lifetime_release_descriptor: int,
    worker_marker: Path,
) -> str:
    """Build a synthetic supervisor with receipt-driven publication/control barriers."""

    publication_action = {
        "delayed": (
            "if os.read(publication_release, 1) != b'R': raise SystemExit(94)\n"
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))"
        ),
        "absent": "os.close(status); status = -1",
        "malformed": "os.write(status, b'not-a-ready-record\\n')",
        "oversized": "os.write(status, b'9' * 4097)",
    }[profile]
    descendant_source = "import signal; signal.pause()"
    worker_source = "\n".join(
        (
            "import os, signal, subprocess, sys",
            "from pathlib import Path",
            "os.close(int(sys.argv[1]))",
            f"child = subprocess.Popen([sys.executable, '-I', '-c', {descendant_source!r}])",
            f"Path({os.fspath(worker_marker)!r}).write_text(f'{{os.getpid()}} {{child.pid}}', encoding='ascii')",
            "signal.pause()",
        )
    )
    return "\n".join(
        (
            "import json, os, subprocess, sys",
            "ROUND12_SYNTHETIC_PUBLICATION_SUPERVISOR = True",
            "status, control, owner_fd, lifetime, receipt, publication_release, release = map(int, sys.argv[1:8])",
            "def publish(value):",
            "    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\\n'",
            "    os.write(receipt, encoded)",
            "owner = f'OWNER {os.getpid()}\\n'.encode('ascii')",
            "os.write(owner_fd, owner)",
            "os.close(owner_fd)",
            "publish({'event': 'owner', 'line': owner.decode('ascii'), 'pid': os.getpid()})",
            publication_action,
            "os.close(publication_release)",
            f"publish({{'event': 'publication', 'profile': {profile!r}}})",
            "command = os.read(control, 1)",
            "if command == b'S' and status >= 0:",
            "    os.write(status, b'CONTROL S\\n')",
            "terminator = os.read(control, 1)",
            "publish({'event': 'control', 'value': command.hex(), 'terminator': 'eof' if not terminator else terminator.hex()})",
            "if command == b'S' and not terminator:",
            f"    subprocess.Popen([sys.executable, '-I', '-c', {worker_source!r}, str(lifetime)], pass_fds=(lifetime,))",
            "if status >= 0:",
            "    os.close(status)",
            "os.close(control)",
            "release_value = os.read(release, 1)",
            "publish({'event': 'lifetime-release', 'value': release_value.hex()})",
            "os.close(release)",
            "publish({'event': 'lifetime', 'state': 'exiting'})",
            "os.close(receipt)",
        )
    )


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
@pytest.mark.parametrize("publication", ["delayed", "absent", "malformed", "oversized"])
def test_round12_publication_control_waits_for_abort_and_lifetime_before_reraise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
    control_type: type[BaseException],
) -> None:
    """Every proven publication action ABORTs and retires lifetime before exact reraising."""

    receipt_reader, receipt_writer = os.pipe()
    publication_release_reader, publication_release_writer = os.pipe()
    lifetime_release_reader, lifetime_release_writer = os.pipe()
    worker_marker = tmp_path / f"{publication}-{control_type.__name__}-worker-domain"
    source = _publication_fault_supervisor_source(
        profile=publication,
        receipt_descriptor=receipt_writer,
        publication_release_descriptor=publication_release_reader,
        lifetime_release_descriptor=lifetime_release_reader,
        worker_marker=worker_marker,
    )
    real_popen_initializer = subprocess.Popen.__init__
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    owner_receipts: list[dict[str, object]] = []
    action_receipts: list[dict[str, object]] = []
    action_observed = threading.Event()
    post_init_fault = threading.Event()
    injected = control_type(f"round12 {publication} {control_type.__name__}")
    real_record_error = isolated_process._StartupOwner._record_error

    def synthetic_supervisor(
        *,
        status_descriptor: int,
        control_descriptor: int,
        owner_descriptor: int,
        lifetime_descriptor: int,
        command: list[str],
    ) -> list[str]:
        del command
        return [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
            str(receipt_writer),
            str(publication_release_reader),
            str(lifetime_release_reader),
        ]

    def initialize_then_control(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        raw_command = args[0] if args else kwargs["args"]
        is_synthetic = type(raw_command) is list and any(
            "ROUND12_SYNTHETIC_PUBLICATION_SUPERVISOR" in str(argument) for argument in raw_command
        )
        if is_synthetic:
            inherited = tuple(kwargs.get("pass_fds", ()))
            kwargs["pass_fds"] = (
                *inherited,
                receipt_writer,
                publication_release_reader,
                lifetime_release_reader,
            )
        real_popen_initializer(process, *args, **kwargs)  # type: ignore[arg-type]
        if is_synthetic:
            handles.append(process)
            identities.append(_session_identity(process.pid))
            _close_descriptor(receipt_writer)
            _close_descriptor(publication_release_reader)
            _close_descriptor(lifetime_release_reader)
            owner_receipts.append(_read_json_receipt(receipt_reader))
            if publication != "delayed":
                action_receipts.append(_read_json_receipt(receipt_reader))
                action_observed.set()
            raise injected

    def observe_recorded_error(
        self: isolated_process._StartupOwner,
        error: BaseException,
    ) -> None:
        real_record_error(self, error)
        if error is injected:
            post_init_fault.set()

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_then_control,
        raising=False,
    )
    monkeypatch.setattr(
        isolated_process._StartupOwner,
        "_record_error",
        observe_recorded_error,
    )
    raised: list[BaseException] = []
    results: list[isolated_process.IsolatedProcessResult] = []

    def invoke() -> None:
        try:
            results.append(
                isolated_process.run_isolated_process(
                    command=[os.fspath(Path(sys.executable).resolve()), "-I", "-c", "pass"],
                    request=b"{}",
                    timeout_seconds=21_600.0,
                    max_response_bytes=32,
                    env=isolated_process.sanitized_worker_environment(),
                )
            )
        except BaseException as exc:
            raised.append(exc)

    controller = threading.Thread(
        target=invoke,
        name=f"round12-publication-{publication}-{control_type.__name__}",
    )
    action_ready = False
    fault_ready = False
    returned_before_publication_release = True
    action_before_publication_release = False
    control_receipt: dict[str, object] = {"test_read_error": "action was not observed"}
    release_receipt: dict[str, object] = {"test_read_error": "action was not observed"}
    lifetime_receipt: dict[str, object] = {"test_read_error": "action was not observed"}
    returned_before_release = True
    joined_after_lifetime = False
    supervisor_not_running = False
    supervisor_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    supervisor_returncode: int | None = -999
    forced_cleanup = False
    try:
        controller.start()
        fault_ready = post_init_fault.wait(_PROTOCOL_WAIT_SECONDS)
        if publication == "delayed" and fault_ready:
            with selectors.DefaultSelector() as selector:
                selector.register(receipt_reader, selectors.EVENT_READ)
                action_before_publication_release = bool(selector.select(0.1))
            returned_before_publication_release = not controller.is_alive()
            os.write(publication_release_writer, b"R")
            _close_descriptor(publication_release_writer)
            action_receipts.append(_read_json_receipt(receipt_reader))
            action_observed.set()
        elif publication != "delayed":
            _close_descriptor(publication_release_writer)
        action_ready = action_observed.wait(_PROTOCOL_WAIT_SECONDS)
        if action_ready:
            control_receipt = _read_json_receipt(receipt_reader)
            returned_before_release = not controller.is_alive()
            os.write(lifetime_release_writer, b"L")
            _close_descriptor(lifetime_release_writer)
            release_receipt = _read_json_receipt(receipt_reader)
            lifetime_receipt = _read_json_receipt(receipt_reader)
        controller.join(timeout=_PROTOCOL_WAIT_SECONDS)
        joined_after_lifetime = not controller.is_alive()
        if handles:
            supervisor_returncode = handles[0].returncode
            supervisor_not_running = _process_is_not_running(handles[0].pid)
            supervisor_reaped = _exact_child_was_reaped(handles[0].pid)
            group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        with suppress(OSError):
            os.write(publication_release_writer, b"R")
        _close_descriptor(publication_release_writer)
        with suppress(OSError):
            os.write(lifetime_release_writer, b"L")
        _close_descriptor(lifetime_release_writer)
        if controller.is_alive():
            forced_cleanup = True
        for handle in handles:
            _cleanup_test_process_group(handle.pid, handle)
        controller.join(timeout=_PROTOCOL_WAIT_SECONDS)
        for descriptor in (
            receipt_reader,
            receipt_writer,
            publication_release_reader,
            publication_release_writer,
            lifetime_release_reader,
            lifetime_release_writer,
        ):
            _close_descriptor(descriptor)

    observation = _PublicationControlObservation(
        owner_receipt=owner_receipts[0] if owner_receipts else None,
        action_receipt=action_receipts[0] if action_receipts else None,
        control_receipt=control_receipt,
        lifetime_receipt=lifetime_receipt,
        raised_is_injected=raised == [injected],
        returned_before_lifetime_release=returned_before_release,
        controller_joined_after_lifetime=joined_after_lifetime,
        supervisor_not_running_before_test_cleanup=supervisor_not_running,
        supervisor_reaped_before_test_cleanup=supervisor_reaped,
        group_after_public_return=group_after_return,
        worker_marker_exists=worker_marker.exists(),
        supervisor_returncode_before_test_cleanup=supervisor_returncode,
        forced_test_cleanup=forced_cleanup,
    )
    assert fault_ready
    assert action_ready
    assert len(handles) == 1
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert observation.owner_receipt == {
        "event": "owner",
        "line": f"OWNER {handles[0].pid}\n",
        "pid": handles[0].pid,
    }
    assert observation.action_receipt == {"event": "publication", "profile": publication}
    assert observation.control_receipt == {
        "event": "control",
        "value": "41",
        "terminator": "eof",
    }
    assert not observation.returned_before_lifetime_release
    assert release_receipt == {"event": "lifetime-release", "value": "4c"}
    assert observation.lifetime_receipt == {"event": "lifetime", "state": "exiting"}
    assert observation.raised_is_injected
    assert results == []
    assert observation.controller_joined_after_lifetime
    assert observation.supervisor_not_running_before_test_cleanup
    assert observation.supervisor_reaped_before_test_cleanup
    assert observation.group_after_public_return == ()
    assert not observation.worker_marker_exists
    assert not observation.forced_test_cleanup
    if publication == "delayed":
        assert not action_before_publication_release
        assert not returned_before_publication_release


def test_round12_post_start_lifetime_eof_cannot_replace_known_group_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After START, leader/lifetime exit is not permission to return with group members alive."""

    receipt_reader, receipt_writer = os.pipe()
    cleanup_release_reader, cleanup_release_writer = os.pipe()
    cleanup_release_receipts: list[bytes] = []
    cleanup_entered = threading.Event()
    cleanup_finished = threading.Event()
    controller_results: list[isolated_process.IsolatedProcessResult] = []
    controller_errors: list[BaseException] = []
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    real_popen_initializer = subprocess.Popen.__init__
    real_cleanup = isolated_process._stop_worker_synchronously
    real_snapshot = migration_process._worker_session_snapshot
    live_group_scanned = threading.Event()
    matching_scan_count = 0
    supervisor_source = "\n".join(
        (
            "import json, os, signal, sys",
            "status, control, owner_fd, lifetime, receipt = map(int, sys.argv[1:6])",
            "owner = f'OWNER {os.getpid()}\\n'.encode('ascii')",
            "os.write(owner_fd, owner)",
            "os.close(owner_fd)",
            "owner_event = json.dumps({'event':'owner','line':owner.decode('ascii'),'pid':os.getpid()}, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\\n'",
            "os.write(receipt, owner_event)",
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))",
            "command = os.read(control, 1)",
            "if command != b'S': raise SystemExit(91)",
            "os.write(status, b'CONTROL S\\n')",
            "if os.read(control, 1): raise SystemExit(93)",
            "ready_reader, ready_writer = os.pipe()",
            "child = os.fork()",
            "if child == 0:",
            "    os.setpgid(0, 0)",
            "    os.close(ready_reader)",
            "    for descriptor in (status, control, lifetime, 0, 1, 2):",
            "        try: os.close(descriptor)",
            "        except OSError: pass",
            "    descendant = os.fork()",
            "    if descendant == 0:",
            "        os.close(ready_writer)",
            "        os.close(receipt)",
            "        signal.pause()",
            "        raise SystemExit(99)",
            "    encoded = json.dumps({'event':'domain','group':os.getpgrp(),'child':os.getpid(),'descendant':descendant}, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\\n'",
            "    os.write(receipt, encoded)",
            "    os.close(receipt)",
            "    os.write(ready_writer, b'R')",
            "    os.close(ready_writer)",
            "    signal.pause()",
            "    raise SystemExit(98)",
            "os.close(ready_writer)",
            "if os.read(ready_reader, 1) != b'R': raise SystemExit(92)",
            "os.close(ready_reader)",
            "os.write(status, f'STARTED {child}\\n'.encode('ascii'))",
            "os.close(status)",
            "os.close(control)",
            'os.write(receipt, b\'{"event":"lifetime-exiting"}\\n\')',
            "os.close(receipt)",
            "raise SystemExit(97)",
        )
    )

    def synthetic_supervisor(
        *,
        status_descriptor: int,
        control_descriptor: int,
        owner_descriptor: int,
        lifetime_descriptor: int,
        command: list[str],
    ) -> list[str]:
        del command
        return [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            supervisor_source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
            str(receipt_writer),
        ]

    def initialize_supervisor(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        raw_command = args[0] if args else kwargs["args"]
        is_synthetic = type(raw_command) is list and any(
            "lifetime-exiting" in str(argument) for argument in raw_command
        )
        if is_synthetic:
            inherited = tuple(kwargs.get("pass_fds", ()))
            kwargs["pass_fds"] = (*inherited, receipt_writer)
        real_popen_initializer(process, *args, **kwargs)  # type: ignore[arg-type]
        if is_synthetic:
            handles.append(process)
            identities.append(_session_identity(process.pid))
            _close_descriptor(receipt_writer)

    def hold_required_group_cleanup(
        worker: object,
    ) -> isolated_process._WorkerCleanupOutcome:
        cleanup_entered.set()
        release = os.read(cleanup_release_reader, 1)
        cleanup_release_receipts.append(release)
        try:
            return real_cleanup(worker)  # type: ignore[arg-type]
        finally:
            cleanup_finished.set()

    def observe_required_group_cleanup(
        worker: migration_process._WorkerSession,
    ) -> migration_process._WorkerSessionSnapshot:
        nonlocal matching_scan_count

        snapshot = real_snapshot(worker)
        expected_members = {
            (process_ids[0], process_group_id),
            (process_ids[1], process_group_id),
        }
        observed_members = {
            (member.process_id, member.process_group_id) for member in snapshot.members
        }
        if (
            snapshot.leader_state.startswith("Z")
            and process_group_id > 0
            and expected_members.issubset(observed_members)
        ):
            matching_scan_count += 1
            if matching_scan_count >= 3:
                live_group_scanned.set()
        return snapshot

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_supervisor,
        raising=False,
    )
    monkeypatch.setattr(isolated_process, "_stop_worker_synchronously", hold_required_group_cleanup)
    monkeypatch.setattr(
        migration_process,
        "_worker_session_snapshot",
        observe_required_group_cleanup,
    )

    def invoke() -> None:
        try:
            controller_results.append(
                isolated_process.run_isolated_process(
                    command=[os.fspath(Path(sys.executable).resolve()), "-I", "-c", "pass"],
                    request=b"{}",
                    timeout_seconds=5.0,
                    max_response_bytes=32,
                    env=isolated_process.sanitized_worker_environment(),
                )
            )
        except BaseException as exc:
            controller_errors.append(exc)

    controller = threading.Thread(target=invoke, name="round12-post-start-group-cleanup")
    owner_receipt: dict[str, object] = {"test_read_error": "controller did not start"}
    domain_receipt: dict[str, object] = {"test_read_error": "controller did not start"}
    lifetime_receipt: dict[str, object] = {"test_read_error": "controller did not start"}
    cleanup_was_entered = False
    returned_before_cleanup_release = True
    members_running_before_release = (False, False)
    members_absent_after_return = (False, False)
    controller_joined = False
    forced_cleanup = False
    process_ids = (0, 0)
    process_group_id = 0
    controller_alive_immediately_before_release = False
    controller_alive_with_scanned_live_group = False
    cleanup_finished_before_return = False
    supervisor_not_running = False
    supervisor_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    worker_identity: tuple[int, int, int] | None = None
    descendant_identity: tuple[int, int, int] | None = None
    controlled_group_release = False
    try:
        controller.start()
        owner_receipt = _read_json_receipt(receipt_reader)
        domain_receipt = _read_json_receipt(receipt_reader)
        lifetime_receipt = _read_json_receipt(receipt_reader)
        if (
            type(domain_receipt.get("child")) is int
            and type(domain_receipt.get("descendant")) is int
            and type(domain_receipt.get("group")) is int
        ):
            process_ids = (
                int(domain_receipt["child"]),
                int(domain_receipt["descendant"]),
            )
            process_group_id = int(domain_receipt["group"])
            worker_identity = _session_identity(process_ids[0])
            descendant_identity = _session_identity(process_ids[1])
        cleanup_was_entered = cleanup_entered.wait(_PROTOCOL_WAIT_SECONDS)
        returned_before_cleanup_release = not controller.is_alive()
        if all(process_id > 0 for process_id in process_ids):
            members_running_before_release = tuple(
                not _process_is_not_running(process_id) for process_id in process_ids
            )  # type: ignore[assignment]
        controller_alive_immediately_before_release = controller.is_alive()
        if cleanup_was_entered:
            os.write(cleanup_release_writer, b"G")
            _close_descriptor(cleanup_release_writer)
            if sys.platform == "darwin":
                assert process_group_id == process_ids[0]
                assert worker_identity == (
                    process_ids[0],
                    process_ids[0],
                    handles[0].pid,
                )
                assert descendant_identity == (
                    process_ids[1],
                    process_ids[0],
                    handles[0].pid,
                )
                assert live_group_scanned.wait(_PROTOCOL_WAIT_SECONDS)
                controller_alive_with_scanned_live_group = controller.is_alive()
                assert controller_results == []
                assert controller_errors == []
                assert tuple(
                    not _process_is_not_running(process_id) for process_id in process_ids
                ) == (True, True)
                os.killpg(process_group_id, signal.SIGKILL)
                controlled_group_release = True
            controller.join(timeout=5.0)
            controller_joined = not controller.is_alive()
            cleanup_finished_before_return = cleanup_finished.is_set()
            if all(process_id > 0 for process_id in process_ids):
                members_absent_after_return = tuple(
                    _process_is_not_running(process_id) for process_id in process_ids
                )  # type: ignore[assignment]
            if handles:
                supervisor_not_running = _process_is_not_running(handles[0].pid)
                supervisor_reaped = _exact_child_was_reaped(handles[0].pid)
                group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        with suppress(OSError):
            os.write(cleanup_release_writer, b"E")
        _close_descriptor(cleanup_release_writer)
        if controller.is_alive():
            forced_cleanup = True
        if process_group_id > 0:
            _cleanup_test_process_group(
                process_group_id,
                handles[0] if handles else None,
            )
        else:
            for handle in handles:
                _cleanup_test_process_group(handle.pid, handle)
        controller.join(timeout=_PROTOCOL_WAIT_SECONDS)
        _close_descriptor(receipt_reader)
        _close_descriptor(receipt_writer)
        _close_descriptor(cleanup_release_reader)
        _close_descriptor(cleanup_release_writer)

    assert len(handles) == 1
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert owner_receipt == {
        "event": "owner",
        "line": f"OWNER {handles[0].pid}\n",
        "pid": handles[0].pid,
    }
    assert domain_receipt == {
        "event": "domain",
        "group": process_group_id,
        "child": process_ids[0],
        "descendant": process_ids[1],
    }
    assert worker_identity == (process_ids[0], process_ids[0], handles[0].pid)
    assert descendant_identity == (
        process_ids[1],
        process_ids[0],
        handles[0].pid,
    )
    assert lifetime_receipt == {"event": "lifetime-exiting"}
    assert cleanup_was_entered
    assert not returned_before_cleanup_release
    assert controller_alive_immediately_before_release
    if sys.platform == "darwin":
        assert controller_alive_with_scanned_live_group
        assert matching_scan_count >= 3
        assert controlled_group_release
    else:
        assert not controller_alive_with_scanned_live_group
        assert not controlled_group_release
    assert cleanup_release_receipts == [b"G"]
    assert members_running_before_release == (True, True)
    assert controller_errors == []
    assert controller_results == [isolated_process.IsolatedProcessResult(None, "crash")]
    assert controller_joined
    assert cleanup_finished_before_return
    assert members_absent_after_return == (True, True)
    assert supervisor_not_running
    assert supervisor_reaped
    assert group_after_return == ()
    assert not forced_cleanup
