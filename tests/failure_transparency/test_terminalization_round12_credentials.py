"""Behavioral regressions for the twelfth Task 9 ownership review."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from invoice_agents import isolated_process

_START = b"S"
_ABORT = b"A"
_SUCCESS_PHASE_SECONDS = 2.0
_SINGLE_WORKER_SUCCESS_SECONDS = 5.0 * _SUCCESS_PHASE_SECONDS
_WORKER_DESCENDANT_SUCCESS_SECONDS = 7.0 * _SUCCESS_PHASE_SECONDS


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


def _bounded_bytes(path: Path, *, timeout: float = 2.0) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = path.read_bytes()
        except OSError:
            pass
        else:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"bounded receipt was not published: {path}")


def _optional_bytes(path: Path, *, timeout: float = 0.3) -> bytes | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return path.read_bytes()
        except OSError:
            time.sleep(0.01)
    return None


def _bounded_pid(path: Path, *, timeout: float = 2.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            encoded = path.read_text(encoding="ascii")
        except (OSError, UnicodeError):
            pass
        else:
            if encoded.isascii() and encoded.isdigit():
                process_id = int(encoded)
                if process_id > 0 and str(process_id) == encoded:
                    return process_id
        time.sleep(0.01)
    raise AssertionError(f"complete PID receipt was not published: {path}")


def _bounded_pid_pair(path: Path, *, timeout: float = 2.0) -> tuple[int, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            encoded = path.read_text(encoding="ascii")
        except (OSError, UnicodeError):
            pass
        else:
            fields = encoded.split()
            if len(fields) == 2 and all(
                field.isascii() and field.isdigit() and int(field) > 0 and str(int(field)) == field
                for field in fields
            ):
                return int(fields[0]), int(fields[1])
        time.sleep(0.01)
    raise AssertionError(f"complete PID-pair receipt was not published: {path}")


def _process_not_running(process_id: int) -> bool:
    completed = subprocess.run(
        ["/bin/ps", "-o", "stat=", "-p", str(process_id)],
        check=False,
        capture_output=True,
        text=True,
        timeout=1.0,
    )
    if completed.returncode not in {0, 1}:
        raise AssertionError(
            f"ps could not inspect PID {process_id}: "
            f"exit={completed.returncode} stderr={completed.stderr!r}"
        )
    state = completed.stdout.strip()
    if completed.returncode == 1:
        if state:
            raise AssertionError(
                f"ps reported PID {process_id} absent but returned state {state!r}"
            )
        return True
    return False


def _process_group_snapshot(process_group_id: int) -> tuple[tuple[int, str], ...]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,stat="],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    members: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if (
            len(fields) == 3
            and fields[0].isdigit()
            and fields[1].isdigit()
            and int(fields[1]) == process_group_id
        ):
            members.append((int(fields[0]), fields[2]))
    return tuple(sorted(members))


def _session_identity(process_id: int) -> tuple[int, int, int]:
    return process_id, os.getpgid(process_id), os.getsid(process_id)


def _exact_child_was_reaped(process_id: int) -> bool:
    try:
        os.waitpid(process_id, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


def _cleanup_test_process_group(process_group_id: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGKILL)
    with suppress(ChildProcessError):
        os.waitpid(process_group_id, 0)


def _terminate_test_domain(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=2.0)


def _publication_supervisor_source(
    *,
    mode: str,
    entered: Path,
    action: Path,
    release: Path,
    control_receipt: Path,
    owner_receipt: Path,
    lifetime_receipt: Path,
    worker_receipt: Path,
) -> str:
    descendant_code = "import time; time.sleep(30)"
    worker_code = "\n".join(
        (
            "import os, subprocess, sys, time",
            "from pathlib import Path",
            "os.close(int(sys.argv[1]))",
            f"child = subprocess.Popen([sys.executable, '-I', '-c', {descendant_code!r}])",
            f"Path({os.fspath(worker_receipt)!r}).write_text(f'{{os.getpid()}} {{child.pid}}', encoding='ascii')",
            "time.sleep(30)",
        )
    )
    publication_action = {
        "delayed": "os.write(publication, f'READY {os.getpid()}\\n'.encode('ascii'))",
        "absent": "os.close(publication); publication = -1",
        "malformed": "os.write(publication, b'not-a-process-id\\n')",
        "oversized": "os.write(publication, b'9' * 32)",
    }[mode]
    delay_barrier = (
        f"while not Path({os.fspath(release)!r}).exists(): time.sleep(0.005)"
        if mode == "delayed"
        else "pass"
    )
    return "\n".join(
        (
            "import os, subprocess, sys, time",
            "from contextlib import suppress",
            "from pathlib import Path",
            "publication = int(sys.argv[1])",
            "control = int(sys.argv[2])",
            "owner_fd = int(sys.argv[3])",
            "lifetime = int(sys.argv[4])",
            f"Path({os.fspath(entered)!r}).write_text(str(os.getpid()), encoding='ascii')",
            "owner = f'OWNER {os.getpid()}\\n'.encode('ascii')",
            "os.write(owner_fd, owner)",
            "os.close(owner_fd)",
            f"Path({os.fspath(owner_receipt)!r}).write_bytes(owner)",
            delay_barrier,
            publication_action,
            f"Path({os.fspath(action)!r}).write_text({mode!r}, encoding='ascii')",
            "if control < 0:",
            "    time.sleep(30)",
            "received = bytearray(os.read(control, 1))",
            "if bytes(received) == b'S' and publication >= 0:",
            "    os.write(publication, b'CONTROL S\\n')",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "os.close(control)",
            "if bytes(received) == b'S':",
            f"    worker = subprocess.Popen([sys.executable, '-I', '-c', {worker_code!r}, str(lifetime)], pass_fds=(lifetime,))",
            "    if publication >= 0:",
            "        os.write(publication, f'STARTED {worker.pid}\\n'.encode('ascii'))",
            "if publication >= 0:",
            "    with suppress(OSError): os.close(publication)",
            "if bytes(received) == b'S':",
            "    time.sleep(30)",
            f"Path({os.fspath(lifetime_receipt)!r}).write_text('exiting', encoding='ascii')",
        )
    )


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
@pytest.mark.parametrize("publication", ["delayed", "absent", "malformed", "oversized"])
def test_round12_publication_fault_aborts_before_worker_and_proves_lifetime_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
    control_type: type[BaseException],
) -> None:
    """Every exercised publication branch ABORTs and exits before reraising control."""

    entered = tmp_path / "entered"
    action = tmp_path / "action"
    release = tmp_path / "release"
    control_receipt = tmp_path / "control"
    owner_receipt = tmp_path / "owner"
    lifetime_receipt = tmp_path / "lifetime"
    worker_receipt = tmp_path / "worker-pids"
    child_source = _publication_supervisor_source(
        mode=publication,
        entered=entered,
        action=action,
        release=release,
        control_receipt=control_receipt,
        owner_receipt=owner_receipt,
        lifetime_receipt=lifetime_receipt,
        worker_receipt=worker_receipt,
    )
    real_popen_initializer = subprocess.Popen.__init__
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    injected = control_type(f"round12 {publication} {control_type.__name__}")
    post_init_fault = threading.Event()
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
            sys.executable,
            "-I",
            "-c",
            child_source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
        ]

    def initialize_then_control(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_popen_initializer(process, *args, **kwargs)
        command = args[0] if args else kwargs["args"]
        if (
            isinstance(command, list)
            and any(os.fspath(entered) in str(argument) for argument in command)
            and not handles
        ):
            handles.append(process)
            identities.append(_session_identity(process.pid))
            assert _bounded_pid(entered) == process.pid
            if publication != "delayed":
                assert _bounded_bytes(action) == publication.encode("ascii")
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
    )
    monkeypatch.setattr(
        isolated_process._StartupOwner,
        "_record_error",
        observe_recorded_error,
    )

    raised: list[BaseException] = []
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    observed_control: bytes | None = None
    observed_owner: bytes | None = None
    lifetime_exiting = False
    worker_ids: tuple[int, int] | None = None
    worker_states: tuple[bool, bool] | None = None
    fault_observed = False
    returned_before_delayed_release = True
    action_before_delayed_release: bytes | None = b"not-delayed"

    def invoke() -> None:
        try:
            isolated_process.run_isolated_process(
                command=[sys.executable, "-I", "-c", "raise SystemExit(0)"],
                request=b"{}",
                timeout_seconds=0.5,
                max_response_bytes=32,
                env=isolated_process.sanitized_worker_environment(),
            )
        except BaseException as exc:
            raised.append(exc)

    controller = threading.Thread(
        target=invoke,
        name=f"round12-credential-publication-{publication}-{control_type.__name__}",
    )
    try:
        controller.start()
        fault_observed = post_init_fault.wait(2.0)
        if publication == "delayed" and fault_observed:
            action_before_delayed_release = _optional_bytes(action, timeout=0.1)
            returned_before_delayed_release = not controller.is_alive()
            release.write_text("release", encoding="ascii")
        controller.join(timeout=3.0)
        assert len(handles) == 1
        handle = handles[0]
        leader_not_running = _process_not_running(handle.pid)
        leader_reaped = _exact_child_was_reaped(handle.pid)
        group_after_return = _process_group_snapshot(handle.pid)
        observed_control = _optional_bytes(control_receipt)
        observed_owner = _optional_bytes(owner_receipt)
        lifetime_exiting = _optional_bytes(lifetime_receipt) == b"exiting"
        if worker_receipt.exists():
            worker_ids = _bounded_pid_pair(worker_receipt)
            worker_states = tuple(_process_not_running(pid) for pid in worker_ids)  # type: ignore[assignment]
    finally:
        release.write_text("release", encoding="ascii")
        controller.join(timeout=2.0)
        for handle in handles:
            _terminate_test_domain(handle)

    assert fault_observed
    assert raised == [injected]
    assert identities == [(handle.pid, handle.pid, handle.pid)]
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()
    assert observed_control == _ABORT
    assert observed_owner == f"OWNER {handle.pid}\n".encode("ascii")
    assert lifetime_exiting
    assert worker_ids is None
    assert worker_states is None
    if publication == "delayed":
        assert action_before_delayed_release is None
        assert not returned_before_delayed_release
        assert _bounded_bytes(action) == b"delayed"


@pytest.mark.parametrize(
    "owner_profile",
    [
        "missing",
        "malformed",
        "partial",
        "oversized",
        "trailing",
        "mismatch",
        "leading-zero",
        "plus",
        "leading-space",
        "trailing-space",
        "zero",
        "negative",
        "non-ascii",
    ],
)
def test_round12_reserved_native_handle_and_invalid_owner_preserve_primary_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_profile: str,
) -> None:
    """A retained native handle cleans invalid OWNER cases before exact reraising."""

    control_receipt = tmp_path / "control"
    exit_receipt = tmp_path / "exiting"
    unrelated = (
        subprocess.Popen(
            [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        if owner_profile == "mismatch"
        else None
    )
    owner_action = {
        "missing": "pass",
        "malformed": "os.write(owner_fd, b'owner malformed\\n')",
        "partial": "os.write(owner_fd, b'OWNER ')",
        "oversized": "os.write(owner_fd, b'OWNER ' + b'9' * 128 + b'\\n')",
        "trailing": "os.write(owner_fd, f'OWNER {os.getpid()}\\nX'.encode('ascii'))",
        "mismatch": (
            f"os.write(owner_fd, b'OWNER {unrelated.pid}\\n')"
            if unrelated is not None
            else "raise AssertionError('missing mismatch fixture')"
        ),
        "leading-zero": "os.write(owner_fd, f'OWNER 0{os.getpid()}\\n'.encode('ascii'))",
        "plus": "os.write(owner_fd, f'OWNER +{os.getpid()}\\n'.encode('ascii'))",
        "leading-space": "os.write(owner_fd, f'OWNER  {os.getpid()}\\n'.encode('ascii'))",
        "trailing-space": "os.write(owner_fd, f'OWNER {os.getpid()} \\n'.encode('ascii'))",
        "zero": "os.write(owner_fd, b'OWNER 0\\n')",
        "negative": "os.write(owner_fd, b'OWNER -1\\n')",
        "non-ascii": "os.write(owner_fd, f'OWNER {os.getpid()}'.encode('ascii') + b'\\xff\\n')",
    }[owner_profile]
    expected_parser_error = {
        "missing": "invalid isolated supervisor ownership publication",
        "malformed": "invalid isolated supervisor status",
        "partial": "invalid isolated supervisor ownership publication",
        "oversized": "invalid isolated supervisor ownership publication",
        "trailing": "invalid isolated supervisor ownership publication",
        "mismatch": None,
        "leading-zero": "invalid isolated supervisor status",
        "plus": "invalid isolated supervisor status",
        "leading-space": "invalid isolated supervisor status",
        "trailing-space": "invalid isolated supervisor status",
        "zero": "invalid isolated supervisor status",
        "negative": "invalid isolated supervisor status",
        "non-ascii": "invalid isolated supervisor status",
    }[owner_profile]
    supervisor_source = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "status, control, owner_fd, lifetime = map(int, sys.argv[1:5])",
            owner_action,
            "os.close(owner_fd)",
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))",
            "received = bytearray(os.read(control, 1))",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "os.close(status)",
            "os.close(control)",
            f"Path({os.fspath(exit_receipt)!r}).write_text('exiting', encoding='ascii')",
        )
    )
    real_popen_initializer = subprocess.Popen.__init__
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    capture_process_ids: list[int] = []
    poll_process_ids: list[int] = []
    kill_process_ids: list[int] = []
    killpg_process_ids: list[int] = []
    waitpid_process_ids: list[int] = []
    owner_parse_observations: list[tuple[str, int | None]] = []
    injected = SystemExit(f"round12 invalid OWNER {owner_profile}")
    real_capture = isolated_process._capture_worker_session
    real_owner_read = isolated_process._read_owner_publication
    real_spawned_poll = isolated_process._SpawnedProcess.poll
    real_kill = os.kill
    real_killpg = os.killpg
    real_waitpid = os.waitpid

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
            sys.executable,
            "-I",
            "-c",
            supervisor_source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
        ]

    def capture_initialized_process(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_popen_initializer(process, *args, **kwargs)
        handles.append(process)
        identities.append(_session_identity(process.pid))

    def inject_after_owner_parse(
        reader: isolated_process.PrivatePipeEndpoint,
        *,
        deadline: float,
    ) -> int:
        try:
            process_id = real_owner_read(reader, deadline=deadline)
        except ValueError as exc:
            assert expected_parser_error is not None
            assert str(exc) == expected_parser_error
            owner_parse_observations.append(("rejected", None))
            raise injected from exc
        if expected_parser_error is not None or unrelated is None or process_id != unrelated.pid:
            raise AssertionError("invalid OWNER unexpectedly parsed as native authority")
        owner_parse_observations.append(("mismatch", process_id))
        raise injected

    def observe_capture(
        process: subprocess.Popen[bytes],
        retained_worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerSession:
        capture_process_ids.append(process.pid)
        if unrelated is not None and process.pid == unrelated.pid:
            raise RuntimeError("unrelated invalid OWNER PID must not be captured")
        return real_capture(process, retained_worker)

    def observe_poll(self: isolated_process._SpawnedProcess) -> int | None:
        poll_process_ids.append(self.pid)
        if unrelated is not None and self.pid == unrelated.pid:
            raise RuntimeError("unrelated invalid OWNER PID must not be polled")
        return real_spawned_poll(self)

    def observe_kill(process_id: int, signal_number: int) -> None:
        kill_process_ids.append(process_id)
        if unrelated is not None and process_id == unrelated.pid:
            raise AssertionError("unrelated invalid OWNER PID must not be signalled")
        real_kill(process_id, signal_number)

    def observe_killpg(process_group_id: int, signal_number: int) -> None:
        killpg_process_ids.append(process_group_id)
        if unrelated is not None and process_group_id == unrelated.pid:
            raise AssertionError("unrelated invalid OWNER PGID must not be signalled")
        real_killpg(process_group_id, signal_number)

    def observe_waitpid(process_id: int, options: int) -> tuple[int, int]:
        waitpid_process_ids.append(process_id)
        if unrelated is not None and process_id == unrelated.pid:
            raise AssertionError("unrelated invalid OWNER PID must not be reaped")
        return real_waitpid(process_id, options)

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        capture_initialized_process,
    )
    monkeypatch.setattr(
        isolated_process,
        "_read_owner_publication",
        inject_after_owner_parse,
    )
    if unrelated is not None:
        monkeypatch.setattr(isolated_process, "_capture_worker_session", observe_capture)
        monkeypatch.setattr(isolated_process._SpawnedProcess, "poll", observe_poll)
        monkeypatch.setattr(os, "kill", observe_kill)
        monkeypatch.setattr(os, "killpg", observe_killpg)
        monkeypatch.setattr(os, "waitpid", observe_waitpid)
    raised: BaseException | None = None
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    observed_control: bytes | None = None
    exiting = False
    unrelated_survived = False
    authority_observation: tuple[tuple[int, ...], ...] | None = None
    try:
        try:
            isolated_process.run_isolated_process(
                command=[sys.executable, "-I", "-c", "pass"],
                request=b"{}",
                timeout_seconds=0.5,
                max_response_bytes=32,
                env=isolated_process.sanitized_worker_environment(),
            )
        except BaseException as exc:
            raised = exc
        assert len(handles) == 1
        leader_not_running = _process_not_running(handles[0].pid)
        leader_reaped = _exact_child_was_reaped(handles[0].pid)
        authority_observation = (
            tuple(capture_process_ids),
            tuple(poll_process_ids),
            tuple(kill_process_ids),
            tuple(killpg_process_ids),
            tuple(waitpid_process_ids),
        )
        monkeypatch.setattr(os, "kill", real_kill)
        monkeypatch.setattr(os, "killpg", real_killpg)
        monkeypatch.setattr(os, "waitpid", real_waitpid)
        unrelated_survived = unrelated is None or unrelated.poll() is None
        group_after_return = _process_group_snapshot(handles[0].pid)
        observed_control = _optional_bytes(control_receipt)
        exiting = _optional_bytes(exit_receipt) == b"exiting"
    finally:
        monkeypatch.setattr(os, "kill", real_kill)
        monkeypatch.setattr(os, "killpg", real_killpg)
        monkeypatch.setattr(os, "waitpid", real_waitpid)
        for handle in handles:
            _terminate_test_domain(handle)
        if unrelated is not None:
            _terminate_test_domain(unrelated)

    assert raised is injected
    assert owner_parse_observations == [
        ("mismatch", unrelated.pid) if unrelated is not None else ("rejected", None)
    ]
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()
    assert observed_control == _ABORT
    assert exiting
    if unrelated is not None:
        assert authority_observation is not None
        assert all(unrelated.pid not in observed for observed in authority_observation)
        assert set(capture_process_ids) <= {handles[0].pid}
        assert set(poll_process_ids) <= {handles[0].pid}
        assert unrelated_survived


def test_round12_reserved_native_cleanup_failure_supersedes_post_init_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible cleanup failure outranks control raised after native initialization."""

    control_receipt = tmp_path / "control"
    supervisor_source = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "status, control, owner_fd, lifetime = map(int, sys.argv[1:5])",
            "os.write(owner_fd, b'owner malformed\\n')",
            "os.close(owner_fd)",
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))",
            "received = bytearray(os.read(control, 1))",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
        )
    )
    real_popen_initializer = subprocess.Popen.__init__
    real_cleanup = isolated_process._stop_worker_synchronously
    handles: list[subprocess.Popen[bytes]] = []
    injected = SystemExit("round12 initialized native cleanup failure")

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
            sys.executable,
            "-I",
            "-c",
            supervisor_source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
        ]

    def initialize_then_control(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_popen_initializer(process, *args, **kwargs)
        handles.append(process)
        raise injected

    def cleanup_then_report_failure(
        worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerCleanupOutcome:
        outcome = real_cleanup(worker)
        assert outcome.contained
        assert outcome.reaped
        outcome.failed = True
        return outcome

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_then_control,
    )
    monkeypatch.setattr(
        isolated_process,
        "_stop_worker_synchronously",
        cleanup_then_report_failure,
    )

    raised: BaseException | None = None
    try:
        try:
            isolated_process.run_isolated_process(
                command=[sys.executable, "-I", "-c", "pass"],
                request=b"{}",
                timeout_seconds=0.5,
                max_response_bytes=32,
                env=isolated_process.sanitized_worker_environment(),
            )
        except BaseException as exc:
            raised = exc
    finally:
        for handle in handles:
            _terminate_test_domain(handle)

    assert len(handles) == 1
    assert isinstance(raised, isolated_process.IsolatedProcessCleanupError)
    assert raised is not injected
    assert _bounded_bytes(control_receipt) == _ABORT
    assert _process_not_running(handles[0].pid)
    assert _exact_child_was_reaped(handles[0].pid)
    assert _process_group_snapshot(handles[0].pid) == ()


@pytest.mark.parametrize("mismatch", ["owner-native", "ready-owner"])
def test_round12_owner_native_and_ready_identities_must_match_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    """Any disagreement among native, OWNER, and READY identities prevents START."""

    owner_receipt = tmp_path / "owner"
    control_receipt = tmp_path / "control"
    exit_receipt = tmp_path / "exiting"
    unrelated = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    owner_pid = str(unrelated.pid) if mismatch == "owner-native" else "os.getpid()"
    ready_pid = str(unrelated.pid) if mismatch == "ready-owner" else "os.getpid()"
    supervisor_source = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "status, control, owner_fd, lifetime = map(int, sys.argv[1:5])",
            f"owner = f'OWNER {{{owner_pid}}}\\n'.encode('ascii')",
            "os.write(owner_fd, owner)",
            "os.close(owner_fd)",
            f"Path({os.fspath(owner_receipt)!r}).write_bytes(owner)",
            f"os.write(status, f'READY {{{ready_pid}}}\\n'.encode('ascii'))",
            "received = bytearray(os.read(control, 1))",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "os.close(status)",
            "os.close(control)",
            f"Path({os.fspath(exit_receipt)!r}).write_text('exiting', encoding='ascii')",
        )
    )
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    cleanup_process_ids: list[int] = []
    capture_process_ids: list[int] = []
    poll_process_ids: list[int] = []
    kill_process_ids: list[int] = []
    killpg_process_ids: list[int] = []
    waitpid_process_ids: list[int] = []
    real_popen_initializer = subprocess.Popen.__init__
    real_cleanup = isolated_process._stop_worker_synchronously
    real_capture = isolated_process._capture_worker_session
    real_spawned_poll = isolated_process._SpawnedProcess.poll
    real_kill = os.kill
    real_killpg = os.killpg
    real_waitpid = os.waitpid

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
            sys.executable,
            "-I",
            "-c",
            supervisor_source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
        ]

    def initialize_and_capture(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_popen_initializer(process, *args, **kwargs)
        raw_command = args[0] if args else kwargs["args"]
        if isinstance(raw_command, list) and any(
            os.fspath(owner_receipt) in str(argument) for argument in raw_command
        ):
            handles.append(process)
            identities.append(_session_identity(process.pid))

    def observe_cleanup(
        worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerCleanupOutcome:
        cleanup_process_ids.append(worker.process_id)
        return real_cleanup(worker)

    def observe_capture(
        process: subprocess.Popen[bytes],
        retained_worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerSession:
        capture_process_ids.append(process.pid)
        if process.pid == unrelated.pid:
            raise RuntimeError("unrelated advertised PID must not be captured")
        return real_capture(process, retained_worker)

    def observe_poll(self: isolated_process._SpawnedProcess) -> int | None:
        poll_process_ids.append(self.pid)
        if self.pid == unrelated.pid:
            raise RuntimeError("unrelated advertised PID must not be polled")
        return real_spawned_poll(self)

    def observe_kill(process_id: int, signal_number: int) -> None:
        kill_process_ids.append(process_id)
        if process_id == unrelated.pid:
            raise AssertionError("unrelated advertised PID must not be signalled")
        real_kill(process_id, signal_number)

    def observe_killpg(process_group_id: int, signal_number: int) -> None:
        killpg_process_ids.append(process_group_id)
        if process_group_id == unrelated.pid:
            raise AssertionError("unrelated advertised PGID must not be signalled")
        real_killpg(process_group_id, signal_number)

    def observe_waitpid(process_id: int, options: int) -> tuple[int, int]:
        waitpid_process_ids.append(process_id)
        if process_id == unrelated.pid:
            raise AssertionError("unrelated advertised PID must not be reaped")
        return real_waitpid(process_id, options)

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_and_capture,
    )
    monkeypatch.setattr(isolated_process, "_stop_worker_synchronously", observe_cleanup)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observe_capture)
    monkeypatch.setattr(isolated_process._SpawnedProcess, "poll", observe_poll)
    monkeypatch.setattr(os, "kill", observe_kill)
    monkeypatch.setattr(os, "killpg", observe_killpg)
    monkeypatch.setattr(os, "waitpid", observe_waitpid)
    result: isolated_process.IsolatedProcessResult | None = None
    leader_not_running = False
    leader_reaped = False
    unrelated_survived = False
    authority_observation: tuple[tuple[int, ...], ...] | None = None
    group_after_return: tuple[tuple[int, str], ...] | None = None
    try:
        result = isolated_process.run_isolated_process(
            command=[sys.executable, "-I", "-c", "pass"],
            request=b"{}",
            timeout_seconds=0.5,
            max_response_bytes=32,
            env=isolated_process.sanitized_worker_environment(),
        )
        assert len(handles) == 1
        leader_not_running = _process_not_running(handles[0].pid)
        leader_reaped = _exact_child_was_reaped(handles[0].pid)
        authority_observation = (
            tuple(capture_process_ids),
            tuple(poll_process_ids),
            tuple(kill_process_ids),
            tuple(killpg_process_ids),
            tuple(waitpid_process_ids),
        )
        monkeypatch.setattr(os, "kill", real_kill)
        monkeypatch.setattr(os, "killpg", real_killpg)
        monkeypatch.setattr(os, "waitpid", real_waitpid)
        unrelated_survived = unrelated.poll() is None
        group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        monkeypatch.setattr(os, "kill", real_kill)
        monkeypatch.setattr(os, "killpg", real_killpg)
        monkeypatch.setattr(os, "waitpid", real_waitpid)
        for handle in handles:
            _terminate_test_domain(handle)
        _terminate_test_domain(unrelated)

    assert len(handles) == 1
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert cleanup_process_ids == [handles[0].pid]
    assert cleanup_process_ids != [unrelated.pid]
    assert authority_observation is not None
    assert all(unrelated.pid not in observed for observed in authority_observation)
    assert set(capture_process_ids) <= {handles[0].pid}
    assert set(poll_process_ids) <= {handles[0].pid}
    assert unrelated_survived
    assert result == isolated_process.IsolatedProcessResult(None, "start")
    assert _bounded_bytes(control_receipt) == _ABORT
    assert _bounded_bytes(exit_receipt) == b"exiting"
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()


def test_round12_reserved_native_handle_survives_until_cleanup_without_competing_reaper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reserved Popen stays retained until exact cleanup has completed."""

    owner_receipt = tmp_path / "owner"
    control_receipt = tmp_path / "control"
    supervisor_source = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "status, control, owner_fd, lifetime = map(int, sys.argv[1:5])",
            "owner = f'OWNER {os.getpid()}\\n'.encode('ascii')",
            "os.write(owner_fd, owner)",
            "os.close(owner_fd)",
            f"Path({os.fspath(owner_receipt)!r}).write_bytes(owner)",
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))",
            "received = bytearray(os.read(control, 1))",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "os.close(status)",
            "os.close(control)",
        )
    )
    process_ids: list[int] = []
    identities: list[tuple[int, int, int]] = []
    process_refs: list[weakref.ReferenceType[subprocess.Popen[bytes]]] = []
    real_popen = subprocess.Popen
    real_popen_initializer = subprocess.Popen.__init__
    real_internal_poll = subprocess.Popen._internal_poll
    real_popen_wait = subprocess.Popen.wait
    real_popen_private_wait = subprocess.Popen._wait
    real_cleanup = isolated_process._stop_worker_synchronously
    injected = SystemExit("round12 post-init control")
    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()
    original_popen_finalized = threading.Event()
    competing_waits: list[tuple[str, int]] = []
    cleanup_workers: list[tuple[int, int, bool]] = []
    raised: list[BaseException] = []

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
            sys.executable,
            "-I",
            "-c",
            supervisor_source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
        ]

    def initialize_then_control(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_popen_initializer(process, *args, **kwargs)
        process_ids.append(process.pid)
        identities.append(_session_identity(process.pid))
        process_refs.append(weakref.ref(process, lambda _reference: original_popen_finalized.set()))
        raise injected

    def observe_competing_waiter(
        self: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> int | None:
        if process_ids and self.pid == process_ids[0]:
            competing_waits.append(("_internal_poll", self.pid))
        return real_internal_poll(self, *args, **kwargs)

    def observe_public_wait(
        self: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if process_ids and self.pid == process_ids[0]:
            competing_waits.append(("wait", self.pid))
        return real_popen_wait(self, *args, **kwargs)

    def observe_private_wait(
        self: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if process_ids and self.pid == process_ids[0]:
            competing_waits.append(("_wait", self.pid))
        return real_popen_private_wait(self, *args, **kwargs)

    def hold_cleanup(
        worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerCleanupOutcome:
        cleanup_workers.append(
            (
                worker.process_id,
                worker.process.pid,
                isinstance(worker.process, isolated_process._SpawnedProcess)
                and worker.process._native_handle is None,
            )
        )
        cleanup_entered.set()
        cleanup_release.wait(2.0)
        return real_cleanup(worker)

    def invoke() -> None:
        try:
            isolated_process.run_isolated_process(
                command=[sys.executable, "-I", "-c", "pass"],
                request=b"{}",
                timeout_seconds=1.0,
                max_response_bytes=32,
                env=isolated_process.sanitized_worker_environment(),
            )
        except BaseException as exc:
            raised.append(exc)

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_then_control,
    )
    monkeypatch.setattr(real_popen, "_internal_poll", observe_competing_waiter)
    monkeypatch.setattr(real_popen, "wait", observe_public_wait)
    monkeypatch.setattr(real_popen, "_wait", observe_private_wait)
    monkeypatch.setattr(isolated_process, "_stop_worker_synchronously", hold_cleanup)
    controller = threading.Thread(target=invoke, name="round12-reserved-popen-control")
    cleanup_was_pending = False
    retained_while_cleanup_pending = False
    retained_after_global_cleanup = False
    finalized_after_cleanup = False
    observed_owner: bytes | None = None
    observed_control: bytes | None = None
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    try:
        controller.start()
        cleanup_was_pending = cleanup_entered.wait(2.0)
        injected.__traceback__ = None
        gc.collect()
        retained_while_cleanup_pending = (
            cleanup_was_pending
            and controller.is_alive()
            and len(process_refs) == 1
            and process_refs[0]() is not None
            and not original_popen_finalized.is_set()
        )
        subprocess._cleanup()
        gc.collect()
        retained_after_global_cleanup = (
            retained_while_cleanup_pending
            and process_refs[0]() is not None
            and not original_popen_finalized.is_set()
        )
        observed_owner = _optional_bytes(owner_receipt)
        observed_control = _optional_bytes(control_receipt)
        cleanup_release.set()
        controller.join(timeout=3.0)
        injected.__traceback__ = None
        gc.collect()
        finalized_after_cleanup = original_popen_finalized.wait(2.0) and process_refs[0]() is None
        if process_ids:
            leader_not_running = _process_not_running(process_ids[0])
            leader_reaped = _exact_child_was_reaped(process_ids[0])
            group_after_return = _process_group_snapshot(process_ids[0])
    finally:
        cleanup_release.set()
        if process_ids:
            _cleanup_test_process_group(process_ids[0])
        controller.join(timeout=2.0)

    assert cleanup_was_pending
    assert retained_while_cleanup_pending
    assert retained_after_global_cleanup
    assert finalized_after_cleanup
    assert raised == [injected]
    assert len(process_ids) == 1
    assert identities == [(process_ids[0], process_ids[0], process_ids[0])]
    assert cleanup_workers == [(process_ids[0], process_ids[0], True)]
    assert observed_owner == f"OWNER {process_ids[0]}\n".encode("ascii")
    assert observed_control == _ABORT
    assert competing_waits == []
    assert not controller.is_alive()
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()


def test_round12_exec_oserror_is_already_reaped_and_returns_start_without_owner_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real exec failure has no child ownership protocol left to consume."""

    initialized: list[subprocess.Popen[bytes]] = []
    owner_reads: list[int] = []
    real_initializer = subprocess.Popen.__init__
    real_owner_read = isolated_process._read_owner_publication

    def initialize_reserved(
        process: subprocess.Popen[bytes],
        _args: object,
        **kwargs: Any,
    ) -> None:
        initialized.append(process)
        real_initializer(
            process,
            ["/definitely/not/a/round12-supervisor"],
            **kwargs,
        )

    def observe_owner_read(
        reader: isolated_process.PrivatePipeEndpoint,
        *,
        deadline: float,
    ) -> int:
        owner_reads.append(reader.fileno())
        return real_owner_read(reader, deadline=deadline)

    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_reserved,
    )
    monkeypatch.setattr(isolated_process, "_read_owner_publication", observe_owner_read)

    result = isolated_process.run_isolated_process(
        command=[sys.executable, "-I", "-c", "pass"],
        request=b"{}",
        timeout_seconds=0.2,
        max_response_bytes=32,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert result == isolated_process.IsolatedProcessResult(None, "start")
    assert len(initialized) == 1
    assert getattr(initialized[0], "_child_created", False) is False
    assert owner_reads == []


def test_round12_pre_child_initializer_failure_has_no_owner_or_cleanup_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure before native child creation retires the blank reservation only."""

    injected = RuntimeError("round12 pre-child initializer failure")
    reservations: list[subprocess.Popen[bytes]] = []
    owner_reads: list[int] = []
    captures: list[int] = []
    cleanup_authorities: list[int] = []

    def fail_before_child(
        process: subprocess.Popen[bytes],
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        reservations.append(process)
        assert getattr(process, "_child_created", False) is False
        raise injected

    def forbid_owner_read(
        reader: isolated_process.PrivatePipeEndpoint,
        *,
        deadline: float,
    ) -> int:
        del deadline
        owner_reads.append(reader.fileno())
        raise AssertionError("pre-child failure must not read OWNER")

    def forbid_capture(
        process: subprocess.Popen[bytes],
        _retained_worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerSession:
        captures.append(process.pid)
        raise AssertionError("pre-child failure must not capture a process")

    def forbid_cleanup(
        worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerCleanupOutcome:
        cleanup_authorities.append(worker.process_id)
        return isolated_process._WorkerCleanupOutcome(
            contained=False,
            reaped=False,
            failed=True,
        )

    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        fail_before_child,
    )
    monkeypatch.setattr(isolated_process, "_read_owner_publication", forbid_owner_read)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", forbid_capture)
    monkeypatch.setattr(isolated_process, "_stop_worker_synchronously", forbid_cleanup)

    result = isolated_process.run_isolated_process(
        command=[sys.executable, "-I", "-c", "pass"],
        request=b"{}",
        timeout_seconds=0.2,
        max_response_bytes=32,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert result == isolated_process.IsolatedProcessResult(None, "start")
    assert len(reservations) == 1
    assert owner_reads == []
    assert captures == []
    assert cleanup_authorities == []


@pytest.mark.parametrize("profile", ["canonical", "malformed", "mismatched"])
def test_round12_native_absent_never_uses_advertised_process_identity(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    """Unbound OWNER/READY PIDs never become capture, signal, or reap authority."""

    unrelated = [
        subprocess.Popen(
            [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        for _index in range(2)
    ]
    advertised = {process.pid for process in unrelated}
    owner_payload = {
        "canonical": f"OWNER {unrelated[0].pid}\n".encode("ascii"),
        "malformed": b"owner malformed\n",
        "mismatched": f"OWNER {unrelated[0].pid}\n".encode("ascii"),
    }[profile]
    ready_process_id = unrelated[1].pid if profile == "mismatched" else unrelated[0].pid
    ready_payload = f"READY {ready_process_id}\n".encode("ascii")
    injected = SystemExit(f"round12 impossible native-absent {profile}")
    adoptions: list[int] = []
    captures: list[int] = []
    cleanup_authorities: list[int] = []
    signal_calls: list[tuple[str, int]] = []
    waitpid_calls: list[int] = []
    real_adopt = isolated_process._StartupOwner._adopt_process_id
    real_capture = isolated_process._capture_worker_session
    real_kill = os.kill
    real_killpg = os.killpg
    real_waitpid = os.waitpid

    def impossible_launch(
        owner: isolated_process._StartupOwner,
        *,
        command: list[str],
        request_stream: Any,
        response_writer: isolated_process.PrivatePipeEndpoint,
        status_writer: isolated_process.PrivatePipeEndpoint,
        control_reader: isolated_process.PrivatePipeEndpoint,
        owner_writer: isolated_process.PrivatePipeEndpoint,
        lifetime_writer: isolated_process.PrivatePipeEndpoint,
        pass_fds: tuple[int, ...],
        env: dict[str, str] | None,
    ) -> None:
        del owner, command, request_stream, response_writer, control_reader, pass_fds, env
        os.write(owner_writer.fileno(), owner_payload)
        os.write(status_writer.fileno(), ready_payload)
        raise injected

    def observe_adoption(
        self: isolated_process._StartupOwner,
        process_id: int,
        command: list[str],
    ) -> None:
        if process_id in advertised:
            adoptions.append(process_id)
        real_adopt(self, process_id, command)

    def forbid_capture(
        process: subprocess.Popen[bytes],
        retained_worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerSession:
        if process.pid in advertised:
            captures.append(process.pid)
            raise RuntimeError("advertised PID must not be captured")
        return real_capture(process, retained_worker)

    def observe_cleanup_authority(
        worker: isolated_process._WorkerSession,
    ) -> isolated_process._WorkerCleanupOutcome:
        if worker.process_id in advertised:
            cleanup_authorities.append(worker.process_id)
        return isolated_process._WorkerCleanupOutcome(
            contained=False,
            reaped=False,
            failed=True,
        )

    def forbid_kill(process_id: int, signal_number: int) -> None:
        if process_id in advertised:
            signal_calls.append(("kill", process_id))
            raise AssertionError("advertised PID must not be signalled")
        real_kill(process_id, signal_number)

    def forbid_killpg(process_group_id: int, signal_number: int) -> None:
        if process_group_id in advertised:
            signal_calls.append(("killpg", process_group_id))
            raise AssertionError("advertised PGID must not be signalled")
        real_killpg(process_group_id, signal_number)

    def forbid_waitpid(process_id: int, options: int) -> tuple[int, int]:
        if process_id in advertised:
            waitpid_calls.append(process_id)
            raise AssertionError("advertised PID must not be reaped")
        return real_waitpid(process_id, options)

    monkeypatch.setattr(isolated_process, "_launch_supervisor", impossible_launch)
    monkeypatch.setattr(isolated_process._StartupOwner, "_adopt_process_id", observe_adoption)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", forbid_capture)
    monkeypatch.setattr(
        isolated_process,
        "_stop_worker_synchronously",
        observe_cleanup_authority,
    )
    monkeypatch.setattr(os, "kill", forbid_kill)
    monkeypatch.setattr(os, "killpg", forbid_killpg)
    monkeypatch.setattr(os, "waitpid", forbid_waitpid)

    raised: BaseException | None = None
    unrelated_survived = False
    try:
        try:
            isolated_process.run_isolated_process(
                command=[sys.executable, "-I", "-c", "pass"],
                request=b"{}",
                timeout_seconds=0.2,
                max_response_bytes=32,
                env=isolated_process.sanitized_worker_environment(),
            )
        except BaseException as exc:
            raised = exc
        monkeypatch.setattr(os, "kill", real_kill)
        monkeypatch.setattr(os, "killpg", real_killpg)
        monkeypatch.setattr(os, "waitpid", real_waitpid)
        unrelated_survived = all(process.poll() is None for process in unrelated)
    finally:
        monkeypatch.setattr(os, "kill", real_kill)
        monkeypatch.setattr(os, "killpg", real_killpg)
        monkeypatch.setattr(os, "waitpid", real_waitpid)
        for process in unrelated:
            _terminate_test_domain(process)

    assert isinstance(raised, isolated_process.IsolatedProcessCleanupError)
    assert raised is not injected
    assert adoptions == []
    assert captures == []
    assert cleanup_authorities == []
    assert signal_calls == []
    assert waitpid_calls == []
    assert unrelated_survived


def test_round12_cancel_after_popen_waits_for_owner_then_aborts_and_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-Popen cancellation cannot outrun canonical ownership publication."""

    owner_release = tmp_path / "release-owner"
    owner_receipt = tmp_path / "owner"
    control_receipt = tmp_path / "control"
    exit_receipt = tmp_path / "exiting"
    supervisor_source = "\n".join(
        (
            "import os, sys, time",
            "from pathlib import Path",
            "status, control, owner_fd, lifetime = map(int, sys.argv[1:5])",
            f"while not Path({os.fspath(owner_release)!r}).exists(): time.sleep(0.005)",
            "owner = f'OWNER {os.getpid()}\\n'.encode('ascii')",
            "os.write(owner_fd, owner)",
            "os.close(owner_fd)",
            f"Path({os.fspath(owner_receipt)!r}).write_bytes(owner)",
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))",
            "received = bytearray(os.read(control, 1))",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "os.close(status)",
            "os.close(control)",
            f"Path({os.fspath(exit_receipt)!r}).write_text('exiting', encoding='ascii')",
        )
    )
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    owner_threads: list[threading.Thread] = []
    spawned = threading.Event()
    real_popen_initializer = subprocess.Popen.__init__
    real_thread = threading.Thread

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
            sys.executable,
            "-I",
            "-c",
            supervisor_source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
        ]

    def initialize_and_capture(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_popen_initializer(process, *args, **kwargs)
        raw_command = args[0] if args else kwargs["args"]
        if isinstance(raw_command, list) and any(
            os.fspath(owner_release) in str(argument) for argument in raw_command
        ):
            handles.append(process)
            identities.append(_session_identity(process.pid))
            spawned.set()

    def capture_thread(*args: Any, **kwargs: Any) -> threading.Thread:
        thread = real_thread(*args, **kwargs)
        target = kwargs.get("target")
        if target is None and len(args) > 1:
            target = args[1]
        if isinstance(getattr(target, "__self__", None), isolated_process._StartupOwner):
            owner_threads.append(thread)
        return thread

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_and_capture,
    )
    monkeypatch.setattr(isolated_process.threading, "Thread", capture_thread)
    cancellation = isolated_process.ProcessCancellation()
    results: list[isolated_process.IsolatedProcessResult] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(
                isolated_process.run_isolated_process(
                    command=[sys.executable, "-I", "-c", "pass"],
                    request=b"{}",
                    timeout_seconds=5.0,
                    max_response_bytes=32,
                    cancel_requested=cancellation,
                    env=isolated_process.sanitized_worker_environment(),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    controller = real_thread(target=invoke, name="round12-cancel-before-owner")
    returned_before_owner = True
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    try:
        controller.start()
        assert spawned.wait(2.0)
        cancellation.set()
        returned_before_owner = not controller.is_alive()
        owner_release.write_text("release", encoding="ascii")
        controller.join(timeout=4.0)
        if handles:
            leader_not_running = _process_not_running(handles[0].pid)
            leader_reaped = _exact_child_was_reaped(handles[0].pid)
            group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        owner_release.write_text("release", encoding="ascii")
        for handle in handles:
            _terminate_test_domain(handle)
        controller.join(timeout=2.0)

    assert not returned_before_owner
    assert errors == []
    assert results == [isolated_process.IsolatedProcessResult(None, "cancelled")]
    assert len(handles) == 1
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert _bounded_bytes(owner_receipt) == f"OWNER {handles[0].pid}\n".encode("ascii")
    assert _bounded_bytes(control_receipt) == _ABORT
    assert _bounded_bytes(exit_receipt) == b"exiting"
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()
    assert len(owner_threads) == 1
    assert all(not thread.is_alive() for thread in owner_threads)


def test_round12_stalled_ready_cancellation_joins_owner_and_descendants_before_outer_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A six-hour READY stall cannot outrun owner join, lifetime exit, or terminal caller work."""

    domain_receipt = tmp_path / "domain-pids"
    control_receipt = tmp_path / "control"
    owner_receipt = tmp_path / "owner"
    lifetime_receipt = tmp_path / "lifetime"
    outer_receipt = tmp_path / "outer.json"
    descendant_source = "import time; time.sleep(30)"
    supervisor_source = "\n".join(
        (
            "import os, signal, subprocess, sys, time",
            "from pathlib import Path",
            "publication, control, owner_fd, lifetime = map(int, sys.argv[1:5])",
            "owner = f'OWNER {os.getpid()}\\n'.encode('ascii')",
            "os.write(owner_fd, owner)",
            "os.close(owner_fd)",
            f"Path({os.fspath(owner_receipt)!r}).write_bytes(owner)",
            f"child = subprocess.Popen([sys.executable, '-I', '-c', {descendant_source!r}])",
            f"Path({os.fspath(domain_receipt)!r}).write_text(f'{{os.getpid()}} {{child.pid}}', encoding='ascii')",
            "if control < 0: time.sleep(30)",
            "received = bytearray()",
            "while len(received) < 2:",
            "    chunk = os.read(control, 2 - len(received))",
            "    if not chunk: break",
            "    received.extend(chunk)",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "os.close(control)",
            "os.kill(child.pid, signal.SIGKILL)",
            "child.wait()",
            f"Path({os.fspath(lifetime_receipt)!r}).write_text('exiting', encoding='ascii')",
        )
    )
    owner_threads: list[threading.Thread] = []
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    real_thread = threading.Thread
    real_popen_initializer = subprocess.Popen.__init__

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
            sys.executable,
            "-I",
            "-c",
            supervisor_source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
        ]

    def capture_thread(*args: Any, **kwargs: Any) -> threading.Thread:
        thread = real_thread(*args, **kwargs)
        target = kwargs.get("target")
        if target is None and len(args) > 1:
            target = args[1]
        if isinstance(getattr(target, "__self__", None), isolated_process._StartupOwner):
            owner_threads.append(thread)
        return thread

    def initialize_and_capture(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_popen_initializer(process, *args, **kwargs)
        command = args[0] if args else kwargs["args"]
        if isinstance(command, list) and any(
            os.fspath(domain_receipt) in str(argument) for argument in command
        ):
            handles.append(process)
            identities.append(_session_identity(process.pid))

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(isolated_process.threading, "Thread", capture_thread)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_and_capture,
    )
    cancellation = isolated_process.ProcessCancellation()
    results: list[isolated_process.IsolatedProcessResult] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(
                isolated_process.run_isolated_process(
                    command=[sys.executable, "-I", "-c", "raise SystemExit(0)"],
                    request=b"{}",
                    timeout_seconds=21_600.0,
                    max_response_bytes=32,
                    cancel_requested=cancellation,
                    env=isolated_process.sanitized_worker_environment(),
                )
            )
            outer_receipt.write_text(
                json.dumps(
                    {
                        "lifetime": lifetime_receipt.exists(),
                        "owner_count": len(owner_threads),
                        "owners_joined": bool(owner_threads)
                        and all(not thread.is_alive() for thread in owner_threads),
                    }
                ),
                encoding="ascii",
            )
        except BaseException as exc:
            errors.append(exc)

    controller = real_thread(target=invoke, name="round12-outer-controller")
    process_ids = (0, 0)
    returned_before_test_cleanup = False
    process_states = (False, False)
    observed_control: bytes | None = None
    observed_owner: bytes | None = None
    lifetime_exiting = False
    outer_state: dict[str, bool] | None = None
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    try:
        controller.start()
        process_ids = _bounded_pid_pair(domain_receipt)
        cancellation.set()
        controller.join(timeout=5.0)
        returned_before_test_cleanup = not controller.is_alive()
        process_states = tuple(_process_not_running(pid) for pid in process_ids)  # type: ignore[assignment]
        observed_control = _optional_bytes(control_receipt)
        observed_owner = _optional_bytes(owner_receipt)
        lifetime_exiting = _optional_bytes(lifetime_receipt) == b"exiting"
        if outer_receipt.exists():
            outer_state = json.loads(outer_receipt.read_text(encoding="ascii"))
        if handles:
            leader_reaped = _exact_child_was_reaped(handles[0].pid)
            group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        if controller.is_alive() and process_ids[0] > 0:
            with suppress(ProcessLookupError):
                os.killpg(process_ids[0], signal.SIGKILL)
        controller.join(timeout=3.0)
        for handle in handles:
            _terminate_test_domain(handle)

    assert returned_before_test_cleanup
    assert errors == []
    assert results == [isolated_process.IsolatedProcessResult(None, "cancelled")]
    assert identities == [(process_ids[0], process_ids[0], process_ids[0])]
    assert process_states == (True, True)
    assert leader_reaped
    assert group_after_return == ()
    assert observed_control == _ABORT
    assert observed_owner == f"OWNER {process_ids[0]}\n".encode("ascii")
    assert lifetime_exiting
    assert len(owner_threads) == 1
    assert all(not thread.is_alive() for thread in owner_threads)
    assert outer_state == {"lifetime": True, "owner_count": 1, "owners_joined": True}


def test_round12_start_barrier_and_lifetime_fd_do_not_cross_exec_or_public_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """START is exact, lifetime remains open before exec, and worker plus child never inherit it."""

    control_receipt = tmp_path / "control"
    owner_receipt = tmp_path / "owner"
    lifetime_pre_start_receipt = tmp_path / "lifetime-pre-start"
    lifetime_post_start_receipt = tmp_path / "lifetime-post-start"
    lifetime_exit_receipt = tmp_path / "lifetime-exit"
    release_exec = tmp_path / "release-exec"
    worker_receipt = tmp_path / "worker.json"
    descendant_receipt = tmp_path / "descendant.json"
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    real_popen_initializer = subprocess.Popen.__init__

    def synthetic_supervisor(
        *,
        status_descriptor: int,
        control_descriptor: int,
        owner_descriptor: int,
        lifetime_descriptor: int,
        command: list[str],
    ) -> list[str]:
        del command
        control = control_descriptor
        lifetime = lifetime_descriptor
        identity: tuple[int, int, int, int] | None = None
        if lifetime >= 0:
            status = os.fstat(lifetime)
            identity = (status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode), status.st_rdev)
        descendant_source = "\n".join(
            (
                "import json, os, sys",
                "from pathlib import Path",
                "descriptor = int(sys.argv[1])",
                "expected = tuple(json.loads(sys.argv[2]))",
                "leaked = False",
                "try: status = os.fstat(descriptor)",
                "except OSError: pass",
                "else: leaked = (status.st_dev, status.st_ino, status.st_mode & 0o170000, status.st_rdev) == expected",
                "Path(sys.argv[3]).write_text(json.dumps({'leaked': leaked}), encoding='ascii')",
            )
        )
        worker_source = "\n".join(
            (
                "import json, os, subprocess, sys",
                "from pathlib import Path",
                "descriptor = int(sys.argv[1])",
                "expected = tuple(json.loads(sys.argv[2]))",
                "leaked = False",
                "try: status = os.fstat(descriptor)",
                "except OSError: pass",
                "else: leaked = (status.st_dev, status.st_ino, status.st_mode & 0o170000, status.st_rdev) == expected",
                f"subprocess.run([sys.executable, '-I', '-c', {descendant_source!r}, sys.argv[1], sys.argv[2], {os.fspath(descendant_receipt)!r}], check=True, close_fds=False)",
                f"Path({os.fspath(worker_receipt)!r}).write_text(json.dumps({{'leaked': leaked}}), encoding='ascii')",
                "sys.stdout.buffer.write(b'ok')",
            )
        )
        supervisor_source = "\n".join(
            (
                "import os, sys, time",
                "from pathlib import Path",
                "publication, control, owner_fd, lifetime = map(int, sys.argv[1:5])",
                "owner = f'OWNER {os.getpid()}\\n'.encode('ascii')",
                "os.write(owner_fd, owner)",
                "os.close(owner_fd)",
                f"Path({os.fspath(owner_receipt)!r}).write_bytes(owner)",
                "lifetime_status = os.fstat(lifetime)",
                "lifetime_identity = (lifetime_status.st_dev, lifetime_status.st_ino, lifetime_status.st_mode & 0o170000, lifetime_status.st_rdev)",
                f"Path({os.fspath(lifetime_pre_start_receipt)!r}).write_text('open' if lifetime_identity == {identity!r} else 'changed', encoding='ascii')",
                "os.write(publication, f'READY {os.getpid()}\\n'.encode('ascii'))",
                "if control < 0: time.sleep(30)",
                "received = bytearray(os.read(control, 1))",
                "if bytes(received) == b'S':",
                "    os.write(publication, b'CONTROL S\\n')",
                "received.extend(os.read(control, 1))",
                f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
                "os.close(control)",
                "if bytes(received) != b'S': raise SystemExit(125)",
                "worker = os.fork()",
                "if worker == 0:",
                "    os.close(publication)",
                "    os.close(lifetime)",
                f"    while not Path({os.fspath(release_exec)!r}).exists(): time.sleep(0.005)",
                f"    os.execvpe(sys.executable, [sys.executable, '-I', '-c', {worker_source!r}, str(lifetime), {json.dumps(identity)!r}], os.environ)",
                "lifetime_status = os.fstat(lifetime)",
                "lifetime_identity = (lifetime_status.st_dev, lifetime_status.st_ino, lifetime_status.st_mode & 0o170000, lifetime_status.st_rdev)",
                f"Path({os.fspath(lifetime_post_start_receipt)!r}).write_text('open' if lifetime_identity == {identity!r} else 'changed', encoding='ascii')",
                "os.write(publication, f'STARTED {worker}\\n'.encode('ascii'))",
                "os.close(publication)",
                "observed_worker, wait_status = os.waitpid(worker, 0)",
                "if observed_worker != worker: raise SystemExit(126)",
                f"Path({os.fspath(lifetime_exit_receipt)!r}).write_text('exiting', encoding='ascii')",
                "raise SystemExit(os.waitstatus_to_exitcode(wait_status))",
            )
        )
        return [
            sys.executable,
            "-I",
            "-c",
            supervisor_source,
            str(status_descriptor),
            str(control),
            str(owner_descriptor),
            str(lifetime),
        ]

    def initialize_and_capture(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_popen_initializer(process, *args, **kwargs)
        command = args[0] if args else kwargs["args"]
        if isinstance(command, list) and any(
            os.fspath(control_receipt) in str(argument) for argument in command
        ):
            handles.append(process)
            identities.append(_session_identity(process.pid))

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_and_capture,
    )
    cancellation = isolated_process.ProcessCancellation()
    results: list[isolated_process.IsolatedProcessResult] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(
                isolated_process.run_isolated_process(
                    command=[sys.executable, "-I", "-c", "raise SystemExit(90)"],
                    request=b"{}",
                    timeout_seconds=_WORKER_DESCENDANT_SUCCESS_SECONDS,
                    max_response_bytes=32,
                    cancel_requested=cancellation,
                    env=isolated_process.sanitized_worker_environment(),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    controller = threading.Thread(target=invoke, name="round12-start-barrier")
    returned_before_release = False
    observed_control: bytes | None = None
    observed_owner: bytes | None = None
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    success_deadline = time.monotonic() + _WORKER_DESCENDANT_SUCCESS_SECONDS
    try:
        controller.start()
        observed_control = _bounded_bytes(
            control_receipt,
            timeout=_remaining_success_seconds(success_deadline),
        )
        observed_owner = _bounded_bytes(
            owner_receipt,
            timeout=_remaining_success_seconds(success_deadline),
        )
        returned_before_release = not controller.is_alive()
        assert not worker_receipt.exists()
        assert not descendant_receipt.exists()
        release_exec.write_text("release", encoding="ascii")
        controller.join(timeout=_remaining_success_seconds(success_deadline))
        if handles:
            leader_not_running = _process_not_running(handles[0].pid)
            leader_reaped = _exact_child_was_reaped(handles[0].pid)
            group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        release_exec.write_text("release", encoding="ascii")
        if controller.is_alive():
            cancellation.set()
        controller.join(timeout=3.0)
        for handle in handles:
            _terminate_test_domain(handle)

    assert observed_control == _START
    assert len(handles) == 1
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert observed_owner == f"OWNER {handles[0].pid}\n".encode("ascii")
    assert not returned_before_release
    assert errors == []
    assert results == [isolated_process.IsolatedProcessResult(b"ok", None)]
    assert (
        _bounded_bytes(
            lifetime_pre_start_receipt,
            timeout=_remaining_success_seconds(success_deadline),
        )
        == b"open"
    )
    assert (
        _bounded_bytes(
            lifetime_post_start_receipt,
            timeout=_remaining_success_seconds(success_deadline),
        )
        == b"open"
    )
    assert (
        _bounded_bytes(
            lifetime_exit_receipt,
            timeout=_remaining_success_seconds(success_deadline),
        )
        == b"exiting"
    )
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()
    assert json.loads(worker_receipt.read_text(encoding="ascii")) == {"leaked": False}
    assert json.loads(descendant_receipt.read_text(encoding="ascii")) == {"leaked": False}


def test_round12_supervisor_uses_resolved_isolated_entrypoint_not_cwd_or_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supervisor launch resolves the trusted absolute script under isolated Python mode."""

    marker = tmp_path / "shadow-executed"
    package = tmp_path / "invoice_agents"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "spawn_supervisor.py").write_text(
        "\n".join(
            (
                "import os, sys",
                "from pathlib import Path",
                f"Path({os.fspath(marker)!r}).write_text('executed', encoding='ascii')",
                "descriptor = int(sys.argv[1])",
                "os.write(descriptor, f'{os.getpid()}\\n'.encode('ascii'))",
                "os.close(descriptor)",
                "os.execvpe(sys.argv[2], sys.argv[2:], os.environ)",
            )
        ),
        encoding="utf-8",
    )
    captured: list[list[str]] = []
    real_popen_initializer = subprocess.Popen.__init__

    def initialize_and_capture_command(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        command = args[0] if args else kwargs["args"]
        is_supervisor = isinstance(command, list) and any(
            "spawn_supervisor" in str(arg) for arg in command
        )
        if is_supervisor:
            captured.append(command)
        real_popen_initializer(process, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_and_capture_command,
    )
    environment = isolated_process.sanitized_worker_environment()
    environment["PYTHONPATH"] = os.fspath(tmp_path)

    result = isolated_process.run_isolated_process(
        command=[sys.executable, "-I", "-c", "import sys; sys.stdout.buffer.write(b'ok')"],
        request=b"{}",
        # Launch+OWNER, READY, CONTROL, STARTED, and response must all finish;
        # expiry remains a hard failure of this positive isolation proof.
        timeout_seconds=_SINGLE_WORKER_SUCCESS_SECONDS,
        max_response_bytes=32,
        env=environment,
    )

    expected = Path(isolated_process.__file__).with_name("spawn_supervisor.py").resolve(strict=True)
    assert result == isolated_process.IsolatedProcessResult(b"ok", None)
    assert not marker.exists()
    assert len(captured) == 1
    assert Path(captured[0][0]).resolve(strict=True) == Path(sys.executable).resolve(strict=True)
    assert captured[0][1] == "-I"
    assert Path(captured[0][2]).is_absolute()
    assert Path(captured[0][2]).resolve(strict=True) == expected


@pytest.mark.parametrize(
    "close_mode",
    ["normal", "pre-error", "pre-control", "post-error", "post-control"],
)
def test_round12_ambiguous_close_is_single_attempt_sticky_and_never_probes_raw_fd(
    close_mode: str,
) -> None:
    """Ambiguous close poisons repeated admission without fstat, retry, or silent clearing."""

    code = "\n".join(
        (
            "import json, os, socket, stat",
            "from invoice_agents import isolated_process as target",
            f"mode = {close_mode!r}",
            "reader, writer = target.private_pipe_channel()",
            "descriptor = reader.fileno()",
            "original = os.fstat(descriptor)",
            "original_identity = (original.st_dev, original.st_ino, stat.S_IFMT(original.st_mode), original.st_rdev)",
            "replacement_source = os.dup(descriptor)",
            "real_close = target.os.close",
            "real_fstat = target.os.fstat",
            "real_closerange = getattr(target.os, 'closerange', None)",
            "real_socket_close = socket.close",
            "counts = {'close': 0, 'retry': 0, 'fstat': 0, 'closerange': 0, 'socket': 0, 'probe': 0}",
            "def injected_close(candidate):",
            "    if candidate != descriptor:",
            "        return real_close(candidate)",
            "    counts['close'] += 1",
            "    if counts['close'] > 1:",
            "        counts['retry'] += 1",
            "        return real_close(candidate)",
            "    if mode == 'normal':",
            "        return real_close(candidate)",
            "    if mode.startswith('post'):",
            "        real_close(candidate)",
            "        os.dup2(replacement_source, candidate, inheritable=False)",
            "    if mode.endswith('control'):",
            "        raise KeyboardInterrupt('round12 ambiguous close control')",
            "    if mode != 'normal':",
            "        raise OSError('round12 ambiguous close error')",
            "    return None",
            "def observed_fstat(candidate):",
            "    if candidate == descriptor:",
            "        counts['fstat'] += 1",
            "        counts['probe'] += 1",
            "    return real_fstat(candidate)",
            "def observed_closerange(low, high):",
            "    if low <= descriptor < high: counts['closerange'] += 1",
            "    if real_closerange is not None: return real_closerange(low, high)",
            "def observed_socket_close(candidate):",
            "    if candidate == descriptor: counts['socket'] += 1",
            "    return real_socket_close(candidate)",
            "def restore_hooks():",
            "    target.os.close = real_close",
            "    target.os.fstat = real_fstat",
            "    if real_closerange is not None: target.os.closerange = real_closerange",
            "    socket.close = real_socket_close",
            "target.os.close = injected_close",
            "target.os.fstat = observed_fstat",
            "if real_closerange is not None: target.os.closerange = observed_closerange",
            "socket.close = observed_socket_close",
            "try:",
            "    reader.close()",
            "except BaseException:",
            "    pass",
            "if mode == 'normal': restore_hooks()",
            "try:",
            "    current = real_fstat(descriptor)",
            "    descriptor_live = True",
            "    current_identity = (current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode), current.st_rdev)",
            "except OSError:",
            "    descriptor_live = False",
            "    current_identity = None",
            "admissions = []",
            "for _ in range(2):",
            "    try:",
            "        new_reader, new_writer = target.private_pipe_channel()",
            "    except target.IsolatedProcessCleanupError:",
            "        admissions.append('blocked')",
            "    else:",
            "        admissions.append('admitted')",
            "        new_reader.close(); new_writer.close()",
            "if descriptor_live: real_close(descriptor)",
            "real_close(replacement_source)",
            "writer.close()",
            "try:",
            "    final_reader, final_writer = target.private_pipe_channel()",
            "except target.IsolatedProcessCleanupError:",
            "    after_cleanup = 'blocked'",
            "else:",
            "    after_cleanup = 'admitted'",
            "    final_reader.close(); final_writer.close()",
            "restore_hooks()",
            "print(json.dumps({'counts': counts, 'descriptor_live': descriptor_live, 'original': original_identity, 'current': current_identity, 'admissions': admissions, 'after_cleanup': after_cleanup}, sort_keys=True))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    observed = json.loads(completed.stdout)

    assert completed.stderr == ""
    assert observed["counts"] == {
        "close": 1,
        "closerange": 0,
        "fstat": 0,
        "probe": 0,
        "retry": 0,
        "socket": 0,
    }
    if close_mode == "normal":
        assert observed["descriptor_live"] is False
        assert observed["admissions"] == ["admitted", "admitted"]
        assert observed["after_cleanup"] == "admitted"
    else:
        assert observed["descriptor_live"] is True
        assert observed["admissions"] == ["blocked", "blocked"]
        assert observed["after_cleanup"] == "blocked"
        if close_mode.startswith("pre"):
            assert tuple(observed["current"]) == tuple(observed["original"])


def test_round12_preset_cancellation_short_circuits_before_any_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-set cancellation needs no child and remains cancelled after local cleanup."""

    calls = 0

    def forbidden_initializer(
        _process: subprocess.Popen[bytes],
        *_args: object,
        **_kwargs: object,
    ) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("pre-set cancellation attempted to spawn")

    cancellation = isolated_process.ProcessCancellation()
    cancellation.set()
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        forbidden_initializer,
    )

    result = isolated_process.run_isolated_process(
        command=[sys.executable, "-I", "-c", "raise SystemExit(0)"],
        request=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=32,
        cancel_requested=cancellation,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert calls == 0
    assert result == isolated_process.IsolatedProcessResult(None, "cancelled")


def test_round12_partial_frame_cancellation_rejects_child_read_without_post_cancel_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation between real frame chunks produces EOF rejection and no credential receipt."""

    reader, writer = isolated_process.private_pipe_channel()
    inherited_reader = reader.detach()
    receipt = tmp_path / "credential.bin"
    rejected = tmp_path / "rejected"
    child_source = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from invoice_agents.lifecycle_worker import _read_private_credential",
            "descriptor = int(sys.argv[1])",
            "try:",
            "    credential = _read_private_credential(descriptor)",
            "except BaseException:",
            f"    Path({os.fspath(rejected)!r}).write_text('rejected', encoding='ascii')",
            "else:",
            f"    Path({os.fspath(receipt)!r}).write_bytes(credential)",
            "    credential[:] = bytes(len(credential))",
        )
    )
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", child_source, str(inherited_reader)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=(inherited_reader,),
        env=isolated_process.sanitized_worker_environment(),
    )
    os.close(inherited_reader)
    credential = bytearray(b"round12-private-frame-canary" * 600)
    header = struct.pack("!I", len(credential))
    cancellation = isolated_process.ProcessCancellation()
    partial_written = threading.Event()
    release_legacy_sender = threading.Event()
    target_descriptor = writer.fileno()
    payload_written = 0
    post_cancel_writes = 0
    real_write = isolated_process.os.write
    results: list[bool] = []
    errors: list[BaseException] = []

    def controlled_write(descriptor: int, payload: object) -> int:
        nonlocal payload_written, post_cancel_writes
        if descriptor != target_descriptor:
            return real_write(descriptor, payload)  # type: ignore[arg-type]
        view = memoryview(payload)  # type: ignore[arg-type]
        encoded = bytes(view)
        if encoded == header:
            return real_write(descriptor, view)
        if payload_written == 0 and encoded and set(encoded) != {0}:
            written = real_write(descriptor, view[:7])
            payload_written += written
            partial_written.set()
            return written
        if not release_legacy_sender.is_set():
            raise BlockingIOError
        if cancellation.is_set():
            post_cancel_writes += 1
        written = real_write(descriptor, view)
        payload_written += written
        return written

    monkeypatch.setattr(isolated_process.os, "write", controlled_write)

    def send() -> None:
        try:
            results.append(
                isolated_process.send_private_frame(
                    writer,
                    credential,
                    max_payload_bytes=len(credential),
                    deadline=time.monotonic() + 3.0,
                    cancel_requested=cancellation,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    sender = threading.Thread(target=send, name="round12-frame-sender")
    canceller = threading.Thread(target=cancellation.set, name="round12-frame-canceller")
    cancellation_returned_before_release = False
    sender_stopped_before_release = False
    child_returncode: int | None = None
    try:
        sender.start()
        assert partial_written.wait(1.0)
        canceller.start()
        canceller.join(timeout=0.5)
        sender.join(timeout=0.5)
        cancellation_returned_before_release = not canceller.is_alive()
        sender_stopped_before_release = not sender.is_alive()
        release_legacy_sender.set()
        sender.join(timeout=2.0)
        canceller.join(timeout=2.0)
        writer.close()
        child_returncode = child.wait(timeout=2.0)
    finally:
        release_legacy_sender.set()
        sender.join(timeout=2.0)
        canceller.join(timeout=2.0)
        if not writer.closed:
            writer.close()
        if child.poll() is None:
            child.kill()
        with suppress(subprocess.TimeoutExpired):
            child.wait(timeout=2.0)
        for index in range(len(credential)):
            credential[index] = 0

    assert cancellation_returned_before_release
    assert sender_stopped_before_release
    assert not sender.is_alive()
    assert not canceller.is_alive()
    assert results == [False]
    assert errors == []
    assert post_cancel_writes == 0
    assert child_returncode == 0
    assert rejected.read_text(encoding="ascii") == "rejected"
    assert not receipt.exists()
