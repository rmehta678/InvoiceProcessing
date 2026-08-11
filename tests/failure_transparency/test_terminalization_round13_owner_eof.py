"""Regression for terminal OWNER publication before endpoint finality."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from invoice_agents import isolated_process


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


def _bounded_bytes(path: Path, *, timeout: float = 2.0) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return path.read_bytes()
        except OSError:
            time.sleep(0.01)
    raise AssertionError(f"bounded receipt was not published: {path}")


def _optional_bytes(path: Path, *, timeout: float = 0.2) -> bytes | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return path.read_bytes()
        except OSError:
            time.sleep(0.01)
    return None


def _process_not_running(process_id: int) -> bool:
    completed = subprocess.run(
        ["/bin/ps", "-o", "stat=", "-p", str(process_id)],
        check=False,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    if completed.returncode not in {0, 1}:
        raise AssertionError(f"ps could not inspect native PID {process_id}")
    return completed.returncode == 1 and not completed.stdout.strip()


def _exact_child_was_reaped(process_id: int) -> bool:
    try:
        os.waitpid(process_id, os.WNOHANG)
    except ChildProcessError:
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


def _force_test_cleanup(process_id: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process_id, signal.SIGKILL)
    with suppress(ChildProcessError):
        os.waitpid(process_id, 0)


def test_delayed_duplicate_owner_after_consumption_never_starts_or_receives_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OWNER acceptance must require endpoint EOF, not a point-in-time trailing read."""

    owner_consumed = tmp_path / "owner-consumed"
    poll_completed = tmp_path / "post-owner-poll-completed"
    owner_waiting_for_eof = tmp_path / "owner-waiting-for-eof"
    allow_duplicate = tmp_path / "allow-duplicate"
    duplicate_written = tmp_path / "duplicate-written"
    control_receipt = tmp_path / "control"
    credential_receipt = tmp_path / "credential"
    reader, writer = isolated_process.private_pipe_channel()
    credential = bytearray(b"round13-owner-eof-private-canary")
    credential_reader = reader.fileno()
    source = "\n".join(
        (
            "import os, struct, sys, time",
            "from pathlib import Path",
            "status, control, owner_fd, lifetime, credential = map(int, sys.argv[1:6])",
            "owner = f'OWNER {os.getpid()}\\n'.encode('ascii')",
            "os.write(owner_fd, owner)",
            f"consumed = Path({os.fspath(owner_consumed)!r})",
            "while not consumed.exists(): time.sleep(0.005)",
            f"poll_completed = Path({os.fspath(poll_completed)!r})",
            "while not poll_completed.exists(): time.sleep(0.005)",
            f"owner_waiting = Path({os.fspath(owner_waiting_for_eof)!r})",
            "while not owner_waiting.exists(): time.sleep(0.005)",
            f"allow_duplicate = Path({os.fspath(allow_duplicate)!r})",
            "while not allow_duplicate.exists(): time.sleep(0.005)",
            "os.write(owner_fd, owner)",
            f"Path({os.fspath(duplicate_written)!r}).write_bytes(owner)",
            "os.close(owner_fd)",
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))",
            "received = bytearray(os.read(control, 1))",
            "if bytes(received) == b'S': os.write(status, b'CONTROL S\\n')",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "if bytes(received) == b'S':",
            "    os.write(status, f'STARTED {os.getpid()}\\n'.encode('ascii'))",
            "    header = os.read(credential, 4)",
            "    if len(header) == 4:",
            "        remaining = struct.unpack('!I', header)[0]",
            "        payload = bytearray()",
            "        while len(payload) < remaining:",
            "            chunk = os.read(credential, remaining - len(payload))",
            "            if not chunk: break",
            "            payload.extend(chunk)",
            f"        Path({os.fspath(credential_receipt)!r}).write_bytes(payload)",
            "    sys.stdout.buffer.write(b'ok')",
            "    sys.stdout.buffer.flush()",
            "os.close(status)",
            "os.close(control)",
            "os.close(lifetime)",
        )
    )
    real_initializer = subprocess.Popen.__init__
    real_read = isolated_process.os.read
    real_selector_factory = isolated_process.selectors.DefaultSelector
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    owner_was_consumed = False
    owner_descriptor: int | None = None
    post_owner_poll_count = 0

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
            source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
            str(credential_reader),
        ]

    def capture_native(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_initializer(process, *args, **kwargs)
        handles.append(process)
        identities.append((process.pid, os.getpgid(process.pid), os.getsid(process.pid)))

    def publish_consumption(descriptor: int, count: int) -> bytes:
        nonlocal owner_descriptor, owner_was_consumed
        payload = real_read(descriptor, count)
        if not owner_was_consumed and payload.startswith(b"OWNER "):
            owner_descriptor = descriptor
            owner_was_consumed = True
            owner_consumed.write_text("consumed", encoding="ascii")
        return payload

    class ObservedSelector:
        """Publish a barrier only after the first post-frame poll has completed."""

        def __init__(self) -> None:
            self._delegate = real_selector_factory()
            self._descriptors: set[int] = set()

        def __enter__(self) -> ObservedSelector:
            self._delegate.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._delegate.__exit__(*args)

        def register(self, fileobj: int, events: int, data: object = None) -> object:
            self._descriptors.add(fileobj)
            return self._delegate.register(fileobj, events, data)

        def select(self, timeout: float | None = None) -> object:
            nonlocal post_owner_poll_count
            events = self._delegate.select(timeout)
            if owner_was_consumed and owner_descriptor in self._descriptors:
                assert events == []
                post_owner_poll_count += 1
                if post_owner_poll_count == 1:
                    poll_completed.write_text("completed", encoding="ascii")
                elif post_owner_poll_count == 2:
                    owner_waiting_for_eof.write_text("waiting", encoding="ascii")
            return events

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(isolated_process, "_initialize_reserved_process", capture_native)
    monkeypatch.setattr(isolated_process.os, "read", publish_consumption)
    monkeypatch.setattr(isolated_process.selectors, "DefaultSelector", ObservedSelector)

    results: list[isolated_process.IsolatedProcessResult] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(
                isolated_process.run_isolated_process(
                    command=[sys.executable, "-I", "-c", "pass"],
                    request=b"{}",
                    timeout_seconds=1.0,
                    max_response_bytes=32,
                    pass_fds=(credential_reader,),
                    private_input=isolated_process.PrivatePipeInput(
                        reader=reader,
                        writer=writer,
                        payload=credential,
                        max_payload_bytes=len(credential),
                    ),
                    env=isolated_process.sanitized_worker_environment(),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    controller = threading.Thread(target=invoke, name="round13-owner-eof-controller")
    pending_after_completed_poll = False
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    observed_control: bytes | None = None
    observed_credential: bytes | None = None
    try:
        controller.start()
        _bounded_bytes(poll_completed)
        _bounded_bytes(owner_waiting_for_eof)
        pending_after_completed_poll = controller.is_alive()
        allow_duplicate.write_text("release", encoding="ascii")
        controller.join(timeout=2.0)
        assert len(handles) == 1
        leader_not_running = _process_not_running(handles[0].pid)
        leader_reaped = _exact_child_was_reaped(handles[0].pid)
        group_after_return = _process_group_snapshot(handles[0].pid)
        observed_control = _optional_bytes(control_receipt)
        observed_credential = _optional_bytes(credential_receipt)
    finally:
        poll_completed.write_text("completed", encoding="ascii")
        owner_waiting_for_eof.write_text("waiting", encoding="ascii")
        allow_duplicate.write_text("release", encoding="ascii")
        controller.join(timeout=2.0)
        if not reader.closed:
            reader.close()
        if not writer.closed:
            writer.close()
        for handle in handles:
            _force_test_cleanup(handle.pid)

    assert _bounded_bytes(duplicate_written).startswith(b"OWNER ")
    assert owner_was_consumed
    assert post_owner_poll_count >= 2
    assert pending_after_completed_poll
    assert not controller.is_alive()
    assert errors == []
    assert results == [isolated_process.IsolatedProcessResult(None, "start")]
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert observed_control == b"A"
    assert observed_credential is None
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()


def test_delayed_lifetime_data_before_ready_never_crosses_start_or_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifetime validity is monitored continuously until START, not sampled once."""

    lifetime_observed_open = tmp_path / "lifetime-observed-open"
    lifetime_data_written = tmp_path / "lifetime-data-written"
    control_receipt = tmp_path / "control"
    credential_receipt = tmp_path / "credential"
    reader, writer = isolated_process.private_pipe_channel()
    credential = bytearray(b"round13-delayed-lifetime-private-canary")
    credential_reader = reader.fileno()
    source = "\n".join(
        (
            "import os, struct, sys, time",
            "from pathlib import Path",
            "status, control, owner_fd, lifetime, credential = map(int, sys.argv[1:6])",
            "os.write(owner_fd, f'OWNER {os.getpid()}\\n'.encode('ascii'))",
            "os.close(owner_fd)",
            f"observed = Path({os.fspath(lifetime_observed_open)!r})",
            "while not observed.exists(): time.sleep(0.005)",
            "os.write(lifetime, b'X')",
            f"Path({os.fspath(lifetime_data_written)!r}).write_bytes(b'X')",
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))",
            "received = bytearray(os.read(control, 1))",
            "if bytes(received) == b'S': os.write(status, b'CONTROL S\\n')",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "if bytes(received) == b'S':",
            "    os.write(status, f'STARTED {os.getpid()}\\n'.encode('ascii'))",
            "    header = os.read(credential, 4)",
            "    if len(header) == 4:",
            "        remaining = struct.unpack('!I', header)[0]",
            "        payload = bytearray()",
            "        while len(payload) < remaining:",
            "            chunk = os.read(credential, remaining - len(payload))",
            "            if not chunk: break",
            "            payload.extend(chunk)",
            f"        Path({os.fspath(credential_receipt)!r}).write_bytes(payload)",
            "    sys.stdout.buffer.write(b'ok')",
            "    sys.stdout.buffer.flush()",
            "os.close(status)",
            "os.close(control)",
            "os.close(lifetime)",
        )
    )
    real_initializer = subprocess.Popen.__init__
    real_read = isolated_process.os.read
    real_fstat = isolated_process.os.fstat
    real_read_ready = isolated_process._read_ready_while_lifetime_open
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    owner_frame_observed = False
    lifetime_open_observed = False
    lifetime_reader_descriptor: int | None = None
    lifetime_reader_identity: tuple[int, int, int, int] | None = None
    lifetime_eagain_descriptors: list[int] = []

    def pipe_identity(descriptor: int) -> tuple[int, int, int, int]:
        status = real_fstat(descriptor)
        return (
            status.st_dev,
            status.st_ino,
            status.st_mode & 0o170000,
            status.st_rdev,
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
            sys.executable,
            "-I",
            "-c",
            source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
            str(credential_reader),
        ]

    def bind_exact_lifetime_reader(
        status_reader: isolated_process.PrivatePipeEndpoint,
        lifetime_reader: isolated_process.PrivatePipeEndpoint,
        **kwargs: Any,
    ) -> bytes:
        nonlocal lifetime_reader_descriptor, lifetime_reader_identity
        descriptor = lifetime_reader.fileno()
        identity = pipe_identity(descriptor)
        if lifetime_reader_descriptor is None:
            lifetime_reader_descriptor = descriptor
            lifetime_reader_identity = identity
        else:
            assert (descriptor, identity) == (
                lifetime_reader_descriptor,
                lifetime_reader_identity,
            )
        return real_read_ready(status_reader, lifetime_reader, **kwargs)

    def capture_native(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_initializer(process, *args, **kwargs)
        handles.append(process)
        identities.append((process.pid, os.getpgid(process.pid), os.getsid(process.pid)))

    def publish_lifetime_observation(descriptor: int, count: int) -> bytes:
        nonlocal lifetime_open_observed, lifetime_reader_descriptor
        nonlocal lifetime_reader_identity, owner_frame_observed
        try:
            payload = real_read(descriptor, count)
        except BlockingIOError:
            if (
                owner_frame_observed
                and lifetime_reader_descriptor is not None
                and descriptor == lifetime_reader_descriptor
                and pipe_identity(descriptor) == lifetime_reader_identity
            ):
                lifetime_eagain_descriptors.append(descriptor)
                lifetime_open_observed = True
                lifetime_observed_open.write_text("open", encoding="ascii")
            raise
        if payload.startswith(b"OWNER "):
            owner_frame_observed = True
        return payload

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(isolated_process, "_initialize_reserved_process", capture_native)
    monkeypatch.setattr(
        isolated_process,
        "_read_ready_while_lifetime_open",
        bind_exact_lifetime_reader,
    )
    monkeypatch.setattr(isolated_process.os, "read", publish_lifetime_observation)

    cancellation = isolated_process.ProcessCancellation()
    results: list[isolated_process.IsolatedProcessResult] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(
                isolated_process.run_isolated_process(
                    command=[sys.executable, "-I", "-c", "pass"],
                    request=b"{}",
                    timeout_seconds=1.0,
                    max_response_bytes=32,
                    pass_fds=(credential_reader,),
                    private_input=isolated_process.PrivatePipeInput(
                        reader=reader,
                        writer=writer,
                        payload=credential,
                        max_payload_bytes=len(credential),
                    ),
                    cancel_requested=cancellation,
                    env=isolated_process.sanitized_worker_environment(),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    controller = threading.Thread(target=invoke, name="round13-delayed-lifetime-controller")
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    observed_control: bytes | None = None
    observed_credential: bytes | None = None
    forced_test_cleanup = False
    try:
        controller.start()
        controller.join(timeout=3.0)
        if controller.is_alive():
            forced_test_cleanup = True
            cancellation.set()
            controller.join(timeout=2.0)
        if controller.is_alive():
            for handle in handles:
                with suppress(ProcessLookupError):
                    os.killpg(handle.pid, signal.SIGTERM)
            controller.join(timeout=2.0)
        if controller.is_alive():
            for handle in handles:
                with suppress(ProcessLookupError):
                    os.killpg(handle.pid, signal.SIGKILL)
            controller.join(timeout=2.0)
        assert len(handles) == 1
        leader_not_running = _process_not_running(handles[0].pid)
        leader_reaped = _exact_child_was_reaped(handles[0].pid)
        group_after_return = _process_group_snapshot(handles[0].pid)
        observed_control = _optional_bytes(control_receipt)
        observed_credential = _optional_bytes(credential_receipt)
    finally:
        if controller.is_alive():
            forced_test_cleanup = True
            cancellation.set()
            for handle in handles:
                _force_test_cleanup(handle.pid)
            controller.join(timeout=2.0)
        if not reader.closed:
            reader.close()
        if not writer.closed:
            writer.close()
        for handle in handles:
            _force_test_cleanup(handle.pid)

    assert lifetime_open_observed
    assert lifetime_reader_descriptor is not None
    assert lifetime_reader_identity is not None
    assert lifetime_eagain_descriptors
    assert set(lifetime_eagain_descriptors) == {lifetime_reader_descriptor}
    assert _bounded_bytes(lifetime_data_written) == b"X"
    assert not controller.is_alive()
    assert errors == []
    assert results == [isolated_process.IsolatedProcessResult(None, "start")]
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert observed_control == b"A"
    assert observed_credential is None
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()
    assert not forced_test_cleanup


def test_control_failure_cleanup_never_uses_fallback_daemon_or_quarantine_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every post-spawn control path joins exact cleanup and proves an empty group."""

    real_initializer = subprocess.Popen.__init__
    real_thread = threading.Thread
    real_fallback = getattr(isolated_process, "_fallback_worker_session", None)
    real_retry_quarantined = getattr(isolated_process, "_retry_quarantined_workers", None)
    real_quarantine = getattr(isolated_process, "_quarantine_worker", None)
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    lifecycle_threads: list[threading.Thread] = []
    fallback_calls: list[int] = []
    quarantine_calls: list[int] = []
    retry_calls = 0
    injected = SystemExit("round13 exact-cleanup control")

    def capture_native(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_initializer(process, *args, **kwargs)
        handles.append(process)
        identities.append((process.pid, os.getpgid(process.pid), os.getsid(process.pid)))

    def capture_thread(*args: Any, **kwargs: Any) -> threading.Thread:
        thread = real_thread(*args, **kwargs)
        if thread.name.startswith("isolated-"):
            lifecycle_threads.append(thread)
        return thread

    def observe_fallback(process: Any) -> object:
        fallback_calls.append(int(process.pid))
        assert callable(real_fallback)
        return real_fallback(process)  # type: ignore[arg-type]

    def observe_retry_quarantined() -> bool:
        nonlocal retry_calls
        retry_calls += 1
        assert callable(real_retry_quarantined)
        return real_retry_quarantined()

    def observe_quarantine(worker: Any) -> None:
        quarantine_calls.append(int(worker.process_id))
        assert callable(real_quarantine)
        real_quarantine(worker)  # type: ignore[arg-type]

    def raise_before_start(_process: object) -> None:
        raise injected

    monkeypatch.setattr(isolated_process, "_initialize_reserved_process", capture_native)
    monkeypatch.setattr(isolated_process.threading, "Thread", capture_thread)
    if real_fallback is not None:
        monkeypatch.setattr(isolated_process, "_fallback_worker_session", observe_fallback)
    if real_retry_quarantined is not None:
        monkeypatch.setattr(
            isolated_process,
            "_retry_quarantined_workers",
            observe_retry_quarantined,
        )
    if real_quarantine is not None:
        monkeypatch.setattr(isolated_process, "_quarantine_worker", observe_quarantine)
    monkeypatch.setattr(isolated_process, "_after_worker_spawn", raise_before_start)

    result: isolated_process.IsolatedProcessResult | None = None
    raised: BaseException | None = None
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    try:
        try:
            result = isolated_process.run_isolated_process(
                command=[
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'ok')",
                ],
                request=b"{}",
                timeout_seconds=2.0,
                max_response_bytes=32,
                env=isolated_process.sanitized_worker_environment(),
            )
        except BaseException as exc:
            raised = exc
        assert len(handles) == 1
        leader_not_running = _process_not_running(handles[0].pid)
        leader_reaped = _exact_child_was_reaped(handles[0].pid)
        group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        for handle in handles:
            _force_test_cleanup(handle.pid)

    assert result is None
    assert raised is injected
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert fallback_calls == []
    assert retry_calls == 0
    assert quarantine_calls == []
    assert lifecycle_threads
    assert all(not thread.daemon for thread in lifecycle_threads)
    assert all(not thread.is_alive() for thread in lifecycle_threads)
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()


@pytest.mark.parametrize("fault_moment", ["pre-action", "post-action"])
@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit])
def test_raw_control_close_ambiguity_never_touches_recycled_descriptor_after_fault(
    monkeypatch: pytest.MonkeyPatch,
    fault_moment: str,
    control_type: type[BaseException],
) -> None:
    """Raw close uncertainty preserves control while retiring the exact numeric FD forever."""

    real_initializer = subprocess.Popen.__init__
    real_thread = threading.Thread
    real_close_control = isolated_process._StartupOwner._close_control_writer
    real_close = isolated_process.os.close
    real_fstat = isolated_process.os.fstat
    real_read = isolated_process.os.read
    real_write = isolated_process.os.write
    real_get_inheritable = isolated_process.os.get_inheritable
    real_set_inheritable = isolated_process.os.set_inheritable
    real_dup = isolated_process.os.dup
    real_dup2 = isolated_process.os.dup2
    replacement_source = os.open(os.devnull, os.O_RDONLY)
    replacement_status = real_fstat(replacement_source)
    replacement_identity = (
        replacement_status.st_dev,
        replacement_status.st_ino,
        replacement_status.st_mode & 0o170000,
        replacement_status.st_rdev,
    )
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []
    lifecycle_threads: list[threading.Thread] = []
    target_descriptor: int | None = None
    target_identity: tuple[int, int, int, int] | None = None
    recycled_identity: tuple[int, int, int, int] | None = None
    raw_close_attempts = 0
    fault_delivered = False
    later_operations: list[tuple[str, int]] = []
    injected = control_type(f"round13 {fault_moment} raw control close")
    poison_lock = getattr(isolated_process, "_PRIVATE_PIPE_STATE_LOCK", None)
    poison_present = hasattr(isolated_process, "_PRIVATE_PIPE_STATE_POISONED")
    poison_before = getattr(isolated_process, "_PRIVATE_PIPE_STATE_POISONED", False)

    def descriptor_identity(descriptor: int) -> tuple[int, int, int, int]:
        status = real_fstat(descriptor)
        return (
            status.st_dev,
            status.st_ino,
            status.st_mode & 0o170000,
            status.st_rdev,
        )

    def observe_later(operation: str, descriptor: int) -> None:
        if fault_delivered and descriptor == target_descriptor:
            later_operations.append((operation, descriptor))

    def capture_native(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_initializer(process, *args, **kwargs)
        handles.append(process)
        identities.append((process.pid, os.getpgid(process.pid), os.getsid(process.pid)))

    def capture_thread(*args: Any, **kwargs: Any) -> threading.Thread:
        thread = real_thread(*args, **kwargs)
        if thread.name.startswith("isolated-"):
            lifecycle_threads.append(thread)
        return thread

    def bind_exact_control_writer(
        self: isolated_process._StartupOwner,
        transport: object,
    ) -> None:
        nonlocal target_descriptor, target_identity
        writer = self.control_writer
        assert writer is not None and not writer.closed
        descriptor = writer.fileno()
        identity = descriptor_identity(descriptor)
        if target_descriptor is None:
            target_descriptor = descriptor
            target_identity = identity
        else:
            assert (descriptor, identity) == (target_descriptor, target_identity)
        real_close_control(self, transport)  # type: ignore[arg-type]

    def ambiguous_close(descriptor: int) -> None:
        nonlocal fault_delivered, raw_close_attempts, recycled_identity
        if descriptor != target_descriptor:
            real_close(descriptor)
            return
        if fault_delivered:
            later_operations.append(("close", descriptor))
            return
        raw_close_attempts += 1
        assert raw_close_attempts == 1
        if fault_moment == "post-action":
            real_close(descriptor)
            real_dup2(replacement_source, descriptor, inheritable=False)
            recycled_identity = descriptor_identity(descriptor)
        fault_delivered = True
        raise injected

    def observed_fstat(descriptor: int) -> os.stat_result:
        observe_later("fstat", descriptor)
        return real_fstat(descriptor)

    def observed_read(descriptor: int, count: int) -> bytes:
        observe_later("read", descriptor)
        return real_read(descriptor, count)

    def observed_write(descriptor: int, payload: object) -> int:
        observe_later("write", descriptor)
        return real_write(descriptor, payload)  # type: ignore[arg-type]

    def observed_get_inheritable(descriptor: int) -> bool:
        observe_later("get_inheritable", descriptor)
        return real_get_inheritable(descriptor)

    def observed_set_inheritable(descriptor: int, inheritable: bool) -> None:
        observe_later("set_inheritable", descriptor)
        real_set_inheritable(descriptor, inheritable)

    def observed_dup(descriptor: int) -> int:
        observe_later("dup", descriptor)
        return real_dup(descriptor)

    def observed_dup2(
        descriptor: int,
        target: int,
        inheritable: bool = True,
    ) -> int:
        observe_later("dup2-source", descriptor)
        observe_later("dup2-target", target)
        return real_dup2(descriptor, target, inheritable=inheritable)

    monkeypatch.setattr(isolated_process, "_initialize_reserved_process", capture_native)
    monkeypatch.setattr(isolated_process.threading, "Thread", capture_thread)
    monkeypatch.setattr(
        isolated_process._StartupOwner,
        "_close_control_writer",
        bind_exact_control_writer,
    )
    monkeypatch.setattr(isolated_process.os, "close", ambiguous_close)
    monkeypatch.setattr(isolated_process.os, "fstat", observed_fstat)
    monkeypatch.setattr(isolated_process.os, "read", observed_read)
    monkeypatch.setattr(isolated_process.os, "write", observed_write)
    monkeypatch.setattr(isolated_process.os, "get_inheritable", observed_get_inheritable)
    monkeypatch.setattr(isolated_process.os, "set_inheritable", observed_set_inheritable)
    monkeypatch.setattr(isolated_process.os, "dup", observed_dup)
    monkeypatch.setattr(isolated_process.os, "dup2", observed_dup2)

    result: isolated_process.IsolatedProcessResult | None = None
    raised: BaseException | None = None
    current_identity: tuple[int, int, int, int] | None = None
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    forced_test_cleanup = False
    try:
        try:
            result = isolated_process.run_isolated_process(
                command=[
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'ok')",
                ],
                request=b"{}",
                timeout_seconds=2.0,
                max_response_bytes=32,
                env=isolated_process.sanitized_worker_environment(),
            )
        except BaseException as exc:
            raised = exc
        assert target_descriptor is not None
        current_identity = descriptor_identity(target_descriptor)
        assert len(handles) == 1
        leader_not_running = _process_not_running(handles[0].pid)
        leader_reaped = _exact_child_was_reaped(handles[0].pid)
        group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        for handle in handles:
            if not _process_not_running(handle.pid) or _process_group_snapshot(handle.pid):
                forced_test_cleanup = True
                _force_test_cleanup(handle.pid)
        if target_descriptor is not None:
            with suppress(OSError):
                real_close(target_descriptor)
        with suppress(OSError):
            real_close(replacement_source)
        if poison_present:
            if poison_lock is None:
                isolated_process._PRIVATE_PIPE_STATE_POISONED = poison_before
            else:
                with poison_lock:
                    isolated_process._PRIVATE_PIPE_STATE_POISONED = poison_before
    assert result is None
    assert raised is injected
    assert fault_delivered
    assert raw_close_attempts == 1
    assert target_descriptor is not None
    assert target_identity is not None
    assert later_operations == []
    assert current_identity == (
        replacement_identity if fault_moment == "post-action" else target_identity
    )
    assert recycled_identity == (replacement_identity if fault_moment == "post-action" else None)
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert lifecycle_threads
    assert all(not thread.daemon for thread in lifecycle_threads)
    assert all(not thread.is_alive() for thread in lifecycle_threads)
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()
    assert not forced_test_cleanup


@pytest.mark.parametrize("profile", ["early-eof", "data"])
def test_invalid_lifetime_sentinel_never_crosses_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    """An already-ended or data-bearing lifetime sentinel is never START authority."""

    control_receipt = tmp_path / "control"
    lifetime_action = {
        "early-eof": "os.close(lifetime); lifetime = -1",
        "data": "os.write(lifetime, b'X')",
    }[profile]
    source = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "status, control, owner_fd, lifetime = map(int, sys.argv[1:5])",
            "os.write(owner_fd, f'OWNER {os.getpid()}\\n'.encode('ascii'))",
            "os.close(owner_fd)",
            lifetime_action,
            "os.write(status, f'READY {os.getpid()}\\n'.encode('ascii'))",
            "received = bytearray(os.read(control, 1))",
            "received.extend(os.read(control, 1))",
            f"Path({os.fspath(control_receipt)!r}).write_bytes(received)",
            "os.close(status)",
            "os.close(control)",
            "if lifetime >= 0: os.close(lifetime)",
        )
    )
    real_initializer = subprocess.Popen.__init__
    handles: list[subprocess.Popen[bytes]] = []
    identities: list[tuple[int, int, int]] = []

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
            source,
            str(status_descriptor),
            str(control_descriptor),
            str(owner_descriptor),
            str(lifetime_descriptor),
        ]

    def capture_native(
        process: subprocess.Popen[bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_initializer(process, *args, **kwargs)
        handles.append(process)
        identities.append((process.pid, os.getpgid(process.pid), os.getsid(process.pid)))

    _install_synthetic_supervisor(monkeypatch, synthetic_supervisor)
    monkeypatch.setattr(isolated_process, "_initialize_reserved_process", capture_native)

    result: isolated_process.IsolatedProcessResult | None = None
    raised: BaseException | None = None
    leader_not_running = False
    leader_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    try:
        try:
            result = isolated_process.run_isolated_process(
                command=[sys.executable, "-I", "-c", "pass"],
                request=b"{}",
                timeout_seconds=1.0,
                max_response_bytes=32,
                env=isolated_process.sanitized_worker_environment(),
            )
        except BaseException as exc:
            raised = exc
        assert len(handles) == 1
        leader_not_running = _process_not_running(handles[0].pid)
        leader_reaped = _exact_child_was_reaped(handles[0].pid)
        group_after_return = _process_group_snapshot(handles[0].pid)
    finally:
        for handle in handles:
            _force_test_cleanup(handle.pid)

    assert raised is None
    assert result == isolated_process.IsolatedProcessResult(None, "start")
    assert identities == [(handles[0].pid, handles[0].pid, handles[0].pid)]
    assert _bounded_bytes(control_receipt) == b"A"
    assert leader_not_running
    assert leader_reaped
    assert group_after_return == ()
