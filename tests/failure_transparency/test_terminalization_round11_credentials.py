"""Regressions for the eleventh Task 9 process-ownership review."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from invoice_agents import isolated_process, lifecycle_process
from invoice_agents.config import Settings
from invoice_agents.db import migration_process
from invoice_agents.db.store import ExecutionClaim


def _lifecycle_inputs(
    tmp_path: Path,
    credential: str,
) -> tuple[Settings, datetime, ExecutionClaim]:
    started_at = datetime.now(UTC)
    return (
        Settings(
            xai_api_key=SecretStr(credential),
            inventory_db=tmp_path / "inventory.db",
            workflow_db=tmp_path / "workflow.db",
            source_archive_dir=tmp_path / "sources",
        ),
        started_at,
        ExecutionClaim(
            "case_round11_credential_transport",
            f"exec_{'b' * 32}",
            1,
            started_at + timedelta(minutes=1),
        ),
    )


def _pid_is_absent(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _bounded_marker_pids(marker: Path) -> tuple[int, int]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            encoded = marker.read_text(encoding="ascii")
            fields = encoded.split()
            if len(fields) == 2 and all(
                field.isascii() and field.isdigit() and int(field) > 0 and str(int(field)) == field
                for field in fields
            ):
                return int(fields[0]), int(fields[1])
        except (OSError, UnicodeError):
            pass
        time.sleep(0.01)
    raise AssertionError("real worker did not publish its complete bounded PID receipt")


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round11_spawn_wrapper_control_cannot_escape_unowned_child_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """A wrapper that raises after a real spawn still leaves a published, reaped domain."""

    marker = tmp_path / f"round11-{control_type.__name__}-spawn.pids"
    descendant_code = "import time; time.sleep(30)"
    worker_code = "\n".join(
        (
            "import os, subprocess, sys, time",
            "from pathlib import Path",
            f"child = subprocess.Popen([sys.executable, '-c', {descendant_code!r}])",
            f"Path({os.fspath(marker)!r}).write_text(f'{{os.getpid()}} {{child.pid}}', encoding='ascii')",
            "time.sleep(30)",
        )
    )
    real_launch = subprocess.Popen
    launched = False
    published_ids: list[tuple[int, int]] = []
    withheld_native_handles: list[subprocess.Popen[bytes]] = []
    barrier_errors: list[Exception] = []

    def spawn_then_control(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        nonlocal launched
        process = real_launch(*args, **kwargs)
        command = args[0] if args else kwargs["args"]
        if (
            not launched
            and isinstance(command, list)
            and "invoice_agents.spawn_supervisor" in command
        ):
            launched = True
            withheld_native_handles.append(process)
            try:
                published_ids.append(_bounded_marker_pids(marker))
            except Exception as exc:
                barrier_errors.append(exc)
            raise control_type("round11 control after real supervisor spawn")
        return process

    monkeypatch.setattr(
        isolated_process.subprocess,
        "Popen",
        spawn_then_control,
    )

    with pytest.raises(control_type, match="round11 control after real supervisor spawn"):
        isolated_process.run_isolated_process(
            command=[sys.executable, "-c", worker_code],
            request=b"{}",
            timeout_seconds=0.3,
            max_response_bytes=128,
            env=isolated_process.sanitized_worker_environment(),
        )

    try:
        assert launched
        assert barrier_errors == []
        assert len(published_ids) == 1
        leader_id, descendant_id = published_ids[0]
        assert _pid_is_absent(leader_id)
        assert _pid_is_absent(descendant_id)
        assert len(withheld_native_handles) == 1
        withheld_handle = withheld_native_handles[0]
        assert withheld_handle.stdin is None
        assert withheld_handle.stdout is None
        assert withheld_handle.stderr is None
        assert withheld_handle.returncode is None
    finally:
        synchronized = [handle.poll() for handle in withheld_native_handles]
    assert synchronized == [0]


def test_round11_expired_reader_drains_ready_publication_before_spawn_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ready raw PID remains ownership evidence after the request deadline."""

    marker = tmp_path / "round11-expired-publication.pids"
    descendant_code = "import time; time.sleep(30)"
    worker_code = "\n".join(
        (
            "import os, subprocess, sys, time",
            "from pathlib import Path",
            f"child = subprocess.Popen([sys.executable, '-c', {descendant_code!r}])",
            f"Path({os.fspath(marker)!r}).write_text(f'{{os.getpid()}} {{child.pid}}', encoding='ascii')",
            "time.sleep(30)",
        )
    )
    real_launch = subprocess.Popen
    launched_handles: list[subprocess.Popen[bytes]] = []
    published_ids: list[tuple[int, int]] = []

    def spawn_stall_then_control(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = real_launch(*args, **kwargs)
        command = args[0] if args else kwargs["args"]
        if isinstance(command, list) and "invoice_agents.spawn_supervisor" in command:
            launched_handles.append(process)
            published_ids.append(_bounded_marker_pids(marker))
            time.sleep(0.08)
            raise SystemExit("round11 control after expired raw PID publication")
        return process

    monkeypatch.setattr(isolated_process.subprocess, "Popen", spawn_stall_then_control)

    try:
        with pytest.raises(
            SystemExit,
            match="round11 control after expired raw PID publication",
        ):
            isolated_process.run_isolated_process(
                command=[sys.executable, "-c", worker_code],
                request=b"{}",
                timeout_seconds=0.05,
                max_response_bytes=128,
                env=isolated_process.sanitized_worker_environment(),
            )

        assert len(published_ids) == 1
        leader_id, descendant_id = published_ids[0]
        assert _pid_is_absent(leader_id)
        assert _pid_is_absent(descendant_id)
    finally:
        for handle in launched_handles:
            with suppress(ProcessLookupError):
                os.killpg(handle.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                handle.wait(timeout=2.0)


def test_round11_parent_mask_is_unchanged_and_child_sigint_mask_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller may block SIGINT; the isolated child must still start unblocked."""

    real_pthread_sigmask = signal.pthread_sigmask
    original_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
    real_pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    parent_calls: list[tuple[int, frozenset[signal.Signals]]] = []

    def observe_parent_mask_change(
        operation: int,
        mask: set[signal.Signals],
    ) -> set[signal.Signals]:
        parent_calls.append((operation, frozenset(mask)))
        if operation == signal.SIG_SETMASK:
            raise OSError(errno.EIO, "round11 injected mask restoration failure")
        return real_pthread_sigmask(operation, mask)

    monkeypatch.setattr(signal, "pthread_sigmask", observe_parent_mask_change)
    try:
        code = "\n".join(
            (
                "import signal, sys",
                "mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())",
                "sys.stdout.buffer.write(b'blocked' if signal.SIGINT in mask else b'unblocked')",
            )
        )
        result = isolated_process.run_isolated_process(
            command=[sys.executable, "-c", code],
            request=b"{}",
            timeout_seconds=1.0,
            max_response_bytes=32,
            env=isolated_process.sanitized_worker_environment(),
        )
        observed_parent_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
    finally:
        real_pthread_sigmask(signal.SIG_SETMASK, original_mask)

    assert result == isolated_process.IsolatedProcessResult(b"unblocked", None)
    assert observed_parent_mask == original_mask | {signal.SIGINT}
    assert parent_calls == []


def _descriptor_identity(descriptor: int) -> tuple[int, int, int, int]:
    status = os.fstat(descriptor)
    return status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode), status.st_rdev


def test_round11_worker_and_descendant_inherit_no_ambient_file_socket_or_pipe(
    tmp_path: Path,
) -> None:
    """Native spawn isolation closes every inheritable descriptor outside its allowlist."""

    with ExitStack() as resources:
        ambient_file = resources.enter_context((tmp_path / "ambient.bin").open("w+b", buffering=0))
        ambient_file.write(b"round11 ambient file")
        ambient_sockets = socket.socketpair()
        resources.callback(ambient_sockets[0].close)
        resources.callback(ambient_sockets[1].close)
        ambient_pipe = os.pipe()
        resources.callback(os.close, ambient_pipe[0])
        resources.callback(os.close, ambient_pipe[1])
        descriptors = (
            ambient_file.fileno(),
            ambient_sockets[0].fileno(),
            ambient_sockets[1].fileno(),
            ambient_pipe[0],
            ambient_pipe[1],
        )
        identities = {descriptor: _descriptor_identity(descriptor) for descriptor in descriptors}
        receipt = tmp_path / "round11-descendant-fds.json"
        for descriptor in descriptors:
            os.set_inheritable(descriptor, True)
        checker = "\n".join(
            (
                "import json, os, sys",
                "from pathlib import Path",
                "expected = {int(fd): tuple(value) for fd, value in json.loads(sys.argv[1]).items()}",
                "leaked = []",
                "for descriptor, identity in expected.items():",
                "    try:",
                "        status = os.fstat(descriptor)",
                "    except OSError:",
                "        continue",
                "    observed = (status.st_dev, status.st_ino, status.st_mode & 0o170000, status.st_rdev)",
                "    if observed == identity:",
                "        leaked.append(descriptor)",
                "Path(sys.argv[2]).write_text(json.dumps(leaked), encoding='ascii')",
            )
        )
        worker = "\n".join(
            (
                "import json, os, subprocess, sys",
                "expected = {int(fd): tuple(value) for fd, value in json.loads(sys.argv[1]).items()}",
                "leaked = []",
                "for descriptor, identity in expected.items():",
                "    try:",
                "        status = os.fstat(descriptor)",
                "    except OSError:",
                "        continue",
                "    observed = (status.st_dev, status.st_ino, status.st_mode & 0o170000, status.st_rdev)",
                "    if observed == identity:",
                "        leaked.append(descriptor)",
                f"child = subprocess.run([sys.executable, '-c', {checker!r}, sys.argv[1], sys.argv[2]], check=True)",
                "sys.stdout.buffer.write(json.dumps(leaked).encode('ascii'))",
            )
        )
        encoded_identities = json.dumps(identities, sort_keys=True)
        result = isolated_process.run_isolated_process(
            command=[sys.executable, "-c", worker, encoded_identities, os.fspath(receipt)],
            request=b"{}",
            timeout_seconds=2.0,
            max_response_bytes=128,
            env=isolated_process.sanitized_worker_environment(),
        )

    assert result == isolated_process.IsolatedProcessResult(b"[]", None)
    assert json.loads(receipt.read_text(encoding="ascii")) == []


def test_round11_spawn_is_fail_closed_when_supervisor_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No direct-spawn fallback may bypass the close-all descriptor boundary."""

    def unavailable(*_args: Any, **_kwargs: Any) -> subprocess.Popen[bytes]:
        raise NotImplementedError("round11 isolated supervisor unavailable")

    monkeypatch.setattr(isolated_process, "_launch_supervisor", unavailable, raising=False)

    result = isolated_process.run_isolated_process(
        command=[sys.executable, "-c", "raise SystemExit(0)"],
        request=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=32,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert result == isolated_process.IsolatedProcessResult(None, "start")


def test_round11_known_spawn_deadline_precedes_unverified_start_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline crossed after PID capture remains a timeout, not a startup crash."""

    real_capture = isolated_process._capture_worker_session
    real_monotonic = time.monotonic
    controller_clock = SimpleNamespace(expired=False)
    controller_clock.monotonic = lambda: (
        real_monotonic() + (2.0 if controller_clock.expired else 0.0)
    )
    controller_clock.sleep = time.sleep

    def expire_after_capture(process: isolated_process._SpawnedProcess) -> object:
        worker = real_capture(process)
        worker.watcher_initialized = False
        worker.identity_verified = False
        controller_clock.expired = True
        return worker

    monkeypatch.setattr(isolated_process, "time", controller_clock)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", expire_after_capture)

    result = isolated_process.run_isolated_process(
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        request=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=32,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert result == isolated_process.IsolatedProcessResult(None, "timeout")


def test_round11_preset_cancel_precedes_publication_deadline_and_keeps_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-set cancellation survives a startup deadline without weakening ownership."""

    real_launch = isolated_process._launch_supervisor
    real_capture = isolated_process._capture_worker_session
    real_monotonic = time.monotonic
    controller_clock = SimpleNamespace(expired=False)
    controller_clock.monotonic = lambda: (
        real_monotonic() + (2.0 if controller_clock.expired else 0.0)
    )
    controller_clock.sleep = time.sleep
    captured_workers: list[Any] = []

    def launch_then_expire(*args: Any, **kwargs: Any) -> None:
        real_launch(*args, **kwargs)
        controller_clock.expired = True

    def observed_capture(process: isolated_process._SpawnedProcess) -> object:
        worker = real_capture(process)
        captured_workers.append(worker)
        return worker

    cancellation = isolated_process.ProcessCancellation()
    cancellation.set()
    monkeypatch.setattr(isolated_process, "time", controller_clock)
    monkeypatch.setattr(isolated_process, "_launch_supervisor", launch_then_expire)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", observed_capture)

    result = isolated_process.run_isolated_process(
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        request=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=32,
        cancel_requested=cancellation,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert result == isolated_process.IsolatedProcessResult(None, "cancelled")
    assert len(captured_workers) == 1
    assert captured_workers[0].cleaned


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round11_startup_endpoint_control_precedes_all_credential_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """A captured close control at startup cannot be deferred past secret delivery."""

    canary = f"round11-startup-close-{control_type.__name__}-credential"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / f"round11-startup-close-{control_type.__name__}.bin"
    worker = "\n".join(
        (
            "import json, sys",
            "from pathlib import Path",
            "from invoice_agents.lifecycle_worker import _read_private_credential",
            "payload = json.loads(sys.stdin.buffer.read())",
            "credential = _read_private_credential(payload['credential_fd'])",
            f"Path({os.fspath(receipt)!r}).write_bytes(credential)",
            "credential[:] = bytes(len(credential))",
            "sys.stdout.buffer.write(b'{\"ok\":true}')",
        )
    )
    real_close = isolated_process.PrivatePipeEndpoint.close
    injected = False

    def close_then_control(endpoint: isolated_process.PrivatePipeEndpoint) -> None:
        nonlocal injected
        real_close(endpoint)
        if not injected and endpoint.writable:
            injected = True
            raise control_type("round11 control during startup endpoint retirement")

    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", worker],
    )
    monkeypatch.setattr(isolated_process.PrivatePipeEndpoint, "close", close_then_control)

    with pytest.raises(control_type, match="startup endpoint retirement"):
        lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )

    assert injected
    assert not receipt.exists()


@pytest.mark.parametrize("endpoint_kind", ["reader", "writer"])
def test_round11_close_fault_never_retries_same_pipe_replacement_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    """A same-inode dup2 replacement is never acted on through stale endpoint ownership."""

    reader, writer = isolated_process.private_pipe_channel()
    endpoint = reader if endpoint_kind == "reader" else writer
    other_endpoint = writer if endpoint_kind == "reader" else reader
    descriptor = endpoint.fileno()
    replacement_source = os.dup(descriptor)
    replacement_identity: tuple[int, int, int, int] | None = None
    real_close = os.close
    injected = False

    def close_replace_then_fail(candidate: int) -> None:
        nonlocal injected, replacement_identity
        if candidate != descriptor or injected:
            real_close(candidate)
            return
        injected = True
        real_close(candidate)
        os.dup2(replacement_source, descriptor, inheritable=False)
        replacement_identity = _descriptor_identity(descriptor)
        raise OSError(errno.EIO, "round11 close failed after same-pipe reuse")

    monkeypatch.setattr(isolated_process.os, "close", close_replace_then_fail)
    try:
        with pytest.raises(OSError, match="same-pipe reuse"):
            endpoint.close()
        admitted_reader, admitted_writer = isolated_process.private_pipe_channel()
        admitted_reader.close()
        admitted_writer.close()
        assert endpoint.closed
        assert _descriptor_identity(descriptor) == replacement_identity
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        with suppress(OSError):
            real_close(descriptor)
        with suppress(OSError):
            real_close(replacement_source)
        other_endpoint.close()


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("exact", True),
        ("wrong-ident", "error"),
        ("wrong-filter", "error"),
        ("missing-note-exit", "error"),
        ("extra-note", "error"),
        ("kernel-error", "error"),
        ("unbound-bool", "error"),
    ],
)
def test_round11_darwin_watcher_accepts_only_exact_bound_note_exit_event(
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    expected: bool | str,
) -> None:
    """A nonempty kqueue result is not evidence unless it binds the exact process exit."""

    process_id = 74011
    proc_filter = -5
    note_exit = 0x80000000
    ev_error = 0x4000
    registration = SimpleNamespace(
        ident=process_id,
        filter=proc_filter,
        flags=0x0011,
        fflags=note_exit,
        data=0,
    )
    returned = {
        "exact": SimpleNamespace(
            ident=process_id,
            filter=proc_filter,
            flags=0,
            fflags=note_exit,
            data=0,
        ),
        "wrong-ident": SimpleNamespace(
            ident=process_id + 1,
            filter=proc_filter,
            flags=0,
            fflags=note_exit,
            data=0,
        ),
        "wrong-filter": SimpleNamespace(
            ident=process_id,
            filter=proc_filter + 1,
            flags=0,
            fflags=note_exit,
            data=0,
        ),
        "missing-note-exit": SimpleNamespace(
            ident=process_id,
            filter=proc_filter,
            flags=0,
            fflags=0,
            data=0,
        ),
        "extra-note": SimpleNamespace(
            ident=process_id,
            filter=proc_filter,
            flags=0,
            fflags=note_exit | 0x02,
            data=0,
        ),
        "kernel-error": SimpleNamespace(
            ident=process_id,
            filter=proc_filter,
            flags=ev_error,
            fflags=note_exit,
            data=errno.ESRCH,
        ),
        "unbound-bool": True,
    }[event]

    class KernelQueue:
        def __init__(self) -> None:
            self.registered = False

        def control(
            self,
            changelist: list[object] | None,
            _max_events: int,
            _timeout: float,
        ) -> object:
            if changelist is not None:
                assert changelist == [registration]
                self.registered = True
                return []
            assert self.registered
            return [returned]

        def close(self) -> None:
            return None

    kernel_queue = KernelQueue()
    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    monkeypatch.setattr(migration_process.select, "kqueue", lambda: kernel_queue, raising=False)
    monkeypatch.setattr(
        migration_process.select,
        "kevent",
        lambda *_args, **_kwargs: registration,
        raising=False,
    )
    monkeypatch.setattr(migration_process.select, "KQ_FILTER_PROC", proc_filter, raising=False)
    monkeypatch.setattr(migration_process.select, "KQ_EV_ADD", 0x0001, raising=False)
    monkeypatch.setattr(migration_process.select, "KQ_EV_ONESHOT", 0x0010, raising=False)
    monkeypatch.setattr(migration_process.select, "KQ_NOTE_EXIT", note_exit, raising=False)
    monkeypatch.setattr(migration_process.select, "KQ_EV_ERROR", ev_error, raising=False)
    watcher = migration_process._WorkerExitWatcher(process_id)

    if expected == "error":
        with pytest.raises(OSError, match="invalid worker exit event"):
            watcher.wait(0)
    else:
        assert watcher.wait(0) is expected


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
@pytest.mark.parametrize("control_stage", ["provider-value", "materialized", "handoff"])
def test_round11_lifecycle_control_zeroes_caller_owned_credential_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
    control_stage: str,
) -> None:
    """The mutable credential copy is caller-owned before materialization through handoff."""

    canary = f"round11-{control_stage}-{control_type.__name__}-credential"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    tracked: list[bytearray] = []
    if control_stage in {"provider-value", "materialized"}:
        real_materialize = getattr(lifecycle_process, "_materialize_provider_credential", None)

        def observed_materialize(*args: Any, **kwargs: Any) -> None:
            assert real_materialize is not None
            credential = args[1]
            assert type(credential) is bytearray and not credential
            tracked.append(credential)
            real_materialize(*args, **kwargs)
            if control_stage == "materialized":
                raise control_type("round11 control after credential materialization")

        monkeypatch.setattr(
            lifecycle_process,
            "_materialize_provider_credential",
            observed_materialize,
            raising=False,
        )
        if control_stage == "provider-value":
            real_provider_key = Settings.provider_key

            def interrupt_after_provider_value(selected: Settings) -> str:
                value = real_provider_key(selected)
                assert value == canary
                raise control_type("round11 control after provider value")

            monkeypatch.setattr(Settings, "provider_key", interrupt_after_provider_value)
    else:

        def interrupt_handoff(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
            private_input = kwargs["private_input"]
            tracked.append(private_input.payload)
            assert bytes(private_input.payload) == canary.encode()
            raise control_type("round11 control during credential handoff")

        monkeypatch.setattr(lifecycle_process, "run_isolated_process", interrupt_handoff)

    with pytest.raises(control_type, match="round11 control"):
        lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )

    assert tracked
    assert all(not any(buffer) for buffer in tracked)
    assert all(canary.encode() not in buffer for buffer in tracked)
