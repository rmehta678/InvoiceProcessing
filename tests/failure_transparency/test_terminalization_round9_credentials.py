"""Public-boundary regressions for the ninth Task 9 credential review."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import hashlib
import json
import logging
import os
import struct
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from invoice_agents import isolated_process, lifecycle_process, lifecycle_worker, terminal_process
from invoice_agents.config import Settings
from invoice_agents.db import migration_process
from invoice_agents.db.store import ExecutionClaim
from invoice_agents.errors import InvoiceAgentsError


def _lifecycle_inputs(
    tmp_path: Path,
    credential: str,
) -> tuple[Settings, datetime, ExecutionClaim]:
    settings = Settings(
        xai_api_key=SecretStr(credential),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_round9_credential_transport",
        f"exec_{'9' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )
    return settings, started_at, claim


def _success_worker_command() -> list[str]:
    child_code = "\n".join(
        (
            "import json, sys",
            "from invoice_agents.lifecycle_worker import _read_private_credential",
            "payload = json.loads(sys.stdin.buffer.read())",
            "credential = _read_private_credential(payload['credential_fd'])",
            "credential[:] = bytes(len(credential))",
            "sys.stdout.buffer.write(b'{\"ok\":true}')",
        )
    )
    return [sys.executable, "-c", child_code]


def _terminal_worker_command(mode: str) -> list[str]:
    tail = {
        "success": "sys.stdout.buffer.write(b'{\"ok\":true}')",
        "crash": "os._exit(23)",
        "timeout": "time.sleep(30)",
        "cancel": "time.sleep(30)",
    }[mode]
    child_code = "\n".join(
        (
            "import json, os, sys, time",
            "from invoice_agents.lifecycle_worker import _read_private_credential",
            "payload = json.loads(sys.stdin.buffer.read())",
            "credential = _read_private_credential(payload['credential_fd'])",
            "credential[:] = bytes(len(credential))",
            tail,
        )
    )
    return [sys.executable, "-c", child_code]


@pytest.fixture(autouse=True)
def _fresh_process_failure_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test models a fresh process without adding a production poison reset."""

    monkeypatch.setattr(isolated_process, "_PRIVATE_PIPE_STATE_POISONED", False)
    monkeypatch.setattr(isolated_process, "_ACTIVE_PRIVATE_PIPE_ENDPOINTS", 0)
    monkeypatch.setattr(
        migration_process,
        "_WORKER_RESOURCE_CLEANUP_POISONED",
        threading.Event(),
    )
    yield
    assert isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS == 0


def _digest_worker_command(receipt: Path, mode: str = "success") -> list[str]:
    tail = {
        "success": "sys.stdout.buffer.write(b'{\"ok\":true}')",
        "crash": "os._exit(29)",
        "timeout": "time.sleep(30)",
    }[mode]
    child_code = "\n".join(
        (
            "import hashlib, json, os, sys, time",
            "from pathlib import Path",
            "from invoice_agents.lifecycle_worker import _read_private_credential",
            "payload = json.loads(sys.stdin.buffer.read())",
            "credential = _read_private_credential(payload['credential_fd'])",
            "digest = hashlib.sha256(credential).hexdigest()",
            "credential[:] = bytes(len(credential))",
            f"Path({os.fspath(receipt)!r}).write_text(digest, encoding='ascii')",
            tail,
        )
    )
    return [sys.executable, "-c", child_code]


def _worker_is_contained(worker: object) -> bool:
    return bool(worker.cleaned) and worker.process.poll() is not None  # type: ignore[attr-defined]


@pytest.mark.parametrize("fault_stage", ["initializer", "classifier", "capture"])
def test_round9_terminal_post_native_spawn_control_reaps_reserved_process_before_reraise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
) -> None:
    """A terminal worker remains owned when process initialization is interrupted."""

    class PostNativeSpawnControl(BaseException):
        pass

    injected = PostNativeSpawnControl("terminal post-native-spawn control")
    spawned: list[subprocess.Popen[bytes]] = []
    captured_sessions: list[Any] = []
    real_initialize = subprocess.Popen.__init__

    def initialize_then_interrupt(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)
        raise injected

    if fault_stage == "initializer":
        monkeypatch.setattr(
            terminal_process,
            "_initialize_reserved_terminal_process",
            initialize_then_interrupt,
            raising=False,
        )
    elif fault_stage == "classifier":
        real_classifier = terminal_process._reserved_process_has_native_child
        interrupted = False

        def classify_then_interrupt(process: subprocess.Popen[bytes]) -> bool:
            nonlocal interrupted
            classified = real_classifier(process)
            if classified and not interrupted:
                interrupted = True
                spawned.append(process)
                raise injected
            return classified

        monkeypatch.setattr(
            terminal_process,
            "_reserved_process_has_native_child",
            classify_then_interrupt,
        )
    else:
        real_capture = terminal_process._capture_worker_session

        def capture_then_interrupt(
            process: subprocess.Popen[bytes],
            *args: object,
            **kwargs: object,
        ) -> object:
            captured = real_capture(process, *args, **kwargs)  # type: ignore[arg-type]
            captured_sessions.append(captured)
            spawned.append(process)
            raise injected

        monkeypatch.setattr(
            terminal_process,
            "_capture_worker_session",
            capture_then_interrupt,
        )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    settings, started_at, claim = _lifecycle_inputs(
        tmp_path,
        "round9-terminal-post-native-spawn-canary",
    )

    try:
        with pytest.raises(PostNativeSpawnControl) as excinfo:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.02,
                started_at=started_at,
            )

        assert excinfo.value is injected
        assert len(spawned) == 1
        assert spawned[0].poll() is not None
        if fault_stage == "capture":
            assert len(captured_sessions) == 1
            captured = captured_sessions[0]
            assert captured.cleaned
            watcher = captured.exit_watcher
            assert watcher is None or (watcher._kqueue is None and watcher._pidfd is None)
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_round9_terminal_poisoned_worker_resource_rejects_before_initializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poison = threading.Event()
    poison.set()
    initialized = False

    def forbid_initializer(*_args: object, **_kwargs: object) -> None:
        nonlocal initialized
        initialized = True
        raise AssertionError("poisoned terminal admission reached initializer")

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        forbid_initializer,
    )
    monkeypatch.setattr(
        "invoice_agents.db.migration_process._WORKER_RESOURCE_CLEANUP_POISONED",
        poison,
    )
    settings, started_at, claim = _lifecycle_inputs(
        tmp_path,
        "round9-terminal-poisoned-resource-canary",
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        terminal_process.run_terminal_process(
            mode="cancel_unstarted",
            settings=settings,
            claim=claim,
            timeout_seconds=0.02,
            started_at=started_at,
        )

    assert excinfo.value.stop_reason == "TERMINAL_WORKER_CLEANUP_FAILED"
    assert not initialized


def _capture_spawned_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> list[isolated_process._SpawnedProcess]:
    spawned: list[isolated_process._SpawnedProcess] = []
    real_launch = isolated_process._launch_supervisor

    def observed_launch(*args: Any, **kwargs: Any) -> None:
        owner = args[0]
        real_launch(*args, **kwargs)
        process = owner.process
        assert process is not None
        spawned.append(process)

    monkeypatch.setattr(isolated_process, "_launch_supervisor", observed_launch)
    return spawned


def test_round9_child_receives_a_read_only_pipe_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bidirectional parent-readable endpoint must never cross ``pass_fds``."""

    canary = "round9-read-only-pipe-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    observed_access_modes: list[int] = []
    real_run = isolated_process.run_isolated_process

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        passed = kwargs["pass_fds"]
        observed_access_modes.extend(
            fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE for descriptor in passed
        )
        return real_run(**kwargs)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(30)"],
    )

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=0.05,
    )

    assert outcome.error_code == "LIFECYCLE_WORKER_TIMED_OUT"
    assert observed_access_modes == [os.O_RDONLY]
    assert canary not in repr(outcome)


def test_round9_unproven_parent_read_close_transmits_zero_credential_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-end close fault must happen before the first credential byte exists."""

    canary = "round9-no-write-before-read-close-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / "credential-receipt.bin"
    child_code = "\n".join(
        (
            "import json, os, sys",
            "from pathlib import Path",
            "payload = json.loads(sys.stdin.buffer.read())",
            "descriptor = payload['credential_fd']",
            "data = os.read(descriptor, 16389)",
            f"Path({os.fspath(receipt)!r}).write_bytes(data)",
            "os.close(descriptor)",
            "sys.stdout.buffer.write(b'{\"ok\":true}')",
        )
    )
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", child_code],
    )
    spawned = _capture_spawned_worker(monkeypatch)
    credential_reader: list[int] = []
    injected = False
    real_close = os.close
    real_run = isolated_process.run_isolated_process

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        credential_reader.extend(kwargs["pass_fds"])
        return real_run(**kwargs)

    def fail_first_parent_read_close(descriptor: int) -> None:
        nonlocal injected
        if credential_reader and descriptor == credential_reader[0] and not injected:
            injected = True
            raise OSError(errno.EIO, "round9 injected parent read close failure")
        real_close(descriptor)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(isolated_process.os, "close", fail_first_parent_read_close)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    observed = receipt.read_bytes() if receipt.exists() else b""
    for descriptor in credential_reader:
        with suppress(OSError):
            real_close(descriptor)
    assert injected
    assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
    assert canary.encode() not in observed
    assert all(process.poll() is not None for process in spawned)
    assert canary not in repr(outcome)
    with pytest.raises(isolated_process.IsolatedProcessCleanupError):
        isolated_process.private_pipe_channel()


def test_round9_repeated_read_release_faults_self_heal_and_do_not_poison_follow_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed pre-write releases close in cleanup and never block a later start."""

    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        _success_worker_command,
    )
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    real_run = isolated_process.run_isolated_process

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    injected = 0

    def fail_before_release(_reader: isolated_process.PrivatePipeEndpoint) -> None:
        nonlocal injected
        injected += 1
        raise OSError(errno.EIO, "round9 injected pre-write reader release failure")

    with monkeypatch.context() as fault:
        fault.setattr(
            isolated_process,
            "_release_parent_reader",
            fail_before_release,
            raising=False,
        )
        for attempt in range(3):
            settings, started_at, claim = _lifecycle_inputs(
                tmp_path,
                f"round9-repeat-reader-close-{attempt}",
            )
            outcome = lifecycle_process.run_lifecycle_process(
                mode="process",
                settings=settings,
                claim=claim,
                started_at=started_at,
                timeout_seconds=1.0,
            )
            assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"

    settings, started_at, claim = _lifecycle_inputs(tmp_path, "round9-recovered-reader-close")
    recovered = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert injected == 3
    assert recovered == lifecycle_process.LifecycleProcessOutcome(True, None)
    assert len(captured_inputs) == 4
    assert all(item.reader.closed and item.writer.closed for item in captured_inputs)
    assert all(not any(item.payload) for item in captured_inputs)


def test_round9_parent_reader_is_closed_before_first_write_and_writer_is_write_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every credential write happens after the parent reader is kernel-absent."""

    canary = "round9-close-before-first-write-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", _success_worker_command)
    readers: list[int] = []
    reader_identities: list[tuple[int, int, int, int]] = []
    write_observations: list[tuple[int, bool]] = []
    retained_payloads: list[bytearray] = []
    requests: list[bytes] = []
    environments: list[dict[str, str]] = []
    real_write = os.write
    real_run = isolated_process.run_isolated_process

    def observed_write(descriptor: int, data: Any) -> int:
        access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        if access_mode == os.O_WRONLY and readers:
            try:
                current = os.fstat(readers[0])
            except OSError as exc:
                reader_absent = exc.errno == errno.EBADF
            else:
                reader_absent = (
                    current.st_dev,
                    current.st_ino,
                    current.st_mode,
                    current.st_rdev,
                ) != reader_identities[0]
            write_observations.append((access_mode, reader_absent))
        return real_write(descriptor, data)

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        readers.extend(kwargs["pass_fds"])
        reader_identities.extend(
            (status.st_dev, status.st_ino, status.st_mode, status.st_rdev)
            for status in (os.fstat(descriptor) for descriptor in kwargs["pass_fds"])
        )
        requests.append(kwargs["request"])
        environments.append(kwargs["env"])
        retained_payloads.append(kwargs["private_input"].payload)
        return real_run(**kwargs)

    monkeypatch.setattr(isolated_process.os, "write", observed_write)
    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert outcome == lifecycle_process.LifecycleProcessOutcome(True, None)
    assert len(write_observations) >= 2
    assert all(mode == os.O_WRONLY and reader_absent for mode, reader_absent in write_observations)
    assert retained_payloads and all(not any(payload) for payload in retained_payloads)
    assert requests and all(canary.encode() not in request for request in requests)
    assert environments and all(canary not in repr(environment) for environment in environments)
    assert all(canary not in argument for argument in _success_worker_command())
    assert canary not in repr(outcome)


@pytest.mark.parametrize(
    ("worker_mode", "expected_code"),
    [
        ("success", None),
        ("crash", "LIFECYCLE_WORKER_CRASHED"),
        ("timeout", "LIFECYCLE_WORKER_TIMED_OUT"),
        ("cancel", "LIFECYCLE_WORKER_CANCELLED"),
    ],
)
def test_round9_worker_terminal_paths_reap_child_and_retire_directional_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_mode: str,
    expected_code: str | None,
) -> None:
    """Success, crash, timeout, and cancellation leave no child or owned endpoint."""

    canary = f"round9-{worker_mode}-terminal-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _terminal_worker_command(worker_mode),
    )
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    spawned = _capture_spawned_worker(monkeypatch)
    real_run = isolated_process.run_isolated_process

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    cancel_requested = isolated_process.ProcessCancellation()
    if worker_mode == "cancel":
        cancel_requested.set()

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=0.08 if worker_mode in {"timeout", "cancel"} else 1.0,
        cancel_requested=cancel_requested,
    )

    assert outcome.error_code == expected_code
    assert len(captured_inputs) == 1
    assert captured_inputs[0].reader.closed
    assert captured_inputs[0].writer.closed
    assert not any(captured_inputs[0].payload)
    if worker_mode == "cancel":
        assert spawned == []
    else:
        assert len(spawned) == 1
        assert spawned[0].poll() is not None
    assert canary not in repr(outcome)


def test_round9_start_failure_retires_both_endpoints_without_transmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed exec has no child, no credential bytes, and no retained endpoint."""

    canary = "round9-start-failure-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [os.fspath(tmp_path / "missing-round9-worker")],
    )
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    real_run = isolated_process.run_isolated_process

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert outcome.error_code == "LIFECYCLE_WORKER_CRASHED"
    assert len(captured_inputs) == 1
    assert captured_inputs[0].reader.closed and captured_inputs[0].writer.closed
    assert not any(captured_inputs[0].payload)
    assert canary not in repr(outcome)


def test_round9_worker_closes_reader_before_descendant_and_inherits_no_ambient_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real frame reader closes before a close_fds=False descendant starts."""

    canary = "round9-descendant-frame-canary"
    aliases = {
        "XAI_KEY": "round9-descendant-xai-alias",
        "PASSWORD": "round9-descendant-password-alias",
        "GITHUB_PAT": "round9-descendant-github-alias",
        "AWS_ACCESS_KEY_ID": "round9-descendant-aws-alias",
    }
    for name, value in aliases.items():
        monkeypatch.setenv(name, value)
    receipt = tmp_path / "round9-descendant.json"
    descendant_code = "\n".join(
        (
            "import json, os, sys",
            "from pathlib import Path",
            "descriptor = int(sys.argv[1])",
            "try:",
            "    os.fstat(descriptor)",
            "    fd_open = True",
            "except OSError:",
            "    fd_open = False",
            f"aliases = {tuple(aliases)!r}",
            "result = {'fd_open': fd_open, 'aliases': sorted(name for name in aliases if name in os.environ), 'digest': sys.argv[2]}",
            f"Path({os.fspath(receipt)!r}).write_text(json.dumps(result, sort_keys=True), encoding='utf-8')",
        )
    )
    child_code = "\n".join(
        (
            "import hashlib, json, subprocess, sys",
            "from invoice_agents.lifecycle_worker import _read_private_credential",
            "payload = json.loads(sys.stdin.buffer.read())",
            "descriptor = payload['credential_fd']",
            "credential = _read_private_credential(descriptor)",
            "digest = hashlib.sha256(credential).hexdigest()",
            "credential[:] = bytes(len(credential))",
            f"subprocess.run([sys.executable, '-c', {descendant_code!r}, str(descriptor), digest], check=True, close_fds=False)",
            "sys.stdout.buffer.write(b'{\"ok\":true}')",
        )
    )
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", child_code],
    )
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=2.0,
    )

    observed = json.loads(receipt.read_text(encoding="utf-8"))
    assert outcome == lifecycle_process.LifecycleProcessOutcome(True, None)
    assert observed == {
        "aliases": [],
        "digest": hashlib.sha256(canary.encode()).hexdigest(),
        "fd_open": False,
    }
    assert canary not in receipt.read_text(encoding="utf-8")
    assert all(value not in receipt.read_text(encoding="utf-8") for value in aliases.values())


def test_round9_provider_output_cannot_promote_credential_to_public_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Credential stdout/stderr is discarded and never reaches errors or artifacts."""

    canary = "round9-provider-output-canary"
    child_code = "\n".join(
        (
            "import json, sys",
            "from invoice_agents.lifecycle_worker import _read_private_credential",
            "payload = json.loads(sys.stdin.buffer.read())",
            "credential = _read_private_credential(payload['credential_fd'])",
            "sys.stderr.buffer.write(credential)",
            "sys.stderr.buffer.flush()",
            "sys.stdout.buffer.write(credential)",
            "sys.stdout.buffer.flush()",
            "credential[:] = bytes(len(credential))",
        )
    )
    command = [sys.executable, "-c", child_code]
    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", lambda: command)
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)

    with caplog.at_level(logging.DEBUG):
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )

    assert outcome.error_code == "LIFECYCLE_WORKER_PROTOCOL_INVALID"
    assert canary not in repr(outcome)
    assert canary not in caplog.text
    assert all(canary not in argument for argument in command)
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert canary.encode() not in artifact.read_bytes()


@pytest.mark.parametrize(
    "frame",
    [
        struct.pack("!I", 0),
        struct.pack("!I", lifecycle_process.LIFECYCLE_MAX_CREDENTIAL_BYTES + 1),
        b"\x00\x01",
        struct.pack("!I", 8) + b"short",
    ],
    ids=["empty", "oversized", "partial-header", "partial-payload"],
)
def test_round9_worker_rejects_malformed_real_pipe_frames(frame: bytes) -> None:
    """Malformed lengths and partial writes fail without waiting for EOF semantics."""

    reader, writer = os.pipe()
    os.write(writer, frame)
    os.close(writer)
    with pytest.raises((OSError, ValueError)):
        lifecycle_worker._read_private_credential(reader)
    with pytest.raises(OSError):
        os.fstat(reader)


def test_round9_worker_rejects_a_write_only_descriptor() -> None:
    """The child reader validates the inherited endpoint's kernel access mode."""

    reader, writer = os.pipe()
    os.close(reader)
    with pytest.raises(ValueError, match="transport"):
        lifecycle_worker._read_private_credential(writer)
    with pytest.raises(OSError):
        os.fstat(writer)


@pytest.mark.parametrize("trailing", [b"", b"ignored-second-frame"])
def test_round9_worker_reads_one_exact_frame_without_waiting_for_eof(trailing: bytes) -> None:
    """A complete frame returns while the writer stays open and ignores later bytes."""

    canary = bytearray(b"round9-exact-frame-canary")
    reader, writer = os.pipe()
    os.write(writer, struct.pack("!I", len(canary)))
    os.write(writer, canary)
    if trailing:
        os.write(writer, trailing)

    credential = lifecycle_worker._read_private_credential(reader)
    try:
        assert credential == canary
        with pytest.raises(OSError):
            os.fstat(reader)
    finally:
        credential[:] = bytes(len(credential))
        canary[:] = bytes(len(canary))
        os.close(writer)


@pytest.mark.parametrize(
    "fault_profile",
    ["before-header", "partial-header", "after-header", "partial-payload"],
)
def test_round9_write_faults_destroy_partial_kernel_buffer_and_reap_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_profile: str,
) -> None:
    """Every ordinary write fault is cleanup failure with no complete child frame."""

    canary = f"round9-{fault_profile}-write-fault-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / f"{fault_profile}-receipt"
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _digest_worker_command(receipt),
    )
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    spawned = _capture_spawned_worker(monkeypatch)
    write_calls = 0
    real_run = isolated_process.run_isolated_process
    real_write = os.write

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    def failing_write(descriptor: int, data: Any) -> int:
        nonlocal write_calls
        if descriptor != captured_inputs[0].writer.fileno():
            return real_write(descriptor, data)
        write_calls += 1
        if fault_profile == "before-header" and write_calls == 1:
            raise OSError(errno.EIO, "round9 write before header")
        if fault_profile == "partial-header":
            if write_calls == 1:
                return real_write(descriptor, data[:2])
            raise OSError(errno.EIO, "round9 write during header")
        if fault_profile == "after-header" and write_calls == 2:
            raise OSError(errno.EIO, "round9 write after header")
        if fault_profile == "partial-payload":
            if write_calls == 2:
                return real_write(descriptor, data[:3])
            if write_calls == 3:
                raise OSError(errno.EIO, "round9 write during payload")
        return real_write(descriptor, data)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(isolated_process.os, "write", failing_write)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
    assert write_calls >= 1
    assert not receipt.exists()
    assert len(captured_inputs) == len(spawned) == 1
    assert captured_inputs[0].reader.closed and captured_inputs[0].writer.closed
    assert not any(captured_inputs[0].payload)
    assert spawned[0].poll() is not None
    assert canary not in repr(outcome)


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
@pytest.mark.parametrize(
    "control_stage",
    ["before-read-close", "after-read-close", "before-write", "during-write", "after-write"],
)
def test_round9_transport_control_waits_for_child_containment_and_zeroes_buffers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
    control_stage: str,
) -> None:
    """Control flow escapes only after the real child is reaped and buffers are zeroed."""

    canary = f"round9-{control_stage}-{control_type.__name__}-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / f"{control_stage}-{control_type.__name__}-digest"
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _digest_worker_command(receipt),
    )
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    spawned = _capture_spawned_worker(monkeypatch)
    real_run = isolated_process.run_isolated_process
    real_reader_release = isolated_process._release_parent_reader
    real_write = os.write
    write_calls = 0

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    def control_reader_release(reader: isolated_process.PrivatePipeEndpoint) -> None:
        if control_stage == "after-read-close":
            real_reader_release(reader)
        raise control_type("round9 reader release control")

    def control_before_write(*_args: Any, **_kwargs: Any) -> None:
        raise control_type("round9 pre-write control")

    def control_during_write(descriptor: int, data: Any) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            return real_write(descriptor, data[:3])
        if write_calls == 3:
            raise control_type("round9 partial-write control")
        return real_write(descriptor, data)

    def control_after_write(_writer: isolated_process.PrivatePipeEndpoint) -> None:
        raise control_type("round9 post-write control")

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    if control_stage in {"before-read-close", "after-read-close"}:
        monkeypatch.setattr(isolated_process, "_release_parent_reader", control_reader_release)
    elif control_stage == "before-write":
        monkeypatch.setattr(isolated_process, "send_private_frame", control_before_write)
    elif control_stage == "during-write":
        monkeypatch.setattr(isolated_process.os, "write", control_during_write)
    else:
        monkeypatch.setattr(isolated_process, "_release_parent_writer", control_after_write)

    with pytest.raises(control_type):
        lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )

    assert len(captured_inputs) == len(spawned) == 1
    assert captured_inputs[0].reader.closed and captured_inputs[0].writer.closed
    assert not any(captured_inputs[0].payload)
    assert spawned[0].poll() is not None
    if receipt.exists():
        assert receipt.read_text(encoding="ascii") == hashlib.sha256(canary.encode()).hexdigest()
    if control_stage != "after-write":
        assert not receipt.exists()


@pytest.mark.parametrize("release_stage", ["before-close", "after-close"])
@pytest.mark.parametrize("worker_mode", ["success", "crash", "timeout"])
def test_round9_writer_release_failure_precedes_worker_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_stage: str,
    worker_mode: str,
) -> None:
    """A post-frame writer release fault remains a visible cleanup failure."""

    canary = f"round9-{worker_mode}-{release_stage}-writer-release-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / f"{worker_mode}-{release_stage}-digest"
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _digest_worker_command(receipt, worker_mode),
    )
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    spawned = _capture_spawned_worker(monkeypatch)
    real_run = isolated_process.run_isolated_process
    real_release = isolated_process._release_parent_writer

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    def fail_writer_release(writer: isolated_process.PrivatePipeEndpoint) -> None:
        if release_stage == "after-close":
            real_release(writer)
        raise OSError(errno.EIO, "round9 injected writer release failure")

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(isolated_process, "_release_parent_writer", fail_writer_release)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
    assert len(captured_inputs) == len(spawned) == 1
    assert captured_inputs[0].reader.closed and captured_inputs[0].writer.closed
    assert not any(captured_inputs[0].payload)
    assert spawned[0].poll() is not None
    if receipt.exists():
        assert receipt.read_text(encoding="ascii") == hashlib.sha256(canary.encode()).hexdigest()
    assert canary not in repr(outcome)


def test_round9_repeated_writer_release_faults_self_heal_and_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retained write-only endpoints close in cleanup and do not poison starts."""

    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", _success_worker_command)
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    real_run = isolated_process.run_isolated_process

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    def fail_before_writer_close(_writer: isolated_process.PrivatePipeEndpoint) -> None:
        raise OSError(errno.EIO, "round9 repeated writer close fault")

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    with monkeypatch.context() as fault:
        fault.setattr(isolated_process, "_release_parent_writer", fail_before_writer_close)
        for attempt in range(3):
            settings, started_at, claim = _lifecycle_inputs(
                tmp_path,
                f"round9-repeat-writer-close-{attempt}",
            )
            outcome = lifecycle_process.run_lifecycle_process(
                mode="process",
                settings=settings,
                claim=claim,
                started_at=started_at,
                timeout_seconds=1.0,
            )
            assert outcome.error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"

    settings, started_at, claim = _lifecycle_inputs(tmp_path, "round9-writer-close-recovered")
    recovered = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert recovered == lifecycle_process.LifecycleProcessOutcome(True, None)
    assert len(captured_inputs) == 4
    assert all(item.reader.closed and item.writer.closed for item in captured_inputs)
    assert all(not any(item.payload) for item in captured_inputs)


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round9_capture_control_transmits_nothing_and_reaps_fallback_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """Capture interruption uses the immediate fallback record for containment."""

    canary = f"round9-capture-{control_type.__name__}-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / f"capture-{control_type.__name__}-digest"
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _digest_worker_command(receipt),
    )
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    spawned = _capture_spawned_worker(monkeypatch)
    real_run = isolated_process.run_isolated_process

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    def interrupt_capture(
        _process: isolated_process._SpawnedProcess,
        _retained_worker: object,
    ) -> object:
        raise control_type("round9 worker capture control")

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", interrupt_capture)

    with pytest.raises(control_type):
        lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )

    assert not receipt.exists()
    assert len(captured_inputs) == len(spawned) == 1
    assert captured_inputs[0].reader.closed and captured_inputs[0].writer.closed
    assert not any(captured_inputs[0].payload)
    assert spawned[0].poll() is not None


def test_round9_capture_failure_is_start_failure_with_zero_credential_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary capture failure is never allowed to start provider work."""

    canary = "round9-capture-failure-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / "capture-failure-digest"
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _digest_worker_command(receipt),
    )
    spawned = _capture_spawned_worker(monkeypatch)

    def fail_capture(
        _process: isolated_process._SpawnedProcess,
        _retained_worker: object,
    ) -> object:
        raise OSError(errno.EIO, "round9 worker capture failure")

    monkeypatch.setattr(isolated_process, "_capture_worker_session", fail_capture)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert outcome.error_code == "LIFECYCLE_WORKER_CRASHED"
    assert not receipt.exists()
    assert len(spawned) == 1 and spawned[0].poll() is not None
    assert canary not in repr(outcome)


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round9_stop_control_is_reraised_only_after_real_child_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """The independent cleanup path reaps before a stop control escapes."""

    canary = f"round9-stop-{control_type.__name__}-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _terminal_worker_command("timeout"),
    )
    captured_workers: list[object] = []
    real_cleanup_phase = isolated_process._run_worker_cleanup_phase

    def interrupt_after_cleanup(worker: object, outcome: object) -> None:
        captured_workers.append(worker)
        real_cleanup_phase(worker, outcome)  # type: ignore[arg-type]
        raise control_type("round9 stop control")

    monkeypatch.setattr(isolated_process, "_run_worker_cleanup_phase", interrupt_after_cleanup)

    with pytest.raises(control_type):
        lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=0.08,
        )

    assert len(captured_workers) == 1
    assert captured_workers[0].cleaned  # type: ignore[attr-defined]
    assert captured_workers[0].process.poll() is not None  # type: ignore[attr-defined]
    assert _worker_is_contained(captured_workers[0])


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round9_cleanup_control_blocks_owner_until_exact_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """A cleanup control cannot release the caller before exact worker reap."""

    canary = f"round9-retained-owner-{control_type.__name__}-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _terminal_worker_command("timeout"),
    )
    captured_workers: list[object] = []
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    returned = threading.Event()
    errors: list[BaseException] = []
    real_cleanup = isolated_process._cleanup_cooperative_worker_session

    def controlled_cleanup(worker: object) -> None:
        captured_workers.append(worker)
        cleanup_entered.set()
        assert release_cleanup.wait(2.0)
        real_cleanup(worker)  # type: ignore[arg-type]
        raise control_type("round9 cleanup control")

    def invoke() -> None:
        try:
            lifecycle_process.run_lifecycle_process(
                mode="process",
                settings=settings,
                claim=claim,
                started_at=started_at,
                timeout_seconds=0.08,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    monkeypatch.setattr(isolated_process, "_cleanup_cooperative_worker_session", controlled_cleanup)
    controller = threading.Thread(target=invoke, name="round9-cleanup-owner")
    controller.start()

    try:
        assert cleanup_entered.wait(2.0)
        assert not returned.is_set()
        assert len(captured_workers) == 1
        assert not captured_workers[0].cleaned  # type: ignore[attr-defined]
        release_cleanup.set()
        controller.join(timeout=2.0)
    finally:
        release_cleanup.set()
        controller.join(timeout=2.0)

    assert not controller.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], control_type)
    assert _worker_is_contained(captured_workers[0])


def test_round9_cleanup_failure_returns_only_after_exact_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary cleanup faults fail closed only after the worker is reaped."""

    settings, started_at, claim = _lifecycle_inputs(tmp_path, "round9-stop-cleanup-canary")
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _terminal_worker_command("timeout"),
    )
    captured_workers: list[object] = []
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    returned = threading.Event()
    outcomes: list[lifecycle_process.LifecycleProcessOutcome] = []
    errors: list[BaseException] = []
    real_cleanup = isolated_process._cleanup_cooperative_worker_session

    def controlled_cleanup(worker: object) -> None:
        captured_workers.append(worker)
        cleanup_entered.set()
        assert release_cleanup.wait(2.0)
        real_cleanup(worker)  # type: ignore[arg-type]
        raise OSError(errno.EIO, "round9 cleanup evidence failure")

    def invoke() -> None:
        try:
            outcomes.append(
                lifecycle_process.run_lifecycle_process(
                    mode="process",
                    settings=settings,
                    claim=claim,
                    started_at=started_at,
                    timeout_seconds=0.08,
                )
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    monkeypatch.setattr(isolated_process, "_cleanup_cooperative_worker_session", controlled_cleanup)
    controller = threading.Thread(target=invoke, name="round9-cleanup-owner")
    controller.start()

    try:
        assert cleanup_entered.wait(2.0)
        assert not returned.is_set()
        assert len(captured_workers) == 1
        assert not captured_workers[0].cleaned  # type: ignore[attr-defined]
        release_cleanup.set()
        controller.join(timeout=2.0)
    finally:
        release_cleanup.set()
        controller.join(timeout=2.0)

    assert not controller.is_alive()
    assert errors == []
    assert len(outcomes) == 1
    assert outcomes[0].error_code == "LIFECYCLE_WORKER_CLEANUP_FAILED"
    assert _worker_is_contained(captured_workers[0])
