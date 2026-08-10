"""Public and kernel-level regressions for the tenth Task 9 review."""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
            "case_round10_credential_transport",
            f"exec_{'a' * 32}",
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


def _raw_frame_receipt_worker(receipt: Path) -> list[str]:
    code = "\n".join(
        (
            "import json, os, struct, sys",
            "from pathlib import Path",
            "payload = json.loads(sys.stdin.buffer.read())",
            "descriptor = payload['credential_fd']",
            "header = os.read(descriptor, 4)",
            "data = b''",
            "if len(header) == 4:",
            "    remaining = struct.unpack('!I', header)[0]",
            "    while len(data) < remaining:",
            "        chunk = os.read(descriptor, remaining - len(data))",
            "        if not chunk:",
            "            break",
            "        data += chunk",
            f"Path({os.fspath(receipt)!r}).write_bytes(header + data)",
            "os.close(descriptor)",
            "sys.stdout.buffer.write(b'{\"ok\":true}')",
        )
    )
    return [sys.executable, "-c", code]


def test_round10_lifecycle_cancellation_has_an_explicit_admission_protocol() -> None:
    """Credential delivery and cancellation must share one ordering owner."""

    cancellation = isolated_process.ProcessCancellation()

    assert not cancellation.is_set()
    cancellation.set()
    assert cancellation.is_set()


@pytest.mark.parametrize(
    ("leader_state", "watcher_observed_exit", "expected"),
    [("Ss", True, True), ("Z", False, True), ("Ss", False, False)],
)
def test_round10_independent_positive_exit_proofs_do_not_require_atomic_visibility(
    leader_state: str,
    watcher_observed_exit: bool,
    expected: bool,
) -> None:
    """A kqueue event and process-table transition can become visible in either order."""

    class ExitWatcher:
        def wait(self, _timeout: float) -> bool:
            return watcher_observed_exit

    worker = migration_process._WorkerSession(
        process=None,  # type: ignore[arg-type]
        process_id=41,
        process_group_id=41,
        session_id=41,
        exit_watcher=ExitWatcher(),  # type: ignore[arg-type]
    )
    snapshot = migration_process._WorkerSessionSnapshot(leader_state, ())

    assert migration_process._leader_exit_is_proven(worker, snapshot) is expected


def test_round10_cancellation_at_send_admission_transmits_zero_frame_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation before the first syscall wins the delivery ordering race."""

    canary = "round10-cancel-before-first-syscall-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / "cancel-admission-frame.bin"
    cancellation = isolated_process.ProcessCancellation()
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _raw_frame_receipt_worker(receipt),
    )
    real_send = isolated_process.send_private_frame
    observed_tokens: list[object] = []

    def cancel_at_admission(*args: Any, **kwargs: Any) -> object:
        observed_tokens.append(kwargs.get("cancel_requested"))
        cancellation.set()
        return real_send(*args, **kwargs)

    monkeypatch.setattr(isolated_process, "send_private_frame", cancel_at_admission)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
        cancel_requested=cancellation,
    )

    observed = receipt.read_bytes() if receipt.exists() else b""
    assert outcome.error_code == "LIFECYCLE_WORKER_CANCELLED"
    assert observed_tokens == [cancellation]
    assert observed == b""
    assert canary.encode() not in observed


def test_round10_every_parent_read_alias_is_closed_before_credential_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spawn inheritance alias cannot keep a parent-readable secret channel."""

    canary = "round10-parent-alias-retirement-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    receipt = tmp_path / "alias-retirement-frame.bin"
    captured_inputs: list[isolated_process.PrivatePipeInput] = []
    captured_aliases: list[isolated_process.PrivatePipeEndpoint] = []
    write_observations: list[tuple[bool, tuple[bool, ...]]] = []
    real_run = isolated_process.run_isolated_process
    real_duplicate = isolated_process._private_pipe_duplicate
    real_write = os.write

    def observed_run(**kwargs: Any) -> isolated_process.IsolatedProcessResult:
        captured_inputs.append(kwargs["private_input"])
        return real_run(**kwargs)

    def observed_duplicate(
        descriptor: int,
        *,
        readable: bool,
        writable: bool,
    ) -> isolated_process.PrivatePipeEndpoint:
        alias = real_duplicate(descriptor, readable=readable, writable=writable)
        captured_aliases.append(alias)
        return alias

    def observed_write(descriptor: int, data: Any) -> int:
        if captured_inputs and descriptor == captured_inputs[0].writer.fileno():
            write_observations.append(
                (
                    captured_inputs[0].reader.closed,
                    tuple(alias.closed for alias in captured_aliases),
                )
            )
        return real_write(descriptor, data)

    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: _raw_frame_receipt_worker(receipt),
    )
    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    monkeypatch.setattr(isolated_process, "_private_pipe_duplicate", observed_duplicate)
    monkeypatch.setattr(isolated_process.os, "write", observed_write)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert outcome.error_code is None
    assert len(captured_inputs) == len(captured_aliases) == 1
    assert write_observations == [(True, (True,)), (True, (True,))]
    assert receipt.read_bytes()[4:] == canary.encode()


@pytest.mark.parametrize(
    "failure_point",
    ["claim", "settings", "started_at", "json"],
)
def test_round10_metadata_faults_happen_before_credential_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """Every fallible metadata operation must precede the mutable secret copy."""

    canary = f"round10-{failure_point}-encode-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    tracked: list[bytearray] = []
    real_bytearray = bytearray

    def tracking_bytearray(value: object = b"", *args: Any, **kwargs: Any) -> bytearray:
        buffer = real_bytearray(value, *args, **kwargs)  # type: ignore[arg-type]
        tracked.append(buffer)
        return buffer

    def fail(*_args: object, **_kwargs: object) -> object:
        raise OSError(f"round10 injected {failure_point} encoding failure")

    monkeypatch.setattr(lifecycle_process, "bytearray", tracking_bytearray, raising=False)
    if failure_point == "claim":
        monkeypatch.setattr(lifecycle_process, "_claim_payload", fail)
    elif failure_point == "settings":
        monkeypatch.setattr(lifecycle_process, "_redacted_settings", fail)
    elif failure_point == "started_at":
        monkeypatch.setattr(lifecycle_process, "_canonical_start", fail)
    else:
        monkeypatch.setattr(lifecycle_process.json, "dumps", fail)

    with pytest.raises(OSError, match=f"round10 injected {failure_point}"):
        lifecycle_process._encode_request(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            credential_fd=7,
        )

    assert tracked == []


@pytest.mark.parametrize("endpoint_kind", ["reader", "writer"])
def test_round10_close_before_action_retains_exact_endpoint_ownership(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    """A pre-action close fault must not discard the still-open pipe identity."""

    reader, writer = isolated_process.private_pipe_channel()
    endpoint = reader if endpoint_kind == "reader" else writer
    other = writer if endpoint_kind == "reader" else reader
    descriptor = endpoint.fileno()
    expected = os.fstat(descriptor)
    real_close = os.close

    def fail_target_close(candidate: int) -> None:
        if candidate == descriptor:
            raise OSError(errno.EIO, "round10 close failed before kernel action")
        real_close(candidate)

    monkeypatch.setattr(isolated_process.os, "close", fail_target_close)
    try:
        with pytest.raises(OSError, match="before kernel action"):
            endpoint.close()

        assert endpoint.fileno() == descriptor
        observed = os.fstat(descriptor)
        assert (observed.st_dev, observed.st_ino, observed.st_mode) == (
            expected.st_dev,
            expected.st_ino,
            expected.st_mode,
        )
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        endpoint.close()
        other.close()


@pytest.mark.parametrize("endpoint_kind", ["reader", "writer"])
def test_round10_uncertain_endpoint_blocks_new_channels_until_reconciled(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    """A retained endpoint is durably quarantined and fail-closes later admission."""

    reader, writer = isolated_process.private_pipe_channel()
    endpoint = reader if endpoint_kind == "reader" else writer
    other = writer if endpoint_kind == "reader" else reader
    descriptor = endpoint.fileno()
    real_close = os.close

    def fail_target_close(candidate: int) -> None:
        if candidate == descriptor:
            raise OSError(errno.EIO, "round10 persistent pre-action close failure")
        real_close(candidate)

    monkeypatch.setattr(isolated_process.os, "close", fail_target_close)
    with pytest.raises(OSError, match="persistent pre-action"):
        endpoint.close()
    other.close()

    unexpectedly_admitted: (
        tuple[
            isolated_process.PrivatePipeEndpoint,
            isolated_process.PrivatePipeEndpoint,
        ]
        | None
    ) = None
    try:
        with pytest.raises(isolated_process.IsolatedProcessCleanupError):
            unexpectedly_admitted = isolated_process.private_pipe_channel()
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        if unexpectedly_admitted is not None:
            unexpectedly_admitted[0].close()
            unexpectedly_admitted[1].close()

    recovered_reader, recovered_writer = isolated_process.private_pipe_channel()
    recovered_reader.close()
    recovered_writer.close()
    assert endpoint.closed
    with pytest.raises(OSError) as excinfo:
        os.fstat(descriptor)
    assert excinfo.value.errno == errno.EBADF


@pytest.mark.parametrize("endpoint_kind", ["reader", "writer"])
def test_round10_close_after_action_never_claims_or_mutates_a_reused_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint_kind: str,
) -> None:
    """Post-action failure proves the pipe gone without owning a reused raw number."""

    reader, writer = isolated_process.private_pipe_channel()
    endpoint = reader if endpoint_kind == "reader" else writer
    other = writer if endpoint_kind == "reader" else reader
    descriptor = endpoint.fileno()
    original = os.fstat(descriptor)
    unrelated_path = tmp_path / f"{endpoint_kind}-unrelated"
    unrelated_path.write_bytes(b"round10 unrelated descriptor")
    real_close = os.close
    replacement_identity: list[tuple[int, int, int]] = []

    def close_reuse_then_fail(candidate: int) -> None:
        if candidate != descriptor or replacement_identity:
            real_close(candidate)
            return
        real_close(candidate)
        replacement = os.open(unrelated_path, os.O_RDONLY)
        if replacement != descriptor:
            os.dup2(replacement, descriptor, inheritable=False)
            real_close(replacement)
        status = os.fstat(descriptor)
        replacement_identity.append((status.st_dev, status.st_ino, status.st_mode))
        raise OSError(errno.EIO, "round10 close failed after kernel action")

    monkeypatch.setattr(isolated_process.os, "close", close_reuse_then_fail)
    try:
        with pytest.raises(OSError, match="after kernel action"):
            endpoint.close()

        assert endpoint.closed
        current = os.fstat(descriptor)
        assert (current.st_dev, current.st_ino, current.st_mode) == replacement_identity[0]
        assert (current.st_dev, current.st_ino, current.st_mode) != (
            original.st_dev,
            original.st_ino,
            original.st_mode,
        )
        endpoint.close()
        current = os.fstat(descriptor)
        assert (current.st_dev, current.st_ino, current.st_mode) == replacement_identity[0]
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        with suppress(OSError):
            real_close(descriptor)
        other.close()


@pytest.mark.parametrize(
    "control_type",
    [KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_round10_spawn_then_control_cannot_escape_before_session_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """A post-spawn interruption must retain and contain leader plus descendants."""

    canary = f"round10-spawn-{control_type.__name__}-canary"
    settings, started_at, claim = _lifecycle_inputs(tmp_path, canary)
    marker = tmp_path / f"{control_type.__name__}.pids"
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
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", worker_code],
    )
    spawned: list[isolated_process._SpawnedProcess] = []
    injected = False

    def spawn_then_interrupt(process: isolated_process._SpawnedProcess) -> None:
        nonlocal injected
        if not injected:
            injected = True
            spawned.append(process)
            deadline = time.monotonic() + 3.0
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), "real worker never published its descendant marker"
            raise control_type("round10 control after real spawn")

    monkeypatch.setattr(isolated_process, "_after_worker_spawn", spawn_then_interrupt)
    leader_id = 0
    descendant_id = 0
    try:
        with pytest.raises(control_type, match="round10 control after real spawn"):
            lifecycle_process.run_lifecycle_process(
                mode="process",
                settings=settings,
                claim=claim,
                started_at=started_at,
                timeout_seconds=1.0,
            )

        assert injected and len(spawned) == 1
        leader_id, descendant_id = map(int, marker.read_text(encoding="ascii").split())
        assert leader_id == spawned[0].pid
        assert spawned[0].poll() is not None
        assert _pid_is_absent(descendant_id)
    finally:
        if marker.exists():
            published_leader, published_descendant = map(
                int,
                marker.read_text(encoding="ascii").split(),
            )
            leader_id = leader_id or published_leader
            descendant_id = descendant_id or published_descendant
        if not leader_id and spawned:
            leader_id = spawned[0].pid
        if leader_id and not _pid_is_absent(leader_id):
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(leader_id, signal.SIGKILL)
        for process in spawned:
            with suppress(BaseException):
                process.wait(timeout=2.0)
        cleanup_deadline = time.monotonic() + 2.0
        while time.monotonic() < cleanup_deadline and any(
            process_id and not _pid_is_absent(process_id)
            for process_id in (leader_id, descendant_id)
        ):
            time.sleep(0.01)
        assert not leader_id or _pid_is_absent(leader_id)
        assert not descendant_id or _pid_is_absent(descendant_id)


def test_round10_control_immediately_after_kernel_spawn_still_owns_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stopped child is already published to the startup owner."""

    settings, started_at, claim = _lifecycle_inputs(
        tmp_path,
        "round10-pre-publication-stop-canary",
    )
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    spawned: list[isolated_process._SpawnedProcess] = []

    def stop_then_interrupt(process: isolated_process._SpawnedProcess) -> None:
        if not spawned:
            spawned.append(process)
            os.kill(process.pid, signal.SIGSTOP)
            raise KeyboardInterrupt("round10 control after kernel spawn")

    monkeypatch.setattr(isolated_process, "_after_worker_spawn", stop_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="after kernel spawn"):
            lifecycle_process.run_lifecycle_process(
                mode="process",
                settings=settings,
                claim=claim,
                started_at=started_at,
                timeout_seconds=0.1,
            )
        assert len(spawned) == 1
        assert spawned[0].poll() is not None
    finally:
        for process in spawned:
            if process.poll() is None:
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
                with suppress(BaseException):
                    process.wait(timeout=2.0)
            assert _pid_is_absent(process.pid)
