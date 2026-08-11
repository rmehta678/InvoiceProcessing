"""Regressions for the eleventh Task 9 process-ownership review."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from contextlib import ExitStack
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

_AMBIENT_DESCRIPTOR_PROCESS_PHASE_SECONDS = 2.0
_SINGLE_WORKER_SUCCESS_SECONDS = 5.0 * _AMBIENT_DESCRIPTOR_PROCESS_PHASE_SECONDS
_AMBIENT_DESCRIPTOR_SUCCESS_SECONDS = 7.0 * _AMBIENT_DESCRIPTOR_PROCESS_PHASE_SECONDS


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


def _strict_process_table_rows() -> tuple[tuple[int, int, str], ...]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,stat="],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        env={**os.environ, "LANG": "C", "LC_ALL": "C"},
        text=False,
        timeout=1.0,
    )
    if completed.stderr or not completed.stdout.endswith(b"\n"):
        raise ValueError("process-table scan was not complete")
    rows: list[tuple[int, int, str]] = []
    row_pattern = re.compile(rb" *([1-9][0-9]*) +([1-9][0-9]*) +([^\x00-\x20\x7f-\xff]+)( *)")
    for raw_line in completed.stdout.splitlines():
        matched = row_pattern.fullmatch(raw_line)
        if matched is None:
            raise ValueError("process-table row was not canonical")
        raw_process_id, raw_group_id, raw_state, raw_padding = matched.groups()
        state = raw_state.decode("ascii")
        if sys.platform == "darwin":
            valid_state = (
                re.fullmatch(
                    r"[?IRSTUZ](?:>|W)?(?:<|N)?(?:A|S)?X?E?V?L?s?\+?",
                    state,
                )
                is not None
            )
            expected_padding = max(0, 4 - len(state))
        elif sys.platform.startswith("linux"):
            valid_state = (
                re.fullmatch(
                    r"[DIKPRStTWXxZ](?:<|N)?L?s?l?\+?",
                    state,
                )
                is not None
            )
            expected_padding = 0
        else:
            raise ValueError("process-table platform was unsupported")
        if not valid_state or len(raw_padding) != expected_padding:
            raise ValueError("process-table state was not canonical")
        rows.append((int(raw_process_id), int(raw_group_id), state))
    if len({process_id for process_id, _group_id, _state in rows}) != len(rows):
        raise ValueError("process-table repeated a process identifier")
    return tuple(sorted(rows))


def _identities_and_group_are_absent(
    process_ids: tuple[int, ...],
    process_group_id: int,
) -> bool:
    rows = _strict_process_table_rows()
    expected_ids = set(process_ids)
    return all(
        process_id not in expected_ids and observed_group_id != process_group_id
        for process_id, observed_group_id, _state in rows
    )


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


def _wait_for_path(path: Path) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.005)
    raise TimeoutError(f"bounded process receipt was not published: {path}")


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
    real_initialize = isolated_process._initialize_reserved_process
    real_status_read = isolated_process._read_supervisor_status_line
    published_ids: list[tuple[int, int]] = []
    withheld_native_handles: list[subprocess.Popen[bytes]] = []
    barrier_errors: list[Exception] = []

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)
        withheld_native_handles.append(process)

    def read_started_then_control(*args: Any, **kwargs: Any) -> bytes:
        status = real_status_read(*args, **kwargs)
        if status.startswith(b"STARTED "):
            try:
                published_ids.append(_bounded_marker_pids(marker))
            except Exception as exc:
                barrier_errors.append(exc)
            raise control_type("round11 control after real supervisor spawn")
        return status

    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(
        isolated_process,
        "_read_supervisor_status_line",
        read_started_then_control,
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
        assert len(withheld_native_handles) == 1
        assert barrier_errors == []
        assert len(published_ids) == 1
        leader_id, descendant_id = published_ids[0]
        assert _identities_and_group_are_absent(
            (leader_id, descendant_id),
            leader_id,
        )
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

    del tmp_path
    worker_code = "import time; time.sleep(30)"
    real_initialize = isolated_process._initialize_reserved_process
    launched_handles: list[subprocess.Popen[bytes]] = []
    published_ids: list[int] = []

    def initialize_stall_then_control(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)
        launched_handles.append(process)
        published_ids.append(process.pid)
        time.sleep(0.08)
        raise SystemExit("round11 control after expired raw PID publication")

    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_stall_then_control,
    )

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
        supervisor_id = published_ids[0]
        assert _identities_and_group_are_absent((supervisor_id,), supervisor_id)
    finally:
        synchronized = [handle.poll() for handle in launched_handles]
    assert synchronized == [0]


def test_round17_unexpected_supervisor_death_cannot_report_contained_worker(
    tmp_path: Path,
) -> None:
    """A crashed trusted owner cannot hide its still-live same-session worker group."""

    marker = tmp_path / "round17-crashed-supervisor-worker"
    crash_receipt = tmp_path / "round17-crashed-supervisor"
    snapshot_receipt = tmp_path / "round17-crashed-supervisor-snapshots"
    outcome_receipt = tmp_path / "round17-crash-outcome"
    descendant_code = "import time; time.sleep(30)"
    worker_code = "\n".join(
        (
            "import os, subprocess, sys, time",
            "from pathlib import Path",
            f"child = subprocess.Popen([sys.executable, '-c', {descendant_code!r}])",
            f"destination = Path({os.fspath(marker)!r})",
            "temporary = destination.with_name(f'{destination.name}.{os.getpid()}.tmp')",
            "temporary.write_text(f'{os.getpid()} {child.pid} {os.getpgid(0)} {os.getsid(0)}', encoding='ascii')",
            "os.replace(temporary, destination)",
            "time.sleep(30)",
        )
    )
    wrapper_source = "\n".join(
        (
            "import json, os, signal, sys, time",
            "from pathlib import Path",
            "source_root, marker, crash_path, snapshot_path, outcome_path, worker_code = sys.argv[1:]",
            "sys.path.insert(0, source_root)",
            "from invoice_agents import isolated_process",
            "from invoice_agents.db import migration_process",
            "real_initialize = isolated_process._initialize_reserved_process",
            "real_status_read = isolated_process._read_supervisor_status_line",
            "real_snapshot = migration_process._worker_session_snapshot",
            "supervisor_id = 0",
            "snapshot_count = 0",
            "def publish(path, payload):",
            "    destination = Path(path)",
            "    temporary = destination.with_name(f'{destination.name}.{os.getpid()}.tmp')",
            "    temporary.write_text(payload, encoding='ascii')",
            "    os.replace(temporary, destination)",
            "def initialize(process, *args, **kwargs):",
            "    global supervisor_id",
            "    real_initialize(process, *args, **kwargs)",
            "    supervisor_id = process.pid",
            "def read_status(*args, **kwargs):",
            "    status = real_status_read(*args, **kwargs)",
            "    if status.startswith(b'STARTED '):",
            "        worker_id = int(status.removeprefix(b'STARTED ').removesuffix(b'\\n'))",
            "        deadline = time.monotonic() + 2.0",
            "        while not Path(marker).exists() and time.monotonic() < deadline: time.sleep(0.005)",
            "        if not Path(marker).exists(): raise TimeoutError('worker tree was not published')",
            "        publish(crash_path, f'{supervisor_id} {worker_id}')",
            "        os.kill(supervisor_id, signal.SIGKILL)",
            "    return status",
            "def snapshot(worker):",
            "    global snapshot_count",
            "    observed = real_snapshot(worker)",
            "    snapshot_count += 1",
            "    publish(snapshot_path, json.dumps({",
            "        'count': snapshot_count,",
            "        'leader_state': observed.leader_state,",
            "        'members': [[member.process_id, member.process_group_id] for member in observed.members],",
            "    }, sort_keys=True))",
            "    return observed",
            "isolated_process._initialize_reserved_process = initialize",
            "isolated_process._read_supervisor_status_line = read_status",
            "migration_process._worker_session_snapshot = snapshot",
            "try:",
            "    result = isolated_process.run_isolated_process(",
            "        command=[sys.executable, '-c', worker_code],",
            "        request=b'{}', timeout_seconds=2.0, max_response_bytes=64,",
            "        env=isolated_process.sanitized_worker_environment(),",
            "    )",
            "    publish(outcome_path, json.dumps({'failure': result.failure}))",
            "except BaseException as exc:",
            "    publish(outcome_path, json.dumps({'raised': type(exc).__name__}))",
        )
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            wrapper_source,
            os.fspath(Path(isolated_process.__file__).resolve().parents[1]),
            os.fspath(marker),
            os.fspath(crash_receipt),
            os.fspath(snapshot_receipt),
            os.fspath(outcome_receipt),
            worker_code,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    supervisor_id = 0
    worker_id = 0
    descendant_id = 0
    try:
        try:
            _wait_for_path(crash_receipt)
        except TimeoutError as exc:
            if process.poll() is not None and process.stderr is not None:
                raise AssertionError(process.stderr.read().decode("utf-8", "replace")) from exc
            raise
        supervisor_id, started_worker_id = map(
            int,
            crash_receipt.read_text(encoding="ascii").split(),
        )
        worker_id, descendant_id, worker_group_id, worker_session_id = map(
            int,
            marker.read_text(encoding="ascii").split(),
        )
        assert started_worker_id == worker_id
        assert worker_group_id == worker_id
        assert worker_session_id == supervisor_id

        if sys.platform == "darwin":
            deadline = time.monotonic() + 2.0
            observed_snapshot: dict[str, object] | None = None
            while time.monotonic() < deadline:
                try:
                    candidate = json.loads(snapshot_receipt.read_text(encoding="ascii"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    pass
                else:
                    if (
                        type(candidate) is dict
                        and type(candidate.get("count")) is int
                        and candidate["count"] >= 3
                        and isinstance(candidate.get("leader_state"), str)
                        and candidate["leader_state"].startswith("Z")
                        and [worker_id, worker_id] in candidate.get("members", [])
                        and [descendant_id, worker_id] in candidate.get("members", [])
                    ):
                        observed_snapshot = candidate
                        break
                time.sleep(0.005)
            assert observed_snapshot is not None
            assert process.poll() is None
            assert not outcome_receipt.exists()
            rows = _strict_process_table_rows()
            assert {
                (process_id, group_id)
                for process_id, group_id, _state in rows
                if process_id in {worker_id, descendant_id}
            } == {
                (worker_id, worker_id),
                (descendant_id, worker_id),
            }
            os.killpg(worker_id, signal.SIGKILL)
        assert process.wait(timeout=5.0) == 0
        outcome = json.loads(outcome_receipt.read_text(encoding="ascii"))
        assert outcome == {"failure": "crash"}
        assert process.stderr is not None
        assert process.stderr.read() == b""
        assert _identities_and_group_are_absent(
            (supervisor_id, worker_id, descendant_id),
            worker_id,
        )
    finally:
        if worker_id > 0:
            remaining_group = [row for row in _strict_process_table_rows() if row[1] == worker_id]
            if remaining_group:
                os.killpg(worker_id, signal.SIGKILL)
        if process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2.0)
        if process.stderr is not None:
            process.stderr.close()


def test_round11_parent_mask_is_unchanged_and_child_sigint_mask_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watcher publication restores the caller mask; the child starts unblocked."""

    real_pthread_sigmask = signal.pthread_sigmask
    original_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
    real_pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    parent_calls: list[tuple[int, frozenset[signal.Signals]]] = []

    def observe_parent_mask_change(
        operation: int,
        mask: set[signal.Signals],
    ) -> set[signal.Signals]:
        parent_calls.append((operation, frozenset(mask)))
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
            # Launch+OWNER, READY, CONTROL, STARTED, and response are all
            # required; the phase-derived deadline remains a hard failure.
            timeout_seconds=_SINGLE_WORKER_SUCCESS_SECONDS,
            max_response_bytes=32,
            env=isolated_process.sanitized_worker_environment(),
        )
        observed_parent_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
    finally:
        real_pthread_sigmask(signal.SIG_SETMASK, original_mask)

    assert result == isolated_process.IsolatedProcessResult(b"unblocked", None)
    assert observed_parent_mask == original_mask | {signal.SIGINT}
    assert parent_calls == [
        (
            signal.SIG_BLOCK,
            frozenset({signal.SIGINT, signal.SIGTERM}),
        ),
        (
            signal.SIG_SETMASK,
            frozenset(original_mask | {signal.SIGINT}),
        ),
    ]


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
        worker_entry = tmp_path / "round11-worker-entry.json"
        worker_before_descendant = tmp_path / "round11-worker-before-descendant.json"
        worker_after_descendant = tmp_path / "round11-worker-after-descendant.json"
        worker_before_response = tmp_path / "round11-worker-before-response.json"
        descendant_entry = tmp_path / "round11-descendant-entry.json"
        descendant_final = tmp_path / "round11-descendant-final.json"
        for descriptor in descriptors:
            os.set_inheritable(descriptor, True)
        checker = "\n".join(
            (
                "import json, os, sys",
                "from pathlib import Path",
                "process_identity = {'pid': os.getpid(), 'pgid': os.getpgrp(), 'sid': os.getsid(0)}",
                "Path(sys.argv[3]).write_text(json.dumps(process_identity, sort_keys=True), encoding='ascii')",
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
                "Path(sys.argv[4]).write_text(json.dumps({'identity': process_identity, 'leaked': leaked}, sort_keys=True), encoding='ascii')",
            )
        )
        worker = "\n".join(
            (
                "import json, os, subprocess, sys",
                "from pathlib import Path",
                "process_identity = {'pid': os.getpid(), 'pgid': os.getpgrp(), 'sid': os.getsid(0)}",
                "Path(sys.argv[3]).write_text(json.dumps(process_identity, sort_keys=True), encoding='ascii')",
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
                "Path(sys.argv[4]).write_text(json.dumps(process_identity, sort_keys=True), encoding='ascii')",
                f"child = subprocess.run([sys.executable, '-c', {checker!r}, sys.argv[1], sys.argv[2], sys.argv[7], sys.argv[8]], check=True)",
                "Path(sys.argv[5]).write_text(json.dumps({'identity': process_identity, 'returncode': child.returncode}, sort_keys=True), encoding='ascii')",
                "Path(sys.argv[6]).write_text(json.dumps({'identity': process_identity, 'leaked': leaked}, sort_keys=True), encoding='ascii')",
                "sys.stdout.buffer.write(json.dumps(leaked).encode('ascii'))",
            )
        )
        encoded_identities = json.dumps(identities, sort_keys=True)
        result = isolated_process.run_isolated_process(
            command=[
                sys.executable,
                "-c",
                worker,
                encoded_identities,
                os.fspath(receipt),
                os.fspath(worker_entry),
                os.fspath(worker_before_descendant),
                os.fspath(worker_after_descendant),
                os.fspath(worker_before_response),
                os.fspath(descendant_entry),
                os.fspath(descendant_final),
            ],
            request=b"{}",
            # The positive proof spans launch+OWNER, READY, CONTROL,
            # STARTED, worker, descendant, and receipt phases.
            timeout_seconds=_AMBIENT_DESCRIPTOR_SUCCESS_SECONDS,
            max_response_bytes=128,
            env=isolated_process.sanitized_worker_environment(),
        )

    assert result == isolated_process.IsolatedProcessResult(b"[]", None)
    assert json.loads(receipt.read_text(encoding="ascii")) == []


def test_round11_spawn_is_fail_closed_when_supervisor_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No direct-spawn fallback may bypass the close-all descriptor boundary."""

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("round11 isolated supervisor unavailable")

    monkeypatch.setattr(isolated_process, "_initialize_reserved_process", unavailable)

    result = isolated_process.run_isolated_process(
        command=[sys.executable, "-c", "raise SystemExit(0)"],
        request=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=32,
        env=isolated_process.sanitized_worker_environment(),
    )

    assert result == isolated_process.IsolatedProcessResult(None, "start")


@pytest.mark.parametrize(
    "stdout",
    [
        b"73 73 Z   ",
        b"73 73 Z   \r\n",
        b"73 73 Z\n",
        b"73 73 Z  \n",
        b"073 73 Z   \n",
        b"73 073 Z   \n",
        b"73 73 z   \n",
        b"73 73 Z!  \n",
        b"73 73 Z   \n73 73 Z   \n",
        b"73 73 Z   \n\n",
        b"1 1 S   \n" * 120_000,
    ],
    ids=[
        "missing-final-lf",
        "crlf",
        "missing-padding",
        "short-padding",
        "pid-leading-zero",
        "pgid-leading-zero",
        "lowercase-state",
        "unknown-state-suffix",
        "duplicate-pid",
        "blank-row",
        "oversized",
    ],
)
def test_round17_session_scanner_rejects_partial_or_noncanonical_full_tables(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
) -> None:
    monkeypatch.setattr(migration_process.sys, "platform", "darwin")

    with pytest.raises(migration_process._WorkerCleanupFailure):
        migration_process._canonical_process_table_rows(stdout, b"")


@pytest.mark.parametrize("state", ["K", "P", "x", "KNLsl+"])
def test_round17_generic_linux_scanner_accepts_documented_primary_states(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    monkeypatch.setattr(migration_process.sys, "platform", "linux")
    row = f"73 73 73 {state}\n".encode("ascii")

    assert migration_process._canonical_process_table_rows(row, b"") == ((73, 73, 73, state),)


@pytest.mark.parametrize(
    "state",
    [
        "R>",
        "RW",
        "R<",
        "RN",
        "RA",
        "RS",
        "RX",
        "RE",
        "RV",
        "RL",
        "Rs",
        "R+",
        "RWNAXEVLs+",
        "RWNSXEVLs+",
    ],
)
def test_round17_generic_darwin_scanner_accepts_documented_modifier_order(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    row = f"73 73 {state:<4}\n".encode("ascii")

    assert migration_process._canonical_process_table_rows(row, b"") == ((73, 73, None, state),)


def test_round17_malformed_session_scan_cannot_advance_empty_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial empty-looking table remains primary after later exact cleanup."""

    monkeypatch.setattr(migration_process.sys, "platform", "darwin")
    command = ["/bin/ps", "-axo", "pid=,pgid=,stat="]
    scans = iter(
        (
            b"73 73 Z   ",
            b"73 73 Z   \n",
            b"73 73 Z   \n",
            b"73 73 Z   \n",
        )
    )
    calls: list[tuple[object, dict[str, object]]] = []

    def run(
        invoked: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((invoked, kwargs))
        return subprocess.CompletedProcess(
            invoked,
            0,
            stdout=next(scans),
            stderr=b"",
        )

    wait_calls: list[float] = []
    process = SimpleNamespace(stdout=None, returncode=None)

    def wait(*, timeout: float) -> int:
        wait_calls.append(timeout)
        process.returncode = 0
        return 0

    process.wait = wait
    worker = migration_process._WorkerSession(
        process=process,  # type: ignore[arg-type]
        process_id=73,
        process_group_id=73,
        session_id=73,
        exit_watcher=None,
    )
    monkeypatch.setattr(migration_process.subprocess, "run", run)
    monkeypatch.setattr(migration_process.time, "sleep", lambda _seconds: None)

    with pytest.raises(migration_process._WorkerCleanupFailure):
        migration_process._cleanup_cooperative_worker_session(worker)

    assert worker.cleaned
    assert wait_calls == [2.0]
    assert len(calls) == 4
    assert all(call[0] == command for call in calls)
    assert all(
        call[1]
        == {
            "capture_output": True,
            "check": True,
            "env": {**os.environ, "LANG": "C", "LC_ALL": "C"},
            "stdin": subprocess.DEVNULL,
            "text": False,
            "timeout": 1.0,
        }
        for call in calls
    )


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

    def expire_after_capture(
        process: isolated_process._SpawnedProcess,
        retained_worker: Any,
    ) -> object:
        worker = real_capture(process, retained_worker)
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


def test_round11_preset_cancel_precedes_publication_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-set cancellation returns before any native ownership can exist."""

    real_initialize = isolated_process._initialize_reserved_process
    real_capture = isolated_process._capture_worker_session
    real_monotonic = time.monotonic
    controller_clock = SimpleNamespace(expired=False)
    controller_clock.monotonic = lambda: (
        real_monotonic() + (2.0 if controller_clock.expired else 0.0)
    )
    controller_clock.sleep = time.sleep
    captured_workers: list[Any] = []
    initializer_calls: list[bool] = []

    def initialize_then_expire(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        initializer_calls.append(True)
        real_initialize(process, *args, **kwargs)
        controller_clock.expired = True

    def observed_capture(
        process: isolated_process._SpawnedProcess,
        retained_worker: Any,
    ) -> object:
        worker = real_capture(process, retained_worker)
        captured_workers.append(worker)
        return worker

    cancellation = isolated_process.ProcessCancellation()
    cancellation.set()
    monkeypatch.setattr(isolated_process, "time", controller_clock)
    monkeypatch.setattr(
        isolated_process,
        "_initialize_reserved_process",
        initialize_then_expire,
    )
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
    assert initializer_calls == []
    assert captured_workers == []


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

    monkeypatch.setattr(isolated_process, "_PRIVATE_PIPE_STATE_POISONED", False)
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
        with pytest.raises(isolated_process.IsolatedProcessCleanupError):
            isolated_process.private_pipe_channel()
        assert endpoint.closed
        assert _descriptor_identity(descriptor) == replacement_identity
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        real_close(descriptor)
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
