"""Focused owned-descriptor close contracts for the trusted supervisor."""

from __future__ import annotations

import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from itertools import chain, repeat
from pathlib import Path

import pytest

from invoice_agents import isolated_process, spawn_supervisor, worker_bootstrap

_ROLE_DESCRIPTORS = {
    "status": 41,
    "control": 43,
    "owner": 45,
    "lifetime": 47,
    "worker": 49,
}
_WORKER_DESCRIPTOR = _ROLE_DESCRIPTORS["worker"]
_WAIT_SECONDS = 2.0
_MAX_PROTOCOL_LINE_BYTES = 64
_PROCESS_TABLE_COMMAND = ("/bin/ps", "-axo", "pid=,pgid=,stat=")
_PROCESS_TABLE_MAX_BYTES = 1_048_576
_LINUX_PROCESS_STATE_PATTERN = r"[DIKPRStTWXxZ](?:<|N)?L?s?l?\+?"
_STARTED_WRITE_FAULT_DESCENDANT_SECONDS = 2.0
_STARTED_WRITE_FAULT_MARKER_SECONDS = _STARTED_WRITE_FAULT_DESCENDANT_SECONDS + _WAIT_SECONDS
_PROCESS_TABLE_SCAN_TIMEOUT_SECONDS = 1.0
_STARTED_WRITE_FAULT_REAP_SECONDS = (
    _STARTED_WRITE_FAULT_MARKER_SECONDS + 2.0 * _PROCESS_TABLE_SCAN_TIMEOUT_SECONDS + _WAIT_SECONDS
)


def _read_line(descriptor: int) -> bytes:
    payload = bytearray()
    deadline = time.monotonic() + _WAIT_SECONDS
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not selector.select(remaining):
                raise TimeoutError("round12 child-close receipt was not published")
            chunk = os.read(descriptor, 1)
            if not chunk:
                raise EOFError("round12 child-close receipt ended early")
            payload.extend(chunk)
            if len(payload) > _MAX_PROTOCOL_LINE_BYTES:
                raise ValueError("round12 child-close receipt exceeded protocol limit")
            if chunk == b"\n":
                return bytes(payload)


def _wait_for_path(
    path: Path,
    *,
    wait_seconds: float = _WAIT_SECONDS,
) -> None:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.005)
    raise TimeoutError(f"bounded child receipt was not published: {path}")


def _read_exact_eof(descriptor: int) -> None:
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        if not selector.select(_WAIT_SECONDS):
            raise TimeoutError("bounded lifetime EOF was not published")
    if os.read(descriptor, 1):
        raise ValueError("lifetime endpoint carried unexpected data")


def _endpoint_is_readable(descriptor: int) -> bool:
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        return bool(selector.select(0.0))


def _read_strict_process_table_for_test() -> tuple[tuple[int, int, str], ...]:
    scan_environment = {**os.environ, "LANG": "C", "LC_ALL": "C"}
    completed = subprocess.run(
        list(_PROCESS_TABLE_COMMAND),
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=False,
        timeout=1.0,
        env=scan_environment,
    )
    return _strict_process_table_rows(completed.stdout, completed.stderr)


def _process_not_running(process_id: int) -> bool:
    return all(
        observed_process_id != process_id
        for observed_process_id, _process_group_id, _state in _read_strict_process_table_for_test()
    )


def _process_group_snapshot(process_group_id: int) -> tuple[tuple[int, str], ...]:
    return tuple(
        (process_id, state)
        for process_id, observed_group_id, state in _read_strict_process_table_for_test()
        if observed_group_id == process_group_id
    )


def _strict_process_table_rows(
    stdout: bytes,
    stderr: bytes,
    *,
    platform: str | None = None,
) -> tuple[tuple[int, int, str], ...]:
    if stderr:
        raise ValueError("process-table scanner produced stderr")
    if not stdout or len(stdout) > _PROCESS_TABLE_MAX_BYTES or not stdout.endswith(b"\n"):
        raise ValueError("process-table scanner output was empty, partial, or oversized")
    rows: list[tuple[int, int, str]] = []
    process_ids: set[int] = set()
    if b"\r" in stdout:
        raise ValueError("process-table scanner output used noncanonical line endings")
    target_platform = sys.platform if platform is None else platform
    row_pattern = re.compile(rb" *([1-9][0-9]*) +([1-9][0-9]*) +([^\x00-\x20\x7f-\xff]+)( *)")
    for raw_line in stdout.splitlines():
        matched = row_pattern.fullmatch(raw_line)
        if matched is None:
            raise ValueError("process-table scanner row was not canonical")
        raw_process_id, raw_process_group_id, raw_state, raw_padding = matched.groups()
        state = raw_state.decode("ascii")
        if target_platform == "darwin":
            valid_state = (
                re.fullmatch(
                    r"[?IRSTUZ](?:>|W)?(?:<|N)?(?:A|S)?X?E?V?L?s?\+?",
                    state,
                )
                is not None
            )
            expected_padding = max(0, 4 - len(state))
        elif target_platform.startswith("linux"):
            valid_state = re.fullmatch(_LINUX_PROCESS_STATE_PATTERN, state) is not None
            expected_padding = 0
        else:
            raise ValueError("process-table scanner platform was unsupported")
        if not valid_state:
            raise ValueError("process-table scanner state was not canonical")
        if len(raw_padding) != expected_padding:
            raise ValueError("process-table scanner state padding was not canonical")
        process_id = int(raw_process_id)
        process_group_id = int(raw_process_group_id)
        if process_id in process_ids:
            raise ValueError("process-table scanner repeated a process identifier")
        process_ids.add(process_id)
        rows.append((process_id, process_group_id, state))
    return tuple(sorted(rows))


def _canonical_process_table_row_for_test(
    process_id: int,
    process_group_id: int,
    state: str,
) -> str:
    if sys.platform == "darwin":
        state_field = f"{state:<4}"
    elif sys.platform.startswith("linux"):
        state_field = state
    else:
        raise RuntimeError("process-table test platform was unsupported")
    return f"{process_id} {process_group_id} {state_field}\n"


def _snapshot_proves_group_extinction_for_test(
    rows: tuple[tuple[int, int, str], ...],
    process_id: int,
) -> bool:
    leader_rows = [row for row in rows if row[0] == process_id]
    return (
        len(leader_rows) == 1
        and leader_rows[0][1] == process_id
        and leader_rows[0][2].startswith("Z")
        and all(
            member_id == process_id or process_group_id != process_id
            for member_id, process_group_id, _state in rows
        )
    )


def _finish_test_owned_process_tree(
    process: subprocess.Popen[bytes],
    *,
    cooperative_cleanup_writer: int,
    process_group_ids: set[int],
) -> None:
    """Use stable capabilities before observing that a test process tree is empty."""

    cleanup_failures: list[BaseException] = []
    try:
        os.close(cooperative_cleanup_writer)
    except BaseException as exc:
        cleanup_failures.append(exc)

    direct_group: tuple[tuple[int, str], ...] = ()
    try:
        direct_group = _process_group_snapshot(process.pid)
    except BaseException as exc:
        cleanup_failures.append(exc)
    if process.returncode is None:
        direct_group_ids = {process_id for process_id, _state in direct_group}
        if not direct_group:
            cleanup_failures.append(
                AssertionError(
                    "direct controller identity vanished before exact reap: "
                    f"pgid={process.pid} members={direct_group!r}"
                )
            )
        elif process.pid not in direct_group_ids:
            cleanup_failures.append(
                AssertionError(
                    "refusing emergency cleanup after direct-child identity was lost: "
                    f"pgid={process.pid} members={direct_group!r}"
                )
            )
        elif direct_group_ids != {process.pid}:
            cleanup_failures.append(
                AssertionError(
                    "test controller retained an unexpected same-group descendant: "
                    f"pgid={process.pid} members={direct_group!r}"
                )
            )
        elif any(not state.startswith("Z") for _process_id, state in direct_group):
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except BaseException as exc:
                cleanup_failures.append(exc)
    if process.returncode is None:
        try:
            process.wait(timeout=_WAIT_SECONDS)
        except BaseException as exc:
            cleanup_failures.append(exc)
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            process.wait(timeout=_WAIT_SECONDS)
        except BaseException as exc:
            cleanup_failures.append(exc)

    remaining_group_ids = {
        process_group_id for process_group_id in process_group_ids if process_group_id > 0
    }
    deadline = time.monotonic() + _WAIT_SECONDS
    last_snapshots: dict[int, tuple[tuple[int, str], ...]] = {}
    while remaining_group_ids and time.monotonic() < deadline:
        last_snapshots = {}
        for process_group_id in remaining_group_ids:
            try:
                last_snapshots[process_group_id] = _process_group_snapshot(process_group_id)
            except BaseException as exc:
                cleanup_failures.append(exc)
        remaining_group_ids = {
            process_group_id for process_group_id, snapshot in last_snapshots.items() if snapshot
        }
        if not remaining_group_ids:
            break
        try:
            time.sleep(0.005)
        except BaseException as exc:
            cleanup_failures.append(exc)
            break

    final_snapshots: dict[int, tuple[tuple[int, str], ...]] = {}
    for process_group_id in remaining_group_ids:
        try:
            final_snapshots[process_group_id] = _process_group_snapshot(process_group_id)
        except BaseException as exc:
            cleanup_failures.append(exc)
    if any(final_snapshots.values()):
        cleanup_failures.append(
            AssertionError(f"bounded emergency cleanup left test-owned groups: {final_snapshots!r}")
        )
    if cleanup_failures:
        raise cleanup_failures[0]


def _complete_test_resource_teardown(
    process: subprocess.Popen[bytes],
    *,
    cooperative_cleanup_writer: int,
    process_group_ids: set[int],
    owned_descriptors: set[int],
    required_eof_descriptors: tuple[int, ...] = (),
) -> None:
    """Finish process teardown and close every resource before exposing a failure."""

    cleanup_failures: list[BaseException] = []
    try:
        _finish_test_owned_process_tree(
            process,
            cooperative_cleanup_writer=cooperative_cleanup_writer,
            process_group_ids=process_group_ids,
        )
    except BaseException as exc:
        cleanup_failures.append(exc)
    for descriptor in required_eof_descriptors:
        try:
            _read_exact_eof(descriptor)
        except BaseException as exc:
            cleanup_failures.append(exc)
    for descriptor in tuple(owned_descriptors):
        owned_descriptors.remove(descriptor)
        try:
            os.close(descriptor)
        except BaseException as exc:
            cleanup_failures.append(exc)
    if process.stderr is not None:
        try:
            process.stderr.close()
        except BaseException as exc:
            cleanup_failures.append(exc)
    if cleanup_failures:
        raise cleanup_failures[0]


def _exact_child_was_reaped(process_id: int) -> bool:
    try:
        os.waitpid(process_id, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


def _install_successful_protocol(
    monkeypatch: pytest.MonkeyPatch,
    *,
    worker_returncode: int,
) -> list[int]:
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_arguments",
        lambda: (
            _ROLE_DESCRIPTORS["status"],
            _ROLE_DESCRIPTORS["control"],
            _ROLE_DESCRIPTORS["owner"],
            _ROLE_DESCRIPTORS["lifetime"],
            (_WORKER_DESCRIPTOR,),
            ["trusted-worker"],
        ),
    )
    monkeypatch.setattr(spawn_supervisor.signal, "pthread_sigmask", lambda *_args: set())
    monkeypatch.setattr(spawn_supervisor.signal, "signal", lambda *_args: signal.SIG_DFL)
    monkeypatch.setattr(spawn_supervisor, "_write_all", lambda *_args: None)
    monkeypatch.setattr(spawn_supervisor, "_read_control", lambda *_args: b"S")

    gate_pairs = iter(((51, 53), (55, 57)))
    monkeypatch.setattr(spawn_supervisor, "_open_directional_gate", lambda: next(gate_pairs))
    readiness = iter((b"R", b""))
    monkeypatch.setattr(spawn_supervisor.os, "read", lambda _descriptor, _size: next(readiness))
    launch_calls: list[int] = []

    def initialize(process: subprocess.Popen[bytes], *_args: object, **_kwargs: object) -> None:
        process.pid = 73
        process.returncode = None
        process._child_created = True
        launch_calls.append(process.pid)

    monkeypatch.setattr(spawn_supervisor, "_initialize_reserved_worker_process", initialize)
    monkeypatch.setattr(spawn_supervisor, "_corroborate_worker_identity", lambda *_args: None)
    monkeypatch.setattr(
        spawn_supervisor,
        "_wait_for_worker",
        lambda process_id: (worker_returncode << 8) if process_id == 73 else (99 << 8),
    )
    return launch_calls


def _inject_one_close_fault(
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str,
    fault_moment: str,
    fault_type: type[BaseException],
) -> tuple[list[int], list[int], Callable[[], int]]:
    target = _ROLE_DESCRIPTORS[role]
    attempts: list[int] = []
    probes: list[int] = []
    kernel_actions = 0

    def close(descriptor: int) -> None:
        nonlocal kernel_actions
        if descriptor not in _ROLE_DESCRIPTORS.values():
            return
        attempts.append(descriptor)
        if attempts.count(descriptor) > 1:
            raise AssertionError("supervisor retried an ambiguously closed raw descriptor")
        if descriptor != target:
            return
        if fault_moment == "post-action":
            kernel_actions += 1
        raise fault_type("round12 injected owned close failure")

    def fstat(descriptor: int) -> object:
        probes.append(descriptor)
        raise AssertionError("supervisor probed a retired raw descriptor")

    monkeypatch.setattr(spawn_supervisor.os, "close", close)
    monkeypatch.setattr(spawn_supervisor.os, "fstat", fstat)
    return attempts, probes, lambda: kernel_actions


def test_round12_supervisor_parent_never_explicitly_closes_lifetime_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent retires lifetime only through kernel close at process exit."""

    launch_calls = _install_successful_protocol(monkeypatch, worker_returncode=0)
    close_attempts: list[int] = []
    monkeypatch.setattr(spawn_supervisor.os, "close", close_attempts.append)

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 0
    assert launch_calls == [73]
    assert [
        descriptor for descriptor in close_attempts if descriptor in _ROLE_DESCRIPTORS.values()
    ] == [
        _ROLE_DESCRIPTORS["owner"],
        _ROLE_DESCRIPTORS["control"],
        _WORKER_DESCRIPTOR,
        _ROLE_DESCRIPTORS["status"],
    ]


def test_round13_worker_descriptor_is_retired_once_before_started_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trusted parent drops worker-only authority immediately after bootstrap spawn."""

    _install_successful_protocol(monkeypatch, worker_returncode=0)
    events: list[tuple[str, int, bytes | None]] = []

    def write(descriptor: int, payload: bytes) -> None:
        events.append(("write", descriptor, payload))

    def close(descriptor: int) -> None:
        events.append(("close", descriptor, None))

    monkeypatch.setattr(spawn_supervisor, "_write_all", write)
    monkeypatch.setattr(spawn_supervisor.os, "close", close)

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 0
    worker_close = ("close", _WORKER_DESCRIPTOR, None)
    started_write = ("write", _ROLE_DESCRIPTORS["status"], b"STARTED 73\n")
    assert events.count(worker_close) == 1
    assert events.index(worker_close) < events.index(started_write)


@pytest.mark.parametrize(
    "write_mode",
    ["full", "zero", "partial", "pre-error", "post-error"],
)
def test_round12_owner_is_one_atomic_raw_write_or_sanitized_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    write_mode: str,
) -> None:
    """OWNER is the first one-shot PIPE_BUF write; uncertainty cannot retry or proceed."""

    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_arguments",
        lambda: (
            _ROLE_DESCRIPTORS["status"],
            _ROLE_DESCRIPTORS["control"],
            _ROLE_DESCRIPTORS["owner"],
            _ROLE_DESCRIPTORS["lifetime"],
            (),
            ["trusted-worker"],
        ),
    )
    monkeypatch.setattr(spawn_supervisor.signal, "pthread_sigmask", lambda *_args: set())
    monkeypatch.setattr(spawn_supervisor.os, "getpid", lambda: 59)
    writes: list[tuple[int, bytes]] = []
    owner_bytes_committed = 0

    def write(descriptor: int, payload: bytes) -> int:
        nonlocal owner_bytes_committed
        writes.append((descriptor, payload))
        if descriptor != _ROLE_DESCRIPTORS["owner"]:
            return len(payload)
        if len([call for call in writes if call[0] == descriptor]) > 1:
            raise AssertionError("supervisor retried the OWNER raw write")
        if write_mode == "full":
            owner_bytes_committed = len(payload)
            return len(payload)
        if write_mode == "zero":
            return 0
        if write_mode == "partial":
            owner_bytes_committed = len(payload) - 1
            return len(payload) - 1
        if write_mode == "post-error":
            owner_bytes_committed = len(payload)
        raise OSError("round12 injected OWNER raw write failure")

    close_attempts: list[int] = []
    control_reads: list[tuple[int, int]] = []
    launch_calls: list[bool] = []
    monkeypatch.setattr(spawn_supervisor.os, "write", write)
    monkeypatch.setattr(spawn_supervisor.os, "close", close_attempts.append)

    def read_control(status_descriptor: int, control_descriptor: int) -> bytes:
        control_reads.append((status_descriptor, control_descriptor))
        return b"A"

    def initialize(*_args: object, **_kwargs: object) -> None:
        launch_calls.append(True)
        raise AssertionError("worker bootstrap launch was forbidden")

    monkeypatch.setattr(spawn_supervisor, "_read_control", read_control)
    monkeypatch.setattr(spawn_supervisor, "_initialize_reserved_worker_process", initialize)

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    owner = b"OWNER 59\n"
    assert len(owner) <= os.pathconf(os.curdir, "PC_PIPE_BUF")
    assert writes[0] == (_ROLE_DESCRIPTORS["owner"], owner)
    assert [call for call in writes if call[0] == _ROLE_DESCRIPTORS["owner"]] == [
        (_ROLE_DESCRIPTORS["owner"], owner)
    ]
    assert launch_calls == []
    if write_mode == "full":
        assert raised.value.code == 0
        assert writes == [
            (_ROLE_DESCRIPTORS["owner"], owner),
            (_ROLE_DESCRIPTORS["status"], b"READY 59\n"),
        ]
        assert owner_bytes_committed == len(owner)
        assert control_reads == [(_ROLE_DESCRIPTORS["status"], _ROLE_DESCRIPTORS["control"])]
    else:
        assert raised.value.code == 127
        assert writes == [(_ROLE_DESCRIPTORS["owner"], owner)]
        assert control_reads == []
        assert (
            owner_bytes_committed
            == {
                "zero": 0,
                "partial": len(owner) - 1,
                "pre-error": 0,
                "post-error": len(owner),
            }[write_mode]
        )
    assert close_attempts == (
        [
            _ROLE_DESCRIPTORS["owner"],
            _ROLE_DESCRIPTORS["control"],
            _ROLE_DESCRIPTORS["status"],
        ]
        if write_mode == "full"
        else [
            _ROLE_DESCRIPTORS["owner"],
            _ROLE_DESCRIPTORS["status"],
            _ROLE_DESCRIPTORS["control"],
        ]
    )


def test_round18_clean_bootstrap_inherits_only_directional_gates_and_worker_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supervisor, control, owner, and lifetime authority cannot enter the bootstrap."""

    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_arguments",
        lambda: (
            _ROLE_DESCRIPTORS["status"],
            _ROLE_DESCRIPTORS["control"],
            _ROLE_DESCRIPTORS["owner"],
            _ROLE_DESCRIPTORS["lifetime"],
            (_WORKER_DESCRIPTOR,),
            ["trusted-worker"],
        ),
    )
    monkeypatch.setattr(spawn_supervisor, "_write_all", lambda *_args: None)
    monkeypatch.setattr(spawn_supervisor, "_read_control", lambda *_args: b"S")
    gate_pairs = iter(((51, 53), (55, 57)))
    monkeypatch.setattr(spawn_supervisor, "_open_directional_gate", lambda: next(gate_pairs))
    monkeypatch.setattr(spawn_supervisor.os, "close", lambda _descriptor: None)
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def reject_before_child(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        captured.append((args, kwargs))
        assert getattr(process, "_child_created", None) is False
        raise RuntimeError("round18 stop before native child creation")

    monkeypatch.setattr(
        spawn_supervisor,
        "_initialize_reserved_worker_process",
        reject_before_child,
    )

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 127
    assert len(captured) == 1
    arguments, options = captured[0]
    bootstrap_command = arguments[0]
    assert isinstance(bootstrap_command, list)
    assert options["close_fds"] is True
    assert options["process_group"] == 0
    assert options["pass_fds"] == (53, 55, _WORKER_DESCRIPTOR)
    assert not set(options["pass_fds"]) & {
        _ROLE_DESCRIPTORS["status"],
        _ROLE_DESCRIPTORS["control"],
        _ROLE_DESCRIPTORS["owner"],
        _ROLE_DESCRIPTORS["lifetime"],
    }
    assert bootstrap_command[:3] == [
        os.fspath(Path(sys.executable).resolve()),
        "-I",
        os.fspath(Path(spawn_supervisor.__file__).with_name("worker_bootstrap.py").resolve()),
    ]
    assert bootstrap_command[3:] == [
        "--ready-fd",
        "53",
        "--start-fd",
        "55",
        "--supervisor-sid",
        str(os.getpid()),
        "--worker-fd",
        str(_WORKER_DESCRIPTOR),
        "--",
        "trusted-worker",
    ]


def test_round17_sigint_after_exact_wnohang_reap_cannot_reopen_group_authority(
    tmp_path: Path,
) -> None:
    """A pending SIGINT becomes visible only after exact reap retires the PGID."""

    status_reader, status_writer = os.pipe()
    control_reader, control_writer = os.pipe()
    owner_reader, owner_writer = os.pipe()
    lifetime_reader, lifetime_writer = os.pipe()
    reaped_receipt = tmp_path / "sigint-after-reap-worker"
    outcome_receipt = tmp_path / "sigint-after-reap-outcome"
    post_reap_receipt = tmp_path / "sigint-after-reap-group-operation"
    wrapper_source = "\n".join(
        (
            "import importlib.util, json, os, signal, sys, time",
            "from pathlib import Path",
            "path, status, control, owner, lifetime, reaped_path, outcome_path, post_reap_path = sys.argv[1:9]",
            "status, control, owner, lifetime = map(int, (status, control, owner, lifetime))",
            "spec = importlib.util.spec_from_file_location('round17_sigint_supervisor', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "real_kill = os.kill",
            "real_killpg = os.killpg",
            "real_waitpid = os.waitpid",
            "real_scan = module._invoke_process_table_scan",
            "real_sleep = time.sleep",
            "reaped = False",
            "def publish(path, payload):",
            "    destination = Path(path)",
            "    temporary = destination.with_name(f'{destination.name}.{os.getpid()}.tmp')",
            "    temporary.write_text(payload, encoding='ascii')",
            "    os.replace(temporary, destination)",
            "def block_on_post_reap(action):",
            "    publish(post_reap_path, action)",
            "    signal.pause()",
            "def kill(process_id, signal_number):",
            "    if reaped: block_on_post_reap(f'kill:{process_id}:{signal_number}')",
            "    return real_kill(process_id, signal_number)",
            "def killpg(process_group_id, signal_number):",
            "    if reaped: block_on_post_reap(f'killpg:{process_group_id}:{signal_number}')",
            "    return real_killpg(process_group_id, signal_number)",
            "def scan():",
            "    if reaped: block_on_post_reap('scan')",
            "    return real_scan()",
            "def sleep(seconds):",
            "    if reaped: block_on_post_reap('sleep')",
            "    return real_sleep(seconds)",
            "def waitpid(process_id, options):",
            "    global reaped",
            "    observed_id, wait_status = real_waitpid(process_id, options)",
            "    if options == os.WNOHANG and observed_id == process_id:",
            "        reaped = True",
            "        publish(reaped_path, str(process_id))",
            "        real_kill(os.getpid(), signal.SIGINT)",
            "    return observed_id, wait_status",
            "module.os.kill = kill",
            "module.os.killpg = killpg",
            "module.os.waitpid = waitpid",
            "module._invoke_process_table_scan = scan",
            "module.time.sleep = sleep",
            "sys.argv = [path, '--status-fd', str(status), '--control-fd', str(control), '--owner-fd', str(owner), '--lifetime-fd', str(lifetime), '--', *sys.argv[9:]]",
            "try:",
            "    module.main()",
            "except BaseException as exc:",
            "    publish(outcome_path, json.dumps({'type': type(exc).__name__, 'code': getattr(exc, 'code', None)}, sort_keys=True))",
            "    raise",
        )
    )
    process = subprocess.Popen(
        [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            wrapper_source,
            os.fspath(Path(spawn_supervisor.__file__).resolve()),
            str(status_writer),
            str(control_reader),
            str(owner_writer),
            str(lifetime_writer),
            os.fspath(reaped_receipt),
            os.fspath(outcome_receipt),
            os.fspath(post_reap_receipt),
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            "pass",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
        pass_fds=(status_writer, control_reader, owner_writer, lifetime_writer),
    )
    for descriptor in (status_writer, control_reader, owner_writer, lifetime_writer):
        os.close(descriptor)
    forced_test_cleanup = False
    started_worker_id = 0
    try:
        assert _read_line(owner_reader) == f"OWNER {process.pid}\n".encode("ascii")
        assert _read_line(status_reader) == f"READY {process.pid}\n".encode("ascii")
        os.write(control_writer, b"S")
        assert _read_line(status_reader) == b"CONTROL S\n"
        os.close(control_writer)
        started = _read_line(status_reader)
        assert started.startswith(b"STARTED ")
        started_worker_id = int(started.removeprefix(b"STARTED ").removesuffix(b"\n"))
        assert process.wait(timeout=_WAIT_SECONDS) == 127
        _wait_for_path(reaped_receipt)
        _wait_for_path(outcome_receipt)
        assert reaped_receipt.read_text(encoding="ascii") == str(started_worker_id)
        assert json.loads(outcome_receipt.read_text(encoding="ascii")) == {
            "code": 127,
            "type": "SystemExit",
        }
        assert not post_reap_receipt.exists()
        assert os.read(lifetime_reader, 1) == b""
        assert process.stderr is not None
        assert process.stderr.read() == b""
        assert _process_not_running(started_worker_id)
        assert _process_group_snapshot(started_worker_id) == ()
    finally:
        if process.poll() is None:
            forced_test_cleanup = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=_WAIT_SECONDS)
        for descriptor in (status_reader, control_writer, owner_reader, lifetime_reader):
            with suppress(OSError):
                os.close(descriptor)
        if process.stderr is not None:
            process.stderr.close()

    assert not forced_test_cleanup


@pytest.mark.parametrize("fault_moment", ["pre-action", "post-action"])
@pytest.mark.parametrize("fault_type_name", ["OSError", "KeyboardInterrupt", "SystemExit"])
def test_round13_worker_fd_close_uncertainty_never_publishes_started_or_admits_credential(
    tmp_path: Path,
    fault_moment: str,
    fault_type_name: str,
) -> None:
    """A real worker-only close fault aborts the gated child before exec or STARTED."""

    status_reader, status_writer = os.pipe()
    control_reader, control_writer = os.pipe()
    owner_reader, owner_writer = os.pipe()
    lifetime_reader, lifetime_writer = os.pipe()
    credential_reader, credential_writer = os.pipe()
    audit_reader, audit_writer = os.pipe()
    credential_receipt = tmp_path / f"{fault_moment}-{fault_type_name}-credential"
    canary = b"round13-worker-close-private-canary"
    worker_source = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "payload = os.read(int(sys.argv[1]), 4096)",
            f"Path({os.fspath(credential_receipt)!r}).write_bytes(payload)",
        )
    )
    wrapper_source = "\n".join(
        (
            "import importlib.util, os, sys",
            "path, status, control, owner, lifetime, worker_fd, audit, moment, fault_name = sys.argv[1:10]",
            "status, control, owner, lifetime, worker_fd, audit = map(int, (status, control, owner, lifetime, worker_fd, audit))",
            "spec = importlib.util.spec_from_file_location('round13_spawn_supervisor', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "parent = os.getpid()",
            "real_close = os.close",
            "fault_type = {'OSError': OSError, 'KeyboardInterrupt': KeyboardInterrupt, 'SystemExit': SystemExit}[fault_name]",
            "attempted = False",
            "def close(descriptor):",
            "    global attempted",
            "    if os.getpid() == parent and descriptor == worker_fd:",
            "        os.write(audit, b'C')",
            "        if attempted: raise AssertionError('worker FD close retried')",
            "        attempted = True",
            "        if moment == 'post-action': real_close(descriptor)",
            "        raise fault_type('round13 injected worker FD close failure')",
            "    return real_close(descriptor)",
            "module.os.close = close",
            "sys.argv = [path, '--status-fd', str(status), '--control-fd', str(control), '--owner-fd', str(owner), '--lifetime-fd', str(lifetime), '--worker-fd', str(worker_fd), '--', *sys.argv[10:]]",
            "module.main()",
        )
    )
    process = subprocess.Popen(
        [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            wrapper_source,
            os.fspath(Path(spawn_supervisor.__file__).resolve()),
            str(status_writer),
            str(control_reader),
            str(owner_writer),
            str(lifetime_writer),
            str(credential_reader),
            str(audit_writer),
            fault_moment,
            fault_type_name,
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            worker_source,
            str(credential_reader),
        ],
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
            audit_writer,
        ),
    )
    supervisor_identity = (process.pid, os.getpgid(process.pid), os.getsid(process.pid))
    for descriptor in (
        status_writer,
        control_reader,
        owner_writer,
        lifetime_writer,
        credential_reader,
        audit_writer,
    ):
        os.close(descriptor)
    status_lines: list[bytes] = []
    returncode: int | None = None
    supervisor_not_running = False
    supervisor_reaped = False
    group_after_return: tuple[tuple[int, str], ...] | None = None
    forced_test_cleanup = False
    close_attempt_receipt = b""
    try:
        assert _read_line(owner_reader) == f"OWNER {process.pid}\n".encode("ascii")
        status_lines.append(_read_line(status_reader))
        os.write(credential_writer, canary)
        os.close(credential_writer)
        os.write(control_writer, b"S")
        status_lines.append(_read_line(status_reader))
        os.close(control_writer)
        while True:
            try:
                status_lines.append(_read_line(status_reader))
            except EOFError:
                break
            if len(status_lines) > 8:
                raise AssertionError("supervisor published an unbounded status stream")
        returncode = process.wait(timeout=_WAIT_SECONDS)
        close_attempt_receipt = os.read(audit_reader, 32)
        assert os.read(lifetime_reader, 1) == b""
        assert process.stderr is not None
        assert process.stderr.read() == b""
        supervisor_not_running = _process_not_running(process.pid)
        supervisor_reaped = _exact_child_was_reaped(process.pid)
        group_after_return = _process_group_snapshot(process.pid)
    finally:
        remaining_group = _process_group_snapshot(process.pid)
        if process.poll() is None or remaining_group:
            forced_test_cleanup = True
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired, ChildProcessError):
                process.wait(timeout=_WAIT_SECONDS)
        for descriptor in (
            status_reader,
            control_writer,
            owner_reader,
            lifetime_reader,
            credential_writer,
            audit_reader,
        ):
            with suppress(OSError):
                os.close(descriptor)
        if process.stderr is not None:
            process.stderr.close()

    assert status_lines == [f"READY {process.pid}\n".encode("ascii"), b"CONTROL S\n"]
    assert all(not line.startswith(b"STARTED ") for line in status_lines)
    assert not credential_receipt.exists()
    assert close_attempt_receipt == b"C"
    assert returncode == 127
    assert supervisor_identity == (process.pid, process.pid, process.pid)
    assert supervisor_not_running
    assert supervisor_reaped
    assert group_after_return == ()
    assert not forced_test_cleanup


def test_round17_start_waits_for_exact_child_bootstrap_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity and exact READY EOF must precede the one-way START publication."""

    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_arguments",
        lambda: (
            _ROLE_DESCRIPTORS["status"],
            _ROLE_DESCRIPTORS["control"],
            _ROLE_DESCRIPTORS["owner"],
            _ROLE_DESCRIPTORS["lifetime"],
            (_WORKER_DESCRIPTOR,),
            ["trusted-worker"],
        ),
    )
    monkeypatch.setattr(spawn_supervisor.signal, "pthread_sigmask", lambda *_args: set())
    monkeypatch.setattr(spawn_supervisor.signal, "signal", lambda *_args: signal.SIG_DFL)
    monkeypatch.setattr(spawn_supervisor, "_read_control", lambda *_args: b"S")
    gate_pairs = iter(((51, 53), (55, 57)))
    monkeypatch.setattr(spawn_supervisor, "_open_directional_gate", lambda: next(gate_pairs))
    events: list[tuple[str, int, bytes | None]] = []

    def initialize(process: subprocess.Popen[bytes], *_args: object, **_kwargs: object) -> None:
        process.pid = 73
        process.returncode = None
        process._child_created = True
        events.append(("initialized", process.pid, None))

    def corroborate(process_id: int, session_id: int) -> None:
        assert (process_id, session_id) == (73, os.getpid())
        events.append(("identity", process_id, None))

    readiness = iter((b"R", b""))

    def read(descriptor: int, size: int) -> bytes:
        assert (descriptor, size) == (51, 1)
        payload = next(readiness)
        events.append(("read", descriptor, payload))
        return payload

    def write(descriptor: int, payload: bytes) -> None:
        events.append(("write", descriptor, payload))

    def close(descriptor: int) -> None:
        events.append(("close", descriptor, None))

    monkeypatch.setattr(spawn_supervisor, "_initialize_reserved_worker_process", initialize)
    monkeypatch.setattr(spawn_supervisor, "_corroborate_worker_identity", corroborate)
    monkeypatch.setattr(spawn_supervisor, "_wait_for_worker", lambda _process_id: 0)
    monkeypatch.setattr(spawn_supervisor.os, "read", read)
    monkeypatch.setattr(spawn_supervisor, "_write_all", write)
    monkeypatch.setattr(spawn_supervisor.os, "close", close)

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 0
    identity = ("identity", 73, None)
    ready = ("read", 51, b"R")
    ready_eof = ("read", 51, b"")
    worker_retired = ("close", _WORKER_DESCRIPTOR, None)
    start = ("write", 57, b"S")
    started = ("write", _ROLE_DESCRIPTORS["status"], b"STARTED 73\n")
    assert events.index(identity) < events.index(ready)
    assert events.index(ready) < events.index(ready_eof)
    assert events.index(ready_eof) < events.index(worker_retired)
    assert events.index(worker_retired) < events.index(start)
    assert events.index(start) < events.index(started)


@pytest.mark.parametrize("fault_moment", ["pre-action", "post-action"])
@pytest.mark.parametrize("fault_type_name", ["OSError", "KeyboardInterrupt", "SystemExit"])
def test_round14_started_write_uncertainty_locally_reaps_admitted_worker_group(
    tmp_path: Path,
    fault_moment: str,
    fault_type_name: str,
) -> None:
    """STARTED uncertainty cannot let an admitted worker outlive supervisor lifetime."""

    status_reader, status_writer = os.pipe()
    control_reader, control_writer = os.pipe()
    owner_reader, owner_writer = os.pipe()
    lifetime_reader, lifetime_writer = os.pipe()
    worker_cleanup_reader, worker_cleanup_writer = os.pipe()
    worker_tree_lifetime_reader, worker_tree_lifetime_writer = os.pipe()
    worker_id_receipt = tmp_path / f"{fault_moment}-{fault_type_name}-worker-id"
    reaped_receipt = tmp_path / f"{fault_moment}-{fault_type_name}-reaped"
    worker_marker = tmp_path / f"{fault_moment}-{fault_type_name}-worker-marker"
    descendant_ready = tmp_path / f"{fault_moment}-{fault_type_name}-descendant-ready"
    descendant_source = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "ready = Path(sys.argv[2])",
            "ready_temp = ready.with_name(f'{ready.name}.{os.getpid()}.tmp')",
            "ready_temp.write_text(str(os.getpid()), encoding='ascii')",
            "os.replace(ready_temp, ready)",
            "os.read(int(sys.argv[1]), 1)",
        )
    )
    worker_source = "\n".join(
        (
            "import os, subprocess, sys, time",
            "from pathlib import Path",
            "cleanup = int(sys.argv[1])",
            "tree_lifetime = int(sys.argv[2])",
            "descendant_ready = Path(sys.argv[3])",
            f"child = subprocess.Popen([sys.executable, '-I', '-c', {descendant_source!r}, str(cleanup), os.fspath(descendant_ready)], pass_fds=(cleanup, tree_lifetime))",
            "ready_deadline = time.monotonic() + 2.0",
            "descendant_blocked = False",
            "while time.monotonic() < ready_deadline:",
            "    if descendant_ready.exists():",
            "        observed = subprocess.run(['/bin/ps', '-o', 'stat=', '-p', str(child.pid)], check=False, capture_output=True, text=True, timeout=2.0)",
            "        if observed.returncode == 0 and observed.stdout.strip().startswith('S'):",
            "            descendant_blocked = True",
            "            break",
            "    time.sleep(0.005)",
            "if not descendant_blocked: raise TimeoutError('round14 descendant did not reach its lifetime read')",
            f"marker = Path({os.fspath(worker_marker)!r})",
            "marker_temp = marker.with_name(f'{marker.name}.{os.getpid()}.tmp')",
            "marker_temp.write_text(f'{os.getpid()} {child.pid}', encoding='ascii')",
            "os.replace(marker_temp, marker)",
            "os.read(cleanup, 1)",
        )
    )
    wrapper_source = "\n".join(
        (
            "import importlib.util, os, sys, time",
            "from pathlib import Path",
            "path, status, control, owner, lifetime, moment, fault_name, worker_id_path, reaped_path, cleanup, tree_lifetime = sys.argv[1:12]",
            "status, control, owner, lifetime, cleanup, tree_lifetime = map(int, (status, control, owner, lifetime, cleanup, tree_lifetime))",
            "spec = importlib.util.spec_from_file_location('round14_spawn_supervisor', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "parent = os.getpid()",
            "real_write_all = module._write_all",
            "real_waitpid = os.waitpid",
            "fault_type = {'OSError': OSError, 'KeyboardInterrupt': KeyboardInterrupt, 'SystemExit': SystemExit}[fault_name]",
            "target = 0",
            "def write_all(descriptor, payload):",
            "    global target",
            "    if os.getpid() == parent and descriptor == status and payload.startswith(b'STARTED '):",
            "        encoded = payload.removeprefix(b'STARTED ').removesuffix(b'\\n')",
            "        target = int(encoded)",
            "        Path(worker_id_path).write_text(str(target), encoding='ascii')",
            f"        deadline = time.monotonic() + {_STARTED_WRITE_FAULT_MARKER_SECONDS!r}",
            f"        marker = Path({os.fspath(worker_marker)!r})",
            "        while not marker.exists() and time.monotonic() < deadline: time.sleep(0.005)",
            "        if not marker.exists(): raise TimeoutError('round14 worker readiness was not published')",
            "        if moment == 'post-action': real_write_all(descriptor, payload)",
            "        raise fault_type('round14 injected STARTED write control')",
            "    return real_write_all(descriptor, payload)",
            "def waitpid(process_id, options):",
            "    observed_id, wait_status = real_waitpid(process_id, options)",
            "    if target and process_id == target and observed_id == target:",
            "        Path(reaped_path).write_text(str(target), encoding='ascii')",
            "    return observed_id, wait_status",
            "module._write_all = write_all",
            "module.os.waitpid = waitpid",
            "sys.argv = [path, '--status-fd', str(status), '--control-fd', str(control), '--owner-fd', str(owner), '--lifetime-fd', str(lifetime), '--worker-fd', str(cleanup), '--worker-fd', str(tree_lifetime), '--', *sys.argv[12:]]",
            "module.main()",
        )
    )
    process = subprocess.Popen(
        [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            wrapper_source,
            os.fspath(Path(spawn_supervisor.__file__).resolve()),
            str(status_writer),
            str(control_reader),
            str(owner_writer),
            str(lifetime_writer),
            fault_moment,
            fault_type_name,
            os.fspath(worker_id_receipt),
            os.fspath(reaped_receipt),
            str(worker_cleanup_reader),
            str(worker_tree_lifetime_writer),
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            worker_source,
            str(worker_cleanup_reader),
            str(worker_tree_lifetime_writer),
            os.fspath(descendant_ready),
        ],
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
            worker_cleanup_reader,
            worker_tree_lifetime_writer,
        ),
    )
    supervisor_identity = (process.pid, os.getpgid(process.pid), os.getsid(process.pid))
    for descriptor in (
        status_writer,
        control_reader,
        owner_writer,
        lifetime_writer,
        worker_cleanup_reader,
        worker_tree_lifetime_writer,
    ):
        os.close(descriptor)
    status_lines: list[bytes] = []
    worker_id = 0
    worker_ids: tuple[int, ...] = ()
    worker_reaped_before_lifetime_eof = False
    worker_not_running_at_lifetime_eof = False
    worker_group_at_lifetime_eof: tuple[tuple[int, str], ...] | None = None
    supervisor_descendants_at_lifetime_eof: tuple[tuple[int, str], ...] | None = None
    returncode: int | None = None
    group_after_return: tuple[tuple[int, str], ...] | None = None
    worker_group_after_return: tuple[tuple[int, str], ...] | None = None
    forced_test_cleanup = False
    owned_test_descriptors = {
        status_reader,
        control_writer,
        owner_reader,
        lifetime_reader,
        worker_tree_lifetime_reader,
    }
    try:
        assert _read_line(owner_reader) == f"OWNER {process.pid}\n".encode("ascii")
        status_lines.append(_read_line(status_reader))
        os.write(control_writer, b"S")
        status_lines.append(_read_line(status_reader))
        owned_test_descriptors.remove(control_writer)
        os.close(control_writer)
        # The intercepted STARTED write can wait for the worker's descendant
        # receipt, then cleanup requires two complete one-second ps scans.
        # One normal protocol window is reserved for scheduling and waitpid.
        _wait_for_path(
            reaped_receipt,
            wait_seconds=_STARTED_WRITE_FAULT_REAP_SECONDS,
        )
        with suppress(EOFError):
            status_lines.append(_read_line(status_reader))
        _read_exact_eof(lifetime_reader)
        worker_id = int(worker_id_receipt.read_text(encoding="ascii"))
        worker_reaped_before_lifetime_eof = (
            reaped_receipt.read_text(encoding="ascii") == str(worker_id)
            if reaped_receipt.exists()
            else False
        )
        worker_not_running_at_lifetime_eof = _process_not_running(worker_id)
        worker_group_at_lifetime_eof = _process_group_snapshot(worker_id)
        supervisor_descendants_at_lifetime_eof = tuple(
            member for member in _process_group_snapshot(process.pid) if member[0] != process.pid
        )
        worker_ids = tuple(
            int(value) for value in worker_marker.read_text(encoding="ascii").split()
        )
        returncode = process.wait(timeout=_WAIT_SECONDS)
        group_after_return = _process_group_snapshot(process.pid)
        worker_group_after_return = _process_group_snapshot(worker_id)
    finally:
        if worker_id_receipt.exists() and not worker_id:
            worker_id = int(worker_id_receipt.read_text(encoding="ascii"))
        if worker_marker.exists():
            worker_ids = tuple(
                int(value) for value in worker_marker.read_text(encoding="ascii").split()
            )
            if worker_ids and not worker_id:
                worker_id = worker_ids[0]
        remaining_groups = {
            process.pid,
            worker_id,
            *(worker_ids or ()),
        }
        if process.returncode is None or any(
            process_group_id > 0 and _process_group_snapshot(process_group_id)
            for process_group_id in remaining_groups
        ):
            forced_test_cleanup = True
        _complete_test_resource_teardown(
            process,
            cooperative_cleanup_writer=worker_cleanup_writer,
            process_group_ids=remaining_groups,
            owned_descriptors=owned_test_descriptors,
            required_eof_descriptors=(worker_tree_lifetime_reader,),
        )

    assert status_lines[:2] == [
        f"READY {process.pid}\n".encode("ascii"),
        b"CONTROL S\n",
    ]
    if fault_moment == "pre-action":
        assert len(status_lines) == 2
    else:
        assert len(status_lines) == 3
        assert status_lines[2] == f"STARTED {worker_id}\n".encode("ascii")
    assert supervisor_identity == (process.pid, process.pid, process.pid)
    assert worker_id > 0
    assert len(worker_ids) == 2
    assert worker_ids[0] == worker_id
    assert descendant_ready.read_text(encoding="ascii") == str(worker_ids[1])
    assert worker_reaped_before_lifetime_eof
    assert worker_not_running_at_lifetime_eof
    assert worker_group_at_lifetime_eof == ()
    assert supervisor_descendants_at_lifetime_eof == ()
    assert all(_process_not_running(process_id) for process_id in worker_ids)
    assert returncode == 127
    assert group_after_return == ()
    assert worker_group_after_return == ()
    assert not forced_test_cleanup


@pytest.mark.parametrize(
    "fault_site",
    ["kill", "waitpid", "killpg", "scan-invoke", "scan-parse", "sleep"],
)
@pytest.mark.parametrize("fault_moment", ["pre-action", "post-action"])
@pytest.mark.parametrize(
    "control_type_name",
    ["OSError", "KeyboardInterrupt", "SystemExit"],
)
def test_round14_cleanup_control_is_raised_only_after_exact_group_reap(
    tmp_path: Path,
    fault_site: str,
    fault_moment: str,
    control_type_name: str,
) -> None:
    """Cleanup preserves its first control but cannot expose it before exact reap."""

    lifetime_reader, lifetime_writer = os.pipe()
    worker_cleanup_reader, worker_cleanup_writer = os.pipe()
    marker = tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-marker"
    worker_id_receipt = tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-worker-id"
    reaped_receipt = tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-reaped"
    control_receipt = tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-control"
    wait_entered = tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-wait"
    wait_group_snapshot = tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-wait-group"
    allow_reap = tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-allow"
    post_reap_group_operation = (
        tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-post-reap-group"
    )
    group_kill_attempts_receipt = (
        tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-group-kills"
    )
    descendant_ready = (
        tmp_path / f"{fault_site}-{fault_moment}-{control_type_name}-descendant-ready"
    )
    descendant_source = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "ready = Path(sys.argv[2])",
            "ready_temp = ready.with_name(f'{ready.name}.{os.getpid()}.tmp')",
            "ready_temp.write_text(str(os.getpid()), encoding='ascii')",
            "os.replace(ready_temp, ready)",
            "os.read(int(sys.argv[1]), 1)",
        )
    )
    worker_source = "\n".join(
        (
            "import os, subprocess, sys, time",
            "from pathlib import Path",
            "cleanup = int(sys.argv[2])",
            "descendant_ready = Path(sys.argv[3])",
            f"child = subprocess.Popen([sys.executable, '-I', '-c', {descendant_source!r}, str(cleanup), os.fspath(descendant_ready)], pass_fds=(cleanup,))",
            "ready_deadline = time.monotonic() + 2.0",
            "descendant_blocked = False",
            "while time.monotonic() < ready_deadline:",
            "    if descendant_ready.exists():",
            "        observed = subprocess.run(['/bin/ps', '-o', 'stat=', '-p', str(child.pid)], check=False, capture_output=True, text=True, timeout=2.0)",
            "        if observed.returncode == 0 and observed.stdout.strip().startswith('S'):",
            "            descendant_blocked = True",
            "            break",
            "    time.sleep(0.005)",
            "if not descendant_blocked: raise TimeoutError('round14 descendant did not reach its lifetime read')",
            "marker = Path(sys.argv[1])",
            "marker_temp = marker.with_name(f'{marker.name}.{os.getpid()}.tmp')",
            "marker_temp.write_text(f'{os.getpid()} {child.pid}', encoding='ascii')",
            "os.replace(marker_temp, marker)",
            "os.read(cleanup, 1)",
        )
    )
    worker_bootstrap_source = "\n".join(
        (
            "import os, sys",
            "gate = int(sys.argv[1])",
            "admitted = os.read(gate, 1)",
            "os.close(gate)",
            "if admitted != b'S': raise SystemExit(127)",
            "os.setsid()",
            "os.execvpe(sys.argv[2], sys.argv[2:], os.environ)",
        )
    )
    wrapper_source = "\n".join(
        (
            "import importlib.util, json, os, re, signal, subprocess, sys, time",
            "from pathlib import Path",
            "path, lifetime, marker, reaped, control, wait_entered, wait_snapshot, allow_reap, post_reap, group_kills, site, moment, control_name, cleanup, worker_id_path = sys.argv[1:16]",
            "lifetime, cleanup = map(int, (lifetime, cleanup))",
            "spec = importlib.util.spec_from_file_location('round14_spawn_supervisor', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "gate_reader, gate_writer = os.pipe()",
            f"worker = subprocess.Popen([sys.executable, '-I', '-c', {worker_bootstrap_source!r}, str(gate_reader), *sys.argv[16:]], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, pass_fds=(gate_reader, cleanup))",
            "os.close(gate_reader)",
            "Path(worker_id_path).write_text(str(worker.pid), encoding='ascii')",
            "if os.write(gate_writer, b'S') != 1: raise OSError('round14 worker gate release was incomplete')",
            "os.close(gate_writer)",
            "marker_deadline = time.monotonic() + 2.0",
            "while not Path(marker).exists() and time.monotonic() < marker_deadline: time.sleep(0.005)",
            "if not Path(marker).exists(): raise TimeoutError('round14 worker group readiness was not published')",
            "real_kill = os.kill",
            "real_killpg = os.killpg",
            "real_waitpid = os.waitpid",
            "real_sleep = time.sleep",
            "real_run = subprocess.run",
            "control_type = {'OSError': OSError, 'KeyboardInterrupt': KeyboardInterrupt, 'SystemExit': SystemExit}[control_name]",
            "injected = control_type(f'round14 {site} {moment} cleanup control')",
            "kill_faulted = False",
            "wait_faulted = False",
            "group_kill_faulted = False",
            "scan_invoke_faulted = False",
            "scan_parse_faulted = False",
            "sleep_faulted = False",
            "group_kill_attempts = 0",
            "scan_invoke_attempts = 0",
            "scan_parse_attempts = 0",
            "sleep_attempts = 0",
            "wait_attempts = 0",
            "leader_kill_attempts = 0",
            "group_signal_delivered = False",
            "actions = []",
            "scan_receipts = []",
            "scan_identifier = 0",
            "scan_command = ['/bin/ps', '-axo', 'pid=,pgid=,stat=']",
            "scan_environment = {**os.environ, 'LC_ALL': 'C', 'LANG': 'C'}",
            "def strict_parse(stdout, stderr):",
            "    if stderr or not stdout or len(stdout) > 1048576 or not stdout.endswith(b'\\n'): raise ValueError('invalid complete process-table stream')",
            "    if b'\\r' in stdout: raise ValueError('noncanonical process-table line ending')",
            "    row_pattern = re.compile(rb' *([1-9][0-9]*) +([1-9][0-9]*) +([^\\x00-\\x20\\x7f-\\xff]+)( *)')",
            "    rows = []",
            "    seen = set()",
            "    for raw_line in stdout.splitlines():",
            "        matched = row_pattern.fullmatch(raw_line)",
            "        if matched is None: raise ValueError('noncanonical process-table row')",
            "        raw_pid, raw_pgid, raw_state, raw_padding = matched.groups()",
            "        pid, pgid = int(raw_pid), int(raw_pgid)",
            "        state = raw_state.decode('ascii')",
            "        if sys.platform == 'darwin':",
            "            valid_state = re.fullmatch(r'[?IRSTUZ](?:>|W)?(?:<|N)?(?:A|S)?X?E?V?L?s?\\+?', state) is not None",
            "            expected_padding = max(0, 4 - len(state))",
            "        elif sys.platform.startswith('linux'):",
            f"            valid_state = re.fullmatch({_LINUX_PROCESS_STATE_PATTERN!r}, state) is not None",
            "            expected_padding = 0",
            "        else: raise ValueError('unsupported process-table platform')",
            "        if not valid_state: raise ValueError('noncanonical process-table state')",
            "        if len(raw_padding) != expected_padding: raise ValueError('noncanonical process-table state padding')",
            "        if pid in seen: raise ValueError('duplicate process-table identifier')",
            "        seen.add(pid)",
            "        rows.append((pid, pgid, state))",
            "    return tuple(sorted(rows))",
            "def kill(process_id, signal_number):",
            "    global kill_faulted, leader_kill_attempts",
            "    if process_id == worker.pid:",
            "        leader_kill_attempts += 1",
            "        actions.append('leader-kill')",
            "        if site == 'kill' and not kill_faulted:",
            "            kill_faulted = True",
            "            if moment == 'post-action': real_kill(process_id, signal_number)",
            "            raise injected",
            "    return real_kill(process_id, signal_number)",
            "def killpg(process_group_id, signal_number):",
            "    global group_kill_faulted, group_kill_attempts, group_signal_delivered",
            "    if process_group_id == worker.pid and Path(reaped).exists():",
            "        Path(post_reap).write_text(f'{process_group_id} {signal_number}', encoding='ascii')",
            "    if process_group_id == worker.pid and signal_number == signal.SIGKILL:",
            "        group_kill_attempts += 1",
            "        actions.append('group-kill')",
            "        Path(group_kills).write_text(str(group_kill_attempts), encoding='ascii')",
            "        if site == 'killpg' and not group_kill_faulted:",
            "            group_kill_faulted = True",
            "            if moment == 'post-action':",
            "                real_killpg(process_group_id, signal_number)",
            "                group_signal_delivered = True",
            "            raise injected",
            "        if site != 'sleep':",
            "            real_killpg(process_group_id, signal_number)",
            "            group_signal_delivered = True",
            "        return None",
            "    return real_killpg(process_group_id, signal_number)",
            "def invoke_process_table_scan():",
            "    global scan_identifier, scan_invoke_attempts, scan_invoke_faulted",
            "    if Path(reaped).exists(): Path(post_reap).write_text('scan-invoke', encoding='ascii')",
            "    scan_invoke_attempts += 1",
            "    scan_identifier += 1",
            "    actions.append('scan-invoke')",
            "    receipt = {'id': scan_identifier, 'stdout_hex': '', 'stderr_hex': '', 'invoke_returned': False, 'parse_returned': False}",
            "    scan_receipts.append(receipt)",
            "    if site == 'scan-invoke' and not scan_invoke_faulted and moment == 'pre-action':",
            "        scan_invoke_faulted = True",
            "        raise injected",
            "    completed = real_run(scan_command, stdin=subprocess.DEVNULL, capture_output=True, check=True, text=False, timeout=1.0, env=scan_environment)",
            "    receipt['stdout_hex'] = completed.stdout.hex()",
            "    receipt['stderr_hex'] = completed.stderr.hex()",
            "    if site == 'scan-invoke' and not scan_invoke_faulted:",
            "        scan_invoke_faulted = True",
            "        raise injected",
            "    receipt['invoke_returned'] = True",
            "    return completed.stdout, completed.stderr",
            "def parse_process_table(stdout, stderr):",
            "    global scan_parse_attempts, scan_parse_faulted",
            "    if Path(reaped).exists(): Path(post_reap).write_text('scan-parse', encoding='ascii')",
            "    scan_parse_attempts += 1",
            "    actions.append('scan-parse')",
            "    receipt = next(item for item in reversed(scan_receipts) if item['invoke_returned'] and not item['parse_returned'])",
            "    if site == 'scan-parse' and not scan_parse_faulted and moment == 'pre-action':",
            "        scan_parse_faulted = True",
            "        raise injected",
            "    rows = strict_parse(stdout, stderr)",
            "    receipt['rows'] = rows",
            "    if site == 'scan-parse' and not scan_parse_faulted:",
            "        scan_parse_faulted = True",
            "        raise injected",
            "    receipt['parse_returned'] = True",
            "    return rows",
            "def sleep(seconds):",
            "    global group_signal_delivered, sleep_attempts, sleep_faulted",
            "    if Path(reaped).exists():",
            "        Path(post_reap).write_text('sleep', encoding='ascii')",
            "        raise AssertionError('sleep was attempted after exact leader reap')",
            "    sleep_attempts += 1",
            "    actions.append('sleep')",
            "    if site == 'sleep' and not sleep_faulted:",
            "        sleep_faulted = True",
            "        if moment == 'post-action': real_sleep(seconds)",
            "        raise injected",
            "    if site == 'sleep' and not group_signal_delivered:",
            "        real_killpg(worker.pid, signal.SIGKILL)",
            "        group_signal_delivered = True",
            "    return real_sleep(seconds)",
            "def waitpid(process_id, options):",
            "    global wait_attempts, wait_faulted",
            "    if process_id == worker.pid:",
            "        wait_attempts += 1",
            "        actions.append('waitpid')",
            "        completed = real_run(scan_command, stdin=subprocess.DEVNULL, capture_output=True, check=True, text=False, timeout=1.0, env=scan_environment)",
            "        barrier_rows = strict_parse(completed.stdout, completed.stderr)",
            "        evidence = {'barrier_stdout_hex': completed.stdout.hex(), 'barrier_stderr_hex': completed.stderr.hex(), 'barrier_rows': barrier_rows, 'scan_receipts': scan_receipts}",
            "        snapshot = Path(wait_snapshot)",
            "        snapshot_temp = snapshot.with_name(f'{snapshot.name}.{os.getpid()}.tmp')",
            "        snapshot_temp.write_text(json.dumps(evidence, sort_keys=True), encoding='ascii')",
            "        os.replace(snapshot_temp, snapshot)",
            "        wait_path = Path(wait_entered)",
            "        wait_temp = wait_path.with_name(f'{wait_path.name}.{os.getpid()}.tmp')",
            "        wait_temp.write_text(str(worker.pid), encoding='ascii')",
            "        os.replace(wait_temp, wait_path)",
            "        release_deadline = time.monotonic() + 2.0",
            "        while not Path(allow_reap).exists() and time.monotonic() < release_deadline: real_sleep(0.005)",
            "        if not Path(allow_reap).exists(): raise TimeoutError('round14 exact reap was not released')",
            "    if site == 'waitpid' and process_id == worker.pid and not wait_faulted:",
            "        wait_faulted = True",
            "        if moment == 'pre-action': raise injected",
            "        observed_id, wait_status = real_waitpid(process_id, options)",
            "        if observed_id == worker.pid: Path(reaped).write_text(str(worker.pid), encoding='ascii')",
            "        raise injected",
            "    observed_id, wait_status = real_waitpid(process_id, options)",
            "    if process_id == worker.pid and observed_id == worker.pid: Path(reaped).write_text(str(worker.pid), encoding='ascii')",
            "    return observed_id, wait_status",
            "module.os.kill = kill",
            "module.os.killpg = killpg",
            "module.os.waitpid = waitpid",
            "module.time = type('Round14Time', (), {'sleep': staticmethod(sleep)})()",
            "module._invoke_process_table_scan = invoke_process_table_scan",
            "module._parse_process_table = parse_process_table",
            "try:",
            "    module._terminate_and_wait_for_worker(worker.pid)",
            "except BaseException as exc:",
            "    module.os.kill = real_kill",
            "    module.os.killpg = real_killpg",
            "    module.os.waitpid = real_waitpid",
            "    module.time = time",
            "    boundary = {'same_object': exc is injected, 'type': type(exc).__name__, 'message': str(exc), 'reaped': Path(reaped).exists() and Path(reaped).read_text(encoding='ascii') == str(worker.pid), 'leader_kill_attempts': leader_kill_attempts, 'group_kill_attempts': group_kill_attempts, 'scan_invoke_attempts': scan_invoke_attempts, 'scan_parse_attempts': scan_parse_attempts, 'sleep_attempts': sleep_attempts, 'wait_attempts': wait_attempts, 'actions': actions, 'group_signal_delivered': group_signal_delivered, 'post_reap_group_operation': Path(post_reap).exists()}",
            "    Path(control).write_text(json.dumps(boundary, sort_keys=True), encoding='ascii')",
            "else:",
            "    module.os.kill = real_kill",
            "    module.os.killpg = real_killpg",
            "    module.os.waitpid = real_waitpid",
            "    module.time = time",
            "    Path(control).write_text(json.dumps({'same_object': False, 'type': 'NO_CONTROL', 'message': '', 'reaped': False, 'leader_kill_attempts': leader_kill_attempts, 'group_kill_attempts': group_kill_attempts, 'scan_invoke_attempts': scan_invoke_attempts, 'scan_parse_attempts': scan_parse_attempts, 'sleep_attempts': sleep_attempts, 'wait_attempts': wait_attempts, 'actions': actions, 'group_signal_delivered': group_signal_delivered, 'post_reap_group_operation': Path(post_reap).exists()}, sort_keys=True), encoding='ascii')",
            "os.close(lifetime)",
        )
    )
    controller = subprocess.Popen(
        [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            wrapper_source,
            os.fspath(Path(spawn_supervisor.__file__).resolve()),
            str(lifetime_writer),
            os.fspath(marker),
            os.fspath(reaped_receipt),
            os.fspath(control_receipt),
            os.fspath(wait_entered),
            os.fspath(wait_group_snapshot),
            os.fspath(allow_reap),
            os.fspath(post_reap_group_operation),
            os.fspath(group_kill_attempts_receipt),
            fault_site,
            fault_moment,
            control_type_name,
            str(worker_cleanup_reader),
            os.fspath(worker_id_receipt),
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            worker_source,
            os.fspath(marker),
            str(worker_cleanup_reader),
            os.fspath(descendant_ready),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
        pass_fds=(lifetime_writer, worker_cleanup_reader),
    )
    for descriptor in (lifetime_writer, worker_cleanup_reader):
        os.close(descriptor)
    worker_id = 0
    descendant_id = 0
    reaped_before_lifetime_eof = False
    group_at_lifetime_eof: tuple[tuple[int, str], ...] | None = None
    control_after_cleanup: dict[str, object] | None = None
    control_hidden_before_release = False
    lifetime_open_before_release = False
    group_before_reap: tuple[tuple[int, str], ...] | None = None
    production_scan_receipts: list[dict[str, object]] | None = None
    forced_test_cleanup = False
    owned_test_descriptors = {lifetime_reader}
    try:
        _wait_for_path(marker)
        worker_id, descendant_id = (
            int(value) for value in marker.read_text(encoding="ascii").split()
        )
        _wait_for_path(wait_entered)
        assert wait_entered.read_text(encoding="ascii") == str(worker_id)
        wait_evidence = json.loads(wait_group_snapshot.read_text(encoding="ascii"))
        assert isinstance(wait_evidence, dict)
        barrier_rows = _strict_process_table_rows(
            bytes.fromhex(str(wait_evidence["barrier_stdout_hex"])),
            bytes.fromhex(str(wait_evidence["barrier_stderr_hex"])),
        )
        assert [list(row) for row in barrier_rows] == wait_evidence["barrier_rows"]
        group_before_reap = tuple(
            (process_id, state)
            for process_id, process_group_id, state in barrier_rows
            if process_group_id == worker_id
        )
        assert len(group_before_reap) == 1
        assert group_before_reap[0][0] == worker_id
        assert group_before_reap[0][1].startswith("Z")
        raw_scan_receipts = wait_evidence["scan_receipts"]
        assert isinstance(raw_scan_receipts, list)
        assert all(isinstance(receipt, dict) for receipt in raw_scan_receipts)
        production_scan_receipts = raw_scan_receipts
        assert len(production_scan_receipts) >= 2
        final_scan_receipts = production_scan_receipts[-2:]
        assert final_scan_receipts[0]["parse_returned"] is True
        assert final_scan_receipts[1]["parse_returned"] is True
        assert int(final_scan_receipts[1]["id"]) == int(final_scan_receipts[0]["id"]) + 1
        for receipt in final_scan_receipts:
            rows = _strict_process_table_rows(
                bytes.fromhex(str(receipt["stdout_hex"])),
                bytes.fromhex(str(receipt["stderr_hex"])),
            )
            assert [list(row) for row in rows] == receipt["rows"]
            assert _snapshot_proves_group_extinction_for_test(rows, worker_id)
        control_hidden_before_release = not control_receipt.exists()
        lifetime_open_before_release = not _endpoint_is_readable(lifetime_reader)
        allow_reap.write_text("release", encoding="ascii")
        _read_exact_eof(lifetime_reader)
        reaped_before_lifetime_eof = (
            reaped_receipt.read_text(encoding="ascii") == str(worker_id)
            if reaped_receipt.exists()
            else False
        )
        group_at_lifetime_eof = _process_group_snapshot(worker_id)
        control_after_cleanup = json.loads(control_receipt.read_text(encoding="ascii"))
        assert controller.wait(timeout=_WAIT_SECONDS) == 0
    finally:
        allow_reap.write_text("release", encoding="ascii")
        if controller.returncode is None:
            with suppress(subprocess.TimeoutExpired):
                controller.wait(timeout=_WAIT_SECONDS)
        if worker_id_receipt.exists() and not worker_id:
            worker_id = int(worker_id_receipt.read_text(encoding="ascii"))
        if marker.exists():
            marker_ids = tuple(int(value) for value in marker.read_text(encoding="ascii").split())
            if len(marker_ids) == 2:
                if not worker_id:
                    worker_id = marker_ids[0]
                if not descendant_id:
                    descendant_id = marker_ids[1]
        if controller.returncode is None or (worker_id > 0 and _process_group_snapshot(worker_id)):
            forced_test_cleanup = True
        _complete_test_resource_teardown(
            controller,
            cooperative_cleanup_writer=worker_cleanup_writer,
            process_group_ids={worker_id, controller.pid},
            owned_descriptors=owned_test_descriptors,
        )

    assert worker_id > 0
    assert descendant_id > 0
    assert descendant_ready.read_text(encoding="ascii") == str(descendant_id)
    assert group_before_reap is not None
    assert production_scan_receipts is not None
    assert control_hidden_before_release
    assert lifetime_open_before_release
    assert reaped_before_lifetime_eof
    assert group_at_lifetime_eof == ()
    assert _process_not_running(worker_id)
    assert _process_not_running(descendant_id)
    assert control_after_cleanup is not None
    assert control_after_cleanup["same_object"] is True
    assert control_after_cleanup["type"] == control_type_name
    assert control_after_cleanup["message"] == (
        f"round14 {fault_site} {fault_moment} cleanup control"
    )
    assert control_after_cleanup["reaped"] is True
    assert control_after_cleanup["group_signal_delivered"] is True
    assert control_after_cleanup["post_reap_group_operation"] is False

    actions = control_after_cleanup["actions"]
    assert isinstance(actions, list)
    assert all(isinstance(action, str) for action in actions)
    leader_kill_attempts = 2 if fault_site == "kill" else 1
    group_kill_attempts = 2 if fault_site == "killpg" else 1
    wait_attempts = 2 if fault_site == "waitpid" else 1
    assert control_after_cleanup["leader_kill_attempts"] == leader_kill_attempts
    assert control_after_cleanup["group_kill_attempts"] == group_kill_attempts
    assert control_after_cleanup["wait_attempts"] == wait_attempts
    assert actions.count("leader-kill") == leader_kill_attempts
    assert actions.count("group-kill") == group_kill_attempts
    assert actions.count("scan-invoke") == control_after_cleanup["scan_invoke_attempts"]
    assert actions.count("scan-parse") == control_after_cleanup["scan_parse_attempts"]
    assert actions.count("sleep") == control_after_cleanup["sleep_attempts"]
    assert actions.count("waitpid") == wait_attempts
    first_group_kill = actions.index("group-kill")
    first_wait = actions.index("waitpid")
    assert actions[:first_group_kill] == ["leader-kill"] * leader_kill_attempts
    assert (
        actions[first_group_kill : first_group_kill + group_kill_attempts]
        == ["group-kill"] * group_kill_attempts
    )
    assert "leader-kill" not in actions[first_group_kill:]
    assert "group-kill" not in actions[first_group_kill + group_kill_attempts :]
    assert actions[first_wait:] == ["waitpid"] * wait_attempts
    if fault_site == "scan-invoke":
        assert actions[first_group_kill + group_kill_attempts :][:7] == [
            "scan-invoke",
            "sleep",
            "scan-invoke",
            "scan-parse",
            "sleep",
            "scan-invoke",
            "scan-parse",
        ]
    if fault_site == "scan-parse":
        assert actions[first_group_kill + group_kill_attempts :][:8] == [
            "scan-invoke",
            "scan-parse",
            "sleep",
            "scan-invoke",
            "scan-parse",
            "sleep",
            "scan-invoke",
            "scan-parse",
        ]
    if fault_site == "sleep":
        assert actions[first_group_kill + group_kill_attempts :][:6] == [
            "scan-invoke",
            "scan-parse",
            "sleep",
            "scan-invoke",
            "scan-parse",
            "sleep",
        ]
    assert not forced_test_cleanup


def test_round15_exact_reap_retires_numeric_process_group_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reaped leader cannot authorize even a read-only scan of a reused PGID."""

    worker_id = 73
    reaped = False
    actions: list[tuple[str, int, int]] = []
    post_reap_group_operations: list[tuple[str, int, int]] = []

    def kill(process_id: int, signal_number: int) -> None:
        assert not reaped
        actions.append(("kill", process_id, signal_number))

    def killpg(process_group_id: int, signal_number: int) -> None:
        if reaped:
            post_reap_group_operations.append(("killpg", process_group_id, signal_number))
            raise ProcessLookupError(process_group_id)
        actions.append(("killpg", process_group_id, signal_number))

    def invoke_process_table_scan() -> tuple[bytes, bytes]:
        if reaped:
            post_reap_group_operations.append(("scan-invoke", worker_id, 0))
            raise AssertionError("numeric PGID was scanned after exact leader reap")
        actions.append(("scan-invoke", worker_id, 0))
        return _canonical_process_table_row_for_test(
            worker_id,
            worker_id,
            "Z",
        ).encode("ascii"), b""

    def parse_process_table(
        stdout: bytes,
        stderr: bytes,
    ) -> tuple[tuple[int, int, str], ...]:
        if reaped:
            post_reap_group_operations.append(("scan-parse", worker_id, 0))
            raise AssertionError("numeric PGID was parsed after exact leader reap")
        actions.append(("scan-parse", worker_id, 0))
        return _strict_process_table_rows(stdout, stderr)

    def waitpid(process_id: int, options: int) -> tuple[int, int]:
        nonlocal reaped
        assert not reaped
        assert options == 0
        actions.append(("waitpid", process_id, options))
        reaped = True
        return process_id, 0

    monkeypatch.setattr(spawn_supervisor.os, "kill", kill)
    monkeypatch.setattr(spawn_supervisor.os, "killpg", killpg)
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)
    monkeypatch.setattr(
        spawn_supervisor,
        "_invoke_process_table_scan",
        invoke_process_table_scan,
        raising=False,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_process_table",
        parse_process_table,
        raising=False,
    )

    spawn_supervisor._terminate_and_wait_for_worker(worker_id)

    assert actions == [
        ("kill", worker_id, signal.SIGKILL),
        ("killpg", worker_id, signal.SIGKILL),
        ("scan-invoke", worker_id, 0),
        ("scan-parse", worker_id, 0),
        ("scan-invoke", worker_id, 0),
        ("scan-parse", worker_id, 0),
        ("waitpid", worker_id, 0),
    ]
    assert post_reap_group_operations == []


def test_round16_process_table_invocation_is_exact_bounded_and_byte_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def run(command: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"73 73 Z   \n81 90 S+  \n",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", run)

    assert spawn_supervisor._invoke_process_table_scan() == (  # type: ignore[attr-defined]
        b"73 73 Z   \n81 90 S+  \n",
        b"",
    )
    assert calls == [
        (
            list(_PROCESS_TABLE_COMMAND),
            {
                "capture_output": True,
                "check": True,
                "env": {**os.environ, "LANG": "C", "LC_ALL": "C"},
                "stdin": subprocess.DEVNULL,
                "text": False,
                "timeout": 1.0,
            },
        )
    ]


def test_round17_bootstrap_readiness_requires_one_byte_then_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = iter((b"R", b""))
    reads: list[tuple[int, int]] = []

    def read(descriptor: int, size: int) -> bytes:
        reads.append((descriptor, size))
        return next(chunks)

    monkeypatch.setattr(spawn_supervisor.os, "read", read)

    spawn_supervisor._read_bootstrap_ready(73)  # type: ignore[attr-defined]

    assert reads == [(73, 1), (73, 1)]


@pytest.mark.parametrize(
    "chunks",
    [
        [b"", b""],
        [b"S", b""],
        [b"R", b"x"],
    ],
)
def test_round17_bootstrap_readiness_rejects_missing_wrong_or_trailing_data(
    chunks: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_iterator = iter(chunks)
    monkeypatch.setattr(
        spawn_supervisor.os,
        "read",
        lambda _descriptor, _size: next(chunks_iterator),
    )

    with pytest.raises(ValueError, match=r"^invalid worker bootstrap readiness$"):
        spawn_supervisor._read_bootstrap_ready(73)  # type: ignore[attr-defined]


def test_round17_bootstrap_readiness_error_precedes_direct_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-admission readiness error retains identity through direct cleanup."""

    worker_id = 73
    readiness_error = ValueError("round17 exact readiness failure")
    cleanup_error = OSError("round17 later direct cleanup failure")
    cleanup_calls: list[int] = []

    def terminate_direct(process_id: int) -> None:
        cleanup_calls.append(process_id)
        raise cleanup_error

    monkeypatch.setattr(
        spawn_supervisor,
        "_terminate_and_wait_for_unadmitted_worker",
        terminate_direct,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_terminate_and_wait_for_worker",
        lambda _process_id: pytest.fail("group cleanup before START was forbidden"),
    )

    with pytest.raises(ValueError) as raised:
        spawn_supervisor._terminate_owned_worker_preserving_error(  # type: ignore[attr-defined]
            worker_id,
            bootstrap_group_proven=False,
            admission_may_have_escaped=False,
            initiating_error=readiness_error,
        )

    assert raised.value is readiness_error
    assert cleanup_calls == [worker_id]


def test_round17_start_send_error_precedes_group_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once START may have escaped, group cleanup cannot replace its error."""

    worker_id = 73
    send_error = OSError("round17 exact START send failure")
    cleanup_error = PermissionError("round17 later group cleanup failure")
    cleanup_calls: list[int] = []

    def terminate_group(process_id: int) -> None:
        cleanup_calls.append(process_id)
        raise cleanup_error

    monkeypatch.setattr(
        spawn_supervisor,
        "_terminate_and_wait_for_worker",
        terminate_group,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_terminate_and_wait_for_unadmitted_worker",
        lambda _process_id: pytest.fail("direct cleanup after START was forbidden"),
    )

    with pytest.raises(OSError) as raised:
        spawn_supervisor._terminate_owned_worker_preserving_error(  # type: ignore[attr-defined]
            worker_id,
            bootstrap_group_proven=True,
            admission_may_have_escaped=True,
            initiating_error=send_error,
        )

    assert raised.value is send_error
    assert cleanup_calls == [worker_id]


@pytest.mark.parametrize(
    "injected",
    [
        OSError("round16 scanner invocation failed"),
        subprocess.TimeoutExpired(list(_PROCESS_TABLE_COMMAND), 1.0),
        KeyboardInterrupt("round16 scanner interrupted"),
        SystemExit("round16 scanner exited"),
    ],
)
def test_round16_process_table_invocation_preserves_exact_failure(
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise injected

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(type(injected)) as raised:
        spawn_supervisor._invoke_process_table_scan()  # type: ignore[attr-defined]

    assert raised.value is injected


@pytest.mark.parametrize(
    ("platform", "stdout", "expected"),
    [
        (
            "darwin",
            b"82 91 ?s  \n81 90 SNXEVLs+\n73 73 Zs  \n",
            ((73, 73, "Zs"), (81, 90, "SNXEVLs+"), (82, 91, "?s")),
        ),
        (
            "linux",
            b"81 90 S<Lsl+\n73 73 Z\n",
            ((73, 73, "Z"), (81, 90, "S<Lsl+")),
        ),
    ],
)
def test_round16_process_table_parser_returns_platform_canonical_full_table(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    stdout: bytes,
    expected: tuple[tuple[int, int, str], ...],
) -> None:
    monkeypatch.setattr(spawn_supervisor.sys, "platform", platform)

    assert (
        spawn_supervisor._parse_process_table(  # type: ignore[attr-defined]
            stdout,
            b"",
        )
        == expected
    )


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
def test_round17_darwin_process_table_accepts_each_documented_modifier_in_order(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    """A legitimate unrelated Darwin row cannot stall extinction proof forever."""

    monkeypatch.setattr(spawn_supervisor.sys, "platform", "darwin")
    row = f"73 73 {state:<4}\n".encode("ascii")

    assert spawn_supervisor._parse_process_table(row, b"") == (  # type: ignore[attr-defined]
        (73, 73, state),
    )


@pytest.mark.parametrize("state", ["K", "P", "x", "KNLsl+"])
def test_round17_linux_process_table_accepts_each_documented_primary_state(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    """A documented procps state cannot stall extinction proof forever."""

    monkeypatch.setattr(spawn_supervisor.sys, "platform", "linux")
    row = f"73 73 {state}\n".encode("ascii")

    assert spawn_supervisor._parse_process_table(row, b"") == (  # type: ignore[attr-defined]
        (73, 73, state),
    )
    assert _strict_process_table_rows(row, b"", platform="linux") == ((73, 73, state),)


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (b"", b""),
        (b"73 73 Z   ", b""),
        (b"73 73\n", b""),
        (b"73 73 Z extra\n", b""),
        (b"073 73 Z   \n", b""),
        (b"73 073 Z   \n", b""),
        (b"0 73 Z   \n", b""),
        (b"73 0 Z   \n", b""),
        (b"73 73 z   \n", b""),
        (b"73 73 Z!  \n", b""),
        (b"73 73 Z\x00  \n", b""),
        (b"73 73 Zss \n", b""),
        (b"73 73 ZsL \n", b""),
        (b"73 73 ZLX \n", b""),
        (b"73 73 RNW \n", b""),
        (b"73 73 R>W \n", b""),
        (b"73 73 RAS \n", b""),
        (b"73 73 RAA \n", b""),
        (b"73\t73 Z   \n", b""),
        (b"73 73 Z\n", b""),
        (b"73 73 Z \n", b""),
        (b"73 73 Z  \n", b""),
        (b"73 73 SNXEVLs+ \n", b""),
        (b"73 73 Z   \r\n", b""),
        (b"73 73 Z   \n73 73 Z   \n", b""),
        (b"73 73 Z   \n\n81 90 S   \n", b""),
        (b"73 73 Z   \n", b"unexpected stderr"),
        (f"73 73 {'Z' * 17}\n".encode("ascii"), b""),
        (b"1 1 S   \n" * 120_000, b""),
    ],
)
def test_round16_darwin_process_table_parser_rejects_malformed_partial_or_unbounded_output(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    stderr: bytes,
) -> None:
    monkeypatch.setattr(spawn_supervisor.sys, "platform", "darwin")

    with pytest.raises(ValueError):
        spawn_supervisor._parse_process_table(stdout, stderr)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "stdout",
    [
        b"73 73 Z   \n",
        b"73 73 SLL\n",
        b"73 73 SL<\n",
        b"73 73 S+l\n",
        b"73 73 SNLs+l\n",
        b"73 73 KN<\n",
        b"73 73 xlL\n",
        b"73 73 U\n",
    ],
)
def test_round16_linux_process_table_parser_rejects_padding_duplicate_or_bad_order(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
) -> None:
    monkeypatch.setattr(spawn_supervisor.sys, "platform", "linux")

    with pytest.raises(ValueError):
        spawn_supervisor._parse_process_table(stdout, b"")  # type: ignore[attr-defined]


def test_round16_process_table_parser_rejects_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spawn_supervisor.sys, "platform", "win32")

    with pytest.raises(
        ValueError,
        match=r"^process-table scanner platform was unsupported$",
    ):
        spawn_supervisor._parse_process_table(  # type: ignore[attr-defined]
            b"73 73 Z\n",
            b"",
        )


@pytest.mark.parametrize(
    "rows",
    [
        (),
        ((73, 74, "Z"),),
        ((73, 73, "S"),),
        ((73, 73, "Z"), (74, 73, "Z")),
        ((73, 73, "Z"), (74, 73, "S")),
        ((73, 73, "Z"), (73, 73, "Z")),
        ((74, 73, "Z"),),
    ],
)
def test_round16_partial_or_mismatched_snapshot_never_proves_group_extinction(
    rows: tuple[tuple[int, int, str], ...],
) -> None:
    assert not spawn_supervisor._snapshot_proves_group_extinction(  # type: ignore[attr-defined]
        rows,
        73,
    )


def test_round16_exact_zombie_leader_snapshot_proves_one_observation() -> None:
    assert spawn_supervisor._snapshot_proves_group_extinction(  # type: ignore[attr-defined]
        ((73, 73, "Zs+"), (81, 90, "S")),
        73,
    )


@pytest.mark.parametrize(
    "fault_site",
    ["kill", "killpg", "scan-invoke", "scan-parse", "sleep", "waitpid"],
)
@pytest.mark.parametrize("fault_moment", ["pre-action", "post-action"])
@pytest.mark.parametrize(
    "control_type",
    [OSError, ValueError, KeyboardInterrupt, SystemExit],
)
def test_round15_cleanup_control_retry_order_is_exact_before_reap(
    monkeypatch: pytest.MonkeyPatch,
    fault_site: str,
    fault_moment: str,
    control_type: type[BaseException],
) -> None:
    """Every ambiguous cleanup action is retried before the stable PGID is retired."""

    worker_id = 73
    injected = control_type(f"round15 {fault_site} {fault_moment} control")
    faulted = False
    reaped = False
    completed_scans = 0
    actions: list[str] = []
    post_reap_group_operations: list[tuple[str, int, int]] = []

    def inject_once(site: str) -> bool:
        nonlocal faulted
        if fault_site != site or faulted:
            return False
        faulted = True
        return True

    def kill(process_id: int, signal_number: int) -> None:
        assert process_id == worker_id
        assert signal_number == signal.SIGKILL
        assert not reaped
        actions.append("leader-kill")
        if inject_once("kill"):
            raise injected

    def killpg(process_group_id: int, signal_number: int) -> None:
        assert process_group_id == worker_id
        if reaped:
            post_reap_group_operations.append(("killpg", process_group_id, signal_number))
            raise ProcessLookupError(process_group_id)
        assert signal_number == signal.SIGKILL
        actions.append("group-kill")
        if inject_once("killpg"):
            raise injected

    def invoke_process_table_scan() -> tuple[bytes, bytes]:
        nonlocal completed_scans
        if reaped:
            post_reap_group_operations.append(("scan-invoke", worker_id, 0))
            raise AssertionError("numeric PGID was scanned after exact leader reap")
        actions.append("scan-invoke")
        if fault_site == "scan-invoke" and not faulted and fault_moment == "pre-action":
            assert inject_once("scan-invoke")
            raise injected
        completed_scans += 1
        if fault_site not in {"scan-invoke", "scan-parse"} and completed_scans == 1:
            stdout = (
                _canonical_process_table_row_for_test(worker_id, worker_id, "Z")
                + _canonical_process_table_row_for_test(
                    worker_id + 1,
                    worker_id,
                    "S",
                )
            ).encode("ascii")
        else:
            stdout = _canonical_process_table_row_for_test(
                worker_id,
                worker_id,
                "Z",
            ).encode("ascii")
        if inject_once("scan-invoke"):
            assert fault_moment == "post-action"
            raise injected
        return stdout, b""

    def parse_process_table(
        stdout: bytes,
        stderr: bytes,
    ) -> tuple[tuple[int, int, str], ...]:
        if reaped:
            post_reap_group_operations.append(("scan-parse", worker_id, 0))
            raise AssertionError("numeric PGID was parsed after exact leader reap")
        actions.append("scan-parse")
        if fault_site == "scan-parse" and not faulted and fault_moment == "pre-action":
            assert inject_once("scan-parse")
            raise injected
        rows = _strict_process_table_rows(stdout, stderr)
        if inject_once("scan-parse"):
            assert fault_moment == "post-action"
            raise injected
        return rows

    def sleep(_seconds: float) -> None:
        assert not reaped
        actions.append("sleep")
        if inject_once("sleep"):
            raise injected

    def waitpid(process_id: int, options: int) -> tuple[int, int]:
        nonlocal reaped
        assert process_id == worker_id
        assert options == 0
        actions.append("waitpid")
        if inject_once("waitpid"):
            if fault_moment == "post-action":
                reaped = True
            raise injected
        if reaped:
            raise ChildProcessError(process_id)
        reaped = True
        return process_id, 0

    monkeypatch.setattr(spawn_supervisor.os, "kill", kill)
    monkeypatch.setattr(spawn_supervisor.os, "killpg", killpg)
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)
    monkeypatch.setattr(spawn_supervisor.time, "sleep", sleep)
    monkeypatch.setattr(
        spawn_supervisor,
        "_invoke_process_table_scan",
        invoke_process_table_scan,
        raising=False,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_process_table",
        parse_process_table,
        raising=False,
    )

    with pytest.raises(control_type) as raised:
        spawn_supervisor._terminate_and_wait_for_worker(worker_id)

    leader_kills = 2 if fault_site == "kill" else 1
    group_kills = 2 if fault_site == "killpg" else 1
    waits = 2 if fault_site == "waitpid" else 1
    standard_proof = [
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "scan-parse",
    ]
    proof_actions = {
        "kill": standard_proof,
        "killpg": standard_proof,
        "scan-invoke": [
            "scan-invoke",
            "sleep",
            "scan-invoke",
            "scan-parse",
            "sleep",
            "scan-invoke",
            "scan-parse",
        ],
        "scan-parse": [
            "scan-invoke",
            "scan-parse",
            "sleep",
            "scan-invoke",
            "scan-parse",
            "sleep",
            "scan-invoke",
            "scan-parse",
        ],
        "sleep": standard_proof,
        "waitpid": standard_proof,
    }[fault_site]
    assert raised.value is injected
    assert faulted
    assert reaped
    assert actions == [
        *(["leader-kill"] * leader_kills),
        *(["group-kill"] * group_kills),
        *proof_actions,
        *(["waitpid"] * waits),
    ]
    assert post_reap_group_operations == []


def test_round16_unsafe_snapshot_resets_two_observation_extinction_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_id = 73
    safe_snapshot = _canonical_process_table_row_for_test(worker_id, worker_id, "Z")
    snapshots = chain(
        (
            _canonical_process_table_row_for_test(worker_id, worker_id, "Z"),
            _canonical_process_table_row_for_test(worker_id, worker_id, "Z")
            + _canonical_process_table_row_for_test(
                worker_id + 1,
                worker_id,
                "S",
            ),
            safe_snapshot,
        ),
        repeat(safe_snapshot),
    )
    actions: list[str] = []
    post_reap_group_operations: list[tuple[int, int]] = []
    reaped = False

    monkeypatch.setattr(
        spawn_supervisor.os,
        "kill",
        lambda _process_id, signal_number: actions.append(f"leader-kill:{signal_number}"),
    )

    def killpg(process_group_id: int, signal_number: int) -> None:
        if reaped:
            post_reap_group_operations.append((process_group_id, signal_number))
            raise ProcessLookupError(process_group_id)
        actions.append(f"group-kill:{signal_number}")

    monkeypatch.setattr(spawn_supervisor.os, "killpg", killpg)

    def invoke_process_table_scan() -> tuple[bytes, bytes]:
        assert not reaped
        actions.append("scan-invoke")
        return next(snapshots).encode("ascii"), b""

    def parse_process_table(
        stdout: bytes,
        stderr: bytes,
    ) -> tuple[tuple[int, int, str], ...]:
        assert not reaped
        actions.append("scan-parse")
        return _strict_process_table_rows(stdout, stderr)

    def waitpid(process_id: int, options: int) -> tuple[int, int]:
        nonlocal reaped
        assert process_id == worker_id
        assert options == 0
        actions.append("waitpid")
        reaped = True
        return process_id, 0

    monkeypatch.setattr(
        spawn_supervisor,
        "_invoke_process_table_scan",
        invoke_process_table_scan,
        raising=False,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_process_table",
        parse_process_table,
        raising=False,
    )
    monkeypatch.setattr(spawn_supervisor.time, "sleep", lambda _seconds: actions.append("sleep"))
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)

    spawn_supervisor._terminate_and_wait_for_worker(worker_id)

    assert reaped
    assert actions == [
        f"leader-kill:{signal.SIGKILL}",
        f"group-kill:{signal.SIGKILL}",
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "waitpid",
    ]
    assert post_reap_group_operations == []


def test_round16_scanner_timeout_is_retried_then_raised_by_identity_after_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_id = 73
    injected = subprocess.TimeoutExpired(list(_PROCESS_TABLE_COMMAND), 1.0)
    invoke_attempts = 0
    reaped = False
    actions: list[str] = []
    post_reap_group_operations: list[tuple[int, int]] = []

    monkeypatch.setattr(spawn_supervisor.os, "kill", lambda *_args: None)

    def killpg(process_group_id: int, signal_number: int) -> None:
        if reaped:
            post_reap_group_operations.append((process_group_id, signal_number))
            raise ProcessLookupError(process_group_id)

    monkeypatch.setattr(spawn_supervisor.os, "killpg", killpg)

    def invoke_process_table_scan() -> tuple[bytes, bytes]:
        nonlocal invoke_attempts
        assert not reaped
        invoke_attempts += 1
        actions.append("scan-invoke")
        if invoke_attempts == 1:
            raise injected
        return _canonical_process_table_row_for_test(
            worker_id,
            worker_id,
            "Z",
        ).encode("ascii"), b""

    def parse_process_table(
        stdout: bytes,
        stderr: bytes,
    ) -> tuple[tuple[int, int, str], ...]:
        assert not reaped
        actions.append("scan-parse")
        return _strict_process_table_rows(stdout, stderr)

    def waitpid(process_id: int, _options: int) -> tuple[int, int]:
        nonlocal reaped
        actions.append("waitpid")
        reaped = True
        return process_id, 0

    monkeypatch.setattr(
        spawn_supervisor,
        "_invoke_process_table_scan",
        invoke_process_table_scan,
        raising=False,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_process_table",
        parse_process_table,
        raising=False,
    )
    monkeypatch.setattr(spawn_supervisor.time, "sleep", lambda _seconds: actions.append("sleep"))
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        spawn_supervisor._terminate_and_wait_for_worker(worker_id)

    assert raised.value is injected
    assert reaped
    assert actions == [
        "scan-invoke",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "waitpid",
    ]
    assert post_reap_group_operations == []


def test_round16_earliest_scanner_error_survives_later_control_and_exact_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_id = 73
    first = OSError("round16 first scanner failure")
    later = SystemExit("round16 later parser control")
    invoke_attempts = 0
    parse_attempts = 0
    reaped = False
    post_reap_group_operations: list[tuple[int, int]] = []

    monkeypatch.setattr(spawn_supervisor.os, "kill", lambda *_args: None)

    def killpg(process_group_id: int, signal_number: int) -> None:
        if reaped:
            post_reap_group_operations.append((process_group_id, signal_number))
            raise ProcessLookupError(process_group_id)

    monkeypatch.setattr(spawn_supervisor.os, "killpg", killpg)

    def invoke_process_table_scan() -> tuple[bytes, bytes]:
        nonlocal invoke_attempts
        assert not reaped
        invoke_attempts += 1
        if invoke_attempts == 1:
            raise first
        return _canonical_process_table_row_for_test(
            worker_id,
            worker_id,
            "Z",
        ).encode("ascii"), b""

    def parse_process_table(
        stdout: bytes,
        stderr: bytes,
    ) -> tuple[tuple[int, int, str], ...]:
        nonlocal parse_attempts
        assert not reaped
        parse_attempts += 1
        rows = _strict_process_table_rows(stdout, stderr)
        if parse_attempts == 1:
            raise later
        return rows

    def waitpid(process_id: int, _options: int) -> tuple[int, int]:
        nonlocal reaped
        reaped = True
        return process_id, 0

    monkeypatch.setattr(
        spawn_supervisor,
        "_invoke_process_table_scan",
        invoke_process_table_scan,
        raising=False,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_process_table",
        parse_process_table,
        raising=False,
    )
    monkeypatch.setattr(spawn_supervisor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)

    with pytest.raises(OSError) as raised:
        spawn_supervisor._terminate_and_wait_for_worker(worker_id)

    assert raised.value is first
    assert invoke_attempts == 4
    assert parse_attempts == 3
    assert reaped
    assert post_reap_group_operations == []


def test_round16_scanner_error_after_safe_snapshot_resets_consecutive_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_id = 73
    injected = OSError("round16 scanner error after first safe snapshot")
    invoke_attempts = 0
    parse_attempts = 0
    reaped = False
    actions: list[str] = []
    post_reap_group_operations: list[tuple[int, int]] = []

    monkeypatch.setattr(spawn_supervisor.os, "kill", lambda *_args: None)

    def killpg(process_group_id: int, signal_number: int) -> None:
        if reaped:
            post_reap_group_operations.append((process_group_id, signal_number))
            raise ProcessLookupError(process_group_id)

    monkeypatch.setattr(spawn_supervisor.os, "killpg", killpg)

    def invoke_process_table_scan() -> tuple[bytes, bytes]:
        nonlocal invoke_attempts
        assert not reaped
        invoke_attempts += 1
        actions.append("scan-invoke")
        if invoke_attempts == 2:
            raise injected
        return _canonical_process_table_row_for_test(
            worker_id,
            worker_id,
            "Z",
        ).encode("ascii"), b""

    def parse_process_table(
        stdout: bytes,
        stderr: bytes,
    ) -> tuple[tuple[int, int, str], ...]:
        nonlocal parse_attempts
        assert not reaped
        parse_attempts += 1
        actions.append("scan-parse")
        return _strict_process_table_rows(stdout, stderr)

    def waitpid(process_id: int, _options: int) -> tuple[int, int]:
        nonlocal reaped
        actions.append("waitpid")
        reaped = True
        return process_id, 0

    monkeypatch.setattr(
        spawn_supervisor,
        "_invoke_process_table_scan",
        invoke_process_table_scan,
        raising=False,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_process_table",
        parse_process_table,
        raising=False,
    )
    monkeypatch.setattr(spawn_supervisor.time, "sleep", lambda _seconds: actions.append("sleep"))
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)

    with pytest.raises(OSError) as raised:
        spawn_supervisor._terminate_and_wait_for_worker(worker_id)

    assert raised.value is injected
    assert invoke_attempts == 4
    assert parse_attempts == 3
    assert reaped
    assert actions == [
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "waitpid",
    ]
    assert post_reap_group_operations == []


@pytest.mark.parametrize("fault_site", ["leader", "group"])
def test_round16_repeated_signal_errors_defer_to_extinction_proof(
    monkeypatch: pytest.MonkeyPatch,
    fault_site: str,
) -> None:
    worker_id = 73
    first = PermissionError(f"round16 first {fault_site} signal failure")
    later = PermissionError(f"round16 repeated {fault_site} signal failure")
    leader_kill_attempts = 0
    group_kill_attempts = 0
    reaped = False
    actions: list[str] = []

    def kill(_process_id: int, _signal_number: int) -> None:
        nonlocal leader_kill_attempts
        leader_kill_attempts += 1
        actions.append("leader-kill")
        if fault_site == "leader" and leader_kill_attempts == 1:
            raise first
        if fault_site == "leader" and leader_kill_attempts == 2:
            raise later

    def killpg(_process_group_id: int, _signal_number: int) -> None:
        nonlocal group_kill_attempts
        group_kill_attempts += 1
        actions.append("group-kill")
        if fault_site == "group" and group_kill_attempts == 1:
            raise first
        if fault_site == "group" and group_kill_attempts == 2:
            raise later

    def invoke_process_table_scan() -> tuple[bytes, bytes]:
        assert not reaped
        actions.append("scan-invoke")
        return _canonical_process_table_row_for_test(
            worker_id,
            worker_id,
            "Z",
        ).encode("ascii"), b""

    def parse_process_table(
        stdout: bytes,
        stderr: bytes,
    ) -> tuple[tuple[int, int, str], ...]:
        assert not reaped
        actions.append("scan-parse")
        return _strict_process_table_rows(stdout, stderr)

    def waitpid(process_id: int, options: int) -> tuple[int, int]:
        nonlocal reaped
        assert process_id == worker_id
        assert options == 0
        actions.append("waitpid")
        reaped = True
        return process_id, 0

    monkeypatch.setattr(spawn_supervisor.os, "kill", kill)
    monkeypatch.setattr(spawn_supervisor.os, "killpg", killpg)
    monkeypatch.setattr(
        spawn_supervisor,
        "_invoke_process_table_scan",
        invoke_process_table_scan,
    )
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_process_table",
        parse_process_table,
    )
    monkeypatch.setattr(
        spawn_supervisor.time,
        "sleep",
        lambda _seconds: actions.append("sleep"),
    )
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)

    with pytest.raises(PermissionError) as raised:
        spawn_supervisor._terminate_and_wait_for_worker(worker_id)

    assert raised.value is first
    assert reaped
    assert actions == [
        *(["leader-kill"] * (2 if fault_site == "leader" else 1)),
        *(["group-kill"] * (2 if fault_site == "group" else 1)),
        "scan-invoke",
        "scan-parse",
        "sleep",
        "scan-invoke",
        "scan-parse",
        "waitpid",
    ]


def test_round17_unadmitted_cleanup_retries_until_exact_nonblocking_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry count cannot substitute for exact direct-child reap evidence."""

    worker_id = 73
    first = PermissionError("round17 first direct-child signal failure")
    actions: list[tuple[str, int]] = []
    kill_attempts = 0
    wait_results = iter(((0, 0), (0, 0), (0, 0), (worker_id, 9)))
    expected_wait_options = iter((os.WNOHANG, os.WNOHANG, os.WNOHANG, 0))

    def kill(process_id: int, signal_number: int) -> None:
        nonlocal kill_attempts
        assert (process_id, signal_number) == (worker_id, signal.SIGKILL)
        kill_attempts += 1
        actions.append(("kill", kill_attempts))
        if kill_attempts == 1:
            raise first
        if kill_attempts <= 3:
            raise PermissionError(f"round17 later direct-child signal failure {kill_attempts}")

    def waitpid(process_id: int, options: int) -> tuple[int, int]:
        assert process_id == worker_id
        assert options == next(expected_wait_options)
        actions.append(("waitpid", options))
        return next(wait_results)

    monkeypatch.setattr(spawn_supervisor.os, "kill", kill)
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)
    monkeypatch.setattr(
        spawn_supervisor.time,
        "sleep",
        lambda _seconds: actions.append(("sleep", 0)),
    )

    with pytest.raises(PermissionError) as raised:
        spawn_supervisor._terminate_and_wait_for_unadmitted_worker(worker_id)

    assert raised.value is first
    assert actions == [
        ("kill", 1),
        ("waitpid", os.WNOHANG),
        ("sleep", 0),
        ("kill", 2),
        ("waitpid", os.WNOHANG),
        ("sleep", 0),
        ("kill", 3),
        ("waitpid", os.WNOHANG),
        ("sleep", 0),
        ("kill", 4),
        ("waitpid", 0),
    ]


def test_round17_unadmitted_post_action_signal_error_requires_exact_wnohang_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-action signal error remains primary after exact child reap proof."""

    worker_id = 73
    injected = OSError("round17 post-action direct-child signal failure")
    actions: list[tuple[str, int]] = []

    def kill(process_id: int, signal_number: int) -> None:
        assert (process_id, signal_number) == (worker_id, signal.SIGKILL)
        actions.append(("kill", signal_number))
        raise injected

    def waitpid(process_id: int, options: int) -> tuple[int, int]:
        assert process_id == worker_id
        assert options == os.WNOHANG
        actions.append(("waitpid", options))
        return worker_id, 9

    monkeypatch.setattr(spawn_supervisor.os, "kill", kill)
    monkeypatch.setattr(spawn_supervisor.os, "waitpid", waitpid)

    with pytest.raises(OSError) as raised:
        spawn_supervisor._terminate_and_wait_for_unadmitted_worker(worker_id)

    assert raised.value is injected
    assert actions == [
        ("kill", signal.SIGKILL),
        ("waitpid", os.WNOHANG),
    ]


def test_round17_unadmitted_persistent_errors_never_enter_blocking_wait(
    tmp_path: Path,
) -> None:
    """Persistent ambiguity stays non-green without a blocking waitpid escape."""

    attempts_receipt = tmp_path / "round17-unadmitted-attempts"
    blocking_receipt = tmp_path / "round17-unadmitted-blocking"
    wrapper_source = "\n".join(
        (
            "import importlib.util, os, signal, sys",
            "from pathlib import Path",
            "path, attempts_path, blocking_path = sys.argv[1:]",
            "spec = importlib.util.spec_from_file_location('round17_direct_cleanup', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "worker_id = 73",
            "attempts = 0",
            "def publish(path, payload):",
            "    destination = Path(path)",
            "    temporary = destination.with_name(f'{destination.name}.{os.getpid()}.tmp')",
            "    temporary.write_text(payload, encoding='ascii')",
            "    os.replace(temporary, destination)",
            "def kill(process_id, signal_number):",
            "    assert (process_id, signal_number) == (worker_id, signal.SIGKILL)",
            "    raise PermissionError('round17 persistent signal ambiguity')",
            "def waitpid(process_id, options):",
            "    global attempts",
            "    assert process_id == worker_id",
            "    if options != os.WNOHANG:",
            "        publish(blocking_path, str(options))",
            "        raise AssertionError('blocking waitpid was forbidden')",
            "    attempts += 1",
            "    publish(attempts_path, str(attempts))",
            "    return 0, 0",
            "module.os.kill = kill",
            "module.os.waitpid = waitpid",
            "module._terminate_and_wait_for_unadmitted_worker(worker_id)",
        )
    )
    process = subprocess.Popen(
        [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            wrapper_source,
            os.fspath(Path(spawn_supervisor.__file__).resolve()),
            os.fspath(attempts_receipt),
            os.fspath(blocking_receipt),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    try:
        deadline = time.monotonic() + _WAIT_SECONDS
        while time.monotonic() < deadline:
            if attempts_receipt.exists() and int(attempts_receipt.read_text(encoding="ascii")) >= 4:
                break
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else b""
                raise AssertionError(f"persistent direct-cleanup harness exited early: {stderr!r}")
            time.sleep(0.005)
        else:
            raise TimeoutError("persistent direct-cleanup attempts were not published")

        assert process.poll() is None
        assert not blocking_receipt.exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=_WAIT_SECONDS)
        if process.stderr is not None:
            process.stderr.close()


@pytest.mark.parametrize("fault_site", ["leader", "group"])
def test_round16_repeated_signal_error_cannot_reap_a_live_group_member(
    tmp_path: Path,
    fault_site: str,
) -> None:
    actions_receipt = tmp_path / f"{fault_site}-signal-actions"
    scans_receipt = tmp_path / f"{fault_site}-signal-scans"
    wait_receipt = tmp_path / f"{fault_site}-signal-wait"
    wrapper_source = "\n".join(
        (
            "import importlib.util, json, os, signal, sys, time",
            "from pathlib import Path",
            "path, site, actions_path, scans_path, wait_path = sys.argv[1:]",
            "spec = importlib.util.spec_from_file_location('round16_signal_supervisor', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "worker_id = 73",
            "actions = []",
            "leader_attempts = 0",
            "group_attempts = 0",
            "scan_attempts = 0",
            "real_sleep = time.sleep",
            "def publish(path, payload):",
            "    destination = Path(path)",
            "    temporary = destination.with_name(f'{destination.name}.{os.getpid()}.tmp')",
            "    temporary.write_text(payload, encoding='ascii')",
            "    os.replace(temporary, destination)",
            "def record(action):",
            "    actions.append(action)",
            "    publish(actions_path, json.dumps(actions))",
            "def kill(process_id, signal_number):",
            "    global leader_attempts",
            "    assert (process_id, signal_number) == (worker_id, signal.SIGKILL)",
            "    leader_attempts += 1",
            "    record('leader-kill')",
            "    if site == 'leader' and leader_attempts <= 2: raise PermissionError('persistent leader signal failure')",
            "def killpg(process_group_id, signal_number):",
            "    global group_attempts",
            "    assert (process_group_id, signal_number) == (worker_id, signal.SIGKILL)",
            "    group_attempts += 1",
            "    record('group-kill')",
            "    if site == 'group' and group_attempts <= 2: raise PermissionError('persistent group signal failure')",
            "def invoke_process_table_scan():",
            "    global scan_attempts",
            "    scan_attempts += 1",
            "    publish(scans_path, str(scan_attempts))",
            "    return b'complete', b''",
            "def parse_process_table(_stdout, _stderr):",
            "    return ((worker_id, worker_id, 'Z'), (worker_id + 1, worker_id, 'S'))",
            "def waitpid(_process_id, _options):",
            "    publish(wait_path, 'forbidden')",
            "    raise AssertionError('live group member was reaped past')",
            "module.os.kill = kill",
            "module.os.killpg = killpg",
            "module.os.waitpid = waitpid",
            "module.time.sleep = real_sleep",
            "module._invoke_process_table_scan = invoke_process_table_scan",
            "module._parse_process_table = parse_process_table",
            "module._terminate_and_wait_for_worker(worker_id)",
        )
    )
    process = subprocess.Popen(
        [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            wrapper_source,
            os.fspath(Path(spawn_supervisor.__file__).resolve()),
            fault_site,
            os.fspath(actions_receipt),
            os.fspath(scans_receipt),
            os.fspath(wait_receipt),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    try:
        deadline = time.monotonic() + _WAIT_SECONDS
        while time.monotonic() < deadline:
            if scans_receipt.exists() and int(scans_receipt.read_text(encoding="ascii")) >= 3:
                break
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else b""
                raise AssertionError(f"live-member signal harness exited early: {stderr!r}")
            time.sleep(0.005)
        else:
            raise TimeoutError("live-member signal harness did not publish scans")

        assert not wait_receipt.exists()
        assert json.loads(actions_receipt.read_text(encoding="ascii")) == [
            *(["leader-kill"] * (2 if fault_site == "leader" else 1)),
            *(["group-kill"] * (2 if fault_site == "group" else 1)),
        ]
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_WAIT_SECONDS)
        if process.stderr is not None:
            process.stderr.close()


@pytest.mark.parametrize("role", ["owner", "status", "control"])
@pytest.mark.parametrize("fault_moment", ["pre-action", "post-action"])
@pytest.mark.parametrize("fault_type", [OSError, KeyboardInterrupt, SystemExit])
def test_round12_successful_supervisor_fails_closed_on_owned_close_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    fault_moment: str,
    fault_type: type[BaseException],
) -> None:
    """An unproven owned close changes success to sanitized protocol failure exactly once."""

    launch_calls = _install_successful_protocol(monkeypatch, worker_returncode=0)
    attempts, probes, observed_kernel_actions = _inject_one_close_fault(
        monkeypatch,
        role=role,
        fault_moment=fault_moment,
        fault_type=fault_type,
    )

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 127
    assert (
        attempts
        == {
            "owner": [
                _ROLE_DESCRIPTORS["owner"],
                _ROLE_DESCRIPTORS["status"],
                _ROLE_DESCRIPTORS["control"],
                _WORKER_DESCRIPTOR,
            ],
            "control": [
                _ROLE_DESCRIPTORS["owner"],
                _ROLE_DESCRIPTORS["control"],
                _ROLE_DESCRIPTORS["status"],
                _WORKER_DESCRIPTOR,
            ],
            "status": [
                _ROLE_DESCRIPTORS["owner"],
                _ROLE_DESCRIPTORS["control"],
                _WORKER_DESCRIPTOR,
                _ROLE_DESCRIPTORS["status"],
            ],
        }[role]
    )
    assert launch_calls == ([] if role in {"owner", "control"} else [73])
    assert probes == []
    assert observed_kernel_actions() == (1 if fault_moment == "post-action" else 0)


@pytest.mark.parametrize("role", ["status"])
@pytest.mark.parametrize("fault_moment", ["pre-action", "post-action"])
@pytest.mark.parametrize("fault_type", [OSError, KeyboardInterrupt, SystemExit])
def test_round12_owned_close_error_preserves_prior_worker_failure_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    fault_moment: str,
    fault_type: type[BaseException],
) -> None:
    """Cleanup failure cannot replace an already-failing worker status or retry its raw FD."""

    launch_calls = _install_successful_protocol(monkeypatch, worker_returncode=23)
    attempts, probes, observed_kernel_actions = _inject_one_close_fault(
        monkeypatch,
        role=role,
        fault_moment=fault_moment,
        fault_type=fault_type,
    )

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 23
    assert attempts == [
        _ROLE_DESCRIPTORS["owner"],
        _ROLE_DESCRIPTORS["control"],
        _WORKER_DESCRIPTOR,
        _ROLE_DESCRIPTORS["status"],
    ]
    assert launch_calls == [73]
    assert probes == []
    assert observed_kernel_actions() == (1 if fault_moment == "post-action" else 0)


@pytest.mark.parametrize("role", ["status", "control"])
@pytest.mark.parametrize("fault_moment", ["pre-action", "post-action"])
@pytest.mark.parametrize("fault_type", [OSError, KeyboardInterrupt, SystemExit])
def test_round12_owned_close_fault_cannot_interrupt_remaining_cleanup_after_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    fault_moment: str,
    fault_type: type[BaseException],
) -> None:
    """Every non-lifetime owner is attempted once even after a primary protocol failure."""

    launch_calls = _install_successful_protocol(monkeypatch, worker_returncode=0)
    monkeypatch.setattr(
        spawn_supervisor,
        "_read_control",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("round12 primary failure")),
    )
    attempts, probes, observed_kernel_actions = _inject_one_close_fault(
        monkeypatch,
        role=role,
        fault_moment=fault_moment,
        fault_type=fault_type,
    )

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 127
    assert attempts == [
        _ROLE_DESCRIPTORS["owner"],
        _ROLE_DESCRIPTORS["status"],
        _ROLE_DESCRIPTORS["control"],
        _WORKER_DESCRIPTOR,
    ]
    assert launch_calls == []
    assert probes == []
    assert observed_kernel_actions() == (1 if fault_moment == "post-action" else 0)


def test_round18_worker_bootstrap_is_clean_execed_with_directional_gates() -> None:
    """Worker admission cannot run Python in the supervisor's raw-fork child."""

    supervisor_source = Path(spawn_supervisor.__file__).read_text(encoding="utf-8")
    bootstrap_path = Path(spawn_supervisor.__file__).with_name("worker_bootstrap.py")

    assert "os.fork(" not in supervisor_source
    assert "socket.socketpair(" not in supervisor_source
    assert "subprocess.Popen.__new__(" in supervisor_source
    assert "_initialize_reserved_worker_process(" in supervisor_source
    controller_source = Path(isolated_process.__file__).read_text(encoding="utf-8")
    assert "_TRUSTED_SUPERVISOR_COMMAND" not in controller_source
    assert " is _supervisor_command" not in controller_source
    assert bootstrap_path.is_file()


def test_round18_clean_worker_bootstrap_waits_for_exact_start_and_preserves_identity(
    tmp_path: Path,
) -> None:
    """The execed bootstrap proves its PGID/SID before an exact one-way admission."""

    ready_reader, ready_writer = os.pipe()
    start_reader, start_writer = os.pipe()
    identity_receipt = tmp_path / "worker-bootstrap-identity"
    bootstrap_path = Path(spawn_supervisor.__file__).with_name("worker_bootstrap.py").resolve()
    worker_source = (
        "import os, sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_text("
        "f'{os.getpid()} {os.getpgrp()} {os.getsid(0)}', encoding='ascii')"
    )
    process = subprocess.Popen(
        [
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            os.fspath(bootstrap_path),
            "--ready-fd",
            str(ready_writer),
            "--start-fd",
            str(start_reader),
            "--supervisor-sid",
            str(os.getsid(0)),
            "--",
            os.fspath(Path(sys.executable).resolve()),
            "-I",
            "-c",
            worker_source,
            os.fspath(identity_receipt),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=(ready_writer, start_reader),
        process_group=0,
    )
    os.close(ready_writer)
    os.close(start_reader)
    try:
        assert os.read(ready_reader, 1) == b"R"
        assert os.read(ready_reader, 1) == b""
        assert not identity_receipt.exists()

        assert os.write(start_writer, b"S") == 1
        os.close(start_writer)
        start_writer = -1

        assert process.wait(timeout=_WAIT_SECONDS) == 0
        assert process.stderr is not None
        assert process.stderr.read() == b""
        assert identity_receipt.read_text(encoding="ascii") == (
            f"{process.pid} {process.pid} {os.getsid(0)}"
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=_WAIT_SECONDS)
        for descriptor in (ready_reader, start_writer):
            with suppress(OSError):
                os.close(descriptor)
        if process.stderr is not None:
            process.stderr.close()


@pytest.mark.parametrize("interrupt_site", ["initializer", "classification"])
def test_round18_post_native_interrupt_retains_and_reaps_exact_handle(
    monkeypatch: pytest.MonkeyPatch,
    interrupt_site: str,
) -> None:
    """A BaseException anywhere after native creation cannot orphan the bootstrap."""

    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_arguments",
        lambda: (
            _ROLE_DESCRIPTORS["status"],
            _ROLE_DESCRIPTORS["control"],
            _ROLE_DESCRIPTORS["owner"],
            _ROLE_DESCRIPTORS["lifetime"],
            (),
            [os.fspath(Path(sys.executable).resolve()), "-I", "-c", "pass"],
        ),
    )
    monkeypatch.setattr(spawn_supervisor, "_write_all", lambda *_args: None)
    monkeypatch.setattr(spawn_supervisor, "_read_control", lambda *_args: b"S")
    real_close = os.close

    def close(descriptor: int) -> None:
        if descriptor in _ROLE_DESCRIPTORS.values():
            return
        real_close(descriptor)

    monkeypatch.setattr(spawn_supervisor.os, "close", close)
    real_initialize = spawn_supervisor._initialize_reserved_worker_process
    retained: list[subprocess.Popen[bytes]] = []
    injected = KeyboardInterrupt("round18 post-child initializer interruption")

    def interrupt_after_child(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)
        retained.append(process)
        if interrupt_site == "initializer":
            raise injected

    monkeypatch.setattr(
        spawn_supervisor,
        "_initialize_reserved_worker_process",
        interrupt_after_child,
    )
    real_classifier = spawn_supervisor._reserved_process_has_native_child
    classifier_calls = 0

    def classify(process: subprocess.Popen[bytes]) -> bool:
        nonlocal classifier_calls
        classifier_calls += 1
        if interrupt_site == "classification" and classifier_calls <= 2:
            raise injected
        return real_classifier(process)

    monkeypatch.setattr(spawn_supervisor, "_reserved_process_has_native_child", classify)

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 127
    assert len(retained) == 1
    process = retained[0]
    assert process.pid > 0
    assert process.returncode is not None
    assert _process_not_running(process.pid)
    assert _exact_child_was_reaped(process.pid)


@pytest.mark.parametrize("fault_stage", ["initializer", "classifier", "capture"])
def test_round18_isolated_supervisor_lifecycle_control_retires_every_native_handle(
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
) -> None:
    """Lifecycle control cannot orphan a supervisor process or its exit watcher."""

    real_initialize = isolated_process._initialize_reserved_process
    real_classifier = isolated_process._reserved_process_has_native_child
    real_capture = isolated_process._capture_worker_session
    retained: list[subprocess.Popen[bytes]] = []
    captured_sessions: list[object] = []
    injected = KeyboardInterrupt(f"round18 isolated supervisor {fault_stage} control")
    interrupted = False

    def initialize(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)
        retained.append(process)
        if fault_stage == "initializer":
            raise injected

    def classify(process: subprocess.Popen[bytes]) -> bool:
        nonlocal interrupted
        native_child = real_classifier(process)
        if fault_stage == "classifier" and native_child and not interrupted:
            interrupted = True
            raise injected
        return native_child

    def capture(
        process: subprocess.Popen[bytes],
        retained_worker: object,
    ) -> object:
        captured = real_capture(process, retained_worker)  # type: ignore[arg-type]
        captured_sessions.append(captured)
        if fault_stage == "capture":
            raise injected
        return captured

    monkeypatch.setattr(isolated_process, "_initialize_reserved_process", initialize)
    monkeypatch.setattr(isolated_process, "_reserved_process_has_native_child", classify)
    monkeypatch.setattr(isolated_process, "_capture_worker_session", capture)

    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            isolated_process.run_isolated_process(
                command=[sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                request=b"{}",
                timeout_seconds=0.2,
                max_response_bytes=128,
                env=isolated_process.sanitized_worker_environment(),
            )

        assert raised.value is injected
        assert len(retained) == 1
        process = retained[0]
        assert process.pid > 0
        assert _process_not_running(process.pid)
        assert _exact_child_was_reaped(process.pid)
        assert len(captured_sessions) == 1
        captured = captured_sessions[0]
        assert captured.cleaned  # type: ignore[attr-defined]
        watcher = captured.exit_watcher  # type: ignore[attr-defined]
        assert watcher is not None
        assert watcher._pidfd is None
        assert watcher._kqueue is None
        assert watcher._close_error is None
    finally:
        for process in retained:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=_WAIT_SECONDS)


def test_round18_poisoned_worker_resource_state_refuses_isolated_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambiguous prior watcher close remains visible and forbids later spawn."""

    initializer_calls: list[bool] = []
    monkeypatch.setattr(
        isolated_process,
        "_worker_resource_cleanup_is_poisoned",
        lambda: True,
    )

    def initialize(*_args: object, **_kwargs: object) -> None:
        initializer_calls.append(True)

    monkeypatch.setattr(isolated_process, "_initialize_reserved_process", initialize)

    with pytest.raises(isolated_process.IsolatedProcessCleanupError):
        isolated_process.run_isolated_process(
            command=[sys.executable, "-I", "-c", "pass"],
            request=b"{}",
            timeout_seconds=0.2,
            max_response_bytes=128,
            env=isolated_process.sanitized_worker_environment(),
        )

    assert initializer_calls == []


@pytest.mark.parametrize("gate_role", ["ready", "start"])
@pytest.mark.parametrize("fault_type", [OSError, KeyboardInterrupt, SystemExit])
def test_round18_bootstrap_gate_close_uncertainty_fails_before_target_exec(
    monkeypatch: pytest.MonkeyPatch,
    gate_role: str,
    fault_type: type[BaseException],
) -> None:
    """Each child gate is retired once; an ambiguous close forbids target execution."""

    ready_descriptor = 41
    start_descriptor = 43
    monkeypatch.setattr(
        worker_bootstrap,
        "_parse_arguments",
        lambda: (ready_descriptor, start_descriptor, 59, (), ["trusted-worker"]),
    )
    monkeypatch.setattr(worker_bootstrap.signal, "pthread_sigmask", lambda *_args: set())
    monkeypatch.setattr(worker_bootstrap.signal, "signal", lambda *_args: signal.SIG_DFL)
    monkeypatch.setattr(worker_bootstrap.os, "getpid", lambda: 73)
    monkeypatch.setattr(worker_bootstrap.os, "getpgrp", lambda: 73)
    monkeypatch.setattr(worker_bootstrap.os, "getsid", lambda _process_id: 59)
    monkeypatch.setattr(worker_bootstrap.os, "write", lambda _descriptor, payload: len(payload))
    admissions = iter((b"S", b""))
    monkeypatch.setattr(
        worker_bootstrap.os,
        "read",
        lambda _descriptor, _size: next(admissions),
    )
    close_calls: list[int] = []
    fault_descriptor = ready_descriptor if gate_role == "ready" else start_descriptor

    def close(descriptor: int) -> None:
        close_calls.append(descriptor)
        if descriptor == fault_descriptor:
            raise fault_type(f"round18 injected {gate_role} close uncertainty")

    exec_calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(worker_bootstrap.os, "close", close)
    monkeypatch.setattr(
        worker_bootstrap.os,
        "execvpe",
        lambda executable, command, _environment: exec_calls.append((executable, command)),
    )

    with pytest.raises(SystemExit) as raised:
        worker_bootstrap.main()

    assert raised.value.code == 127
    assert close_calls == (
        [ready_descriptor] if gate_role == "ready" else [ready_descriptor, start_descriptor]
    )
    assert close_calls.count(fault_descriptor) == 1
    assert exec_calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "worker_bootstrap.py",
            "--ready-fd",
            "03",
            "--start-fd",
            "5",
            "--supervisor-sid",
            "7",
            "--",
            "worker",
        ],
        [
            "worker_bootstrap.py",
            "--ready-fd",
            "3",
            "--start-fd",
            "3",
            "--supervisor-sid",
            "7",
            "--",
            "worker",
        ],
        [
            "worker_bootstrap.py",
            "--ready-fd",
            "3",
            "--start-fd",
            "5",
            "--supervisor-sid",
            "07",
            "--",
            "worker",
        ],
        [
            "worker_bootstrap.py",
            "--ready-fd",
            "3",
            "--start-fd",
            "5",
            "--supervisor-sid",
            "7",
            "--worker-fd",
            "3",
            "--",
            "worker",
        ],
    ],
)
def test_round18_bootstrap_argv_rejects_noncanonical_or_duplicate_roles(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(worker_bootstrap.sys, "argv", arguments)

    with pytest.raises(ValueError):
        worker_bootstrap._parse_arguments()  # type: ignore[attr-defined]


@pytest.mark.parametrize("identity_fault", ["pgid", "sid"])
def test_round18_bootstrap_rejects_wrong_identity_before_ready(
    monkeypatch: pytest.MonkeyPatch,
    identity_fault: str,
) -> None:
    monkeypatch.setattr(
        worker_bootstrap,
        "_parse_arguments",
        lambda: (41, 43, 59, (), ["trusted-worker"]),
    )
    monkeypatch.setattr(worker_bootstrap.signal, "pthread_sigmask", lambda *_args: set())
    monkeypatch.setattr(worker_bootstrap.signal, "signal", lambda *_args: signal.SIG_DFL)
    monkeypatch.setattr(worker_bootstrap.os, "getpid", lambda: 73)
    monkeypatch.setattr(
        worker_bootstrap.os,
        "getpgrp",
        lambda: 79 if identity_fault == "pgid" else 73,
    )
    monkeypatch.setattr(
        worker_bootstrap.os,
        "getsid",
        lambda _process_id: 61 if identity_fault == "sid" else 59,
    )
    publications: list[bytes] = []
    exec_calls: list[str] = []
    monkeypatch.setattr(
        worker_bootstrap.os,
        "write",
        lambda _descriptor, payload: publications.append(payload) or len(payload),
    )
    monkeypatch.setattr(
        worker_bootstrap.os,
        "execvpe",
        lambda executable, _command, _environment: exec_calls.append(executable),
    )

    with pytest.raises(SystemExit) as raised:
        worker_bootstrap.main()

    assert raised.value.code == 127
    assert publications == []
    assert exec_calls == []


def test_round18_supervisor_rejects_colliding_gate_roles_before_native_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_arguments",
        lambda: (
            _ROLE_DESCRIPTORS["status"],
            _ROLE_DESCRIPTORS["control"],
            _ROLE_DESCRIPTORS["owner"],
            _ROLE_DESCRIPTORS["lifetime"],
            (),
            ["trusted-worker"],
        ),
    )
    monkeypatch.setattr(spawn_supervisor, "_write_all", lambda *_args: None)
    monkeypatch.setattr(spawn_supervisor, "_read_control", lambda *_args: b"S")
    gate_pairs = iter(((51, 53), (53, 57)))
    monkeypatch.setattr(spawn_supervisor, "_open_directional_gate", lambda: next(gate_pairs))
    monkeypatch.setattr(spawn_supervisor.os, "close", lambda _descriptor: None)
    launch_calls: list[bool] = []

    def initialize(*_args: object, **_kwargs: object) -> None:
        launch_calls.append(True)

    monkeypatch.setattr(spawn_supervisor, "_initialize_reserved_worker_process", initialize)

    with pytest.raises(SystemExit) as raised:
        spawn_supervisor.main()

    assert raised.value.code == 127
    assert launch_calls == []


@pytest.mark.parametrize(
    "fault_site",
    [
        "close-ready-writer",
        "close-start-reader",
        "identity",
        "read-ready",
        "close-ready-reader",
        "write-start",
        "close-start-writer",
    ],
)
def test_round18_each_supervisor_gate_uncertainty_exactly_reaps_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    fault_site: str,
) -> None:
    """No parent-side gate ambiguity can publish STARTED or leave a bootstrap alive."""

    real_pipe = os.pipe
    real_close = os.close
    role_pairs = {role: real_pipe() for role in ("status", "control", "owner", "lifetime")}
    role_descriptors = {
        "status": role_pairs["status"][1],
        "control": role_pairs["control"][0],
        "owner": role_pairs["owner"][1],
        "lifetime": role_pairs["lifetime"][1],
    }
    monkeypatch.setattr(
        spawn_supervisor,
        "_parse_arguments",
        lambda: (
            role_descriptors["status"],
            role_descriptors["control"],
            role_descriptors["owner"],
            role_descriptors["lifetime"],
            (),
            [os.fspath(Path(sys.executable).resolve()), "-I", "-c", "pass"],
        ),
    )
    monkeypatch.setattr(spawn_supervisor, "_read_control", lambda *_args: b"S")
    real_initialize = spawn_supervisor._initialize_reserved_worker_process
    real_bootstrap_command = spawn_supervisor._worker_bootstrap_command
    gate_pairs = [real_pipe(), real_pipe()]
    available_gate_pairs = iter(gate_pairs)
    retained: list[subprocess.Popen[bytes]] = []
    initializer_errors: list[BaseException] = []
    status_publications: list[bytes] = []

    def pipe() -> tuple[int, int]:
        return next(available_gate_pairs)

    def initialize(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        retained.append(process)
        try:
            real_initialize(process, *args, **kwargs)
        except BaseException as exc:
            initializer_errors.append(exc)
            raise

    def fault_descriptor() -> int | None:
        if len(gate_pairs) != 2:
            return None
        ready_reader, ready_writer = gate_pairs[0]
        start_reader, start_writer = gate_pairs[1]
        return {
            "close-ready-writer": ready_writer,
            "close-start-reader": start_reader,
            "close-ready-reader": ready_reader,
            "close-start-writer": start_writer,
        }.get(fault_site)

    def close_gate(descriptor: int) -> None:
        if descriptor == fault_descriptor():
            raise OSError(f"round18 injected {fault_site} uncertainty")
        real_close(descriptor)

    def bootstrap_command(
        ready_descriptor: int,
        start_descriptor: int,
        _supervisor_session_id: int,
        worker_descriptors: tuple[int, ...],
        command: list[str],
    ) -> list[str]:
        return real_bootstrap_command(
            ready_descriptor,
            start_descriptor,
            os.getsid(0),
            worker_descriptors,
            command,
        )

    real_write_all = spawn_supervisor._write_all

    def write_all(descriptor: int, payload: bytes) -> None:
        if descriptor in {role_descriptors["owner"], role_descriptors["status"]}:
            if descriptor == role_descriptors["status"]:
                status_publications.append(payload)
            return
        if len(gate_pairs) == 2 and descriptor == gate_pairs[1][1] and fault_site == "write-start":
            raise OSError("round18 injected START write uncertainty")
        real_write_all(descriptor, payload)

    monkeypatch.setattr(spawn_supervisor, "_open_directional_gate", pipe)
    monkeypatch.setattr(spawn_supervisor, "_close_gate_descriptor", close_gate)
    monkeypatch.setattr(spawn_supervisor, "_write_all", write_all)
    monkeypatch.setattr(spawn_supervisor, "_initialize_reserved_worker_process", initialize)
    monkeypatch.setattr(spawn_supervisor, "_worker_bootstrap_command", bootstrap_command)
    if fault_site == "identity":
        monkeypatch.setattr(
            spawn_supervisor,
            "_corroborate_worker_identity",
            lambda *_args: (_ for _ in ()).throw(OSError("round18 identity uncertainty")),
        )
    else:
        monkeypatch.setattr(spawn_supervisor, "_corroborate_worker_identity", lambda *_args: None)
    if fault_site == "read-ready":
        monkeypatch.setattr(
            spawn_supervisor,
            "_read_bootstrap_ready",
            lambda _descriptor: (_ for _ in ()).throw(OSError("round18 READY uncertainty")),
        )

    try:
        with pytest.raises(SystemExit) as raised:
            spawn_supervisor.main()
    finally:
        for pair in (*role_pairs.values(), *gate_pairs):
            for descriptor in pair:
                with suppress(OSError):
                    real_close(descriptor)

    assert raised.value.code == 127
    assert len(retained) == 1
    assert initializer_errors == []
    process = retained[0]
    assert process.returncode is not None
    assert _process_not_running(process.pid)
    assert _exact_child_was_reaped(process.pid)
    assert all(not payload.startswith(b"STARTED ") for payload in status_publications)


def test_round19_pipe_construction_close_ambiguity_permanently_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw pipe never becomes managed when construction fails, so close must poison."""

    real_close = os.close
    reader_descriptor, writer_descriptor = os.pipe()
    close_attempts: list[int] = []
    with isolated_process._PRIVATE_PIPE_STATE_LOCK:
        poison_before = isolated_process._PRIVATE_PIPE_STATE_POISONED
        active_before = isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS

    def reject_endpoint(
        _descriptor: int,
        *,
        readable: bool,
        writable: bool,
    ) -> object:
        del readable, writable
        raise RuntimeError("round19 endpoint construction failed")

    def ambiguous_close(descriptor: int) -> None:
        close_attempts.append(descriptor)
        if descriptor == reader_descriptor:
            raise OSError("round19 raw pipe close was ambiguous")
        real_close(descriptor)

    monkeypatch.setattr(isolated_process.os, "pipe", lambda: (reader_descriptor, writer_descriptor))
    monkeypatch.setattr(isolated_process, "PrivatePipeEndpoint", reject_endpoint)
    monkeypatch.setattr(isolated_process.os, "close", ambiguous_close)
    try:
        with pytest.raises(OSError, match="raw pipe close was ambiguous") as raised:
            isolated_process._owned_pipe_channel(nonblocking_writer=True)

        assert isinstance(raised.value.__cause__, RuntimeError)
        assert close_attempts == [reader_descriptor, writer_descriptor]
        with isolated_process._PRIVATE_PIPE_STATE_LOCK:
            assert isolated_process._PRIVATE_PIPE_STATE_POISONED
            assert active_before == isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS
        with pytest.raises(isolated_process.IsolatedProcessCleanupError):
            isolated_process._owned_pipe_channel(nonblocking_writer=True)
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        with suppress(OSError):
            real_close(reader_descriptor)
        with suppress(OSError):
            real_close(writer_descriptor)
        with isolated_process._PRIVATE_PIPE_STATE_LOCK:
            isolated_process._PRIVATE_PIPE_STATE_POISONED = poison_before
            isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS = active_before


def test_round19_pipe_duplicate_close_ambiguity_permanently_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed duplicate wrapper retires its one raw descriptor and poisons ambiguity."""

    real_close = os.close
    real_dup = os.dup
    source_reader, source_writer = os.pipe()
    duplicate_descriptors: list[int] = []
    close_attempts: list[int] = []
    with isolated_process._PRIVATE_PIPE_STATE_LOCK:
        poison_before = isolated_process._PRIVATE_PIPE_STATE_POISONED
        active_before = isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS

    def capture_duplicate(descriptor: int) -> int:
        duplicate = real_dup(descriptor)
        duplicate_descriptors.append(duplicate)
        return duplicate

    def reject_endpoint(
        _descriptor: int,
        *,
        readable: bool,
        writable: bool,
    ) -> object:
        del readable, writable
        raise RuntimeError("round19 duplicate construction failed")

    def ambiguous_close(descriptor: int) -> None:
        close_attempts.append(descriptor)
        if duplicate_descriptors and descriptor == duplicate_descriptors[0]:
            raise OSError("round19 duplicate close was ambiguous")
        real_close(descriptor)

    monkeypatch.setattr(isolated_process.os, "dup", capture_duplicate)
    monkeypatch.setattr(isolated_process, "PrivatePipeEndpoint", reject_endpoint)
    monkeypatch.setattr(isolated_process.os, "close", ambiguous_close)
    try:
        with pytest.raises(OSError, match="duplicate close was ambiguous") as raised:
            isolated_process._private_pipe_duplicate(
                source_reader,
                readable=True,
                writable=False,
            )

        assert isinstance(raised.value.__cause__, RuntimeError)
        assert len(duplicate_descriptors) == 1
        assert close_attempts == duplicate_descriptors
        with isolated_process._PRIVATE_PIPE_STATE_LOCK:
            assert isolated_process._PRIVATE_PIPE_STATE_POISONED
            assert active_before == isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS
        with pytest.raises(isolated_process.IsolatedProcessCleanupError):
            isolated_process._private_pipe_duplicate(
                source_reader,
                readable=True,
                writable=False,
            )
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        for descriptor in duplicate_descriptors:
            with suppress(OSError):
                real_close(descriptor)
        real_close(source_reader)
        real_close(source_writer)
        with isolated_process._PRIVATE_PIPE_STATE_LOCK:
            isolated_process._PRIVATE_PIPE_STATE_POISONED = poison_before
            isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS = active_before


def test_round19_set_blocking_cleanup_attempts_both_endpoints_and_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed cleanup retires both endpoints even when one close is ambiguous."""

    real_close = os.close
    reader_descriptor, writer_descriptor = os.pipe()
    close_attempts: list[int] = []
    with isolated_process._PRIVATE_PIPE_STATE_LOCK:
        poison_before = isolated_process._PRIVATE_PIPE_STATE_POISONED
        active_before = isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS

    def reject_set_blocking(_descriptor: int, _blocking: bool) -> None:
        raise OSError("round19 set_blocking failed")

    def ambiguous_close(descriptor: int) -> None:
        close_attempts.append(descriptor)
        if descriptor == reader_descriptor:
            raise OSError("round19 managed close was ambiguous")
        real_close(descriptor)

    monkeypatch.setattr(isolated_process.os, "pipe", lambda: (reader_descriptor, writer_descriptor))
    monkeypatch.setattr(isolated_process.os, "set_blocking", reject_set_blocking)
    monkeypatch.setattr(isolated_process.os, "close", ambiguous_close)
    try:
        with pytest.raises(OSError, match="set_blocking failed"):
            isolated_process._owned_pipe_channel(nonblocking_writer=True)

        assert close_attempts == [reader_descriptor, writer_descriptor]
        with isolated_process._PRIVATE_PIPE_STATE_LOCK:
            assert isolated_process._PRIVATE_PIPE_STATE_POISONED
            assert active_before == isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS
        with pytest.raises(isolated_process.IsolatedProcessCleanupError):
            isolated_process._owned_pipe_channel(nonblocking_writer=True)
    finally:
        monkeypatch.setattr(isolated_process.os, "close", real_close)
        with suppress(OSError):
            real_close(reader_descriptor)
        with suppress(OSError):
            real_close(writer_descriptor)
        with isolated_process._PRIVATE_PIPE_STATE_LOCK:
            isolated_process._PRIVATE_PIPE_STATE_POISONED = poison_before
            isolated_process._ACTIVE_PRIVATE_PIPE_ENDPOINTS = active_before
