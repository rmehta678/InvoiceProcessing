"""Trusted START/ABORT supervisor for one isolated worker process group."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

_START = b"S"
_ABORT = b"A"
_BOOTSTRAP_READY = b"R"
_PROTOCOL_FAILURE = 127
_PROCESS_TABLE_COMMAND = ("/bin/ps", "-axo", "pid=,pgid=,stat=")
_PROCESS_TABLE_MAX_BYTES = 1_048_576


class _SupervisorTerminationRequested(BaseException):
    """Internal control raised when the owning controller requests cleanup."""


def _request_supervisor_termination(
    _signal_number: int,
    _frame: object,
) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    finally:
        raise _SupervisorTerminationRequested


def _write_all(descriptor: int, payload: bytes) -> None:
    written = os.write(descriptor, payload)
    if written != len(payload):
        raise OSError("isolated supervisor process publication was incomplete")


def _write_owner_once(descriptor: int, process_id: int) -> None:
    """Publish the lifetime owner in one exact, non-retryable pipe write."""

    payload = f"OWNER {process_id}\n".encode("ascii")
    pipe_buffer = os.pathconf(os.curdir, "PC_PIPE_BUF")
    if len(payload) > pipe_buffer:
        raise OSError("isolated supervisor ownership publication exceeded PIPE_BUF")
    _write_all(descriptor, payload)


def _canonical_descriptor(value: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError("invalid supervisor descriptor")
    descriptor = int(value)
    if descriptor < 3 or str(descriptor) != value:
        raise ValueError("invalid supervisor descriptor")
    return descriptor


def _parse_arguments() -> tuple[int, int, int, int, tuple[int, ...], list[str]]:
    expected = ("--status-fd", "--control-fd", "--owner-fd", "--lifetime-fd")
    if (
        len(sys.argv) < 11
        or (
            sys.argv[1],
            sys.argv[3],
            sys.argv[5],
            sys.argv[7],
        )
        != expected
    ):
        raise ValueError("invalid supervisor arguments")
    status = _canonical_descriptor(sys.argv[2])
    control = _canonical_descriptor(sys.argv[4])
    owner = _canonical_descriptor(sys.argv[6])
    lifetime = _canonical_descriptor(sys.argv[8])
    worker_descriptors: list[int] = []
    argument_index = 9
    while argument_index < len(sys.argv) and sys.argv[argument_index] == "--worker-fd":
        if argument_index + 1 >= len(sys.argv):
            raise ValueError("invalid supervisor arguments")
        worker_descriptors.append(_canonical_descriptor(sys.argv[argument_index + 1]))
        argument_index += 2
    if argument_index >= len(sys.argv) or sys.argv[argument_index] != "--":
        raise ValueError("invalid supervisor arguments")
    protocol_descriptors = (status, control, owner, lifetime)
    all_descriptors = (*protocol_descriptors, *worker_descriptors)
    if len(set(all_descriptors)) != len(all_descriptors):
        raise ValueError("supervisor descriptors must be role separated")
    command = sys.argv[argument_index + 1 :]
    if not command or any(not argument for argument in command):
        raise ValueError("invalid supervisor worker command")
    return status, control, owner, lifetime, tuple(worker_descriptors), command


def _read_control(status_descriptor: int, control_descriptor: int) -> bytes | None:
    """Read one EOF-terminated byte, acknowledging START before its EOF."""

    first = os.read(control_descriptor, 1)
    if not first:
        return None
    if first == _START:
        _write_all(status_descriptor, b"CONTROL S\n")
    trailing = os.read(control_descriptor, 1)
    if trailing:
        raise ValueError("invalid supervisor control frame")
    if first == _ABORT:
        return _ABORT
    if first != _START:
        raise ValueError("invalid supervisor control frame")
    return _START


def _read_bootstrap_ready(descriptor: int) -> None:
    readiness = os.read(descriptor, 1)
    trailing = os.read(descriptor, 1)
    if readiness != _BOOTSTRAP_READY or trailing:
        raise ValueError("invalid worker bootstrap readiness")


def _worker_bootstrap_command(
    ready_descriptor: int,
    start_descriptor: int,
    supervisor_session_id: int,
    worker_descriptors: tuple[int, ...],
    command: list[str],
) -> list[str]:
    bootstrap_path = Path(__file__).with_name("worker_bootstrap.py").resolve(strict=True)
    arguments = [
        os.fspath(Path(sys.executable).resolve(strict=True)),
        "-I",
        os.fspath(bootstrap_path),
        "--ready-fd",
        str(ready_descriptor),
        "--start-fd",
        str(start_descriptor),
        "--supervisor-sid",
        str(supervisor_session_id),
    ]
    for descriptor in worker_descriptors:
        arguments.extend(("--worker-fd", str(descriptor)))
    arguments.extend(("--", *command))
    return arguments


def _corroborate_worker_identity(process_id: int, supervisor_session_id: int) -> None:
    if os.getpgid(process_id) != process_id or os.getsid(process_id) != supervisor_session_id:
        raise RuntimeError("worker bootstrap process identity was not isolated")


def _record_reaped_status(
    process: subprocess.Popen[bytes] | None,
    wait_status: int,
) -> None:
    if process is not None:
        process.returncode = int(os.waitstatus_to_exitcode(wait_status))


def _close_gate_descriptor(descriptor: int) -> None:
    os.close(descriptor)


def _open_directional_gate() -> tuple[int, int]:
    return os.pipe()


def _initialize_reserved_worker_process(
    process: subprocess.Popen[bytes],
    *args: object,
    **kwargs: object,
) -> None:
    """Initialize one worker handle that the supervisor already retains."""

    subprocess.Popen.__init__(process, *args, **kwargs)  # type: ignore[call-overload]


def _reserved_process_has_native_child(process: subprocess.Popen[bytes]) -> bool:
    process_id = getattr(process, "pid", None)
    return (
        getattr(process, "_child_created", False) is True
        and type(process_id) is int
        and process_id > 0
        and getattr(process, "returncode", None) is None
    )


def _wait_for_worker(process_id: int) -> int | None:
    """Poll once and return raw status only after exactly reaping the child."""

    while True:
        try:
            observed_id, status = os.waitpid(process_id, os.WNOHANG)
        except InterruptedError:
            continue
        if observed_id == 0:
            return None
        if observed_id != process_id:
            raise ChildProcessError("supervisor reaped an unexpected worker")
        return status


def _invoke_process_table_scan() -> tuple[bytes, bytes]:
    scan_environment = {**os.environ, "LANG": "C", "LC_ALL": "C"}
    completed = subprocess.run(
        list(_PROCESS_TABLE_COMMAND),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        env=scan_environment,
        text=False,
        timeout=1.0,
    )
    return completed.stdout, completed.stderr


def _parse_process_table(
    stdout: bytes,
    stderr: bytes,
) -> tuple[tuple[int, int, str], ...]:
    if stderr:
        raise ValueError("process-table scanner produced stderr")
    if not stdout or len(stdout) > _PROCESS_TABLE_MAX_BYTES or not stdout.endswith(b"\n"):
        raise ValueError("process-table scanner output was empty, partial, or oversized")
    if b"\r" in stdout:
        raise ValueError("process-table scanner output used noncanonical line endings")

    rows: list[tuple[int, int, str]] = []
    process_ids: set[int] = set()
    row_pattern = re.compile(rb" *([1-9][0-9]*) +([1-9][0-9]*) +([^\x00-\x20\x7f-\xff]+)( *)")
    for raw_line in stdout.splitlines():
        matched = row_pattern.fullmatch(raw_line)
        if matched is None:
            raise ValueError("process-table scanner row was not canonical")
        raw_process_id, raw_process_group_id, raw_state, raw_padding = matched.groups()
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


def _snapshot_proves_group_extinction(
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


def _terminate_and_wait_for_unadmitted_worker(
    process_id: int,
    *,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    """Reap a trusted bootstrap child that was never offered its exec gate."""

    first_error: BaseException | None = None

    def preserve_first(error: BaseException) -> None:
        nonlocal first_error
        if first_error is None:
            first_error = error

    child_reaped = False
    while True:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            break
        except BaseException as exc:
            preserve_first(exc)
            try:
                observed_id, wait_status = os.waitpid(process_id, os.WNOHANG)
            except InterruptedError:
                continue
            except ChildProcessError as wait_error:
                preserve_first(wait_error)
                child_reaped = True
                break
            except BaseException as wait_error:
                preserve_first(wait_error)
            else:
                if observed_id == process_id:
                    _record_reaped_status(process, wait_status)
                    child_reaped = True
                    break
                if observed_id != 0:
                    preserve_first(ChildProcessError("supervisor reaped an unexpected worker"))
            try:
                time.sleep(0.001)
            except BaseException as sleep_error:
                preserve_first(sleep_error)
        else:
            break

    if not child_reaped:
        wait_was_ambiguous = False
        while True:
            try:
                observed_id, wait_status = os.waitpid(process_id, 0)
            except InterruptedError:
                continue
            except ChildProcessError as exc:
                if not wait_was_ambiguous:
                    preserve_first(exc)
                break
            except BaseException as exc:
                preserve_first(exc)
                wait_was_ambiguous = True
                continue
            if observed_id != process_id:
                preserve_first(ChildProcessError("supervisor reaped an unexpected worker"))
                wait_was_ambiguous = True
                continue
            _record_reaped_status(process, wait_status)
            break

    if first_error is not None:
        raise first_error


def _terminate_and_wait_for_worker(
    process_id: int,
    *,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    """Retire one admitted worker group before exposing any cleanup control."""

    first_error: BaseException | None = None

    def preserve_first(error: BaseException) -> None:
        nonlocal first_error
        if first_error is None:
            first_error = error

    leader_signal_was_ambiguous = False
    while True:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            break
        except BaseException as exc:
            preserve_first(exc)
            if leader_signal_was_ambiguous:
                break
            leader_signal_was_ambiguous = True
        else:
            break

    group_signal_was_ambiguous = False
    while True:
        try:
            os.killpg(process_id, signal.SIGKILL)
        except ProcessLookupError:
            break
        except BaseException as exc:
            preserve_first(exc)
            if group_signal_was_ambiguous:
                break
            group_signal_was_ambiguous = True
        else:
            break

    consecutive_extinction_observations = 0
    while consecutive_extinction_observations < 2:
        try:
            stdout, stderr = _invoke_process_table_scan()
            rows = _parse_process_table(stdout, stderr)
        except BaseException as exc:
            preserve_first(exc)
            consecutive_extinction_observations = 0
        else:
            if _snapshot_proves_group_extinction(rows, process_id):
                consecutive_extinction_observations += 1
            else:
                consecutive_extinction_observations = 0
        if consecutive_extinction_observations >= 2:
            break
        try:
            time.sleep(0.001)
        except BaseException as exc:
            preserve_first(exc)
            consecutive_extinction_observations = 0

    wait_was_ambiguous = False
    while True:
        try:
            observed_id, wait_status = os.waitpid(process_id, 0)
        except InterruptedError:
            continue
        except ChildProcessError as exc:
            if not wait_was_ambiguous:
                preserve_first(exc)
            break
        except BaseException as exc:
            preserve_first(exc)
            wait_was_ambiguous = True
            continue
        if observed_id != process_id:
            preserve_first(ChildProcessError("supervisor reaped an unexpected worker"))
            wait_was_ambiguous = True
            continue
        _record_reaped_status(process, wait_status)
        break

    if first_error is not None:
        raise first_error


def _terminate_owned_worker_preserving_error(
    process_id: int,
    *,
    bootstrap_group_proven: bool,
    admission_may_have_escaped: bool,
    initiating_error: BaseException | None,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    """Clean one child without replacing the failure that initiated cleanup."""

    cleanup_error: BaseException | None = None
    try:
        if admission_may_have_escaped:
            if not bootstrap_group_proven:
                raise RuntimeError("worker admission escaped before process-group proof")
            if process is None:
                _terminate_and_wait_for_worker(process_id)
            else:
                _terminate_and_wait_for_worker(process_id, process=process)
        else:
            if process is None:
                _terminate_and_wait_for_unadmitted_worker(process_id)
            else:
                _terminate_and_wait_for_unadmitted_worker(process_id, process=process)
    except BaseException as exc:
        cleanup_error = exc
    if initiating_error is not None:
        raise initiating_error
    if cleanup_error is not None:
        raise cleanup_error


def _run() -> int:
    """Authorize exactly one worker only after START plus control EOF."""

    (
        status_descriptor,
        control_descriptor,
        owner_descriptor,
        _lifetime_descriptor,
        worker_descriptors,
        command,
    ) = _parse_arguments()
    signal.pthread_sigmask(signal.SIG_SETMASK, set())
    owned_descriptors = {
        status_descriptor,
        control_descriptor,
        owner_descriptor,
        *worker_descriptors,
    }
    close_failed = False
    worker_process: subprocess.Popen[bytes] | None = None
    worker_owned = False
    worker_initialization_completed = False
    worker_classification_completed = False
    worker_native_child = False
    bootstrap_group_proven = False
    admission_may_have_escaped = False

    def close_owned(descriptor: int) -> bool:
        """Retire an owned descriptor before its sole, potentially ambiguous close."""

        nonlocal close_failed
        if descriptor not in owned_descriptors:
            return True
        owned_descriptors.remove(descriptor)
        try:
            os.close(descriptor)
        except BaseException:
            close_failed = True
            return False
        return True

    return_code = _PROTOCOL_FAILURE
    try:
        process_id = os.getpid()
        _write_owner_once(owner_descriptor, process_id)
        if not close_owned(owner_descriptor):
            return _PROTOCOL_FAILURE
        _write_all(status_descriptor, f"READY {process_id}\n".encode("ascii"))
        control = _read_control(status_descriptor, control_descriptor)
        if close_owned(control_descriptor):
            if control is None or control == _ABORT:
                return_code = 0
            else:
                signal.signal(signal.SIGINT, _request_supervisor_termination)
                signal.signal(signal.SIGTERM, _request_supervisor_termination)
                gate_descriptors: set[int] = set()
                launch_error: BaseException | None = None

                def close_gate(descriptor: int) -> None:
                    if descriptor not in gate_descriptors:
                        return
                    gate_descriptors.remove(descriptor)
                    _close_gate_descriptor(descriptor)

                try:
                    ready_reader, ready_writer = _open_directional_gate()
                    gate_descriptors.update((ready_reader, ready_writer))
                    start_reader, start_writer = _open_directional_gate()
                    gate_descriptors.update((start_reader, start_writer))
                    gate_roles = (ready_reader, ready_writer, start_reader, start_writer)
                    if len(set(gate_roles)) != len(gate_roles) or set(gate_roles) & {
                        status_descriptor,
                        control_descriptor,
                        owner_descriptor,
                        _lifetime_descriptor,
                        *worker_descriptors,
                    }:
                        raise RuntimeError("worker bootstrap gate descriptors were not role separated")
                    worker_process = cast(
                        "subprocess.Popen[bytes]",
                        subprocess.Popen.__new__(subprocess.Popen),
                    )
                    worker_process._child_created = False  # type: ignore[attr-defined]
                    _initialize_reserved_worker_process(
                        worker_process,
                        _worker_bootstrap_command(
                            ready_writer,
                            start_reader,
                            process_id,
                            worker_descriptors,
                            command,
                        ),
                        stdin=None,
                        stdout=None,
                        stderr=None,
                        close_fds=True,
                        pass_fds=(ready_writer, start_reader, *worker_descriptors),
                        process_group=0,
                    )
                    worker_initialization_completed = True
                    worker_native_child = _reserved_process_has_native_child(worker_process)
                    worker_classification_completed = True
                    if not worker_native_child:
                        raise RuntimeError(
                            "worker bootstrap initialization produced no native child"
                        )
                except BaseException as exc:
                    launch_error = exc
                    if worker_process is not None:
                        if (
                            worker_initialization_completed
                            and not worker_classification_completed
                        ):
                            # Successful Popen initialization is positive ownership
                            # evidence even when classification itself is interrupted.
                            worker_native_child = True
                        elif not worker_classification_completed:
                            while True:
                                try:
                                    worker_native_child = _reserved_process_has_native_child(
                                        worker_process
                                    )
                                    worker_classification_completed = True
                                    break
                                except BaseException:
                                    try:
                                        time.sleep(0.001)
                                    except BaseException:
                                        continue
                        if not worker_native_child:
                            worker_process._child_created = False  # type: ignore[attr-defined]
                            worker_process = None
                if worker_process is None:
                    for descriptor in tuple(gate_descriptors):
                        try:
                            close_gate(descriptor)
                        except BaseException as exc:
                            if launch_error is None:
                                launch_error = exc
                    if launch_error is None:
                        launch_error = RuntimeError("worker bootstrap launch returned no process")
                    raise launch_error

                worker_id = worker_process.pid
                worker_owned = True
                post_spawn_error: BaseException | None = launch_error

                def preserve_post_spawn_error(error: BaseException) -> None:
                    nonlocal post_spawn_error
                    if post_spawn_error is None:
                        post_spawn_error = error

                def terminate_owned_worker() -> None:
                    nonlocal worker_owned
                    try:
                        _terminate_owned_worker_preserving_error(
                            worker_id,
                            bootstrap_group_proven=bootstrap_group_proven,
                            admission_may_have_escaped=admission_may_have_escaped,
                            initiating_error=post_spawn_error,
                            process=worker_process,
                        )
                    finally:
                        worker_owned = False

                try:
                    gate_proven = post_spawn_error is None
                    for descriptor in (ready_writer, start_reader):
                        try:
                            close_gate(descriptor)
                        except BaseException as exc:
                            preserve_post_spawn_error(exc)
                            gate_proven = False
                    if gate_proven:
                        try:
                            _corroborate_worker_identity(worker_id, process_id)
                        except BaseException as exc:
                            preserve_post_spawn_error(exc)
                            gate_proven = False
                        else:
                            bootstrap_group_proven = True
                    if gate_proven:
                        try:
                            _read_bootstrap_ready(ready_reader)
                        except BaseException as exc:
                            preserve_post_spawn_error(exc)
                            gate_proven = False
                    try:
                        close_gate(ready_reader)
                    except BaseException as exc:
                        preserve_post_spawn_error(exc)
                        gate_proven = False
                    for descriptor in worker_descriptors:
                        if not close_owned(descriptor):
                            preserve_post_spawn_error(
                                OSError("isolated supervisor worker descriptor retirement failed")
                            )
                            gate_proven = False
                    if gate_proven:
                        admission_may_have_escaped = True
                        try:
                            _write_all(start_writer, _START)
                        except BaseException as exc:
                            preserve_post_spawn_error(exc)
                            gate_proven = False
                    try:
                        close_gate(start_writer)
                    except BaseException as exc:
                        preserve_post_spawn_error(exc)
                        gate_proven = False
                    if gate_proven:
                        _write_all(
                            status_descriptor,
                            f"STARTED {worker_id}\n".encode("ascii"),
                        )
                        close_owned(status_descriptor)
                        while worker_owned:
                            previous_mask = signal.pthread_sigmask(
                                signal.SIG_BLOCK,
                                {signal.SIGINT, signal.SIGTERM},
                            )
                            try:
                                wait_status = _wait_for_worker(worker_id)
                                if wait_status is not None:
                                    _record_reaped_status(worker_process, wait_status)
                                    worker_owned = False
                                    return_code = int(os.waitstatus_to_exitcode(wait_status))
                            finally:
                                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                            if worker_owned:
                                time.sleep(0.001)
                    else:
                        return_code = _PROTOCOL_FAILURE
                except BaseException as exc:
                    preserve_post_spawn_error(exc)
                finally:
                    for descriptor in tuple(gate_descriptors):
                        try:
                            close_gate(descriptor)
                        except BaseException as exc:
                            preserve_post_spawn_error(exc)
                    if worker_owned:
                        try:
                            terminate_owned_worker()
                        except BaseException as exc:
                            preserve_post_spawn_error(exc)
                if post_spawn_error is not None:
                    raise post_spawn_error
    except BaseException:
        return_code = _PROTOCOL_FAILURE
    finally:
        if (
            worker_process is not None
            and worker_native_child
            and worker_process.returncode is None
        ):
            try:
                _terminate_owned_worker_preserving_error(
                    worker_process.pid,
                    bootstrap_group_proven=bootstrap_group_proven,
                    admission_may_have_escaped=admission_may_have_escaped,
                    initiating_error=None,
                    process=worker_process,
                )
            except BaseException:
                return_code = _PROTOCOL_FAILURE
            finally:
                worker_owned = False
        for descriptor in (
            owner_descriptor,
            status_descriptor,
            control_descriptor,
            *worker_descriptors,
        ):
            close_owned(descriptor)
    if return_code == 0 and close_failed:
        return _PROTOCOL_FAILURE
    return return_code


def main() -> None:
    """Run the supervisor without exposing protocol details on stderr."""

    try:
        return_code = _run()
    except BaseException:
        return_code = _PROTOCOL_FAILURE
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
