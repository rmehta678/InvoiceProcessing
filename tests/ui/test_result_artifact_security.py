"""Result downloads are validated and sanitized, never raw file serving."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import multiprocessing
import os
import socket
import stat
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path

import httpx
import pytest
from factories import make_failed_case, make_succeeded_case
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ResultArtifactBinding, WorkflowStore
from invoice_agents.isolated_process import (
    IsolatedProcessCleanupError,
    IsolatedProcessResult,
    ProcessCancellation,
)
from invoice_agents.models import ErrorRecord
from invoice_agents.ui import routes
from invoice_agents.ui import server as ui_server


def _legacy_result(settings: Settings, case_id: str, marker: str) -> str:
    store = WorkflowStore(settings)
    result = store.load_result(case_id)
    assert result is not None and result.payment is not None
    payment = result.payment.model_copy(
        update={"error": f"cookie=session={marker}; preference={marker}-continuation"},
        deep=True,
    )
    error = ErrorRecord(
        category="PROVIDER",
        message=f"provider rejected sk-abcd\u2061efgh_{marker}",
        case_id=case_id,
        stop_reason="PROVIDER_REQUEST_FAILED",
    )
    legacy = result.model_copy(update={"payment": payment, "errors": [error]}, deep=True)
    encoded = legacy.model_dump_json(indent=2)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (encoded, case_id),
        )
        connection.commit()
    return encoded


def _binding_for_file(
    case_id: str,
    generation: int,
    target: Path,
) -> ResultArtifactBinding:
    payload = target.read_bytes()
    identity = target.stat(follow_symlinks=False)
    assert stat.S_ISREG(identity.st_mode)
    assert identity.st_nlink == 1
    return ResultArtifactBinding(
        case_id=case_id,
        execution_generation=generation,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_device=identity.st_dev,
        artifact_inode=identity.st_ino,
        artifact_file_type=stat.S_IFMT(identity.st_mode),
        artifact_size_bytes=identity.st_size,
    )


def _persist_exact_binding(
    settings: Settings,
    case_id: str,
    target: Path,
) -> ResultArtifactBinding:
    store = WorkflowStore(settings)
    result, generation, _binding = store.load_result_with_artifact_binding(case_id)
    assert result is not None
    binding = _binding_for_file(case_id, generation, target)
    store.save_result_artifact_binding(binding, result)
    assert store.load_result_artifact_binding(case_id) == binding
    return binding


def _synthetic_binding(case_id: str) -> ResultArtifactBinding:
    return ResultArtifactBinding(
        case_id=case_id,
        execution_generation=1,
        artifact_sha256="a" * 64,
        artifact_device=1,
        artifact_inode=1,
        artifact_file_type=stat.S_IFREG,
        artifact_size_bytes=1,
    )


def _install_worker_binding(
    monkeypatch: pytest.MonkeyPatch,
    binding: ResultArtifactBinding,
) -> None:
    monkeypatch.setattr(routes, "_RESULT_ARTIFACT_WORKER_BINDING", binding)


def _descriptor_cleanup_probe(
    connection: Connection,
    target: str,
    binding: ResultArtifactBinding,
    fault_index: int,
    timing: str,
) -> None:
    real_open = os.open
    real_close = os.close
    opened: list[int] = []
    attempts: list[int] = []
    replacements: list[int] = []

    def tracking_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def faulting_close(descriptor: int) -> None:
        attempt_index = len(attempts)
        attempts.append(descriptor)
        if attempt_index == fault_index:
            if timing == "after":
                real_close(descriptor)
                replacements.append(real_open(os.devnull, os.O_RDONLY))
            raise OSError(errno.EIO, "injected descriptor close failure")
        real_close(descriptor)

    routes.os.open = tracking_open
    routes.os.close = faulting_close
    routes._RESULT_ARTIFACT_WORKER_BINDING = binding
    error_type: str | None = None
    error_number: int | None = None
    try:
        routes._read_bounded_regular_file(Path(target))
    except BaseException as exc:
        error_type = type(exc).__name__
        error_number = exc.errno if isinstance(exc, OSError) else None
    alive: list[int] = []
    for descriptor in dict.fromkeys([*opened, *replacements]):
        try:
            os.fstat(descriptor)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
        else:
            alive.append(descriptor)
    connection.send(
        {
            "alive": alive,
            "attempts": attempts,
            "error_number": error_number,
            "error_type": error_type,
            "opened": opened,
            "replacements": replacements,
        }
    )
    connection.close()


def _run_descriptor_cleanup_probe(
    target: Path,
    binding: ResultArtifactBinding,
    *,
    fault_index: int = -1,
    timing: str = "before",
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_descriptor_cleanup_probe,
        args=(child_connection, str(target), binding, fault_index, timing),
    )
    try:
        process.start()
        child_connection.close()
        assert parent_connection.poll(10), "descriptor probe did not report a bounded outcome"
        report = parent_connection.recv()
        process.join(10)
        assert not process.is_alive(), "descriptor probe survived its ownership domain"
        assert process.exitcode == 0
        assert isinstance(report, dict)
        return report
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent_connection.close()
        child_connection.close()


def test_result_artifact_is_bounded_bound_to_database_and_newly_sanitized(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    marker = "round2-result-marker"
    raw = _legacy_result(settings, case_id, marker)
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    target.write_text(raw, encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["content-disposition"] == f'attachment; filename="{case_id}.json"'
    assert marker not in response.text
    payload = response.json()
    assert payload["case_id"] == case_id
    assert payload["payment"]["error"] == "cookie=[REDACTED]"
    assert payload["errors"][0]["message"] == "provider rejected [REDACTED]"
    assert response.text == WorkflowStore(settings).load_result(case_id).model_dump_json()


@pytest.mark.parametrize(
    "mutation",
    [
        "malformed",
        "duplicate",
        "mismatched_case",
        "mismatched_authority",
        "oversize",
        "noncanonical_datetime",
    ],
)
def test_result_artifact_rejects_invalid_or_unbound_content_without_echoing_it(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    mutation: str,
) -> None:
    case_id = make_succeeded_case(settings)
    marker = f"round2-{mutation}-marker"
    raw = _legacy_result(settings, case_id, marker)
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    if mutation == "malformed":
        candidate = f'{{"case_id":"{case_id}","marker":"{marker}"'
    elif mutation == "duplicate":
        candidate = f'{{"case_id":"{case_id}","marker":"{marker}",' + raw.lstrip()[1:]
    elif mutation == "mismatched_case":
        payload = json.loads(raw)
        payload["case_id"] = "case_other"
        candidate = json.dumps(payload)
    elif mutation == "mismatched_authority":
        payload = json.loads(raw)
        payload["stop_reason"] = marker
        candidate = json.dumps(payload)
    elif mutation == "oversize":
        candidate = raw + (" " * 1_048_577) + marker
    else:
        payload = json.loads(raw)
        payload["started_at"] = payload["started_at"].replace("Z", "+00:00")
        payload["errors"][0]["message"] = marker
        candidate = json.dumps(payload)
    target.write_text(candidate, encoding="utf-8")
    if mutation != "oversize":
        _persist_exact_binding(settings, case_id, target)
    else:
        target.write_text(raw, encoding="utf-8")
        _persist_exact_binding(settings, case_id, target)
        target.write_text(candidate, encoding="utf-8")

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert marker not in response.text
    expected_stop_reason = (
        "RESULT_ARTIFACT_BINDING_UNRESOLVED"
        if mutation == "oversize"
        else "RESULT_ARTIFACT_INVALID"
    )
    assert expected_stop_reason in response.text


@pytest.mark.parametrize("node_kind", ["dangling_symlink", "directory", "fifo"])
def test_result_artifact_rejects_hostile_filesystem_nodes(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    node_kind: str,
) -> None:
    case_id = make_succeeded_case(settings)
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    stored = WorkflowStore(settings).load_result(case_id)
    assert stored is not None
    target.write_text(stored.model_dump_json(), encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)
    target.unlink()
    if node_kind == "dangling_symlink":
        target.symlink_to(artifact_dir / "does-not-exist.json")
    elif node_kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"


def test_result_artifact_rejects_excessive_nesting_without_parser_crash(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    marker = "round3-deep-artifact-canary"
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    target.write_text(
        '{"case_id":"'
        + case_id
        + '","nested":'
        + ("[" * 16_000)
        + '"'
        + marker
        + '"'
        + ("]" * 16_000)
        + "}",
        encoding="utf-8",
    )
    _persist_exact_binding(settings, case_id, target)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert marker not in response.text
    assert "RESULT_ARTIFACT_INVALID" in response.text


def test_result_artifact_open_is_descriptor_first_and_nonblocking(
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round3-descriptor-canary")
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = (artifact_dir / f"{case_id}.json").absolute()
    target.write_text(raw, encoding="utf-8")
    binding = _binding_for_file(case_id, 1, target)
    _install_worker_binding(monkeypatch, binding)
    real_open = routes.os.open
    real_is_file = Path.is_file

    def checked_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(path).name == target.name:
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK
            assert dir_fd is not None
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def forbidden_is_file(path: Path) -> bool:
        if path.absolute() == target:
            raise AssertionError("result route performed a path precheck")
        return real_is_file(path)

    monkeypatch.setattr(routes.os, "open", checked_open)
    monkeypatch.setattr(Path, "is_file", forbidden_is_file)

    assert routes._read_bounded_regular_file(target) == raw.encode("utf-8")


@pytest.mark.parametrize("parent_kind", ["live_symlink", "dangling_symlink"])
def test_result_artifact_rejects_symlinked_results_parent_as_invalid(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    parent_kind: str,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round4-parent-symlink-canary")
    artifact_root = ui_workdir / "artifacts"
    results = artifact_root / "results"
    results.mkdir(parents=True)
    target = results / f"{case_id}.json"
    target.write_text(raw, encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)
    moved_results = artifact_root / "moved-results"
    results.rename(moved_results)
    if parent_kind == "live_symlink":
        outside = ui_workdir / "outside-results"
        outside.mkdir()
        (outside / f"{case_id}.json").write_text(raw, encoding="utf-8")
        results.symlink_to(outside, target_is_directory=True)
    else:
        results.symlink_to(ui_workdir / "missing-results", target_is_directory=True)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"


def test_result_artifact_reports_missing_parent_as_invalid_not_missing_file(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    stored = WorkflowStore(settings).load_result(case_id)
    assert stored is not None
    artifact_root = ui_workdir / "artifacts"
    artifact_dir = artifact_root / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    target.write_text(stored.model_dump_json(), encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)
    target.unlink()
    artifact_dir.rmdir()
    artifact_root.rmdir()

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"


def test_result_artifact_rejects_hardlinked_file(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round4-hardlink-canary")
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    outside = ui_workdir / "outside-result.json"
    target = artifact_dir / f"{case_id}.json"
    target.write_text(raw, encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)
    os.link(target, outside)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"


def test_result_artifact_rejects_unix_socket_without_blocking(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    stored = WorkflowStore(settings).load_result(case_id)
    assert stored is not None
    target.write_text(stored.model_dump_json(), encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)
    target.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(Path("artifacts/results") / target.name))

        response = client.get(f"/cases/{case_id}/result.json")
    finally:
        server.close()

    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"


def test_result_artifact_rejects_parent_swap_during_descriptor_walk(
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round4-parent-swap-canary")
    artifact_root = ui_workdir / "artifacts"
    results = artifact_root / "results"
    results.mkdir(parents=True)
    (results / f"{case_id}.json").write_text(raw, encoding="utf-8")
    binding = _binding_for_file(case_id, 1, results / f"{case_id}.json")
    _install_worker_binding(monkeypatch, binding)
    outside = ui_workdir / "outside-results"
    outside.mkdir()
    (outside / f"{case_id}.json").write_text(raw, encoding="utf-8")
    moved_results = artifact_root / "moved-results"
    real_open = routes.os.open
    swapped = False

    def swapping_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and Path(path).name == f"{case_id}.json":
            results.rename(moved_results)
            results.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(routes.os, "open", swapping_open)

    with pytest.raises(ValueError, match="parent changed"):
        routes._read_bounded_regular_file((results / f"{case_id}.json").absolute())


def test_result_artifact_parent_swap_cannot_turn_final_enoent_into_missing(
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    artifact_root = ui_workdir / "artifacts"
    results = artifact_root / "results"
    results.mkdir(parents=True)
    target = results / f"{case_id}.json"
    target.write_bytes(b"{}")
    binding = _binding_for_file(case_id, 1, target)
    _install_worker_binding(monkeypatch, binding)
    outside = ui_workdir / "outside-results"
    outside.mkdir()
    moved_results = artifact_root / "moved-results"
    real_open = routes.os.open
    swapped = False

    def swapping_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and Path(path).name == f"{case_id}.json":
            results.rename(moved_results)
            (moved_results / f"{case_id}.json").unlink()
            results.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(routes.os, "open", swapping_open)

    with pytest.raises(ValueError, match="parent changed"):
        routes._read_bounded_regular_file((results / f"{case_id}.json").absolute())


@pytest.mark.parametrize(
    "required_flag",
    ["O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"],
)
def test_result_artifact_fails_closed_when_required_open_flag_is_unavailable(
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    required_flag: str,
) -> None:
    case_id = make_succeeded_case(settings)
    monkeypatch.delattr(routes.os, required_flag)

    target = (ui_workdir / "artifacts" / "results" / f"{case_id}.json").absolute()
    _install_worker_binding(monkeypatch, _synthetic_binding(case_id))
    with pytest.raises(ValueError, match="required artifact open flag"):
        routes._read_bounded_regular_file(target)


@pytest.mark.asyncio
async def test_result_artifact_blocking_work_does_not_stall_the_event_loop(
    app: FastAPI,
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round4-worker-thread-canary")
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    target.write_text(raw, encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)
    original_read = routes._read_bounded_regular_file_isolated

    def slow_read(target: Path, binding: ResultArtifactBinding) -> bytes | None:
        time.sleep(0.15)
        return original_read(target, binding)

    monkeypatch.setattr(routes, "_read_bounded_regular_file_isolated", slow_read)
    ticked_at: list[float] = []
    began = 0.0

    async def watchdog() -> None:
        await asyncio.sleep(0.02)
        ticked_at.append(time.monotonic())

    async with app.router.lifespan_context(app):
        began = time.monotonic()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as async_client:
            response, _ = await asyncio.gather(
                async_client.get(f"/cases/{case_id}/result.json"),
                watchdog(),
            )

    assert response.status_code == 200
    assert ticked_at[0] - began < 0.1


def test_result_artifact_cleanup_attempts_every_owned_descriptor_and_contains_reuse(
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round5-cleanup-canary")
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = (artifact_dir / f"{case_id}.json").absolute()
    target.write_text(raw, encoding="utf-8")
    binding = _binding_for_file(case_id, 1, target)
    baseline = _run_descriptor_cleanup_probe(target, binding)
    opened = baseline["opened"]
    assert isinstance(opened, list)
    assert baseline["error_type"] is None
    descriptor_count = len(opened)
    assert descriptor_count >= 4
    baseline_parent_fds = len(os.listdir("/dev/fd"))

    for timing in ("before", "after"):
        for fault_index in range(descriptor_count):
            report = _run_descriptor_cleanup_probe(
                target,
                binding,
                fault_index=fault_index,
                timing=timing,
            )
            assert report["error_type"] == "OSError"
            assert report["error_number"] == errno.EIO
            assert report["attempts"] == report["opened"][::-1]
            assert report["alive"] == [report["attempts"][fault_index]]
            if timing == "after":
                assert report["replacements"] == [report["attempts"][fault_index]]
            else:
                assert report["replacements"] == []

    assert len(os.listdir("/dev/fd")) == baseline_parent_fds


@pytest.mark.parametrize(
    ("primary", "expected_error"),
    [("missing", "_ResultArtifactMissing"), ("directory", "ValueError")],
)
def test_result_artifact_primary_error_precedes_descriptor_cleanup_error(
    ui_workdir: Path,
    primary: str,
    expected_error: str,
) -> None:
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    case_id = f"case_round5_{primary}"
    target = (artifact_dir / f"{case_id}.json").absolute()
    target.write_bytes(b"{}")
    binding = _binding_for_file(case_id, 1, target)
    target.unlink()
    if primary == "directory":
        target.mkdir()

    report = _run_descriptor_cleanup_probe(
        target,
        binding,
        fault_index=0,
        timing="before",
    )

    assert report["error_type"] == expected_error
    assert report["attempts"] == report["opened"][::-1]


def test_result_artifact_reader_uses_only_the_public_isolated_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"case_id":"case_public_controller"}'
    cancellation = ProcessCancellation()
    calls: list[dict[str, object]] = []

    def controlled(**kwargs: object) -> IsolatedProcessResult:
        calls.append(kwargs)
        return IsolatedProcessResult(routes._RESULT_ARTIFACT_OK + raw, None)

    monkeypatch.setattr(routes, "run_isolated_process", controlled)
    target = (tmp_path / "case_public_controller.json").absolute()
    binding = _synthetic_binding("case_public_controller")

    assert (
        routes._read_bounded_regular_file_isolated(
            target,
            binding,
            cancel_requested=cancellation,
        )
        == raw
    )
    assert len(calls) == 1
    call = calls[0]
    assert set(call) == {
        "cancel_requested",
        "command",
        "env",
        "max_response_bytes",
        "request",
        "timeout_seconds",
    }
    assert call["cancel_requested"] is cancellation
    assert call["env"] == routes.sanitized_worker_environment()
    request = call["request"]
    assert isinstance(request, bytes)
    assert routes._decode_result_artifact_worker_request(request) == target
    assert binding == routes._RESULT_ARTIFACT_WORKER_BINDING
    command = call["command"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert Path(command[0]).is_absolute()
    assert Path(command[0]).is_file()
    assert os.access(command[0], os.X_OK)
    assert command[1] == "-I"
    assert Path(command[2]).is_absolute()
    assert Path(command[2]).name == "result_artifact_worker.py"


@pytest.mark.parametrize(
    "mutated",
    [
        b'{"path":"relative.json","protocol_version":1}',
        b'{"path":"/tmp/case.json","protocol_version":true}',
        b'{"path":"/tmp/case.json","protocol_version":1,"unexpected":null}',
        b'{ "path":"/tmp/case.json","protocol_version":1}',
        b'{"path":"/tmp/case.json","path":"/tmp/other.json","protocol_version":1}',
    ],
)
def test_result_artifact_worker_request_is_strict_and_canonical(mutated: bytes) -> None:
    with pytest.raises(ValueError):
        routes._decode_result_artifact_worker_request(mutated)


def test_result_artifact_root_is_captured_before_concurrent_cwd_changes(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round5-cwd-canary")
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    target.write_text(raw, encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)
    attacker_cwd = tmp_path / "attacker-cwd"
    attacker_cwd.mkdir()

    monkeypatch.chdir(attacker_cwd)
    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 200
    assert response.json()["case_id"] == case_id


@pytest.mark.parametrize(
    ("outcome", "expected_code"),
    [
        (IsolatedProcessResult(None, "start"), "RESULT_ARTIFACT_WORKER_CRASHED"),
        (IsolatedProcessResult(None, "timeout"), "RESULT_ARTIFACT_WORKER_TIMED_OUT"),
        (IsolatedProcessResult(None, "cancelled"), "RESULT_ARTIFACT_WORKER_CANCELLED"),
        (IsolatedProcessResult(None, "crash"), "RESULT_ARTIFACT_WORKER_CRASHED"),
        (IsolatedProcessResult(None, "protocol"), "RESULT_ARTIFACT_WORKER_PROTOCOL_INVALID"),
        (IsolatedProcessResult(None, None), "RESULT_ARTIFACT_WORKER_PROTOCOL_INVALID"),
    ],
)
def test_result_artifact_process_failure_taxonomy_never_admits_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: IsolatedProcessResult,
    expected_code: str,
) -> None:
    marker = "artifact-worker-untrusted-output"
    selected = IsolatedProcessResult(marker.encode(), outcome.failure)
    monkeypatch.setattr(routes, "run_isolated_process", lambda **_kwargs: selected)
    target = (tmp_path / "case_failure.json").absolute()
    binding = _synthetic_binding("case_failure")

    with pytest.raises(routes._ResultArtifactWorkerError) as failure:
        routes._read_bounded_regular_file_isolated(target, binding)

    assert failure.value.error_code == expected_code
    assert marker not in str(failure.value)


def test_result_artifact_cleanup_ambiguity_poison_blocks_later_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes, "_RESULT_ARTIFACT_OWNERSHIP_POISONED", False)
    calls = 0

    def cleanup_ambiguous(**_kwargs: object) -> IsolatedProcessResult:
        nonlocal calls
        calls += 1
        raise IsolatedProcessCleanupError

    monkeypatch.setattr(routes, "run_isolated_process", cleanup_ambiguous)
    target = (tmp_path / "case_cleanup.json").absolute()
    binding = _synthetic_binding("case_cleanup")

    with pytest.raises(routes._ResultArtifactWorkerError) as first:
        routes._read_bounded_regular_file_isolated(target, binding)
    assert first.value.error_code == "RESULT_ARTIFACT_WORKER_CLEANUP_FAILED"

    with pytest.raises(routes._ResultArtifactWorkerError) as successor:
        routes._read_bounded_regular_file_isolated(target, binding)
    assert successor.value.error_code == "RESULT_ARTIFACT_OWNERSHIP_UNRESOLVED"
    assert calls == 1


@pytest.mark.parametrize(
    ("response", "expected", "expected_code"),
    [
        (routes._RESULT_ARTIFACT_OK + b"{}", b"{}", None),
        (routes._RESULT_ARTIFACT_MISSING, None, None),
        (routes._RESULT_ARTIFACT_INVALID, None, None),
        (b"artifact-untrusted-frame", None, "RESULT_ARTIFACT_WORKER_PROTOCOL_INVALID"),
        (
            routes._RESULT_ARTIFACT_OK + b"x" * (routes.RESULT_ARTIFACT_MAX_BYTES + 1),
            None,
            "RESULT_ARTIFACT_WORKER_PROTOCOL_INVALID",
        ),
    ],
)
def test_result_artifact_worker_frames_are_strict_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: bytes,
    expected: bytes | None,
    expected_code: str | None,
) -> None:
    monkeypatch.setattr(
        routes,
        "run_isolated_process",
        lambda **_kwargs: IsolatedProcessResult(response, None),
    )
    target = (tmp_path / "case_frame.json").absolute()
    binding = _synthetic_binding("case_frame")

    if expected_code is not None:
        with pytest.raises(routes._ResultArtifactWorkerError) as failure:
            routes._read_bounded_regular_file_isolated(target, binding)
        assert failure.value.error_code == expected_code
    else:
        assert routes._read_bounded_regular_file_isolated(target, binding) == expected


def test_result_artifact_failure_response_is_sanitized_and_specific(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "artifact-response-secret")
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    target.write_text(raw, encoding="utf-8")
    _persist_exact_binding(settings, case_id, target)
    marker = "artifact-supervisor-private-error"
    monkeypatch.setattr(
        routes,
        "run_isolated_process",
        lambda **_kwargs: IsolatedProcessResult(marker.encode(), "protocol"),
    )

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert "RESULT_ARTIFACT_WORKER_PROTOCOL_INVALID" in response.text
    assert marker not in response.text


def test_result_artifact_preexisting_cancellation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = ProcessCancellation()
    cancellation.set()

    def cancelled_controller(**kwargs: object) -> IsolatedProcessResult:
        assert kwargs["cancel_requested"] is cancellation
        return IsolatedProcessResult(None, "cancelled")

    monkeypatch.setattr(routes, "run_isolated_process", cancelled_controller)
    target = (tmp_path / "case_cancelled.json").absolute()
    binding = _synthetic_binding("case_cancelled")

    with pytest.raises(routes._ResultArtifactWorkerError) as failure:
        routes._read_bounded_regular_file_isolated(
            target,
            binding,
            cancel_requested=cancellation,
        )
    assert failure.value.error_code == "RESULT_ARTIFACT_WORKER_CANCELLED"


def test_result_artifact_rejects_invalid_cancellation_owner_before_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def controller_not_admitted(**_kwargs: object) -> IsolatedProcessResult:
        nonlocal calls
        calls += 1
        return IsolatedProcessResult(routes._RESULT_ARTIFACT_OK + b"{}", None)

    monkeypatch.setattr(routes, "run_isolated_process", controller_not_admitted)
    target = (tmp_path / "case_invalid_cancellation.json").absolute()
    binding = _synthetic_binding("case_invalid_cancellation")

    with pytest.raises(ValueError, match="cancellation owner"):
        routes._read_bounded_regular_file_isolated(
            target,
            binding,
            cancel_requested=object(),  # type: ignore[arg-type]
        )

    assert calls == 0


def test_result_artifact_rejects_noncanonical_controller_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes,
        "run_isolated_process",
        lambda **_kwargs: object(),
    )
    target = (tmp_path / "case_invalid_controller_result.json").absolute()
    binding = _synthetic_binding("case_invalid_controller_result")

    with pytest.raises(routes._ResultArtifactWorkerError) as failure:
        routes._read_bounded_regular_file_isolated(target, binding)

    assert failure.value.error_code == "RESULT_ARTIFACT_WORKER_PROTOCOL_INVALID"


def test_multiple_apps_keep_distinct_artifact_roots_across_cwd_mutation(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_case_id = make_succeeded_case(settings)
    second_case_id = make_failed_case(settings, "invoice_1002.txt")
    first_root = tmp_path / "first-app"
    second_root = tmp_path / "second-app"
    attacker_root = tmp_path / "attacker"
    for root in (first_root, second_root, attacker_root):
        root.mkdir()
    for root, case_id in (
        (first_root, first_case_id),
        (second_root, second_case_id),
    ):
        stored = WorkflowStore(settings).load_result(case_id)
        assert stored is not None
        artifact_dir = root / "artifacts" / "results"
        artifact_dir.mkdir(parents=True)
        target = artifact_dir / f"{case_id}.json"
        target.write_text(stored.model_dump_json(), encoding="utf-8")
        _persist_exact_binding(settings, case_id, target)

    settings.ui_session_secret = SecretStr("test-only-result-artifact-session-secret-000000000000")
    monkeypatch.chdir(first_root)
    first_app = ui_server.create_app(settings, allowed_hosts=("testserver",))
    monkeypatch.chdir(second_root)
    second_app = ui_server.create_app(settings, allowed_hosts=("testserver",))
    monkeypatch.chdir(attacker_root)

    with TestClient(first_app) as first_client, TestClient(second_app) as second_client:
        first_response = first_client.get(f"/cases/{first_case_id}/result.json")
        second_response = second_client.get(f"/cases/{second_case_id}/result.json")

    assert first_app.state.result_artifact_root == first_root / "artifacts" / "results"
    assert second_app.state.result_artifact_root == second_root / "artifacts" / "results"
    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["case_id"] == first_case_id
    assert second_response.json()["case_id"] == second_case_id
