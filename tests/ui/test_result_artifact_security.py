"""Result downloads are validated and sanitized, never raw file serving."""

from __future__ import annotations

import asyncio
import errno
import json
import multiprocessing
import os
import socket
import time
from multiprocessing.connection import Connection
from pathlib import Path

import httpx
import pytest
from factories import make_succeeded_case
from fastapi import FastAPI
from fastapi.testclient import TestClient

from invoice_agents import isolated_process
from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import WorkflowStore
from invoice_agents.models import ErrorRecord
from invoice_agents.ui import routes


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


def _descriptor_cleanup_probe(
    connection: Connection,
    target: str,
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
    *,
    fault_index: int = -1,
    timing: str = "before",
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_descriptor_cleanup_probe,
        args=(child_connection, str(target), fault_index, timing),
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
    (artifact_dir / f"{case_id}.json").write_text(raw, encoding="utf-8")

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

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert marker not in response.text
    assert "RESULT_ARTIFACT_INVALID" in response.text


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
    if node_kind == "dangling_symlink":
        target.symlink_to(artifact_dir / "does-not-exist.json")
    elif node_kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert "RESULT_ARTIFACT_INVALID" in response.text


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
    artifact_root.mkdir()
    if parent_kind == "live_symlink":
        outside = ui_workdir / "outside-results"
        outside.mkdir()
        (outside / f"{case_id}.json").write_text(raw, encoding="utf-8")
        (artifact_root / "results").symlink_to(outside, target_is_directory=True)
    else:
        (artifact_root / "results").symlink_to(
            ui_workdir / "missing-results", target_is_directory=True
        )

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert "RESULT_ARTIFACT_INVALID" in response.text


def test_result_artifact_reports_missing_parent_as_invalid_not_missing_file(
    client: TestClient,
    settings: Settings,
) -> None:
    case_id = make_succeeded_case(settings)

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert "RESULT_ARTIFACT_INVALID" in response.text


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
    outside.write_text(raw, encoding="utf-8")
    os.link(outside, artifact_dir / f"{case_id}.json")

    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 409
    assert "RESULT_ARTIFACT_INVALID" in response.text


def test_result_artifact_rejects_unix_socket_without_blocking(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
) -> None:
    case_id = make_succeeded_case(settings)
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    target = artifact_dir / f"{case_id}.json"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(Path("artifacts/results") / target.name))

        response = client.get(f"/cases/{case_id}/result.json")
    finally:
        server.close()

    assert response.status_code == 409
    assert "RESULT_ARTIFACT_INVALID" in response.text


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
    (artifact_dir / f"{case_id}.json").write_text(raw, encoding="utf-8")
    original_read = routes._read_bounded_regular_file_isolated

    def slow_read(target: Path) -> bytes:
        time.sleep(0.15)
        return original_read(target)

    monkeypatch.setattr(routes, "_read_bounded_regular_file_isolated", slow_read)
    ticked_at: list[float] = []
    began = time.monotonic()

    async def watchdog() -> None:
        await asyncio.sleep(0.02)
        ticked_at.append(time.monotonic())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
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
    baseline = _run_descriptor_cleanup_probe(target)
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
    target = (artifact_dir / f"case_round5_{primary}.json").absolute()
    if primary == "directory":
        target.mkdir()

    report = _run_descriptor_cleanup_probe(target, fault_index=0, timing="before")

    assert report["error_type"] == expected_error
    assert report["attempts"] == report["opened"][::-1]


def test_result_artifact_requests_use_reaped_descriptor_ownership_processes(
    client: TestClient,
    settings: Settings,
    ui_workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    raw = _legacy_result(settings, case_id, "round5-process-canary")
    artifact_dir = ui_workdir / "artifacts" / "results"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / f"{case_id}.json").write_text(raw, encoding="utf-8")
    real_popen = isolated_process.subprocess.Popen
    artifact_processes: list[object] = []

    def capturing_popen(*args: object, **kwargs: object) -> object:
        process = real_popen(*args, **kwargs)
        command = args[0] if args else kwargs.get("args")
        if "invoice_agents.ui.result_artifact_worker" in str(command):
            artifact_processes.append(process)
        return process

    monkeypatch.setattr(isolated_process.subprocess, "Popen", capturing_popen)
    baseline_parent_fds = len(os.listdir("/dev/fd"))

    responses = [client.get(f"/cases/{case_id}/result.json") for _ in range(4)]

    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    assert len(artifact_processes) == 4
    assert all(process.poll() is not None for process in artifact_processes)  # type: ignore[attr-defined]
    assert len(os.listdir("/dev/fd")) == baseline_parent_fds


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
    (artifact_dir / f"{case_id}.json").write_text(raw, encoding="utf-8")
    attacker_cwd = tmp_path / "attacker-cwd"
    attacker_cwd.mkdir()

    monkeypatch.chdir(attacker_cwd)
    response = client.get(f"/cases/{case_id}/result.json")

    assert response.status_code == 200
    assert response.json()["case_id"] == case_id
