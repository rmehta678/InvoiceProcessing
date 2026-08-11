"""Fail-closed coupling between filesystem publication and its SQLite binding."""

from __future__ import annotations

import asyncio
import errno
import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invoice_agents import orchestration
from invoice_agents.config import Settings
from invoice_agents.db.store import ResultArtifactBinding, WorkflowStore
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.models import CaseResult, CaseStatus
from invoice_agents.source_store import snapshot_source
from invoice_agents.ui import routes as ui_routes
from invoice_agents.ui.server import create_app

SOURCE_PATH = Path(__file__).resolve().parents[2] / "data/invoices/invoice_1001.txt"
STARTED_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _terminal_case(settings: Settings) -> tuple[WorkflowStore, CaseResult, int]:
    case_id = "case_binding_publication"
    source = snapshot_source(
        SOURCE_PATH,
        settings.source_archive_dir,
        max_bytes=10_485_760,
    )
    store = WorkflowStore(settings)
    store.register_source(source)
    store.create_case(case_id, source, STARTED_AT)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    result = CaseResult(
        case_id=case_id,
        source_id=source.source_id,
        status=CaseStatus.FAILED,
        stop_reason="BINDING_PUBLICATION_TEST",
        started_at=STARTED_AT,
        finished_at=STARTED_AT + timedelta(seconds=1),
    )
    store.finish_case(result, claim)
    loaded_result, generation, binding = store.load_result_with_artifact_binding(case_id)
    assert loaded_result is not None
    assert generation >= 1
    assert binding is None
    return store, loaded_result, generation


def _rollback_paths(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.rollback-*"))


def _instrument_directory_durability(
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
) -> list[tuple[str, Path | None]]:
    original_open = os.open
    original_fsync = os.fsync
    original_link = os.link
    original_replace = os.replace
    original_unlink = os.unlink
    directory_descriptors: set[int] = set()
    events: list[tuple[str, Path | None]] = []

    def resolve(path: os.PathLike[str] | str, dir_fd: int | None) -> Path:
        return Path(path) if dir_fd is None else directory / Path(path)

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if resolve(path, dir_fd) == directory:
            directory_descriptors.add(descriptor)
        return descriptor

    def observe_fsync(descriptor: int) -> None:
        original_fsync(descriptor)
        if descriptor in directory_descriptors:
            events.append(("directory_fsync", None))

    def observe_unlink(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        resolved = resolve(path, dir_fd)
        original_unlink(path, dir_fd=dir_fd)
        events.append(("unlink", resolved))

    def observe_link(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        events.append(("link", resolve(destination, dst_dir_fd)))

    def observe_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        events.append(("replace", resolve(destination, dst_dir_fd)))

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(os, "link", observe_link)
    monkeypatch.setattr(os, "replace", observe_replace)
    monkeypatch.setattr(os, "unlink", observe_unlink)
    return events


def _assert_directory_fsync_after_final_namespace_mutation(
    events: list[tuple[str, Path | None]],
) -> None:
    mutations = [
        index
        for index, (event, _path) in enumerate(events)
        if event in {"link", "replace", "unlink"}
    ]
    assert mutations
    assert ("directory_fsync", None) in events[max(mutations) + 1 :]


def _binding_matches_payload(
    binding: ResultArtifactBinding,
    *,
    payload: bytes,
    generation: int,
) -> None:
    assert binding.execution_generation == generation
    assert binding.artifact_sha256 == sha256(payload).hexdigest()
    assert binding.artifact_size_bytes == len(payload)


@dataclass(slots=True)
class _DirectoryCapabilityObservations:
    events: list[str] = field(default_factory=list)
    attempted_flags: list[int] = field(default_factory=list)
    descriptors: list[int] = field(default_factory=list)
    descriptor_events: list[tuple[int, int]] = field(default_factory=list)
    production_fstats: list[tuple[int, int, int, int]] = field(default_factory=list)


def _install_result_directory_substitution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    intended: Path,
    unintended: Path,
    displaced: Path,
    timing: str,
) -> _DirectoryCapabilityObservations:
    """Swap the final directory name at one exact capability-open boundary."""

    original_open = os.open
    original_fstat = os.fstat
    parent_identity = intended.parent.stat()
    observations = _DirectoryCapabilityObservations()
    substituted = False

    def opens_final_directory(
        path: os.PathLike[str] | str,
        flags: int,
        dir_fd: int | None,
    ) -> bool:
        if not flags & os.O_DIRECTORY:
            return False
        if dir_fd is None:
            return Path(path) == intended
        if Path(path) != Path(intended.name):
            return False
        opened_parent = original_fstat(dir_fd)
        return (opened_parent.st_dev, opened_parent.st_ino) == (
            parent_identity.st_dev,
            parent_identity.st_ino,
        )

    def substitute() -> None:
        nonlocal substituted
        assert not substituted
        intended.rename(displaced)
        if timing == "before-first-symlink":
            intended.symlink_to(unintended, target_is_directory=True)
        else:
            unintended.rename(intended)
        substituted = True
        observations.events.append("substituted")

    def adversarial_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        is_final_directory = opens_final_directory(path, flags, dir_fd)
        if is_final_directory:
            observations.attempted_flags.append(flags)
            observations.events.append("attempt-final-directory")
        if is_final_directory and timing == "before-first-symlink" and not substituted:
            substitute()
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if is_final_directory:
            observations.descriptors.append(descriptor)
            observations.events.append("opened-final-directory")
            observations.descriptor_events.append((descriptor, len(observations.events) - 1))
            if timing == "after-first-directory" and not substituted:
                substitute()
        return descriptor

    def observe_fstat(descriptor: int) -> os.stat_result:
        identity = original_fstat(descriptor)
        if descriptor in observations.descriptors:
            observations.production_fstats.append(
                (
                    descriptor,
                    identity.st_dev,
                    identity.st_ino,
                    stat.S_IFMT(identity.st_mode),
                )
            )
        return identity

    monkeypatch.setattr(os, "open", adversarial_open)
    monkeypatch.setattr(os, "fstat", observe_fstat)
    return observations


def _directory_inventory(
    directory: Path,
) -> dict[str, tuple[int, int, int, int, str | None, str | None]]:
    inventory: dict[str, tuple[int, int, int, int, str | None, str | None]] = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            identity = entry.stat(follow_symlinks=False)
            file_type = stat.S_IFMT(identity.st_mode)
            inventory[entry.name] = (
                identity.st_dev,
                identity.st_ino,
                file_type,
                identity.st_size,
                os.readlink(entry.path) if stat.S_ISLNK(identity.st_mode) else None,
                sha256(Path(entry.path).read_bytes()).hexdigest()
                if stat.S_ISREG(identity.st_mode)
                else None,
            )
    return inventory


def _assert_directory_capability_attempts(
    observations: _DirectoryCapabilityObservations,
    *,
    timing: str,
) -> None:
    assert observations.attempted_flags
    assert all(flags & os.O_DIRECTORY for flags in observations.attempted_flags)
    assert all(flags & os.O_NOFOLLOW for flags in observations.attempted_flags)
    if timing == "before-first-symlink":
        assert observations.events[:2] == ["attempt-final-directory", "substituted"]
        return
    substitution = observations.events.index("substituted")
    assert "attempt-final-directory" in observations.events[:substitution]
    assert "attempt-final-directory" in observations.events[substitution + 1 :]
    before = [
        descriptor
        for descriptor, event_index in observations.descriptor_events
        if event_index < substitution
    ]
    after = [
        descriptor
        for descriptor, event_index in observations.descriptor_events
        if event_index > substitution
    ]
    assert before
    assert after
    first, second = before[-1], after[0]
    fstats = {
        descriptor: (device, inode, file_type)
        for descriptor, device, inode, file_type in observations.production_fstats
    }
    assert first in fstats
    assert second in fstats
    assert fstats[first][2] == stat.S_IFDIR
    assert fstats[second][2] == stat.S_IFDIR
    assert fstats[first][:2] != fstats[second][:2]


def _assert_worker_directory_capability_attempt(
    observations: _DirectoryCapabilityObservations,
    *,
    timing: str,
) -> None:
    """The worker holds one descriptor and revalidates its parent relation at EOF."""

    assert observations.attempted_flags
    assert all(flags & os.O_DIRECTORY for flags in observations.attempted_flags)
    assert all(flags & os.O_NOFOLLOW for flags in observations.attempted_flags)
    if timing == "before-first-symlink":
        assert observations.events[:2] == ["attempt-final-directory", "substituted"]
        assert observations.descriptors == []
        return
    assert observations.events == [
        "attempt-final-directory",
        "opened-final-directory",
        "substituted",
    ]
    assert len(observations.descriptors) == 1
    descriptor = observations.descriptors[0]
    assert len(observations.production_fstats) == 1
    observed_descriptor, _device, _inode, file_type = observations.production_fstats[0]
    assert observed_descriptor == descriptor
    assert file_type == stat.S_IFDIR


@pytest.mark.parametrize(
    "failure_type",
    [OSError, KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_pre_action_binding_failure_restores_exact_prior_artifact(
    failure_type: type[BaseException],
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / f"{result.case_id}.json"
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    prior_identity = target.stat()
    failure = failure_type("pre-action private binding sentinel")
    events = _instrument_directory_durability(monkeypatch, output)

    def fail_before_action(
        _store: WorkflowStore,
        _binding: ResultArtifactBinding,
        _result: CaseResult,
    ) -> None:
        raise failure

    monkeypatch.setattr(WorkflowStore, "save_result_artifact_binding", fail_before_action)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(BaseException) as excinfo:
        orchestration._write_result_for_generation(store, result, generation)

    assert excinfo.value is failure
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    restored = target.stat()
    assert target.read_bytes() == prior
    assert (restored.st_dev, restored.st_ino) == (prior_identity.st_dev, prior_identity.st_ino)
    assert store.load_result_artifact_binding(result.case_id) is None
    assert not target.with_name(f"{target.name}.tmp").exists()
    assert _rollback_paths(target) == []
    assert set(output.iterdir()) == {target}
    _assert_directory_fsync_after_final_namespace_mutation(events)
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        response = client.get(f"/cases/{result.case_id}/result.json")
    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"
    assert result.model_dump_json(indent=2).encode("utf-8") not in response.content


def test_pre_action_binding_failure_removes_new_candidate_durably(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    target = tmp_path / "artifacts" / "results" / f"{result.case_id}.json"
    failure = OSError("pre-action private binding sentinel")
    events = _instrument_directory_durability(monkeypatch, target.parent)

    monkeypatch.setattr(
        WorkflowStore,
        "save_result_artifact_binding",
        lambda _store, _binding, _result: (_ for _ in ()).throw(failure),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(OSError) as excinfo:
        orchestration._write_result_for_generation(store, result, generation)

    assert excinfo.value is failure
    assert not target.exists()
    assert not target.with_name(f"{target.name}.tmp").exists()
    assert _rollback_paths(target) == []
    assert list(target.parent.iterdir()) == []
    target_unlink = events.index(("unlink", target))
    assert ("directory_fsync", None) in events[target_unlink + 1 :]


@pytest.mark.parametrize(
    "failure_type",
    [OSError, KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_post_action_binding_fault_uses_exact_independent_readback(
    failure_type: type[BaseException],
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    original_save = WorkflowStore.save_result_artifact_binding
    failure = failure_type("post-action private binding sentinel")

    def save_then_fail(
        active_store: WorkflowStore,
        binding: ResultArtifactBinding,
        stored_result: CaseResult,
    ) -> None:
        original_save(active_store, binding, stored_result)
        raise failure

    monkeypatch.setattr(WorkflowStore, "save_result_artifact_binding", save_then_fail)
    monkeypatch.chdir(tmp_path)
    payload = result.model_dump_json(indent=2).encode("utf-8")

    raised: BaseException | None = None
    try:
        orchestration._write_result_for_generation(store, result, generation)
    except BaseException as exc:
        raised = exc

    if isinstance(failure, Exception):
        assert raised is None
    else:
        assert raised is failure
        assert raised.__cause__ is None
        assert raised.__context__ is None
    target = tmp_path / "artifacts" / "results" / f"{result.case_id}.json"
    assert target.read_bytes() == payload
    identity = os.lstat(target)
    binding = store.load_result_artifact_binding(result.case_id)
    assert binding is not None
    _binding_matches_payload(binding, payload=payload, generation=generation)
    assert binding.artifact_device == identity.st_dev
    assert binding.artifact_inode == identity.st_ino
    assert binding.artifact_file_type == stat.S_IFMT(identity.st_mode) == stat.S_IFREG
    assert _rollback_paths(target) == []
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        response = client.get(f"/cases/{result.case_id}/result.json")
    assert response.status_code == 200
    assert CaseResult.model_validate_json(response.content) == result
    assert response.content == result.model_dump_json().encode("utf-8")


@pytest.mark.parametrize(
    "failure_type",
    [OSError, KeyboardInterrupt, SystemExit, asyncio.CancelledError],
)
def test_ambiguous_binding_readback_preserves_candidate_and_prior_evidence(
    failure_type: type[BaseException],
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / f"{result.case_id}.json"
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    prior_identity = target.stat()
    failure = failure_type("binding action private sentinel")
    events = _instrument_directory_durability(monkeypatch, output)
    original_binding_loader = WorkflowStore.load_result_with_artifact_binding

    monkeypatch.setattr(
        WorkflowStore,
        "save_result_artifact_binding",
        lambda _store, _binding, _result: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        WorkflowStore,
        "load_result_with_artifact_binding",
        lambda _store, _case_id: (_ for _ in ()).throw(
            OSError("binding readback private sentinel")
        ),
    )
    monkeypatch.chdir(tmp_path)

    raised: BaseException | None = None
    try:
        orchestration._write_result_for_generation(store, result, generation)
    except BaseException as exc:
        raised = exc

    if isinstance(failure, Exception):
        assert isinstance(raised, InvoiceAgentsError)
        assert raised.stop_reason == "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED"
        assert "private" not in raised.message
    else:
        assert raised is failure
    assert raised is not None
    assert raised.__cause__ is None
    assert raised.__context__ is None
    candidate = result.model_dump_json(indent=2).encode("utf-8")
    assert target.read_bytes() == candidate
    rollback = _rollback_paths(target)
    assert len(rollback) == 1
    rollback_identity = rollback[0].stat()
    assert rollback[0].read_bytes() == prior
    assert (rollback_identity.st_dev, rollback_identity.st_ino) == (
        prior_identity.st_dev,
        prior_identity.st_ino,
    )
    _assert_directory_fsync_after_final_namespace_mutation(events)
    assert set(output.iterdir()) == {target, rollback[0]}
    assert not target.with_name(f"{target.name}.tmp").exists()
    monkeypatch.setattr(
        WorkflowStore,
        "load_result_with_artifact_binding",
        original_binding_loader,
    )
    with TestClient(
        create_app(settings),
        base_url="http://127.0.0.1:8787",
        raise_server_exceptions=False,
    ) as client:
        response = client.get(f"/cases/{result.case_id}/result.json")
    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"
    assert result.model_dump_json(indent=2).encode("utf-8") not in response.content


def test_binding_absence_plus_filesystem_rollback_fault_preserves_both_evidence_files(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / f"{result.case_id}.json"
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    events = _instrument_directory_durability(monkeypatch, output)
    original_replace = os.replace
    rollback_attempts = 0

    def fail_rollback_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal rollback_attempts
        source_path = Path(source) if src_dir_fd is None else output / Path(source)
        destination_path = Path(destination) if dst_dir_fd is None else output / Path(destination)
        if source_path.name.startswith(f".{target.name}.rollback-"):
            assert destination_path == target
            rollback_attempts += 1
            raise OSError("rollback private binding sentinel")
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(
        WorkflowStore,
        "save_result_artifact_binding",
        lambda _store, _binding, _result: (_ for _ in ()).throw(
            OSError("binding action private sentinel")
        ),
    )
    monkeypatch.setattr(os, "replace", fail_rollback_replace)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        orchestration._write_result_for_generation(store, result, generation)

    assert rollback_attempts == 1
    assert excinfo.value.stop_reason == "ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert "private" not in excinfo.value.message
    assert target.read_bytes() == result.model_dump_json(indent=2).encode("utf-8")
    rollback = _rollback_paths(target)
    assert len(rollback) == 1
    assert rollback[0].read_bytes() == prior
    assert set(output.iterdir()) == {target, rollback[0]}
    assert not target.with_name(f"{target.name}.tmp").exists()
    _assert_directory_fsync_after_final_namespace_mutation(events)
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        response = client.get(f"/cases/{result.case_id}/result.json")
    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"
    assert result.model_dump_json(indent=2).encode("utf-8") not in response.content


def test_mismatched_binding_readback_preserves_both_files_and_fails_closed(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / f"{result.case_id}.json"
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    events = _instrument_directory_durability(monkeypatch, output)
    mismatched = ResultArtifactBinding(
        case_id=result.case_id,
        execution_generation=generation,
        artifact_sha256="f" * 64,
        artifact_device=1,
        artifact_inode=1,
        artifact_file_type=32_768,
        artifact_size_bytes=1,
    )

    monkeypatch.setattr(
        WorkflowStore,
        "save_result_artifact_binding",
        lambda _store, _binding, _result: (_ for _ in ()).throw(
            OSError("binding action private sentinel")
        ),
    )
    monkeypatch.setattr(
        WorkflowStore,
        "load_result_with_artifact_binding",
        lambda _store, _case_id: (result, generation, mismatched),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        orchestration._write_result_for_generation(store, result, generation)

    assert excinfo.value.stop_reason == "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert "private" not in excinfo.value.message
    assert target.read_bytes() == result.model_dump_json(indent=2).encode("utf-8")
    rollback = _rollback_paths(target)
    assert len(rollback) == 1
    assert rollback[0].read_bytes() == prior
    _assert_directory_fsync_after_final_namespace_mutation(events)
    assert set(output.iterdir()) == {target, rollback[0]}
    assert not target.with_name(f"{target.name}.tmp").exists()
    with TestClient(
        create_app(settings),
        base_url="http://127.0.0.1:8787",
        raise_server_exceptions=False,
    ) as client:
        response = client.get(f"/cases/{result.case_id}/result.json")
    assert response.status_code == 409
    assert response.json()["stop_reason"] == "RESULT_ARTIFACT_BINDING_UNRESOLVED"
    assert result.model_dump_json(indent=2).encode("utf-8") not in response.content


@pytest.mark.parametrize(
    "timing",
    ["before-first-symlink", "after-first-directory"],
)
def test_publication_rejects_result_directory_entry_substitution_before_binding(
    timing: str,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    intended = tmp_path / "artifacts" / "results"
    intended.mkdir(parents=True)
    unintended = tmp_path / "unintended-results"
    unintended.mkdir()
    unintended_identity = unintended.stat()
    intended_inventory = _directory_inventory(intended)
    unintended_inventory = _directory_inventory(unintended)
    displaced = tmp_path / "artifacts" / "displaced-results"
    observations = _install_result_directory_substitution(
        monkeypatch,
        intended=intended,
        unintended=unintended,
        displaced=displaced,
        timing=timing,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        orchestration._write_result_for_generation(store, result, generation)

    assert excinfo.value.stop_reason == "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    _assert_directory_capability_attempts(observations, timing=timing)
    assert store.load_result_artifact_binding(result.case_id) is None
    assert _directory_inventory(displaced) == intended_inventory
    if timing == "before-first-symlink":
        assert intended.is_symlink()
        assert intended.resolve() == unintended
        assert _directory_inventory(unintended) == unintended_inventory
    else:
        substituted_identity = intended.stat()
        assert (substituted_identity.st_dev, substituted_identity.st_ino) == (
            unintended_identity.st_dev,
            unintended_identity.st_ino,
        )
        assert _directory_inventory(intended) == unintended_inventory


@pytest.mark.parametrize(
    "timing",
    ["before-first-symlink", "after-first-directory"],
)
def test_result_worker_and_route_reject_directory_entry_substitution_before_serving(
    timing: str,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    monkeypatch.chdir(tmp_path)
    target = orchestration._write_result_for_generation(store, result, generation)
    payload = target.read_bytes()
    intended = target.parent
    unintended = tmp_path / "unintended-results"
    unintended.mkdir()
    unintended_target = unintended / target.name
    unintended_target.write_bytes(payload)
    unintended_identity = unintended.stat()
    intended_inventory = _directory_inventory(intended)
    unintended_inventory = _directory_inventory(unintended)
    displaced = tmp_path / "artifacts" / "displaced-results"
    original_binding = store.load_result_artifact_binding(result.case_id)
    assert original_binding is not None
    observations = _install_result_directory_substitution(
        monkeypatch,
        intended=intended,
        unintended=unintended,
        displaced=displaced,
        timing=timing,
    )

    monkeypatch.setattr(ui_routes, "_RESULT_ARTIFACT_WORKER_BINDING", original_binding)
    if timing == "before-first-symlink":
        with pytest.raises(OSError) as excinfo:
            ui_routes._read_bounded_regular_file(target.absolute())
        assert excinfo.value.errno in {errno.ELOOP, errno.ENOTDIR}
    else:
        with pytest.raises(ValueError, match="parent changed"):
            ui_routes._read_bounded_regular_file(target.absolute())

    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        response = client.get(f"/cases/{result.case_id}/result.json")

    assert response.status_code == 409
    assert response.json() == {
        "case_id": result.case_id,
        "status": result.status.value,
        "stop_reason": "RESULT_ARTIFACT_BINDING_UNRESOLVED",
    }
    assert response.content != payload
    _assert_worker_directory_capability_attempt(observations, timing=timing)
    assert store.load_result_artifact_binding(result.case_id) == original_binding
    substituted_directory = unintended if timing == "before-first-symlink" else intended
    substituted_identity = substituted_directory.stat()
    assert (substituted_identity.st_dev, substituted_identity.st_ino) == (
        unintended_identity.st_dev,
        unintended_identity.st_ino,
    )
    assert _directory_inventory(substituted_directory) == unintended_inventory
    assert _directory_inventory(displaced) == intended_inventory


def test_publication_revalidates_named_directory_after_candidate_fsync_before_binding(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    intended = tmp_path / "artifacts" / "results"
    intended.mkdir(parents=True)
    intended_identity = intended.stat()
    unintended = tmp_path / "unintended-results"
    unintended.mkdir()
    displaced = tmp_path / "artifacts" / "displaced-results"
    intended_inventory = _directory_inventory(intended)
    unintended_inventory = _directory_inventory(unintended)
    original_fsync = os.fsync
    original_fstat = os.fstat
    substituted = False
    monkeypatch.chdir(tmp_path)
    observations = _install_result_directory_substitution(
        monkeypatch,
        intended=intended,
        unintended=unintended,
        displaced=displaced,
        timing="external",
    )

    def substitute_after_candidate_directory_fsync(descriptor: int) -> None:
        nonlocal substituted
        original_fsync(descriptor)
        identity = original_fstat(descriptor)
        if (
            not substituted
            and stat.S_ISDIR(identity.st_mode)
            and (identity.st_dev, identity.st_ino)
            == (intended_identity.st_dev, intended_identity.st_ino)
            and (intended / f"{result.case_id}.json").exists()
        ):
            intended.rename(displaced)
            unintended.rename(intended)
            substituted = True
            observations.events.append("substituted")

    monkeypatch.setattr(os, "fsync", substitute_after_candidate_directory_fsync)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        orchestration._write_result_for_generation(store, result, generation)

    assert substituted
    assert excinfo.value.stop_reason == "ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    _assert_directory_capability_attempts(
        observations,
        timing="after-first-directory",
    )
    assert store.load_result_artifact_binding(result.case_id) is None
    assert _directory_inventory(intended) == unintended_inventory
    assert _directory_inventory(displaced) == intended_inventory


def test_result_worker_and_route_revalidate_named_directory_after_artifact_eof(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result, generation = _terminal_case(settings)
    monkeypatch.chdir(tmp_path)
    target = orchestration._write_result_for_generation(store, result, generation)
    payload = target.read_bytes()
    target_identity = target.stat()
    intended = target.parent
    unintended = tmp_path / "unintended-results"
    unintended.mkdir()
    unintended_target = unintended / target.name
    unintended_target.write_bytes(payload)
    intended_inventory = _directory_inventory(intended)
    unintended_inventory = _directory_inventory(unintended)
    displaced = tmp_path / "artifacts" / "displaced-results"
    original_read = os.read
    original_fstat = os.fstat
    substituted = False
    original_binding = store.load_result_artifact_binding(result.case_id)
    assert original_binding is not None
    observations = _install_result_directory_substitution(
        monkeypatch,
        intended=intended,
        unintended=unintended,
        displaced=displaced,
        timing="external",
    )

    def substitute_after_artifact_eof(descriptor: int, size: int) -> bytes:
        nonlocal substituted
        chunk = original_read(descriptor, size)
        identity = original_fstat(descriptor)
        if (
            not substituted
            and not chunk
            and stat.S_ISREG(identity.st_mode)
            and (identity.st_dev, identity.st_ino)
            == (target_identity.st_dev, target_identity.st_ino)
        ):
            intended.rename(displaced)
            unintended.rename(intended)
            substituted = True
            observations.events.append("substituted")
        return chunk

    monkeypatch.setattr(os, "read", substitute_after_artifact_eof)

    monkeypatch.setattr(ui_routes, "_RESULT_ARTIFACT_WORKER_BINDING", original_binding)
    with pytest.raises(ValueError, match="parent changed"):
        ui_routes._read_bounded_regular_file(target.absolute())

    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        response = client.get(f"/cases/{result.case_id}/result.json")

    assert substituted
    assert response.status_code == 409
    assert response.json() == {
        "case_id": result.case_id,
        "status": result.status.value,
        "stop_reason": "RESULT_ARTIFACT_BINDING_UNRESOLVED",
    }
    assert response.content != payload
    _assert_worker_directory_capability_attempt(
        observations,
        timing="after-first-directory",
    )
    assert store.load_result_artifact_binding(result.case_id) == original_binding
    assert _directory_inventory(intended) == unintended_inventory
    assert _directory_inventory(displaced) == intended_inventory
