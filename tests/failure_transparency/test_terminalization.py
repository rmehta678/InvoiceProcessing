"""Every claimed execution reaches durable terminal evidence or fails explicitly."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from autogen_agentchat.base import TaskResult
from fastapi.testclient import TestClient
from ui.factories import make_pending_review_case, make_succeeded_case

from invoice_agents import orchestration
from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.hitl.service import record_human_decision
from invoice_agents.models import CaseResult, CaseStatus, ErrorRecord, HumanDecisionKind
from invoice_agents.ui import runs as ui_runs
from invoice_agents.ui import sse
from invoice_agents.ui.runs import RunRegistry
from invoice_agents.ui.server import create_app
from invoice_agents.ui.sse import terminal_payload


class _ClosingClient:
    async def close(self) -> None:
        return None


class _CancelledTeam:
    async def run_stream(self, task: object) -> AsyncIterator[object]:
        del task
        raise asyncio.CancelledError
        yield  # pragma: no cover - makes this an async generator


class _MaxMessagesTeam:
    def __init__(self, *, state_error: BaseException | None = None) -> None:
        self.state_error = state_error

    async def run_stream(self, task: object) -> AsyncIterator[object]:
        del task
        yield TaskResult(messages=[], stop_reason="maximum number of messages")

    async def save_state(self) -> dict[str, object]:
        if self.state_error is not None:
            raise self.state_error
        return {"saved": True}


def _prepared_case(invoice_dir: Path, settings: Settings) -> tuple[str, datetime]:
    prepared = orchestration.prepare_case(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    return prepared


async def _run_prepared_with_new_claim(
    case_id: str, started_at: datetime, settings: Settings
) -> CaseResult:
    claim = WorkflowStore(settings).claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        orchestration.EXECUTION_LEASE_SECONDS,
    )
    return await orchestration.run_prepared_case(
        case_id,
        started_at,
        settings,
        claim=claim,
    )


async def _resume_with_new_claim(case_id: str, settings: Settings) -> CaseResult:
    claim = orchestration.claim_resumable_case(case_id, settings)
    return await orchestration.resume_case(case_id, settings, claim=claim)


@pytest.mark.asyncio
async def test_setup_failure_after_claim_is_durably_terminal(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the outer lifecycle boundary must let this sentinel escape."""

    case_id, started_at = _prepared_case(invoice_dir, settings)

    def fail_extraction_load(*_args: object, **_kwargs: object) -> object:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "safe extraction-load sentinel",
            case_id=case_id,
            stop_reason="EXTRACTION_LOAD_FAILED",
        )

    monkeypatch.setattr(WorkflowStore, "promote_predecessor_extraction", fail_extraction_load)
    monkeypatch.chdir(tmp_path)

    try:
        result = await _run_prepared_with_new_claim(case_id, started_at, settings)
    except InvoiceAgentsError as exc:
        pytest.fail(f"post-claim setup failure escaped terminalization: {exc.stop_reason}")

    stored = WorkflowStore(settings.workflow_db).load_result(case_id)
    assert stored == result
    assert stored is not None
    assert stored.status is CaseStatus.FAILED
    assert stored.stop_reason == "EXTRACTION_LOAD_FAILED"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT execution_state, lease_expires_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert tuple(row) == ("FINISHED", None)


@pytest.mark.asyncio
async def test_cancellation_is_durable_and_reraised(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catching cancellation as an ordinary failure must break this contract."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _CancelledTeam())
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    stored = WorkflowStore(settings.workflow_db).load_result(case_id)
    assert stored is not None
    assert stored.status is CaseStatus.INCOMPLETE
    assert stored.stop_reason == "CANCELLED"
    assert [error.stop_reason for error in stored.errors] == ["CANCELLED"]


@pytest.mark.asyncio
async def test_cancellation_during_evidence_reconciliation_remains_cancelled(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    monkeypatch.setattr(
        WorkflowStore,
        "merge_relational_case_evidence",
        lambda _self, _result: (_ for _ in ()).throw(asyncio.CancelledError),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    stored = WorkflowStore(settings).load_result(case_id)
    assert stored is not None
    assert stored.status is CaseStatus.INCOMPLETE
    assert stored.stop_reason == "CANCELLED"
    assert [error.stop_reason for error in stored.errors] == [
        "CANCELLED",
        "TERMINAL_EVIDENCE_RECONCILIATION_CANCELLED",
    ]


@pytest.mark.asyncio
async def test_cancellation_during_terminal_evidence_refresh_is_recovery_durable(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    original_record = orchestration.AuditRecorder.record

    def fail_final_audit(
        self: orchestration.AuditRecorder,
        event_type: str,
        *args: object,
        **kwargs: object,
    ) -> str:
        if event_type == "case.finished":
            raise sqlite3.OperationalError("final audit private sentinel")
        return original_record(self, event_type, *args, **kwargs)

    monkeypatch.setattr(orchestration.AuditRecorder, "record", fail_final_audit)
    monkeypatch.setattr(
        WorkflowStore,
        "update_finished_case_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(asyncio.CancelledError),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    recovery_path = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert recovery["case_result"]["status"] == "INCOMPLETE"
    assert recovery["case_result"]["stop_reason"] == "CANCELLED"
    assert [error["stop_reason"] for error in recovery["case_result"]["errors"]] == [
        "FINAL_AUDIT_WRITE_FAILED",
        "CANCELLED",
        "TERMINAL_RESULT_UPDATE_CANCELLED",
    ]
    assert "private sentinel" not in recovery_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_process_control_during_terminal_evidence_refresh_is_reraised(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    original_record = orchestration.AuditRecorder.record

    def fail_final_audit(
        self: orchestration.AuditRecorder,
        event_type: str,
        *args: object,
        **kwargs: object,
    ) -> str:
        if event_type == "case.finished":
            raise sqlite3.OperationalError("final audit private sentinel")
        return original_record(self, event_type, *args, **kwargs)

    monkeypatch.setattr(orchestration.AuditRecorder, "record", fail_final_audit)
    monkeypatch.setattr(
        WorkflowStore,
        "update_finished_case_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemExit("terminal refresh sk-proj-private-sentinel")
        ),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    recovery_path = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert recovery["case_result"]["stop_reason"] == "MAX_MESSAGES_EXHAUSTED"
    assert [error["stop_reason"] for error in recovery["case_result"]["errors"]] == [
        "FINAL_AUDIT_WRITE_FAILED",
        "TERMINAL_PERSISTENCE_FAILED",
    ]
    assert "private-sentinel" not in recovery_path.read_text(encoding="utf-8")


def test_result_publication_never_exposes_partial_final_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing atomic publication with a direct final-path write must fail."""

    result = CaseResult(
        case_id="case_atomic_publication",
        source_id=None,
        status=CaseStatus.FAILED,
        stop_reason="SYNTHETIC_FAILURE",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    monkeypatch.chdir(tmp_path)

    def fail_replace(_source: os.PathLike[str] | str, _target: os.PathLike[str] | str) -> None:
        raise OSError("safe replace sentinel")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="safe replace sentinel"):
        orchestration._write_result(result)

    output_dir = tmp_path / "artifacts" / "results"
    assert not (output_dir / f"{result.case_id}.json").exists()
    assert list(output_dir.glob(f"{result.case_id}.json.tmp")) == []


def test_result_publication_removes_temp_after_partial_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = CaseResult(
        case_id="case_partial_write",
        source_id=None,
        status=CaseStatus.FAILED,
        stop_reason="SYNTHETIC_FAILURE",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    original_write = os.write
    calls = 0

    def partial_then_fail(file_descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(file_descriptor, payload[:7])
        raise OSError("safe partial-write sentinel")

    monkeypatch.setattr(os, "write", partial_then_fail)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(OSError, match="partial-write sentinel"):
        orchestration._write_result(result)

    output_dir = tmp_path / "artifacts" / "results"
    assert not (output_dir / f"{result.case_id}.json").exists()
    assert not (output_dir / f"{result.case_id}.json.tmp").exists()


def test_result_publication_removes_temp_after_file_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = CaseResult(
        case_id="case_file_fsync",
        source_id=None,
        status=CaseStatus.FAILED,
        stop_reason="SYNTHETIC_FAILURE",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    monkeypatch.setattr(
        os,
        "fsync",
        lambda _file_descriptor: (_ for _ in ()).throw(OSError("safe file-fsync sentinel")),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(OSError, match="file-fsync sentinel"):
        orchestration._write_result(result)

    output_dir = tmp_path / "artifacts" / "results"
    assert not (output_dir / f"{result.case_id}.json").exists()
    assert not (output_dir / f"{result.case_id}.json.tmp").exists()


def test_directory_fsync_failure_leaves_only_complete_final_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = CaseResult(
        case_id="case_directory_fsync",
        source_id=None,
        status=CaseStatus.FAILED,
        stop_reason="SYNTHETIC_FAILURE",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    original_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(file_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("safe directory-fsync sentinel")
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(OSError, match="directory-fsync sentinel"):
        orchestration._write_result(result)

    output_dir = tmp_path / "artifacts" / "results"
    final_path = output_dir / f"{result.case_id}.json"
    assert CaseResult.model_validate_json(final_path.read_text(encoding="utf-8")) == result
    assert not (output_dir / f"{result.case_id}.json.tmp").exists()


def test_result_publication_success_is_complete_and_leaves_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = CaseResult(
        case_id="case_atomic_success",
        source_id=None,
        status=CaseStatus.FAILED,
        stop_reason="SYNTHETIC_FAILURE",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    monkeypatch.chdir(tmp_path)
    final_path = orchestration._write_result(result)

    assert CaseResult.model_validate_json(final_path.read_text(encoding="utf-8")) == result
    assert not final_path.with_name(f"{final_path.name}.tmp").exists()


def test_expired_recovery_is_atomic_audited_and_fences_stale_worker(
    invoice_dir: Path, settings: Settings
) -> None:
    """A recovery lacking CAS, audit, or generation advancement must fail here."""

    case_id, _started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings.workflow_db)
    stale_claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            (expired_at.isoformat(), case_id),
        )
        connection.commit()

    recover = getattr(store, "recover_expired_executions", None)
    assert callable(recover), "WorkflowStore must expose durable expired-execution recovery"
    assert recover(now=datetime.now(UTC)) == [case_id]

    recovered = store.load_result(case_id)
    assert recovered is not None
    assert recovered.status is CaseStatus.INCOMPLETE
    assert recovered.stop_reason == "ORPHANED_EXECUTION"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        case_row = connection.execute(
            "SELECT execution_generation, execution_state, lease_expires_at "
            "FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE case_id = ? "
            "AND event_type = 'case.execution_recovered'",
            (case_id,),
        ).fetchone()[0]
    assert tuple(case_row) == (stale_claim.generation + 1, "FINISHED", None)
    assert event_count == 1

    stale_result = recovered.model_copy(update={"stop_reason": "STALE_OVERWRITE"})
    with pytest.raises(InvoiceAgentsError) as stale_error:
        store.finish_case(stale_result, stale_claim)
    assert stale_error.value.stop_reason == "STALE_EXECUTION_CLAIM"
    with pytest.raises(InvoiceAgentsError) as stale_update_error:
        store.update_finished_case_result(stale_result, stale_claim)
    assert stale_update_error.value.stop_reason == "STALE_EXECUTION_CLAIM"


def test_recovery_uses_exact_utc_lease_boundary_and_rejects_naive_clock(
    invoice_dir: Path, settings: Settings
) -> None:
    case_id, _ = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    boundary = datetime.now(UTC)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            (boundary.isoformat(), case_id),
        )
        connection.commit()

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        store.recover_expired_executions(now=boundary.replace(tzinfo=None))
    with connect_database(settings.workflow_db, read_only=True) as connection:
        unchanged = connection.execute(
            "SELECT execution_token, execution_generation, execution_state, "
            "lease_expires_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert tuple(unchanged) == (
        claim.token,
        claim.generation,
        "RUNNING",
        boundary.isoformat(),
    )

    assert store.recover_expired_executions(now=boundary) == [case_id]
    recovered = store.load_result(case_id)
    assert recovered is not None
    assert recovered.stop_reason == "ORPHANED_EXECUTION"


def test_terminal_payload_recovers_restart_without_memory_task(
    invoice_dir: Path, settings: Settings
) -> None:
    """In-memory task absence must not leave a lease-free case streaming forever."""

    case_id, _started_at = _prepared_case(invoice_dir, settings)

    payload = terminal_payload(settings.workflow_db, case_id, RunRegistry())

    assert payload is not None
    assert payload["status"] == "INCOMPLETE"
    assert payload["stop_reason"] == "ORPHANED_EXECUTION"
    stored = WorkflowStore(settings.workflow_db).load_result(case_id)
    assert stored is not None
    assert stored.stop_reason == "ORPHANED_EXECUTION"


@pytest.mark.asyncio
async def test_sse_active_lease_emits_heartbeat_and_closes_promptly(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy storage lease must not produce an indefinitely silent stream."""

    case_id, _ = _prepared_case(invoice_dir, settings)
    WorkflowStore(settings).claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    assert terminal_payload(settings.workflow_db, case_id, RunRegistry()) is None
    monkeypatch.setattr(sse, "POLL_SECONDS", 0.005)
    monkeypatch.setattr(sse, "HEARTBEAT_SECONDS", 0.01, raising=False)
    stream = sse.case_event_stream(
        settings.workflow_db, case_id, RunRegistry(), after_seq=1_000_000
    )

    event = await asyncio.wait_for(anext(stream), timeout=0.2)

    assert event.encode() == b": heartbeat\r\n\r\n"
    await asyncio.wait_for(stream.aclose(), timeout=0.2)


@pytest.mark.asyncio
async def test_ui_single_launch_hands_off_durable_claim_before_return(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ExecutionClaim | None] = []
    started = asyncio.Event()

    async def block_run(
        case_id: str,
        started_at: datetime,
        selected_settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        del case_id, started_at, selected_settings
        captured.append(claim)
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable run return")

    monkeypatch.setattr(ui_runs, "run_prepared_case", block_run)
    registry = RunRegistry()
    outcome = await registry.start_process(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(outcome, str)
    lease_was_durable = WorkflowStore(settings).has_valid_execution_lease(outcome)
    handle = registry.handle(outcome)
    assert handle is not None and handle.task is not None
    await asyncio.wait_for(started.wait(), timeout=0.2)
    handle.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.task
    if captured[0] is not None:
        WorkflowStore(settings).release_case_execution(captured[0])

    assert lease_was_durable
    assert captured[0] is not None


@pytest.mark.asyncio
async def test_ui_resume_claims_before_scheduling(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, _review = make_pending_review_case(settings)
    captured: list[ExecutionClaim | None] = []
    started = asyncio.Event()

    async def block_resume(
        selected_case_id: str,
        selected_settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        del selected_case_id, selected_settings
        captured.append(claim)
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable resume return")

    monkeypatch.setattr(ui_runs, "resume_case", block_resume)
    registry = RunRegistry()
    handle = registry.start_resume(case_id, settings)
    lease_was_durable = WorkflowStore(settings).has_valid_execution_lease(case_id)
    assert handle.task is not None
    await asyncio.wait_for(started.wait(), timeout=0.2)
    handle.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.task
    if captured[0] is not None:
        WorkflowStore(settings).release_case_execution(captured[0])

    assert lease_was_durable
    assert captured[0] is not None


@pytest.mark.asyncio
async def test_ui_batch_retains_each_claim_while_entries_are_queued(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ExecutionClaim | None] = []
    both_started = asyncio.Event()

    async def block_run(
        case_id: str,
        started_at: datetime,
        selected_settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        del case_id, started_at, selected_settings
        captured.append(claim)
        if len(captured) == 2:
            both_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable batch return")

    monkeypatch.setattr(ui_runs, "run_prepared_case", block_run)
    registry = RunRegistry()
    batch = await registry.start_batch(
        [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
        settings,
        concurrency=2,
    )
    prepared_case_ids = [entry.case_id for entry in batch.entries if entry.prepared]
    durable_before_schedule = [
        WorkflowStore(settings).has_valid_execution_lease(case_id) for case_id in prepared_case_ids
    ]
    assert batch.task is not None
    await asyncio.wait_for(both_started.wait(), timeout=0.2)
    batch.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch.task
    for claim in captured:
        if claim is not None:
            WorkflowStore(settings).release_case_execution(claim)

    assert durable_before_schedule == [True, True]
    assert len(captured) == 2
    assert all(claim is not None for claim in captured)


@pytest.mark.asyncio
async def test_core_process_entrypoints_retain_claims_until_each_run_starts(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, ExecutionClaim | None, bool]] = []
    queued_authorities: list[tuple[str, str]] = []

    async def capture_run(
        case_id: str,
        started_at: datetime,
        selected_settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        del started_at
        store = WorkflowStore(selected_settings)
        observed.append((case_id, claim, store.has_valid_execution_lease(case_id)))
        if len(observed) > 1 and not queued_authorities:
            prior_case_ids = {observed_case_id for observed_case_id, _claim, _valid in observed}
            queued_authorities.extend(
                (row["case_id"], row["execution_state"])
                for row in _nonterminal_authority_rows(selected_settings)
                if row["case_id"] not in prior_case_ids
            )
        if claim is not None:
            store.release_case_execution(claim)
        return CaseResult(
            case_id=case_id,
            source_id=store.load_case_source_id(case_id),
            status=CaseStatus.INCOMPLETE,
            stop_reason="CAPTURED_EXECUTION",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )

    monkeypatch.setattr(orchestration, "run_prepared_case", capture_run)

    single = await orchestration.process_invoice(invoice_dir / "invoice_1001.txt", settings)
    batch = await orchestration.process_batch(
        [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
        settings,
        concurrency=1,
    )

    assert single.stop_reason == "CAPTURED_EXECUTION"
    assert [result.stop_reason for result in batch] == [
        "CAPTURED_EXECUTION",
        "CAPTURED_EXECUTION",
    ]
    assert len(observed) == 3
    assert all(claim is not None and lease_valid for _case_id, claim, lease_valid in observed)
    # With concurrency one, the second batch case is still durably RUNNING while
    # the first case starts rather than being exposed as recoverable IDLE work.
    assert queued_authorities
    assert all(state == "RUNNING" for _case_id, state in queued_authorities)


@pytest.mark.asyncio
async def test_core_batch_preparation_abort_terminalizes_prior_handed_off_claims(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later preparation failure cannot abandon claims acquired earlier in the batch."""

    real_prepare = orchestration.prepare_claimed_case
    prepared: list[tuple[str, datetime, ExecutionClaim]] = []
    calls = 0

    def abort_second(path: Path, selected_settings: Settings) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "batch preparation could not establish terminal evidence",
                stop_reason="TERMINAL_RECOVERY_ARTIFACT_FAILED",
            ) from None
        outcome = real_prepare(path, selected_settings)
        assert isinstance(outcome, tuple)
        prepared.append(outcome)
        return outcome

    monkeypatch.setattr(orchestration, "prepare_claimed_case", abort_second)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration.process_batch(
            [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
            settings,
            concurrency=1,
        )

    assert excinfo.value.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
    assert len(prepared) == 1
    case_id, _started_at, _claim = prepared[0]
    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "CANCELLED"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        authority = connection.execute(
            "SELECT execution_state, lease_expires_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert tuple(authority) == ("FINISHED", None)


def _nonterminal_authority_rows(settings: Settings) -> list[sqlite3.Row]:
    with connect_database(settings.workflow_db, read_only=True) as connection:
        return list(
            connection.execute(
                "SELECT case_id, execution_state FROM cases "
                "WHERE status IN ('INCOMPLETE', 'NEEDS_HUMAN') ORDER BY started_at"
            ).fetchall()
        )


@pytest.mark.asyncio
async def test_delayed_runner_cannot_overwrite_recovered_orphan_with_stale_claim(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, stale_claim = prepared
    store = WorkflowStore(settings)
    _expire_running_case(settings, case_id)
    assert store.recover_expired_executions(now=datetime.now(UTC)) == [case_id]
    recovered = store.load_result(case_id)
    assert recovered is not None and recovered.stop_reason == "ORPHANED_EXECUTION"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        before = connection.execute(
            "SELECT execution_token, execution_generation, execution_state FROM cases "
            "WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert before is not None
    monkeypatch.setattr(
        orchestration,
        "create_model_client",
        lambda _settings: (_ for _ in ()).throw(
            AssertionError("stale delayed runner reached provider construction")
        ),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration.run_prepared_case(
            case_id,
            started_at,
            settings,
            claim=stale_claim,
        )
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"
    assert excinfo.value.__cause__ is None

    with connect_database(settings.workflow_db, read_only=True) as connection:
        after = connection.execute(
            "SELECT execution_token, execution_generation, execution_state FROM cases "
            "WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert tuple(after) == tuple(before)
    assert store.load_result(case_id) == recovered


def test_sse_active_resume_lease_hides_stale_human_review_result(
    settings: Settings,
) -> None:
    case_id, _review = make_pending_review_case(settings)
    store = WorkflowStore(settings)
    previous = store.load_result(case_id)
    assert previous is not None and previous.status is CaseStatus.NEEDS_HUMAN
    store.claim_case_execution(case_id, frozenset({CaseStatus.NEEDS_HUMAN}), lease_seconds=60)

    payload = terminal_payload(settings.workflow_db, case_id, RunRegistry())

    assert payload is None


def test_sse_never_emits_result_observed_before_concurrent_finish(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, _review = make_pending_review_case(settings)
    store = WorkflowStore(settings)
    stale = store.load_result(case_id)
    assert stale is not None and stale.status is CaseStatus.NEEDS_HUMAN
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.NEEDS_HUMAN}),
        lease_seconds=60,
    )
    completed = stale.model_copy(
        update={
            "status": CaseStatus.SUCCEEDED,
            "stop_reason": "DECISION_APPROVE",
            "finished_at": datetime.now(UTC),
        },
        deep=True,
    )
    writer_go = Event()
    writer_done = Event()

    def finish_concurrently() -> None:
        assert writer_go.wait(timeout=1)
        WorkflowStore(settings).finish_case(completed, claim)
        writer_done.set()

    original_load_result = WorkflowStore.load_result

    def stale_load_then_finish(self: WorkflowStore, selected_case_id: str) -> CaseResult | None:
        result = original_load_result(self, selected_case_id)
        writer_go.set()
        assert writer_done.wait(timeout=1)
        return result

    monkeypatch.setattr(WorkflowStore, "load_result", stale_load_then_finish)
    original_snapshot = getattr(WorkflowStore, "load_case_execution_snapshot", None)
    if original_snapshot is not None:

        def snapshot_after_finish(self: WorkflowStore, selected_case_id: str) -> object:
            writer_go.set()
            assert writer_done.wait(timeout=1)
            return original_snapshot(self, selected_case_id)

        monkeypatch.setattr(
            WorkflowStore,
            "load_case_execution_snapshot",
            snapshot_after_finish,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(finish_concurrently)
        payload = terminal_payload(settings.workflow_db, case_id, RunRegistry())
        writer.result(timeout=1)

    assert payload is not None
    assert payload["status"] == "SUCCEEDED"
    assert payload["stop_reason"] == "DECISION_APPROVE"


def test_sse_reloads_terminal_result_when_concurrent_recovery_wins(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, _started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    _expire_running_case(settings, case_id)
    original_recover = WorkflowStore.recover_expired_executions

    def lose_after_other_recovery(self: WorkflowStore, **kwargs: object) -> list[str]:
        assert original_recover(self, **kwargs) == [case_id]
        return []

    monkeypatch.setattr(
        WorkflowStore,
        "recover_expired_executions",
        lose_after_other_recovery,
    )

    payload = terminal_payload(settings.workflow_db, case_id, RunRegistry())

    assert payload is not None
    assert payload["status"] == "INCOMPLETE"
    assert payload["stop_reason"] == "ORPHANED_EXECUTION"


def _expire_running_case(settings: Settings, case_id: str) -> tuple[int, str]:
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    with connect_database(settings.workflow_db) as connection:
        row = connection.execute(
            "SELECT execution_generation, execution_token FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        assert row is not None and row["execution_token"]
        connection.execute(
            "UPDATE cases SET status = 'INCOMPLETE', execution_state = 'RUNNING', "
            "lease_expires_at = ? WHERE case_id = ?",
            (expired_at.isoformat(), case_id),
        )
        connection.commit()
    return int(row["execution_generation"]), str(row["execution_token"])


def test_recovery_retains_paid_decision_and_human_review_truth(settings: Settings) -> None:
    """Recovery must not erase or resume already-authorized durable facts."""

    paid_case = make_succeeded_case(settings)
    paid_generation, _paid_token = _expire_running_case(settings, paid_case)
    human_case, pending_review = make_pending_review_case(settings)
    human_generation, _human_token = _expire_running_case(settings, human_case)

    recovered = WorkflowStore(settings).recover_expired_executions(now=datetime.now(UTC))

    assert recovered == sorted([human_case, paid_case])
    paid = WorkflowStore(settings).load_result(paid_case)
    assert paid is not None
    assert paid.status is CaseStatus.INCOMPLETE
    assert paid.stop_reason == "ORPHANED_EXECUTION"
    assert paid.final_decision is not None
    assert paid.final_decision.decision == "APPROVE"
    assert paid.payment is not None and paid.payment.status == "PAID"
    human = WorkflowStore(settings).load_result(human_case)
    assert human is not None
    assert human.status is CaseStatus.INCOMPLETE
    assert human.stop_reason == "ORPHANED_EXECUTION"
    assert human.review_request is not None
    assert human.review_request.review_id == pending_review.review_id
    with connect_database(settings.workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT case_id, execution_generation, execution_state FROM cases "
            "WHERE case_id IN (?, ?) ORDER BY case_id",
            tuple(sorted([human_case, paid_case])),
        ).fetchall()
    generations = {paid_case: paid_generation, human_case: human_generation}
    assert [
        (row["case_id"], row["execution_generation"], row["execution_state"]) for row in rows
    ] == [
        (case_id, generations[case_id] + 1, "FINISHED")
        for case_id in sorted([human_case, paid_case])
    ]


def test_recovery_reconstructs_committed_payment_when_result_json_is_missing(
    settings: Settings,
) -> None:
    """Committed relational approval/payment evidence must survive a torn result row."""

    case_id = make_succeeded_case(settings)
    _generation, _token = _expire_running_case(settings, case_id)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET result_json = NULL WHERE case_id = ?",
            (case_id,),
        )
        connection.commit()

    assert WorkflowStore(settings).recover_expired_executions(now=datetime.now(UTC)) == [case_id]

    recovered = WorkflowStore(settings).load_result(case_id)
    assert recovered is not None
    assert recovered.stop_reason == "ORPHANED_EXECUTION"
    assert recovered.final_decision is not None
    assert recovered.final_decision.decision == "APPROVE"
    assert recovered.payment is not None
    assert recovered.payment.status == "PAID"
    assert recovered.payment.payment_id is not None


def test_recovery_overlays_relational_payment_on_stale_prior_aggregate(
    settings: Settings,
) -> None:
    """A torn resumed run may retain its predecessor aggregate after payment commits."""

    case_id = make_succeeded_case(settings)
    store = WorkflowStore(settings)
    committed = store.load_result(case_id)
    assert committed is not None
    assert committed.final_decision is not None and committed.payment is not None
    generation, _token = _expire_running_case(settings, case_id)
    stale = committed.model_copy(
        update={
            "status": CaseStatus.NEEDS_HUMAN,
            "stop_reason": "HUMAN_REVIEW_REQUESTED",
            "final_decision": None,
            "payment": None,
        },
        deep=True,
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET status = ?, stop_reason = ?, result_json = ? WHERE case_id = ?",
            (stale.status, stale.stop_reason, stale.model_dump_json(), case_id),
        )
        connection.commit()

    assert store.recover_expired_executions(now=datetime.now(UTC)) == [case_id]

    recovered = store.load_result(case_id)
    assert recovered is not None
    assert recovered.stop_reason == "ORPHANED_EXECUTION"
    assert recovered.final_decision == committed.final_decision
    assert recovered.payment == committed.payment
    assert recovered.errors[-1].details["abandoned_execution_generation"] == generation


def test_missing_relational_payment_cannot_preserve_paid_aggregate(
    settings: Settings,
) -> None:
    case_id = make_succeeded_case(settings)
    store = WorkflowStore(settings)
    committed = store.load_result(case_id)
    assert committed is not None and committed.payment is not None
    generation, token = _expire_running_case(settings, case_id)
    with connect_database(settings.workflow_db) as connection:
        trigger_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'payments'"
        ).fetchall()
        for trigger_row in trigger_rows:
            trigger_name = str(trigger_row["name"]).replace('"', '""')
            connection.execute(f'DROP TRIGGER "{trigger_name}"')
        connection.execute("DELETE FROM payments WHERE case_id = ?", (case_id,))
        authority = connection.execute(
            "SELECT lease_expires_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as merge_error:
        store.merge_relational_case_evidence(committed)
    assert merge_error.value.stop_reason == "PERSISTED_RESULT_INVALID"
    with pytest.raises(InvoiceAgentsError) as recovery_error:
        store.recover_expired_executions(now=datetime.now(UTC))
    assert recovery_error.value.stop_reason == "PERSISTED_RESULT_INVALID"

    with connect_database(settings.workflow_db, read_only=True) as connection:
        unchanged = connection.execute(
            "SELECT execution_token, execution_generation, execution_state, lease_expires_at "
            "FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        recovery_events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE case_id = ? "
            "AND event_type = 'case.execution_recovered'",
            (case_id,),
        ).fetchone()[0]
    assert tuple(unchanged) == (
        token,
        generation,
        "RUNNING",
        authority["lease_expires_at"],
    )
    assert recovery_events == 0


def test_contradictory_relational_payment_evidence_rolls_back_recovery(
    settings: Settings,
) -> None:
    case_id = make_succeeded_case(settings)
    generation, token = _expire_running_case(settings, case_id)
    with connect_database(settings.workflow_db) as connection:
        trigger_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'final_decisions'"
        ).fetchall()
        for trigger_row in trigger_rows:
            trigger_name = str(trigger_row["name"]).replace('"', '""')
            connection.execute(f'DROP TRIGGER "{trigger_name}"')
        connection.execute(
            "UPDATE final_decisions SET source_id = ? WHERE case_id = ?",
            ("src_contradictory_recovery_evidence", case_id),
        )
        connection.execute(
            "UPDATE cases SET result_json = NULL WHERE case_id = ?",
            (case_id,),
        )
        authority = connection.execute(
            "SELECT lease_expires_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as error:
        WorkflowStore(settings).recover_expired_executions(now=datetime.now(UTC))
    assert error.value.stop_reason == "PERSISTED_RESULT_INVALID"

    with connect_database(settings.workflow_db, read_only=True) as connection:
        unchanged = connection.execute(
            "SELECT execution_token, execution_generation, execution_state, "
            "lease_expires_at, result_json FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        recovery_events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE case_id = ? "
            "AND event_type = 'case.execution_recovered'",
            (case_id,),
        ).fetchone()[0]
    assert tuple(unchanged) == (
        token,
        generation,
        "RUNNING",
        authority["lease_expires_at"],
        None,
    )
    assert recovery_events == 0


def test_recovery_workers_are_exact_once_and_active_lease_is_untouched(
    invoice_dir: Path, settings: Settings
) -> None:
    expired_case, _ = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    expired_claim = store.claim_case_execution(
        expired_case, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), expired_case),
        )
        connection.commit()

    active_case, _ = _prepared_case(invoice_dir, settings)
    active_claim = store.claim_case_execution(
        active_case, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    recovery_time = datetime.now(UTC)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                lambda _index: WorkflowStore(settings).recover_expired_executions(
                    now=recovery_time
                ),
                range(2),
            )
        )

    assert sorted(outcomes, key=len) == [[], [expired_case]]
    assert store.load_result(active_case) is None
    with connect_database(settings.workflow_db, read_only=True) as connection:
        active = connection.execute(
            "SELECT execution_generation, execution_token, execution_state, lease_expires_at "
            "FROM cases WHERE case_id = ?",
            (active_case,),
        ).fetchone()
    assert tuple(active) == (
        active_claim.generation,
        active_claim.token,
        "RUNNING",
        active_claim.expires_at.isoformat(),
    )
    assert expired_claim.generation >= 1


def test_recovery_audit_failure_rolls_back_terminalization(
    invoice_dir: Path, settings: Settings
) -> None:
    case_id, _ = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            (expired_at.isoformat(), case_id),
        )
        connection.execute(
            "CREATE TRIGGER fail_recovery_audit BEFORE INSERT ON events "
            "WHEN NEW.event_type = 'case.execution_recovered' "
            "BEGIN SELECT RAISE(ABORT, 'RECOVERY_AUDIT_SENTINEL'); END"
        )
        connection.commit()

    with pytest.raises(Exception, match="RECOVERY_AUDIT_SENTINEL"):
        store.recover_expired_executions(now=datetime.now(UTC))

    assert store.load_result(case_id) is None
    with connect_database(settings.workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT execution_generation, execution_token, execution_state, lease_expires_at "
            "FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert tuple(row) == (
        claim.generation,
        claim.token,
        "RUNNING",
        expired_at.isoformat(),
    )


def test_terminal_payload_reports_corrupt_persisted_result_explicitly(
    invoice_dir: Path, settings: Settings
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.finish_case(
        CaseResult(
            case_id=case_id,
            source_id=None,
            status=CaseStatus.FAILED,
            stop_reason="FIXTURE_TERMINAL",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
        claim,
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            ("{corrupt-json", case_id),
        )
        connection.commit()

    payload = terminal_payload(settings.workflow_db, case_id, RunRegistry())

    assert payload is not None
    assert payload["status"] == "INCOMPLETE"
    assert payload["stop_reason"] == "PERSISTED_RESULT_INVALID"
    assert payload["recovery_error"]["stop_reason"] == "PERSISTED_RESULT_INVALID"
    assert "corrupt-json" not in json.dumps(payload)


def test_startup_runs_recovery_and_propagates_failure(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fail_recovery(self: WorkflowStore, **_kwargs: object) -> list[str]:
        calls.append(self.path)
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "safe startup recovery sentinel",
            stop_reason="EXECUTION_RECOVERY_FAILED",
        )

    monkeypatch.setattr(WorkflowStore, "recover_expired_executions", fail_recovery, raising=False)

    with pytest.raises(InvoiceAgentsError) as error, TestClient(create_app(settings)):
        pass

    assert error.value.stop_reason == "EXECUTION_RECOVERY_FAILED"
    assert calls == [settings.workflow_db.resolve()]


def test_startup_recovers_expired_claim_and_preserves_valid_lease(
    invoice_dir: Path, settings: Settings
) -> None:
    expired_case, _ = _prepared_case(invoice_dir, settings)
    expired_store = WorkflowStore(settings)
    expired_store.claim_case_execution(
        expired_case, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), expired_case),
        )
        connection.commit()
    active_case, _ = _prepared_case(invoice_dir, settings)
    active_claim = WorkflowStore(settings).claim_case_execution(
        active_case, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with TestClient(create_app(settings)):
        pass

    recovered = WorkflowStore(settings).load_result(expired_case)
    assert recovered is not None
    assert recovered.stop_reason == "ORPHANED_EXECUTION"
    assert WorkflowStore(settings).load_result(active_case) is None
    with connect_database(settings.workflow_db, read_only=True) as connection:
        active = connection.execute(
            "SELECT execution_token, execution_generation, execution_state, "
            "lease_expires_at FROM cases WHERE case_id = ?",
            (active_case,),
        ).fetchone()
    assert tuple(active) == (
        active_claim.token,
        active_claim.generation,
        "RUNNING",
        active_claim.expires_at.isoformat(),
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "audit_setup",
        "audit_initial_write",
        "client_construction",
        "team_construction",
        "state_save",
    ],
)
@pytest.mark.asyncio
async def test_setup_boundary_faults_are_sanitized_and_terminal(
    boundary: str,
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    secret = "sk-proj-setup-boundary-secret"
    fault = RuntimeError(f"{boundary}: {secret}")
    original_record = orchestration.AuditRecorder.record

    if boundary == "audit_setup":

        def fail_audit_setup(*_args: object, **_kwargs: object) -> object:
            raise fault

        monkeypatch.setattr(orchestration, "AuditRecorder", fail_audit_setup)
    elif boundary == "audit_initial_write":

        def fail_initial_audit(
            self: orchestration.AuditRecorder,
            event_type: str,
            *args: object,
            **kwargs: object,
        ) -> str:
            if event_type == "provider.configuration":
                raise fault
            return original_record(self, event_type, *args, **kwargs)

        monkeypatch.setattr(orchestration.AuditRecorder, "record", fail_initial_audit)
    elif boundary == "client_construction":

        def fail_client(_settings: Settings) -> object:
            raise fault

        monkeypatch.setattr(orchestration, "create_model_client", fail_client)
    elif boundary == "team_construction":
        monkeypatch.setattr(
            orchestration, "create_model_client", lambda _settings: _ClosingClient()
        )

        def fail_team(_context: object, _client: object) -> object:
            raise fault

        monkeypatch.setattr(orchestration, "build_team", fail_team)
    else:
        monkeypatch.setattr(
            orchestration, "create_model_client", lambda _settings: _ClosingClient()
        )
        monkeypatch.setattr(
            orchestration,
            "build_team",
            lambda _context, _client: _MaxMessagesTeam(state_error=fault),
        )
    monkeypatch.chdir(tmp_path)

    result = await _run_prepared_with_new_claim(case_id, started_at, settings)

    stored = WorkflowStore(settings).load_result(case_id)
    assert stored == result
    assert result.status is CaseStatus.FAILED
    assert result.stop_reason == "UNEXPECTED_RUNTIME_ERROR"
    serialized = result.model_dump_json()
    assert secret not in serialized
    assert result.errors[0].message == "unexpected runtime failure"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        authority = connection.execute(
            "SELECT execution_state, lease_expires_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert tuple(authority) == ("FINISHED", None)


@pytest.mark.parametrize(
    "boundary",
    [
        "extraction_load",
        "audit_initial_write",
        "client_construction",
        "team_construction",
        "state_save",
    ],
)
@pytest.mark.asyncio
async def test_cancellation_at_setup_boundaries_is_durable_and_reraised(
    boundary: str,
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    original_record = orchestration.AuditRecorder.record

    if boundary == "extraction_load":
        monkeypatch.setattr(
            WorkflowStore,
            "promote_predecessor_extraction",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(asyncio.CancelledError),
        )
    elif boundary == "audit_initial_write":

        def cancel_initial_audit(
            self: orchestration.AuditRecorder,
            event_type: str,
            *args: object,
            **kwargs: object,
        ) -> str:
            if event_type == "provider.configuration":
                raise asyncio.CancelledError
            return original_record(self, event_type, *args, **kwargs)

        monkeypatch.setattr(orchestration.AuditRecorder, "record", cancel_initial_audit)
    elif boundary == "client_construction":
        monkeypatch.setattr(
            orchestration,
            "create_model_client",
            lambda _settings: (_ for _ in ()).throw(asyncio.CancelledError),
        )
    elif boundary == "team_construction":
        monkeypatch.setattr(
            orchestration, "create_model_client", lambda _settings: _ClosingClient()
        )
        monkeypatch.setattr(
            orchestration,
            "build_team",
            lambda _context, _client: (_ for _ in ()).throw(asyncio.CancelledError),
        )
    else:
        monkeypatch.setattr(
            orchestration, "create_model_client", lambda _settings: _ClosingClient()
        )
        monkeypatch.setattr(
            orchestration,
            "build_team",
            lambda _context, _client: _MaxMessagesTeam(state_error=asyncio.CancelledError()),
        )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "CANCELLED"
    assert result.errors[0].stop_reason == "CANCELLED"


@pytest.mark.asyncio
async def test_process_control_exception_is_terminalized_then_reraised(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(
        WorkflowStore,
        "promote_predecessor_extraction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemExit("system-exit sk-proj-private-sentinel")
        ),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.FAILED
    assert result.stop_reason == "UNEXPECTED_RUNTIME_ERROR"
    assert "private-sentinel" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_heartbeat_process_control_failure_reaches_outer_terminalizer(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)

    def exit_renewal(_claim: ExecutionClaim, _lease_seconds: int) -> ExecutionClaim:
        raise SystemExit("heartbeat-renewal sk-proj-private-sentinel")

    class BlockingTeam:
        async def run_stream(self, task: object) -> AsyncIterator[object]:
            del task
            await asyncio.Event().wait()
            yield  # pragma: no cover - makes this an async generator

    original_heartbeat = orchestration._run_with_lease_heartbeat

    async def fast_heartbeat(
        operation: Awaitable[CaseResult],
        *,
        renew: Callable[[ExecutionClaim, int], ExecutionClaim],
        claim: ExecutionClaim,
        lease_seconds: int,
        renewal_interval_seconds: float,
    ) -> CaseResult:
        del renew, renewal_interval_seconds
        return await original_heartbeat(
            operation,
            renew=exit_renewal,
            claim=claim,
            lease_seconds=lease_seconds,
            renewal_interval_seconds=0.001,
        )

    monkeypatch.setattr(orchestration, "_run_with_lease_heartbeat", fast_heartbeat)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: BlockingTeam())
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.FAILED
    assert result.stop_reason == "UNEXPECTED_RUNTIME_ERROR"
    assert "private-sentinel" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_process_control_during_client_close_preserves_primary_result(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)

    class ExitCloseClient:
        async def close(self) -> None:
            raise SystemExit("close sk-proj-private-sentinel")

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: ExitCloseClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "MAX_MESSAGES_EXHAUSTED"
    assert [error.stop_reason for error in result.errors] == ["CLIENT_CLOSE_FAILED"]
    assert "private-sentinel" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_primary_process_control_survives_secondary_cleanup_cancellation(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)

    class CancelCloseClient:
        async def close(self) -> None:
            raise asyncio.CancelledError

    class ExitTeam:
        async def run_stream(self, task: object) -> AsyncIterator[object]:
            del task
            raise SystemExit("primary sk-proj-private-sentinel")
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: CancelCloseClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: ExitTeam())
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.FAILED
    assert result.stop_reason == "UNEXPECTED_RUNTIME_ERROR"
    assert [error.stop_reason for error in result.errors] == [
        "UNEXPECTED_RUNTIME_ERROR",
        "CLIENT_CLOSE_CANCELLED",
    ]
    assert "private-sentinel" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_multiple_secondary_failures_preserve_primary_and_order(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    secret = "xai-secondary-secret"

    class FailingCloseClient:
        async def close(self) -> None:
            raise RuntimeError(f"close {secret}")

    monkeypatch.setattr(
        orchestration, "create_model_client", lambda _settings: FailingCloseClient()
    )
    monkeypatch.setattr(
        orchestration,
        "build_team",
        lambda _context, _client: _MaxMessagesTeam(state_error=RuntimeError(f"state {secret}")),
    )
    original_count = WorkflowStore.count_events

    def fail_retry_count(self: WorkflowStore, counted_case: str, event_type: str) -> int:
        if counted_case == case_id and event_type == "provider.retry":
            raise sqlite3.OperationalError(f"retry {secret}")
        return original_count(self, counted_case, event_type)

    monkeypatch.setattr(WorkflowStore, "count_events", fail_retry_count)
    original_record = orchestration.AuditRecorder.record

    def fail_final_audit(
        self: orchestration.AuditRecorder,
        event_type: str,
        *args: object,
        **kwargs: object,
    ) -> str:
        if event_type == "case.finished":
            raise sqlite3.OperationalError(f"audit {secret}")
        return original_record(self, event_type, *args, **kwargs)

    monkeypatch.setattr(orchestration.AuditRecorder, "record", fail_final_audit)
    original_replace = os.replace

    def fail_result_replace(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        if str(target).endswith(f"{case_id}.json"):
            raise OSError(f"artifact {secret}")
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_result_replace)
    monkeypatch.chdir(tmp_path)

    result = await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert result.status is CaseStatus.FAILED
    assert result.stop_reason == "UNEXPECTED_RUNTIME_ERROR"
    assert [error.stop_reason for error in result.errors] == [
        "UNEXPECTED_RUNTIME_ERROR",
        "CLIENT_CLOSE_FAILED",
        "RETRY_COUNT_FAILED",
        "FINAL_AUDIT_WRITE_FAILED",
        "RESULT_ARTIFACT_WRITE_FAILED",
    ]
    assert secret not in result.model_dump_json()
    assert WorkflowStore(settings).load_result(case_id) == result
    result_dir = tmp_path / "artifacts" / "results"
    assert not (result_dir / f"{case_id}.json").exists()
    assert not (result_dir / f"{case_id}.json.tmp").exists()


@pytest.mark.asyncio
async def test_terminal_db_failure_writes_atomic_recovery_artifact(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())

    def fail_terminal_write(_self: WorkflowStore, _result: CaseResult, _claim: object) -> None:
        raise sqlite3.OperationalError("terminal write sk-proj-secret")

    monkeypatch.setattr(WorkflowStore, "finish_case", fail_terminal_write)
    monkeypatch.chdir(tmp_path)

    result = await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "MAX_MESSAGES_EXHAUSTED"
    assert result.errors[-1].stop_reason == "TERMINAL_PERSISTENCE_FAILED"
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert recovery.is_file()
    assert not (recovery.parent / f"{case_id}.recovery.json.tmp").exists()
    payload = json.loads(recovery.read_text(encoding="utf-8"))
    assert payload["recovery_format"] == 1
    assert payload["case_result"]["case_id"] == case_id
    assert payload["terminal_persistence_error"]["stop_reason"] == ("TERMINAL_PERSISTENCE_FAILED")
    assert "sk-proj-secret" not in recovery.read_text(encoding="utf-8")
    assert WorkflowStore(settings).load_result(case_id) is None


@pytest.mark.asyncio
async def test_terminal_db_and_recovery_artifact_failure_raise_explicitly(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())

    def fail_terminal_write(_self: WorkflowStore, _result: CaseResult, _claim: object) -> None:
        raise sqlite3.OperationalError("terminal database private sentinel")

    def fail_recovery_replace(
        _source: os.PathLike[str] | str, target: os.PathLike[str] | str
    ) -> None:
        assert str(target).endswith(f"{case_id}.recovery.json")
        raise OSError("recovery artifact private sentinel")

    monkeypatch.setattr(WorkflowStore, "finish_case", fail_terminal_write)
    monkeypatch.setattr(os, "replace", fail_recovery_replace)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as error:
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert error.value.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
    assert error.value.__cause__ is None
    assert "private sentinel" not in error.value.message
    output_dir = tmp_path / "artifacts" / "results"
    assert not (output_dir / f"{case_id}.json").exists()
    assert not (output_dir / f"{case_id}.recovery.json").exists()
    assert list(output_dir.glob(f"{case_id}*.tmp")) == []
    assert WorkflowStore(settings).load_result(case_id) is None


@pytest.mark.asyncio
async def test_recovery_artifact_process_control_becomes_one_chainless_failure(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    closed = False

    class ObservedClient:
        async def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: ObservedClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    monkeypatch.setattr(
        WorkflowStore,
        "finish_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("terminal database private sentinel")
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "_write_recovery_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemExit("recovery publication sk-proj-private-sentinel")
        ),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert excinfo.value.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert closed
    assert WorkflowStore(settings).load_result(case_id) is None


@pytest.mark.asyncio
async def test_cancellation_during_client_cleanup_is_durable_and_reraised(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)

    class CancelCloseClient:
        async def close(self) -> None:
            raise asyncio.CancelledError

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: CancelCloseClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "CANCELLED"
    assert [error.stop_reason for error in result.errors] == [
        "CANCELLED",
        "CLIENT_CLOSE_CANCELLED",
    ]


@pytest.mark.asyncio
async def test_cancelled_execution_bounds_hung_client_cleanup(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    close_finished = asyncio.Event()

    class NeverCloseClient:
        async def close(self) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                close_finished.set()

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: NeverCloseClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _CancelledTeam())
    monkeypatch.setattr(orchestration, "CLIENT_CLOSE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            _run_prepared_with_new_claim(case_id, started_at, settings),
            timeout=0.2,
        )

    await asyncio.wait_for(close_finished.wait(), timeout=0.2)
    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert [error.stop_reason for error in result.errors] == [
        "CANCELLED",
        "CLIENT_CLOSE_TIMEOUT",
    ]


@pytest.mark.asyncio
async def test_repeated_cancellation_during_cleanup_is_drained_and_reraised(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    stream_started = asyncio.Event()
    close_started = asyncio.Event()
    close_finished = asyncio.Event()

    class BlockingTeam:
        async def run_stream(self, task: object) -> AsyncIterator[object]:
            del task
            stream_started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover - makes this an async generator

    class BlockingCloseClient:
        async def close(self) -> None:
            close_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                close_finished.set()

    monkeypatch.setattr(
        orchestration, "create_model_client", lambda _settings: BlockingCloseClient()
    )
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: BlockingTeam())
    monkeypatch.chdir(tmp_path)
    run_task = asyncio.create_task(_run_prepared_with_new_claim(case_id, started_at, settings))
    await asyncio.wait_for(stream_started.wait(), timeout=0.2)
    run_task.cancel()
    await asyncio.wait_for(close_started.wait(), timeout=0.2)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=0.2)

    await asyncio.wait_for(close_finished.wait(), timeout=0.2)
    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert [error.stop_reason for error in result.errors] == [
        "CANCELLED",
        "CLIENT_CLOSE_CANCELLED",
    ]


@pytest.mark.asyncio
async def test_cancellation_during_terminal_write_retries_cancel_result_once(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    original_finish = WorkflowStore.finish_case
    calls = 0

    def cancel_first_finish(self: WorkflowStore, result: CaseResult, claim: ExecutionClaim) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        original_finish(self, result, claim)

    monkeypatch.setattr(WorkflowStore, "finish_case", cancel_first_finish)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert calls == 2
    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "CANCELLED"
    assert result.errors[-1].stop_reason == "CANCELLED"


@pytest.mark.asyncio
async def test_cancelled_execution_does_not_repeat_failed_terminal_write(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary terminal DB failure gets one attempt and one recovery artifact."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _CancelledTeam())
    calls = 0

    def fail_terminal_write(_self: WorkflowStore, _result: CaseResult, _claim: object) -> None:
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("terminal write private credential")

    monkeypatch.setattr(WorkflowStore, "finish_case", fail_terminal_write)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert calls == 1
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert recovery.is_file()
    assert "private credential" not in recovery.read_text(encoding="utf-8")
    assert WorkflowStore(settings).load_result(case_id) is None


@pytest.mark.asyncio
async def test_cancelled_primary_survives_cleanup_failure(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)

    class FailingCloseClient:
        async def close(self) -> None:
            raise RuntimeError("close xai-private-secret")

    monkeypatch.setattr(
        orchestration, "create_model_client", lambda _settings: FailingCloseClient()
    )
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _CancelledTeam())
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "CANCELLED"
    assert [error.stop_reason for error in result.errors] == [
        "CANCELLED",
        "CLIENT_CLOSE_FAILED",
    ]
    assert "private-secret" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_resume_state_load_failure_after_claim_is_terminal(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    source_id = store.load_extraction(case_id).source.source_id
    store.finish_case(
        CaseResult(
            case_id=case_id,
            source_id=source_id,
            status=CaseStatus.NEEDS_HUMAN,
            stop_reason="HUMAN_REVIEW_REQUESTED",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
        claim,
    )
    monkeypatch.setattr(
        WorkflowStore,
        "load_case_review",
        lambda _self, _case_id: SimpleNamespace(status="RESOLVED", human_decision=object()),
    )

    def fail_state_load(_self: WorkflowStore, _case_id: str) -> object:
        raise RuntimeError("state load sk-proj-private-secret")

    monkeypatch.setattr(WorkflowStore, "load_team_state", fail_state_load)
    monkeypatch.chdir(tmp_path)

    result = await _resume_with_new_claim(case_id, settings)

    assert result.status is CaseStatus.FAILED
    assert result.stop_reason == "UNEXPECTED_RUNTIME_ERROR"
    assert "private-secret" not in result.model_dump_json()
    assert WorkflowStore(settings).load_result(case_id) == result


@pytest.mark.asyncio
async def test_resume_failure_preserves_latest_resolved_review(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, pending = make_pending_review_case(settings)
    resolved = record_human_decision(
        pending.review_id,
        "terminalization reviewer",
        HumanDecisionKind.REQUEST_CORRECTION,
        "request a corrected source",
        WorkflowStore(settings),
        settings.inventory_db,
    )

    def fail_state_load(_self: WorkflowStore, _case_id: str) -> object:
        raise RuntimeError("resume state load private sentinel")

    monkeypatch.setattr(WorkflowStore, "load_team_state", fail_state_load)
    monkeypatch.chdir(tmp_path)

    result = await _resume_with_new_claim(case_id, settings)

    assert result.status is CaseStatus.FAILED
    assert result.review_request == resolved
    assert result.review_request is not None
    assert result.review_request.human_decision is not None
    assert "private sentinel" not in result.model_dump_json()
    assert WorkflowStore(settings).load_result(case_id) == result


@pytest.mark.asyncio
async def test_execution_failure_overlays_already_committed_payment(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = make_succeeded_case(settings)
    store = WorkflowStore(settings)
    committed = store.load_result(case_id)
    assert committed is not None
    assert committed.final_decision is not None and committed.payment is not None
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET status = 'INCOMPLETE', stop_reason = 'TORN_EXECUTION', "
            "result_json = NULL WHERE case_id = ?",
            (case_id,),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    async def fail_after_payment(_execution: object) -> CaseResult:
        raise RuntimeError("post-payment private sentinel")

    monkeypatch.chdir(tmp_path)
    result = await orchestration._execute_claimed_case(
        case_id,
        committed.started_at,
        store,
        claim,
        fail_after_payment,
        finished_event_type="case.finished",
    )

    assert result.status is CaseStatus.FAILED
    assert result.final_decision == committed.final_decision
    assert result.payment == committed.payment
    assert "private sentinel" not in result.model_dump_json()
    assert store.load_result(case_id) == result


@pytest.mark.asyncio
async def test_missing_claim_cannot_reclaim_recovered_terminal_case(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed caller without exact authority cannot replace recovered truth."""

    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, _stale_claim = prepared
    store = WorkflowStore(settings)
    _expire_running_case(settings, case_id)
    assert store.recover_expired_executions(now=datetime.now(UTC)) == [case_id]
    recovered = store.load_result(case_id)
    assert recovered is not None and recovered.stop_reason == "ORPHANED_EXECUTION"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        before = tuple(
            connection.execute(
                "SELECT status, stop_reason, result_json, execution_token, "
                "execution_generation, execution_state, lease_expires_at FROM cases "
                "WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        )
    provider_calls = 0

    def provider_sentinel(_settings: Settings) -> object:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("missing authority reached provider construction")

    monkeypatch.setattr(orchestration, "create_model_client", provider_sentinel)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration.run_prepared_case(case_id, started_at, settings, claim=None)

    assert excinfo.value.stop_reason == "EXECUTION_CLAIM_MISSING"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    with connect_database(settings.workflow_db, read_only=True) as connection:
        after = tuple(
            connection.execute(
                "SELECT status, stop_reason, result_json, execution_token, "
                "execution_generation, execution_state, lease_expires_at FROM cases "
                "WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        )
    assert after == before
    assert store.load_result(case_id) == recovered
    assert provider_calls == 0
    assert not (tmp_path / "artifacts").exists()


def test_prepare_setup_process_control_is_terminalized_then_reraised(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation must not convert process-control into an ordinary return."""

    original_record = orchestration.AuditRecorder.record
    captured_case_id: str | None = None

    def exit_prepared_audit(
        self: orchestration.AuditRecorder,
        event_type: str,
        *args: object,
        **kwargs: object,
    ) -> str:
        nonlocal captured_case_id
        if event_type == "case.prepared":
            captured_case_id = self.case_id
            raise SystemExit("prepare audit sk-proj-private-sentinel")
        return original_record(self, event_type, *args, **kwargs)

    monkeypatch.setattr(orchestration.AuditRecorder, "record", exit_prepared_audit)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)

    assert excinfo.value.__cause__ is None
    assert captured_case_id is not None
    stored = WorkflowStore(settings).load_result(captured_case_id)
    assert stored is not None
    assert stored.status is CaseStatus.FAILED
    assert stored.stop_reason == "UNEXPECTED_RUNTIME_ERROR"
    assert "private-sentinel" not in stored.model_dump_json()
    with connect_database(settings.workflow_db, read_only=True) as connection:
        authority = connection.execute(
            "SELECT execution_state, lease_expires_at FROM cases WHERE case_id = ?",
            (captured_case_id,),
        ).fetchone()
    assert tuple(authority) == ("FINISHED", None)


def test_prepare_setup_terminal_db_failure_publishes_only_recovery_artifact(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal-looking preparation return needs durable recovery evidence."""

    def fail_extraction(_source: object) -> object:
        raise RuntimeError("prepare extraction sk-proj-private-sentinel")

    def fail_finish(_self: WorkflowStore, _result: CaseResult, _claim: ExecutionClaim) -> None:
        raise sqlite3.OperationalError("prepare terminal write private sentinel")

    monkeypatch.setattr(orchestration, "extract_invoice_evidence", fail_extraction)
    monkeypatch.setattr(WorkflowStore, "finish_case", fail_finish)
    monkeypatch.chdir(tmp_path)

    result = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)

    assert isinstance(result, CaseResult)
    recovery = tmp_path / "artifacts" / "results" / f"{result.case_id}.recovery.json"
    normal = tmp_path / "artifacts" / "results" / f"{result.case_id}.json"
    assert recovery.is_file()
    assert not normal.exists()
    payload = json.loads(recovery.read_text(encoding="utf-8"))
    assert payload["case_result"]["case_id"] == result.case_id
    assert payload["terminal_persistence_error"]["stop_reason"] == ("TERMINAL_PERSISTENCE_FAILED")
    assert "private sentinel" not in recovery.read_text(encoding="utf-8")
    assert list(recovery.parent.glob(f"{result.case_id}*.tmp")) == []


@pytest.mark.parametrize(
    ("fault_type", "expected_control"),
    [
        (RuntimeError, None),
        (asyncio.CancelledError, asyncio.CancelledError),
        (SystemExit, SystemExit),
        (KeyboardInterrupt, KeyboardInterrupt),
    ],
)
@pytest.mark.parametrize("persistence_mode", ["success", "db_failure", "artifact_failure"])
def test_prepare_post_claim_fault_matrix_is_durable_and_preserves_precedence(
    fault_type: type[BaseException],
    expected_control: type[BaseException] | None,
    persistence_mode: str,
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every preparation control path has DB or one recovery authority."""

    original_record = orchestration.AuditRecorder.record
    case_ids: list[str] = []

    def fail_prepared_audit(
        self: orchestration.AuditRecorder,
        event_type: str,
        *args: object,
        **kwargs: object,
    ) -> str:
        if event_type == "case.prepared":
            case_ids.append(self.case_id)
            raise fault_type(f"{fault_type.__name__} sk-proj-private-sentinel")
        return original_record(self, event_type, *args, **kwargs)

    monkeypatch.setattr(orchestration.AuditRecorder, "record", fail_prepared_audit)
    if persistence_mode != "success":
        monkeypatch.setattr(
            WorkflowStore,
            "finish_case",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                sqlite3.OperationalError("prepare terminal persistence private sentinel")
            ),
        )
    if persistence_mode == "artifact_failure":
        monkeypatch.setattr(
            orchestration,
            "_write_recovery_artifact",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("prepare recovery publication private sentinel")
            ),
        )
    monkeypatch.chdir(tmp_path)

    if persistence_mode == "artifact_failure":
        with pytest.raises(InvoiceAgentsError) as excinfo:
            orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
        assert excinfo.value.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__context__ is None
    elif expected_control is not None:
        with pytest.raises(expected_control):
            orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    else:
        returned = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
        assert isinstance(returned, CaseResult)

    assert len(case_ids) == 1
    case_id = case_ids[0]
    stored = WorkflowStore(settings).load_result(case_id)
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    normal = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    if persistence_mode == "success":
        assert stored is not None
        assert stored.status is (
            CaseStatus.INCOMPLETE if fault_type is asyncio.CancelledError else CaseStatus.FAILED
        )
        assert not recovery.exists()
    elif persistence_mode == "db_failure":
        assert stored is None
        assert recovery.is_file()
        assert not normal.exists()
    else:
        assert stored is None
        assert not recovery.exists()
        assert not normal.exists()
    artifacts = tmp_path / "artifacts" / "results"
    if artifacts.exists():
        assert list(artifacts.glob(f"{case_id}*.tmp")) == []
        assert "private-sentinel" not in "".join(
            path.read_text(encoding="utf-8") for path in artifacts.glob(f"{case_id}*.json")
        )


@pytest.mark.asyncio
async def test_ui_batch_cancellation_terminalizes_queued_exact_claims_before_return(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling while one child owns the semaphore cannot strand queued leases."""

    started = asyncio.Event()

    class BlockingTeam:
        async def run_stream(self, task: object) -> AsyncIterator[object]:
            del task
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: BlockingTeam())
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    batch = await registry.start_batch(
        [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
        settings,
        concurrency=1,
    )
    assert batch.task is not None
    await asyncio.wait_for(started.wait(), timeout=0.5)

    batch.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(batch.task, timeout=1.0)

    store = WorkflowStore(settings)
    assert len(batch.entries) == 2
    for entry in batch.entries:
        result = store.load_result(entry.case_id)
        assert result is not None
        assert result.status is CaseStatus.INCOMPLETE
        assert result.stop_reason == "CANCELLED"
        handle = registry.handle(entry.case_id)
        assert handle is not None and handle.state == "done"
        with connect_database(settings.workflow_db, read_only=True) as connection:
            authority = connection.execute(
                "SELECT execution_state, lease_expires_at FROM cases WHERE case_id = ?",
                (entry.case_id,),
            ).fetchone()
        assert tuple(authority) == ("FINISHED", None)


@pytest.mark.asyncio
async def test_ui_batch_cancel_during_preparation_accounts_for_completed_thread_claim(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation cannot abandon a claim returned by an in-flight prep thread."""

    real_prepare = ui_runs.prepare_claimed_invoice
    second_started = Event()
    release_second = Event()
    prepared: list[tuple[str, datetime, ExecutionClaim]] = []
    calls = 0

    def controlled_prepare(path: Path, selected_settings: Settings) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            second_started.set()
            assert release_second.wait(timeout=2)
        outcome = real_prepare(path, selected_settings)
        if isinstance(outcome, tuple):
            prepared.append(outcome)
        return outcome

    monkeypatch.setattr(ui_runs, "prepare_claimed_invoice", controlled_prepare)
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    start_task = asyncio.create_task(
        registry.start_batch(
            [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
            settings,
            concurrency=1,
        )
    )
    assert await asyncio.to_thread(second_started.wait, 1)

    start_task.cancel()
    release_second.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=2)
    deadline = asyncio.get_running_loop().time() + 2
    while len(prepared) < 2 and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    assert len(prepared) == 2
    store = WorkflowStore(settings)
    for case_id, _started_at, _claim in prepared:
        result = store.load_result(case_id)
        assert result is not None
        assert result.status is CaseStatus.INCOMPLETE
        assert result.stop_reason == "CANCELLED"
        with connect_database(settings.workflow_db, read_only=True) as connection:
            authority = connection.execute(
                "SELECT execution_state, lease_expires_at FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        assert tuple(authority) == ("FINISHED", None)


@pytest.mark.asyncio
async def test_ui_batch_repeated_cancellation_drains_active_and_queued_claims(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated caller cancellation cannot interrupt terminal durability work."""

    started = asyncio.Event()

    class BlockingTeam:
        async def run_stream(self, task: object) -> AsyncIterator[object]:
            del task
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: BlockingTeam())
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    batch = await registry.start_batch(
        [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
        settings,
        concurrency=1,
    )
    assert batch.task is not None
    await asyncio.wait_for(started.wait(), timeout=0.5)

    batch.task.cancel()
    await asyncio.sleep(0)
    batch.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(batch.task, timeout=1.0)

    store = WorkflowStore(settings)
    for entry in batch.entries:
        result = store.load_result(entry.case_id)
        assert result is not None
        assert result.status is CaseStatus.INCOMPLETE
        assert result.stop_reason == "CANCELLED"
        handle = registry.handle(entry.case_id)
        assert handle is not None and handle.state == "done"


@pytest.mark.asyncio
async def test_ui_batch_cancellation_racing_child_completion_is_terminal(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion and cancellation may race, but the exact claim always finishes."""

    release = asyncio.Event()
    started = asyncio.Event()

    class CompletingTeam(_MaxMessagesTeam):
        async def run_stream(self, task: object) -> AsyncIterator[object]:
            del task
            started.set()
            await release.wait()
            yield TaskResult(messages=[], stop_reason="maximum number of messages")

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: CompletingTeam())
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    batch = await registry.start_batch(
        [invoice_dir / "invoice_1001.txt"],
        settings,
        concurrency=1,
    )
    assert batch.task is not None
    await asyncio.wait_for(started.wait(), timeout=0.5)

    release.set()
    await asyncio.sleep(0)
    cancellation_accepted = batch.task.cancel()
    if cancellation_accepted:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(batch.task, timeout=1.0)
    else:
        await asyncio.wait_for(batch.task, timeout=1.0)

    entry = batch.entries[0]
    result = WorkflowStore(settings).load_result(entry.case_id)
    assert result is not None
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason in {"CANCELLED", "MAX_MESSAGES_EXHAUSTED"}
    with connect_database(settings.workflow_db, read_only=True) as connection:
        authority = connection.execute(
            "SELECT execution_state, lease_expires_at FROM cases WHERE case_id = ?",
            (entry.case_id,),
        ).fetchone()
    assert tuple(authority) == ("FINISHED", None)
    handle = registry.handle(entry.case_id)
    assert handle is not None and handle.state == "done"


@pytest.mark.asyncio
async def test_ui_batch_cancel_surfaces_active_terminal_recovery_failure(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing both terminal stores supersedes cancellation with one stable failure."""

    started = asyncio.Event()

    class BlockingTeam:
        async def run_stream(self, task: object) -> AsyncIterator[object]:
            del task
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: BlockingTeam())
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    batch = await registry.start_batch(
        [invoice_dir / "invoice_1001.txt"],
        settings,
        concurrency=1,
    )
    assert batch.task is not None
    await asyncio.wait_for(started.wait(), timeout=0.5)

    monkeypatch.setattr(
        WorkflowStore,
        "finish_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("batch terminal persistence private sentinel")
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "_write_recovery_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("batch recovery publication private sentinel")
        ),
    )

    batch.task.cancel()
    with pytest.raises(InvoiceAgentsError) as excinfo:
        await asyncio.wait_for(batch.task, timeout=1.0)

    assert excinfo.value.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    handle = registry.handle(batch.entries[0].case_id)
    assert handle is not None and handle.state == "done"


@pytest.mark.asyncio
async def test_ui_batch_cancel_publishes_recovery_for_active_and_queued_claims(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DB terminal-write failure still accounts for every batch claim by artifact."""

    started = asyncio.Event()

    class BlockingTeam:
        async def run_stream(self, task: object) -> AsyncIterator[object]:
            del task
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: BlockingTeam())
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    batch = await registry.start_batch(
        [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
        settings,
        concurrency=1,
    )
    assert batch.task is not None
    await asyncio.wait_for(started.wait(), timeout=0.5)
    monkeypatch.setattr(
        WorkflowStore,
        "finish_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("batch terminal persistence private sentinel")
        ),
    )

    batch.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(batch.task, timeout=1.0)

    for entry in batch.entries:
        recovery = tmp_path / "artifacts" / "results" / f"{entry.case_id}.recovery.json"
        payload = json.loads(recovery.read_text(encoding="utf-8"))
        assert payload["case_result"]["status"] == "INCOMPLETE"
        assert payload["case_result"]["stop_reason"] == "CANCELLED"
        assert payload["terminal_persistence_error"]["stop_reason"] == (
            "TERMINAL_PERSISTENCE_FAILED"
        )
        assert not recovery.with_name(f"{entry.case_id}.json").exists()
        handle = registry.handle(entry.case_id)
        assert handle is not None and handle.state == "done"


@pytest.mark.asyncio
async def test_one_terminal_db_failure_publishes_recovery_exactly_once(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No later audit or artifact branch may republish recovery evidence."""

    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())

    def fail_finish(_self: WorkflowStore, _result: CaseResult, _claim: ExecutionClaim) -> None:
        raise sqlite3.OperationalError("single terminal persistence failure")

    publications: list[str] = []
    original_publish = orchestration._write_recovery_artifact

    def count_publish(result: CaseResult, error: ErrorRecord) -> Path:
        publications.append(result.case_id)
        return original_publish(result, error)

    monkeypatch.setattr(WorkflowStore, "finish_case", fail_finish)
    monkeypatch.setattr(orchestration, "_write_recovery_artifact", count_publish)
    monkeypatch.chdir(tmp_path)

    result = await orchestration.run_prepared_case(case_id, started_at, settings, claim=claim)

    assert result.errors[-1].stop_reason == "TERMINAL_PERSISTENCE_FAILED"
    assert publications == [case_id]
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert recovery.is_file()
    assert not (recovery.parent / f"{case_id}.json").exists()
    assert list(recovery.parent.glob(f"{case_id}*.tmp")) == []
