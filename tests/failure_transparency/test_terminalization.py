"""Every claimed execution reaches durable terminal evidence or fails explicitly."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import sqlite3
import stat
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from threading import Barrier, Event, Thread, current_thread
from types import SimpleNamespace

import pytest
from autogen_agentchat.base import TaskResult
from fastapi.testclient import TestClient
from sse_starlette import ServerSentEvent
from ui.factories import make_pending_review_case, make_succeeded_case

from invoice_agents import lifecycle_process, orchestration, terminal_process
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


@pytest.fixture(autouse=True)
def _exercise_child_side_lifecycle_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy fault injection inside the lifecycle now hosted by the worker."""

    def forbid_external_provider(_settings: Settings) -> object:
        raise AssertionError("non-live Task 9 test reached an unstubbed provider boundary")

    monkeypatch.setattr(orchestration, "create_model_client", forbid_external_provider)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [
            sys.executable,
            "-c",
            "import sys; sys.exit(86)",
        ],
    )

    monkeypatch.setattr(
        orchestration,
        "run_prepared_case",
        orchestration._run_prepared_case_in_process,
    )
    monkeypatch.setattr(
        orchestration,
        "resume_case",
        orchestration._resume_case_in_process,
    )
    monkeypatch.setattr(
        ui_runs,
        "run_prepared_case",
        orchestration._run_prepared_case_in_process,
    )
    monkeypatch.setattr(
        ui_runs,
        "resume_case",
        orchestration._resume_case_in_process,
    )

    async def prepare_in_worker_body(
        path: Path,
        settings: Settings,
    ) -> tuple[str, datetime, ExecutionClaim] | CaseResult:
        return orchestration.prepare_claimed_invoice(path, settings)

    monkeypatch.setattr(
        orchestration,
        "prepare_claimed_invoice_async",
        prepare_in_worker_body,
    )
    monkeypatch.setattr(
        ui_runs,
        "prepare_claimed_invoice_async",
        prepare_in_worker_body,
    )


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


def _failed_terminal_worker(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
    """Protocol-level terminal failure for fresh-interpreter boundary tests."""

    if kwargs["mode"] == "publish_cancel_recovery":
        settings = kwargs["settings"]
        claim = kwargs["claim"]
        started_at = kwargs["started_at"]
        worker_error_code = kwargs["worker_error_code"]
        assert isinstance(settings, Settings)
        assert isinstance(claim, ExecutionClaim)
        assert isinstance(started_at, datetime)
        assert isinstance(worker_error_code, str)
        store = WorkflowStore(settings)
        evidence = orchestration._inspect_exact_claim_evidence(store, claim)
        if evidence.state is orchestration._ExactClaimEvidenceState.DURABLE_DATABASE_RESULT:
            return SimpleNamespace(
                result=evidence.result,
                error_code=None,
                evidence_state=evidence.state.value,
                evidence_result=evidence.result,
            )  # type: ignore[return-value]
        source_id = store.load_authoritative_case_source_id(claim)
        previous = store.load_result(claim.case_id)
        result = orchestration._cancelled_result(
            claim.case_id,
            source_id,
            started_at,
            previous,
        )
        result = store.merge_relational_case_evidence(result)
        error = orchestration._canonical_recovery_persistence_error(
            claim.case_id,
            (
                "TERMINAL_DURABILITY_TIMEOUT"
                if worker_error_code == "TERMINAL_WORKER_TIMEOUT"
                else "TERMINAL_PERSISTENCE_FAILED"
            ),
        )
        result = orchestration._recovery_only_result(result, error)
        orchestration._recovery_artifact_or_raise(result, error, store=store, claim=claim)
        return SimpleNamespace(
            result=result,
            error_code=None,
            evidence_state="RECOVERABLE_RUNNING",
            evidence_result=None,
        )  # type: ignore[return-value]
    return SimpleNamespace(
        result=None,
        error_code="TERMINAL_WORKER_FAILED",
        evidence_state="RECOVERABLE_RUNNING",
        evidence_result=None,
    )  # type: ignore[return-value]


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
    original_terminal_process = orchestration.run_terminal_process

    def cancel_update(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        if kwargs["mode"] == "update":
            raise asyncio.CancelledError
        return original_terminal_process(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(orchestration, "run_terminal_process", cancel_update)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    recovery_path = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert not recovery_path.exists()
    stored = WorkflowStore(settings).load_result(case_id)
    assert stored is not None
    assert stored.stop_reason == "MAX_MESSAGES_EXHAUSTED"
    assert "private sentinel" not in stored.model_dump_json()


@pytest.mark.asyncio
async def test_terminal_helper_done_race_preserves_caught_cancellation(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation delivered with a completed helper is never swallowed."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    execution = orchestration._ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
    )
    injected = asyncio.CancelledError("terminal helper done race")

    def cancel_after_completion(
        worker: asyncio.Future[object],
    ) -> asyncio.Future[object] | asyncio.Task[object]:
        async def wait_then_cancel() -> object:
            await worker
            assert worker.done()
            raise injected

        return asyncio.create_task(wait_then_cancel())

    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
    monkeypatch.setattr(orchestration.asyncio, "shield", cancel_after_completion)

    write = await orchestration._terminal_process_write(
        execution,
        orchestration._cancelled_result(case_id, None, started_at),
        mode="finish",
    )

    assert write.evidence.state is orchestration._ExactClaimEvidenceState.RECOVERABLE_RUNNING
    assert write.control_exception is injected


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_post_helper_exact_evidence_read(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated cancellation cannot escape before the exact claim reread completes."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    execution = orchestration._ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
    )
    helper_started = Event()
    release_helper = Event()
    inspection_started = Event()
    release_inspection = Event()

    def controlled_helper(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        if kwargs["mode"] != "inspect_claim":
            helper_started.set()
            assert release_helper.wait(2.0)
            return terminal_process.TerminalProcessOutcome(None, "TERMINAL_WORKER_FAILED")
        inspection_started.set()
        assert release_inspection.wait(2.0)
        return SimpleNamespace(
            result=None,
            error_code=None,
            evidence_state="RECOVERABLE_RUNNING",
            evidence_result=None,
        )  # type: ignore[return-value]

    monkeypatch.setattr(orchestration, "run_terminal_process", controlled_helper)
    monkeypatch.setattr(
        orchestration,
        "_inspect_exact_claim_evidence",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("parent-owned exact evidence read escaped the helper process")
        ),
    )
    task = asyncio.create_task(
        orchestration._terminal_process_write(
            execution,
            orchestration._cancelled_result(case_id, None, started_at),
            mode="finish",
        )
    )

    try:
        assert await asyncio.to_thread(helper_started.wait, 2.0)
        task.cancel("first terminal boundary cancellation")
        release_helper.set()
        assert await asyncio.to_thread(inspection_started.wait, 2.0)
        task.cancel("second terminal boundary cancellation")
        await asyncio.sleep(0)
        assert not task.done()
        release_inspection.set()
        write = await asyncio.wait_for(task, timeout=2.0)
    finally:
        release_helper.set()
        release_inspection.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert write.evidence.state is orchestration._ExactClaimEvidenceState.RECOVERABLE_RUNNING
    assert isinstance(write.control_exception, asyncio.CancelledError)
    assert write.control_exception.args == ("first terminal boundary cancellation",)


@pytest.mark.asyncio
async def test_full_terminal_persistence_reuses_one_bounded_exact_evidence_read(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full persistence caller cannot start a second unowned claim reread."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    execution = orchestration._ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
    )
    inspection_started = Event()
    release_inspection = Event()
    modes: list[str] = []
    worker_control = SystemExit("terminal worker process control")

    def process_control_worker(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        mode = str(kwargs["mode"])
        modes.append(mode)
        if mode != "inspect_claim":
            raise worker_control
        inspection_started.set()
        assert release_inspection.wait(2.0)
        return SimpleNamespace(
            result=None,
            error_code=None,
            evidence_state="RECOVERABLE_RUNNING",
            evidence_result=None,
        )  # type: ignore[return-value]

    monkeypatch.setattr(orchestration, "run_terminal_process", process_control_worker)
    monkeypatch.setattr(
        orchestration,
        "_inspect_exact_claim_evidence",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("parent-owned exact evidence read escaped the helper process")
        ),
    )
    task = asyncio.create_task(
        orchestration._persist_terminal_result_safely(
            execution,
            orchestration._cancelled_result(case_id, None, started_at),
            None,
        )
    )

    try:
        assert await asyncio.to_thread(inspection_started.wait, 2.0)
        task.cancel("first full persistence cancellation")
        await asyncio.sleep(0)
        task.cancel("second full persistence cancellation")
        await asyncio.sleep(0)
        assert not task.done()
        release_inspection.set()
        persistence = await asyncio.wait_for(task, timeout=2.0)
    finally:
        release_inspection.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert modes == ["finish", "inspect_claim"]
    assert not persistence.persisted
    assert persistence.persistence_error is not None
    assert persistence.persistence_error.stop_reason == "TERMINAL_PERSISTENCE_FAILED"
    assert persistence.control_exception is worker_control


@pytest.mark.asyncio
async def test_normal_terminal_worker_deadline_drains_before_timeout_return(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal-path helper timeout returns only after its process is reaped."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    execution = orchestration._ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
    )
    worker_pid = tmp_path / "normal-terminal-worker.pid"
    late_mutation = tmp_path / "normal-terminal-worker-late.txt"
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [
            sys.executable,
            "-c",
            (
                "import os,time; from pathlib import Path; "
                f"Path({str(worker_pid)!r}).write_text(str(os.getpid())); "
                "time.sleep(10); "
                f"Path({str(late_mutation)!r}).write_text('late')"
            ),
        ],
    )
    monkeypatch.setattr(
        orchestration,
        "_inspect_exact_claim_evidence",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("parent-owned exact evidence read escaped the helper process")
        ),
    )
    monkeypatch.setattr(orchestration, "DURABILITY_DEADLINE_SECONDS", 0.01)
    monkeypatch.setattr(orchestration, "TERMINAL_WORKER_CLEANUP_GRACE_SECONDS", 0.01)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as raised:
        await orchestration._terminal_process_write(
            execution,
            orchestration._cancelled_result(case_id, None, started_at),
            mode="finish",
        )

    assert raised.value.stop_reason == "TERMINAL_DURABILITY_TIMEOUT"
    pid = int(worker_pid.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await asyncio.sleep(0.03)
    assert not late_mutation.exists()


def test_terminal_response_read_is_skipped_after_absolute_operation_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch/capture exhaustion cannot open a fresh response-read budget."""

    read_called = False

    def forbidden_read(_worker: object, _timeout_seconds: float) -> bytes:
        nonlocal read_called
        read_called = True
        raise AssertionError("response read started after absolute deadline")

    monkeypatch.setattr(terminal_process.time, "monotonic", lambda: 4.0)
    monkeypatch.setattr(terminal_process, "_read_response", forbidden_read)

    with pytest.raises(TimeoutError):
        terminal_process._read_response_until(  # type: ignore[attr-defined]
            object(),
            deadline=4.0,
        )

    assert not read_called


def test_terminal_response_read_receives_only_remaining_operation_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response I/O receives the remainder, never the original full timeout."""

    observed_deadline: float | None = None

    def capture_read(_worker: object, *, deadline: float) -> bytes:
        nonlocal observed_deadline
        observed_deadline = deadline
        return b"exact response"

    monkeypatch.setattr(terminal_process.time, "monotonic", lambda: 2.5)
    monkeypatch.setattr(terminal_process, "_read_response", capture_read)

    response = terminal_process._read_response_until(  # type: ignore[attr-defined]
        object(),
        deadline=4.0,
    )

    assert response == b"exact response"
    assert observed_deadline == 4.0
    assert observed_deadline - 2.5 == 1.5


def test_terminal_response_reader_cannot_reanchor_after_setup_consumes_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reader setup cannot extend the original launch-to-response deadline."""

    clock = [1.0]
    setup_continued = False
    watcher_waited = False

    class AdvancingStdout:
        def fileno(self) -> int:
            clock[0] = 4.0
            return 17

    class ForbiddenWatcher:
        def wait(self, _timeout_seconds: float) -> bool:
            nonlocal watcher_waited
            watcher_waited = True
            raise AssertionError("watcher waited after the absolute deadline")

    def forbidden_set_blocking(_descriptor: int, _blocking: bool) -> None:
        nonlocal setup_continued
        setup_continued = True
        raise AssertionError("reader setup continued after the absolute deadline")

    worker = SimpleNamespace(
        process=SimpleNamespace(stdout=AdvancingStdout()),
        exit_watcher=ForbiddenWatcher(),
    )
    monkeypatch.setattr(terminal_process.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(terminal_process.os, "set_blocking", forbidden_set_blocking)

    with pytest.raises(TimeoutError):
        terminal_process._read_response_until(  # type: ignore[attr-defined]
            worker,
            deadline=4.0,
        )

    assert not setup_continued
    assert not watcher_waited


def test_terminal_request_preparation_expiry_never_spawns_worker(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request preparation that exhausts the operation budget cannot spawn."""

    clock = [1.0]
    initializer_called = False
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_request_preparation_expiry",
        f"exec_{'b' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def consume_operation_budget(**_kwargs: object) -> bytes:
        clock[0] = 2.0
        return b"bounded terminal request"

    def forbidden_initializer(*_args: object, **_kwargs: object) -> None:
        nonlocal initializer_called
        initializer_called = True
        raise AssertionError("terminal worker spawned after operation deadline")

    monkeypatch.setattr(terminal_process.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(terminal_process, "_encode_request", consume_operation_budget)
    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        forbidden_initializer,
    )

    outcome = terminal_process.run_terminal_process(
        mode="cancel_unstarted",
        settings=settings,
        claim=claim,
        timeout_seconds=1.0,
        started_at=started_at,
    )

    assert outcome == terminal_process.TerminalProcessOutcome(
        None,
        "TERMINAL_WORKER_TIMEOUT",
    )
    assert not initializer_called


def test_terminal_response_watcher_success_after_deadline_is_still_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truthy exit wait cannot make evidence timely after the absolute deadline."""

    clock = [1.0]
    observed_wait: float | None = None

    class EmptyStdout:
        def fileno(self) -> int:
            return 19

    class DeadlineCrossingWatcher:
        def wait(self, timeout_seconds: float) -> bool:
            nonlocal observed_wait
            observed_wait = timeout_seconds
            clock[0] = 2.1
            return True

    class ReadySelector:
        def __enter__(self) -> ReadySelector:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def register(self, *_args: object) -> None:
            return None

        def select(self, _timeout_seconds: float) -> list[tuple[object, object]]:
            return [(object(), object())]

    worker = SimpleNamespace(
        process=SimpleNamespace(stdout=EmptyStdout()),
        exit_watcher=DeadlineCrossingWatcher(),
    )
    monkeypatch.setattr(terminal_process.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(terminal_process.os, "set_blocking", lambda *_args: None)
    monkeypatch.setattr(terminal_process.os, "read", lambda *_args: b"")
    monkeypatch.setattr(terminal_process.selectors, "DefaultSelector", ReadySelector)

    with pytest.raises(TimeoutError):
        terminal_process._read_response(worker, deadline=2.0)  # type: ignore[call-arg]

    assert observed_wait == 1.0


@pytest.mark.parametrize("budget_consumer", ["command", "environment"])
def test_terminal_worker_argument_preparation_expiry_never_calls_initializer(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    budget_consumer: str,
) -> None:
    """Command and environment preparation remain inside the pre-spawn deadline."""

    clock = [1.0]
    initializer_called = False
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        f"case_terminal_{budget_consumer}_preparation_expiry",
        f"exec_{'c' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def prepare_command() -> list[str]:
        if budget_consumer == "command":
            clock[0] = 2.0
        return [sys.executable, "-m", "invoice_agents.terminal_worker"]

    def prepare_environment() -> dict[str, str]:
        if budget_consumer == "environment":
            clock[0] = 2.0
        return {}

    def forbidden_initializer(*_args: object, **_kwargs: object) -> None:
        nonlocal initializer_called
        initializer_called = True
        raise AssertionError("terminal initializer called after argument preparation expiry")

    monkeypatch.setattr(terminal_process.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(terminal_process, "_terminal_worker_command", prepare_command)
    monkeypatch.setattr(
        terminal_process,
        "sanitized_worker_environment",
        prepare_environment,
    )
    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        forbidden_initializer,
    )

    outcome = terminal_process.run_terminal_process(
        mode="cancel_unstarted",
        settings=settings,
        claim=claim,
        timeout_seconds=1.0,
        started_at=started_at,
    )

    assert outcome == terminal_process.TerminalProcessOutcome(
        None,
        "TERMINAL_WORKER_TIMEOUT",
    )
    assert not initializer_called


def test_terminal_uncertain_session_failure_still_cleans_owned_native_child(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retained native-child ownership cannot escape when session wrapping fails."""

    spawned: list[subprocess.Popen[bytes]] = []
    uncertain_calls = 0
    stop_calls = 0
    real_initialize = subprocess.Popen.__init__
    real_stop = terminal_process._stop_worker
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_uncertain_session_failure",
        f"exec_{'d' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    def fail_uncertain_session(_process: subprocess.Popen[bytes]) -> object:
        nonlocal uncertain_calls
        uncertain_calls += 1
        raise RuntimeError("uncertain terminal session construction failed")

    def track_stop(worker: object) -> object:
        nonlocal stop_calls
        stop_calls += 1
        return real_stop(worker)  # type: ignore[arg-type]

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(
        terminal_process,
        "_reserved_process_has_native_child",
        lambda _process: True,
    )
    monkeypatch.setattr(
        terminal_process,
        "_uncertain_worker_session",
        fail_uncertain_session,
    )
    monkeypatch.setattr(terminal_process, "_stop_worker", track_stop)
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        cleanup_failure: InvoiceAgentsError | None = None
        try:
            outcome = terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.1,
                started_at=started_at,
            )
        except InvoiceAgentsError as exc:
            cleanup_failure = exc
        if cleanup_failure is None:
            assert outcome.error_code == "TERMINAL_WORKER_FAILED"
        else:
            assert cleanup_failure.stop_reason == "TERMINAL_WORKER_CLEANUP_FAILED"
        assert uncertain_calls >= 1
        assert stop_calls == 1
        assert len(spawned) == 1
        assert (
            spawned[0].poll() is not None
            or terminal_process._worker_resource_cleanup_is_poisoned()
        )
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_terminal_selector_error_after_deadline_is_timeout_and_reaped(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary selector error after expiry is timeout, never protocol failure."""

    clock = [1.0]
    spawned: list[subprocess.Popen[bytes]] = []
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_selector_error_after_deadline",
        f"exec_{'e' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    class ExpiringSelector:
        def __enter__(self) -> ExpiringSelector:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def register(self, *_args: object) -> None:
            return None

        def select(self, _timeout_seconds: float) -> list[tuple[object, object]]:
            clock[0] = 2.1
            raise OSError("selector failed after deadline")

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    monkeypatch.setattr(
        terminal_process,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(terminal_process.selectors, "DefaultSelector", ExpiringSelector)
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        outcome = terminal_process.run_terminal_process(
            mode="cancel_unstarted",
            settings=settings,
            claim=claim,
            timeout_seconds=1.0,
            started_at=started_at,
        )

        assert outcome.error_code == "TERMINAL_WORKER_TIMEOUT"
        assert len(spawned) == 1
        assert spawned[0].poll() is not None
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize("budget_consumer", ["command", "environment"])
def test_terminal_argument_error_after_deadline_is_timeout_without_spawn(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    budget_consumer: str,
) -> None:
    """Expired ordinary argument-preparation failures classify as timeout."""

    clock = [1.0]
    initializer_called = False
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        f"case_terminal_{budget_consumer}_error_after_deadline",
        f"exec_{'f' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def prepare_command() -> list[str]:
        if budget_consumer == "command":
            clock[0] = 2.0
            raise OSError("command preparation failed after deadline")
        return [sys.executable, "-m", "invoice_agents.terminal_worker"]

    def prepare_environment() -> dict[str, str]:
        if budget_consumer == "environment":
            clock[0] = 2.0
            raise OSError("environment preparation failed after deadline")
        return {}

    def forbidden_initializer(*_args: object, **_kwargs: object) -> None:
        nonlocal initializer_called
        initializer_called = True
        raise AssertionError("initializer called after argument preparation failure")

    monkeypatch.setattr(
        terminal_process,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    monkeypatch.setattr(terminal_process, "_terminal_worker_command", prepare_command)
    monkeypatch.setattr(
        terminal_process,
        "sanitized_worker_environment",
        prepare_environment,
    )
    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        forbidden_initializer,
    )

    outcome = terminal_process.run_terminal_process(
        mode="cancel_unstarted",
        settings=settings,
        claim=claim,
        timeout_seconds=1.0,
        started_at=started_at,
    )

    assert outcome.error_code == "TERMINAL_WORKER_TIMEOUT"
    assert not initializer_called


def test_terminal_cleanup_reservation_control_reaps_before_reraise(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-spawn cleanup reservation control cannot escape a live native child."""

    injected = SystemExit("terminal cleanup reservation control")
    spawned: list[subprocess.Popen[bytes]] = []
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_cleanup_reservation_control",
        f"exec_{'1' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(
        terminal_process,
        "_reserved_process_has_native_child",
        lambda _process: True,
    )
    monkeypatch.setattr(
        terminal_process,
        "_reserve_terminal_cleanup_session",
        lambda _process: (_ for _ in ()).throw(injected),
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.1,
                started_at=started_at,
            )
        assert raised.value is injected
        assert len(spawned) == 1
        assert (
            spawned[0].poll() is not None
            or terminal_process._worker_resource_cleanup_is_poisoned()
        )
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_terminal_cleanup_control_retries_until_reaped_before_reraise(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First cleanup control is retained while later cleanup proves child extinction."""

    injected = SystemExit("first terminal cleanup control")
    spawned: list[subprocess.Popen[bytes]] = []
    stop_calls = 0
    real_initialize = subprocess.Popen.__init__
    real_stop = terminal_process._stop_worker
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_cleanup_control_retry",
        f"exec_{'2' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    def interrupt_first_stop(worker: object) -> object:
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            raise injected
        return real_stop(worker)  # type: ignore[arg-type]

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(terminal_process, "_stop_worker", interrupt_first_stop)
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.02,
                started_at=started_at,
            )
        assert raised.value is injected
        assert stop_calls >= 2
        assert len(spawned) == 1
        assert (
            spawned[0].poll() is not None
            or terminal_process._worker_resource_cleanup_is_poisoned()
        )
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_terminal_cleanup_poison_check_expiry_skips_request_encoding(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A false poison check cannot consume the budget then admit request work."""

    clock = [1.0]
    encoding_started = False
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_poison_check_expiry",
        f"exec_{'3' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def consume_budget_then_report_clean() -> bool:
        clock[0] = 2.0
        return False

    def forbidden_encode(**_kwargs: object) -> bytes:
        nonlocal encoding_started
        encoding_started = True
        raise AssertionError("request encoding started after poison-check expiry")

    monkeypatch.setattr(
        terminal_process,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    monkeypatch.setattr(
        terminal_process,
        "_worker_resource_cleanup_is_poisoned",
        consume_budget_then_report_clean,
    )
    monkeypatch.setattr(terminal_process, "_encode_request", forbidden_encode)

    outcome = terminal_process.run_terminal_process(
        mode="cancel_unstarted",
        settings=settings,
        claim=claim,
        timeout_seconds=1.0,
        started_at=started_at,
    )

    assert outcome.error_code == "TERMINAL_WORKER_TIMEOUT"
    assert not encoding_started


def test_terminal_cleanup_binding_control_reaps_or_poisons_before_reraise(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated identity-binding control cannot escape an owned native child."""

    injected = SystemExit("terminal cleanup identity binding control")
    spawned: list[subprocess.Popen[bytes]] = []
    bind_calls = 0
    cleanup_poisoned = False
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_cleanup_binding_control",
        f"exec_{'4' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    def fail_binding(*_args: object) -> None:
        nonlocal bind_calls
        bind_calls += 1
        raise injected

    def poison_cleanup() -> None:
        nonlocal cleanup_poisoned
        cleanup_poisoned = True

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(
        terminal_process,
        "_bind_terminal_cleanup_session",
        fail_binding,
    )
    monkeypatch.setattr(
        terminal_process,
        "_poison_worker_resource_cleanup",
        poison_cleanup,
    )
    monkeypatch.setattr(
        terminal_process,
        "_worker_resource_cleanup_is_poisoned",
        lambda: cleanup_poisoned,
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.1,
                started_at=started_at,
            )
        assert raised.value is injected
        assert bind_calls >= 2
        assert len(spawned) == 1
        assert (
            spawned[0].poll() is not None
            or terminal_process._worker_resource_cleanup_is_poisoned()
        )
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_terminal_response_deadline_recheck_control_cleans_before_reraise(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response-handler deadline control cannot escape a live native child."""

    injected = SystemExit("terminal response deadline recheck control")
    spawned: list[subprocess.Popen[bytes]] = []
    response_failed = False
    cleanup_poisoned = False
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_response_deadline_recheck_control",
        f"exec_{'5' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def monotonic() -> float:
        if response_failed:
            raise injected
        return 1.0

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    def fail_response(*_args: object, **_kwargs: object) -> bytes:
        nonlocal response_failed
        response_failed = True
        raise OSError("response failed before deadline recheck")

    def poison_cleanup() -> None:
        nonlocal cleanup_poisoned
        cleanup_poisoned = True

    monkeypatch.setattr(
        terminal_process,
        "time",
        SimpleNamespace(monotonic=monotonic),
    )
    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(terminal_process, "_read_response_until", fail_response)
    monkeypatch.setattr(
        terminal_process,
        "_poison_worker_resource_cleanup",
        poison_cleanup,
    )
    monkeypatch.setattr(
        terminal_process,
        "_worker_resource_cleanup_is_poisoned",
        lambda: cleanup_poisoned,
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=1.0,
                started_at=started_at,
            )
        assert raised.value is injected
        assert len(spawned) == 1
        assert spawned[0].poll() is not None or cleanup_poisoned
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_terminal_finalizer_binding_check_control_is_contained(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalizer binding inspection control poisons before its exact rethrow."""

    injected = SystemExit("terminal finalizer binding check control")
    spawned: list[subprocess.Popen[bytes]] = []
    cleanup_poisoned = False
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_finalizer_binding_check_control",
        f"exec_{'6' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    def fail_binding(*_args: object) -> None:
        raise RuntimeError("terminal cleanup binding failed")

    def poison_cleanup() -> None:
        nonlocal cleanup_poisoned
        cleanup_poisoned = True

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(terminal_process, "_bind_terminal_cleanup_session", fail_binding)
    monkeypatch.setattr(
        terminal_process,
        "_terminal_cleanup_binding_is_exact",
        lambda *_args: (_ for _ in ()).throw(injected),
    )
    monkeypatch.setattr(
        terminal_process,
        "_poison_worker_resource_cleanup",
        poison_cleanup,
    )
    monkeypatch.setattr(
        terminal_process,
        "_worker_resource_cleanup_is_poisoned",
        lambda: cleanup_poisoned,
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.1,
                started_at=started_at,
            )
        assert raised.value is injected
        assert len(spawned) == 1
        assert spawned[0].poll() is not None or cleanup_poisoned
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_terminal_finalizer_owned_stop_control_is_contained(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owned-stop control reaches direct cleanup before exact rethrow."""

    injected = SystemExit("terminal finalizer owned stop control")
    spawned: list[subprocess.Popen[bytes]] = []
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_finalizer_owned_stop_control",
        f"exec_{'7' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(
        terminal_process,
        "_stop_terminal_worker_owned",
        lambda _worker: (_ for _ in ()).throw(injected),
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.02,
                started_at=started_at,
            )
        assert raised.value is injected
        assert len(spawned) == 1
        assert (
            spawned[0].poll() is not None
            or terminal_process._worker_resource_cleanup_is_poisoned()
        )
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize("interrupted_publications", [1, 3])
def test_terminal_poison_publication_control_waits_for_visible_proof(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_publications: int,
) -> None:
    """Poison-publication control is rethrown only after poison is observable."""

    injected = SystemExit("terminal cleanup poison publication control")
    spawned: list[subprocess.Popen[bytes]] = []
    poison_calls = 0
    cleanup_poisoned = False
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        f"case_terminal_poison_publication_{interrupted_publications}",
        f"exec_{'8' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def initialize_and_retain(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)

    def fail_binding(*_args: object) -> None:
        raise RuntimeError("terminal cleanup binding is unavailable")

    def publish_poison() -> None:
        nonlocal cleanup_poisoned, poison_calls
        poison_calls += 1
        if poison_calls <= interrupted_publications:
            raise injected
        cleanup_poisoned = True

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_and_retain,
    )
    monkeypatch.setattr(terminal_process, "_bind_terminal_cleanup_session", fail_binding)
    monkeypatch.setattr(
        terminal_process,
        "_terminal_cleanup_binding_is_exact",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        terminal_process,
        "_poison_worker_resource_cleanup",
        publish_poison,
    )
    monkeypatch.setattr(
        terminal_process,
        "_worker_resource_cleanup_is_poisoned",
        lambda: cleanup_poisoned,
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.1,
                started_at=started_at,
            )
        assert raised.value is injected
        assert poison_calls == interrupted_publications + 1
        assert cleanup_poisoned
        assert len(spawned) == 1
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_terminal_uncertain_raw_ownership_inspection_cleans_or_poisons(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted raw ownership inspection is possible ownership, never no-child proof."""

    injected = SystemExit("terminal raw ownership inspection control")
    spawned: list[subprocess.Popen[bytes]] = []
    cleanup_poisoned = False
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_uncertain_raw_ownership_inspection",
        f"exec_{'9' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def spawn_then_interrupt(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)
        raise RuntimeError("initializer interrupted before ownership publication")

    def fail_raw_inspection(*_args: object, **_kwargs: object) -> object:
        raise injected

    def poison_cleanup() -> None:
        nonlocal cleanup_poisoned
        cleanup_poisoned = True

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        spawn_then_interrupt,
    )
    monkeypatch.setattr(
        terminal_process,
        "_classify_reserved_terminal_process_until",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("classifier interrupted before ownership publication")
        ),
    )
    monkeypatch.setattr(terminal_process, "getattr", fail_raw_inspection, raising=False)
    monkeypatch.setattr(
        terminal_process,
        "_poison_worker_resource_cleanup",
        poison_cleanup,
    )
    monkeypatch.setattr(
        terminal_process,
        "_worker_resource_cleanup_is_poisoned",
        lambda: cleanup_poisoned,
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.1,
                started_at=started_at,
            )
        assert raised.value is injected
        assert len(spawned) == 1
        assert spawned[0].poll() is not None or cleanup_poisoned
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_terminal_positive_pid_is_ownership_when_child_created_is_false(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A published positive PID is ownership before CPython's child flag publication."""

    spawned: list[subprocess.Popen[bytes]] = []
    cleanup_poisoned = False
    real_initialize = subprocess.Popen.__init__
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_terminal_positive_pid_publish_window",
        f"exec_{'a' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    def spawn_in_publish_window(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)
        assert type(process.pid) is int and process.pid > 0
        process._child_created = False  # type: ignore[attr-defined]
        raise RuntimeError("initializer interrupted before child-created publication")

    def poison_cleanup() -> None:
        nonlocal cleanup_poisoned
        cleanup_poisoned = True

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        spawn_in_publish_window,
    )
    monkeypatch.setattr(
        terminal_process,
        "_classify_reserved_terminal_process_until",
        lambda *_args, **_kwargs: (False, None),
    )
    monkeypatch.setattr(
        terminal_process,
        "_poison_worker_resource_cleanup",
        poison_cleanup,
    )
    monkeypatch.setattr(
        terminal_process,
        "_worker_resource_cleanup_is_poisoned",
        lambda: cleanup_poisoned,
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    try:
        outcome = terminal_process.run_terminal_process(
            mode="cancel_unstarted",
            settings=settings,
            claim=claim,
            timeout_seconds=0.1,
            started_at=started_at,
        )
        assert outcome.error_code == "TERMINAL_WORKER_FAILED"
        assert len(spawned) == 1
        assert spawned[0].poll() is not None or cleanup_poisoned
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.asyncio
async def test_terminal_exact_evidence_is_owned_by_the_reaped_helper_session(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal evidence is returned by the helper, never a parent executor read."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    execution = orchestration._ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
    )
    source_id = store.load_authoritative_case_source_id(claim)
    result = orchestration._cancelled_result(case_id, source_id, started_at)
    monkeypatch.setattr(
        orchestration,
        "_inspect_exact_claim_evidence",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("parent-owned exact evidence read escaped the helper process")
        ),
    )

    write = await orchestration._terminal_process_write(execution, result, mode="finish")

    assert write.evidence.state is orchestration._ExactClaimEvidenceState.DURABLE_DATABASE_RESULT
    assert write.evidence.result == result


def test_terminal_inspect_claim_returns_strict_exact_evidence(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """A read-only helper returns the exact claim state in its strict response."""

    case_id, _started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )

    outcome = terminal_process.run_terminal_process(
        mode="inspect_claim",  # type: ignore[arg-type]
        settings=settings,
        claim=claim,
        timeout_seconds=1.0,
    )

    assert outcome.error_code is None
    assert outcome.result is None
    assert outcome.evidence_state == "RECOVERABLE_RUNNING"
    assert outcome.evidence_result is None


def _terminal_response_payload(
    claim: ExecutionClaim,
    *,
    response_kind: str,
    started_at: datetime,
) -> dict[str, object]:
    echoed_claim = terminal_process._claim_payload(claim)
    if response_kind == "inspect_success":
        return {
            "ok": True,
            "claim": echoed_claim,
            "result": None,
            "error_code": None,
            "evidence_state": "RECOVERABLE_RUNNING",
            "evidence_result": None,
        }
    if response_kind == "failure_with_evidence":
        return {
            "ok": False,
            "claim": echoed_claim,
            "result": None,
            "error_code": "TERMINAL_WORKER_FAILED",
            "evidence_state": "RECOVERABLE_RUNNING",
            "evidence_result": None,
        }
    result = orchestration._cancelled_result(claim.case_id, None, started_at)
    dumped = result.model_dump(mode="json")
    return {
        "ok": True,
        "claim": echoed_claim,
        "result": dumped,
        "error_code": None,
        "evidence_state": "DURABLE_DATABASE_RESULT",
        "evidence_result": dumped,
    }


def test_terminal_response_accepts_one_exact_canonical_claim_echo(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """Result-free inspection evidence still requires the exact canonical claim echo."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    claim = WorkflowStore(settings).claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    encoded = terminal_process.encode_terminal_response(
        _terminal_response_payload(
            claim,
            response_kind="inspect_success",
            started_at=started_at,
        )
    )

    outcome = terminal_process._decode_response(encoded, expected_claim=claim)  # type: ignore[call-arg]

    assert outcome.error_code is None
    assert outcome.evidence_state == "RECOVERABLE_RUNNING"
    assert outcome.evidence_result is None


@pytest.mark.parametrize(
    "response_kind",
    ["inspect_success", "failure_with_evidence", "success_with_result"],
)
@pytest.mark.parametrize("claim_mutation", ["token", "generation", "expires_at"])
def test_terminal_response_rejects_every_nonexact_claim_echo(
    invoice_dir: Path,
    settings: Settings,
    response_kind: str,
    claim_mutation: str,
) -> None:
    """Same-case evidence cannot cross token, generation, or lease authority."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    claim = WorkflowStore(settings).claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    payload = _terminal_response_payload(
        claim,
        response_kind=response_kind,
        started_at=started_at,
    )
    echoed = payload["claim"]
    assert isinstance(echoed, dict)
    if claim_mutation == "token":
        echoed["token"] = f"exec_{'f' * 32}"
    elif claim_mutation == "generation":
        echoed["generation"] = claim.generation + 1
    else:
        echoed["expires_at"] = (claim.expires_at + timedelta(seconds=1)).isoformat()
    encoded = terminal_process.encode_terminal_response(payload)

    with pytest.raises(ValueError, match="terminal worker response claim"):
        terminal_process._decode_response(encoded, expected_claim=claim)  # type: ignore[call-arg]


def test_terminal_inspect_claim_rejects_evidence_for_a_different_claim(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict response cannot substitute durable evidence from another case."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    source_id = store.load_authoritative_case_source_id(claim)
    wrong_result = orchestration._cancelled_result(
        "case_malicious_terminal_evidence",
        source_id,
        started_at,
    )
    response = terminal_process.encode_terminal_response(
        {
            "ok": True,
            "claim": terminal_process._claim_payload(claim),
            "result": None,
            "error_code": None,
            "evidence_state": "DURABLE_DATABASE_RESULT",
            "evidence_result": wrong_result.model_dump(mode="json"),
        }
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [
            sys.executable,
            "-c",
            f"import sys; sys.stdout.buffer.write({response!r})",
        ],
    )

    outcome = terminal_process.run_terminal_process(
        mode="inspect_claim",
        settings=settings,
        claim=claim,
        timeout_seconds=1.0,
    )

    assert outcome == terminal_process.TerminalProcessOutcome(
        result=None,
        error_code="TERMINAL_WORKER_FAILED",
    )


def test_terminal_publish_cancel_recovery_is_atomic_and_claim_bound(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery reconciliation and publication occur in one terminable helper."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    monkeypatch.chdir(tmp_path)

    outcome = terminal_process.run_terminal_process(
        mode="publish_cancel_recovery",  # type: ignore[arg-type]
        settings=settings,
        claim=claim,
        timeout_seconds=1.0,
        started_at=started_at,
        worker_error_code="TERMINAL_WORKER_FAILED",  # type: ignore[call-arg]
    )

    assert outcome.error_code is None
    assert outcome.result is not None
    assert outcome.result.stop_reason == "TERMINAL_PERSISTENCE_FAILED"
    assert outcome.evidence_state == "RECOVERABLE_RUNNING"
    assert outcome.evidence_result is None
    assert orchestration._recovery_artifact_is_valid(case_id, store, claim)
    assert store.load_result(case_id) is None


def test_terminal_publish_cancel_recovery_timeout_reaps_before_return(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck recovery helper cannot mutate after its timeout outcome returns."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    worker_pid = tmp_path / "recovery-terminal-worker.pid"
    late_mutation = tmp_path / "recovery-terminal-worker-late.txt"
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [
            sys.executable,
            "-c",
            (
                "import os,time; from pathlib import Path; "
                f"Path({str(worker_pid)!r}).write_text(str(os.getpid())); "
                "time.sleep(10); "
                f"Path({str(late_mutation)!r}).write_text('late')"
            ),
        ],
    )

    outcome = terminal_process.run_terminal_process(
        mode="publish_cancel_recovery",  # type: ignore[arg-type]
        settings=settings,
        claim=claim,
        timeout_seconds=0.02,
        started_at=started_at,
        worker_error_code="TERMINAL_WORKER_FAILED",  # type: ignore[call-arg]
    )

    assert outcome.error_code == "TERMINAL_WORKER_TIMEOUT"
    pid = int(worker_pid.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not late_mutation.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["initial", "retry", "update"])
async def test_later_terminal_helper_process_control_precedes_prior_cancellation(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    """A prior caller cancellation cannot mask later helper process control."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    execution = orchestration._ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
    )
    source_id = store.load_authoritative_case_source_id(claim)
    result = orchestration._cancelled_result(case_id, source_id, started_at)
    prior_cancellation = asyncio.CancelledError("prior terminal cancellation")
    helper_control = SystemExit(f"{surface} terminal helper control")
    calls = 0

    if surface == "update":
        store.finish_case(result, claim)

    def controlled_worker(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        nonlocal calls
        mode = kwargs["mode"]
        if mode == "inspect_claim":
            return SimpleNamespace(
                result=None,
                error_code=None,
                evidence_state="RECOVERABLE_RUNNING",
                evidence_result=None,
            )  # type: ignore[return-value]
        calls += 1
        if surface == "retry" and calls == 1:
            raise asyncio.CancelledError("first finish cancellation")
        raise helper_control

    monkeypatch.setattr(orchestration, "run_terminal_process", controlled_worker)
    if surface == "update":
        outcome = await orchestration._refresh_terminal_evidence_safely(
            execution,
            result,
            persisted=True,
            persistence_error=None,
            control_exception=prior_cancellation,
        )
    else:
        outcome = await orchestration._persist_terminal_result_safely(
            execution,
            result,
            prior_cancellation if surface == "initial" else None,
        )

    assert outcome.control_exception is helper_control
    assert calls == (2 if surface == "retry" else 1)


@pytest.mark.asyncio
async def test_caller_cancellation_precedes_later_ordinary_terminal_helper_failure(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later RuntimeError is reconciled as data and cannot replace first cancellation."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    execution = orchestration._ClaimedExecution(
        store=store,
        claim=claim,
        case_id=case_id,
        started_at=started_at,
    )
    helper_started = Event()
    release_helper = Event()

    def controlled_helper(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        if kwargs["mode"] == "inspect_claim":
            return SimpleNamespace(
                result=None,
                error_code=None,
                evidence_state="RECOVERABLE_RUNNING",
                evidence_result=None,
            )  # type: ignore[return-value]
        helper_started.set()
        assert release_helper.wait(2.0)
        raise RuntimeError("later ordinary terminal helper failure")

    monkeypatch.setattr(orchestration, "run_terminal_process", controlled_helper)
    task = asyncio.create_task(
        orchestration._terminal_process_write(
            execution,
            orchestration._cancelled_result(case_id, None, started_at),
            mode="finish",
        )
    )

    try:
        assert await asyncio.to_thread(helper_started.wait, 2.0)
        task.cancel("first terminal cancellation")
        release_helper.set()
        write = await asyncio.wait_for(task, timeout=2.0)
    finally:
        release_helper.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert isinstance(write.control_exception, asyncio.CancelledError)
    assert write.control_exception.args == ("first terminal cancellation",)


def test_terminal_control_selector_preserves_cancellation_over_ordinary_failure() -> None:
    """Ordinary helper failure data cannot replace the caller's cancellation."""

    cancellation = asyncio.CancelledError("first terminal cancellation")

    selected = orchestration._select_terminal_control_exception(
        cancellation,
        RuntimeError("later ordinary helper failure"),
    )

    assert selected is cancellation


def test_terminal_control_selector_prefers_explicit_durability_failure() -> None:
    """Named durability loss remains load-bearing after caller cancellation."""

    cancellation = asyncio.CancelledError("first terminal cancellation")
    durability_failure = InvoiceAgentsError(
        ErrorCategory.ORCHESTRATION,
        "atomic terminal recovery artifact publication failed",
        case_id="case_terminal_selector_durability",
        stop_reason="TERMINAL_RECOVERY_ARTIFACT_FAILED",
    )

    selected = orchestration._select_terminal_control_exception(
        cancellation,
        durability_failure,
    )

    assert selected is durability_failure


def test_terminal_classifier_deadline_bounds_repeated_process_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent classification control ends at the caller's existing deadline."""

    injected = SystemExit("repeated terminal classifier process control")
    calls = 0
    clock = iter([1.0, 1.01, 1.02])

    def fail_classification(_process: object) -> bool:
        nonlocal calls
        calls += 1
        raise injected

    monkeypatch.setattr(
        terminal_process,
        "_reserved_process_has_native_child",
        fail_classification,
    )
    monkeypatch.setattr(terminal_process.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(terminal_process.time, "sleep", lambda _seconds: None)

    native_child, classification_error = (
        terminal_process._classify_reserved_terminal_process_until(  # type: ignore[attr-defined]
            object(),
            deadline=1.02,
        )
    )

    assert native_child is None
    assert classification_error is injected
    assert calls == 3


def test_terminal_repeated_classifier_control_is_reaped_and_reraised(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient repeated classifier control stays visible after exact child cleanup."""

    injected = SystemExit("terminal classifier process control")
    spawned: list[subprocess.Popen[bytes]] = []
    classifier_calls = 0
    real_initialize = subprocess.Popen.__init__
    real_classifier = terminal_process._reserved_process_has_native_child

    def initialize_then_fail(
        process: subprocess.Popen[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        real_initialize(process, *args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)
        raise RuntimeError("initializer interrupted after native spawn")

    def repeated_control(process: subprocess.Popen[bytes]) -> bool:
        nonlocal classifier_calls
        classifier_calls += 1
        if classifier_calls <= 3:
            raise injected
        return real_classifier(process)

    monkeypatch.setattr(
        terminal_process,
        "_initialize_reserved_terminal_process",
        initialize_then_fail,
    )
    monkeypatch.setattr(
        terminal_process,
        "_reserved_process_has_native_child",
        repeated_control,
    )
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_bounded_terminal_classifier_control",
        f"exec_{'a' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )

    try:
        with pytest.raises(SystemExit) as raised:
            terminal_process.run_terminal_process(
                mode="cancel_unstarted",
                settings=settings,
                claim=claim,
                timeout_seconds=0.1,
                started_at=started_at,
            )
        assert raised.value is injected
        assert classifier_calls == 4
        assert len(spawned) == 1
        assert spawned[0].poll() is not None
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


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
    original_terminal_process = orchestration.run_terminal_process

    def exit_update(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        if kwargs["mode"] == "update":
            raise SystemExit("terminal refresh sk-proj-private-sentinel")
        return original_terminal_process(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(orchestration, "run_terminal_process", exit_update)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    recovery_path = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert not recovery_path.exists()
    stored = WorkflowStore(settings).load_result(case_id)
    assert stored is not None
    assert stored.stop_reason == "MAX_MESSAGES_EXHAUSTED"
    assert "private-sentinel" not in stored.model_dump_json()


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

    def fail_replace(
        _source: os.PathLike[str] | str,
        _target: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        del src_dir_fd, dst_dir_fd
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


def test_directory_fsync_failure_rolls_back_an_unpublished_new_artifact(
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
    output_dir = tmp_path / "artifacts" / "results"
    original_open = os.open
    original_fsync = os.fsync
    original_unlink = os.unlink
    directory_descriptors: set[int] = set()
    injected = OSError("safe directory-fsync sentinel")
    fault_calls = 0
    rollback_unlinks = 0
    rollback_syncs = 0

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output_dir)
        if resolved == output_dir:
            directory_descriptors.add(descriptor)
        return descriptor

    def fail_directory_fsync(file_descriptor: int) -> None:
        nonlocal fault_calls, rollback_syncs
        if file_descriptor in directory_descriptors and fault_calls == 0:
            fault_calls += 1
            raise injected
        original_fsync(file_descriptor)
        if file_descriptor in directory_descriptors and fault_calls:
            rollback_syncs += 1

    def observe_unlink(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal rollback_unlinks
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output_dir)
        if resolved == output_dir / f"{result.case_id}.json":
            rollback_unlinks += 1
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(os, "unlink", observe_unlink)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(OSError) as excinfo:
        orchestration._write_result(result)

    assert excinfo.value is injected
    assert fault_calls == 1
    assert rollback_unlinks == 1
    assert rollback_syncs == 1
    final_path = output_dir / f"{result.case_id}.json"
    assert not final_path.exists()
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


def _assert_atomic_cleanup_unresolved(error: BaseException) -> None:
    assert isinstance(error, InvoiceAgentsError)
    assert error.stop_reason == "ARTIFACT_PUBLICATION_CLEANUP_UNRESOLVED"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "private" not in error.message


def _assert_atomic_namespace_rejected(error: BaseException, *, stop_reason: str) -> None:
    assert isinstance(error, InvoiceAgentsError)
    assert error.stop_reason == stop_reason
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "private" not in error.message


def _directory_entry_inventory(directory: Path) -> dict[str, tuple[int, int, int, str | None]]:
    inventory: dict[str, tuple[int, int, int, str | None]] = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            identity = entry.stat(follow_symlinks=False)
            inventory[entry.name] = (
                identity.st_dev,
                identity.st_ino,
                stat.S_IFMT(identity.st_mode),
                os.readlink(entry.path) if entry.is_symlink() else None,
            )
    return inventory


def _resolve_dirfd_test_path(
    path: os.PathLike[str] | str,
    *,
    dir_fd: int | None,
    directory: Path,
) -> Path:
    """Resolve a test-observed *at(2) path without reopening the namespace."""

    return Path(path) if dir_fd is None else directory / Path(path)


@pytest.mark.parametrize(
    "fault_point",
    [
        "file_close",
        "temp_unlink",
        "directory_open",
        "directory_fsync",
        "directory_close",
    ],
)
def test_atomic_publication_fault_preserves_seeded_final_bytes(
    fault_point: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed replacement attempt must not destroy the last published record."""

    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_seeded_atomic.json"
    temporary = target.with_name(f"{target.name}.tmp")
    prior = b'{"generation":"prior"}\n'
    candidate = b'{"generation":"candidate"}\n'
    target.write_bytes(prior)
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync
    original_replace = os.replace
    original_unlink = os.unlink
    temp_descriptor: int | None = None
    directory_descriptors: set[int] = set()
    injected = OSError(f"{fault_point} private publication sentinel")
    fault_calls = 0
    rollback_syncs = 0
    replacements_to_target: list[Path] = []
    candidate_replaced = False

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal temp_descriptor, fault_calls
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if (
            fault_point == "directory_open"
            and resolved == output
            and candidate_replaced
            and fault_calls == 0
        ):
            fault_calls += 1
            raise injected
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if resolved == temporary:
            temp_descriptor = descriptor
        elif resolved == output:
            directory_descriptors.add(descriptor)
        return descriptor

    def observe_fsync(descriptor: int) -> None:
        nonlocal fault_calls, rollback_syncs
        if (
            fault_point == "directory_fsync"
            and descriptor in directory_descriptors
            and fault_calls == 0
        ):
            fault_calls += 1
            raise injected
        original_fsync(descriptor)
        if descriptor in directory_descriptors and fault_calls:
            rollback_syncs += 1

    def observe_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal candidate_replaced
        source_path = _resolve_dirfd_test_path(
            source, dir_fd=src_dir_fd, directory=output
        )
        destination_path = _resolve_dirfd_test_path(
            destination, dir_fd=dst_dir_fd, directory=output
        )
        if destination_path == target:
            replacements_to_target.append(source_path)
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if source_path == temporary and destination_path == target:
            candidate_replaced = True

    def observe_close(descriptor: int) -> None:
        nonlocal fault_calls
        if fault_point == "file_close" and descriptor == temp_descriptor and fault_calls == 0:
            original_close(descriptor)
            fault_calls += 1
            raise injected
        if (
            fault_point == "directory_close"
            and descriptor in directory_descriptors
            and fault_calls == 0
        ):
            original_close(descriptor)
            fault_calls += 1
            raise injected
        original_close(descriptor)

    def observe_unlink(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal fault_calls
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if fault_point == "temp_unlink" and resolved == temporary and fault_calls == 0:
            fault_calls += 1
            raise injected
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(os, "replace", observe_replace)
    if fault_point == "temp_unlink":
        monkeypatch.setattr(
            os,
            "write",
            lambda _descriptor, _payload: (_ for _ in ()).throw(
                OSError("primary write private publication sentinel")
            ),
        )
        monkeypatch.setattr(os, "unlink", observe_unlink)

    raised: BaseException | None = None
    try:
        orchestration._atomic_publish(target, candidate)
    except BaseException as exc:
        raised = exc

    if fault_point == "directory_open" and fault_calls == 0:
        assert raised is None
        assert candidate_replaced
        assert target.read_bytes() == candidate
        assert replacements_to_target == [temporary]
        assert not temporary.exists()
        assert set(output.iterdir()) == {target}
        return

    assert fault_calls == 1
    assert raised is not None
    assert target.read_bytes() == prior
    if fault_point == "temp_unlink":
        _assert_atomic_cleanup_unresolved(raised)
        assert temporary.is_file()
    elif fault_point in {"file_close", "directory_close"}:
        _assert_atomic_cleanup_unresolved(raised)
        assert not temporary.exists()
    elif fault_point == "directory_open":
        _assert_atomic_namespace_rejected(
            raised,
            stop_reason="ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED",
        )
        assert not temporary.exists()
    else:
        assert raised is injected
        assert not temporary.exists()
    if fault_point in {"directory_open", "directory_fsync", "directory_close"}:
        assert candidate_replaced
        assert len(replacements_to_target) == 2
        assert replacements_to_target[0] == temporary
        assert replacements_to_target[1] != temporary
        assert rollback_syncs == 1
    assert set(output.iterdir()) == ({target, temporary} if temporary.exists() else {target})


@pytest.mark.parametrize(
    "fault_point",
    ["directory_open", "directory_close"],
)
def test_post_replace_fault_restores_target_absence_when_no_prior_record_exists(
    fault_point: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_new_atomic.json"
    temporary = target.with_name(f"{target.name}.tmp")
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync
    original_replace = os.replace
    original_unlink = os.unlink
    directory_descriptors: set[int] = set()
    injected = OSError(f"{fault_point} private publication sentinel")
    fault_calls = 0
    rollback_unlinks = 0
    rollback_syncs = 0
    candidate_replaced = False

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal fault_calls
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if (
            fault_point == "directory_open"
            and resolved == output
            and candidate_replaced
            and fault_calls == 0
        ):
            fault_calls += 1
            raise injected
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if resolved == output:
            directory_descriptors.add(descriptor)
        return descriptor

    def observe_fsync(descriptor: int) -> None:
        nonlocal fault_calls, rollback_syncs
        if (
            fault_point == "directory_fsync"
            and descriptor in directory_descriptors
            and fault_calls == 0
        ):
            fault_calls += 1
            raise injected
        original_fsync(descriptor)
        if descriptor in directory_descriptors and fault_calls:
            rollback_syncs += 1

    def observe_close(descriptor: int) -> None:
        nonlocal fault_calls
        if (
            fault_point == "directory_close"
            and descriptor in directory_descriptors
            and fault_calls == 0
        ):
            original_close(descriptor)
            fault_calls += 1
            raise injected
        original_close(descriptor)

    def observe_unlink(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal rollback_unlinks
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == target:
            rollback_unlinks += 1
        original_unlink(path, dir_fd=dir_fd)

    def observe_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal candidate_replaced
        source_path = _resolve_dirfd_test_path(
            source, dir_fd=src_dir_fd, directory=output
        )
        destination_path = _resolve_dirfd_test_path(
            destination, dir_fd=dst_dir_fd, directory=output
        )
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if source_path == temporary and destination_path == target:
            candidate_replaced = True

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(os, "replace", observe_replace)
    monkeypatch.setattr(os, "unlink", observe_unlink)

    raised: BaseException | None = None
    try:
        orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')
    except BaseException as exc:
        raised = exc

    if fault_point == "directory_open" and fault_calls == 0:
        assert raised is None
        assert candidate_replaced
        assert target.read_bytes() == b'{"generation":"candidate"}\n'
        assert not temporary.exists()
        assert set(output.iterdir()) == {target}
        return

    assert fault_calls == 1
    assert candidate_replaced
    assert raised is not None
    assert rollback_unlinks == 1
    assert rollback_syncs == 1
    assert not target.exists()
    assert not temporary.exists()
    assert list(output.iterdir()) == []
    if fault_point == "directory_close":
        _assert_atomic_cleanup_unresolved(raised)
    else:
        _assert_atomic_namespace_rejected(
            raised,
            stop_reason="ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED",
        )


def test_seeded_target_replace_failure_preserves_prior_bytes_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_seeded_replace.json"
    temporary = target.with_name(f"{target.name}.tmp")
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    original_replace = os.replace
    injected = OSError("replace private publication sentinel")
    replace_attempts = 0

    def fail_candidate_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal replace_attempts
        source_path = _resolve_dirfd_test_path(
            source, dir_fd=src_dir_fd, directory=output
        )
        destination_path = _resolve_dirfd_test_path(
            destination, dir_fd=dst_dir_fd, directory=output
        )
        if source_path == temporary and destination_path == target:
            replace_attempts += 1
            raise injected
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", fail_candidate_replace)

    with pytest.raises(OSError) as excinfo:
        orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')

    assert excinfo.value is injected
    assert replace_attempts == 1
    assert target.read_bytes() == prior
    assert not temporary.exists()
    assert set(output.iterdir()) == {target}


def test_rollback_directory_sync_failure_is_explicitly_durability_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_rollback_sync.json"
    temporary = target.with_name(f"{target.name}.tmp")
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync
    original_replace = os.replace
    next_token = 0
    current_tokens: dict[int, int] = {}
    token_paths: dict[int, Path] = {}
    acquired_tokens: list[int] = []
    closed_tokens: list[int] = []
    close_attempts: list[tuple[int, int | None]] = []
    retired_tokens: dict[int, list[int]] = {}
    candidate_replaced = False
    rollback_replaced = False
    publication_faults = 0
    rollback_faults = 0

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal next_token
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == output or resolved.parent == output:
            next_token += 1
            current_tokens[descriptor] = next_token
            token_paths[next_token] = resolved
            acquired_tokens.append(next_token)
        return descriptor

    def observe_close(descriptor: int) -> None:
        token = current_tokens.get(descriptor)
        close_attempts.append((descriptor, token))
        if token is not None:
            del current_tokens[descriptor]
            closed_tokens.append(token)
            retired_tokens.setdefault(descriptor, []).append(token)
        original_close(descriptor)

    def observe_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal candidate_replaced, rollback_replaced
        source_path = _resolve_dirfd_test_path(
            source, dir_fd=src_dir_fd, directory=output
        )
        destination_path = _resolve_dirfd_test_path(
            destination, dir_fd=dst_dir_fd, directory=output
        )
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination_path != target:
            return
        if source_path == temporary:
            candidate_replaced = True
        else:
            rollback_replaced = True

    def fail_publication_and_rollback_sync(descriptor: int) -> None:
        nonlocal publication_faults, rollback_faults
        token = current_tokens.get(descriptor)
        is_directory = token is not None and token_paths[token] == output
        if is_directory and candidate_replaced:
            if not rollback_replaced and publication_faults == 0:
                publication_faults += 1
                raise OSError("publication durability private sentinel")
            if rollback_replaced and rollback_faults == 0:
                rollback_faults += 1
                raise OSError("rollback durability private sentinel")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    monkeypatch.setattr(os, "replace", observe_replace)
    monkeypatch.setattr(os, "fsync", fail_publication_and_rollback_sync)

    raised: BaseException | None = None
    leaked_descriptors_before_test_cleanup: dict[int, int] = {}
    try:
        try:
            orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')
        except BaseException as exc:
            raised = exc
    finally:
        leaked_descriptors_before_test_cleanup = dict(current_tokens)
        for descriptor in leaked_descriptors_before_test_cleanup:
            with suppress(OSError):
                original_close(descriptor)

    assert isinstance(raised, InvoiceAgentsError)
    assert candidate_replaced
    assert rollback_replaced
    assert publication_faults == 1
    assert rollback_faults == 1
    assert raised.stop_reason == "ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED"
    assert raised.__cause__ is None
    assert raised.__context__ is None
    assert "private" not in raised.message
    assert leaked_descriptors_before_test_cleanup == {}
    assert sorted(closed_tokens) == sorted(acquired_tokens)
    assert len(closed_tokens) == len(set(closed_tokens))
    assert all(token is not None for _descriptor, token in close_attempts)
    assert sorted(token for tokens in retired_tokens.values() for token in tokens) == sorted(
        acquired_tokens
    )
    assert target.read_bytes() == prior
    assert not temporary.exists()
    assert set(output.iterdir()) == {target}


def test_post_replace_rollback_closes_every_acquired_descriptor_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_rollback_resources.json"
    temporary = target.with_name(f"{target.name}.tmp")
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync
    original_replace = os.replace
    next_token = 0
    current_tokens: dict[int, int] = {}
    token_paths: dict[int, Path] = {}
    acquired_tokens: list[int] = []
    closed_tokens: list[int] = []
    close_attempts: list[tuple[int, int | None]] = []
    retired_tokens: dict[int, list[int]] = {}
    candidate_replaced = False
    rollback_replaced = False
    publication_faults = 0
    rollback_syncs = 0

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal next_token
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == output or resolved.parent == output:
            next_token += 1
            current_tokens[descriptor] = next_token
            token_paths[next_token] = resolved
            acquired_tokens.append(next_token)
        return descriptor

    def observe_close(descriptor: int) -> None:
        token = current_tokens.get(descriptor)
        close_attempts.append((descriptor, token))
        if token is not None:
            del current_tokens[descriptor]
            closed_tokens.append(token)
            retired_tokens.setdefault(descriptor, []).append(token)
        original_close(descriptor)

    def observe_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal candidate_replaced, rollback_replaced
        source_path = _resolve_dirfd_test_path(
            source, dir_fd=src_dir_fd, directory=output
        )
        destination_path = _resolve_dirfd_test_path(
            destination, dir_fd=dst_dir_fd, directory=output
        )
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination_path != target:
            return
        if source_path == temporary:
            candidate_replaced = True
        else:
            rollback_replaced = True

    def fail_first_publication_directory_sync(descriptor: int) -> None:
        nonlocal publication_faults, rollback_syncs
        token = current_tokens.get(descriptor)
        is_directory = token is not None and token_paths[token] == output
        if is_directory and candidate_replaced and not rollback_replaced:
            publication_faults += 1
            raise OSError("publication durability sentinel")
        original_fsync(descriptor)
        if is_directory and rollback_replaced:
            rollback_syncs += 1

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    monkeypatch.setattr(os, "replace", observe_replace)
    monkeypatch.setattr(os, "fsync", fail_first_publication_directory_sync)

    raised: BaseException | None = None
    leaked_descriptors_before_test_cleanup: dict[int, int] = {}
    try:
        try:
            orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')
        except BaseException as exc:
            raised = exc
    finally:
        leaked_descriptors_before_test_cleanup = dict(current_tokens)
        for descriptor in leaked_descriptors_before_test_cleanup:
            with suppress(OSError):
                original_close(descriptor)

    assert isinstance(raised, OSError)
    assert "publication durability sentinel" in str(raised)
    assert candidate_replaced
    assert rollback_replaced
    assert publication_faults == 1
    assert rollback_syncs == 1
    assert acquired_tokens
    assert leaked_descriptors_before_test_cleanup == {}
    assert sorted(closed_tokens) == sorted(acquired_tokens)
    assert len(closed_tokens) == len(set(closed_tokens))
    assert all(token is not None for _descriptor, token in close_attempts)
    assert sorted(token for tokens in retired_tokens.values() for token in tokens) == sorted(
        acquired_tokens
    )
    assert target.read_bytes() == prior
    assert set(output.iterdir()) == {target}


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_post_replace_rollback_control_is_reraised_after_durable_containment(
    control_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_rollback_control.json"
    temporary = target.with_name(f"{target.name}.tmp")
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync
    original_replace = os.replace
    next_token = 0
    current_tokens: dict[int, int] = {}
    token_paths: dict[int, Path] = {}
    acquired_tokens: list[int] = []
    closed_tokens: list[int] = []
    close_attempts: list[tuple[int, int | None]] = []
    retired_tokens: dict[int, list[int]] = {}
    control = control_type("rollback control private sentinel")
    candidate_replaced = False
    rollback_replaced = False
    publication_faults = 0
    rollback_syncs = 0

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal next_token
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == output or resolved.parent == output:
            next_token += 1
            current_tokens[descriptor] = next_token
            token_paths[next_token] = resolved
            acquired_tokens.append(next_token)
        return descriptor

    def observe_close(descriptor: int) -> None:
        token = current_tokens.get(descriptor)
        close_attempts.append((descriptor, token))
        if token is not None:
            del current_tokens[descriptor]
            closed_tokens.append(token)
            retired_tokens.setdefault(descriptor, []).append(token)
        original_close(descriptor)

    def fail_after_rollback_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal candidate_replaced, rollback_replaced
        source_path = _resolve_dirfd_test_path(
            source, dir_fd=src_dir_fd, directory=output
        )
        destination_path = _resolve_dirfd_test_path(
            destination, dir_fd=dst_dir_fd, directory=output
        )
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination_path != target:
            return
        if source_path == temporary:
            candidate_replaced = True
            return
        rollback_replaced = True
        raise control

    def fail_publication_sync(descriptor: int) -> None:
        nonlocal publication_faults, rollback_syncs
        token = current_tokens.get(descriptor)
        is_directory = token is not None and token_paths[token] == output
        if is_directory and candidate_replaced:
            if not rollback_replaced and publication_faults == 0:
                publication_faults += 1
                raise OSError("publication durability sentinel")
            original_fsync(descriptor)
            if rollback_replaced:
                rollback_syncs += 1
            return
        original_fsync(descriptor)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    monkeypatch.setattr(os, "replace", fail_after_rollback_replace)
    monkeypatch.setattr(os, "fsync", fail_publication_sync)

    raised: BaseException | None = None
    leaked_descriptors_before_test_cleanup: dict[int, int] = {}
    try:
        try:
            orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')
        except BaseException as exc:
            raised = exc
    finally:
        leaked_descriptors_before_test_cleanup = dict(current_tokens)
        for descriptor in leaked_descriptors_before_test_cleanup:
            with suppress(OSError):
                original_close(descriptor)

    assert raised is control
    assert raised.__cause__ is None
    assert raised.__context__ is None
    assert candidate_replaced
    assert rollback_replaced
    assert publication_faults == 1
    assert rollback_syncs == 1
    assert leaked_descriptors_before_test_cleanup == {}
    assert sorted(closed_tokens) == sorted(acquired_tokens)
    assert len(closed_tokens) == len(set(closed_tokens))
    assert all(token is not None for _descriptor, token in close_attempts)
    assert sorted(token for tokens in retired_tokens.values() for token in tokens) == sorted(
        acquired_tokens
    )
    assert target.read_bytes() == prior
    assert not temporary.exists()
    assert set(output.iterdir()) == {target}


@pytest.mark.parametrize(
    ("publication_control_type", "rollback_control_type"),
    [
        (asyncio.CancelledError, SystemExit),
        (SystemExit, KeyboardInterrupt),
        (KeyboardInterrupt, asyncio.CancelledError),
    ],
)
def test_earliest_post_replace_control_survives_later_rollback_control_and_uncertainty(
    publication_control_type: type[BaseException],
    rollback_control_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_rollback_control_precedence.json"
    temporary = target.with_name(f"{target.name}.tmp")
    prior = b'{"generation":"prior"}\n'
    target.write_bytes(prior)
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync
    original_replace = os.replace
    next_token = 0
    current_tokens: dict[int, int] = {}
    token_paths: dict[int, Path] = {}
    acquired_tokens: list[int] = []
    closed_tokens: list[int] = []
    close_attempts: list[tuple[int, int | None]] = []
    retired_tokens: dict[int, list[int]] = {}
    publication_control = publication_control_type("publication control private sentinel")
    rollback_control = rollback_control_type("rollback control private sentinel")
    candidate_replaced = False
    rollback_replaced = False
    publication_faults = 0
    rollback_control_faults = 0
    rollback_durability_faults = 0

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal next_token
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == output or resolved.parent == output:
            next_token += 1
            current_tokens[descriptor] = next_token
            token_paths[next_token] = resolved
            acquired_tokens.append(next_token)
        return descriptor

    def observe_close(descriptor: int) -> None:
        token = current_tokens.get(descriptor)
        close_attempts.append((descriptor, token))
        if token is not None:
            del current_tokens[descriptor]
            closed_tokens.append(token)
            retired_tokens.setdefault(descriptor, []).append(token)
        original_close(descriptor)

    def fail_after_rollback_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal candidate_replaced, rollback_replaced, rollback_control_faults
        source_path = _resolve_dirfd_test_path(
            source, dir_fd=src_dir_fd, directory=output
        )
        destination_path = _resolve_dirfd_test_path(
            destination, dir_fd=dst_dir_fd, directory=output
        )
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination_path != target:
            return
        if source_path == temporary:
            candidate_replaced = True
            return
        rollback_replaced = True
        rollback_control_faults += 1
        raise rollback_control

    def fail_publication_and_rollback_sync(descriptor: int) -> None:
        nonlocal publication_faults, rollback_durability_faults
        token = current_tokens.get(descriptor)
        is_directory = token is not None and token_paths[token] == output
        if is_directory and candidate_replaced:
            if not rollback_replaced and publication_faults == 0:
                publication_faults += 1
                raise publication_control
            if rollback_replaced and rollback_durability_faults == 0:
                rollback_durability_faults += 1
                raise OSError("rollback durability private sentinel")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    monkeypatch.setattr(os, "replace", fail_after_rollback_replace)
    monkeypatch.setattr(os, "fsync", fail_publication_and_rollback_sync)

    raised: BaseException | None = None
    leaked_descriptors_before_test_cleanup: dict[int, int] = {}
    try:
        try:
            orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')
        except BaseException as exc:
            raised = exc
    finally:
        leaked_descriptors_before_test_cleanup = dict(current_tokens)
        for descriptor in leaked_descriptors_before_test_cleanup:
            with suppress(OSError):
                original_close(descriptor)

    assert raised is publication_control
    assert raised.__cause__ is None
    assert raised.__context__ is None
    assert candidate_replaced
    assert rollback_replaced
    assert publication_faults == 1
    assert rollback_control_faults == 1
    assert rollback_durability_faults == 1
    assert leaked_descriptors_before_test_cleanup == {}
    assert sorted(closed_tokens) == sorted(acquired_tokens)
    assert len(closed_tokens) == len(set(closed_tokens))
    assert all(token is not None for _descriptor, token in close_attempts)
    assert sorted(token for tokens in retired_tokens.values() for token in tokens) == sorted(
        acquired_tokens
    )
    assert target.read_bytes() == prior
    assert not temporary.exists()
    assert set(output.iterdir()) == {target}


@pytest.mark.parametrize("resource", ["file", "directory"])
def test_atomic_publication_never_operates_on_a_reused_raw_descriptor(
    resource: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused numeric descriptor must remain an unrelated live kernel resource."""

    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_descriptor_reuse.json"
    target.write_bytes(b'{"generation":"prior"}\n')
    temporary = target.with_name(f"{target.name}.tmp")
    original_open = os.open
    original_close = os.close
    original_fstat = os.fstat
    original_fsync = os.fsync
    original_get_inheritable = os.get_inheritable
    original_set_inheritable = os.set_inheritable
    original_read = os.read
    original_write = os.write
    original_dup = os.dup
    original_dup2 = os.dup2
    owned_descriptor: int | None = None
    reused_descriptor: int | None = None
    sentinel_writer: int | None = None
    subsequent_operations: list[str] = []
    injected = OSError(f"{resource} close private publication sentinel")

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal owned_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if (resource == "file" and resolved == temporary) or (
            resource == "directory" and resolved == output and owned_descriptor is None
        ):
            owned_descriptor = descriptor
        return descriptor

    def fail_after_close_and_reuse(descriptor: int) -> None:
        nonlocal reused_descriptor, sentinel_writer
        if descriptor == owned_descriptor and reused_descriptor is None:
            original_close(descriptor)
            reader, writer = os.pipe()
            if reader != descriptor:
                original_dup2(reader, descriptor)
                original_close(reader)
            reused_descriptor = descriptor
            sentinel_writer = writer
            raise injected
        if reused_descriptor is not None and descriptor == reused_descriptor:
            subsequent_operations.append("close")
            raise OSError("test protected reused descriptor")
        original_close(descriptor)

    def observe_fstat(descriptor: int) -> os.stat_result:
        if reused_descriptor is not None and descriptor == reused_descriptor:
            subsequent_operations.append("fstat")
            raise OSError("test protected reused descriptor")
        return original_fstat(descriptor)

    def observe_fsync(descriptor: int) -> None:
        if reused_descriptor is not None and descriptor == reused_descriptor:
            subsequent_operations.append("fsync")
            raise OSError("test protected reused descriptor")
        original_fsync(descriptor)

    def observe_get_inheritable(descriptor: int) -> bool:
        if reused_descriptor is not None and descriptor == reused_descriptor:
            subsequent_operations.append("get_inheritable")
            raise OSError("test protected reused descriptor")
        return original_get_inheritable(descriptor)

    def observe_set_inheritable(descriptor: int, inheritable: bool) -> None:
        if reused_descriptor is not None and descriptor == reused_descriptor:
            subsequent_operations.append("set_inheritable")
            raise OSError("test protected reused descriptor")
        original_set_inheritable(descriptor, inheritable)

    def observe_read(descriptor: int, length: int) -> bytes:
        if reused_descriptor is not None and descriptor == reused_descriptor:
            subsequent_operations.append("read")
            raise OSError("test protected reused descriptor")
        return original_read(descriptor, length)

    def observe_write(descriptor: int, payload: bytes) -> int:
        if reused_descriptor is not None and descriptor == reused_descriptor:
            subsequent_operations.append("write")
            raise OSError("test protected reused descriptor")
        return original_write(descriptor, payload)

    def observe_dup(descriptor: int) -> int:
        if reused_descriptor is not None and descriptor == reused_descriptor:
            subsequent_operations.append("dup")
            raise OSError("test protected reused descriptor")
        return original_dup(descriptor)

    def observe_dup2(descriptor: int, destination: int, inheritable: bool = True) -> int:
        if reused_descriptor is not None and (
            descriptor == reused_descriptor or destination == reused_descriptor
        ):
            subsequent_operations.append("dup2")
            raise OSError("test protected reused descriptor")
        return original_dup2(descriptor, destination, inheritable=inheritable)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", fail_after_close_and_reuse)
    monkeypatch.setattr(os, "fstat", observe_fstat)
    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(os, "get_inheritable", observe_get_inheritable)
    monkeypatch.setattr(os, "set_inheritable", observe_set_inheritable)
    monkeypatch.setattr(os, "read", observe_read)
    monkeypatch.setattr(os, "write", observe_write)
    monkeypatch.setattr(os, "dup", observe_dup)
    monkeypatch.setattr(os, "dup2", observe_dup2)
    try:
        with pytest.raises(BaseException) as excinfo:
            orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')

        assert owned_descriptor is not None
        assert reused_descriptor == owned_descriptor
        assert sentinel_writer is not None
        _assert_atomic_cleanup_unresolved(excinfo.value)
        assert subsequent_operations == []
        original_fstat(reused_descriptor)
        original_write(sentinel_writer, b"K")
        assert original_read(reused_descriptor, 1) == b"K"
    finally:
        if reused_descriptor is not None:
            with suppress(OSError):
                original_close(reused_descriptor)
        if sentinel_writer is not None:
            with suppress(OSError):
                original_close(sentinel_writer)


def test_successful_close_retires_ownership_before_numeric_descriptor_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_successful_close_reuse.json"
    temporary = target.with_name(f"{target.name}.tmp")
    original_open = os.open
    original_close = os.close
    original_fstat = os.fstat
    original_fsync = os.fsync
    original_get_inheritable = os.get_inheritable
    original_set_inheritable = os.set_inheritable
    original_read = os.read
    original_write = os.write
    original_dup = os.dup
    original_dup2 = os.dup2
    next_token = 0
    active_tokens: dict[int, int] = {}
    token_paths: dict[int, Path] = {}
    retired_tokens: dict[int, list[int]] = {}
    close_attempts: list[tuple[int, int | None]] = []
    phase_timeline: list[tuple[str, int, int | None]] = []
    foreign_descriptor: int | None = None
    foreign_writer: int | None = None
    foreign_operations: list[str] = []

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal next_token
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == output or resolved.parent == output:
            next_token += 1
            active_tokens[descriptor] = next_token
            token_paths[next_token] = resolved
            phase_timeline.append(("open", descriptor, next_token))
        return descriptor

    def observe_close(descriptor: int) -> None:
        nonlocal foreign_descriptor, foreign_writer
        token = active_tokens.get(descriptor)
        close_attempts.append((descriptor, token))
        phase_timeline.append(("close_attempt", descriptor, token))
        if foreign_descriptor is not None and descriptor == foreign_descriptor:
            foreign_operations.append("close")
            raise OSError("test protected foreign descriptor")
        if token is not None:
            del active_tokens[descriptor]
            retired_tokens.setdefault(descriptor, []).append(token)
        original_close(descriptor)
        if token is not None and token_paths[token] == temporary:
            reader, writer = os.pipe()
            if reader != descriptor:
                original_dup2(reader, descriptor)
                original_close(reader)
            foreign_descriptor = descriptor
            foreign_writer = writer
            phase_timeline.append(("foreign_reuse", descriptor, token))

    def reject_foreign(operation: str, descriptor: int) -> None:
        if foreign_descriptor is not None and descriptor == foreign_descriptor:
            foreign_operations.append(operation)
            raise OSError("test protected foreign descriptor")

    def observe_fstat(descriptor: int) -> os.stat_result:
        reject_foreign("fstat", descriptor)
        return original_fstat(descriptor)

    def observe_fsync(descriptor: int) -> None:
        reject_foreign("fsync", descriptor)
        original_fsync(descriptor)

    def observe_get_inheritable(descriptor: int) -> bool:
        reject_foreign("get_inheritable", descriptor)
        return original_get_inheritable(descriptor)

    def observe_set_inheritable(descriptor: int, inheritable: bool) -> None:
        reject_foreign("set_inheritable", descriptor)
        original_set_inheritable(descriptor, inheritable)

    def observe_read(descriptor: int, length: int) -> bytes:
        reject_foreign("read", descriptor)
        return original_read(descriptor, length)

    def observe_write(descriptor: int, payload: bytes) -> int:
        reject_foreign("write", descriptor)
        return original_write(descriptor, payload)

    def observe_dup(descriptor: int) -> int:
        reject_foreign("dup", descriptor)
        return original_dup(descriptor)

    def observe_dup2(descriptor: int, destination: int, inheritable: bool = True) -> int:
        reject_foreign("dup2", descriptor)
        reject_foreign("dup2", destination)
        return original_dup2(descriptor, destination, inheritable=inheritable)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    monkeypatch.setattr(os, "fstat", observe_fstat)
    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(os, "get_inheritable", observe_get_inheritable)
    monkeypatch.setattr(os, "set_inheritable", observe_set_inheritable)
    monkeypatch.setattr(os, "read", observe_read)
    monkeypatch.setattr(os, "write", observe_write)
    monkeypatch.setattr(os, "dup", observe_dup)
    monkeypatch.setattr(os, "dup2", observe_dup2)
    raised: BaseException | None = None
    try:
        try:
            orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')
        except BaseException as exc:
            raised = exc

        assert raised is None
        assert foreign_descriptor is not None
        assert foreign_writer is not None
        assert foreign_operations == []
        assert active_tokens == {}
        assert all(token is not None for _descriptor, token in close_attempts)
        assert sorted(token for tokens in retired_tokens.values() for token in tokens) == sorted(
            token_paths
        )
        assert any(
            phase == "foreign_reuse" and descriptor == foreign_descriptor
            for phase, descriptor, _token in phase_timeline
        )
        original_fstat(foreign_descriptor)
        original_write(foreign_writer, b"S")
        assert original_read(foreign_descriptor, 1) == b"S"
        assert target.read_bytes() == b'{"generation":"candidate"}\n'
        assert set(output.iterdir()) == {target}
    finally:
        leaked_descriptors_before_test_cleanup = dict(active_tokens)
        for descriptor in leaked_descriptors_before_test_cleanup:
            with suppress(OSError):
                original_close(descriptor)
        if foreign_descriptor is not None:
            with suppress(OSError):
                original_close(foreign_descriptor)
        if foreign_writer is not None:
            with suppress(OSError):
                original_close(foreign_writer)

    assert leaked_descriptors_before_test_cleanup == {}


@pytest.mark.parametrize(
    ("primary_type", "close_type", "unlink_type", "expected_control"),
    [
        (RuntimeError, OSError, OSError, None),
        (KeyboardInterrupt, OSError, OSError, "primary"),
        (SystemExit, OSError, OSError, "primary"),
        (asyncio.CancelledError, OSError, OSError, "primary"),
        (RuntimeError, KeyboardInterrupt, OSError, "close"),
        (RuntimeError, SystemExit, OSError, "close"),
        (RuntimeError, asyncio.CancelledError, OSError, "close"),
        (RuntimeError, OSError, KeyboardInterrupt, "unlink"),
        (RuntimeError, OSError, SystemExit, "unlink"),
        (RuntimeError, OSError, asyncio.CancelledError, "unlink"),
        (asyncio.CancelledError, SystemExit, KeyboardInterrupt, "primary"),
    ],
)
def test_atomic_publication_attempts_all_cleanup_once_and_preserves_control_precedence(
    primary_type: type[BaseException],
    close_type: type[BaseException],
    unlink_type: type[BaseException],
    expected_control: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_cleanup_precedence.json"
    temporary = target.with_name(f"{target.name}.tmp")
    primary = primary_type("primary private publication sentinel")
    close_failure = close_type("close private publication sentinel")
    unlink_failure = unlink_type("unlink private publication sentinel")
    original_open = os.open
    original_close = os.close
    original_fstat = os.fstat
    original_unlink = os.unlink
    temp_descriptor: int | None = None
    cleanup_calls: list[str] = []

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal temp_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == temporary:
            temp_descriptor = descriptor
        return descriptor

    def fail_close(descriptor: int) -> None:
        if descriptor == temp_descriptor:
            cleanup_calls.append("file_close")
            original_close(descriptor)
            raise close_failure
        original_close(descriptor)

    def fail_unlink(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == temporary:
            cleanup_calls.append("temp_unlink")
            original_unlink(path, dir_fd=dir_fd)
            raise unlink_failure
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", fail_close)
    monkeypatch.setattr(
        os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(primary),
    )
    monkeypatch.setattr(os, "unlink", fail_unlink)

    raised: BaseException | None = None
    leaked_descriptor_before_test_cleanup: int | None = None
    try:
        try:
            orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')
        except BaseException as exc:
            raised = exc
    finally:
        if temp_descriptor is not None:
            try:
                original_fstat(temp_descriptor)
            except OSError:
                pass
            else:
                leaked_descriptor_before_test_cleanup = temp_descriptor
                original_close(temp_descriptor)

    assert cleanup_calls == ["file_close", "temp_unlink"]
    assert leaked_descriptor_before_test_cleanup is None
    assert not target.exists()
    assert not temporary.exists()
    if expected_control == "primary":
        assert raised is primary
        assert raised.__cause__ is None
        assert raised.__context__ is None
    elif expected_control == "close":
        assert raised is close_failure
        assert raised.__cause__ is None
        assert raised.__context__ is None
    elif expected_control == "unlink":
        assert raised is unlink_failure
        assert raised.__cause__ is None
        assert raised.__context__ is None
    else:
        assert raised is not None
        _assert_atomic_cleanup_unresolved(raised)


@pytest.mark.parametrize("target_kind", ["dangling_symlink", "file_symlink"])
def test_atomic_publication_rejects_symlink_target_without_following_it(
    target_kind: str,
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_symlink.json"
    referent = output / "referent.json"
    if target_kind == "file_symlink":
        referent.write_bytes(b'{"referent":"prior"}\n')
    target.symlink_to(referent.name)
    original_link_text = os.readlink(target)
    prior_inventory = _directory_entry_inventory(output)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')

    _assert_atomic_namespace_rejected(
        excinfo.value,
        stop_reason="ARTIFACT_PUBLICATION_TARGET_UNSAFE",
    )
    assert target.is_symlink()
    assert os.readlink(target) == original_link_text
    if target_kind == "file_symlink":
        assert referent.read_bytes() == b'{"referent":"prior"}\n'
    else:
        assert not referent.exists()
    assert _directory_entry_inventory(output) == prior_inventory


def test_atomic_publication_rejects_fifo_target_without_opening_or_replacing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_fifo.json"
    os.mkfifo(target, 0o600)
    prior_inventory = _directory_entry_inventory(output)
    original_open = os.open
    process_context = get_context("fork")
    outcome_receiver, outcome_sender = process_context.Pipe(duplex=False)
    open_attempt_receiver, open_attempt_sender = process_context.Pipe(duplex=False)
    target_open_attempts = 0

    def forbid_target_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal target_open_attempts
        if Path(path) == target or (path == target.name and dir_fd is not None):
            target_open_attempts += 1
            open_attempt_sender.send(("attempt", target_open_attempts))
            raise AssertionError("FIFO target must be classified without opening it")
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", forbid_target_open)

    def invoke() -> None:
        outcome_receiver.close()
        open_attempt_receiver.close()
        try:
            orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')
        except BaseException as exc:
            outcome_sender.send(
                (
                    type(exc).__name__,
                    getattr(exc, "stop_reason", None),
                    exc.__cause__ is None,
                    exc.__context__ is None,
                    str(getattr(exc, "message", exc)),
                )
            )
        else:
            outcome_sender.send(("RETURNED", None, True, True, ""))
        finally:
            open_attempt_sender.send(("final", target_open_attempts))
            open_attempt_sender.close()
            outcome_sender.close()

    process = process_context.Process(target=invoke, name="fifo-publication-probe")
    process.start()
    open_attempt_sender.close()
    outcome_sender.close()
    timed_out = False
    outcome: tuple[str, str | None, bool, bool, str] | None = None
    open_attempt_evidence: list[tuple[str, int]] = []
    open_attempt_evidence_complete = False
    process_alive_after_cleanup = False
    try:
        process.join(timeout=1.0)
        timed_out = process.is_alive()
        if not timed_out and outcome_receiver.poll(timeout=0.5):
            outcome = outcome_receiver.recv()
        while open_attempt_receiver.poll(timeout=0.5):
            try:
                open_attempt_evidence.append(open_attempt_receiver.recv())
            except EOFError:
                open_attempt_evidence_complete = True
                break
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        process_alive_after_cleanup = process.is_alive()
        open_attempt_receiver.close()
        outcome_receiver.close()

    try:
        assert not timed_out, "atomic publication blocked on a FIFO target"
        assert not process_alive_after_cleanup, "FIFO publication probe survived forced cleanup"
        assert process.exitcode == 0
        assert outcome is not None
        assert open_attempt_evidence_complete
        assert open_attempt_evidence == [("final", 0)]
        error_type, stop_reason, cause_clear, context_clear, message = outcome
        assert error_type == "InvoiceAgentsError"
        assert stop_reason == "ARTIFACT_PUBLICATION_TARGET_UNSAFE"
        assert cause_clear
        assert context_clear
        assert "private" not in message
        assert stat.S_ISFIFO(os.lstat(target).st_mode)
        assert _directory_entry_inventory(output) == prior_inventory
    finally:
        if target.exists():
            target.unlink()


@pytest.mark.parametrize(
    "injected",
    [
        PermissionError(errno.EACCES, "lstat private publication sentinel"),
        FileNotFoundError(errno.ESTALE, "stale lstat private publication sentinel"),
    ],
    ids=["permission", "file-not-found-non-enoent"],
)
def test_atomic_publication_lstat_uncertainty_is_chainless_and_dirfd_relative(
    injected: OSError,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_lstat_error.json"
    target.write_bytes(b'{"generation":"prior"}\n')
    prior_inventory = _directory_entry_inventory(output)
    original_lstat = os.lstat
    observed: list[tuple[object, int | None]] = []

    def fail_target_lstat(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> os.stat_result:
        if path == target.name or Path(path) == target:
            observed.append((path, dir_fd))
            raise injected
        return original_lstat(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "lstat", fail_target_lstat)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')

    _assert_atomic_namespace_rejected(
        excinfo.value,
        stop_reason="ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED",
    )
    assert len(observed) == 1
    observed_path, observed_dir_fd = observed[0]
    assert observed_path == target.name
    assert observed_dir_fd is not None
    assert target.read_bytes() == b'{"generation":"prior"}\n'
    assert _directory_entry_inventory(output) == prior_inventory


def test_atomic_publication_treats_only_exact_enoent_lstat_as_target_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_exact_enoent.json"
    payload = b'{"generation":"candidate"}\n'
    original_lstat = os.lstat
    observed: list[tuple[object, int | None]] = []

    def inject_exact_enoent_once(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> os.stat_result:
        if path == target.name and dir_fd is not None and not observed:
            observed.append((path, dir_fd))
            raise FileNotFoundError(errno.ENOENT, "exact absence publication sentinel")
        return original_lstat(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "lstat", inject_exact_enoent_once)

    orchestration._atomic_publish(target, payload)

    assert len(observed) == 1
    observed_path, observed_dir_fd = observed[0]
    assert observed_path == target.name
    assert observed_dir_fd is not None
    assert target.read_bytes() == payload
    assert set(_directory_entry_inventory(output)) == {target.name}


def test_atomic_publication_rejects_identity_swap_after_target_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_identity_swap.json"
    displaced_prior = output / "displaced-prior.json"
    intruder = output / "intruder.json"
    prior = b'{"generation":"prior"}\n'
    intruder_bytes = b'{"generation":"intruder"}\n'
    target.write_bytes(prior)
    intruder.write_bytes(intruder_bytes)
    prior_inventory = _directory_entry_inventory(output)
    original_lstat = os.lstat
    original_replace = os.replace
    classifications = 0

    def swap_after_first_lstat(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> os.stat_result:
        nonlocal classifications
        identity = original_lstat(path, dir_fd=dir_fd)
        if path == target.name and dir_fd is not None and classifications == 0:
            classifications += 1
            original_replace(target, displaced_prior)
            original_replace(intruder, target)
        return identity

    monkeypatch.setattr(os, "lstat", swap_after_first_lstat)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        orchestration._atomic_publish(target, b'{"generation":"candidate"}\n')

    _assert_atomic_namespace_rejected(
        excinfo.value,
        stop_reason="ARTIFACT_PUBLICATION_NAMESPACE_UNRESOLVED",
    )
    assert classifications == 1
    assert target.read_bytes() == intruder_bytes
    assert displaced_prior.read_bytes() == prior
    assert _directory_entry_inventory(output) == {
        target.name: prior_inventory[intruder.name],
        displaced_prior.name: prior_inventory[target.name],
    }


def test_atomic_publication_does_not_overwrite_identity_swapped_during_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_rollback_identity_swap.json"
    temporary = target.with_name(f"{target.name}.tmp")
    displaced_candidate = output / "displaced-candidate.json"
    intruder = output / "rollback-intruder.json"
    prior = b'{"generation":"prior"}\n'
    candidate = b'{"generation":"candidate"}\n'
    intruder_bytes = b'{"generation":"intruder"}\n'
    target.write_bytes(prior)
    intruder.write_bytes(intruder_bytes)
    prior_inventory = _directory_entry_inventory(output)
    original_open = os.open
    original_fsync = os.fsync
    original_replace = os.replace
    directory_descriptors: set[int] = set()
    candidate_replaced = False
    candidate_identity: tuple[int, int, int] | None = None
    publication_faults = 0
    prior_evidence_syncs = 0
    rollback_swaps = 0
    rollback_path: Path | None = None

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = (
            original_open(path, flags, mode)
            if dir_fd is None
            else original_open(path, flags, mode, dir_fd=dir_fd)
        )
        resolved_path = Path(path) if dir_fd is None else output / Path(path)
        if resolved_path == output:
            directory_descriptors.add(descriptor)
        return descriptor

    def fail_publication_sync(descriptor: int) -> None:
        nonlocal prior_evidence_syncs, publication_faults
        if descriptor in directory_descriptors and candidate_replaced and publication_faults == 0:
            publication_faults += 1
            raise OSError("directory durability private publication sentinel")
        original_fsync(descriptor)
        if descriptor in directory_descriptors and rollback_swaps:
            prior_evidence_syncs += 1

    def swap_before_rollback(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal candidate_identity, candidate_replaced, rollback_path, rollback_swaps
        source_path = Path(source) if src_dir_fd is None else output / Path(source)
        destination_path = Path(destination) if dst_dir_fd is None else output / Path(destination)
        if source_path == temporary and destination_path == target:
            original_replace(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            replaced_identity = os.lstat(target)
            candidate_identity = (
                replaced_identity.st_dev,
                replaced_identity.st_ino,
                stat.S_IFMT(replaced_identity.st_mode),
            )
            candidate_replaced = True
            return
        if destination_path == target and source_path != temporary:
            rollback_swaps += 1
            rollback_path = source_path
            original_replace(target, displaced_candidate)
            original_replace(intruder, target)
            raise OSError("rollback identity private publication sentinel")
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "fsync", fail_publication_sync)
    monkeypatch.setattr(os, "replace", swap_before_rollback)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        orchestration._atomic_publish(target, candidate)

    _assert_atomic_namespace_rejected(
        excinfo.value,
        stop_reason="ARTIFACT_PUBLICATION_DURABILITY_UNRESOLVED",
    )
    assert publication_faults == 1
    assert rollback_swaps == 1
    assert prior_evidence_syncs == 1
    assert candidate_identity is not None
    assert target.read_bytes() == intruder_bytes
    assert displaced_candidate.read_bytes() == candidate
    assert rollback_path is not None
    assert rollback_path.read_bytes() == prior
    rollback_identity = os.lstat(rollback_path)
    prior_identity = prior_inventory[target.name]
    assert (
        rollback_identity.st_dev,
        rollback_identity.st_ino,
        stat.S_IFMT(rollback_identity.st_mode),
    ) == prior_identity[:3]
    final_inventory = _directory_entry_inventory(output)
    assert set(final_inventory) == {
        target.name,
        displaced_candidate.name,
        rollback_path.name,
    }
    assert final_inventory[target.name] == prior_inventory[intruder.name]
    assert final_inventory[displaced_candidate.name][:3] == candidate_identity
    prior_identity_entries = [
        entry_name
        for entry_name, identity in final_inventory.items()
        if identity[:3] == prior_identity[:3] and (output / entry_name).read_bytes() == prior
    ]
    prior_byte_entries = [
        entry_name
        for entry_name, identity in final_inventory.items()
        if stat.S_ISREG(identity[2]) and (output / entry_name).read_bytes() == prior
    ]
    assert prior_identity_entries == [rollback_path.name]
    assert prior_byte_entries == [rollback_path.name]


def test_atomic_publication_serializes_same_target_transactions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts" / "results"
    output.mkdir(parents=True)
    target = output / "case_serialized.json"
    first_payload = b'{"generation":"first"}\n'
    second_payload = b'{"generation":"second"}\n'
    original_open = os.open
    original_fsync = os.fsync
    original_replace = os.replace
    directory_descriptors: set[int] = set()
    first_replaced = Event()
    release_first = Event()
    second_arrival_boundary = Barrier(2)
    second_invoking_publication = Event()
    second_entered_transaction = Event()
    failures: list[BaseException] = []

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if (
            resolved == target.with_name(f"{target.name}.tmp")
            and current_thread().name == "second-publisher"
        ):
            second_entered_transaction.set()
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if resolved == output:
            directory_descriptors.add(descriptor)
        return descriptor

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
        destination_path = (
            Path(destination) if dst_dir_fd is None else output / Path(destination)
        )
        if destination_path == target and current_thread().name == "first-publisher":
            first_replaced.set()

    def block_first_publication_sync(descriptor: int) -> None:
        if (
            descriptor in directory_descriptors
            and current_thread().name == "first-publisher"
            and first_replaced.is_set()
        ):
            assert release_first.wait(timeout=2.0)
        original_fsync(descriptor)

    def publish(payload: bytes) -> None:
        try:
            if current_thread().name == "second-publisher":
                second_arrival_boundary.wait(timeout=2.0)
                second_invoking_publication.set()
            orchestration._atomic_publish(target, payload)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "replace", observe_replace)
    monkeypatch.setattr(os, "fsync", block_first_publication_sync)
    first = Thread(target=publish, args=(first_payload,), name="first-publisher")
    second = Thread(target=publish, args=(second_payload,), name="second-publisher")
    first.start()
    second_started = False
    second_crossed_exclusion_boundary: bool | None = None
    try:
        assert first_replaced.wait(timeout=2.0)
        second.start()
        second_started = True
        second_arrival_boundary.wait(timeout=2.0)
        assert second_invoking_publication.wait(timeout=2.0)
        second_crossed_exclusion_boundary = second_entered_transaction.wait(timeout=1.0)
    finally:
        release_first.set()
        first.join(timeout=2.0)
        if second_started:
            second.join(timeout=2.0)

    assert second_crossed_exclusion_boundary is False
    assert not first.is_alive()
    assert second_started
    assert not second.is_alive()
    assert failures == []
    assert second_entered_transaction.is_set()
    assert target.read_bytes() == second_payload
    assert set(output.iterdir()) == {target}


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

    assert lease_was_durable
    assert captured[0] is not None
    snapshot = WorkflowStore(settings).load_case_execution_snapshot(outcome)
    assert snapshot is not None and snapshot.execution_state == "FINISHED"
    assert snapshot.result is not None and snapshot.result.stop_reason == "CANCELLED"


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
    handle = await registry.start_resume(case_id, settings)
    lease_was_durable = WorkflowStore(settings).has_valid_execution_lease(case_id)
    assert handle.task is not None
    await asyncio.wait_for(started.wait(), timeout=0.2)
    handle.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.task

    assert lease_was_durable
    assert captured[0] is not None
    snapshot = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert snapshot is not None and snapshot.execution_state == "FINISHED"
    assert snapshot.result is not None and snapshot.result.stop_reason == "CANCELLED"


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
    terminal_snapshots = [
        WorkflowStore(settings).load_case_execution_snapshot(case_id)
        for case_id in prepared_case_ids
    ]

    assert durable_before_schedule == [True, True]
    assert len(captured) == 2
    assert all(claim is not None for claim in captured)
    assert all(
        snapshot is not None
        and snapshot.execution_state == "FINISHED"
        and snapshot.result is not None
        and snapshot.result.stop_reason == "CANCELLED"
        for snapshot in terminal_snapshots
    )


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

    real_prepare = orchestration.prepare_claimed_invoice
    prepared: list[tuple[str, datetime, ExecutionClaim]] = []
    calls = 0

    async def abort_second(path: Path, selected_settings: Settings) -> object:
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

    monkeypatch.setattr(orchestration, "prepare_claimed_invoice_async", abort_second)
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
            source_id=store.load_case_source_id(case_id),
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
        replace_claim: Callable[[ExecutionClaim], None],
        lease_seconds: int,
        renewal_interval_seconds: float,
    ) -> CaseResult:
        del renew, renewal_interval_seconds
        return await original_heartbeat(
            operation,
            renew=exit_renewal,
            claim=claim,
            replace_claim=replace_claim,
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

    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
    monkeypatch.chdir(tmp_path)

    result = await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "MAX_MESSAGES_EXHAUSTED"
    assert result.errors[-1].stop_reason == "TERMINAL_PERSISTENCE_FAILED"
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert recovery.is_file()
    assert not (recovery.parent / f"{case_id}.recovery.json.tmp").exists()
    payload = json.loads(recovery.read_text(encoding="utf-8"))
    assert payload["recovery_format"] == 2
    assert payload["case_id"] == case_id
    assert payload["case_result"]["case_id"] == case_id
    assert payload["terminal_persistence_error"]["stop_reason"] == ("TERMINAL_PERSISTENCE_FAILED")
    assert "sk-proj-secret" not in recovery.read_text(encoding="utf-8")
    assert WorkflowStore(settings).load_result(case_id) is None


def _terminal_audit_payloads(settings: Settings, case_id: str) -> list[dict[str, object]]:
    with connect_database(settings.workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM events WHERE case_id = ? "
            "AND event_type = 'case.finished' ORDER BY rowid",
            (case_id,),
        ).fetchall()
    return [json.loads(str(row["payload_json"])) for row in rows]


async def _assert_terminal_stream_is_reconciled(
    *,
    settings: Settings,
    case_id: str,
    expected: CaseResult,
    forbidden_candidate_stop: str | None = None,
) -> None:
    stream = sse.case_event_stream(
        settings.workflow_db,
        case_id,
        RunRegistry(),
        settings=settings,
    )

    async def consume() -> list[ServerSentEvent]:
        return [event async for event in stream]

    try:
        events = await asyncio.wait_for(consume(), timeout=0.5)
    finally:
        await asyncio.wait_for(stream.aclose(), timeout=0.2)
    terminal_events = [event for event in events if event.event == "terminal"]
    assert len(terminal_events) == 1
    assert events[-1] is terminal_events[0]
    terminal_payload_data = {
        "case_id": case_id,
        "status": str(expected.status),
        "stop_reason": expected.stop_reason,
        "run_error": None,
    }
    assert json.loads(str(terminal_events[0].data)) == terminal_payload_data
    assert (
        terminal_events[0].encode()
        == (
            "event: terminal\r\n"
            f"data: {json.dumps(terminal_payload_data, ensure_ascii=False, default=str)}\r\n\r\n"
        ).encode()
    )
    terminal_case_events = []
    for event in events:
        if event.event != "case-event":
            continue
        encoded = event.encode()
        assert encoded.startswith(b"id: ")
        assert b"\r\nevent: case-event\r\ndata: " in encoded
        payload = json.loads(str(event.data))
        if payload["event_type"] in {"case.finished", "case.resumed_finished", "case.failed"}:
            terminal_case_events.append(payload)
    assert [(payload["status"], payload["stop_reason"]) for payload in terminal_case_events] == [
        (str(expected.status), expected.stop_reason)
    ]
    if forbidden_candidate_stop is not None:
        assert forbidden_candidate_stop.encode("utf-8") not in b"".join(
            event.encode() for event in events
        )


async def _assert_terminal_result_views_are_identical(
    *,
    settings: Settings,
    case_id: str,
    tmp_path: Path,
    expected: CaseResult,
    forbidden_candidate_stop: str | None = None,
) -> None:
    stored = WorkflowStore(settings).load_result(case_id)
    assert stored == expected
    artifact = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    assert CaseResult.model_validate_json(artifact.read_text(encoding="utf-8")) == expected
    assert _terminal_audit_payloads(settings, case_id) == [expected.model_dump(mode="json")]
    await _assert_terminal_stream_is_reconciled(
        settings=settings,
        case_id=case_id,
        expected=expected,
        forbidden_candidate_stop=forbidden_candidate_stop,
    )
    with TestClient(create_app(settings)) as client:
        response = client.get(f"/cases/{case_id}/result.json")
    assert response.status_code == 200
    assert CaseResult.model_validate(response.json()) == expected
    assert not artifact.with_name(f"{artifact.name}.tmp").exists()


def _assert_artifact_binding_conflict(
    status_code: int,
    payload: object,
    result: CaseResult,
) -> None:
    assert status_code == 409
    assert payload == {
        "case_id": result.case_id,
        "status": result.status.value,
        "stop_reason": "RESULT_ARTIFACT_BINDING_UNRESOLVED",
    }


def _load_result_artifact_binding(store: WorkflowStore, case_id: str) -> object | None:
    loader = getattr(store, "load_result_artifact_binding", None)
    assert callable(loader), "WorkflowStore must expose durable result-artifact bindings"
    return loader(case_id)


def _binding_field(binding: object, field: str) -> object:
    assert hasattr(binding, field), f"result-artifact binding is missing {field}"
    return getattr(binding, field)


async def _publish_case_for_artifact_binding(
    *,
    case_id: str,
    started_at: datetime,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> CaseResult:
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    claim = WorkflowStore(settings).claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        orchestration.EXECUTION_LEASE_SECONDS,
    )
    return await orchestration._run_prepared_case_in_process(
        case_id,
        started_at,
        settings,
        claim=claim,
        terminal_writes_in_process=True,
    )


@pytest.mark.asyncio
async def test_result_route_never_serves_candidate_before_directory_durability(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    output = tmp_path / "artifacts" / "results"
    target = output / f"{case_id}.json"
    original_open = os.open
    original_fsync = os.fsync
    original_replace = os.replace
    directory_descriptors: set[int] = set()
    candidate_replaced = Event()
    publication_blocked = Event()
    release_publication = Event()

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = (
            original_open(path, flags, mode)
            if dir_fd is None
            else original_open(path, flags, mode, dir_fd=dir_fd)
        )
        resolved_path = Path(path) if dir_fd is None else output / Path(path)
        if resolved_path == output:
            directory_descriptors.add(descriptor)
        return descriptor

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
        destination_path = (
            Path(destination) if dst_dir_fd is None else output / Path(destination)
        )
        if destination_path == target:
            candidate_replaced.set()

    def block_pre_durability(descriptor: int) -> None:
        if (
            descriptor in directory_descriptors
            and candidate_replaced.is_set()
            and not publication_blocked.is_set()
        ):
            publication_blocked.set()
            assert release_publication.wait(timeout=3.0)
        original_fsync(descriptor)

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "replace", observe_replace)
    monkeypatch.setattr(os, "fsync", block_pre_durability)
    monkeypatch.chdir(tmp_path)
    claim = WorkflowStore(settings).claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        orchestration.EXECUTION_LEASE_SECONDS,
    )

    def invoke() -> CaseResult:
        return asyncio.run(
            orchestration._run_prepared_case_in_process(
                case_id,
                started_at,
                settings,
                claim=claim,
                terminal_writes_in_process=True,
            )
        )

    process_context = get_context("fork")
    command_receiver, command_sender = process_context.Pipe(duplex=False)
    response_receiver, response_sender = process_context.Pipe(duplex=False)

    def fetch_result_in_process() -> None:
        command_sender.close()
        response_receiver.close()
        try:
            command = command_receiver.recv()
            if command != "fetch":
                raise AssertionError("invalid route probe command")
            with TestClient(create_app(settings)) as client:
                response = client.get(f"/cases/{case_id}/result.json")
            response_sender.send(("response", response.status_code, response.json()))
        except BaseException as exc:
            response_sender.send(("error", type(exc).__name__, None))
        finally:
            command_receiver.close()
            response_sender.close()

    route_process = process_context.Process(
        target=fetch_result_in_process,
        name="result-route-publication-probe",
    )
    route_process.start()
    command_receiver.close()
    response_sender.close()
    publication = asyncio.create_task(asyncio.to_thread(invoke))
    route_outcome: tuple[str, object, object] | None = None
    route_timed_out = False
    route_terminated = False
    result: CaseResult | None = None

    try:
        assert await asyncio.to_thread(publication_blocked.wait, 2.0)
        stored_during_publication = WorkflowStore(settings).load_result(case_id)
        assert stored_during_publication is not None
        assert target.is_file()
        command_sender.send("fetch")
        if response_receiver.poll(timeout=1.0):
            route_outcome = response_receiver.recv()
        else:
            route_timed_out = True
    finally:
        release_publication.set()
        try:
            result = await asyncio.wait_for(publication, timeout=3.0)
        finally:
            command_sender.close()
            route_process.join(timeout=1.0)
            if route_process.is_alive():
                route_process.terminate()
                route_process.join(timeout=1.0)
            if route_process.is_alive():
                route_process.kill()
                route_process.join(timeout=1.0)
            route_terminated = not route_process.is_alive()
            response_receiver.close()

    assert not route_timed_out, "result route blocked behind an in-progress publication"
    assert route_terminated
    assert route_process.exitcode == 0
    assert route_outcome is not None
    outcome_kind, status_code, payload = route_outcome
    assert outcome_kind == "response"
    assert type(status_code) is int
    assert result is not None
    _assert_artifact_binding_conflict(
        status_code,
        payload,
        stored_during_publication,
    )
    with TestClient(create_app(settings)) as client:
        response_after_durability = client.get(f"/cases/{case_id}/result.json")
    assert response_after_durability.status_code == 200
    assert CaseResult.model_validate(response_after_durability.json()) == result


@pytest.mark.parametrize("mutation", ["absent", "hash_mismatch", "identity_mismatch"])
@pytest.mark.asyncio
async def test_result_route_conflicts_when_durable_artifact_binding_is_not_exact(
    mutation: str,
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.chdir(tmp_path)
    result = await _publish_case_for_artifact_binding(
        case_id=case_id,
        started_at=started_at,
        settings=settings,
        monkeypatch=monkeypatch,
    )
    target = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    published_bytes = target.read_bytes()
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/cases/{case_id}/result.json").status_code == 200

    if mutation == "absent":
        target.unlink()
    elif mutation == "hash_mismatch":
        target.write_bytes(b'{"case_id":"tampered"}\n')
    else:
        replacement = target.with_name(f"{target.name}.replacement")
        replacement.write_bytes(published_bytes)
        os.replace(replacement, target)

    with TestClient(create_app(settings)) as client:
        response = client.get(f"/cases/{case_id}/result.json")
    _assert_artifact_binding_conflict(response.status_code, response.json(), result)


@pytest.mark.asyncio
async def test_new_terminal_database_generation_cannot_advertise_seeded_prior_artifact(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    monkeypatch.chdir(tmp_path)
    first = await _publish_case_for_artifact_binding(
        case_id=case_id,
        started_at=started_at,
        settings=settings,
        monkeypatch=monkeypatch,
    )
    target = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    temporary = target.with_name(f"{target.name}.tmp")
    prior_bytes = target.read_bytes()
    prior_identity = os.lstat(target)
    store = WorkflowStore(settings)
    first_snapshot = store.load_case_execution_snapshot(case_id)
    assert first_snapshot is not None
    first_binding = _load_result_artifact_binding(store, case_id)
    assert first_binding is not None
    assert _binding_field(first_binding, "case_id") == case_id
    assert _binding_field(first_binding, "execution_generation") == (
        first_snapshot.execution_generation
    )
    assert _binding_field(first_binding, "artifact_sha256") == sha256(prior_bytes).hexdigest()
    assert _binding_field(first_binding, "artifact_device") == prior_identity.st_dev
    assert _binding_field(first_binding, "artifact_inode") == prior_identity.st_ino
    assert _binding_field(first_binding, "artifact_size_bytes") == len(prior_bytes)
    original_replace = os.replace
    replace_failures = 0

    def fail_candidate_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal replace_failures
        source_path = Path(source) if src_dir_fd is None else target.parent / Path(source)
        destination_path = (
            Path(destination) if dst_dir_fd is None else target.parent / Path(destination)
        )
        if source_path == temporary and destination_path == target:
            replace_failures += 1
            raise OSError("pre-replace private publication sentinel")
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", fail_candidate_replace)
    second = await _publish_case_for_artifact_binding(
        case_id=case_id,
        started_at=first.started_at,
        settings=settings,
        monkeypatch=monkeypatch,
    )

    assert replace_failures == 1
    assert second != first
    assert second.errors[-1].stop_reason == "RESULT_ARTIFACT_WRITE_FAILED"
    second_snapshot = store.load_case_execution_snapshot(case_id)
    assert second_snapshot is not None
    assert second_snapshot.execution_generation > first_snapshot.execution_generation
    assert store.load_result(case_id) == second
    assert _load_result_artifact_binding(store, case_id) is None
    assert target.read_bytes() == prior_bytes
    current_identity = os.lstat(target)
    assert (current_identity.st_dev, current_identity.st_ino) == (
        prior_identity.st_dev,
        prior_identity.st_ino,
    )
    with TestClient(create_app(settings)) as client:
        response = client.get(f"/cases/{case_id}/result.json")
    _assert_artifact_binding_conflict(response.status_code, response.json(), second)


@pytest.mark.parametrize(
    ("fault_type", "expected_error_stop"),
    [
        (OSError, "RESULT_ARTIFACT_WRITE_FAILED"),
        (asyncio.CancelledError, "RESULT_ARTIFACT_WRITE_CANCELLED"),
    ],
)
@pytest.mark.asyncio
async def test_post_replace_fault_reconciles_result_audit_sse_and_json_route(
    fault_type: type[BaseException],
    expected_error_stop: str,
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No pre-fault candidate may remain visible through one terminal view."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    output = tmp_path / "artifacts" / "results"
    injected = fault_type("post-replace private publication sentinel")
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync
    next_token = 0
    current_tokens: dict[int, int] = {}
    token_paths: dict[int, Path] = {}
    acquired_tokens: list[int] = []
    closed_tokens: list[int] = []
    close_attempts: list[tuple[int, int | None]] = []
    retired_tokens: dict[int, list[int]] = {}
    fault_calls = 0

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal next_token
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == output or resolved.parent == output:
            next_token += 1
            current_tokens[descriptor] = next_token
            token_paths[next_token] = resolved
            acquired_tokens.append(next_token)
        return descriptor

    def observe_close(descriptor: int) -> None:
        token = current_tokens.get(descriptor)
        close_attempts.append((descriptor, token))
        if token is not None:
            del current_tokens[descriptor]
            closed_tokens.append(token)
            retired_tokens.setdefault(descriptor, []).append(token)
        original_close(descriptor)

    def fail_first_result_directory_sync(descriptor: int) -> None:
        nonlocal fault_calls
        token = current_tokens.get(descriptor)
        is_directory = token is not None and token_paths[token] == output
        if is_directory and fault_calls == 0:
            fault_calls += 1
            raise injected
        original_fsync(descriptor)

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    monkeypatch.setattr(os, "fsync", fail_first_result_directory_sync)
    monkeypatch.chdir(tmp_path)
    claim = WorkflowStore(settings).claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        orchestration.EXECUTION_LEASE_SECONDS,
    )

    raised: BaseException | None = None
    result: CaseResult | None = None
    leaked_descriptors_before_test_cleanup: dict[int, int] = {}
    try:
        try:
            result = await orchestration._run_prepared_case_in_process(
                case_id,
                started_at,
                settings,
                claim=claim,
                terminal_writes_in_process=True,
            )
        except BaseException as exc:
            raised = exc
    finally:
        leaked_descriptors_before_test_cleanup = dict(current_tokens)
        for descriptor in leaked_descriptors_before_test_cleanup:
            with suppress(OSError):
                original_close(descriptor)

    if fault_type is asyncio.CancelledError:
        assert raised is injected
        result = WorkflowStore(settings).load_result(case_id)
        assert result is not None
    else:
        assert raised is None
        assert result is not None

    assert fault_calls == 1
    assert leaked_descriptors_before_test_cleanup == {}
    assert sorted(closed_tokens) == sorted(acquired_tokens)
    assert len(closed_tokens) == len(set(closed_tokens))
    assert all(token is not None for _descriptor, token in close_attempts)
    assert sorted(token for tokens in retired_tokens.values() for token in tokens) == sorted(
        acquired_tokens
    )
    assert result.errors[-1].stop_reason == expected_error_stop
    assert "private" not in result.model_dump_json()
    await _assert_terminal_result_views_are_identical(
        settings=settings,
        case_id=case_id,
        tmp_path=tmp_path,
        expected=result,
        forbidden_candidate_stop=(
            "MAX_MESSAGES_EXHAUSTED" if fault_type is asyncio.CancelledError else None
        ),
    )


@pytest.mark.asyncio
async def test_unproven_post_replace_rollback_is_explicit_and_not_advertised(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    output = tmp_path / "artifacts" / "results"
    target = output / f"{case_id}.json"
    original_open = os.open
    original_fsync = os.fsync
    original_unlink = os.unlink
    directory_descriptors: set[int] = set()
    directory_faults = 0
    rollback_attempts = 0

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == output:
            directory_descriptors.add(descriptor)
        return descriptor

    def fail_first_result_directory_sync(descriptor: int) -> None:
        nonlocal directory_faults
        if descriptor in directory_descriptors and directory_faults == 0:
            directory_faults += 1
            raise OSError("directory durability private publication sentinel")
        original_fsync(descriptor)

    def fail_target_rollback_unlink(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal rollback_attempts
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == target:
            rollback_attempts += 1
            raise OSError("rollback private publication sentinel")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "fsync", fail_first_result_directory_sync)
    monkeypatch.setattr(os, "unlink", fail_target_rollback_unlink)
    monkeypatch.chdir(tmp_path)

    result = await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert directory_faults == 1
    assert rollback_attempts == 1
    assert result.status is CaseStatus.INCOMPLETE
    assert result.stop_reason == "RESULT_ARTIFACT_DURABILITY_UNRESOLVED"
    assert result.errors[-1].stop_reason == "RESULT_ARTIFACT_DURABILITY_UNRESOLVED"
    assert WorkflowStore(settings).load_result(case_id) == result
    assert _terminal_audit_payloads(settings, case_id) == [result.model_dump(mode="json")]
    await _assert_terminal_stream_is_reconciled(
        settings=settings,
        case_id=case_id,
        expected=result,
        forbidden_candidate_stop="MAX_MESSAGES_EXHAUSTED",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get(f"/cases/{case_id}/result.json")
    assert response.status_code == 409
    assert response.json() == {
        "case_id": case_id,
        "status": "INCOMPLETE",
        "stop_reason": "RESULT_ARTIFACT_DURABILITY_UNRESOLVED",
    }
    assert b"MAX_MESSAGES_EXHAUSTED" not in response.content
    assert target.is_file(), "the injected rollback fault must leave the candidate observable"
    assert not target.with_name(f"{target.name}.tmp").exists()
    assert "private" not in result.model_dump_json()


@pytest.mark.parametrize("cleanup_fault", ["file_close", "temp_unlink", "directory_close"])
@pytest.mark.asyncio
async def test_recovery_publication_cleanup_fault_is_stable_chainless_and_explicit(
    cleanup_fault: str,
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id, started_at = _prepared_case(invoice_dir, settings)
    output = tmp_path / "artifacts" / "results"
    target = output / f"{case_id}.recovery.json"
    temporary = target.with_name(f"{target.name}.tmp")
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync
    original_unlink = os.unlink
    temp_descriptor: int | None = None
    directory_descriptors: set[int] = set()
    fault_calls = 0

    def observe_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal temp_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if resolved == temporary:
            temp_descriptor = descriptor
        elif resolved == output:
            directory_descriptors.add(descriptor)
        return descriptor

    def fail_cleanup_close(descriptor: int) -> None:
        nonlocal fault_calls
        if cleanup_fault == "file_close" and descriptor == temp_descriptor and fault_calls == 0:
            original_close(descriptor)
            fault_calls += 1
            raise OSError("file close private recovery sentinel")
        if (
            cleanup_fault == "directory_close"
            and descriptor in directory_descriptors
            and fault_calls == 0
        ):
            original_close(descriptor)
            fault_calls += 1
            raise OSError("directory close private recovery sentinel")
        original_close(descriptor)

    def fail_temp_fsync(descriptor: int) -> None:
        if cleanup_fault == "temp_unlink" and descriptor == temp_descriptor:
            raise OSError("primary fsync private recovery sentinel")
        original_fsync(descriptor)

    def fail_temp_unlink(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal fault_calls
        resolved = _resolve_dirfd_test_path(path, dir_fd=dir_fd, directory=output)
        if cleanup_fault == "temp_unlink" and resolved == temporary and fault_calls == 0:
            fault_calls += 1
            raise OSError("temp unlink private recovery sentinel")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _ClosingClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: _MaxMessagesTeam())
    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", fail_cleanup_close)
    monkeypatch.setattr(os, "fsync", fail_temp_fsync)
    if cleanup_fault == "temp_unlink":
        monkeypatch.setattr(os, "unlink", fail_temp_unlink)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert fault_calls == 1
    assert excinfo.value.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert excinfo.value.details["artifact_publication_stop_reason"] == (
        "ARTIFACT_PUBLICATION_CLEANUP_UNRESOLVED"
    )
    assert "private" not in excinfo.value.message
    assert WorkflowStore(settings).load_result(case_id) is None
    assert not target.exists()
    if cleanup_fault == "temp_unlink":
        assert temporary.is_file()
    else:
        assert not temporary.exists()


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

    def fail_recovery_replace(
        _source: os.PathLike[str] | str, target: os.PathLike[str] | str
    ) -> None:
        assert str(target).endswith(f"{case_id}.recovery.json")
        raise OSError("recovery artifact private sentinel")

    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
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
    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
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
            timeout=2.0,
        )

    await asyncio.wait_for(close_finished.wait(), timeout=0.2)
    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None
    assert [error.stop_reason for error in result.errors] == [
        "CANCELLED",
        "CLIENT_CLOSE_TIMEOUT",
    ]


@pytest.mark.asyncio
async def test_cancellation_resistant_client_close_is_drained_without_task_cancellation() -> None:
    """Caller cancellation cannot cancel or abandon the owned close operation."""

    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_cancelled = asyncio.Event()

    class CancellationResistantClient:
        async def close(self) -> None:
            close_started.set()
            try:
                await release_close.wait()
            except asyncio.CancelledError:
                close_cancelled.set()
                await release_close.wait()

    execution = SimpleNamespace(
        client=CancellationResistantClient(),
        case_id="case_close_ownership",
    )
    close_owner = asyncio.create_task(orchestration._close_claimed_client(execution))
    await asyncio.wait_for(close_started.wait(), timeout=2.0)
    close_owner.cancel("outer cleanup cancellation")
    await asyncio.sleep(0)

    assert not close_owner.done()
    assert not close_cancelled.is_set()

    release_close.set()
    outcome = await asyncio.wait_for(close_owner, timeout=2.0)

    assert isinstance(outcome.control_exception, asyncio.CancelledError)
    assert outcome.control_exception.args == ("outer cleanup cancellation",)
    assert outcome.error is not None
    assert outcome.error.stop_reason == "CLIENT_CLOSE_CANCELLED"
    assert not close_cancelled.is_set()


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
    release_close = asyncio.Event()

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
                await release_close.wait()
            finally:
                close_finished.set()

    monkeypatch.setattr(
        orchestration, "create_model_client", lambda _settings: BlockingCloseClient()
    )
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: BlockingTeam())
    monkeypatch.chdir(tmp_path)
    assertion_timeout = 2.0
    run_task = asyncio.create_task(_run_prepared_with_new_claim(case_id, started_at, settings))
    await asyncio.wait_for(stream_started.wait(), timeout=assertion_timeout)
    run_task.cancel()
    await asyncio.wait_for(close_started.wait(), timeout=assertion_timeout)
    run_task.cancel()
    await asyncio.sleep(0)
    assert not run_task.done()
    assert not close_finished.is_set()
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=assertion_timeout)

    await asyncio.wait_for(close_finished.wait(), timeout=assertion_timeout)
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
    calls = 0

    original_terminal_process = orchestration.run_terminal_process

    def cancel_first_finish(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        nonlocal calls
        if kwargs["mode"] == "finish":
            calls += 1
            if calls == 1:
                raise asyncio.CancelledError
        return original_terminal_process(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(orchestration, "run_terminal_process", cancel_first_finish)
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
    modes: list[str] = []

    def fail_terminal_write(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        modes.append(str(kwargs["mode"]))
        return _failed_terminal_worker(**kwargs)

    monkeypatch.setattr(orchestration, "run_terminal_process", fail_terminal_write)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await _run_prepared_with_new_claim(case_id, started_at, settings)

    assert modes == ["finish"]
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
    """A fresh nonterminal generation retains only exact immutable payment evidence."""

    case_id = make_succeeded_case(settings)
    store = WorkflowStore(settings)
    committed = store.load_result(case_id)
    assert committed is not None
    assert committed.final_decision is not None and committed.payment is not None
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET status = 'INCOMPLETE', stop_reason = 'TORN_EXECUTION', "
            "result_json = NULL, finished_at = NULL, execution_token = NULL, "
            "execution_state = 'IDLE', lease_expires_at = NULL WHERE case_id = ?",
            (case_id,),
        )
        connection.commit()
    assert store.load_result(case_id) is None
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
async def test_cancel_unstarted_reconciliation_drains_repeated_cancellation_before_recovery(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No cancellation can escape between helper completion and recovery publication."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    publication_started = Event()
    release_publication = Event()
    calls: list[str] = []
    real_publish = orchestration._recovery_artifact_or_raise
    real_inspect = orchestration._inspect_exact_claim_evidence
    publication_owner: Thread | None = None

    def controlled_boundary(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        nonlocal publication_owner
        mode = str(kwargs["mode"])
        calls.append(mode)
        if mode == "cancel_unstarted":
            return SimpleNamespace(
                result=None,
                error_code="TERMINAL_WORKER_FAILED",
                evidence_state="RECOVERABLE_RUNNING",
                evidence_result=None,
            )  # type: ignore[return-value]
        assert mode == "publish_cancel_recovery"
        publication_owner = current_thread()
        publication_started.set()
        assert release_publication.wait(2.0)
        selected_store = WorkflowStore(settings)
        source_id = selected_store.load_authoritative_case_source_id(claim)
        previous = selected_store.load_result(case_id)
        result = orchestration._cancelled_result(case_id, source_id, started_at, previous)
        result = selected_store.merge_relational_case_evidence(result)
        error = orchestration._canonical_recovery_persistence_error(
            case_id,
            "TERMINAL_PERSISTENCE_FAILED",
        )
        real_publish(
            result,
            error,
            store=selected_store,
            claim=claim,
        )
        return SimpleNamespace(
            result=result,
            error_code=None,
            evidence_state="RECOVERABLE_RUNNING",
            evidence_result=None,
        )  # type: ignore[return-value]

    def helper_owned_inspection(
        selected_store: WorkflowStore,
        selected_claim: ExecutionClaim,
    ) -> object:
        if current_thread() is not publication_owner:
            raise AssertionError("parent-owned cancellation reconciliation escaped its helper")
        return real_inspect(selected_store, selected_claim)

    monkeypatch.setattr(orchestration, "run_terminal_process", controlled_boundary)
    monkeypatch.setattr(
        orchestration,
        "_inspect_exact_claim_evidence",
        helper_owned_inspection,
    )
    monkeypatch.setattr(
        orchestration,
        "_recovery_artifact_or_raise",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parent-owned recovery publication escaped its helper")
        ),
    )
    monkeypatch.chdir(tmp_path)
    task = asyncio.create_task(
        orchestration._durably_cancel_unstarted_claim(
            case_id,
            started_at,
            settings,
            claim,
        )
    )

    try:
        assert await asyncio.to_thread(publication_started.wait, 2.0)
        task.cancel("first cancel-unstarted reconciliation cancellation")
        await asyncio.sleep(0)
        task.cancel("second cancel-unstarted reconciliation cancellation")
        await asyncio.sleep(0)
        assert not task.done()
        release_publication.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, timeout=2.0)
    finally:
        release_publication.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert raised.value.args == ("first cancel-unstarted reconciliation cancellation",)
    assert calls == ["cancel_unstarted", "publish_cancel_recovery"]
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    payload = json.loads(recovery.read_text(encoding="utf-8"))
    assert payload["case_result"]["stop_reason"] == "TERMINAL_PERSISTENCE_FAILED"
    assert payload["terminal_persistence_error"]["stop_reason"] == (
        "TERMINAL_PERSISTENCE_FAILED"
    )
    assert store.load_result(case_id) is None


@pytest.mark.asyncio
async def test_cancel_unstarted_deadline_drains_owner_and_starts_no_later_stage(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired shared deadline drains its owner and admits no new helper stage."""

    case_id, started_at = _prepared_case(invoice_dir, settings)
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    owner_finished = Event()
    calls: list[str] = []

    def consume_shared_deadline(**kwargs: object) -> terminal_process.TerminalProcessOutcome:
        calls.append(str(kwargs["mode"]))
        try:
            assert kwargs["mode"] == "cancel_unstarted"
            Event().wait(0.04)
            return SimpleNamespace(
                result=None,
                error_code="TERMINAL_WORKER_FAILED",
                evidence_state="RECOVERABLE_RUNNING",
                evidence_result=None,
            )  # type: ignore[return-value]
        finally:
            owner_finished.set()

    monkeypatch.setattr(orchestration, "run_terminal_process", consume_shared_deadline)
    monkeypatch.setattr(orchestration, "DURABILITY_DEADLINE_SECONDS", 0.01)
    monkeypatch.setattr(orchestration, "TERMINAL_WORKER_CLEANUP_GRACE_SECONDS", 0.01)
    started = asyncio.get_running_loop().time()

    with pytest.raises(InvoiceAgentsError) as raised:
        await orchestration._durably_cancel_unstarted_claim(case_id, started_at, settings, claim)

    assert raised.value.stop_reason == "TERMINAL_DURABILITY_TIMEOUT"
    assert asyncio.get_running_loop().time() - started >= 0.035
    assert owner_finished.is_set()
    assert calls == ["cancel_unstarted"]


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
        await asyncio.wait_for(
            batch.task,
            timeout=(
                (3 * ui_runs.DURABILITY_DEADLINE_SECONDS)
                + ui_runs.TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                + 1.0
            ),
        )

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
async def test_ui_batch_cancel_during_preparation_accounts_for_owned_async_claim(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation cannot abandon a claim owned by the async prep boundary."""

    second_started = asyncio.Event()
    prepared: list[tuple[str, datetime, ExecutionClaim]] = []
    calls = 0

    async def controlled_prepare(path: Path, selected_settings: Settings) -> object:
        nonlocal calls
        calls += 1
        outcome = orchestration.prepare_claimed_invoice(path, selected_settings)
        if isinstance(outcome, tuple):
            prepared.append(outcome)
        if calls == 2:
            second_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert isinstance(outcome, tuple)
                await orchestration._durably_cancel_unstarted_claim(
                    outcome[0],
                    outcome[1],
                    selected_settings,
                    outcome[2],
                )
                raise
        return outcome

    monkeypatch.setattr(ui_runs, "prepare_claimed_invoice_async", controlled_prepare)
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    start_task = asyncio.create_task(
        registry.start_batch(
            [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
            settings,
            concurrency=1,
        )
    )
    await asyncio.wait_for(second_started.wait(), timeout=1)

    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=2)

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
        await asyncio.wait_for(
            batch.task,
            timeout=(
                (3 * ui_runs.DURABILITY_DEADLINE_SECONDS)
                + ui_runs.TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                + 1.0
            ),
        )

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
            await asyncio.wait_for(
                batch.task,
                timeout=(
                    (3 * ui_runs.DURABILITY_DEADLINE_SECONDS)
                    + ui_runs.TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                    + 1.0
                ),
            )
    else:
        await asyncio.wait_for(
            batch.task,
            timeout=(
                (3 * ui_runs.DURABILITY_DEADLINE_SECONDS)
                + ui_runs.TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                + 1.0
            ),
        )

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

    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
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
    assert handle is not None and handle.state == "unresolved"


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
    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)

    batch.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(batch.task, timeout=1.0)

    for entry in batch.entries:
        recovery = tmp_path / "artifacts" / "results" / f"{entry.case_id}.recovery.json"
        payload = json.loads(recovery.read_text(encoding="utf-8"))
        assert payload["case_result"]["status"] == "INCOMPLETE"
        assert payload["case_result"]["stop_reason"] == "TERMINAL_PERSISTENCE_FAILED"
        assert payload["terminal_persistence_error"]["stop_reason"] == (
            "TERMINAL_PERSISTENCE_FAILED"
        )
        assert payload["case_result"]["errors"][-1] == payload["terminal_persistence_error"]
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

    publications: list[str] = []
    original_publish = orchestration._write_recovery_artifact

    def count_publish(result: CaseResult, error: ErrorRecord) -> Path:
        publications.append(result.case_id)
        return original_publish(result, error)

    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
    monkeypatch.setattr(orchestration, "_write_recovery_artifact", count_publish)
    monkeypatch.chdir(tmp_path)

    result = await orchestration.run_prepared_case(case_id, started_at, settings, claim=claim)

    assert result.errors[-1].stop_reason == "TERMINAL_PERSISTENCE_FAILED"
    assert publications == [case_id]
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert recovery.is_file()
    assert not (recovery.parent / f"{case_id}.json").exists()
    assert list(recovery.parent.glob(f"{case_id}*.tmp")) == []


@pytest.mark.asyncio
async def test_core_batch_preserves_recovery_only_preparation_publication(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovery-only preparation outcome must never gain a normal result file."""

    monkeypatch.setattr(
        orchestration,
        "extract_invoice_evidence",
        lambda _source: (_ for _ in ()).throw(RuntimeError("preparation sentinel")),
    )
    monkeypatch.setattr(
        WorkflowStore,
        "finish_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("terminal write sentinel")
        ),
    )
    monkeypatch.chdir(tmp_path)

    results = await orchestration.process_batch(
        [invoice_dir / "invoice_1001.txt"], settings, concurrency=1
    )

    assert len(results) == 1
    case_id = results[0].case_id
    output = tmp_path / "artifacts" / "results"
    assert (output / f"{case_id}.recovery.json").is_file()
    assert not (output / f"{case_id}.json").exists()
    assert list(output.glob(f"{case_id}*.tmp")) == []


@pytest.mark.parametrize(
    ("fault_type", "expected_control"),
    [
        (RuntimeError, None),
        (asyncio.CancelledError, asyncio.CancelledError),
        (SystemExit, SystemExit),
        (KeyboardInterrupt, KeyboardInterrupt),
    ],
)
@pytest.mark.asyncio
async def test_core_batch_post_claim_preparation_failure_has_one_recovery_owner(
    fault_type: type[BaseException],
    expected_control: type[BaseException] | None,
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every post-claim control path keeps recovery-only publication exclusive."""

    monkeypatch.setattr(
        orchestration,
        "extract_invoice_evidence",
        lambda _source: (_ for _ in ()).throw(fault_type("preparation sentinel")),
    )
    monkeypatch.setattr(
        WorkflowStore,
        "finish_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("terminal write sentinel")
        ),
    )
    monkeypatch.chdir(tmp_path)

    if expected_control is None:
        results = await orchestration.process_batch(
            [invoice_dir / "invoice_1001.txt"], settings, concurrency=1
        )
        assert len(results) == 1
    else:
        with pytest.raises(expected_control):
            await orchestration.process_batch(
                [invoice_dir / "invoice_1001.txt"], settings, concurrency=1
            )

    output = tmp_path / "artifacts" / "results"
    recoveries = list(output.glob("case_*.recovery.json"))
    assert len(recoveries) == 1
    case_id = recoveries[0].name.removesuffix(".recovery.json")
    assert not (output / f"{case_id}.json").exists()
    assert list(output.glob(f"{case_id}*.tmp")) == []


@pytest.mark.asyncio
async def test_ui_batch_same_tick_cancellation_has_installed_drain_ownership(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned batch task is not cancellable before every claim has an owner."""

    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    batch = await registry.start_batch(
        [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
        settings,
        concurrency=1,
    )
    assert batch.task is not None

    batch.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch.task

    store = WorkflowStore(settings)
    for entry in batch.entries:
        result = store.load_result(entry.case_id)
        assert result is not None
        assert result.stop_reason == "CANCELLED"
        handle = registry.handle(entry.case_id)
        assert handle is not None and handle.state == "done"


@pytest.mark.asyncio
async def test_core_batch_first_await_cancellation_terminalizes_unstarted_tasks(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation queued before child first turns still has a direct owner."""

    real_prepare = orchestration.prepare_claimed_invoice
    prepared: list[tuple[str, datetime, ExecutionClaim]] = []

    async def cancel_after_second_prepare(path: Path, selected_settings: Settings) -> object:
        outcome = real_prepare(path, selected_settings)
        assert isinstance(outcome, tuple)
        prepared.append(outcome)
        if len(prepared) == 2:
            task = asyncio.current_task()
            assert task is not None
            asyncio.get_running_loop().call_soon(task.cancel)
        return outcome

    monkeypatch.setattr(
        orchestration,
        "prepare_claimed_invoice_async",
        cancel_after_second_prepare,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await orchestration.process_batch(
            [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
            settings,
            concurrency=1,
        )

    assert len(prepared) == 2
    store = WorkflowStore(settings)
    for case_id, _started_at, _claim in prepared:
        result = store.load_result(case_id)
        assert result is not None
        assert result.stop_reason == "CANCELLED"


@pytest.mark.asyncio
async def test_ui_start_batch_cancellation_during_startup_handshake_accounts_all_claims(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation before child ownership is acknowledged uses the direct owner."""

    handshake_entered = asyncio.Event()
    captured_batches: list[ui_runs.BatchState] = []

    async def delayed_run_batch(
        self: RunRegistry,
        batch: ui_runs.BatchState,
        entries: list[ui_runs.BatchEntry],
        selected_settings: Settings,
        ownership_installed: asyncio.Event,
    ) -> None:
        del self, entries, selected_settings, ownership_installed
        captured_batches.append(batch)
        handshake_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(RunRegistry, "_run_batch", delayed_run_batch)
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    startup = asyncio.create_task(
        registry.start_batch(
            [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
            settings,
            concurrency=1,
        )
    )
    await asyncio.wait_for(handshake_entered.wait(), timeout=1)

    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(startup, timeout=1)

    assert len(captured_batches) == 1
    store = WorkflowStore(settings)
    for entry in captured_batches[0].entries:
        result = store.load_result(entry.case_id)
        assert result is not None
        assert result.stop_reason == "CANCELLED"
        handle = registry.handle(entry.case_id)
        assert handle is not None and handle.state == "done"


@pytest.mark.asyncio
async def test_ui_batch_recovery_failure_precedes_ordinary_child_failure(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every drained failure is inspected even when the first child fails ordinarily."""

    calls = 0

    async def fail_first_runner(*_args: object, **_kwargs: object) -> CaseResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ordinary child sentinel")
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(ui_runs, "run_prepared_case", fail_first_runner)
    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
    monkeypatch.setattr(
        orchestration,
        "_write_recovery_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("recovery publication sentinel")),
    )
    drained: list[object] = []
    original_select = orchestration._select_drained_failure

    def capture_selection(
        primary: BaseException,
        outcomes: list[object],
        durability: dict[str, BaseException | None],
    ) -> BaseException:
        drained.extend(outcomes)
        return original_select(primary, outcomes, durability)

    monkeypatch.setattr(ui_runs, "_select_drained_failure", capture_selection)
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    batch = await registry.start_batch(
        [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
        settings,
        concurrency=1,
    )
    assert batch.task is not None

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await batch.task

    assert any(
        isinstance(outcome, InvoiceAgentsError)
        and outcome.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
        for outcome in drained
    ), [(type(outcome).__name__, getattr(outcome, "stop_reason", None)) for outcome in drained]
    assert excinfo.value.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert any(
        (handle := registry.handle(entry.case_id)) is not None and handle.state != "done"
        for entry in batch.entries
    )


@pytest.mark.asyncio
async def test_core_batch_recovery_failure_precedes_later_ordinary_child_failure(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core selection is independent of the opposite drained-outcome ordering."""

    calls = 0

    async def block_then_fail(*_args: object, **_kwargs: object) -> CaseResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("later ordinary child sentinel")
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(orchestration, "run_prepared_case", block_then_fail)
    monkeypatch.setattr(orchestration, "run_terminal_process", _failed_terminal_worker)
    monkeypatch.setattr(
        orchestration,
        "_write_recovery_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("recovery publication sentinel")),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration.process_batch(
            [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
            settings,
            concurrency=2,
        )

    assert excinfo.value.stop_reason == "TERMINAL_RECOVERY_ARTIFACT_FAILED"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


@pytest.mark.asyncio
async def test_durability_wait_has_a_monotonic_deadline() -> None:
    """A stuck durability task must fail explicitly instead of draining forever."""

    stuck = asyncio.create_task(asyncio.Event().wait(), name="task9-stuck-durability")
    try:
        with pytest.raises(InvoiceAgentsError) as excinfo:
            await orchestration._await_task_despite_cancellation(
                stuck,
                deadline=orchestration.monotonic() - 0.001,
            )
        assert excinfo.value.stop_reason == "TERMINAL_DURABILITY_TIMEOUT"
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__context__ is None
    finally:
        stuck.cancel()
        await asyncio.gather(stuck, return_exceptions=True)


@pytest.mark.asyncio
async def test_blocking_terminal_write_times_out_without_blocking_event_loop(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck synchronous finish has one bounded timeout recovery owner."""

    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    worker_pid = tmp_path / "terminal-worker.pid"
    late_mutation = tmp_path / "terminal-worker-late.txt"

    publications: list[str] = []
    original_publish = orchestration._write_recovery_artifact

    def count_publish(result: CaseResult, error: ErrorRecord) -> Path:
        publications.append(result.case_id)
        return original_publish(result, error)

    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [
            sys.executable,
            "-c",
            (
                "import os,time; from pathlib import Path; "
                f"Path({str(worker_pid)!r}).write_text(str(os.getpid())); time.sleep(10); "
                f"Path({str(late_mutation)!r}).write_text('late')"
            ),
        ],
    )
    monkeypatch.setattr(orchestration, "_write_recovery_artifact", count_publish)
    monkeypatch.setattr(orchestration, "DURABILITY_DEADLINE_SECONDS", 0.03)
    monkeypatch.chdir(tmp_path)
    before = asyncio.get_running_loop().time()
    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration._durably_cancel_unstarted_claim(case_id, started_at, settings, claim)
    elapsed = asyncio.get_running_loop().time() - before
    assert excinfo.value.stop_reason == "TERMINAL_DURABILITY_TIMEOUT"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert elapsed < 2.0
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert not recovery.exists()
    assert publications == []
    pid = int(worker_pid.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await asyncio.sleep(0.03)
    assert not late_mutation.exists()


@pytest.mark.asyncio
async def test_ui_batch_timeout_starts_no_late_recovery_and_leaves_handle_unresolved(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exhausted shared deadline publishes nothing and remains unresolved."""

    runner_started = asyncio.Event()

    async def blocking_runner(*_args: object, **_kwargs: object) -> CaseResult:
        runner_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    publications: list[str] = []
    original_publish = orchestration._write_recovery_artifact

    def count_publish(result: CaseResult, error: ErrorRecord) -> Path:
        publications.append(result.case_id)
        return original_publish(result, error)

    worker_pid = tmp_path / "ui-timeout-terminal-worker.pid"
    late_mutation = tmp_path / "ui-timeout-terminal-worker-late.txt"

    monkeypatch.setattr(ui_runs, "run_prepared_case", blocking_runner)
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [
            sys.executable,
            "-c",
            (
                "import os,time; from pathlib import Path; "
                f"Path({str(worker_pid)!r}).write_text(str(os.getpid())); time.sleep(10); "
                f"Path({str(late_mutation)!r}).write_text('late')"
            ),
        ],
    )
    monkeypatch.setattr(orchestration, "_write_recovery_artifact", count_publish)
    monkeypatch.setattr(orchestration, "DURABILITY_DEADLINE_SECONDS", 0.03)
    monkeypatch.setattr(ui_runs, "DURABILITY_DEADLINE_SECONDS", 0.03)
    monkeypatch.chdir(tmp_path)
    registry = RunRegistry()
    batch = await registry.start_batch([invoice_dir / "invoice_1001.txt"], settings, concurrency=1)
    assert batch.task is not None
    await asyncio.wait_for(runner_started.wait(), timeout=1)
    batch.task.cancel()
    with pytest.raises(InvoiceAgentsError) as excinfo:
        await asyncio.wait_for(
            batch.task,
            timeout=(
                (3 * ui_runs.DURABILITY_DEADLINE_SECONDS)
                + ui_runs.TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                + 1.0
            ),
        )
    assert excinfo.value.stop_reason == "TERMINAL_DURABILITY_TIMEOUT"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    recovery = tmp_path / "artifacts" / "results" / f"{batch.entries[0].case_id}.recovery.json"
    assert not recovery.exists()
    assert publications == []
    store = WorkflowStore(settings)
    assert store.load_result(batch.entries[0].case_id) is None
    snapshot = store.load_case_execution_snapshot(batch.entries[0].case_id)
    assert snapshot is not None
    assert snapshot.execution_state == "RUNNING"
    assert snapshot.lease_expires_at is not None
    pid = int(worker_pid.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not late_mutation.exists()
    handle = registry.handle(batch.entries[0].case_id)
    assert handle is not None and handle.state == "unresolved"


@pytest.mark.asyncio
async def test_batch_preflight_process_control_is_not_a_business_result(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SystemExit before any claim stops the batch and publishes no result artifact."""

    publications: list[str] = []
    monkeypatch.setattr(
        orchestration,
        "preflight",
        lambda _settings: (_ for _ in ()).throw(SystemExit("preflight sentinel")),
    )
    monkeypatch.setattr(
        orchestration,
        "_write_result",
        lambda result: publications.append(result.case_id),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        await orchestration.process_batch(
            [invoice_dir / "invoice_1001.txt", invoice_dir / "invoice_1002.txt"],
            settings,
            concurrency=1,
        )

    assert publications == []
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("control_type", [asyncio.CancelledError, SystemExit, KeyboardInterrupt])
@pytest.mark.asyncio
async def test_preflight_control_exceptions_escape_single_and_batch_without_artifacts(
    control_type: type[BaseException],
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-claim process control is never converted into application data."""

    publications: list[str] = []

    def stop_preflight(_settings: Settings) -> None:
        raise control_type("preflight process-control sentinel")

    monkeypatch.setattr(orchestration, "preflight", stop_preflight)
    monkeypatch.setattr(
        orchestration,
        "_write_result",
        lambda result: publications.append(result.case_id),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(control_type):
        orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    with pytest.raises(control_type):
        await orchestration.process_batch(
            [invoice_dir / "invoice_1001.txt"], settings, concurrency=1
        )

    assert publications == []
    assert not (tmp_path / "artifacts").exists()
