"""RED reproducers for the fourth Task 9 reliability review."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from invoice_agents import orchestration
from invoice_agents.config import Settings
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult, CaseStatus, ErrorRecord
from invoice_agents.source_store import snapshot_source
from invoice_agents.ui.runs import RunHandle, RunRegistry


def _finish_with_successor_generation(
    invoice_path: Path,
    settings: Settings,
) -> tuple[str, datetime, ExecutionClaim, CaseResult]:
    prepared = orchestration.prepare_claimed_invoice(invoice_path, settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, inspected_claim = prepared
    store = WorkflowStore(settings)
    store.release_case_execution(inspected_claim)
    successor = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    assert successor.generation == inspected_claim.generation + 1
    successor_result = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(successor),
        status=CaseStatus.INCOMPLETE,
        stop_reason="SUCCESSOR_TERMINAL",
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    store.finish_case(successor_result, successor)
    return case_id, started_at, inspected_claim, successor_result


@pytest.mark.parametrize("surface", ["inspection", "post_helper_reread", "unstarted_cancel"])
@pytest.mark.asyncio
async def test_round4_successor_finished_never_satisfies_an_older_claim(
    invoice_dir: Path,
    settings: Settings,
    surface: str,
) -> None:
    """Generation N+1 FINISHED is a contradiction for an inspected generation N claim."""

    case_id, started_at, inspected_claim, successor_result = _finish_with_successor_generation(
        invoice_dir / "invoice_1001.txt", settings
    )

    if surface == "inspection":
        outcomes = await orchestration._inspect_claim_durability(
            [(case_id, started_at, inspected_claim)],
            settings,
        )
        failure = outcomes[case_id]
        assert isinstance(failure, InvoiceAgentsError)
        assert failure.stop_reason == "TERMINAL_DURABILITY_UNRESOLVED"
        assert failure.__cause__ is None and failure.__context__ is None
        return

    if surface == "post_helper_reread":
        execution = orchestration._ClaimedExecution(
            store=WorkflowStore(settings),
            claim=inspected_claim,
            case_id=case_id,
            started_at=started_at,
        )
        with pytest.raises(InvoiceAgentsError) as excinfo:
            await orchestration._terminal_process_write(
                execution,
                successor_result,
                mode="finish",
            )
        assert excinfo.value.stop_reason == "TERMINAL_DURABILITY_UNRESOLVED"
        assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None
        return

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration._terminalize_unstarted_claim(
            case_id,
            started_at,
            settings,
            inspected_claim,
        )
    assert excinfo.value.stop_reason == "TERMINAL_DURABILITY_UNRESOLVED"
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None


def _write_recovery_fixture(
    invoice_path: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, WorkflowStore, ExecutionClaim, Path]:
    prepared = orchestration.prepare_claimed_invoice(invoice_path, settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    result = orchestration._cancelled_result(
        case_id,
        store.load_authoritative_case_source_id(claim),
        started_at,
    )
    persistence_error = ErrorRecord(
        category=ErrorCategory.DATABASE,
        message="terminal database write failed",
        case_id=case_id,
        stop_reason="TERMINAL_PERSISTENCE_FAILED",
    )
    monkeypatch.chdir(tmp_path)
    orchestration._recovery_artifact_or_raise(
        result,
        persistence_error,
        store=store,
        claim=claim,
    )
    target = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert target.is_file()
    return case_id, store, claim, target


def test_round4_recovery_writer_emits_one_canonical_failure_frame(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery is a canonical failed-persistence frame, never a second result channel."""

    _case_id, _store, _claim, target = _write_recovery_fixture(
        invoice_dir / "invoice_1001.txt",
        settings,
        tmp_path,
        monkeypatch,
    )
    raw = target.read_bytes()
    payload = json.loads(raw)
    assert raw == json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result = payload["case_result"]
    persistence_error = payload["terminal_persistence_error"]
    assert result["status"] == "INCOMPLETE"
    assert result["stop_reason"] == persistence_error["stop_reason"]
    assert result["errors"][-1] == persistence_error
    assert persistence_error == {
        "case_id": result["case_id"],
        "category": "DATABASE",
        "details": {},
        "message": "terminal database write failed",
        "provider_request_id": None,
        "stop_reason": "TERMINAL_PERSISTENCE_FAILED",
    }


_RECOVERY_SEMANTIC_FORGERIES = (
    "trailing_whitespace",
    "success_status",
    "payment_success_stop",
    "error_category",
    "error_message",
    "error_provider_id",
    "error_details",
    "result_error_missing",
    "reverse_chronology",
)


@pytest.mark.parametrize("forgery", _RECOVERY_SEMANTIC_FORGERIES)
def test_round4_recovery_parser_rejects_semantically_forgeable_frames(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    """Canonical bytes and the exact parent-owned failure relation are mandatory."""

    case_id, store, claim, target = _write_recovery_fixture(
        invoice_dir / "invoice_1001.txt",
        settings,
        tmp_path,
        monkeypatch,
    )
    raw = target.read_bytes()
    payload = json.loads(raw)
    if forgery == "trailing_whitespace":
        target.write_bytes(raw + b"\n")
    else:
        if forgery == "success_status":
            payload["case_result"]["status"] = "SUCCEEDED"
            payload["case_result"]["stop_reason"] = "TEAM_COMPLETED"
        elif forgery == "payment_success_stop":
            payload["case_result"]["status"] = "SUCCEEDED"
            payload["case_result"]["stop_reason"] = "APPROVED_PAYMENT_RECORDED"
        elif forgery == "error_category":
            payload["terminal_persistence_error"]["category"] = "ORCHESTRATION"
        elif forgery == "error_message":
            payload["terminal_persistence_error"]["message"] = "forged persistence claim"
        elif forgery == "error_provider_id":
            payload["terminal_persistence_error"]["provider_request_id"] = "req_forged"
        elif forgery == "error_details":
            payload["terminal_persistence_error"]["details"] = {"forged": "claim"}
        elif forgery == "result_error_missing":
            payload["case_result"]["errors"] = []
        elif forgery == "reverse_chronology":
            payload["case_result"]["finished_at"] = "2000-01-01T00:00:00+00:00"
        target.write_bytes(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )

    assert orchestration._recovery_artifact_is_valid(case_id, store, claim) is False


def test_round4_preparation_worker_kills_its_session_before_returning(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled claimed-preparation helper cannot commit or write after return."""

    preparation_process = importlib.import_module("invoice_agents.preparation_process")
    late_sentinel = tmp_path / "late-preparation-write"
    child_code = (
        "import subprocess,sys,time; "
        "sys.stdin.buffer.read(); "
        f"subprocess.Popen([sys.executable,'-c',{('import time; from pathlib import Path; time.sleep(0.25); Path(' + repr(str(late_sentinel)) + ').write_text("late", encoding="utf-8")')!r}]); "
        "time.sleep(30)"
    )
    command = [sys.executable, "-c", child_code]
    monkeypatch.setattr(preparation_process, "_preparation_worker_command", lambda: command)
    secret = "sk-proj-preparation-must-not-cross"
    selected = settings.model_copy(update={"xai_api_key": SecretStr(secret)})
    cancel_requested = threading.Event()
    timer = threading.Timer(0.08, cancel_requested.set)
    timer.start()
    try:
        outcome = preparation_process.run_preparation_process(
            path=invoice_dir / "invoice_1001.txt",
            settings=selected,
            case_id="case_round4_preparation_process",
            started_at=datetime.now(UTC),
            preparation_token="exec_11111111111111111111111111111111",
            run_token="exec_22222222222222222222222222222222",
            timeout_seconds=1.0,
            cancel_requested=cancel_requested,
        )
    finally:
        timer.cancel()
    assert outcome.error_code == "PREPARATION_WORKER_CANCELLED"
    assert secret not in repr(outcome)
    time.sleep(0.4)
    assert not late_sentinel.exists()


def test_round4_real_preparation_worker_hands_off_parent_issued_token(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """The durable run authority is chosen by the parent before spawning."""

    preparation_process = importlib.import_module("invoice_agents.preparation_process")
    case_id = "case_round4_parent_owned_handoff"
    preparation_token = "exec_33333333333333333333333333333333"
    run_token = "exec_44444444444444444444444444444444"
    outcome = preparation_process.run_preparation_process(
        path=invoice_dir / "invoice_1001.txt",
        settings=settings,
        case_id=case_id,
        started_at=datetime.now(UTC),
        preparation_token=preparation_token,
        run_token=run_token,
        timeout_seconds=10.0,
    )

    assert outcome.acknowledged is True
    assert outcome.error_code is None
    store = WorkflowStore(settings)
    snapshot = store.load_case_execution_snapshot(case_id)
    assert snapshot is not None
    assert snapshot.execution_state == "RUNNING"
    assert snapshot.execution_token == run_token
    assert snapshot.execution_generation == 2
    assert snapshot.lease_expires_at is not None
    store.release_case_execution(ExecutionClaim(case_id, run_token, 2, snapshot.lease_expires_at))


@pytest.mark.asyncio
async def test_round4_async_preparation_recovers_handoff_after_response_loss(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact run token in storage survives a lost worker acknowledgement."""

    preparation_process = importlib.import_module("invoice_agents.preparation_process")
    real_worker = preparation_process.run_preparation_process

    def lose_response(**kwargs: object) -> object:
        actual = real_worker(**kwargs)
        assert actual.acknowledged is True
        return preparation_process.PreparationProcessOutcome(
            False,
            "PREPARATION_WORKER_PROTOCOL_INVALID",
        )

    monkeypatch.setattr(preparation_process, "run_preparation_process", lose_response)
    prepared = await orchestration.prepare_claimed_invoice_async(
        invoice_dir / "invoice_1001.txt",
        settings,
    )

    assert isinstance(prepared, tuple)
    case_id, _started_at, claim = prepared
    snapshot = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert snapshot is not None
    assert snapshot.execution_state == "RUNNING"
    assert snapshot.execution_token == claim.token
    assert snapshot.execution_generation == claim.generation == 2
    WorkflowStore(settings).release_case_execution(claim)


def test_round4_lifecycle_worker_uses_private_credential_fd_and_kills_descendants(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation kills provider work and descendants without exposing credentials."""

    lifecycle_process = importlib.import_module("invoice_agents.lifecycle_process")
    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    _case_id, started_at, claim = prepared
    canary = "sk-proj-lifecycle-private-fd-canary"
    receipt = tmp_path / "credential-receipt"
    late_sentinel = tmp_path / "late-provider-call"
    descendant_code = (
        "import time; from pathlib import Path; time.sleep(0.25); "
        f"Path({str(late_sentinel)!r}).write_text('late', encoding='utf-8')"
    )
    child_code = (
        "import hashlib,json,os,subprocess,sys,time; "
        "from invoice_agents.lifecycle_worker import _read_private_credential; "
        "p=json.loads(sys.stdin.buffer.read()); "
        "s=_read_private_credential(p['credential_fd']); "
        "d=hashlib.sha256(s).hexdigest(); "
        "s[:]=bytes(len(s)); "
        f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}]); "
        f"open({str(receipt)!r},'w',encoding='utf-8').write(d); "
        "time.sleep(30)"
    )
    command = [sys.executable, "-c", child_code]
    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", lambda: command)
    selected = settings.model_copy(update={"xai_api_key": SecretStr(canary)})
    encoded, _secret = lifecycle_process._encode_request(
        mode="process",
        settings=selected,
        claim=claim,
        started_at=started_at,
        credential_fd=99,
    )
    assert canary.encode() not in encoded
    _secret[:] = bytes(len(_secret))
    assert all(canary not in argument for argument in command)
    cancel_requested = threading.Event()

    def cancel_after_worker_started() -> None:
        deadline = time.monotonic() + 0.75
        while time.monotonic() < deadline and not receipt.exists():
            time.sleep(0.01)
        cancel_requested.set()

    cancellation = threading.Thread(target=cancel_after_worker_started)
    cancellation.start()
    try:
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=selected,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
            cancel_requested=cancel_requested,
        )
    finally:
        cancellation.join(timeout=1.0)
    assert outcome.error_code == "LIFECYCLE_WORKER_CANCELLED"
    assert canary not in repr(outcome)
    assert receipt.read_text(encoding="utf-8") == hashlib.sha256(canary.encode()).hexdigest()
    time.sleep(0.4)
    assert not late_sentinel.exists()


@pytest.mark.asyncio
async def test_round4_public_lifecycle_trusts_exact_db_finish_after_worker_reap(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker acknowledgement is only a hint; exact fenced DB truth is returned."""

    lifecycle_process = importlib.import_module("invoice_agents.lifecycle_process")
    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    expected = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND4_EXACT_WORKER_FINISH",
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )

    def fake_worker(**_kwargs: object) -> object:
        store.finish_case(expected, claim)
        return lifecycle_process.LifecycleProcessOutcome(True, None)

    monkeypatch.setattr(lifecycle_process, "run_lifecycle_process", fake_worker)

    def forbidden_parent_provider(_settings: Settings) -> object:
        raise AssertionError("provider lifecycle ran in the parent process")

    monkeypatch.setattr(orchestration, "create_model_client", forbidden_parent_provider)
    monkeypatch.chdir(tmp_path)
    actual = await orchestration.run_prepared_case(
        case_id,
        started_at,
        settings,
        claim=claim,
    )

    assert actual == expected
    artifact = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    assert CaseResult.model_validate_json(artifact.read_bytes(), strict=True) == expected


@pytest.mark.asyncio
async def test_round4_lifecycle_reuses_exact_recovery_after_worker_response_loss(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lifecycle recovery frame remains the sole result when its acknowledgement is lost."""

    case_id, store, claim, _target = _write_recovery_fixture(
        invoice_dir / "invoice_1001.txt",
        settings,
        tmp_path,
        monkeypatch,
    )
    expected = orchestration._load_valid_recovery_result(case_id, store, claim)
    assert expected is not None

    async def forbidden_terminalization(**_kwargs: object) -> CaseResult:
        raise AssertionError("parent tried to publish beside exact lifecycle recovery")

    monkeypatch.setattr(
        orchestration,
        "_terminalize_lifecycle_boundary_failure",
        forbidden_terminalization,
    )
    actual = await orchestration._reconcile_lifecycle_boundary(
        case_id=case_id,
        started_at=expected.started_at,
        settings=settings,
        issued_claim=claim,
        outcome=orchestration._LifecycleBoundaryOutcome(True, None),
        cancellation=None,
    )

    snapshot = store.load_case_execution_snapshot(case_id)
    assert actual == expected
    assert snapshot is not None and snapshot.execution_state == "RUNNING"
    assert snapshot.execution_token == claim.token
    assert not (tmp_path / "artifacts" / "results" / f"{case_id}.json").exists()


@pytest.mark.asyncio
async def test_round4_cancel_after_exact_lifecycle_finish_updates_normal_artifact(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent cancellation update keeps the normal artifact equal to exact DB truth."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    worker_result = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND4_WORKER_FINISHED_BEFORE_CANCEL",
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    store.finish_case(worker_result, claim)
    monkeypatch.chdir(tmp_path)
    cancellation = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await orchestration._reconcile_lifecycle_boundary(
            case_id=case_id,
            started_at=started_at,
            settings=settings,
            issued_claim=claim,
            outcome=orchestration._LifecycleBoundaryOutcome(True, None),
            cancellation=cancellation,
        )

    snapshot = store.load_case_execution_snapshot(case_id)
    artifact = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    assert excinfo.value is cancellation
    assert snapshot is not None and snapshot.result is not None
    assert snapshot.result.stop_reason == "CANCELLED"
    assert CaseResult.model_validate_json(artifact.read_bytes(), strict=True) == snapshot.result


@pytest.mark.asyncio
async def test_round4_parent_lifecycle_terminalization_publishes_normal_result_artifact(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reaped worker crash is terminalized and published entirely by the parent."""

    lifecycle_process = importlib.import_module("invoice_agents.lifecycle_process")
    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared

    def failed_worker(**_kwargs: object) -> object:
        return lifecycle_process.LifecycleProcessOutcome(
            False,
            "LIFECYCLE_WORKER_CRASHED",
        )

    monkeypatch.setattr(lifecycle_process, "run_lifecycle_process", failed_worker)
    monkeypatch.setattr(
        orchestration,
        "create_model_client",
        lambda _settings: (_ for _ in ()).throw(
            AssertionError("provider lifecycle ran in the parent process")
        ),
    )
    monkeypatch.chdir(tmp_path)

    result = await orchestration.run_prepared_case(
        case_id,
        started_at,
        settings,
        claim=claim,
    )

    stored = WorkflowStore(settings).load_result(case_id)
    artifact = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    assert result.stop_reason == "LIFECYCLE_WORKER_CRASHED"
    assert stored == result
    assert CaseResult.model_validate_json(artifact.read_bytes(), strict=True) == result


@pytest.mark.asyncio
async def test_round4_resume_load_failure_is_terminalized_by_exact_claim_owner(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed resume cannot strand RUNNING authority when prior-result loading fails."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    original_load_result = WorkflowStore.load_result

    def fail_parent_load(self: WorkflowStore, loaded_case_id: str) -> CaseResult | None:
        if loaded_case_id == case_id:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "persisted result failed strict validation",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            )
        return original_load_result(self, loaded_case_id)

    async def forbidden_worker(**_kwargs: object) -> object:
        raise AssertionError("resume lifecycle started after prior-result load failure")

    monkeypatch.setattr(WorkflowStore, "load_result", fail_parent_load)
    monkeypatch.setattr(orchestration, "_invoke_lifecycle_boundary", forbidden_worker)
    monkeypatch.chdir(tmp_path)

    result = await orchestration.resume_case(case_id, settings, claim=claim)

    snapshot = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    artifact = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    assert snapshot is not None
    assert snapshot.execution_state == "FINISHED"
    assert snapshot.execution_token == claim.token
    assert snapshot.execution_generation == claim.generation
    assert snapshot.result == result
    assert result.started_at == started_at
    assert result.stop_reason == "PERSISTED_RESULT_INVALID"
    assert CaseResult.model_validate_json(artifact.read_bytes(), strict=True) == result


@pytest.mark.asyncio
async def test_round4_preparation_reuses_exact_recovery_after_worker_response_loss(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid child recovery frame prevents a conflicting parent terminal result."""

    case_id = "case_round4_preparation_recovery"
    started_at = datetime.now(UTC)
    preparation_token = "exec_55555555555555555555555555555555"
    run_token = "exec_66666666666666666666666666666666"
    source = snapshot_source(
        invoice_dir / "invoice_1001.txt",
        settings.source_archive_dir,
        settings.source_max_bytes,
    )
    store = WorkflowStore(settings)
    store.register_source(source)
    store.create_case(case_id, source, started_at)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        orchestration.EXECUTION_LEASE_SECONDS,
        requested_token=preparation_token,
    )
    result = orchestration._failed_result(
        case_id,
        source.source_id,
        started_at,
        InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "isolated preparation failed",
            case_id=case_id,
            stop_reason="PREPARATION_FAILED",
        ),
    )
    persistence_error = ErrorRecord(
        category=ErrorCategory.DATABASE,
        message="terminal database write failed",
        case_id=case_id,
        stop_reason="TERMINAL_PERSISTENCE_FAILED",
    )
    monkeypatch.chdir(tmp_path)
    orchestration._recovery_artifact_or_raise(
        result,
        persistence_error,
        store=store,
        claim=claim,
    )
    expected = orchestration._load_valid_recovery_result(case_id, store, claim)
    assert expected is not None

    async def forbidden_terminalization(**_kwargs: object) -> CaseResult:
        raise AssertionError("parent tried to publish a second terminal result")

    monkeypatch.setattr(
        orchestration,
        "_terminalize_lifecycle_boundary_failure",
        forbidden_terminalization,
    )
    actual = await orchestration._reconcile_preparation_boundary(
        case_id=case_id,
        started_at=started_at,
        settings=settings,
        preparation_token=preparation_token,
        run_token=run_token,
        outcome=orchestration._PreparationBoundaryOutcome(False, "PREPARATION_FAILED"),
        cancellation=None,
    )

    snapshot = store.load_case_execution_snapshot(case_id)
    assert actual == expected
    assert snapshot is not None and snapshot.execution_state == "RUNNING"
    assert snapshot.execution_token == preparation_token
    assert not (tmp_path / "artifacts" / "results" / f"{case_id}.json").exists()


@pytest.mark.asyncio
async def test_round4_cancel_after_exact_preparation_finish_publishes_artifact(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed preparation failure is published before parent cancellation returns."""

    case_id = "case_round4_preparation_finish_cancel"
    started_at = datetime.now(UTC)
    preparation_token = "exec_99999999999999999999999999999999"
    run_token = "exec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    source = snapshot_source(
        invoice_dir / "invoice_1001.txt",
        settings.source_archive_dir,
        settings.source_max_bytes,
    )
    store = WorkflowStore(settings)
    store.register_source(source)
    store.create_case(case_id, source, started_at)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        orchestration.EXECUTION_LEASE_SECONDS,
        requested_token=preparation_token,
    )
    result = orchestration._failed_result(
        case_id,
        source.source_id,
        started_at,
        InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "isolated preparation failed",
            case_id=case_id,
            stop_reason="PREPARATION_FAILED",
        ),
    )
    store.finish_case(result, claim)
    monkeypatch.chdir(tmp_path)
    cancellation = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await orchestration._reconcile_preparation_boundary(
            case_id=case_id,
            started_at=started_at,
            settings=settings,
            preparation_token=preparation_token,
            run_token=run_token,
            outcome=orchestration._PreparationBoundaryOutcome(False, "PREPARATION_FAILED"),
            cancellation=cancellation,
        )

    artifact = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    assert excinfo.value is cancellation
    assert CaseResult.model_validate_json(artifact.read_bytes(), strict=True) == result


@pytest.mark.asyncio
async def test_round4_preparation_never_claims_an_id_collision_with_another_start(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """An IDLE row is owned only when its persisted start equals the parent request."""

    case_id = "case_round4_idle_collision"
    persisted_start = datetime.now(UTC)
    requested_start = persisted_start + timedelta(seconds=1)
    preparation_token = "exec_77777777777777777777777777777777"
    run_token = "exec_88888888888888888888888888888888"
    source = snapshot_source(
        invoice_dir / "invoice_1001.txt",
        settings.source_archive_dir,
        settings.source_max_bytes,
    )
    store = WorkflowStore(settings)
    store.register_source(source)
    store.create_case(case_id, source, persisted_start)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration._reconcile_preparation_boundary(
            case_id=case_id,
            started_at=requested_start,
            settings=settings,
            preparation_token=preparation_token,
            run_token=run_token,
            outcome=orchestration._PreparationBoundaryOutcome(
                False,
                "PREPARATION_WORKER_CRASHED",
            ),
            cancellation=None,
        )

    snapshot = store.load_case_execution_snapshot(case_id)
    assert excinfo.value.stop_reason == "TERMINAL_DURABILITY_UNRESOLVED"
    assert snapshot is not None
    assert snapshot.execution_state == "IDLE"
    assert snapshot.execution_generation == 0
    assert snapshot.execution_token is None


@pytest.mark.asyncio
async def test_round4_registry_detaches_completed_raw_task_and_traceback() -> None:
    """Completed public registry state retains only one stable parent-owned error code."""

    canary = "xai-registry-retained-exception-canary /private/provider/path"

    async def fail() -> CaseResult:
        raise RuntimeError(canary)

    task = asyncio.create_task(fail())
    handle = RunHandle(
        case_id="case_round4_registry",
        kind="process",
        state="running",
        started_at=datetime.now(UTC),
        task=task,
    )
    registry = RunRegistry()
    registry._runs[handle.case_id] = handle
    with pytest.raises(RuntimeError, match="retained-exception-canary"):
        await task
    registry._finish(handle, None)

    assert handle.error == "UNEXPECTED_RUNTIME_ERROR"
    assert handle.task is None
    retained = repr(handle) + repr(registry._runs) + repr(registry._batches)
    assert canary not in retained
    assert task.exception() is not None
    assert task.exception().__traceback__ is None
