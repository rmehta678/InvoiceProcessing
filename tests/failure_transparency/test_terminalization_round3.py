"""Minimal RED reproducers for the third Task 9 reliability review."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import SecretStr

from invoice_agents import lifecycle_process, orchestration, terminal_process
from invoice_agents.agents.team import AgentCaseContext
from invoice_agents.config import Settings
from invoice_agents.db import store as store_module
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult, CaseStatus, ErrorRecord
from invoice_agents.payment import service as payment_service
from invoice_agents.ui import runs as ui_runs
from invoice_agents.ui.runs import RunHandle, RunRegistry


@pytest.fixture(autouse=True)
def _forbid_external_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", "import sys; sys.exit(86)"],
    )

    def forbidden(_settings: Settings) -> object:
        raise AssertionError("non-live round3 test reached an unstubbed provider boundary")

    monkeypatch.setattr(orchestration, "create_model_client", forbidden)


@pytest.mark.asyncio
async def test_round3_single_same_tick_cancel_has_durability_owner(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the exposed task before its first turn must account for its claim."""

    provider_calls = 0

    async def forbidden_runner(*_args: object, **_kwargs: object) -> CaseResult:
        nonlocal provider_calls
        provider_calls += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(ui_runs, "run_prepared_case", forbidden_runner)
    registry = RunRegistry()
    outcome = await registry.start_process(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(outcome, str)
    handle = registry.handle(outcome)
    assert handle is not None and handle.task is not None

    handle.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.task
    await asyncio.sleep(0)

    snapshot = WorkflowStore(settings).load_case_execution_snapshot(outcome)
    assert provider_calls == 0
    assert handle.state in {"done", "unresolved"}
    assert snapshot is not None
    if handle.state == "done":
        assert snapshot.execution_state == "FINISHED"
        assert snapshot.result is not None
        assert snapshot.result.stop_reason == "CANCELLED"
    else:
        assert not snapshot.has_valid_lease


@pytest.mark.asyncio
async def test_round3_terminal_timeout_cannot_later_commit_after_recovery(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out terminal worker must not outlive recovery publication."""

    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(10)"],
    )
    monkeypatch.setattr(orchestration, "DURABILITY_DEADLINE_SECONDS", 0.02)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration._durably_cancel_unstarted_claim(case_id, started_at, settings, claim)
    assert excinfo.value.stop_reason == "TERMINAL_DURABILITY_TIMEOUT"
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert recovery.is_file()

    snapshot = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert snapshot is not None and snapshot.execution_state != "FINISHED"
    await asyncio.sleep(0.05)
    later = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert later is not None and later.execution_state != "FINISHED"


@pytest.mark.asyncio
async def test_round3_active_terminal_db_write_never_blocks_event_loop(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active terminal persistence must run outside the event-loop thread."""

    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    original_finish = WorkflowStore.finish_case

    def slow_finish(
        selected_store: WorkflowStore,
        result: CaseResult,
        exact_claim: ExecutionClaim,
    ) -> None:
        time.sleep(0.15)
        original_finish(selected_store, result, exact_claim)

    async def lifecycle(_execution: object) -> CaseResult:
        return CaseResult(
            case_id=case_id,
            source_id=store.load_case_source_id(case_id),
            status=CaseStatus.INCOMPLETE,
            stop_reason="TEST_TERMINAL",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    monkeypatch.setattr(WorkflowStore, "finish_case", slow_finish)
    ticked_at: list[float] = []
    began = time.monotonic()

    async def watchdog() -> None:
        await asyncio.sleep(0.02)
        ticked_at.append(time.monotonic())

    await asyncio.gather(
        watchdog(),
        orchestration._execute_claimed_case(
            case_id,
            started_at,
            store,
            claim,
            lifecycle,
            finished_event_type="case.test_finished",
        ),
    )

    assert ticked_at[0] - began < 0.08


def test_round3_recovery_artifact_is_bound_to_exact_claim_generation(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale generation's artifact cannot satisfy the current live claim."""

    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    source_id = WorkflowStore(settings).load_case_source_id(case_id)
    result = orchestration._cancelled_result(case_id, source_id, started_at)
    persistence_error = ErrorRecord(
        category=ErrorCategory.DATABASE,
        message="terminal database write failed",
        case_id=case_id,
        stop_reason="TERMINAL_PERSISTENCE_FAILED",
    )
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "recovery_format": 1,
                "case_result": result.model_dump(mode="json"),
                "terminal_persistence_error": persistence_error.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    assert claim.generation > 1
    assert orchestration._claim_has_durable_terminal_evidence(case_id, settings, claim) is False


@pytest.mark.parametrize("invalid_concurrency", [0, -1, 9, True, 1.0, "1"])
@pytest.mark.asyncio
async def test_round3_invalid_concurrency_is_rejected_before_preflight(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    invalid_concurrency: object,
) -> None:
    """Zero is invalid input, not a request for the configured default."""

    preflight_calls = 0

    def forbidden_preflight(_settings: Settings) -> None:
        nonlocal preflight_calls
        preflight_calls += 1
        raise AssertionError("preflight must not run for invalid concurrency")

    monkeypatch.setattr(orchestration, "preflight", forbidden_preflight)
    with pytest.raises(ValueError, match="concurrency"):
        await orchestration.process_batch(
            [invoice_dir / "invoice_1001.txt"],
            settings,
            concurrency=invalid_concurrency,  # type: ignore[arg-type]
        )
    assert preflight_calls == 0


@pytest.mark.asyncio
async def test_round3_source_identity_failure_never_publishes_normal_result(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result without transaction-validated source identity is not DB success."""

    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared

    def fail_source(_store: WorkflowStore, _claim: ExecutionClaim) -> str:
        raise sqlite3.OperationalError("transient source lookup failure")

    async def forbidden_lifecycle(_execution: object) -> CaseResult:
        raise AssertionError("lifecycle must not run after source lookup failure")

    monkeypatch.setattr(WorkflowStore, "load_authoritative_case_source_id", fail_source)
    monkeypatch.chdir(tmp_path)
    await orchestration._execute_claimed_case(
        case_id,
        started_at,
        WorkflowStore(settings),
        claim,
        forbidden_lifecycle,
        finished_event_type="case.test_finished",
    )

    normal = tmp_path / "artifacts" / "results" / f"{case_id}.json"
    recovery = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert not normal.exists()
    assert recovery.is_file()


def test_round3_float_generation_is_rejected_before_store_access(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """Python float/int equality cannot authorize any claim boundary."""

    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    _case_id, _started_at, issued = prepared
    forged = replace(issued, generation=float(issued.generation))

    with pytest.raises(InvoiceAgentsError) as excinfo:
        WorkflowStore(settings).require_current_execution_claim(forged)
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"


@pytest.mark.asyncio
async def test_round3_registry_never_stores_raw_exception_text() -> None:
    """Parent-owned registry state exposes only a stable sanitized error code."""

    canary = "xai-secret-canary /private/provider/path"

    async def fail() -> CaseResult:
        raise RuntimeError(canary)

    task = asyncio.create_task(fail())
    handle = RunHandle(
        case_id="case_registry_canary",
        kind="process",
        state="running",
        started_at=datetime.now(UTC),
        task=task,
        claim=ExecutionClaim(
            "case_registry_canary",
            "exec_registry_canary",
            1,
            datetime.now(UTC) + timedelta(minutes=1),
        ),
    )
    with pytest.raises(RuntimeError, match="xai-secret-canary"):
        await task

    RunRegistry()._finish(handle, None)

    assert handle.error == "UNEXPECTED_RUNTIME_ERROR"
    assert canary not in (handle.error or "")


_CLAIM_MUTATIONS = (
    "generation_float",
    "generation_bool",
    "generation_zero",
    "generation_negative",
    "generation_string",
    "case_non_string",
    "case_empty",
    "case_padded",
    "token_non_string",
    "token_empty",
    "token_padded",
    "expiry_naive",
    "expiry_noncanonical_utc",
)


def _mutate_claim(claim: ExecutionClaim, mutation: str) -> ExecutionClaim:
    values: dict[str, object]
    if mutation == "generation_float":
        values = {"generation": float(claim.generation)}
    elif mutation == "generation_bool":
        values = {"generation": True}
    elif mutation == "generation_zero":
        values = {"generation": 0}
    elif mutation == "generation_negative":
        values = {"generation": -1}
    elif mutation == "generation_string":
        values = {"generation": str(claim.generation)}
    elif mutation == "case_non_string":
        values = {"case_id": 7}
    elif mutation == "case_empty":
        values = {"case_id": ""}
    elif mutation == "case_padded":
        values = {"case_id": f" {claim.case_id} "}
    elif mutation == "token_non_string":
        values = {"token": 7}
    elif mutation == "token_empty":
        values = {"token": ""}
    elif mutation == "token_padded":
        values = {"token": f" {claim.token} "}
    elif mutation == "expiry_naive":
        values = {"expires_at": claim.expires_at.replace(tzinfo=None)}
    elif mutation == "expiry_noncanonical_utc":
        values = {"expires_at": claim.expires_at.astimezone(ZoneInfo("UTC"))}
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(mutation)
    return replace(claim, **values)  # type: ignore[arg-type]


@pytest.mark.parametrize("mutation", _CLAIM_MUTATIONS)
def test_round3_invalid_claim_shape_precedes_any_database_access(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    issued = prepared[2]
    forged = _mutate_claim(issued, mutation)
    database_calls = 0

    def forbidden_connect(*_args: object, **_kwargs: object) -> object:
        nonlocal database_calls
        database_calls += 1
        raise AssertionError("invalid claim reached database access")

    monkeypatch.setattr(store_module, "connect_database", forbidden_connect)
    with pytest.raises(InvoiceAgentsError) as excinfo:
        WorkflowStore(settings).require_current_execution_claim(forged)
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"
    assert database_calls == 0


@pytest.mark.parametrize("mutation", _CLAIM_MUTATIONS[:5])
@pytest.mark.asyncio
async def test_round3_invalid_claim_never_reaches_run_lifecycle_or_provider(
    invoice_dir: Path,
    settings: Settings,
    mutation: str,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, issued = prepared
    forged = _mutate_claim(issued, mutation)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration.run_prepared_case(case_id, started_at, settings, claim=forged)
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"
    snapshot = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert snapshot is not None and snapshot.execution_state == "RUNNING"
    assert snapshot.execution_generation == issued.generation


@pytest.mark.parametrize("mutation", _CLAIM_MUTATIONS[:5])
def test_round3_invalid_claim_is_rejected_at_tool_context_construction(
    settings: Settings,
    mutation: str,
) -> None:
    issued = ExecutionClaim(
        "case_tool_claim",
        "exec_tool_claim",
        1,
        datetime.now(UTC) + timedelta(minutes=1),
    )
    forged = _mutate_claim(issued, mutation)
    with pytest.raises(InvoiceAgentsError) as excinfo:
        AgentCaseContext(
            case_id=issued.case_id,
            settings=settings,
            store=WorkflowStore(settings),
            audit=object(),  # type: ignore[arg-type]
            claim=forged,
        )
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"


@pytest.mark.parametrize("mutation", _CLAIM_MUTATIONS[:5])
def test_round3_invalid_claim_never_reaches_payment_database(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, _started_at, issued = prepared
    store = WorkflowStore(settings)
    invoice = store.load_extraction(case_id)
    forged = _mutate_claim(issued, mutation)
    database_calls = 0

    def forbidden_connect(*_args: object, **_kwargs: object) -> object:
        nonlocal database_calls
        database_calls += 1
        raise AssertionError("invalid payment claim reached database access")

    monkeypatch.setattr(payment_service, "connect_database", forbidden_connect)
    with pytest.raises(InvoiceAgentsError) as excinfo:
        payment_service.mock_payment(
            case_id,
            invoice,
            store,
            settings.workflow_db,
            forged,
        )
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"
    assert database_calls == 0


@pytest.mark.parametrize("mutation", _CLAIM_MUTATIONS[:5])
@pytest.mark.asyncio
async def test_round3_invalid_renewed_claim_never_replaces_live_authority(
    mutation: str,
) -> None:
    issued = ExecutionClaim(
        "case_heartbeat_claim",
        "exec_heartbeat_claim",
        1,
        datetime.now(UTC) + timedelta(minutes=1),
    )
    replacement_calls = 0

    async def operation() -> None:
        await asyncio.Event().wait()

    def renew(_claim: ExecutionClaim, _lease_seconds: int) -> ExecutionClaim:
        return _mutate_claim(issued, mutation)

    def replace_claim(_claim: ExecutionClaim) -> None:
        nonlocal replacement_calls
        replacement_calls += 1

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await orchestration._run_with_lease_heartbeat(
            operation(),
            renew=renew,
            claim=issued,
            replace_claim=replace_claim,
            lease_seconds=60,
            renewal_interval_seconds=0.001,
        )
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"
    assert replacement_calls == 0


_RECOVERY_MUTATIONS = (
    "legacy_format",
    "missing_root_field",
    "extra_root_field",
    "outer_case_mismatch",
    "token_mismatch",
    "generation_float",
    "generation_bool",
    "generation_string",
    "generation_zero",
    "lease_mismatch",
    "lease_noncanonical",
    "result_case_mismatch",
    "result_source_mismatch",
    "error_case_mismatch",
    "error_stop_unrecognized",
    "extra_nested_field",
    "duplicate_root_key",
    "nonfinite_number",
    "invalid_utf8",
    "oversize",
)


@pytest.mark.parametrize("mutation", _RECOVERY_MUTATIONS)
def test_round3_recovery_envelope_rejects_every_untrusted_mutation(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    source_id = store.load_authoritative_case_source_id(claim)
    result = orchestration._cancelled_result(case_id, source_id, started_at)
    error = ErrorRecord(
        category=ErrorCategory.DATABASE,
        message="terminal database write failed",
        case_id=case_id,
        stop_reason="TERMINAL_PERSISTENCE_FAILED",
    )
    monkeypatch.chdir(tmp_path)
    orchestration._recovery_artifact_or_raise(result, error, store=store, claim=claim)
    target = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    assert orchestration._recovery_artifact_is_valid(case_id, store, claim)
    payload = json.loads(target.read_text(encoding="utf-8"))

    if mutation == "legacy_format":
        payload["recovery_format"] = 1
    elif mutation == "missing_root_field":
        del payload["execution_token"]
    elif mutation == "extra_root_field":
        payload["unexpected"] = "value"
    elif mutation == "outer_case_mismatch":
        payload["case_id"] = "case_other"
    elif mutation == "token_mismatch":
        payload["execution_token"] = "exec_other"
    elif mutation == "generation_float":
        payload["execution_generation"] = float(claim.generation)
    elif mutation == "generation_bool":
        payload["execution_generation"] = True
    elif mutation == "generation_string":
        payload["execution_generation"] = str(claim.generation)
    elif mutation == "generation_zero":
        payload["execution_generation"] = 0
    elif mutation == "lease_mismatch":
        payload["lease_expires_at"] = (claim.expires_at + timedelta(seconds=1)).isoformat()
    elif mutation == "lease_noncanonical":
        payload["lease_expires_at"] = claim.expires_at.isoformat().replace("+00:00", "Z")
    elif mutation == "result_case_mismatch":
        payload["case_result"]["case_id"] = "case_other"
    elif mutation == "result_source_mismatch":
        payload["case_result"]["source_id"] = "src_other"
    elif mutation == "error_case_mismatch":
        payload["terminal_persistence_error"]["case_id"] = "case_other"
    elif mutation == "error_stop_unrecognized":
        payload["terminal_persistence_error"]["stop_reason"] = "OTHER"
    elif mutation == "extra_nested_field":
        payload["case_result"]["unexpected"] = "value"

    if mutation == "duplicate_root_key":
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        needle = f'"case_id":"{case_id}"'
        target.write_text(
            encoded.replace(needle, f'"case_id":"case_other",{needle}', 1),
            encoding="utf-8",
        )
    elif mutation == "nonfinite_number":
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        target.write_text(
            encoded.replace(
                f'"execution_generation":{claim.generation}',
                '"execution_generation":NaN',
                1,
            ),
            encoding="utf-8",
        )
    elif mutation == "invalid_utf8":
        target.write_bytes(b"\xff\xfe")
    elif mutation == "oversize":
        target.write_bytes(b" " * (orchestration.RECOVERY_ARTIFACT_MAX_BYTES + 1))
    else:
        target.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )

    assert orchestration._recovery_artifact_is_valid(case_id, store, claim) is False


def test_round3_recovery_envelope_is_invalid_after_generation_changes(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    result = orchestration._cancelled_result(
        case_id,
        store.load_authoritative_case_source_id(claim),
        started_at,
    )
    error = ErrorRecord(
        category=ErrorCategory.DATABASE,
        message="terminal database write failed",
        case_id=case_id,
        stop_reason="TERMINAL_PERSISTENCE_FAILED",
    )
    monkeypatch.chdir(tmp_path)
    orchestration._recovery_artifact_or_raise(result, error, store=store, claim=claim)
    assert orchestration._recovery_artifact_is_valid(case_id, store, claim)

    store.release_case_execution(claim)
    successor = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    assert successor.generation == claim.generation + 1
    assert orchestration._recovery_artifact_is_valid(case_id, store, claim) is False
    assert orchestration._recovery_artifact_is_valid(case_id, store, successor) is False


def test_round3_terminal_protocol_omits_provider_secret(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    _case_id, started_at, claim = prepared
    canary = "sk-proj-terminal-protocol-canary"
    secret_settings = settings.model_copy(update={"xai_api_key": SecretStr(canary)})

    encoded = terminal_process._encode_request(
        mode="cancel_unstarted",
        settings=secret_settings,
        claim=claim,
        started_at=started_at,
    )
    assert canary.encode() not in encoded
    payload = json.loads(encoded)
    assert payload["settings"]["xai_api_key"] is None


def test_round3_terminal_protocol_never_returns_raw_worker_output(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    _case_id, started_at, claim = prepared
    canary = "sk-proj-worker-canary /private/provider/path"
    monkeypatch.setattr(
        terminal_process,
        "_terminal_worker_command",
        lambda: [
            sys.executable,
            "-c",
            f"import sys; sys.stdout.write({canary!r})",
        ],
    )

    outcome = terminal_process.run_terminal_process(
        mode="cancel_unstarted",
        settings=settings,
        claim=claim,
        timeout_seconds=1.0,
        started_at=started_at,
    )
    assert outcome == terminal_process.TerminalProcessOutcome(
        result=None,
        error_code="TERMINAL_WORKER_FAILED",
    )
    assert canary not in repr(outcome)


def test_round3_real_terminal_helper_commits_exact_claim_without_provider_secret(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    secret_settings = settings.model_copy(
        update={"xai_api_key": SecretStr("sk-proj-real-helper-canary")}
    )

    outcome = terminal_process.run_terminal_process(
        mode="cancel_unstarted",
        settings=secret_settings,
        claim=claim,
        timeout_seconds=2.0,
        started_at=started_at,
    )
    assert outcome.error_code is None
    assert outcome.result is not None and outcome.result.stop_reason == "CANCELLED"
    snapshot = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert snapshot is not None and snapshot.execution_state == "FINISHED"


@pytest.mark.parametrize("invalid_concurrency", [0, -1, 9, True, 1.0, "1"])
@pytest.mark.asyncio
async def test_round3_ui_registry_rejects_invalid_concurrency_before_mutation(
    invoice_dir: Path,
    settings: Settings,
    invalid_concurrency: object,
) -> None:
    registry = RunRegistry()
    with pytest.raises(ValueError, match="concurrency"):
        await registry.start_batch(
            [invoice_dir / "invoice_1001.txt"],
            settings,
            invalid_concurrency,  # type: ignore[arg-type]
        )
    assert registry._batches == {}
    assert registry._runs == {}


@pytest.mark.asyncio
async def test_round3_cancellation_resistant_single_child_is_fenced_before_owner_completion(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_started = asyncio.Event()
    release_child = asyncio.Event()
    late_write_finished = asyncio.Event()
    late_stop_reason: list[str] = []

    async def resistant_runner(
        case_id: str,
        started_at: datetime,
        selected_settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        assert claim is not None
        child_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_child.wait()
        store = WorkflowStore(selected_settings)
        result = CaseResult(
            case_id=case_id,
            source_id=store.load_case_source_id(case_id),
            status=CaseStatus.FAILED,
            stop_reason="LATE_CHILD_RESULT",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
        try:
            await asyncio.to_thread(store.finish_case, result, claim)
        except InvoiceAgentsError as exc:
            late_stop_reason.append(exc.stop_reason or "")
        finally:
            late_write_finished.set()
        return result

    monkeypatch.setattr(ui_runs, "run_prepared_case", resistant_runner)
    monkeypatch.setattr(ui_runs, "DURABILITY_DEADLINE_SECONDS", 0.02)
    registry = RunRegistry()
    case_id = await registry.start_process(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(case_id, str)
    handle = registry.handle(case_id)
    assert handle is not None and handle.task is not None
    await asyncio.wait_for(child_started.wait(), timeout=1)

    handle.task.cancel()
    with pytest.raises(InvoiceAgentsError) as excinfo:
        await asyncio.wait_for(handle.task, timeout=3)
    assert excinfo.value.stop_reason == "TERMINAL_DURABILITY_TIMEOUT"
    snapshot = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert snapshot is not None and snapshot.execution_state == "FINISHED"
    assert snapshot.result is not None and snapshot.result.stop_reason == "CANCELLED"

    release_child.set()
    await asyncio.wait_for(late_write_finished.wait(), timeout=1)
    later = WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert later == snapshot
    assert late_stop_reason == ["STALE_EXECUTION_CLAIM"]


@pytest.mark.parametrize("source_id", [None, "src_wrong"])
def test_round3_terminal_store_rejects_wrong_source_without_consuming_claim(
    invoice_dir: Path,
    settings: Settings,
    source_id: str | None,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    result = CaseResult(
        case_id=case_id,
        source_id=source_id,
        status=CaseStatus.FAILED,
        stop_reason="SOURCE_ID_TEST",
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.finish_case(result, claim)
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    snapshot = store.load_case_execution_snapshot(case_id)
    assert snapshot is not None and snapshot.execution_state == "RUNNING"
    assert snapshot.result is None


def test_round3_finished_result_update_cannot_change_source_identity(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    prepared = orchestration.prepare_claimed_invoice(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    original = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.FAILED,
        stop_reason="SOURCE_ID_TEST",
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    store.finish_case(original, claim)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.update_finished_case_result(
            original.model_copy(update={"source_id": "src_wrong"}),
            claim,
        )
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    assert store.load_result(case_id) == original
