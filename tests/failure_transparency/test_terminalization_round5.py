"""Focused RED reproducers for the fifth Task 9 reliability review."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import struct
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from invoice_agents import (
    isolated_process,
    lifecycle_process,
    lifecycle_worker,
    orchestration,
    preparation_process,
    terminal_process,
)
from invoice_agents.config import Settings
from invoice_agents.db import core as db_core
from invoice_agents.db import migration_process
from invoice_agents.db.core import DatabaseKind
from invoice_agents.db.store import ExecutionClaim, WorkflowStore, validate_execution_claim
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult, CaseStatus, ErrorRecord
from invoice_agents.ui import runs as ui_runs
from invoice_agents.ui.runs import RunRegistry
from invoice_agents.ui.server import build_templates
from invoice_agents.ui.sse import _stored_terminal_payload


@pytest.fixture(autouse=True)
def _forbid_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every round-5 behavior test fails immediately at an unstubbed provider seam."""

    def forbidden(_settings: Settings) -> object:
        raise AssertionError("round5 test reached an unstubbed provider boundary")

    monkeypatch.setattr(orchestration, "create_model_client", forbidden)
    monkeypatch.setattr(
        lifecycle_process,
        "_lifecycle_worker_command",
        lambda: [sys.executable, "-c", "import sys; sys.exit(86)"],
    )


def test_round5_terminal_chronology_must_match_authoritative_start(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """A terminal result cannot replace the case's authoritative start timestamp."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, authoritative_start, claim = prepared
    store = WorkflowStore(settings)
    wrong_start = authoritative_start + timedelta(microseconds=1)
    result = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND5_WRONG_START",
        started_at=wrong_start,
        finished_at=max(datetime.now(UTC), wrong_start),
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.finish_case(result, claim)

    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    snapshot = store.load_case_execution_snapshot(case_id)
    assert snapshot is not None
    assert snapshot.execution_state == "RUNNING"
    assert snapshot.result is None


@pytest.mark.parametrize(
    "mutation",
    [
        "naive_start",
        "non_utc_start",
        "naive_finish",
        "non_utc_finish",
        "finish_before_start",
    ],
)
def test_round5_terminal_chronology_rejects_noncanonical_or_reversed_clocks(
    invoice_dir: Path,
    settings: Settings,
    mutation: str,
) -> None:
    """Both terminal clocks are canonical UTC and monotonically ordered."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, authoritative_start, claim = prepared
    store = WorkflowStore(settings)
    started_at = authoritative_start
    finished_at = max(datetime.now(UTC), authoritative_start)
    if mutation == "naive_start":
        started_at = authoritative_start.replace(tzinfo=None)
    elif mutation == "non_utc_start":
        started_at = authoritative_start.astimezone(timezone(timedelta(hours=1)))
    elif mutation == "naive_finish":
        finished_at = finished_at.replace(tzinfo=None)
    elif mutation == "non_utc_finish":
        finished_at = finished_at.astimezone(timezone(timedelta(hours=-6)))
    else:
        finished_at = authoritative_start - timedelta(microseconds=1)
    result = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND5_INVALID_CLOCK",
        started_at=started_at,
        finished_at=finished_at,
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.finish_case(result, claim)

    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    snapshot = store.load_case_execution_snapshot(case_id)
    assert snapshot is not None and snapshot.execution_state == "RUNNING"
    assert snapshot.result is None


def test_round5_normal_terminal_result_round_trips_with_pydantic_utc_wire_type(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """Strict chronology accepts the normal Pydantic Z wire representation."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    result = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND5_VALID_CLOCK",
        started_at=started_at,
        finished_at=max(datetime.now(UTC), started_at),
    )

    store.finish_case(result, claim)

    loaded = store.load_result(case_id)
    snapshot = store.load_case_execution_snapshot(case_id)
    assert loaded == result
    assert snapshot is not None and snapshot.result == result
    assert loaded is not None and loaded.started_at.utcoffset() == timedelta(0)


def test_round5_non_z_result_wire_timestamp_never_certifies_durability(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """A semantically-zero +00:00 result wire value is not canonical Pydantic Z."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    result = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND5_WIRE_CLOCK",
        started_at=started_at,
        finished_at=max(datetime.now(UTC), started_at),
    )
    store.finish_case(result, claim)
    with sqlite3.connect(settings.workflow_db) as connection:
        raw = connection.execute(
            "SELECT result_json FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]
        payload = json.loads(raw)
        assert payload["started_at"].endswith("Z")
        payload["started_at"] = payload["started_at"][:-1] + "+00:00"
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (json.dumps(payload, separators=(",", ":"), sort_keys=True), case_id),
        )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.load_result(case_id)
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    with pytest.raises(InvoiceAgentsError):
        orchestration._claim_has_durable_terminal_evidence(case_id, settings, claim)


def _publish_round5_recovery(
    invoice_path: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, datetime, ExecutionClaim, WorkflowStore, Path]:
    prepared = orchestration.prepare_claimed_invoice(invoice_path, settings)
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
    target = tmp_path / "artifacts" / "results" / f"{case_id}.recovery.json"
    return case_id, started_at, claim, store, target


def test_round5_recovery_envelope_start_is_bound_to_authoritative_case(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canonical-looking recovery result with a forged start is inadmissible."""

    case_id, started_at, claim, store, target = _publish_round5_recovery(
        invoice_dir / "invoice_1001.txt",
        settings,
        tmp_path,
        monkeypatch,
    )
    payload = json.loads(target.read_bytes())
    forged_start = started_at - timedelta(seconds=1)
    payload["case_result"]["started_at"] = forged_start.isoformat().replace("+00:00", "Z")
    target.write_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )

    assert orchestration._load_valid_recovery_result(case_id, store, claim) is None
    assert not orchestration._recovery_artifact_is_valid(case_id, store, claim)


def test_round5_recovery_rejects_malformed_prior_chronology_transactionally(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """Expired recovery cannot preserve a prior aggregate with a forged start."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    malformed = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND5_PRIOR",
        started_at=started_at - timedelta(seconds=1),
        finished_at=started_at,
    )
    with sqlite3.connect(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (malformed.model_dump_json(), case_id),
        )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.recover_expired_executions(now=claim.expires_at + timedelta(seconds=1))

    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    with sqlite3.connect(settings.workflow_db) as connection:
        row = connection.execute(
            "SELECT execution_state, result_json FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert row == ("RUNNING", malformed.model_dump_json())


@pytest.mark.asyncio
async def test_round5_run_prepared_case_rejects_caller_start_before_lifecycle(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public runner loads authoritative start and never trusts its caller's clock."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    lifecycle_calls = 0

    async def local_boundary(**_kwargs: object) -> tuple[object, None]:
        nonlocal lifecycle_calls
        lifecycle_calls += 1
        return orchestration._LifecycleBoundaryOutcome(False, "LIFECYCLE_FAILED"), None

    monkeypatch.setattr(orchestration, "_invoke_lifecycle_boundary", local_boundary)
    result = await orchestration.run_prepared_case(
        case_id,
        started_at + timedelta(seconds=1),
        settings,
        claim=claim,
    )

    assert lifecycle_calls == 0
    assert result.started_at == started_at
    assert result.stop_reason == "PERSISTED_RESULT_INVALID"


@pytest.mark.asyncio
async def test_round5_completed_public_registry_has_no_execution_claim(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed public handle cannot expose reusable execution authority."""

    case_id = "case_round5_public_registry"
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        case_id,
        f"exec_{'a' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )
    expected = CaseResult(
        case_id=case_id,
        source_id=None,
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND5_LOCAL_RESULT",
        started_at=started_at,
        finished_at=started_at,
    )

    async def durable(
        claims: list[tuple[str, datetime, ExecutionClaim]],
        _settings: Settings,
    ) -> dict[str, BaseException | None]:
        return {selected_case_id: None for selected_case_id, _started, _claim in claims}

    async def local_result() -> CaseResult:
        return expected

    monkeypatch.setattr(ui_runs, "_inspect_claim_durability", durable)
    registry = RunRegistry()
    handle = await registry._launch(
        case_id,
        "process",
        local_result(),
        claim=claim,
        source_path=None,
        settings=settings,
        claimed_started_at=started_at,
    )
    owner_task = handle.task
    assert owner_task is not None
    assert await owner_task == expected
    await asyncio.sleep(0)

    retained = repr(handle) + repr(registry._runs) + repr(registry._batches)
    assert claim.token not in retained
    assert getattr(handle, "claim", None) is None


@pytest.mark.asyncio
async def test_round5_public_batch_nested_entries_logs_templates_and_sse_hide_claims(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every public batch/display surface is claim-free while private owners execute."""

    started_at = datetime.now(UTC)
    paths = [tmp_path / "one.txt", tmp_path / "two.txt"]
    claims = {
        path: ExecutionClaim(
            f"case_round5_{path.stem}",
            f"exec_{character * 32}",
            1,
            started_at + timedelta(minutes=1),
        )
        for path, character in zip(paths, ("e", "f"), strict=True)
    }

    async def prepare(path: Path, _settings: Settings) -> tuple[str, datetime, ExecutionClaim]:
        claim = claims[path]
        return claim.case_id, started_at, claim

    async def run(
        case_id: str,
        selected_start: datetime,
        _settings: Settings,
        *,
        claim: ExecutionClaim,
    ) -> CaseResult:
        assert claim == claims[next(path for path in paths if path.stem in case_id)]
        return CaseResult(
            case_id=case_id,
            source_id=None,
            status=CaseStatus.INCOMPLETE,
            stop_reason="ROUND5_BATCH_LOCAL",
            started_at=selected_start,
            finished_at=selected_start,
        )

    async def durable(
        selected_claims: list[tuple[str, datetime, ExecutionClaim]],
        _settings: Settings,
    ) -> dict[str, BaseException | None]:
        return {case_id: None for case_id, _started, _claim in selected_claims}

    monkeypatch.setattr(ui_runs, "_prepare_claimed_for_launch", prepare)
    monkeypatch.setattr(ui_runs, "run_prepared_case", run)
    monkeypatch.setattr(ui_runs, "_inspect_claim_durability", durable)
    registry = RunRegistry()
    batch = await registry.start_batch(paths, settings, concurrency=2)
    public_while_running = repr(batch) + repr(batch.entries) + repr(registry._runs)
    for claim in claims.values():
        assert claim.token not in public_while_running
    owner_task = batch.task
    assert owner_task is not None
    await owner_task
    await asyncio.sleep(0)

    rows = [
        {
            "entry": entry,
            "header": None,
            "run_state": registry.run_state(entry.case_id),
            "run_error": registry.run_error(entry.case_id),
        }
        for entry in batch.entries
    ]
    rendered = (
        build_templates()
        .env.get_template("_batch_rows.html")
        .render(
            batch=batch,
            rows=rows,
        )
    )
    logger = logging.getLogger("round5.registry")
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("batch=%r entries=%r runs=%r", batch, batch.entries, registry._runs)
    terminal = _stored_terminal_payload(
        batch.entries[0].case_id,
        CaseResult(
            case_id=batch.entries[0].case_id,
            source_id=None,
            status=CaseStatus.INCOMPLETE,
            stop_reason="ROUND5_BATCH_LOCAL",
            started_at=started_at,
            finished_at=started_at,
        ),
        registry,
    )
    public_after = (
        repr(batch)
        + repr(batch.entries)
        + repr(registry._runs)
        + caplog.text
        + rendered
        + json.dumps(terminal, sort_keys=True)
    )
    for entry in batch.entries:
        assert getattr(entry, "claim", None) is None
    for claim in claims.values():
        assert claim.token not in public_after
    assert registry._lifecycle_owners == {}


@pytest.mark.asyncio
async def test_round5_public_handle_cannot_authorize_finished_result_update(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """The private owner finishes authoritatively, then no public reusable claim remains."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, started_at, claim = prepared
    store = WorkflowStore(settings)
    result = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.INCOMPLETE,
        stop_reason="ROUND5_PRIVATE_OWNER",
        started_at=started_at,
        finished_at=max(datetime.now(UTC), started_at),
    )

    async def finish_locally() -> CaseResult:
        store.finish_case(result, claim)
        return result

    registry = RunRegistry()
    handle = await registry._launch(
        case_id,
        "process",
        finish_locally(),
        claim=claim,
        source_path=None,
        settings=settings,
        claimed_started_at=started_at,
    )
    owner_task = handle.task
    assert owner_task is not None
    assert await owner_task == result
    await asyncio.sleep(0)
    assert registry._lifecycle_owners == {}
    public_authority = getattr(handle, "claim", None)
    assert public_authority is None

    updated = result.model_copy(update={"finished_at": datetime.now(UTC)}, deep=True)
    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.update_finished_case_result(updated, public_authority)
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"


def test_round5_wire_settings_and_execution_tokens_reject_coercion(
    tmp_path: Path,
) -> None:
    """Wire metadata cannot use BaseSettings coercion or a noncanonical claim token."""

    settings = Settings(
        xai_api_key=SecretStr("round5-local-only-key"),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    started_at = datetime.now(UTC)
    encoded = preparation_process._encode_request(
        path=tmp_path / "invoice.txt",
        settings=settings,
        case_id="case_round5_wire",
        started_at=started_at,
        preparation_token=f"exec_{'b' * 32}",
        run_token=f"exec_{'c' * 32}",
    )
    payload = json.loads(encoded)
    payload["settings"]["unexpected_wire_field"] = "ignored by BaseSettings"
    payload["settings"]["source_max_bytes"] = str(payload["settings"]["source_max_bytes"])
    malformed_wire = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    violations: list[str] = []
    try:
        preparation_process.decode_preparation_request(malformed_wire)
    except ValueError:
        pass
    else:
        violations.append("coercive or unknown wire settings were accepted")

    malformed_claim = ExecutionClaim(
        "case_round5_wire",
        "exec_not_canonical",
        1,
        started_at + timedelta(minutes=1),
    )
    try:
        validate_execution_claim(malformed_claim)
    except InvoiceAgentsError:
        pass
    else:
        violations.append("a noncanonical execution token was accepted")

    assert violations == []


def _round5_protocol_request(
    protocol: str,
    settings: Settings,
    tmp_path: Path,
) -> tuple[bytes, Any]:
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_round5_protocol",
        f"exec_{'1' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )
    if protocol == "preparation":
        encoded = preparation_process._encode_request(
            path=tmp_path / "invoice.txt",
            settings=settings,
            case_id=claim.case_id,
            started_at=started_at,
            preparation_token=f"exec_{'2' * 32}",
            run_token=f"exec_{'3' * 32}",
        )
        return encoded, preparation_process.decode_preparation_request
    if protocol == "lifecycle":
        credential = bytearray()
        encoded = lifecycle_process._encode_request(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            credential_fd=99,
            credential=credential,
        )
        for index in range(len(credential)):
            credential[index] = 0
        return encoded, lifecycle_process.decode_lifecycle_request
    if protocol == "terminal":
        encoded = terminal_process._encode_request(
            mode="cancel_unstarted",
            settings=settings,
            claim=claim,
            started_at=started_at,
        )
        return encoded, terminal_process.decode_terminal_request
    encoded = migration_process._encode_request(
        tmp_path / "workflow.db",
        DatabaseKind.WORKFLOW,
        settings,
    )
    return encoded, migration_process.decode_worker_request


_WIRE_MUTATIONS = (
    "unknown",
    "missing",
    "coerced_int",
    "bool_int",
    "int_float",
    "decimal_number",
    "date_number",
    "nonfinite",
    "secret",
)


@pytest.mark.parametrize("protocol", ["preparation", "lifecycle", "terminal", "migration"])
@pytest.mark.parametrize("mutation", _WIRE_MUTATIONS)
def test_round5_all_worker_protocols_reject_invalid_wire_settings(
    protocol: str,
    mutation: str,
    tmp_path: Path,
) -> None:
    """Every process boundary rejects malformed settings before BaseSettings construction."""

    settings = Settings(
        xai_api_key=SecretStr("round5-local-only-key"),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    encoded, decoder = _round5_protocol_request(protocol, settings, tmp_path)
    payload = json.loads(encoded)
    wire = payload["settings"]
    assert isinstance(wire, dict)
    if mutation == "unknown":
        wire["unknown"] = "value"
    elif mutation == "missing":
        del wire["source_max_bytes"]
    elif mutation == "coerced_int":
        wire["source_max_bytes"] = str(wire["source_max_bytes"])
    elif mutation == "bool_int":
        wire["pdf_max_pages"] = True
    elif mutation == "int_float":
        wire["model_timeout_seconds"] = 120
    elif mutation == "decimal_number":
        wire["review_threshold_amount"] = 10000.0
    elif mutation == "date_number":
        wire["review_threshold_effective_date"] = 20260806
    elif mutation == "nonfinite":
        wire["model_timeout_seconds"] = float("nan")
    else:
        wire["xai_api_key"] = "must-never-cross-wire"
    malformed = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    with pytest.raises((ValueError, InvoiceAgentsError)):
        decoder(malformed)


@pytest.mark.parametrize("protocol", ["preparation", "lifecycle", "terminal", "migration"])
def test_round5_all_worker_protocols_reject_duplicate_wire_setting(
    protocol: str,
    tmp_path: Path,
) -> None:
    """Duplicate nested settings keys are rejected rather than last-value-wins decoded."""

    settings = Settings(
        xai_api_key=SecretStr("round5-local-only-key"),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    encoded, decoder = _round5_protocol_request(protocol, settings, tmp_path)
    text = encoded.decode("utf-8")
    needle = f'"source_max_bytes":{settings.source_max_bytes}'
    assert needle in text
    duplicated = text.replace(needle, f"{needle},{needle}", 1).encode("utf-8")

    with pytest.raises((ValueError, InvoiceAgentsError)):
        decoder(duplicated)


@pytest.mark.parametrize("protocol", ["preparation", "lifecycle", "terminal", "migration"])
def test_round5_valid_wire_settings_round_trip_exactly(
    protocol: str,
    tmp_path: Path,
) -> None:
    """The exact native serialized field set remains a valid deterministic round trip."""

    settings = Settings(
        xai_api_key=SecretStr("round5-local-only-key"),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    encoded, decoder = _round5_protocol_request(protocol, settings, tmp_path)

    decoded = decoder(encoded)
    decoded_settings = decoded[2] if protocol == "migration" else decoded[1]
    assert isinstance(decoded_settings, Settings)
    assert decoded_settings.xai_api_key is None
    assert decoded_settings.source_max_bytes == settings.source_max_bytes
    assert decoded_settings.model_timeout_seconds == settings.model_timeout_seconds
    assert decoded_settings.review_threshold_amount == settings.review_threshold_amount
    assert (
        decoded_settings.review_threshold_effective_date == settings.review_threshold_effective_date
    )


@pytest.mark.parametrize("protocol", ["preparation", "lifecycle", "terminal"])
def test_round5_process_claim_decoders_reject_noncanonical_tokens(
    protocol: str,
    tmp_path: Path,
) -> None:
    """Malformed execution tokens are rejected at every child process admission boundary."""

    settings = Settings(
        xai_api_key=SecretStr("round5-local-only-key"),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    encoded, decoder = _round5_protocol_request(protocol, settings, tmp_path)
    payload = json.loads(encoded)
    if protocol == "preparation":
        payload["run_token"] = "exec_not_canonical"
    else:
        payload["claim"]["token"] = "exec_not_canonical"
    malformed = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    with pytest.raises((ValueError, InvoiceAgentsError)):
        decoder(malformed)


def test_round5_application_authority_tuple_rejects_malformed_token_without_trigger(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """Application validation independently rejects a malformed persisted authority token."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, _started_at, claim = prepared
    with sqlite3.connect(settings.workflow_db) as connection:
        connection.execute("DROP TRIGGER trg_cases_execution_authority_insert")
        connection.execute("DROP TRIGGER trg_cases_execution_authority_update")
        connection.execute("DROP TRIGGER trg_cases_execution_token_grammar_insert")
        connection.execute("DROP TRIGGER trg_cases_execution_token_grammar_update")
        connection.execute(
            "UPDATE cases SET execution_token = ? WHERE case_id = ?",
            ("exec_not_canonical", case_id),
        )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        WorkflowStore(settings).load_case_execution_snapshot(case_id)
    assert excinfo.value.stop_reason == "EXECUTION_AUTHORITY_CORRUPT"
    assert claim.token != "exec_not_canonical"


def test_round5_database_trigger_enforces_exact_execution_token_grammar(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """Direct SQL accepts only exec_ followed by exactly 32 lowercase hex characters."""

    prepared = orchestration.prepare_claimed_invoice(
        invoice_dir / "invoice_1001.txt",
        settings,
    )
    assert isinstance(prepared, tuple)
    case_id, _started_at, claim = prepared
    WorkflowStore(settings).release_case_execution(claim)
    lease = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    malformed_tokens = (
        "exec_short",
        f"exec_{'g' * 32}",
        f"exec_{'A' * 32}",
        f"exec_{'a' * 31}",
        f"exec_{'a' * 33}",
        f"recovery_{'a' * 32}",
    )
    with sqlite3.connect(settings.workflow_db) as connection:
        for malformed in malformed_tokens:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE cases SET execution_token = ?, execution_generation = 3, "
                    "execution_state = 'RUNNING', lease_expires_at = ? WHERE case_id = ?",
                    (malformed, lease, case_id),
                )
        legitimate = f"exec_{'0f' * 16}"
        connection.execute(
            "UPDATE cases SET execution_token = ?, execution_generation = 3, "
            "execution_state = 'RUNNING', lease_expires_at = ? WHERE case_id = ?",
            (legitimate, lease, case_id),
        )
        stored = connection.execute(
            "SELECT execution_token FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]
    assert stored == legitimate


def test_round5_forward_migration_history_contains_exact_token_grammar_hash(
    workflow_db: Path,
) -> None:
    """Fresh workflow databases record the forward migration as an exact history prefix."""

    migration = (
        Path(__file__).resolve().parents[2]
        / "src/invoice_agents/db/migrations/workflow/004_execution_token_grammar.sql"
    )
    assert migration.is_file()
    expected_hash = hashlib.sha256(migration.read_bytes()).hexdigest()
    with sqlite3.connect(workflow_db) as connection:
        versions = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_version ORDER BY version"
            ).fetchall()
        ]
        history = connection.execute(
            "SELECT ordinal, version, migration_sha256 FROM schema_migration_history "
            "ORDER BY ordinal"
        ).fetchall()
    assert versions == [1, 2, 3, 4]
    assert [row[:2] for row in history] == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert history[-1][2] == expected_hash


def _round5_build_v3_with_malformed_authority(path: Path) -> None:
    migrations = Path(__file__).resolve().parents[2] / "src/invoice_agents/db/migrations/workflow"
    with sqlite3.connect(path) as connection:
        connection.executescript((migrations / "001_initial.sql").read_text(encoding="utf-8"))
        connection.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (1, ?)",
            ("2026-08-10T12:00:00+00:00",),
        )
        connection.execute(
            "INSERT INTO cases(case_id, status, stop_reason, started_at, updated_at) "
            "VALUES (?, 'INCOMPLETE', 'CASE_CREATED', ?, ?)",
            (
                "case_round5_migration",
                "2026-08-10T12:00:00+00:00",
                "2026-08-10T12:00:00+00:00",
            ),
        )
        connection.commit()
        connection.executescript(
            (migrations / "002_review_sequence.sql").read_text(encoding="utf-8")
        )
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (2, ?)",
            ("2026-08-10T12:00:01+00:00",),
        )
        connection.commit()
        connection.executescript(
            (migrations / "003_execution_fencing.sql").read_text(encoding="utf-8")
        )
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (3, ?)",
            ("2026-08-10T12:00:02+00:00",),
        )
        connection.execute(
            "UPDATE cases SET execution_token = 'exec_not_canonical', "
            "execution_generation = 1, execution_state = 'FINISHED' "
            "WHERE case_id = 'case_round5_migration'"
        )
        connection.commit()


def test_round5_token_migration_rejects_bad_history_atomically(tmp_path: Path) -> None:
    """An invalid v3 authority aborts before old triggers or history are changed."""

    target = tmp_path / "workflow-v3.db"
    _round5_build_v3_with_malformed_authority(target)
    migration = (
        Path(__file__).resolve().parents[2]
        / "src/invoice_agents/db/migrations/workflow/004_execution_token_grammar.sql"
    )
    assert migration.is_file()
    statements = db_core._migration_statements(migration.read_text(encoding="utf-8"))
    with sqlite3.connect(target) as connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.IntegrityError):
            for statement in statements:
                connection.execute(statement)
        connection.rollback()
        token = connection.execute(
            "SELECT execution_token FROM cases WHERE case_id = 'case_round5_migration'"
        ).fetchone()[0]
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'trg_cases_execution_authority_%'"
            ).fetchall()
        }
        guard = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'execution_token_migration_guard'"
        ).fetchone()[0]
    assert token == "exec_not_canonical"
    assert triggers == {
        "trg_cases_execution_authority_insert",
        "trg_cases_execution_authority_update",
    }
    assert guard == 0


def test_round5_credentials_use_no_filesystem_transport_or_ambient_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential transport must be anonymous and worker environment explicitly allowlisted."""

    aliases = {
        "XAI_KEY": "round5-xai-alias",
        "PASSWORD": "round5-password-alias",
        "GITHUB_PAT": "round5-pat-alias",
        "AWS_ACCESS_KEY_ID": "round5-access-key-alias",
        "CLOUD_ACCESS_KEY": "round5-cloud-key-alias",
    }
    for name, value in aliases.items():
        monkeypatch.setenv(name, value)

    class FilesystemCredentialTransportUsed(Exception):
        pass

    class TemporaryFileTripwire:
        @staticmethod
        def TemporaryFile(*_args: Any, **_kwargs: Any) -> object:
            raise FilesystemCredentialTransportUsed

    monkeypatch.setattr(
        lifecycle_process,
        "tempfile",
        TemporaryFileTripwire,
        raising=False,
    )
    settings = Settings(
        xai_api_key=SecretStr("round5-private-credential"),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_round5_credential",
        f"exec_{'d' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )
    violations: list[str] = []
    try:
        lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=0.2,
        )
    except FilesystemCredentialTransportUsed:
        violations.append("provider credential used TemporaryFile transport")

    worker_environment = lifecycle_process.sanitized_worker_environment()
    inherited_aliases = sorted(set(aliases).intersection(worker_environment))
    if inherited_aliases:
        violations.append(f"ambient credential aliases were inherited: {inherited_aliases!r}")

    assert worker_environment == {
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "TZ": "UTC",
    }
    assert violations == []


def _round5_lifecycle_inputs(
    tmp_path: Path,
    credential: str,
) -> tuple[Settings, datetime, ExecutionClaim]:
    settings = Settings(
        xai_api_key=SecretStr(credential),
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
        source_archive_dir=tmp_path / "sources",
    )
    started_at = datetime.now(UTC)
    claim = ExecutionClaim(
        "case_round5_private_channel",
        f"exec_{'e' * 32}",
        1,
        started_at + timedelta(minutes=1),
    )
    return settings, started_at, claim


def test_round5_private_channel_delivers_once_and_zeroes_parent_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The secret crosses one frame only; metadata, env, argv, and buffers stay clean."""

    canary = "round5-private-datagram-canary"
    settings, started_at, claim = _round5_lifecycle_inputs(tmp_path, canary)
    child_code = (
        "import json,os,sys; "
        "from invoice_agents.lifecycle_worker import _read_private_credential; "
        "p=json.loads(sys.stdin.buffer.read()); "
        "b=_read_private_credential(p['credential_fd']); "
        "b[:]=bytes(len(b)); "
        "sys.stdout.buffer.write(b'{\"ok\":true}')"
    )
    command = [sys.executable, "-c", child_code]
    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", lambda: command)
    retained_buffers: list[bytearray] = []

    real_run = isolated_process.run_isolated_process
    observed_requests: list[bytes] = []
    observed_environments: list[dict[str, str]] = []
    observed_descriptors: list[int] = []

    def observed_run(**kwargs: Any) -> object:
        observed_requests.append(kwargs["request"])
        observed_environments.append(kwargs["env"])
        descriptor = kwargs["pass_fds"][0]
        os.fstat(descriptor)
        observed_descriptors.append(descriptor)
        retained_buffers.append(kwargs["private_input"].payload)
        return real_run(**kwargs)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=1.0,
    )

    assert outcome == lifecycle_process.LifecycleProcessOutcome(True, None)
    assert retained_buffers and all(not any(buffer) for buffer in retained_buffers)
    assert len(observed_requests) == len(observed_environments) == len(observed_descriptors) == 1
    assert canary.encode() not in observed_requests[0]
    assert canary not in repr(observed_environments[0])
    assert all(canary not in argument for argument in command)
    assert canary not in repr(outcome)
    with pytest.raises(OSError):
        os.fstat(observed_descriptors[0])


def test_round5_descendant_inherits_neither_credential_fd_nor_ambient_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker closes its one-shot FD before a close_fds=False descendant starts."""

    canary = "round5-descendant-private-canary"
    aliases = {
        "XAI_KEY": "round5-descendant-xai-alias",
        "PASSWORD": "round5-descendant-password-alias",
        "GITHUB_PAT": "round5-descendant-github-alias",
        "AWS_ACCESS_KEY_ID": "round5-descendant-aws-alias",
        "UNFAMILIAR_LOGIN_MATERIAL": "round5-descendant-unfamiliar-alias",
    }
    for name, value in aliases.items():
        monkeypatch.setenv(name, value)
    receipt = tmp_path / "descendant-isolation.json"
    descendant_code = "\n".join(
        (
            "import json, os, sys",
            "from pathlib import Path",
            "descriptor = int(sys.argv[1])",
            "try:",
            "    os.fstat(descriptor)",
            "    fd_open = True",
            "except OSError:",
            "    fd_open = False",
            f"aliases = {tuple(aliases)!r}",
            "payload = {'fd_open': fd_open, 'present_aliases': sorted(name for name in aliases if name in os.environ), 'credential_sha256': sys.argv[2]}",
            f"Path({os.fspath(receipt)!r}).write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')",
        )
    )
    child_code = "\n".join(
        (
            "import hashlib, json, os, subprocess, sys",
            "from invoice_agents.lifecycle_worker import _read_private_credential",
            "payload = json.loads(sys.stdin.buffer.read())",
            "descriptor = payload['credential_fd']",
            "credential = _read_private_credential(descriptor)",
            "digest = hashlib.sha256(credential).hexdigest()",
            "credential[:] = bytes(len(credential))",
            f"subprocess.run([sys.executable, '-c', {descendant_code!r}, str(descriptor), digest], check=True, close_fds=False)",
            "sys.stdout.buffer.write(b'{\"ok\":true}')",
        )
    )
    command = [sys.executable, "-c", child_code]
    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", lambda: command)
    settings, started_at, claim = _round5_lifecycle_inputs(tmp_path, canary)

    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=2.0,
    )

    assert outcome == lifecycle_process.LifecycleProcessOutcome(True, None)
    observed = json.loads(receipt.read_text(encoding="utf-8"))
    assert observed == {
        "credential_sha256": hashlib.sha256(canary.encode()).hexdigest(),
        "fd_open": False,
        "present_aliases": [],
    }
    serialized_receipt = receipt.read_text(encoding="utf-8")
    assert canary not in serialized_receipt
    assert all(value not in serialized_receipt for value in aliases.values())


@pytest.mark.parametrize(
    ("worker_mode", "expected_code"),
    [
        ("success", None),
        ("crash", "LIFECYCLE_WORKER_CRASHED"),
        ("timeout", "LIFECYCLE_WORKER_TIMED_OUT"),
        ("cancel", "LIFECYCLE_WORKER_CANCELLED"),
        ("start", "LIFECYCLE_WORKER_CRASHED"),
    ],
)
def test_round5_parent_credential_descriptor_closes_on_every_worker_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_mode: str,
    expected_code: str | None,
) -> None:
    """Parent copies never survive success, crash, timeout, cancellation, or start failure."""

    settings, started_at, claim = _round5_lifecycle_inputs(
        tmp_path,
        f"round5-{worker_mode}-fd-canary",
    )
    if worker_mode == "start":
        command = [os.fspath(tmp_path / "missing-worker-executable")]
    else:
        tail = {
            "success": (
                "os.read(descriptor,16385); os.close(descriptor); "
                "sys.stdout.buffer.write(b'{\"ok\":true}')"
            ),
            "crash": "os._exit(9)",
            "timeout": "time.sleep(30)",
            "cancel": "time.sleep(30)",
        }[worker_mode]
        command = [
            sys.executable,
            "-c",
            "import json,os,sys,time; "
            "payload=json.loads(sys.stdin.buffer.read()); "
            "descriptor=payload['credential_fd']; " + tail,
        ]
    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", lambda: command)
    real_run = isolated_process.run_isolated_process
    descriptors: list[int] = []

    def observed_run(**kwargs: Any) -> object:
        descriptors.extend(kwargs["pass_fds"])
        return real_run(**kwargs)

    monkeypatch.setattr(lifecycle_process, "run_isolated_process", observed_run)
    cancel_requested = isolated_process.ProcessCancellation()
    if worker_mode == "cancel":
        cancel_requested.set()
    outcome = lifecycle_process.run_lifecycle_process(
        mode="process",
        settings=settings,
        claim=claim,
        started_at=started_at,
        timeout_seconds=0.15 if worker_mode in {"timeout", "cancel"} else 1.0,
        cancel_requested=cancel_requested,
    )

    assert outcome.error_code == expected_code
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_round5_worker_reads_one_bounded_frame_and_closes_before_return() -> None:
    """The real worker reader consumes one anonymous frame and closes its descriptor."""

    canary = bytearray(b"round5-worker-reader-canary")
    reader, writer = lifecycle_process._private_credential_channel()
    descriptor = reader.detach()
    try:
        lifecycle_process._send_private_credential(writer, canary)
    finally:
        writer.close()
    credential = lifecycle_worker._read_private_credential(descriptor)
    try:
        assert credential == canary
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        credential[:] = bytes(len(credential))
        canary[:] = bytes(len(canary))


@pytest.mark.parametrize("declared", [0, lifecycle_process.LIFECYCLE_MAX_CREDENTIAL_BYTES + 1])
def test_round5_worker_rejects_invalid_frame_size_and_closes_reader(
    declared: int,
) -> None:
    """Empty/oversized frames are rejected through a real anonymous pipe."""

    descriptor, writer = os.pipe()
    os.write(writer, struct.pack("!I", declared))
    os.close(writer)
    with pytest.raises(ValueError, match="credential size"):
        lifecycle_worker._read_private_credential(descriptor)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_round5_worker_closes_transport_when_private_read_fails() -> None:
    """A partial real frame cannot strand the inherited credential descriptor."""

    descriptor, writer = os.pipe()
    os.write(writer, b"\x00\x00")
    os.close(writer)
    with pytest.raises(ValueError, match="incomplete private credential frame"):
        lifecycle_worker._read_private_credential(descriptor)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_round5_provider_output_cannot_expose_private_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raw worker stdout/stderr is never promoted into errors, logs, or artifacts."""

    canary = "round5-provider-output-canary"
    settings, started_at, claim = _round5_lifecycle_inputs(tmp_path, canary)
    child_code = (
        "import json,os,sys; "
        "p=json.loads(sys.stdin.buffer.read()); "
        "b=bytearray(os.read(p['credential_fd'],16385)); "
        "os.close(p['credential_fd']); "
        "sys.stderr.buffer.write(b); sys.stderr.buffer.flush(); "
        "sys.stdout.buffer.write(b); sys.stdout.buffer.flush(); "
        "b[:]=bytes(len(b))"
    )
    command = [sys.executable, "-c", child_code]
    monkeypatch.setattr(lifecycle_process, "_lifecycle_worker_command", lambda: command)

    with caplog.at_level(logging.DEBUG):
        outcome = lifecycle_process.run_lifecycle_process(
            mode="process",
            settings=settings,
            claim=claim,
            started_at=started_at,
            timeout_seconds=1.0,
        )

    assert outcome.error_code == "LIFECYCLE_WORKER_PROTOCOL_INVALID"
    assert canary not in repr(outcome)
    assert canary not in caplog.text
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert canary.encode() not in artifact.read_bytes()
