"""Focused RED/GREEN coverage for Task 9 round-6 terminal chronology."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest

from invoice_agents import orchestration
from invoice_agents.config import Settings
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.models import CaseResult, CaseStatus
from invoice_agents.ui.runs import RunRegistry
from invoice_agents.ui.sse import terminal_payload

RelationalFinishMutation = Literal[
    "sql_null",
    "malformed",
    "z_spelling",
    "equivalent_offset",
    "mismatch",
    "reversed",
]
_RELATIONAL_FINISH_MUTATIONS: tuple[RelationalFinishMutation, ...] = (
    "sql_null",
    "malformed",
    "z_spelling",
    "equivalent_offset",
    "mismatch",
    "reversed",
)


@pytest.fixture(autouse=True)
def _forbid_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chronology tests are local-only and must never cross a provider seam."""

    def forbidden(_settings: Settings) -> object:
        raise AssertionError("round6 chronology test reached a provider boundary")

    monkeypatch.setattr(orchestration, "create_model_client", forbidden)


def _finish_legitimate_case(
    invoice_dir: Path,
    settings: Settings,
    *,
    status: CaseStatus = CaseStatus.SUCCEEDED,
) -> tuple[str, ExecutionClaim, WorkflowStore, CaseResult]:
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
        status=status,
        stop_reason="ROUND6_LEGITIMATE_SUCCESS",
        started_at=started_at,
        finished_at=max(datetime.now(UTC), started_at),
    )
    store.finish_case(result, claim)
    return case_id, claim, store, result


def _mutated_relational_finish(
    result: CaseResult,
    mutation: RelationalFinishMutation,
) -> str | None:
    if mutation == "sql_null":
        return None
    if mutation == "malformed":
        return "not-a-terminal-clock"
    if mutation == "z_spelling":
        return result.finished_at.isoformat().replace("+00:00", "Z")
    if mutation == "equivalent_offset":
        return result.finished_at.astimezone(timezone(timedelta(hours=-6))).isoformat()
    if mutation == "mismatch":
        return (result.finished_at + timedelta(microseconds=1)).isoformat()
    return (result.started_at - timedelta(microseconds=1)).isoformat()


def _tamper_relational_finish(
    settings: Settings,
    case_id: str,
    result: CaseResult,
    mutation: RelationalFinishMutation,
) -> str | None:
    stored = _mutated_relational_finish(result, mutation)
    with sqlite3.connect(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET finished_at = ? WHERE case_id = ?",
            (stored, case_id),
        )
    return stored


def test_round6_sql_null_relational_finish_never_certifies_terminal_result(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    """Explicit SQL NULL cannot mean that the relational finish check was omitted."""

    case_id, _claim, store, result = _finish_legitimate_case(invoice_dir, settings)
    _tamper_relational_finish(settings, case_id, result, "sql_null")

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.load_result(case_id)
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"


@pytest.mark.parametrize(
    "mutation",
    ["malformed", "z_spelling", "equivalent_offset", "mismatch", "reversed"],
)
def test_round6_read_rejects_every_noncanonical_or_unequal_relational_finish(
    invoice_dir: Path,
    settings: Settings,
    mutation: RelationalFinishMutation,
) -> None:
    """A persisted aggregate is readable only with its one exact relational finish."""

    case_id, _claim, store, result = _finish_legitimate_case(invoice_dir, settings)
    _tamper_relational_finish(settings, case_id, result, mutation)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.load_result(case_id)
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"


@pytest.mark.parametrize("mutation", _RELATIONAL_FINISH_MUTATIONS)
def test_round6_snapshot_and_durability_reject_corrupt_relational_finish(
    invoice_dir: Path,
    settings: Settings,
    mutation: RelationalFinishMutation,
) -> None:
    """A corrupt relational finish cannot become a durable exact-claim result."""

    case_id, claim, store, result = _finish_legitimate_case(invoice_dir, settings)
    _tamper_relational_finish(settings, case_id, result, mutation)

    with pytest.raises(InvoiceAgentsError) as snapshot_error:
        store.load_case_execution_snapshot(case_id)
    assert snapshot_error.value.stop_reason == "PERSISTED_RESULT_INVALID"
    with pytest.raises(InvoiceAgentsError) as durability_error:
        orchestration._claim_has_durable_terminal_evidence(case_id, settings, claim)
    assert durability_error.value.stop_reason == "PERSISTED_RESULT_INVALID"


@pytest.mark.parametrize("mutation", _RELATIONAL_FINISH_MUTATIONS)
def test_round6_sse_never_publishes_success_from_corrupt_relational_finish(
    invoice_dir: Path,
    settings: Settings,
    mutation: RelationalFinishMutation,
) -> None:
    """SSE exposes a fail-closed corruption terminal instead of stored success."""

    case_id, _claim, _store, result = _finish_legitimate_case(invoice_dir, settings)
    _tamper_relational_finish(settings, case_id, result, mutation)

    payload = terminal_payload(
        settings.workflow_db,
        case_id,
        RunRegistry(),
        settings=settings,
    )

    assert payload is not None
    assert payload["status"] == "INCOMPLETE"
    assert payload["stop_reason"] == "PERSISTED_RESULT_INVALID"
    assert payload["status"] != "SUCCEEDED"


@pytest.mark.parametrize("mutation", _RELATIONAL_FINISH_MUTATIONS)
def test_round6_recovery_reconciliation_rejects_corrupt_stored_aggregate(
    invoice_dir: Path,
    settings: Settings,
    mutation: RelationalFinishMutation,
) -> None:
    """Recovery reconciliation cannot admit a result beside a corrupt stored aggregate."""

    case_id, _claim, store, result = _finish_legitimate_case(invoice_dir, settings)
    _tamper_relational_finish(settings, case_id, result, mutation)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.merge_relational_case_evidence(result)
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"


@pytest.mark.parametrize("mutation", _RELATIONAL_FINISH_MUTATIONS)
def test_round6_resumption_admission_rejects_corrupt_finished_predecessor(
    invoice_dir: Path,
    settings: Settings,
    mutation: RelationalFinishMutation,
) -> None:
    """A new generation cannot claim a corrupt finished predecessor."""

    case_id, claim, store, result = _finish_legitimate_case(
        invoice_dir,
        settings,
        status=CaseStatus.INCOMPLETE,
    )
    _tamper_relational_finish(settings, case_id, result, mutation)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.claim_case_execution(
            case_id,
            frozenset({CaseStatus.INCOMPLETE}),
            lease_seconds=60,
        )
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    with sqlite3.connect(settings.workflow_db) as connection:
        authority = connection.execute(
            "SELECT execution_token, execution_generation, execution_state, lease_expires_at "
            "FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert authority == (claim.token, claim.generation, "FINISHED", None)


@pytest.mark.parametrize("mutation", _RELATIONAL_FINISH_MUTATIONS)
def test_round6_finished_result_update_cannot_overwrite_relational_finish_corruption(
    invoice_dir: Path,
    settings: Settings,
    mutation: RelationalFinishMutation,
) -> None:
    """An exact claim cannot use a later update to conceal prior terminal corruption."""

    case_id, claim, store, result = _finish_legitimate_case(
        invoice_dir,
        settings,
        status=CaseStatus.INCOMPLETE,
    )
    tampered_finish = _tamper_relational_finish(settings, case_id, result, mutation)
    replacement = result.model_copy(
        update={
            "stop_reason": "ROUND6_REPLACEMENT",
            "finished_at": result.finished_at + timedelta(microseconds=2),
        },
        deep=True,
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.update_finished_case_result(replacement, claim)
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    with sqlite3.connect(settings.workflow_db) as connection:
        stored = connection.execute(
            "SELECT result_json, finished_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert stored == (result.model_dump_json(), tampered_finish)


@pytest.mark.parametrize("mutation", _RELATIONAL_FINISH_MUTATIONS)
def test_round6_expired_recovery_rejects_corrupt_predecessor_transactionally(
    invoice_dir: Path,
    settings: Settings,
    mutation: RelationalFinishMutation,
) -> None:
    """Expired recovery neither masks predecessor corruption nor advances authority."""

    case_id, _finished_claim, store, result = _finish_legitimate_case(
        invoice_dir,
        settings,
        status=CaseStatus.INCOMPLETE,
    )
    resumed_claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    tampered_finish = _tamper_relational_finish(settings, case_id, result, mutation)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.recover_expired_executions(
            now=resumed_claim.expires_at + timedelta(microseconds=1),
            case_id=case_id,
        )
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"
    with sqlite3.connect(settings.workflow_db) as connection:
        authority = connection.execute(
            "SELECT execution_token, execution_generation, execution_state, lease_expires_at, "
            "result_json, finished_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        recovery_events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE case_id = ? "
            "AND event_type = 'case.execution_recovered'",
            (case_id,),
        ).fetchone()[0]
    assert authority == (
        resumed_claim.token,
        resumed_claim.generation,
        "RUNNING",
        resumed_claim.expires_at.isoformat(),
        result.model_dump_json(),
        tampered_finish,
    )
    assert recovery_events == 0
