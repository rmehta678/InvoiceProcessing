"""Adversarial database races for execution ownership and payment authorization."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from autogen_agentchat.base import TaskResult

import invoice_agents.db.store as store_module
import invoice_agents.payment.service as payment_module
from invoice_agents import orchestration
from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.hitl.service import create_review_request, record_human_decision
from invoice_agents.models import (
    CaseResult,
    CaseStatus,
    Critique,
    DecisionKind,
    FinalDecision,
    HumanDecisionKind,
    PaymentStatus,
    RiskAssessment,
)
from invoice_agents.orchestration import resume_case
from invoice_agents.payment.service import mock_payment
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.comparison import (
    InventoryReader,
    build_risk_assessment,
    compare_inventory,
    compute_invoice_totals,
)
from invoice_agents.tools.evidence import extract_invoice_evidence


def _persist_case(settings: Settings, case_id: str) -> WorkflowStore:
    source = snapshot_source(
        Path("data/invoices/invoice_1001.txt"),
        settings.source_archive_dir,
        max_bytes=settings.source_max_bytes,
    )
    invoice = extract_invoice_evidence(source)
    store = WorkflowStore(settings.workflow_db)
    store.register_source(source)
    store.create_case(case_id, source, datetime.now(UTC))
    store.save_extraction(case_id, invoice)
    comparisons, _unresolved = compare_inventory(invoice, InventoryReader(settings.inventory_db))
    risk = build_risk_assessment(
        invoice,
        comparisons,
        [],
        compute_invoice_totals(invoice),
        settings,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"))
    store.save_critique(
        case_id,
        Critique(
            supported_findings=["all deterministic evidence supports approval"],
            challenged_findings=[],
            missing_evidence=[],
            requested_follow_up=[],
            recommended_disposition=DecisionKind.APPROVE,
            rationale=["race-test fixture has no unresolved evidence"],
        ),
    )
    return store


def _approve(store: WorkflowStore, case_id: str, claim: ExecutionClaim) -> FinalDecision:
    decision = FinalDecision(
        decision=DecisionKind.APPROVE,
        reasons=["all evidence supports payment"],
        critic_disposition=DecisionKind.APPROVE,
        payment_eligible=True,
    )
    store.save_final_decision(case_id, decision, claim)
    return decision


def test_two_resume_claims_have_exactly_one_database_owner(settings: Settings) -> None:
    _persist_case(settings, "case_resume_race")
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET status = ?, stop_reason = ? WHERE case_id = ?",
            (CaseStatus.NEEDS_HUMAN, "HUMAN_REVIEW_REQUESTED", "case_resume_race"),
        )
        connection.commit()
    barrier = threading.Barrier(2)

    def resume_attempt() -> ExecutionClaim | InvoiceAgentsError:
        barrier.wait(timeout=5)
        try:
            return WorkflowStore(settings.workflow_db).claim_case_execution(
                "case_resume_race",
                frozenset({CaseStatus.NEEDS_HUMAN}),
                lease_seconds=60,
            )
        except InvoiceAgentsError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: resume_attempt(), range(2)))

    owners = [outcome for outcome in outcomes if isinstance(outcome, ExecutionClaim)]
    rejected = [outcome for outcome in outcomes if isinstance(outcome, InvoiceAgentsError)]
    assert len(owners) == 1
    assert len(rejected) == 1
    assert rejected[0].stop_reason == "CASE_ALREADY_CLAIMED"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT execution_token, execution_generation, execution_state "
            "FROM cases WHERE case_id = ?",
            ("case_resume_race",),
        ).fetchone()
    assert row is not None
    assert row["execution_token"] == owners[0].token
    assert row["execution_generation"] == owners[0].generation
    assert row["execution_state"] == "RUNNING"


class _StubModelClient:
    async def close(self) -> None:
        return None


class _PausedResumeTeam:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def load_state(self, _state: dict[str, Any]) -> None:
        return None

    async def run_stream(self, task: object) -> AsyncIterator[object]:
        self.entered.set()
        await self.release.wait()
        yield TaskResult(messages=[], stop_reason="race fixture completed")

    async def save_state(self) -> dict[str, object]:
        return {"race": "winner"}


@pytest.mark.asyncio
async def test_two_orchestration_resume_attempts_have_one_database_owner(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = "case_resume_entrypoint_race"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    risk = store.load_comparison(case_id, "risk")
    review = create_review_request(
        case_id,
        invoice,
        RiskAssessment.model_validate(risk),
        store.load_critique(case_id),
        DecisionKind.HOLD,
        ["race fixture requires a persisted ruling"],
        store,
        extra_reasons=["race fixture requires a persisted ruling"],
    )
    setup_claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.save_team_state(case_id, {"fixture": "stopped"}, setup_claim)
    store.finish_case(
        CaseResult(
            case_id=case_id,
            source_id=invoice.source.source_id,
            status=CaseStatus.NEEDS_HUMAN,
            stop_reason="HUMAN_REVIEW_REQUESTED",
            review_request=review,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        ),
        setup_claim,
    )
    record_human_decision(
        review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.REJECT,
        "the invoice is rejected",
        store,
        settings.inventory_db,
    )
    team = _PausedResumeTeam()
    monkeypatch.setattr(orchestration, "create_model_client", lambda _settings: _StubModelClient())
    monkeypatch.setattr(orchestration, "build_team", lambda _context, _client: team)

    attempts = [asyncio.create_task(resume_case(case_id, settings)) for _ in range(2)]
    await asyncio.wait_for(team.entered.wait(), timeout=5)
    await asyncio.sleep(0)
    loser = next((attempt for attempt in attempts if attempt.done()), None)
    assert loser is not None
    team.release.set()
    outcomes = await asyncio.gather(*attempts, return_exceptions=True)

    completed = [outcome for outcome in outcomes if isinstance(outcome, CaseResult)]
    rejected = [outcome for outcome in outcomes if isinstance(outcome, InvoiceAgentsError)]
    assert len(completed) == 1
    assert len(rejected) == 1
    assert rejected[0].stop_reason == "CASE_ALREADY_CLAIMED"


def test_expired_claim_cannot_write_after_new_generation_owns_case(settings: Settings) -> None:
    store = _persist_case(settings, "case_stale_claim")
    stale = store.claim_case_execution(
        "case_stale_claim", frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            ("2000-01-01T00:00:00+00:00", "case_stale_claim"),
        )
        connection.commit()
    current = store.claim_case_execution(
        "case_stale_claim", frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    assert current.generation == stale.generation + 1

    stale_result = CaseResult(
        case_id="case_stale_claim",
        source_id=store.load_extraction("case_stale_claim").source.source_id,
        status=CaseStatus.FAILED,
        stop_reason="STALE_WRITER_MUST_NOT_COMMIT",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    stale_writes = (
        lambda: store.save_team_state("case_stale_claim", {"owner": "stale"}, stale),
        lambda: store.save_final_decision(
            "case_stale_claim",
            FinalDecision(
                decision=DecisionKind.REJECT,
                reasons=["stale writer"],
                critic_disposition=DecisionKind.REJECT,
                payment_eligible=False,
            ),
            stale,
        ),
        lambda: store.finish_case(stale_result, stale),
    )
    for write in stale_writes:
        with pytest.raises(InvoiceAgentsError) as excinfo:
            write()
        assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"

    store.save_team_state("case_stale_claim", {"owner": "current"}, current)
    assert store.load_team_state("case_stale_claim") == {"owner": "current"}


def test_only_current_unexpired_claim_can_renew(settings: Settings) -> None:
    store = _persist_case(settings, "case_claim_renewal")
    claim = store.claim_case_execution(
        "case_claim_renewal", frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    renewed = store.renew_case_execution(claim, lease_seconds=120)
    assert renewed.token == claim.token
    assert renewed.generation == claim.generation
    assert renewed.expires_at > claim.expires_at
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            ("2000-01-01T00:00:00+00:00", claim.case_id),
        )
        connection.commit()
    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.renew_case_execution(renewed, lease_seconds=120)
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"


class _ConnectionProxy:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        before_execute: Any,
        after_execute: Any,
    ) -> None:
        self._connection = connection
        self._before_execute = before_execute
        self._after_execute = after_execute

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        self._before_execute(sql)
        cursor = self._connection.execute(sql, parameters)
        self._after_execute(sql)
        return cursor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def test_payment_and_competing_final_decision_cannot_commit_impossible_state(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = "case_payment_decision_race"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    approved = _approve(store, case_id, claim)
    authorization_read = threading.Event()
    writer_begin_attempted = threading.Event()
    payment_connect = payment_module.connect_database
    store_connect = store_module.connect_database

    @contextmanager
    def pause_payment(
        path: Path, *, read_only: bool = False
    ) -> Iterator[sqlite3.Connection]:
        with payment_connect(path, read_only=read_only) as connection:
            def after_execute(sql: str) -> None:
                if "SELECT c.execution_token" in sql:
                    authorization_read.set()
                    assert writer_begin_attempted.wait(timeout=5)

            yield _ConnectionProxy(
                connection,
                before_execute=lambda _sql: None,
                after_execute=after_execute,
            )  # type: ignore[misc]

    @contextmanager
    def observe_competing_writer(
        path: Path, *, read_only: bool = False
    ) -> Iterator[sqlite3.Connection]:
        with store_connect(path, read_only=read_only) as connection:
            def before_execute(sql: str) -> None:
                if sql.strip().upper() == "BEGIN IMMEDIATE":
                    writer_begin_attempted.set()

            yield _ConnectionProxy(
                connection,
                before_execute=before_execute,
                after_execute=lambda _sql: None,
            )  # type: ignore[misc]

    monkeypatch.setattr(payment_module, "connect_database", pause_payment)
    monkeypatch.setattr(store_module, "connect_database", observe_competing_writer)

    payment_outcome: list[Any] = []
    writer_outcome: list[BaseException | None] = []

    def pay() -> None:
        try:
            payment_outcome.append(
                mock_payment(case_id, invoice, store, settings.workflow_db, claim)
            )
        except BaseException as exc:  # retained for exact post-race assertion
            payment_outcome.append(exc)

    def replace_decision() -> None:
        try:
            store.save_final_decision(
                case_id,
                FinalDecision(
                    decision=DecisionKind.REJECT,
                    reasons=["competing non-approval"],
                    critic_disposition=DecisionKind.REJECT,
                    payment_eligible=False,
                ),
                claim,
            )
        except BaseException as exc:
            writer_outcome.append(exc)
        else:
            writer_outcome.append(None)

    payment_thread = threading.Thread(target=pay)
    payment_thread.start()
    assert authorization_read.wait(timeout=5)
    writer_thread = threading.Thread(target=replace_decision)
    writer_thread.start()
    payment_thread.join(timeout=10)
    writer_thread.join(timeout=10)
    assert not payment_thread.is_alive()
    assert not writer_thread.is_alive()

    assert len(payment_outcome) == 1
    assert not isinstance(payment_outcome[0], BaseException)
    assert payment_outcome[0].status is PaymentStatus.PAID
    assert len(writer_outcome) == 1
    assert isinstance(writer_outcome[0], sqlite3.IntegrityError)
    assert "PAID_FINAL_DECISION_IMMUTABLE" in str(writer_outcome[0])
    assert store.load_final_decision(case_id) == approved
    with connect_database(settings.workflow_db, read_only=True) as connection:
        paid = connection.execute(
            "SELECT COUNT(*) FROM payments WHERE case_id = ? AND status = 'PAID'", (case_id,)
        ).fetchone()[0]
    assert paid == 1
    with (
        connect_database(settings.workflow_db) as connection,
        pytest.raises(sqlite3.IntegrityError, match="PAID_FINAL_DECISION_IMMUTABLE"),
    ):
        connection.execute("DELETE FROM final_decisions WHERE case_id = ?", (case_id,))
