"""Adversarial database races for execution ownership and payment authorization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from autogen_agentchat.base import TaskResult

import invoice_agents.db.store as store_module
import invoice_agents.payment.service as payment_module
from invoice_agents import orchestration
from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind, connect_database, migrate_database, verify_database
from invoice_agents.db.store import (
    ExecutionClaim,
    WorkflowStore,
    load_generation_evidence_snapshot,
)
from invoice_agents.errors import DatabaseVerificationError, InvoiceAgentsError
from invoice_agents.hitl.service import create_review_request, record_human_decision
from invoice_agents.models import (
    CaseResult,
    CaseStatus,
    Critique,
    DecisionKind,
    ExtractedInvoice,
    FinalDecision,
    HumanDecisionKind,
    InventoryComparison,
    InventoryStatus,
    PaymentStatus,
    RiskAssessment,
)
from invoice_agents.orchestration import resume_case
from invoice_agents.payment.service import mock_payment
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.comparison import (
    InventoryReader,
    apply_mapping_evidence,
    build_risk_assessment,
    compare_inventory_evidence,
    compute_invoice_totals,
    find_prior_invoice_candidates,
)
from invoice_agents.tools.evidence import extract_invoice_evidence


def _persist_case(settings: Settings, case_id: str) -> WorkflowStore:
    source = snapshot_source(
        Path("data/invoices/invoice_1001.txt"),
        settings.source_archive_dir,
        max_bytes=settings.source_max_bytes,
    )
    invoice = extract_invoice_evidence(source)
    store = WorkflowStore(settings)
    store.register_source(source)
    store.create_case(case_id, source, datetime.now(UTC))
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.save_extraction(case_id, invoice, claim)
    mappings, comparisons, unresolved = compare_inventory_evidence(
        invoice, InventoryReader(settings.inventory_db)
    )
    invoice = apply_mapping_evidence(invoice, mappings, unresolved)
    store.save_extraction(case_id, invoice, claim)
    identity = find_prior_invoice_candidates(case_id, invoice, store)
    store.save_identity(
        case_id,
        [item.model_dump(mode="json") for item in identity],
        claim,
    )
    store.save_comparison(
        case_id,
        "inventory",
        {
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        },
        claim,
    )
    risk = build_risk_assessment(
        invoice,
        comparisons,
        identity,
        compute_invoice_totals(invoice),
        settings,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
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
        claim,
    )
    store.release_case_execution(claim)
    return store


def _approve(store: WorkflowStore, case_id: str, claim: ExecutionClaim) -> FinalDecision:
    store.adopt_latest_evidence(claim)
    decision = FinalDecision(
        decision=DecisionKind.APPROVE,
        reasons=["all evidence supports payment"],
        critic_disposition=DecisionKind.APPROVE,
        payment_eligible=True,
    )
    store.save_final_decision(case_id, decision, claim)
    return decision


def _approve_with_resolved_review(
    store: WorkflowStore,
    case_id: str,
    settings: Settings,
) -> tuple[ExtractedInvoice, ExecutionClaim]:
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.adopt_latest_evidence(claim)
    invoice = store.load_current_extraction(claim)
    risk = RiskAssessment.model_validate(store.load_current_comparison(claim, "risk"))
    critique = store.load_current_critique(claim)
    review = create_review_request(
        case_id,
        invoice,
        risk,
        critique,
        DecisionKind.HOLD,
        ["payment authorization requires an attributable human ruling"],
        store,
        claim,
        extra_reasons=["payment authorization requires an attributable human ruling"],
    )
    resolved = record_human_decision(
        review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.APPROVE,
        "the current evidence is authorized",
        store,
        settings.inventory_db,
    )
    assert resolved.human_decision is not None
    store.save_final_decision(
        case_id,
        FinalDecision(
            decision=DecisionKind.APPROVE,
            reasons=["the resolved review authorizes payment"],
            critic_disposition=DecisionKind.APPROVE,
            human_outcome=resolved.human_decision,
            payment_eligible=True,
        ),
        claim,
    )
    return invoice, claim


def _tampered_total(invoice: ExtractedInvoice) -> ExtractedInvoice:
    return invoice.model_copy(update={"declared_total": Decimal("88888.00")}, deep=True)


def _source_inaccurate_price_snapshot(invoice: ExtractedInvoice) -> ExtractedInvoice:
    """Rewrite source-derived prices and totals into a self-consistent $88,888 claim."""

    forged = invoice.model_copy(deep=True)
    target_total = Decimal("88888.00")
    unchanged_total = sum(
        (line.calculated_line_total for line in forged.lines[1:]),
        start=Decimal("0"),
    )
    first = forged.lines[0]
    first_total = target_total - unchanged_total
    first_price = first_total / first.quantity
    first.raw_unit_price = f"${first_price:.2f}"
    first.unit_price = first_price
    first.calculated_line_total = first_total
    first.evidence[0].raw_value = (
        f"{first.raw_item}|{first.raw_quantity}|${first_price:.2f}|"
        f"{first.raw_declared_line_total}"
    )
    forged.declared_subtotal = target_total
    forged.declared_tax_amount = Decimal("0.00")
    forged.declared_total = target_total
    return forged


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
    setup_claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.adopt_latest_evidence(setup_claim)
    review = create_review_request(
        case_id,
        invoice,
        RiskAssessment.model_validate(risk),
        store.load_critique(case_id),
        DecisionKind.HOLD,
        ["race fixture requires a persisted ruling"],
        store,
        setup_claim,
        extra_reasons=["race fixture requires a persisted ruling"],
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
    def pause_payment(path: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
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
    assert isinstance(writer_outcome[0], InvoiceAgentsError)
    assert writer_outcome[0].stop_reason == "PAID_FINAL_DECISION_IMMUTABLE"
    assert "immutable final decision" in str(writer_outcome[0])
    assert store.load_final_decision(case_id) == approved
    with connect_database(settings.workflow_db, read_only=True) as connection:
        paid = connection.execute(
            "SELECT COUNT(*) FROM payments WHERE case_id = ? AND status = 'PAID'", (case_id,)
        ).fetchone()[0]
    assert paid == 1
    with (
        connect_database(settings.workflow_db) as connection,
        pytest.raises(sqlite3.IntegrityError, match="FINAL_DECISION_IMMUTABLE"),
    ):
        connection.execute("DELETE FROM final_decisions WHERE case_id = ?", (case_id,))


@pytest.mark.parametrize(
    "trigger_name",
    [
        "trg_final_decisions_no_insert_after_paid",
        "trg_final_decisions_no_update_after_paid",
        "trg_final_decisions_no_delete_after_paid",
        "trg_final_decisions_authorization_insert",
        "trg_final_decisions_immutable_update",
        "trg_final_decisions_immutable_delete",
        "trg_payments_authorization_insert",
        "trg_payments_immutable_update",
        "trg_payments_immutable_delete",
        "trg_review_requests_snapshot_digest_insert",
        "trg_extractions_immutable_after_final_insert",
        "trg_human_decisions_immutable_update",
    ],
)
def test_preflight_rejects_missing_or_drifted_required_trigger(
    settings: Settings, trigger_name: str
) -> None:
    with connect_database(settings.workflow_db) as connection:
        connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.commit()
    with pytest.raises(DatabaseVerificationError) as missing:
        verify_database(settings.workflow_db, DatabaseKind.WORKFLOW)
    assert missing.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"

    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            f"CREATE TRIGGER {trigger_name} BEFORE UPDATE ON final_decisions "
            "BEGIN SELECT RAISE(ABORT, 'DRIFTED_TRIGGER'); END"
        )
        connection.commit()
    with pytest.raises(DatabaseVerificationError) as drifted:
        verify_database(settings.workflow_db, DatabaseKind.WORKFLOW)
    assert drifted.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"


def test_store_rejects_post_paid_decision_mutation_even_if_triggers_were_dropped(
    settings: Settings,
) -> None:
    case_id = "case_paid_without_triggers"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    approved = _approve(store, case_id, claim)
    payment = mock_payment(case_id, invoice, store, settings.workflow_db, claim)
    assert payment.status is PaymentStatus.PAID
    with connect_database(settings.workflow_db) as connection:
        for name in (
            "trg_final_decisions_no_insert_after_paid",
            "trg_final_decisions_no_update_after_paid",
            "trg_final_decisions_no_delete_after_paid",
            "trg_final_decisions_authorization_insert",
            "trg_final_decisions_immutable_update",
            "trg_final_decisions_immutable_delete",
        ):
            connection.execute(f"DROP TRIGGER {name}")
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_final_decision(
            case_id,
            FinalDecision(
                decision=DecisionKind.REJECT,
                reasons=["must not replace paid approval"],
                critic_disposition=DecisionKind.REJECT,
                payment_eligible=False,
            ),
            claim,
        )

    assert excinfo.value.stop_reason == "PAID_FINAL_DECISION_IMMUTABLE"
    assert store.load_final_decision(case_id) == approved


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_final_decision_row_is_immutable_immediately_after_insert(
    settings: Settings, operation: str
) -> None:
    case_id = f"case_final_row_immutable_{operation}"
    store = _persist_case(settings, case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    approved = _approve(store, case_id, claim)
    sql = (
        "UPDATE final_decisions SET payload_json = payload_json WHERE case_id = ?"
        if operation == "update"
        else "DELETE FROM final_decisions WHERE case_id = ?"
    )

    with (
        connect_database(settings.workflow_db) as connection,
        pytest.raises(sqlite3.IntegrityError, match="FINAL_DECISION_IMMUTABLE"),
    ):
        connection.execute(sql, (case_id,))

    assert store.load_final_decision(case_id) == approved


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("idempotency_key", "0" * 64),
        ("vendor", "Different Vendor"),
        ("amount", "88888.00"),
        ("currency", "EUR"),
    ],
)
def test_payment_insert_requires_exact_authorized_invoice_fields(
    settings: Settings, field: str, invalid_value: str
) -> None:
    case_id = f"case_payment_insert_mismatch_{field}"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    _approve(store, case_id, claim)
    with connect_database(settings.workflow_db) as connection:
        final_row = connection.execute(
            "SELECT evidence_snapshot_digest, source_id, invoice_number, review_id "
            "FROM final_decisions WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        values = {
            "idempotency_key": payment_module.payment_idempotency_key(invoice),
            "vendor": str(invoice.vendor.normalized_value),
            "amount": str(invoice.declared_total),
            "currency": str(invoice.currency.normalized_value),
            "source_id": str(final_row["source_id"]),
            "invoice_number": str(final_row["invoice_number"]),
            "review_id": final_row["review_id"],
        }
        values[field] = invalid_value
        with pytest.raises(sqlite3.IntegrityError, match="PAYMENT_AUTHORIZATION_INVALID"):
            connection.execute(
                "INSERT INTO payments(payment_id, case_id, idempotency_key, vendor, amount, "
                "currency, status, error, created_at, decision_generation, "
                "evidence_snapshot_digest, source_id, invoice_number, review_id) "
                "VALUES (?, ?, ?, ?, ?, ?, 'PAID', NULL, ?, ?, ?, ?, ?, ?)",
                (
                    f"pay_invalid_{field}",
                    case_id,
                    values["idempotency_key"],
                    values["vendor"],
                    values["amount"],
                    values["currency"],
                    datetime.now(UTC).isoformat(),
                    claim.generation,
                    final_row["evidence_snapshot_digest"],
                    values["source_id"],
                    values["invoice_number"],
                    values["review_id"],
                ),
            )


def test_paid_entry_cannot_coexist_with_rejected_final_or_arbitrary_amount(
    settings: Settings,
) -> None:
    case_id = "case_paid_rejected_amount_invariant"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    approved = _approve(store, case_id, claim)
    rejected = FinalDecision(
        decision=DecisionKind.REJECT,
        reasons=["relational invariant probe"],
        critic_disposition=DecisionKind.REJECT,
        payment_eligible=False,
    )
    with connect_database(settings.workflow_db) as connection:
        final_row = connection.execute(
            "SELECT evidence_snapshot_digest FROM final_decisions WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError, match="FINAL_DECISION_IMMUTABLE"):
            connection.execute(
                "UPDATE final_decisions SET payload_json = ? WHERE case_id = ?",
                (rejected.model_dump_json(), case_id),
            )
            connection.execute(
                "INSERT INTO payments(payment_id, case_id, idempotency_key, vendor, amount, "
                "currency, status, error, created_at, decision_generation, "
                "evidence_snapshot_digest) VALUES (?, ?, ?, ?, ?, ?, 'PAID', NULL, ?, ?, ?)",
                (
                    "pay_rejected_arbitrary_amount",
                    case_id,
                    payment_module.payment_idempotency_key(invoice),
                    str(invoice.vendor.normalized_value),
                    "88888.00",
                    str(invoice.currency.normalized_value),
                    datetime.now(UTC).isoformat(),
                    claim.generation,
                    final_row["evidence_snapshot_digest"],
                ),
            )
            connection.commit()

    assert store.load_final_decision(case_id) == approved
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM payments WHERE case_id = ?", (case_id,)
        ).fetchone()[0] == 0


@pytest.mark.parametrize("inject_failure", [False, True], ids=["paid", "failed"])
@pytest.mark.parametrize("operation", ["update", "delete"])
def test_payment_row_is_immutable_after_insert(
    settings: Settings, inject_failure: bool, operation: str
) -> None:
    case_id = f"case_payment_row_immutable_{operation}_{inject_failure}"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    _approve(store, case_id, claim)
    payment = mock_payment(
        case_id,
        invoice,
        store,
        settings.workflow_db,
        claim,
        inject_failure=inject_failure,
    )
    assert payment.payment_id is not None
    sql = (
        "UPDATE payments SET error = error WHERE payment_id = ?"
        if operation == "update"
        else "DELETE FROM payments WHERE payment_id = ?"
    )

    with (
        connect_database(settings.workflow_db) as connection,
        pytest.raises(sqlite3.IntegrityError, match="PAYMENT_IMMUTABLE"),
    ):
        connection.execute(sql, (payment.payment_id,))

    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM payments WHERE payment_id = ?", (payment.payment_id,)
        ).fetchone()[0] == 1


def test_duplicate_payment_revalidates_source_case_snapshot(settings: Settings) -> None:
    store = _persist_case(settings, "case_paid_source")
    first_invoice = store.load_extraction("case_paid_source")
    first_claim = store.claim_case_execution(
        "case_paid_source", frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    _approve(store, "case_paid_source", first_claim)
    first = mock_payment(
        "case_paid_source", first_invoice, store, settings.workflow_db, first_claim
    )
    assert first.status is PaymentStatus.PAID

    second_store = _persist_case(settings, "case_duplicate_attempt")
    second_invoice, second_claim = _approve_with_resolved_review(
        second_store,
        "case_duplicate_attempt",
        settings,
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "DROP TRIGGER IF EXISTS trg_comparison_results_immutable_after_final_delete"
        )
        connection.execute(
            "DELETE FROM comparison_results WHERE case_id = ? AND comparison_type = 'risk'",
            ("case_paid_source",),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as excinfo:
        mock_payment(
            "case_duplicate_attempt",
            second_invoice,
            second_store,
            settings.workflow_db,
            second_claim,
        )
    assert excinfo.value.stop_reason == "PAYMENT_LEDGER_INCONSISTENT"


def test_every_execution_evidence_write_rejects_stale_generation(settings: Settings) -> None:
    case_id = "case_stale_evidence"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    risk = RiskAssessment.model_validate(store.load_comparison(case_id, "risk"))
    stale = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.adopt_latest_evidence(stale)
    review = create_review_request(
        case_id,
        invoice,
        risk,
        store.load_current_critique(stale),
        DecisionKind.HOLD,
        ["stale writer must not persist"],
        store,
        stale,
        extra_reasons=["stale writer must not persist"],
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            ("2000-01-01T00:00:00+00:00", case_id),
        )
        connection.commit()
    store.claim_case_execution(case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60)
    retry_review = review.model_copy(
        update={"review_id": "rev_stale_retry", "status": "PENDING"}, deep=True
    )
    writes = (
        lambda: store.save_extraction(case_id, invoice, stale),
        lambda: store.save_identity(case_id, [], stale),
        lambda: store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), stale),
        lambda: store.save_critique(case_id, store.load_critique(case_id), stale),
        lambda: store.save_review(retry_review, stale),
    )
    for write in writes:
        with pytest.raises(InvoiceAgentsError) as excinfo:
            write()
        assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"

    reads = (
        lambda: store.load_current_extraction(stale),
        lambda: store.load_current_identity(stale),
        lambda: store.load_current_comparison(stale, "risk"),
        lambda: store.load_current_critique(stale),
        lambda: store.load_current_review(stale),
        lambda: store.load_current_final_decision(stale),
    )
    for read in reads:
        with pytest.raises(InvoiceAgentsError) as excinfo:
            read()
        assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"


def test_corrupt_future_lease_tuple_is_not_reclaimed(settings: Settings) -> None:
    case_id = "case_corrupt_authority"
    _persist_case(settings, case_id)
    with connect_database(settings.workflow_db) as connection:
        connection.execute("DROP TRIGGER trg_cases_execution_authority_update")
        connection.execute(
            "UPDATE cases SET execution_state = 'IDLE', execution_token = NULL, "
            "lease_expires_at = '2999-01-01T00:00:00+00:00' WHERE case_id = ?",
            (case_id,),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as excinfo:
        WorkflowStore(settings.workflow_db).claim_case_execution(
            case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
        )
    assert excinfo.value.stop_reason == "EXECUTION_AUTHORITY_CORRUPT"


@pytest.mark.parametrize(
    "lease",
    [
        "2000-01-01 00:00:00",
        "2000-01-01T00:00:00.0+00:00",
        "2000-01-01T00:00:00.000000+00:00",
        "2000-02-30T00:00:00+00:00",
        "2000-01-01T24:00:00+00:00",
        "0000-01-01T00:00:00+00:00",
    ],
)
def test_schema_rejects_noncanonical_execution_lease(settings: Settings, lease: str) -> None:
    case_id = "case_noncanonical_lease_trigger"
    _persist_case(settings, case_id)

    with (
        connect_database(settings.workflow_db) as connection,
        pytest.raises(sqlite3.IntegrityError, match="INVALID_EXECUTION_AUTHORITY"),
    ):
        connection.execute(
            "UPDATE cases SET execution_state = 'RUNNING', execution_token = ?, "
            "execution_generation = execution_generation + 1, lease_expires_at = ? "
            "WHERE case_id = ?",
            ("exec_noncanonical", lease, case_id),
        )


def test_claim_rejects_noncanonical_expired_lease_before_cas_adoption(
    settings: Settings,
) -> None:
    case_id = "case_noncanonical_lease_claim"
    _persist_case(settings, case_id)
    with connect_database(settings.workflow_db) as connection:
        connection.execute("DROP TRIGGER trg_cases_execution_authority_update")
        connection.execute(
            "UPDATE cases SET execution_state = 'RUNNING', execution_token = ?, "
            "execution_generation = execution_generation + 1, lease_expires_at = ? "
            "WHERE case_id = ?",
            ("exec_noncanonical", "2000-01-01 00:00:00", case_id),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as excinfo:
        WorkflowStore(settings.workflow_db).claim_case_execution(
            case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
        )
    assert excinfo.value.stop_reason == "EXECUTION_AUTHORITY_CORRUPT"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT execution_token, execution_generation, lease_expires_at FROM cases "
            "WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert row["execution_token"] == "exec_noncanonical"
    assert row["execution_generation"] == 2
    assert row["lease_expires_at"] == "2000-01-01 00:00:00"


@pytest.mark.parametrize(
    "expired_lease",
    ["2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00.000001+00:00"],
)
def test_canonical_expired_lease_is_taken_over(settings: Settings, expired_lease: str) -> None:
    case_id = "case_canonical_expired_lease"
    store = _persist_case(settings, case_id)
    first = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            (expired_lease, case_id),
        )
        connection.commit()

    successor = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    assert successor.generation == first.generation + 1
    assert successor.expires_at.tzinfo is UTC
    assert successor.expires_at.isoformat().endswith("+00:00")


def test_renew_rejects_noncanonical_future_lease_without_overwriting(
    settings: Settings,
) -> None:
    case_id = "case_noncanonical_lease_renew"
    store = _persist_case(settings, case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    malformed = "2999-01-01 00:00:00"
    with connect_database(settings.workflow_db) as connection:
        connection.execute("DROP TRIGGER trg_cases_execution_authority_update")
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            (malformed, case_id),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.renew_case_execution(claim, lease_seconds=60)

    assert excinfo.value.stop_reason == "EXECUTION_AUTHORITY_CORRUPT"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        stored = connection.execute(
            "SELECT lease_expires_at FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
    assert stored == malformed


def test_adoption_rejects_future_generation_instead_of_copying_global_latest(
    settings: Settings,
) -> None:
    case_id = "case_future_generation_evidence"
    store = _persist_case(settings, case_id)
    forged_source = snapshot_source(
        Path("data/invoices/invoice_1002.txt"),
        settings.source_archive_dir,
        max_bytes=settings.source_max_bytes,
    )
    forged_invoice = extract_invoice_evidence(forged_source)
    with connect_database(settings.workflow_db) as connection:
        next_version = connection.execute(
            "SELECT MAX(version) + 1 FROM extractions WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extractions(extraction_id, case_id, version, payload_json, "
            "created_at, execution_generation) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "ext_forged_future",
                case_id,
                next_version,
                forged_invoice.model_dump_json(),
                datetime.now(UTC).isoformat(),
                999,
            ),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        current = connection.execute(
            "SELECT COUNT(*) FROM extractions WHERE case_id = ? AND execution_generation = ?",
            (case_id, claim.generation),
        ).fetchone()[0]
    assert current == 0


def test_adoption_rejects_partial_predecessor_snapshot(settings: Settings) -> None:
    case_id = "case_partial_predecessor"
    store = _persist_case(settings, case_id)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "DELETE FROM critique_results WHERE case_id = ? AND execution_generation = 1",
            (case_id,),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        adopted = connection.execute(
            "SELECT COUNT(*) FROM extractions WHERE case_id = ? AND execution_generation = ?",
            (case_id, claim.generation),
        ).fetchone()[0]
    assert adopted == 0


@pytest.mark.parametrize("tampering", ["declared_total", "source_metadata"])
def test_adoption_rejects_semantically_incoherent_predecessor_snapshot(
    settings: Settings, tampering: str
) -> None:
    case_id = f"case_incoherent_predecessor_{tampering}"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    if tampering == "declared_total":
        forged = _tampered_total(invoice)
    else:
        forged = invoice.model_copy(
            update={
                "source": invoice.source.model_copy(
                    update={"canonical_path": Path("/forged/source/invoice.txt")}
                )
            },
            deep=True,
        )
    with connect_database(settings.workflow_db) as connection:
        next_version = connection.execute(
            "SELECT MAX(version) + 1 FROM extractions WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extractions(extraction_id, case_id, version, payload_json, "
            "created_at, execution_generation) VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"ext_incoherent_{tampering}",
                case_id,
                next_version,
                forged.model_dump_json(),
                datetime.now(UTC).isoformat(),
                1,
            ),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        adopted = connection.execute(
            "SELECT COUNT(*) FROM extractions WHERE case_id = ? AND execution_generation = ?",
            (case_id, claim.generation),
        ).fetchone()[0]
    assert adopted == 0


def test_adoption_rejects_inventory_relationship_tamper(settings: Settings) -> None:
    case_id = "case_incoherent_predecessor_inventory"
    store = _persist_case(settings, case_id)
    with connect_database(settings.workflow_db) as connection:
        inventory_row = connection.execute(
            "SELECT comparison_id, payload_json FROM comparison_results "
            "WHERE case_id = ? AND comparison_type = 'inventory'",
            (case_id,),
        ).fetchone()
        risk_row = connection.execute(
            "SELECT comparison_id, payload_json FROM comparison_results "
            "WHERE case_id = ? AND comparison_type = 'risk'",
            (case_id,),
        ).fetchone()
        inventory = json.loads(str(inventory_row["payload_json"]))
        risk = json.loads(str(risk_row["payload_json"]))
        forged = {
            **inventory["comparisons"][0],
            "sku": None,
            "available_stock": None,
            "queried_row": None,
            "status": "AVAILABLE",
        }
        inventory["comparisons"][0] = forged
        risk["inventory"][0] = forged
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE comparison_id = ?",
            (json.dumps(inventory), inventory_row["comparison_id"]),
        )
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE comparison_id = ?",
            (json.dumps(risk), risk_row["comparison_id"]),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        adopted = connection.execute(
            "SELECT COUNT(*) FROM extractions WHERE case_id = ? AND execution_generation = ?",
            (case_id, claim.generation),
        ).fetchone()[0]
    assert adopted == 0


def test_adoption_rejects_self_consistent_snapshot_that_disagrees_with_source_bytes(
    settings: Settings,
) -> None:
    configured = settings.model_copy(
        update={"review_threshold_amount": Decimal("100000.00")}, deep=True
    )
    case_id = "case_source_inaccurate_snapshot"
    store = _persist_case(configured, case_id)
    invoice = store.load_extraction(case_id)
    inaccurate = _source_inaccurate_price_snapshot(invoice)
    inventory_payload = store.load_comparison(case_id, "inventory")
    comparisons = [
        InventoryComparison.model_validate(item) for item in inventory_payload["comparisons"]
    ]
    for comparison in comparisons:
        comparison.evidence = [
            reference
            for line in inaccurate.lines
            if line.raw_item in comparison.raw_items
            for reference in line.evidence
        ]
    inventory_payload["comparisons"] = [
        item.model_dump(mode="json") for item in comparisons
    ]
    inaccurate_risk = build_risk_assessment(
        inaccurate,
        comparisons,
        [],
        compute_invoice_totals(inaccurate),
        configured,
    )
    assert inaccurate.declared_total == Decimal("88888.00")
    assert inaccurate_risk.financial.exact
    assert not inaccurate_risk.policy_review_reasons
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE extractions SET payload_json = ? WHERE case_id = ? "
            "AND execution_generation = 1",
            (inaccurate.model_dump_json(), case_id),
        )
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE case_id = ? "
            "AND comparison_type = 'inventory' AND execution_generation = 1",
            (json.dumps(inventory_payload), case_id),
        )
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE case_id = ? "
            "AND comparison_type = 'risk' AND execution_generation = 1",
            (inaccurate_risk.model_dump_json(), case_id),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        paid = connection.execute(
            "SELECT COUNT(*) FROM payments WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
    assert paid == 0


def test_adoption_rejects_omitted_configured_threshold_review_reason(
    settings: Settings,
) -> None:
    configured = settings.model_copy(
        update={"review_threshold_amount": Decimal("1000.00")}, deep=True
    )
    case_id = "case_threshold_reason_omitted"
    store = _persist_case(configured, case_id)
    risk = store.load_comparison(case_id, "risk")
    assert risk["policy_review_reasons"]
    risk["policy_review_reasons"] = []
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE case_id = ? "
            "AND comparison_type = 'risk' AND execution_generation = 1",
            (json.dumps(risk), case_id),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"


def test_adoption_rejects_omitted_authoritative_identity_candidate(
    settings: Settings,
) -> None:
    _persist_case(settings, "case_identity_scope_prior")
    case_id = "case_identity_scope_omitted"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    inventory = [
        InventoryComparison.model_validate(item)
        for item in store.load_comparison(case_id, "inventory")["comparisons"]
    ]
    risk = build_risk_assessment(
        invoice, inventory, [], compute_invoice_totals(invoice), settings
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE identity_results SET payload_json = '[]' WHERE case_id = ? "
            "AND execution_generation = 1",
            (case_id,),
        )
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE case_id = ? "
            "AND comparison_type = 'risk' AND execution_generation = 1",
            (risk.model_dump_json(), case_id),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"


def test_adoption_rejects_dates_not_rederived_from_extraction(settings: Settings) -> None:
    case_id = "case_risk_dates_omitted"
    store = _persist_case(settings, case_id)
    risk = store.load_comparison(case_id, "risk")
    assert risk["dates"]
    risk["dates"] = []
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE case_id = ? "
            "AND comparison_type = 'risk' AND execution_generation = 1",
            (json.dumps(risk), case_id),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"


def test_adoption_rejects_inventory_row_not_in_authoritative_database(
    settings: Settings,
) -> None:
    case_id = "case_inventory_row_inaccurate"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    inventory_payload = store.load_comparison(case_id, "inventory")
    comparisons = [
        InventoryComparison.model_validate(item) for item in inventory_payload["comparisons"]
    ]
    comparison = comparisons[0]
    assert comparison.queried_row is not None
    comparison.queried_row = comparison.queried_row.model_copy(
        update={"available_stock": 0}, deep=True
    )
    comparison.available_stock = 0
    comparison.status = InventoryStatus.OUT_OF_STOCK
    comparison.explanation = (
        f"aggregate quantity {comparison.requested_quantity} compared with exact row "
        f"{comparison.queried_row.model_dump()}"
    )
    inventory_payload["comparisons"] = [
        item.model_dump(mode="json") for item in comparisons
    ]
    risk = build_risk_assessment(
        invoice, comparisons, [], compute_invoice_totals(invoice), settings
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE case_id = ? "
            "AND comparison_type = 'inventory' AND execution_generation = 1",
            (json.dumps(inventory_payload), case_id),
        )
        connection.execute(
            "UPDATE comparison_results SET payload_json = ? WHERE case_id = ? "
            "AND comparison_type = 'risk' AND execution_generation = 1",
            (risk.model_dump_json(), case_id),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.adopt_latest_evidence(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"


def test_snapshot_digest_uses_explicit_domain_version_and_payload_length(
    settings: Settings,
) -> None:
    case_id = "case_snapshot_digest_framing"
    _persist_case(settings, case_id)
    with connect_database(settings.workflow_db, read_only=True) as connection:
        snapshot = load_generation_evidence_snapshot(connection, case_id, 1, settings)
    envelope = {
        "case_id": case_id,
        "invoice": snapshot.invoice.model_dump(mode="json"),
        "identity": [item.model_dump(mode="json") for item in snapshot.identity],
        "identity_evaluated_at": snapshot.identity_evaluated_at.isoformat(),
        "inventory": snapshot.inventory_payload,
        "risk": snapshot.risk.model_dump(mode="json"),
        "critique": snapshot.critique.model_dump(mode="json"),
    }
    payload = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    expected = hashlib.sha256(
        b"galatiq.invoice-agents/evidence-snapshot\x00"
        + (1).to_bytes(4, "big")
        + len(payload).to_bytes(8, "big")
        + payload
    ).hexdigest()

    assert snapshot.digest == expected


def test_human_approval_cannot_pay_a_later_same_generation_total(settings: Settings) -> None:
    case_id = "case_reviewed_total_changed_before_payment"
    store = _persist_case(settings, case_id)
    invoice, claim = _approve_with_resolved_review(store, case_id, settings)
    forged = _tampered_total(invoice)
    with connect_database(settings.workflow_db) as connection:
        connection.execute("DROP TRIGGER IF EXISTS trg_extractions_immutable_after_final_insert")
        next_version = connection.execute(
            "SELECT MAX(version) + 1 FROM extractions WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extractions(extraction_id, case_id, version, payload_json, "
            "created_at, execution_generation) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "ext_changed_after_human_approval",
                case_id,
                next_version,
                forged.model_dump_json(),
                datetime.now(UTC).isoformat(),
                claim.generation,
            ),
        )
        connection.commit()

    result = mock_payment(case_id, forged, store, settings.workflow_db, claim)

    assert result.status is PaymentStatus.NOT_ELIGIBLE
    assert result.error is not None
    assert "snapshot" in result.error
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0


@pytest.mark.parametrize(
    "write_kind",
    ["extraction", "identity", "inventory", "risk", "critique", "review"],
)
def test_store_rejects_every_authorization_evidence_write_after_final_decision(
    settings: Settings, write_kind: str
) -> None:
    case_id = f"case_post_final_store_write_{write_kind}"
    store = _persist_case(settings, case_id)
    invoice, claim = _approve_with_resolved_review(store, case_id, settings)
    risk = RiskAssessment.model_validate(store.load_current_comparison(claim, "risk"))
    review = store.load_current_review(claim)
    assert review is not None
    writes = {
        "extraction": lambda: store.save_extraction(case_id, invoice, claim),
        "identity": lambda: store.save_identity(case_id, [], claim),
        "inventory": lambda: store.save_comparison(
            case_id, "inventory", store.load_current_comparison(claim, "inventory"), claim
        ),
        "risk": lambda: store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim),
        "critique": lambda: store.save_critique(case_id, store.load_current_critique(claim), claim),
        "review": lambda: store.save_review(
            review.model_copy(
                update={"review_id": f"rev_post_final_{write_kind}", "status": "PENDING"},
                deep=True,
            ),
            claim,
        ),
    }

    with pytest.raises(InvoiceAgentsError) as excinfo:
        writes[write_kind]()

    assert excinfo.value.stop_reason == "AUTHORIZATION_EVIDENCE_IMMUTABLE"


@pytest.mark.parametrize(
    "table",
    [
        "extractions",
        "identity_results",
        "comparison_results",
        "critique_results",
        "review_requests",
    ],
)
def test_schema_rejects_authorization_evidence_update_after_final_decision(
    settings: Settings, table: str
) -> None:
    case_id = f"case_post_final_schema_write_{table}"
    store = _persist_case(settings, case_id)
    _invoice, _claim = _approve_with_resolved_review(store, case_id, settings)

    with (
        connect_database(settings.workflow_db) as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match=r"(?:AUTHORIZATION_EVIDENCE|RESOLVED_REVIEW)_IMMUTABLE",
        ),
    ):
        connection.execute(
            f"UPDATE {table} SET payload_json = payload_json WHERE case_id = ?",
            (case_id,),
        )


def test_schema_rejects_human_decision_update_after_final_decision(
    settings: Settings,
) -> None:
    case_id = "case_post_final_schema_write_human_decision"
    store = _persist_case(settings, case_id)
    _invoice, claim = _approve_with_resolved_review(store, case_id, settings)
    review = store.load_current_review(claim)
    assert review is not None

    with (
        connect_database(settings.workflow_db) as connection,
        pytest.raises(sqlite3.IntegrityError, match="HUMAN_DECISION_IMMUTABLE"),
    ):
        connection.execute(
            "UPDATE human_decisions SET reason = reason WHERE review_id = ?",
            (review.review_id,),
        )


def test_public_extraction_write_fails_after_paid_state(settings: Settings) -> None:
    case_id = "case_paid_extraction_freeze"
    store = _persist_case(settings, case_id)
    invoice, claim = _approve_with_resolved_review(store, case_id, settings)
    paid = mock_payment(case_id, invoice, store, settings.workflow_db, claim)
    assert paid.status is PaymentStatus.PAID

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_extraction(case_id, _tampered_total(invoice), claim)

    assert excinfo.value.stop_reason == "AUTHORIZATION_EVIDENCE_IMMUTABLE"


def test_fresh_extraction_promotion_rejects_future_generation(
    settings: Settings,
) -> None:
    case_id = "case_future_generation_promotion"
    store = _persist_case(settings, case_id)
    invoice = store.load_extraction(case_id)
    with connect_database(settings.workflow_db) as connection:
        next_version = connection.execute(
            "SELECT MAX(version) + 1 FROM extractions WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extractions(extraction_id, case_id, version, payload_json, "
            "created_at, execution_generation) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "ext_forged_promotion",
                case_id,
                next_version,
                invoice.model_copy(
                    update={
                        "invoice_number": invoice.invoice_number.model_copy(
                            update={"normalized_value": "INV-1002"}
                        )
                    }
                ).model_dump_json(),
                datetime.now(UTC).isoformat(),
                999,
            ),
        )
        connection.commit()
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.promote_predecessor_extraction(claim)

    assert excinfo.value.stop_reason == "EVIDENCE_PROVENANCE_INVALID"


@pytest.mark.parametrize("corruption", ["pending", "missing", "mismatched"])
def test_payment_reconciles_review_with_authoritative_human_decision_rows(
    settings: Settings, corruption: str
) -> None:
    case_id = f"case_review_reconciliation_{corruption}"
    store = _persist_case(settings, case_id)
    invoice, claim = _approve_with_resolved_review(store, case_id, settings)
    review = store.load_current_review(claim)
    assert review is not None
    with connect_database(settings.workflow_db) as connection:
        if corruption == "pending":
            connection.execute(
                "DROP TRIGGER IF EXISTS trg_review_requests_immutable_after_final_update"
            )
            connection.execute("DROP TRIGGER IF EXISTS trg_resolved_review_immutable_update")
            connection.execute(
                "UPDATE review_requests SET status = 'PENDING', resolved_at = NULL "
                "WHERE review_id = ?",
                (review.review_id,),
            )
        elif corruption == "missing":
            connection.execute("DROP TRIGGER IF EXISTS trg_human_decisions_immutable_delete")
            connection.execute(
                "DELETE FROM human_decisions WHERE review_id = ?", (review.review_id,)
            )
        else:
            connection.execute("DROP TRIGGER IF EXISTS trg_human_decisions_immutable_update")
            connection.execute(
                "UPDATE human_decisions SET reason = ? WHERE review_id = ?",
                ("contradictory relational reason", review.review_id),
            )
        connection.commit()

    result = mock_payment(case_id, invoice, store, settings.workflow_db, claim)

    assert result.status is PaymentStatus.NOT_ELIGIBLE
    assert result.error == "review authorization records are inconsistent"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0


def test_duplicate_reconciles_paid_source_human_decision_rows(settings: Settings) -> None:
    source_case = "case_paid_review_source"
    store = _persist_case(settings, source_case)
    source_invoice, source_claim = _approve_with_resolved_review(store, source_case, settings)
    paid = mock_payment(
        source_case,
        source_invoice,
        store,
        settings.workflow_db,
        source_claim,
    )
    assert paid.status is PaymentStatus.PAID

    duplicate_case = "case_paid_review_duplicate"
    _persist_case(settings, duplicate_case)
    duplicate_invoice, duplicate_claim = _approve_with_resolved_review(
        store,
        duplicate_case,
        settings,
    )
    review = store.load_current_review(source_claim)
    assert review is not None
    with connect_database(settings.workflow_db) as connection:
        connection.execute("DROP TRIGGER IF EXISTS trg_human_decisions_immutable_delete")
        connection.execute("DELETE FROM human_decisions WHERE review_id = ?", (review.review_id,))
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as excinfo:
        mock_payment(
            duplicate_case,
            duplicate_invoice,
            store,
            settings.workflow_db,
            duplicate_claim,
        )

    assert excinfo.value.stop_reason == "PAYMENT_LEDGER_INCONSISTENT"


def test_migration_003_rolls_back_and_is_retryable_after_mid_migration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import invoice_agents.db.core as core_module

    path = tmp_path / "workflow-atomic.db"
    real_connect = core_module.connect_database
    armed = True

    class FailingConnection(_ConnectionProxy):
        def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
            nonlocal armed
            if armed and "CREATE INDEX idx_cases_execution_lease" in sql:
                armed = False
                raise sqlite3.OperationalError("injected migration interruption")
            return self._connection.execute(sql, parameters)

    @contextmanager
    def failing_connect(target: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
        with real_connect(target, read_only=read_only) as connection:
            yield FailingConnection(
                connection,
                before_execute=lambda _sql: None,
                after_execute=lambda _sql: None,
            )  # type: ignore[misc]

    monkeypatch.setattr(core_module, "connect_database", failing_connect)
    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW)
    assert excinfo.value.stop_reason == "MIGRATION_FAILED"
    with real_connect(path, read_only=True) as connection:
        columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(cases)")}
        versions = [
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_version ORDER BY version")
        ]
    assert "execution_token" not in columns
    assert versions == [1, 2]

    legacy_evaluated_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC).isoformat()
    with real_connect(path) as connection:
        connection.execute(
            "INSERT INTO source_artifacts(source_id, canonical_path, source_hash, "
            "source_format, size_bytes, modified_at, metadata_json, created_at) "
            "VALUES ('src_v2', '/tmp/v2.txt', ?, 'txt', 1, ?, '{}', ?)",
            ("a" * 64, legacy_evaluated_at, legacy_evaluated_at),
        )
        connection.execute(
            "INSERT INTO cases(case_id, source_id, status, started_at, updated_at) "
            "VALUES ('case_v2', 'src_v2', 'INCOMPLETE', ?, ?)",
            (legacy_evaluated_at, legacy_evaluated_at),
        )
        connection.execute(
            "INSERT INTO identity_results(identity_id, case_id, payload_json, created_at) "
            "VALUES ('ident_v2', 'case_v2', '[]', ?)",
            (legacy_evaluated_at,),
        )
        connection.commit()

    assert migrate_database(path, DatabaseKind.WORKFLOW) == [3]
    assert verify_database(path, DatabaseKind.WORKFLOW)["schema_version"] == 3
    with real_connect(path, read_only=True) as connection:
        identity_columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(identity_results)")
        }
        final_columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(final_decisions)")
        }
        payment_columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(payments)")
        }
        trigger_names = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
        backfilled_at = connection.execute(
            "SELECT evaluated_at FROM identity_results WHERE identity_id = 'ident_v2'"
        ).fetchone()["evaluated_at"]
    assert "evaluated_at" in identity_columns
    assert {
        "source_id",
        "invoice_number",
        "vendor",
        "authorized_amount",
        "authorized_currency",
        "payment_idempotency_key",
        "review_id",
    } <= final_columns
    assert {"source_id", "invoice_number", "review_id"} <= payment_columns
    assert {
        "trg_final_decisions_authorization_insert",
        "trg_final_decisions_immutable_update",
        "trg_final_decisions_immutable_delete",
        "trg_payments_authorization_insert",
        "trg_payments_immutable_update",
        "trg_payments_immutable_delete",
    } <= trigger_names
    assert backfilled_at == legacy_evaluated_at


@pytest.mark.asyncio
async def test_lease_heartbeat_propagates_renewal_failure_without_masking() -> None:
    from invoice_agents.orchestration import _run_with_lease_heartbeat

    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()
    renew_attempted = asyncio.Event()
    claim = ExecutionClaim(
        case_id="case_heartbeat",
        token="exec_heartbeat",
        generation=1,
        expires_at=datetime.now(UTC),
    )

    async def operation() -> str:
        operation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()
        return "unreachable"

    def fail_renewal(_claim: ExecutionClaim, _lease_seconds: int) -> ExecutionClaim:
        renew_attempted.set()
        raise InvoiceAgentsError(
            category=orchestration.ErrorCategory.ORCHESTRATION,
            message="lease was taken over",
            case_id=claim.case_id,
            stop_reason="STALE_EXECUTION_CLAIM",
        )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        await _run_with_lease_heartbeat(
            operation(),
            renew=fail_renewal,
            claim=claim,
            lease_seconds=60,
            renewal_interval_seconds=0.001,
        )
    assert operation_started.is_set()
    assert renew_attempted.is_set()
    assert operation_cancelled.is_set()
    assert excinfo.value.stop_reason == "STALE_EXECUTION_CLAIM"


def test_claim_and_renew_compute_authoritative_time_after_write_lock(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = "case_lock_time"
    _persist_case(settings, case_id)
    real_connect = store_module.connect_database
    begin_attempted = threading.Event()
    initial = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    after_claim_lock = initial + timedelta(minutes=5)
    after_renew_lock = after_claim_lock + timedelta(minutes=5)

    class ControlledDatetime(datetime):
        current = initial

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return cls.current

    @contextmanager
    def observed_connect(path: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
        with real_connect(path, read_only=read_only) as connection:

            def before_execute(sql: str) -> None:
                if sql.strip().upper() == "BEGIN IMMEDIATE":
                    begin_attempted.set()

            yield _ConnectionProxy(
                connection,
                before_execute=before_execute,
                after_execute=lambda _sql: None,
            )  # type: ignore[misc]

    monkeypatch.setattr(store_module, "datetime", ControlledDatetime)
    monkeypatch.setattr(store_module, "connect_database", observed_connect)
    claim_outcome: list[ExecutionClaim | BaseException] = []
    with real_connect(settings.workflow_db) as blocker:
        blocker.execute("BEGIN IMMEDIATE")

        def claim_case() -> None:
            try:
                claim_outcome.append(
                    WorkflowStore(settings.workflow_db).claim_case_execution(
                        case_id,
                        frozenset({CaseStatus.INCOMPLETE}),
                        lease_seconds=60,
                    )
                )
            except BaseException as exc:
                claim_outcome.append(exc)

        worker = threading.Thread(target=claim_case)
        worker.start()
        assert begin_attempted.wait(timeout=5)
        ControlledDatetime.current = after_claim_lock
        blocker.commit()
        worker.join(timeout=10)
    assert len(claim_outcome) == 1
    assert isinstance(claim_outcome[0], ExecutionClaim)
    claim = claim_outcome[0]
    assert claim.expires_at == after_claim_lock + timedelta(seconds=60)

    begin_attempted.clear()
    renew_outcome: list[ExecutionClaim | BaseException] = []
    with real_connect(settings.workflow_db) as blocker:
        blocker.execute("BEGIN IMMEDIATE")

        def renew_case() -> None:
            try:
                renew_outcome.append(
                    WorkflowStore(settings.workflow_db).renew_case_execution(
                        claim, lease_seconds=120
                    )
                )
            except BaseException as exc:
                renew_outcome.append(exc)

        worker = threading.Thread(target=renew_case)
        worker.start()
        assert begin_attempted.wait(timeout=5)
        ControlledDatetime.current = after_renew_lock
        blocker.commit()
        worker.join(timeout=10)
    assert len(renew_outcome) == 1
    assert isinstance(renew_outcome[0], InvoiceAgentsError)
    assert renew_outcome[0].stop_reason == "STALE_EXECUTION_CLAIM"
