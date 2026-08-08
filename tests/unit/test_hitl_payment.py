"""Persisted review decisions and payment idempotency."""

from datetime import UTC, datetime
from pathlib import Path

from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import WorkflowStore
from invoice_agents.hitl.service import create_review_request, record_human_decision
from invoice_agents.models import (
    Critique,
    DecisionKind,
    FinalDecision,
    HumanDecisionKind,
    PaymentStatus,
)
from invoice_agents.payment.service import mock_payment
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.comparison import (
    InventoryReader,
    build_risk_assessment,
    compare_inventory,
    compute_invoice_totals,
)
from invoice_agents.tools.evidence import extract_invoice_evidence


def load(invoice_dir: Path, name: str, archive: Path):  # type: ignore[no-untyped-def]
    source = snapshot_source(invoice_dir / name, archive, max_bytes=10_485_760)
    return extract_invoice_evidence(source)


def persist_case(store: WorkflowStore, case_id: str, invoice) -> None:  # type: ignore[no-untyped-def]
    store.register_source(invoice.source)
    store.create_case(case_id, invoice.source, datetime.now(UTC))
    store.save_extraction(case_id, invoice)


def critique(disposition: DecisionKind = DecisionKind.HOLD) -> Critique:
    return Critique(
        supported_findings=["deterministic evidence reviewed"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=disposition,
        rationale=["review policy applies"],
    )


def approve_final(store: WorkflowStore, case_id: str) -> None:
    store.save_final_decision(
        case_id,
        FinalDecision(
            decision=DecisionKind.APPROVE,
            reasons=["approved evidence"],
            critic_disposition=DecisionKind.APPROVE,
            payment_eligible=True,
        ),
    )


def test_review_request_and_human_decision_are_persisted(
    invoice_dir: Path,
    inventory_db: Path,
    workflow_db: Path,
    settings: Settings,
) -> None:
    store = WorkflowStore(workflow_db)
    inv = load(invoice_dir, "invoice_1002.txt", workflow_db.parent / "sources")
    persist_case(store, "case_review", inv)
    comparisons, _ = compare_inventory(inv, InventoryReader(inventory_db))
    risk = build_risk_assessment(inv, comparisons, [], compute_invoice_totals(inv), settings)
    review = create_review_request(
        "case_review", inv, risk, critique(), DecisionKind.HOLD, ["stock exceeds"], store
    )
    assert review.status == "PENDING"
    resolved = record_human_decision(
        review.review_id,
        "vp@example.com",
        HumanDecisionKind.REJECT,
        "quantity is not authorized",
        store,
        inventory_db,
    )
    assert resolved.status == "RESOLVED"
    assert resolved.human_decision is not None
    assert resolved.human_decision.reviewer == "vp@example.com"
    assert resolved.agent_recommendation is DecisionKind.HOLD


def test_mock_payment_pays_once_across_duplicate_representations(
    invoice_dir: Path, workflow_db: Path
) -> None:
    store = WorkflowStore(workflow_db)
    text = load(invoice_dir, "invoice_1011.txt", workflow_db.parent / "sources")
    pdf = load(invoice_dir, "invoice_1011.pdf", workflow_db.parent / "sources")
    persist_case(store, "case_text", text)
    approve_final(store, "case_text")
    first = mock_payment("case_text", text, store, workflow_db)
    assert first.status is PaymentStatus.PAID
    persist_case(store, "case_pdf", pdf)
    approve_final(store, "case_pdf")
    duplicate = mock_payment("case_pdf", pdf, store, workflow_db)
    assert duplicate.status is PaymentStatus.DUPLICATE
    assert duplicate.case_id == "case_pdf"
    assert duplicate.duplicate_of == first.payment_id
    with connect_database(workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 1


def test_rejected_and_injected_failure_never_report_payment_success(
    invoice_dir: Path, workflow_db: Path
) -> None:
    store = WorkflowStore(workflow_db)
    rejected_invoice = load(invoice_dir, "invoice_1001.txt", workflow_db.parent / "sources")
    persist_case(store, "case_rejected", rejected_invoice)
    store.save_final_decision(
        "case_rejected",
        FinalDecision(
            decision=DecisionKind.REJECT,
            reasons=["rejected"],
            critic_disposition=DecisionKind.REJECT,
            payment_eligible=False,
        ),
    )
    rejected = mock_payment("case_rejected", rejected_invoice, store, workflow_db)
    assert rejected.status is PaymentStatus.NOT_ELIGIBLE

    failed_invoice = load(invoice_dir, "invoice_1004.json", workflow_db.parent / "sources")
    persist_case(store, "case_failed_payment", failed_invoice)
    approve_final(store, "case_failed_payment")
    failed = mock_payment(
        "case_failed_payment", failed_invoice, store, workflow_db, inject_failure=True
    )
    assert failed.status is PaymentStatus.FAILED
    assert failed.error == "injected mock-payment failure"
