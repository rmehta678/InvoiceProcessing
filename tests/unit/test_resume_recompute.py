"""G6 deterministic halves: recompute after a human mapping resolves or re-blocks a case."""

from decimal import Decimal
from pathlib import Path

import pytest

from invoice_agents import orchestration
from invoice_agents.agents.decision_rules import blocking_evidence, validate_final_decision
from invoice_agents.config import Settings
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.hitl.service import create_review_request, record_human_decision
from invoice_agents.models import (
    CanonicalMapping,
    CaseStatus,
    Critique,
    DecisionKind,
    HumanDecisionKind,
    InventoryStatus,
    ReviewRequest,
    RiskAssessment,
)
from invoice_agents.observability.audit import AuditRecorder
from invoice_agents.tools.comparison import (
    InventoryReader,
    apply_mapping_evidence,
    build_risk_assessment,
    compare_inventory_evidence,
    compute_invoice_totals,
    find_prior_invoice_candidates,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def make_critique(disposition: DecisionKind = DecisionKind.HOLD) -> Critique:
    return Critique(
        supported_findings=["deterministic evidence reviewed"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=disposition,
        rationale=["mapping ambiguity requires human review"],
    )


def prepare(path: Path, settings: Settings) -> str:
    prepared = orchestration.prepare_case(path, settings)
    assert isinstance(prepared, tuple), f"prepare_case failed: {prepared}"
    return prepared[0]


def first_pass_review(
    case_id: str, settings: Settings, store: WorkflowStore
) -> tuple[RiskAssessment, ReviewRequest, ExecutionClaim]:
    """Run the deterministic pass-1 evidence chain and persist it like the team does."""

    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    invoice = store.promote_predecessor_extraction(claim)
    mappings, comparisons, unresolved = compare_inventory_evidence(
        invoice, InventoryReader(settings.inventory_db)
    )
    invoice = apply_mapping_evidence(invoice, mappings, unresolved)
    store.save_extraction(case_id, invoice, claim)
    identity = find_prior_invoice_candidates(case_id, invoice, store)
    risk = build_risk_assessment(
        invoice, comparisons, identity, compute_invoice_totals(invoice), settings
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
    store.save_identity(
        case_id,
        [candidate.model_dump(mode="json") for candidate in identity],
        claim,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    case_critique = make_critique()
    store.save_critique(case_id, case_critique, claim)
    review = create_review_request(
        case_id,
        invoice,
        risk,
        case_critique,
        DecisionKind.HOLD,
        ["ambiguous mapping"],
        store,
        claim,
        pdf_policy=settings.pdf_policy(),
    )
    return risk, review, claim


def establish_mapping_and_recompute(
    case_id: str,
    review: ReviewRequest,
    raw_item: str,
    reason: str,
    settings: Settings,
    store: WorkflowStore,
    claim: ExecutionClaim,
) -> ReviewRequest:
    resolved = record_human_decision(
        review.review_id,
        "test-reviewer",
        HumanDecisionKind.ESTABLISH_MAPPING,
        reason,
        store,
        settings.inventory_db,
        mappings=[CanonicalMapping(raw_item=raw_item, sku="SKU-WIDGET-A", basis="human_decision")],
    )
    assert resolved.human_decision is not None
    orchestration._recompute_after_mapping(
        case_id,
        resolved.human_decision,
        settings,
        store,
        AuditRecorder(settings.workflow_db, case_id),
        claim,
    )
    return resolved


def latest_risk(store: WorkflowStore, case_id: str) -> RiskAssessment:
    return RiskAssessment.model_validate(store.load_comparison(case_id, "risk"))


def test_mapping_recompute_resolves_blocker_and_authorizes_approve(
    invoice_dir: Path, settings: Settings
) -> None:
    case_id = prepare(invoice_dir / "invoice_1010.txt", settings)
    store = WorkflowStore(settings)
    first_risk, review, claim = first_pass_review(case_id, settings, store)
    assert any(item.status is InventoryStatus.AMBIGUOUS for item in first_risk.inventory)
    resolved = establish_mapping_and_recompute(
        case_id,
        review,
        "WidgetA (rush order)",
        "maps rush order",
        settings,
        store,
        claim,
    )
    risk = latest_risk(store, case_id)
    assert all(item.status is not InventoryStatus.AMBIGUOUS for item in risk.inventory)
    widget_a = next(item for item in risk.inventory if item.sku == "SKU-WIDGET-A")
    assert widget_a.requested_quantity == Decimal("12")
    assert widget_a.status is InventoryStatus.AVAILABLE
    assert blocking_evidence(risk) == []
    rush_line = next(
        line
        for line in store.load_extraction(case_id).lines
        if line.raw_item == "WidgetA (rush order)"
    )
    assert rush_line.canonical_sku == "SKU-WIDGET-A"
    assert (
        validate_final_decision(
            DecisionKind.APPROVE, True, risk, make_critique(DecisionKind.HOLD), resolved
        )
        is None
    )
    assert store.count_events(case_id, "recompute.after_human_mapping") == 1
    store.release_case_execution(claim)


def test_mapping_recompute_with_exceeding_stock_stays_blocked_and_needs_second_review(
    settings: Settings,
) -> None:
    case_id = prepare(FIXTURE_DIR / "invoice_2001_bulk_alias.txt", settings)
    store = WorkflowStore(settings)
    first_risk, review, claim = first_pass_review(case_id, settings, store)
    bulk = next(item for item in first_risk.inventory if item.raw_items == ["WidgetA (bulk)"])
    assert bulk.status is InventoryStatus.AMBIGUOUS
    resolved = establish_mapping_and_recompute(
        case_id,
        review,
        "WidgetA (bulk)",
        "maps bulk alias",
        settings,
        store,
        claim,
    )
    risk = latest_risk(store, case_id)
    widget_a = next(item for item in risk.inventory if item.sku == "SKU-WIDGET-A")
    assert widget_a.requested_quantity == Decimal("20")
    assert widget_a.status is InventoryStatus.EXCEEDS_STOCK
    assert blocking_evidence(risk)
    assert (
        validate_final_decision(
            DecisionKind.HOLD, False, risk, make_critique(DecisionKind.HOLD), resolved
        )
        is None
    )
    with pytest.raises(InvoiceAgentsError) as approval_error:
        validate_final_decision(
            DecisionKind.APPROVE, True, risk, make_critique(DecisionKind.APPROVE), resolved
        )
    assert approval_error.value.stop_reason == "BLOCKING_EVIDENCE_UNRESOLVED"
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.REJECT, False, risk, make_critique(DecisionKind.HOLD), resolved
        )
    assert excinfo.value.stop_reason == "HUMAN_AGENT_DECISION_CONFLICT"
    second = create_review_request(
        case_id,
        store.load_extraction(case_id),
        risk,
        make_critique(),
        DecisionKind.HOLD,
        ["blocking evidence remains after mapping"],
        store,
        claim,
        pdf_policy=settings.pdf_policy(),
    )
    assert second.sequence == 2
    latest = store.load_case_review(case_id)
    assert latest is not None
    assert latest.review_id == second.review_id
    assert latest.sequence == 2
    assert latest.status == "PENDING"
    store.release_case_execution(claim)
