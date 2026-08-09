"""Fabricate persisted cases through the same store/service calls the pipeline uses.

Pages then render exactly what a real run would have stored. No model calls;
the only replaced boundary in route tests is run_prepared_case / resume_case.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.hitl.service import create_review_request
from invoice_agents.models import (
    CaseResult,
    CaseStatus,
    Critique,
    DecisionKind,
    ErrorRecord,
    FinalDecision,
    ReviewRequest,
    RiskAssessment,
    UsageSummary,
)
from invoice_agents.orchestration import prepare_case
from invoice_agents.payment.service import mock_payment
from invoice_agents.tools.comparison import (
    InventoryReader,
    build_risk_assessment,
    compare_inventory_evidence,
    compute_invoice_totals,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "invoices"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def make_critique(disposition: DecisionKind) -> Critique:
    return Critique(
        supported_findings=["deterministic evidence reviewed"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=disposition,
        rationale=["fixture critique for template rendering"],
    )


def _now() -> datetime:
    return datetime.now(UTC)


def prepare_fixture_case(settings: Settings, source: Path) -> tuple[str, datetime]:
    prepared = prepare_case(source, settings)
    assert isinstance(prepared, tuple), f"prepare_case failed: {prepared}"
    return prepared


def record_case_evidence(
    settings: Settings, case_id: str, critic_disposition: DecisionKind
) -> RiskAssessment:
    """Persist inventory/risk/critique evidence exactly as the agent tools do."""

    store = WorkflowStore(settings.workflow_db)
    invoice = store.load_extraction(case_id)
    _mappings, comparisons, unresolved = compare_inventory_evidence(
        invoice, InventoryReader(settings.inventory_db)
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
    )
    risk = build_risk_assessment(
        invoice, comparisons, [], compute_invoice_totals(invoice), settings
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"))
    store.save_critique(case_id, make_critique(critic_disposition))
    return risk


def make_succeeded_case(settings: Settings, name: str = "invoice_1001.txt") -> str:
    """A fully persisted approved case paid through the real mock-payment service."""

    case_id, started_at = prepare_fixture_case(settings, DATA_DIR / name)
    store = WorkflowStore(settings.workflow_db)
    invoice = store.load_extraction(case_id)
    record_case_evidence(settings, case_id, DecisionKind.APPROVE)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    decision = FinalDecision(
        decision=DecisionKind.APPROVE,
        reasons=["all deterministic checks passed"],
        critic_disposition=DecisionKind.APPROVE,
        payment_eligible=True,
    )
    store.save_final_decision(case_id, decision, claim)
    payment = mock_payment(case_id, invoice, store, settings.workflow_db, claim)
    result = CaseResult(
        case_id=case_id,
        source_id=invoice.source.source_id,
        status=CaseStatus.SUCCEEDED,
        stop_reason="APPROVED_PAYMENT_RECORDED",
        final_decision=decision,
        payment=payment,
        usage=UsageSummary(prompt_tokens=1000, completion_tokens=100, model_calls=5),
        started_at=started_at,
        finished_at=_now(),
    )
    store.finish_case(result, claim)
    return case_id


def make_pending_review_case(
    settings: Settings, source: Path | None = None
) -> tuple[str, ReviewRequest]:
    """A NEEDS_HUMAN case with a persisted pending review package and team state."""

    case_id, started_at = prepare_fixture_case(settings, source or (DATA_DIR / "invoice_1002.txt"))
    store = WorkflowStore(settings.workflow_db)
    invoice = store.load_extraction(case_id)
    risk = record_case_evidence(settings, case_id, DecisionKind.HOLD)
    assert risk.policy_review_reasons, "fixture invoice must trigger review policy"
    review = create_review_request(
        case_id,
        invoice,
        risk,
        make_critique(DecisionKind.HOLD),
        DecisionKind.HOLD,
        ["policy triggers require human review"],
        store,
    )
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.save_team_state(case_id, {"fixture": "stopped-team-state"}, claim)
    result = CaseResult(
        case_id=case_id,
        source_id=invoice.source.source_id,
        status=CaseStatus.NEEDS_HUMAN,
        stop_reason="HUMAN_REVIEW_REQUESTED",
        review_request=review,
        started_at=started_at,
        finished_at=_now(),
    )
    store.finish_case(result, claim)
    return case_id, review


def make_failed_case(settings: Settings, name: str = "invoice_1001.txt") -> str:
    """A FAILED case whose error records must render at the top of the page."""

    case_id, started_at = prepare_fixture_case(settings, DATA_DIR / name)
    store = WorkflowStore(settings.workflow_db)
    invoice = store.load_extraction(case_id)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    result = CaseResult(
        case_id=case_id,
        source_id=invoice.source.source_id,
        status=CaseStatus.FAILED,
        stop_reason="PROVIDER_TIMEOUT",
        errors=[
            ErrorRecord(
                category="TIMEOUT",
                message="provider request exceeded the configured timeout",
                case_id=case_id,
                stop_reason="PROVIDER_TIMEOUT",
                provider_request_id="req_fixture_123",
            )
        ],
        started_at=started_at,
        finished_at=_now(),
    )
    store.finish_case(result, claim)
    return case_id
