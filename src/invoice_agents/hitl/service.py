"""Human-review workflow that persists stopped state instead of blocking a live run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import (
    CanonicalMapping,
    Critique,
    DecisionKind,
    ExtractedInvoice,
    HumanDecision,
    HumanDecisionKind,
    Money,
    ReviewRequest,
    RiskAssessment,
    SourceArtifact,
)
from invoice_agents.source_store import verified_source_path
from invoice_agents.tools.evidence import render_pdf_page


def _render_review_pages(source: SourceArtifact, review_id: str) -> list[dict[str, object]]:
    """Render original-layout PDF evidence for the review package (§9).

    Page 1 always renders; documents of up to three pages render completely. A render
    failure raises SourceEvidenceError - the review is not created without the
    evidence it promises.
    """

    if source.source_format != "pdf":
        return []
    verified_source_path(source)
    page_count = source.page_count or 1
    pages = range(1, page_count + 1) if page_count <= 3 else range(1, 2)
    output_dir = Path("artifacts/reviews").resolve() / review_id
    return [render_pdf_page(source, page, output_dir) for page in pages]


def create_review_request(
    case_id: str,
    invoice: ExtractedInvoice,
    risk: RiskAssessment,
    critique: Critique,
    agent_recommendation: DecisionKind,
    agent_rationale: list[str],
    store: WorkflowStore,
    claim: ExecutionClaim,
    *,
    extra_reasons: list[str] | None = None,
) -> ReviewRequest:
    """Persist a complete evidence package for every policy or ambiguity trigger.

    extra_reasons carries deterministic non-policy triggers (today: the recorded
    critic/agent disposition disagreement); at least one reason must exist overall.
    """

    from invoice_agents.agents.decision_rules import blocking_evidence

    reasons = list(dict.fromkeys([*risk.policy_review_reasons, *(extra_reasons or [])]))
    if not reasons:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "review request requires at least one evidence-backed review reason",
            case_id=case_id,
            stop_reason="REVIEW_REASON_MISSING",
        )
    amount = None
    if invoice.declared_total is not None and invoice.currency.normalized_value:
        amount = Money(
            amount=invoice.declared_total,
            currency=invoice.currency.normalized_value,
        )
    questions = [
        "Do the source evidence, normalized values, and calculated deltas support payment?",
        "Resolve every listed inventory, identity, date, currency, and field ambiguity.",
        "If establishing an alias, identify the exact SKU and explain the evidence for the mapping.",
        "If this is a revision, identify which prior case is superseded and whether it was paid.",
    ]
    review_id = f"rev_{uuid4().hex}"
    review = ReviewRequest(
        review_id=review_id,
        case_id=case_id,
        status="PENDING",
        reasons=reasons,
        amount=amount,
        source=invoice.source,
        evidence_bundle={
            "invoice": invoice.model_dump(mode="json"),
            "financial": risk.financial.model_dump(mode="json"),
            "inventory": [item.model_dump(mode="json") for item in risk.inventory],
            "identity_candidates": [
                item.model_dump(mode="json") for item in risk.identity_candidates
            ],
            "dates": [item.model_dump(mode="json") for item in risk.dates],
            "suspicious_signals": risk.suspicious_signals,
            "unavailable_reconciliations": risk.unavailable_reconciliations,
            "blocking_evidence": [
                blocker.model_dump(mode="json") for blocker in blocking_evidence(risk)
            ],
            "rendered_pages": _render_review_pages(invoice.source, review_id),
        },
        agent_recommendation=agent_recommendation,
        agent_rationale=agent_rationale,
        critic=critique,
        questions=questions,
        created_at=datetime.now(UTC),
    )
    return store.save_review(review, claim)


def record_human_decision(
    review_id: str,
    reviewer: str,
    decision: HumanDecisionKind,
    reason: str,
    store: WorkflowStore,
    inventory_db: Path,
    *,
    mappings: list[CanonicalMapping] | None = None,
    superseded_case_id: str | None = None,
    addressed_blocker_ids: list[str] | None = None,
) -> ReviewRequest:
    """Record one evidence-bound decision and any aliases in a single SQLite commit."""

    selected_blocker_ids = list(
        dict.fromkeys(
            blocker_id.strip() for blocker_id in addressed_blocker_ids or [] if blocker_id.strip()
        )
    )
    selected_mappings = [
        mapping.model_copy(
            update={
                "raw_item": mapping.raw_item.strip(),
                "sku": mapping.sku.strip(),
                "basis": "human_decision",
            },
            deep=True,
        )
        for mapping in mappings or []
    ]
    human = (
        HumanDecision(
            review_id=review_id,
            reviewer=reviewer.strip(),
            decision=decision,
            reason=reason.strip(),
            decided_at=datetime.now(UTC),
            mappings=selected_mappings,
            superseded_case_id=(superseded_case_id or "").strip() or None,
            addressed_blocker_ids=selected_blocker_ids,
        )
        if reviewer.strip() and reason.strip()
        else None
    )
    replay = store.classify_human_decision_replay(review_id, human)
    if replay is not None:
        return replay
    if human is None:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "reviewer and reason are required",
            stop_reason="HUMAN_DECISION_INVALID",
        )
    return store.save_human_decision(human, inventory_db)
