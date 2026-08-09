"""Pure final-decision rules; every violation raises and nothing defaults to approval."""

from __future__ import annotations

from decimal import Decimal

from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import (
    Critique,
    DecisionKind,
    EvidenceBlocker,
    HumanDecision,
    HumanDecisionKind,
    InventoryStatus,
    ReviewRequest,
    RiskAssessment,
)
from invoice_agents.tools.comparison import normalize_alias

AUTHORIZING_HUMAN_DECISIONS = frozenset(
    {
        HumanDecisionKind.APPROVE,
        HumanDecisionKind.ESTABLISH_MAPPING,
        HumanDecisionKind.SUPERSEDE_REVISION,
    }
)

# Evidence an authorizing decision does not implicitly clear: payment against
# unavailable/unknown stock or a conflicting declared total always needs its own ruling.
BLOCKING_INVENTORY_STATUSES = frozenset(
    {
        InventoryStatus.EXCEEDS_STOCK,
        InventoryStatus.OUT_OF_STOCK,
        InventoryStatus.UNKNOWN,
        InventoryStatus.INVALID_QUANTITY,
        InventoryStatus.ERROR,
    }
)


def _inventory_evidence_id(sku: str | None, raw_items: list[str]) -> str:
    """Return an order-independent domain identity, never a display position or prose."""

    if sku and sku.strip():
        return sku.strip()
    normalized = sorted({value for raw in raw_items if (value := normalize_alias(raw))})
    return "+".join(normalized) if normalized else "unidentified"


def blocking_evidence(risk: RiskAssessment) -> list[EvidenceBlocker]:
    """Build typed blockers whose IDs depend only on stable domain material."""

    blockers: list[EvidenceBlocker] = []
    for item in risk.inventory:
        if item.status not in BLOCKING_INVENTORY_STATUSES:
            continue
        evidence_id = _inventory_evidence_id(item.sku, item.raw_items)
        blockers.append(
            EvidenceBlocker(
                blocker_id=f"inventory:{evidence_id}:{item.status.value}",
                kind="inventory",
                evidence_id=evidence_id,
                description=(
                    f"inventory {item.status.value}: {', '.join(item.raw_items)} "
                    f"requested={item.requested_quantity} stock={item.available_stock}"
                ),
            )
        )
    total_delta = risk.financial.total_delta
    if total_delta is not None and total_delta != Decimal("0"):
        blockers.append(
            EvidenceBlocker(
                blocker_id="financial:declared-total-delta",
                kind="financial",
                evidence_id="declared-total-delta",
                description=f"declared/calculated total delta is {total_delta}",
            )
        )
    return blockers


def unaddressed_blockers(
    risk: RiskAssessment, human: HumanDecision | None
) -> list[EvidenceBlocker]:
    """Return current blockers unless one authorizing ruling names the exact current set.

    Exact-set comparison is deliberate: an unknown, stale, or extra ID invalidates the
    authorization rather than silently widening a prior human ruling to changed evidence.
    """

    current = blocking_evidence(risk)
    if not current or human is None or human.decision not in AUTHORIZING_HUMAN_DECISIONS:
        return current
    current_ids = {blocker.blocker_id for blocker in current}
    addressed_ids = set(human.addressed_blocker_ids)
    return [] if addressed_ids == current_ids else current


def assert_new_review_cycle_permitted(
    latest: ReviewRequest | None, case_id: str | None = None
) -> None:
    """Refuse to open a review cycle over a final (non-authorizing) human ruling.

    REJECT and REQUEST_CORRECTION already rule on every listed blocker; the only
    lawful continuation is the forced final decision, so re-escalation is refused
    loudly instead of looping the queue. Authorizing decisions with remaining
    blocking evidence stay eligible for a further cycle (remediation §3.5).
    """

    if latest is None or latest.status != "RESOLVED" or latest.human_decision is None:
        return
    decision = latest.human_decision.decision
    if decision not in AUTHORIZING_HUMAN_DECISIONS:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            f"human decision {decision} on review {latest.review_id} is final; submit the "
            "matching final decision instead of requesting another review",
            case_id=case_id,
            stop_reason="HUMAN_DECISION_MUST_BE_OBEYED",
        )


def validate_final_decision(
    selected: DecisionKind,
    payment_eligible: bool,
    risk: RiskAssessment,
    critique: Critique,
    review: ReviewRequest | None,
    case_id: str | None = None,
) -> None:
    """Raise for every rule violation; returning means the decision may be persisted.

    Rules, in order:
    1. Policy/ambiguity triggers require a resolved human review.
    2. A recorded human decision constrains the agent: REJECT forces REJECT,
       REQUEST_CORRECTION forces HOLD, and an authorizing decision permits APPROVE -
       or HOLD when blocking evidence the decision did not address remains. It never
       forces APPROVE against remaining blocking evidence, and REJECT after an
       authorizing decision stays a conflict.
    3. APPROVE additionally requires the independent critic's APPROVE or a resolved
       authorizing human decision; an unresolved disagreement stops the case.
    4. APPROVE and payment eligibility must agree exactly.
    """

    if risk.policy_review_reasons and (review is None or review.status != "RESOLVED"):
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "policy/ambiguity triggers require resolved human review before final decision",
            case_id=case_id,
            stop_reason="HUMAN_REVIEW_UNRESOLVED",
        )
    human = review.human_decision if review is not None and review.status == "RESOLVED" else None
    remaining = unaddressed_blockers(risk, human)
    if selected is DecisionKind.APPROVE and remaining:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "APPROVE requires explicit human authorization for every current blocking "
            f"evidence ID: {[blocker.blocker_id for blocker in remaining]}",
            case_id=case_id,
            stop_reason="BLOCKING_EVIDENCE_UNRESOLVED",
        )
    if human is not None:
        if human.decision in AUTHORIZING_HUMAN_DECISIONS:
            hold_permitted = selected is DecisionKind.HOLD and remaining
            if selected is not DecisionKind.APPROVE and not hold_permitted:
                raise InvoiceAgentsError(
                    ErrorCategory.TOOL,
                    "agent final decision conflicts with authorizing human decision",
                    case_id=case_id,
                    stop_reason="HUMAN_AGENT_DECISION_CONFLICT",
                )
        if human.decision is HumanDecisionKind.REJECT and selected is not DecisionKind.REJECT:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                "agent final decision conflicts with human rejection",
                case_id=case_id,
                stop_reason="HUMAN_AGENT_DECISION_CONFLICT",
            )
        if (
            human.decision is HumanDecisionKind.REQUEST_CORRECTION
            and selected is not DecisionKind.HOLD
        ):
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                "request-correction human decision requires HOLD",
                case_id=case_id,
                stop_reason="HUMAN_AGENT_DECISION_CONFLICT",
            )
    if selected is DecisionKind.APPROVE and critique.recommended_disposition is not (
        DecisionKind.APPROVE
    ):
        authorized = human is not None and human.decision in AUTHORIZING_HUMAN_DECISIONS
        if not authorized:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                f"critic recommends {critique.recommended_disposition} and no human decision "
                "authorizes APPROVE; the disagreement requires human review",
                case_id=case_id,
                stop_reason="CRITIC_DISAGREEMENT_UNRESOLVED",
            )
    if selected is DecisionKind.APPROVE and not payment_eligible:
        raise InvoiceAgentsError(
            ErrorCategory.SCHEMA,
            "APPROVE must set payment_eligible=true",
            case_id=case_id,
            stop_reason="FINAL_DECISION_INVALID",
        )
    if selected is not DecisionKind.APPROVE and payment_eligible:
        raise InvoiceAgentsError(
            ErrorCategory.SCHEMA,
            "non-APPROVE decision cannot be payment eligible",
            case_id=case_id,
            stop_reason="FINAL_DECISION_INVALID",
        )
