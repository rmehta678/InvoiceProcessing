"""Pure final-decision rules; every violation raises and nothing defaults to approval."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from invoice_agents.db.store import WorkflowStore

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

MappingProvenance = tuple[str, str, str, str]


def validate_human_decision_applicability(
    review: ReviewRequest,
    decision: HumanDecision,
    *,
    inventory_skus: frozenset[str],
    valid_superseded_case_ids: frozenset[str],
    persisted_mapping_provenance: Mapping[str, MappingProvenance] | None,
) -> tuple[tuple[str, str], ...]:
    """Purely validate one ruling against explicit authoritative review facts.

    Callers obtain inventory existence, supersession identity, and (for replay or
    audit) persisted alias provenance transactionally.  This function owns every
    decision-kind field rule so write, replay, and preflight cannot drift apart.
    """

    if decision.review_id != review.review_id:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "human decision does not identify its review",
            case_id=review.case_id,
            stop_reason="HUMAN_DECISION_INVALID",
        )

    raw_blockers = review.evidence_bundle.get("blocking_evidence")
    if not isinstance(raw_blockers, list):
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "review blocking evidence is not an exact list",
            case_id=review.case_id,
            stop_reason="BLOCKER_AUTHORIZATION_INVALID",
        )
    package_blocker_ids = [
        entry.get("blocker_id") if isinstance(entry, dict) else None for entry in raw_blockers
    ]
    addressed = decision.addressed_blocker_ids
    package_set = {blocker_id for blocker_id in package_blocker_ids if isinstance(blocker_id, str)}
    addressed_set = set(addressed)
    blockers_are_exact = (
        all(
            isinstance(blocker_id, str) and blocker_id.strip() == blocker_id
            for blocker_id in package_blocker_ids
        )
        and len(package_set) == len(package_blocker_ids)
        and all(blocker_id and blocker_id.strip() == blocker_id for blocker_id in addressed)
        and len(addressed_set) == len(addressed)
        and addressed_set.issubset(package_set)
    )
    authorizing = decision.decision in AUTHORIZING_HUMAN_DECISIONS
    # Empty means "this ruling does not authorize any blocker" and remains valid;
    # once IDs are supplied, only the complete exact set carries authorization.
    exact_set_required = bool(addressed)
    if (
        not blockers_are_exact
        or (addressed and not authorizing)
        or (exact_set_required and addressed_set != package_set)
    ):
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "blocker authorization must use one exact review-package blocker-ID set",
            case_id=review.case_id,
            stop_reason="BLOCKER_AUTHORIZATION_INVALID",
        )

    mappings = decision.mappings
    if mappings and decision.decision is not HumanDecisionKind.ESTABLISH_MAPPING:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "mappings are permitted only for ESTABLISH_MAPPING",
            case_id=review.case_id,
            stop_reason="HUMAN_MAPPING_INVALID",
        )
    if decision.decision is HumanDecisionKind.ESTABLISH_MAPPING and not mappings:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "ESTABLISH_MAPPING requires at least one explicit mapping",
            case_id=review.case_id,
            stop_reason="HUMAN_MAPPING_MISSING",
        )

    unresolved_aliases: set[str] = set()
    raw_inventory = review.evidence_bundle.get("inventory")
    if not isinstance(raw_inventory, list):
        raw_inventory = []
    for entry in raw_inventory:
        if not isinstance(entry, dict) or entry.get("sku"):
            continue
        raw_items = entry.get("raw_items")
        if isinstance(raw_items, list):
            unresolved_aliases.update(
                normalized
                for raw_item in raw_items
                if isinstance(raw_item, str) and (normalized := normalize_alias(raw_item))
            )

    validated_mappings: dict[str, str] = {}
    for mapping in mappings:
        normalized = normalize_alias(mapping.raw_item)
        if (
            not normalized
            or mapping.raw_item.strip() != mapping.raw_item
            or mapping.sku.strip() != mapping.sku
            or mapping.basis != "human_decision"
            or mapping.evidence
            or normalized in validated_mappings
        ):
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                "mapping contents are not exact canonical human-decision inputs",
                case_id=review.case_id,
                stop_reason="HUMAN_MAPPING_INVALID",
            )
        if normalized not in unresolved_aliases:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                f"mapping alias is not unresolved inventory evidence in this review: {mapping.raw_item}",
                case_id=review.case_id,
                stop_reason="MAPPING_ALIAS_NOT_IN_REVIEW",
            )
        if mapping.sku not in inventory_skus:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"mapping target SKU does not exist: {mapping.sku}",
                case_id=review.case_id,
                stop_reason="MAPPING_SKU_NOT_FOUND",
            )
        validated_mappings[normalized] = mapping.sku

    if persisted_mapping_provenance is not None:
        expected_provenance = {
            alias: (
                sku,
                f"human_review:{review.review_id}",
                decision.reviewer,
                decision.decided_at.isoformat(),
            )
            for alias, sku in validated_mappings.items()
        }
        if dict(persisted_mapping_provenance) != expected_provenance:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "persisted mapping provenance does not match the exact human ruling",
                case_id=review.case_id,
                stop_reason="HUMAN_MAPPING_PROVENANCE_INVALID",
            )

    superseded = decision.superseded_case_id
    if superseded is not None and decision.decision is not HumanDecisionKind.SUPERSEDE_REVISION:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "superseded_case_id is permitted only for SUPERSEDE_REVISION",
            case_id=review.case_id,
            stop_reason="SUPERSEDED_CASE_INVALID",
        )
    if decision.decision is HumanDecisionKind.SUPERSEDE_REVISION:
        if superseded is None:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                "SUPERSEDE_REVISION requires a superseded case ID",
                case_id=review.case_id,
                stop_reason="SUPERSEDED_CASE_MISSING",
            )
        if superseded not in valid_superseded_case_ids:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                "superseded case lacks exact review and relational provenance",
                case_id=review.case_id,
                stop_reason="SUPERSEDED_CASE_INVALID",
            )

    return tuple(sorted(validated_mappings.items()))


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


def _assert_critique_sequence_complete(
    case_id: str,
    critiques: list[Critique],
) -> None:
    if not critiques:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "no persisted critique cycle exists",
            case_id=case_id,
            stop_reason="CRITIQUE_MISSING",
        )
    if len(critiques) > 2:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "the persisted critic cycle exceeds its finite limit",
            case_id=case_id,
            stop_reason="CRITIQUE_CYCLE_LIMIT",
        )
    first = critiques[0]
    if first.cycle != 1 or first.responds_to_critique_id is not None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "the persisted first critique has an invalid response relationship",
            case_id=case_id,
            stop_reason="CRITIQUE_RESPONSE_INVALID",
        )
    follow_up_required = bool(
        first.challenged_findings
        or first.missing_evidence
        or first.requested_follow_up
    )
    if len(critiques) == 1:
        if follow_up_required:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "the first critique requires one persisted follow-up cycle",
                case_id=case_id,
                stop_reason="CRITIQUE_FOLLOW_UP_REQUIRED",
            )
        return
    second = critiques[1]
    if second.cycle != 2 or second.responds_to_critique_id is None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "the persisted second critique has an invalid response relationship",
            case_id=case_id,
            stop_reason="CRITIQUE_RESPONSE_INVALID",
        )
    _assert_critique_follow_up_addressed(case_id, first, second)


def _assert_critique_follow_up_addressed(
    case_id: str,
    first: Critique,
    second: Critique,
) -> None:
    """Require cycle two to account for every fact that forced the follow-up."""

    if second.requested_follow_up:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "the second critique requests an unpersistable third cycle",
            case_id=case_id,
            stop_reason="CRITIQUE_CYCLE_LIMIT",
        )
    addressed = {
        *second.supported_findings,
        *second.challenged_findings,
        *second.missing_evidence,
    }
    required = {
        *first.challenged_findings,
        *first.missing_evidence,
        *first.requested_follow_up,
    }
    if any(item not in addressed for item in required):
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "the persisted second critique omits an exact required follow-up item",
            case_id=case_id,
            stop_reason="CRITIQUE_FOLLOW_UP_UNADDRESSED",
        )


def assert_critique_cycle_complete(case_id: str, store: WorkflowStore) -> None:
    """Require the exact persisted finite critic sequence before finalization."""

    _assert_critique_sequence_complete(case_id, store.list_critiques(case_id))


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
