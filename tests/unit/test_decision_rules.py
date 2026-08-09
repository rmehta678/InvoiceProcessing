"""Pure final-decision rules: validate_final_decision and blocking_evidence on minimal models."""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_agents.agents import decision_rules
from invoice_agents.agents.decision_rules import (
    assert_new_review_cycle_permitted,
    blocking_evidence,
    validate_final_decision,
)
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.models import (
    Critique,
    DecisionKind,
    FinancialComparison,
    HumanDecision,
    HumanDecisionKind,
    InventoryComparison,
    InventoryStatus,
    ReviewRequest,
    RiskAssessment,
    RiskPolicy,
    SourceArtifact,
)

AUTHORIZING = [
    HumanDecisionKind.APPROVE,
    HumanDecisionKind.ESTABLISH_MAPPING,
    HumanDecisionKind.SUPERSEDE_REVISION,
]
BLOCKING_STATUSES = [
    InventoryStatus.EXCEEDS_STOCK,
    InventoryStatus.OUT_OF_STOCK,
    InventoryStatus.UNKNOWN,
    InventoryStatus.INVALID_QUANTITY,
    InventoryStatus.ERROR,
]


def make_financial(total_delta: Decimal | None = None) -> FinancialComparison:
    return FinancialComparison(
        calculated_subtotal=Decimal("100"),
        declared_subtotal=None,
        subtotal_delta=None,
        calculated_tax=Decimal("0"),
        declared_tax=None,
        tax_delta=None,
        tax_recomputable=True,
        tax_basis="test basis",
        calculated_fees=Decimal("0"),
        calculated_total=Decimal("100"),
        declared_total=None,
        total_delta=total_delta,
        line_deltas={},
        exact=total_delta in (None, Decimal("0")),
    )


def make_comparison(status: InventoryStatus, stock: int | None = 15) -> InventoryComparison:
    return InventoryComparison(
        sku="SKU-WIDGET-A",
        raw_items=["WidgetA"],
        requested_quantity=Decimal("20"),
        available_stock=stock,
        status=status,
        explanation="test evidence",
    )


def make_risk(
    inventory: list[InventoryComparison] | None = None,
    total_delta: Decimal | None = None,
    reasons: list[str] | None = None,
) -> RiskAssessment:
    return RiskAssessment(
        policy=RiskPolicy(
            review_threshold_amount=Decimal("10000.00"),
            review_threshold_currency="USD",
            review_threshold_effective_date=date(2026, 8, 6),
            due_date_tolerance_days=3,
        ),
        financial=make_financial(total_delta),
        dates=[],
        inventory=inventory or [],
        identity_candidates=[],
        suspicious_signals=[],
        unavailable_reconciliations=[],
        policy_review_reasons=reasons or [],
    )


def make_critique(disposition: DecisionKind) -> Critique:
    return Critique(
        supported_findings=["evidence reviewed"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=disposition,
        rationale=["test rationale"],
    )


def make_review(
    decision: HumanDecisionKind, addressed_blocker_ids: list[str] | None = None
) -> ReviewRequest:
    return ReviewRequest(
        review_id="rev_test",
        case_id="case_test",
        status="RESOLVED",
        reasons=["policy trigger"],
        amount=None,
        source=SourceArtifact(
            source_id="src_test",
            canonical_path=Path("invoice_test.txt"),
            sha256="0" * 64,
            source_format="txt",
            size_bytes=1,
            modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        evidence_bundle={},
        agent_recommendation=DecisionKind.HOLD,
        agent_rationale=["needs human review"],
        critic=make_critique(DecisionKind.HOLD),
        questions=[],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        human_decision=HumanDecision(
            review_id="rev_test",
            reviewer="reviewer@example.com",
            decision=decision,
            reason="attributable human ruling",
            decided_at=datetime(2026, 1, 2, tzinfo=UTC),
            superseded_case_id="case_prior"
            if decision is HumanDecisionKind.SUPERSEDE_REVISION
            else None,
            addressed_blocker_ids=addressed_blocker_ids or [],
        ),
    )


@pytest.mark.parametrize("decision", AUTHORIZING)
def test_authorizing_human_decision_permits_approve_despite_critic(
    decision: HumanDecisionKind,
) -> None:
    result = validate_final_decision(
        DecisionKind.APPROVE,
        True,
        make_risk(reasons=["policy trigger"]),
        make_critique(DecisionKind.HOLD),
        make_review(decision),
    )
    assert result is None


def test_authorizing_decision_permits_hold_when_blocking_evidence_remains() -> None:
    risk = make_risk(
        inventory=[make_comparison(InventoryStatus.EXCEEDS_STOCK)],
        reasons=["policy trigger"],
    )
    result = validate_final_decision(
        DecisionKind.HOLD,
        False,
        risk,
        make_critique(DecisionKind.HOLD),
        make_review(HumanDecisionKind.APPROVE),
    )
    assert result is None


def test_hold_without_blocking_evidence_conflicts_with_authorizing_decision() -> None:
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.HOLD,
            False,
            make_risk(reasons=["policy trigger"]),
            make_critique(DecisionKind.HOLD),
            make_review(HumanDecisionKind.APPROVE),
        )
    assert excinfo.value.stop_reason == "HUMAN_AGENT_DECISION_CONFLICT"


@pytest.mark.parametrize(
    "inventory",
    [[], [make_comparison(InventoryStatus.EXCEEDS_STOCK)]],
    ids=["no_blockers", "with_blockers"],
)
def test_reject_after_authorizing_decision_is_a_conflict(
    inventory: list[InventoryComparison],
) -> None:
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.REJECT,
            False,
            make_risk(inventory=inventory, reasons=["policy trigger"]),
            make_critique(DecisionKind.REJECT),
            make_review(HumanDecisionKind.ESTABLISH_MAPPING),
        )
    assert excinfo.value.stop_reason == "HUMAN_AGENT_DECISION_CONFLICT"


def test_human_reject_forbids_agent_approve() -> None:
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.APPROVE,
            True,
            make_risk(reasons=["policy trigger"]),
            make_critique(DecisionKind.APPROVE),
            make_review(HumanDecisionKind.REJECT),
        )
    assert excinfo.value.stop_reason == "HUMAN_AGENT_DECISION_CONFLICT"


@pytest.mark.parametrize(
    ("selected", "payment_eligible"),
    [(DecisionKind.REJECT, False), (DecisionKind.APPROVE, True)],
)
def test_request_correction_forbids_anything_but_hold(
    selected: DecisionKind, payment_eligible: bool
) -> None:
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            selected,
            payment_eligible,
            make_risk(reasons=["policy trigger"]),
            make_critique(DecisionKind.APPROVE),
            make_review(HumanDecisionKind.REQUEST_CORRECTION),
        )
    assert excinfo.value.stop_reason == "HUMAN_AGENT_DECISION_CONFLICT"


def test_request_correction_permits_hold() -> None:
    result = validate_final_decision(
        DecisionKind.HOLD,
        False,
        make_risk(reasons=["policy trigger"]),
        make_critique(DecisionKind.HOLD),
        make_review(HumanDecisionKind.REQUEST_CORRECTION),
    )
    assert result is None


def test_approve_with_payment_ineligible_is_invalid() -> None:
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.APPROVE,
            False,
            make_risk(),
            make_critique(DecisionKind.APPROVE),
            None,
        )
    assert excinfo.value.stop_reason == "FINAL_DECISION_INVALID"


def test_non_approve_with_payment_eligible_is_invalid() -> None:
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.HOLD,
            True,
            make_risk(),
            make_critique(DecisionKind.HOLD),
            None,
        )
    assert excinfo.value.stop_reason == "FINAL_DECISION_INVALID"


@pytest.mark.parametrize("selected", [DecisionKind.REJECT, DecisionKind.HOLD])
def test_stricter_decision_than_critic_approve_is_never_blocked(selected: DecisionKind) -> None:
    result = validate_final_decision(
        selected,
        False,
        make_risk(),
        make_critique(DecisionKind.APPROVE),
        None,
    )
    assert result is None


@pytest.mark.parametrize("status", BLOCKING_STATUSES)
def test_blocking_evidence_flags_each_blocking_inventory_status(status: InventoryStatus) -> None:
    entries = blocking_evidence(make_risk(inventory=[make_comparison(status)]))
    assert [entry.model_dump() for entry in entries] == [
        {
            "blocker_id": f"inventory:SKU-WIDGET-A:{status.value}",
            "kind": "inventory",
            "evidence_id": "SKU-WIDGET-A",
            "description": (
                f"inventory {status.value}: WidgetA requested=20 stock=15"
            ),
        }
    ]


def test_blocking_evidence_flags_nonzero_total_delta() -> None:
    entries = blocking_evidence(make_risk(total_delta=Decimal("12.50")))
    assert [entry.model_dump() for entry in entries] == [
        {
            "blocker_id": "financial:declared-total-delta",
            "kind": "financial",
            "evidence_id": "declared-total-delta",
            "description": "declared/calculated total delta is 12.50",
        }
    ]


def test_blocker_ids_ignore_order_and_mutable_explanatory_prose() -> None:
    first = make_comparison(InventoryStatus.EXCEEDS_STOCK)
    changed = first.model_copy(
        update={
            "raw_items": ["Second description", "WidgetA"],
            "requested_quantity": Decimal("999"),
            "available_stock": 0,
            "explanation": "model-generated prose changed",
        }
    )
    first_id = blocking_evidence(make_risk(inventory=[first]))[0].blocker_id
    changed_id = blocking_evidence(make_risk(inventory=[changed]))[0].blocker_id
    assert first_id == changed_id == "inventory:SKU-WIDGET-A:EXCEEDS_STOCK"


def test_blocker_id_uses_normalized_raw_domain_identity_when_sku_is_absent() -> None:
    comparison = make_comparison(InventoryStatus.UNKNOWN, stock=None).model_copy(
        update={"sku": None, "raw_items": ["  Widget A (rush)!  "]}
    )
    blocker = blocking_evidence(make_risk(inventory=[comparison]))[0]
    assert blocker.blocker_id == "inventory:widgetarush:UNKNOWN"
    assert blocker.evidence_id == "widgetarush"


def test_financial_blocker_id_does_not_change_with_delta_value() -> None:
    first = blocking_evidence(make_risk(total_delta=Decimal("1.00")))[0]
    second = blocking_evidence(make_risk(total_delta=Decimal("999.00")))[0]
    assert first.blocker_id == second.blocker_id == "financial:declared-total-delta"


@pytest.mark.parametrize("status", BLOCKING_STATUSES)
def test_approve_rejects_each_unaddressed_inventory_blocker(
    status: InventoryStatus,
) -> None:
    risk = make_risk(
        inventory=[make_comparison(status)], reasons=["policy trigger"]
    )
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.APPROVE,
            True,
            risk,
            make_critique(DecisionKind.APPROVE),
            make_review(HumanDecisionKind.APPROVE),
        )
    assert excinfo.value.stop_reason == "BLOCKING_EVIDENCE_UNRESOLVED"


def test_approve_rejects_unaddressed_nonzero_total_delta() -> None:
    risk = make_risk(total_delta=Decimal("12.50"), reasons=["policy trigger"])
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.APPROVE,
            True,
            risk,
            make_critique(DecisionKind.APPROVE),
            make_review(HumanDecisionKind.APPROVE),
        )
    assert excinfo.value.stop_reason == "BLOCKING_EVIDENCE_UNRESOLVED"


@pytest.mark.parametrize("decision", AUTHORIZING)
def test_exact_current_blocker_ids_explicitly_authorize_approve(
    decision: HumanDecisionKind,
) -> None:
    risk = make_risk(
        inventory=[make_comparison(InventoryStatus.EXCEEDS_STOCK)],
        total_delta=Decimal("12.50"),
        reasons=["policy trigger"],
    )
    current_ids = [blocker.blocker_id for blocker in blocking_evidence(risk)]
    review = make_review(decision, current_ids)
    assert decision_rules.unaddressed_blockers(risk, review.human_decision) == []
    assert (
        validate_final_decision(
            DecisionKind.APPROVE,
            True,
            risk,
            make_critique(DecisionKind.HOLD),
            review,
        )
        is None
    )


@pytest.mark.parametrize(
    "addressed",
    [
        [],
        ["inventory:SKU-WIDGET-A:OUT_OF_STOCK"],
        [
            "inventory:SKU-WIDGET-A:EXCEEDS_STOCK",
            "financial:declared-total-delta",
            "inventory:STALE:UNKNOWN",
        ],
    ],
    ids=["missing", "stale", "extra"],
)
@pytest.mark.parametrize("decision", AUTHORIZING)
def test_approve_mapping_or_supersession_needs_exact_current_blocker_ids(
    addressed: list[str],
    decision: HumanDecisionKind,
) -> None:
    risk = make_risk(
        inventory=[make_comparison(InventoryStatus.EXCEEDS_STOCK)],
        total_delta=Decimal("12.50"),
        reasons=["policy trigger"],
    )
    review = make_review(decision, addressed)
    remaining = decision_rules.unaddressed_blockers(risk, review.human_decision)
    assert remaining
    with pytest.raises(InvoiceAgentsError) as excinfo:
        validate_final_decision(
            DecisionKind.APPROVE,
            True,
            risk,
            make_critique(DecisionKind.APPROVE),
            review,
        )
    assert excinfo.value.stop_reason == "BLOCKING_EVIDENCE_UNRESOLVED"


def test_blocking_evidence_ignores_non_blocking_statuses_and_zero_delta() -> None:
    clean = make_risk(
        inventory=[
            make_comparison(InventoryStatus.AVAILABLE),
            make_comparison(InventoryStatus.AMBIGUOUS, stock=None),
        ],
        total_delta=Decimal("0"),
    )
    assert blocking_evidence(clean) == []
    assert blocking_evidence(make_risk()) == []


@pytest.mark.parametrize(
    "decision", [HumanDecisionKind.REJECT, HumanDecisionKind.REQUEST_CORRECTION]
)
def test_final_human_ruling_forbids_a_new_review_cycle(decision: HumanDecisionKind) -> None:
    with pytest.raises(InvoiceAgentsError) as excinfo:
        assert_new_review_cycle_permitted(make_review(decision), case_id="case_test")
    assert excinfo.value.stop_reason == "HUMAN_DECISION_MUST_BE_OBEYED"
    assert excinfo.value.case_id == "case_test"


@pytest.mark.parametrize("decision", AUTHORIZING)
def test_authorizing_ruling_permits_a_new_review_cycle(decision: HumanDecisionKind) -> None:
    assert assert_new_review_cycle_permitted(make_review(decision)) is None


def test_absent_or_pending_review_permits_a_new_cycle() -> None:
    assert assert_new_review_cycle_permitted(None) is None
    pending = make_review(HumanDecisionKind.REJECT).model_copy(
        update={"status": "PENDING", "human_decision": None}, deep=True
    )
    assert assert_new_review_cycle_permitted(pending) is None
