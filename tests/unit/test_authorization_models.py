"""Strict semantic contracts for persisted review and human authorization payloads."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from invoice_agents.agents import decision_rules
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.models import (
    CanonicalMapping,
    Critique,
    DecisionKind,
    HumanDecision,
    HumanDecisionKind,
    ReviewRequest,
    SourceArtifact,
)


def _human(**updates: Any) -> HumanDecision:
    values: dict[str, Any] = {
        "review_id": "rev_semantic",
        "reviewer": "reviewer@example.com",
        "decision": HumanDecisionKind.REJECT,
        "reason": "the evidence does not support payment",
        "decided_at": datetime(2026, 8, 8, 12, 1, tzinfo=UTC),
    }
    values.update(updates)
    return HumanDecision(**values)


def _review(**updates: Any) -> ReviewRequest:
    values: dict[str, Any] = {
        "review_id": "rev_semantic",
        "case_id": "case_semantic",
        "status": "PENDING",
        "reasons": ["policy threshold requires review"],
        "amount": None,
        "source": SourceArtifact(
            source_id="src_semantic",
            canonical_path=Path("semantic.txt"),
            sha256="a" * 64,
            source_format="txt",
            size_bytes=1,
            modified_at=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
        ),
        "evidence_bundle": {},
        "agent_recommendation": DecisionKind.HOLD,
        "agent_rationale": ["the policy threshold requires an attributable review"],
        "critic": Critique(
            supported_findings=["the amount is above policy"],
            challenged_findings=[],
            missing_evidence=[],
            requested_follow_up=[],
            recommended_disposition=DecisionKind.HOLD,
            rationale=["human review is required"],
        ),
        "critic_disagreement_reason": None,
        "questions": ["Does the evidence support payment?"],
        "created_at": datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
    }
    values.update(updates)
    return ReviewRequest(**values)


@pytest.mark.parametrize("field", ["reviewer", "reason"])
def test_human_decision_rejects_whitespace_only_attribution(field: str) -> None:
    with pytest.raises(ValidationError):
        _human(**{field: " \t "})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasons", []),
        ("reasons", [" \t"]),
        ("agent_rationale", []),
        ("agent_rationale", [""]),
        ("questions", []),
        ("questions", ["\n"]),
    ],
)
def test_review_rejects_empty_or_malformed_required_text_lists(
    field: str,
    value: list[str],
) -> None:
    with pytest.raises(ValidationError):
        _review(**{field: value})


def test_review_status_human_link_and_chronology_are_exact() -> None:
    with pytest.raises(ValidationError):
        _review(human_decision=_human())
    with pytest.raises(ValidationError):
        _review(status="RESOLVED")
    with pytest.raises(ValidationError):
        _review(
            status="RESOLVED",
            human_decision=_human(
                decided_at=datetime(2026, 8, 8, 12, 0, tzinfo=UTC) - timedelta(microseconds=1)
            ),
        )
    with pytest.raises(ValidationError):
        _review(status="RESOLVED", human_decision=_human(review_id="rev_other"))


def test_review_requires_deterministic_structured_critic_disagreement_fact() -> None:
    critic = _review().critic.model_copy(
        update={"recommended_disposition": DecisionKind.HOLD},
        deep=True,
    )
    expected = "agent recommendation APPROVE conflicts with critic recommendation HOLD"

    bound = _review(
        agent_recommendation=DecisionKind.APPROVE,
        critic=critic,
        critic_disagreement_reason=expected,
    )

    assert bound.critic_disagreement_reason == expected
    with pytest.raises(ValidationError):
        _review(
            agent_recommendation=DecisionKind.APPROVE,
            critic=critic,
            reasons=["arbitrary extra prose claims that the critic disagreed"],
        )
    with pytest.raises(ValidationError):
        _review(
            agent_recommendation=DecisionKind.APPROVE,
            critic=critic,
            critic_disagreement_reason="a caller-controlled disagreement description",
        )
    with pytest.raises(ValidationError):
        _review(critic_disagreement_reason="there is no disagreement")


def _mapping(raw_item: str = "WidgetA (bulk)", sku: str = "SKU-WIDGET-A") -> CanonicalMapping:
    return CanonicalMapping(raw_item=raw_item, sku=sku, basis="human_decision")


def _decision_review() -> ReviewRequest:
    return _review(
        evidence_bundle={
            "inventory": [
                {
                    "sku": None,
                    "raw_items": ["WidgetA (bulk)"],
                }
            ],
            "blocking_evidence": [
                {"blocker_id": "inventory:widgetabulk:UNKNOWN"},
                {"blocker_id": "financial:declared-total-delta"},
            ],
        }
    )


@pytest.mark.parametrize(
    ("decision", "stop_reason"),
    [
        (
            _human(decision=HumanDecisionKind.REJECT, mappings=[_mapping()]),
            "HUMAN_MAPPING_INVALID",
        ),
        (
            _human(decision=HumanDecisionKind.ESTABLISH_MAPPING),
            "HUMAN_MAPPING_MISSING",
        ),
        (
            _human(
                decision=HumanDecisionKind.APPROVE,
                superseded_case_id="case_prior",
                addressed_blocker_ids=[
                    "inventory:widgetabulk:UNKNOWN",
                    "financial:declared-total-delta",
                ],
            ),
            "SUPERSEDED_CASE_INVALID",
        ),
        (
            _human(decision=HumanDecisionKind.SUPERSEDE_REVISION),
            "SUPERSEDED_CASE_MISSING",
        ),
        (
            _human(
                decision=HumanDecisionKind.REJECT,
                addressed_blocker_ids=["inventory:widgetabulk:UNKNOWN"],
            ),
            "BLOCKER_AUTHORIZATION_INVALID",
        ),
        (
            _human(
                decision=HumanDecisionKind.APPROVE,
                addressed_blocker_ids=["inventory:widgetabulk:UNKNOWN"],
            ),
            "BLOCKER_AUTHORIZATION_INVALID",
        ),
    ],
    ids=[
        "mapping-on-reject",
        "missing-mapping",
        "supersession-on-approve",
        "missing-supersession",
        "blockers-on-reject",
        "partial-blocker-set",
    ],
)
def test_shared_human_decision_validator_rejects_incompatible_fields_and_blocker_sets(
    decision: HumanDecision,
    stop_reason: str,
) -> None:
    with pytest.raises(InvoiceAgentsError) as excinfo:
        decision_rules.validate_human_decision_applicability(
            _decision_review(),
            decision,
            inventory_skus=frozenset({"SKU-WIDGET-A"}),
            valid_superseded_case_ids=frozenset({"case_prior"}),
            persisted_mapping_provenance=None,
        )

    assert excinfo.value.stop_reason == stop_reason


def test_shared_human_decision_validator_requires_exact_mapping_contents_and_provenance() -> None:
    review = _decision_review()
    decided_at = datetime(2026, 8, 8, 12, 1, tzinfo=UTC)
    decision = _human(
        decision=HumanDecisionKind.ESTABLISH_MAPPING,
        mappings=[_mapping()],
        decided_at=decided_at,
    )
    exact_provenance = {
        "widgetabulk": (
            "SKU-WIDGET-A",
            "human_review:rev_semantic",
            "reviewer@example.com",
            decided_at.isoformat(),
        )
    }

    assert decision_rules.validate_human_decision_applicability(
        review,
        decision,
        inventory_skus=frozenset({"SKU-WIDGET-A"}),
        valid_superseded_case_ids=frozenset(),
        persisted_mapping_provenance=exact_provenance,
    ) == (("widgetabulk", "SKU-WIDGET-A"),)
    with pytest.raises(InvoiceAgentsError) as provenance_error:
        decision_rules.validate_human_decision_applicability(
            review,
            decision,
            inventory_skus=frozenset({"SKU-WIDGET-A"}),
            valid_superseded_case_ids=frozenset(),
            persisted_mapping_provenance={
                "widgetabulk": (
                    "SKU-WIDGET-A",
                    "manual:unbound",
                    "reviewer@example.com",
                    decided_at.isoformat(),
                )
            },
        )
    assert provenance_error.value.stop_reason == "HUMAN_MAPPING_PROVENANCE_INVALID"

    invalid_contents = decision.model_copy(
        update={
            "mappings": [
                CanonicalMapping(
                    raw_item="WidgetA (bulk)",
                    sku="SKU-WIDGET-A",
                    basis="exact_item_name",
                )
            ]
        },
        deep=True,
    )
    with pytest.raises(InvoiceAgentsError) as contents_error:
        decision_rules.validate_human_decision_applicability(
            review,
            invalid_contents,
            inventory_skus=frozenset({"SKU-WIDGET-A"}),
            valid_superseded_case_ids=frozenset(),
            persisted_mapping_provenance=None,
        )
    assert contents_error.value.stop_reason == "HUMAN_MAPPING_INVALID"
