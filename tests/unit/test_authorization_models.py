"""Strict semantic contracts for persisted review and human authorization payloads."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from invoice_agents.models import (
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
