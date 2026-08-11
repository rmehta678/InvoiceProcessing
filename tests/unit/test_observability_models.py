"""Secret redaction and strict structured-output validation."""

import pytest
from pydantic import ValidationError

from invoice_agents.models import DecisionKind, FinalDecision
from invoice_agents.observability.audit import redact


def test_recursive_redaction() -> None:
    value = {
        "Authorization": "Bearer abc.secret",
        "nested": {"xai_api_key": "secret-value", "safe": "keep"},
        "message": "Authorization was Bearer token123",
    }
    cleaned = redact(value)
    assert cleaned["Authorization"] == "[REDACTED]"
    assert cleaned["nested"]["xai_api_key"] == "[REDACTED]"
    assert cleaned["nested"]["safe"] == "keep"
    assert "token123" not in cleaned["message"]


@pytest.mark.parametrize("usage_key", ["prompt_tokens", "completion_tokens"])
def test_redaction_preserves_only_canonical_numeric_usage_token_counts(usage_key: str) -> None:
    value = {
        usage_key: 17,
        "nested": {usage_key: 0},
        "token": "credential",
        "access_token": "credential",
        "refresh_token": "credential",
        "authorization": "credential",
        "cookie": "credential",
        "api_key": "credential",
        f"noncanonical_{usage_key}": "17",
        usage_key.upper(): 17,
        f"prefixed_{usage_key}": 17,
        usage_key.replace("_", "-"): 17,
    }

    cleaned = redact(value)

    assert cleaned[usage_key] == 17
    assert cleaned["nested"][usage_key] == 0
    for key in (
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "cookie",
        "api_key",
        f"noncanonical_{usage_key}",
        usage_key.upper(),
        f"prefixed_{usage_key}",
        usage_key.replace("_", "-"),
    ):
        assert cleaned[key] == "[REDACTED]"


@pytest.mark.parametrize("invalid_value", [True, -1, 1.0, "1", None])
def test_redaction_rejects_noncanonical_usage_token_counts(invalid_value: object) -> None:
    cleaned = redact(
        {
            "prompt_tokens": invalid_value,
            "completion_tokens": invalid_value,
        }
    )

    assert cleaned == {
        "prompt_tokens": "[REDACTED]",
        "completion_tokens": "[REDACTED]",
    }


def test_invalid_structured_output_is_not_defaulted() -> None:
    with pytest.raises(ValidationError):
        FinalDecision.model_validate(
            {
                "decision": "MAYBE",
                "reasons": [],
                "critic_disposition": DecisionKind.HOLD,
                "payment_eligible": True,
                "unexpected": "field",
            }
        )
    with pytest.raises(ValidationError, match="only APPROVE"):
        FinalDecision(
            decision=DecisionKind.REJECT,
            reasons=["no"],
            critic_disposition=DecisionKind.REJECT,
            payment_eligible=True,
        )
