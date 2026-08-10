"""No-cost AutoGen schema contracts that complement the opt-in live suite."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest
from autogen_core.tools import FunctionTool

from invoice_agents import compatibility
from invoice_agents.config import Settings


def test_invalid_strict_tool_schema_fails_at_construction() -> None:
    def defaulted_argument(value: int = 1) -> int:
        return value

    with pytest.raises(ValueError, match="Default arguments are not allowed"):
        tool = FunctionTool(
            defaulted_argument,
            description="An intentionally unsupported strict schema.",
            strict=True,
        )
        _ = tool.schema


@pytest.mark.asyncio
async def test_round1_live_evidence_uses_only_stable_safe_fields_without_network(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ModelRawResponse:
        def __init__(self) -> None:
            self.headers = {"x-request-id": "req_round1_model"}

        @staticmethod
        def parse() -> object:
            return SimpleNamespace(model="grok-4.5-round1-response-model-marker")

    class _ModelClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    with_raw_response=SimpleNamespace(create=self.create_model)
                )
            )

        async def create_model(self, **_kwargs: object) -> _ModelRawResponse:
            return _ModelRawResponse()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(compatibility.openai, "AsyncOpenAI", lambda **_kwargs: _ModelClient())
    model_check = await compatibility._server_echoed_model_identity(settings)

    exception_marker = "round1-provider-exception-marker"
    body_marker = "round1-provider-body-marker"
    cookie_marker = "round1-provider-cookie-marker"
    request = httpx.Request(
        "POST",
        "https://local.invalid/v1/chat/completions",
        headers={"Cookie": f"session={cookie_marker}"},
    )
    auth_response = httpx.Response(
        401,
        request=request,
        headers={
            "x-request-id": "req_round1_authentication",
            "Set-Cookie": f"session={cookie_marker}",
        },
        json={"error": {"message": body_marker}},
    )
    authentication_error = openai.AuthenticationError(
        exception_marker,
        response=auth_response,
        body={
            "error": {
                "type": "authentication_error",
                "code": "invalid_api_key",
                "message": body_marker,
            }
        },
    )
    schema_response = httpx.Response(
        400,
        request=request,
        headers={"x-request-id": "req_round1_bad_schema"},
    )
    schema_error = openai.BadRequestError(
        exception_marker,
        response=schema_response,
        body={
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_json_schema",
                "param": "response_format",
                "message": body_marker,
            }
        },
    )

    class _RejectingClient:
        def __init__(self, failure: openai.APIError) -> None:
            self.failure = failure
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **_kwargs: object) -> object:
            raise self.failure

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        compatibility.openai,
        "AsyncOpenAI",
        lambda **_kwargs: _RejectingClient(authentication_error),
    )
    invalid_key_check = await compatibility._live_invalid_key_rejection()
    monkeypatch.setattr(
        compatibility.openai,
        "AsyncOpenAI",
        lambda **_kwargs: _RejectingClient(schema_error),
    )
    schema_check = await compatibility._structured_output_rejection_live(settings)

    class _AcceptingClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **_kwargs: object) -> object:
            return object()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        compatibility.openai,
        "AsyncOpenAI",
        lambda **_kwargs: _AcceptingClient(),
    )
    accepted_invalid_key_check = await compatibility._live_invalid_key_rejection()

    assert model_check.passed is True
    assert invalid_key_check.passed is True
    assert schema_check.passed is True
    assert model_check.evidence == (
        "category=MODEL_IDENTITY; status=MATCHED; provider_request_id=req_round1_model"
    )
    assert invalid_key_check.evidence == (
        "category=AUTHENTICATION_REJECTION; status=401; "
        "provider_request_id=req_round1_authentication"
    )
    assert schema_check.evidence == (
        "category=SCHEMA_REJECTION; status=400; provider_request_id=req_round1_bad_schema"
    )
    assert accepted_invalid_key_check.passed is False
    assert accepted_invalid_key_check.evidence == (
        "category=AUTHENTICATION; status=UNEXPECTED_ACCEPT; provider_request_id=<absent>"
    )
    encoded = " ".join(
        (
            model_check.evidence,
            invalid_key_check.evidence,
            schema_check.evidence,
            accepted_invalid_key_check.evidence,
        )
    )
    for marker in (
        "round1-response-model-marker",
        exception_marker,
        body_marker,
        cookie_marker,
    ):
        assert marker not in encoded
