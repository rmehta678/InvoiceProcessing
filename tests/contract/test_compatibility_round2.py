"""Fail-closed local classifications for live compatibility probes."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from invoice_agents import compatibility
from invoice_agents.config import Settings


class _RejectingClient:
    def __init__(self, failure: BaseException) -> None:
        self.failure = failure
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **_kwargs: object) -> object:
        raise self.failure

    async def close(self) -> None:
        return None


def _status_error(
    error_type: type[openai.APIStatusError],
    status: int,
    body: object,
) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://local.invalid/v1/chat/completions")
    response = httpx.Response(
        status,
        request=request,
        headers={"x-request-id": f"req_round2_{status}"},
    )
    return error_type("round2-provider-prose", response=response, body=body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        _status_error(
            openai.AuthenticationError,
            401,
            {"error": {"type": "authentication_error", "code": "invalid_api_key"}},
        ),
        _status_error(
            openai.RateLimitError,
            429,
            {"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded"}},
        ),
        openai.APIConnectionError(
            request=httpx.Request("POST", "https://local.invalid/v1/chat/completions")
        ),
        openai.APITimeoutError(httpx.Request("POST", "https://local.invalid/v1/chat/completions")),
        _status_error(
            openai.InternalServerError,
            500,
            {"error": {"type": "server_error", "code": "internal_error"}},
        ),
        _status_error(
            openai.BadRequestError,
            400,
            {"error": {"type": "invalid_request_error", "code": "unrelated_parameter"}},
        ),
    ],
)
async def test_schema_probe_rejects_non_schema_failures(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    monkeypatch.setattr(
        compatibility.openai,
        "AsyncOpenAI",
        lambda **_kwargs: _RejectingClient(failure),
    )

    check = await compatibility._structured_output_rejection_live(settings)

    assert check.passed is False
    assert "round2-provider-prose" not in check.evidence


@pytest.mark.asyncio
async def test_schema_probe_accepts_only_exact_structured_schema_rejection(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = _status_error(
        openai.BadRequestError,
        400,
        {
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_json_schema",
                "param": "response_format",
                "message": "round2-provider-prose",
            }
        },
    )
    monkeypatch.setattr(
        compatibility.openai,
        "AsyncOpenAI",
        lambda **_kwargs: _RejectingClient(failure),
    )

    check = await compatibility._structured_output_rejection_live(settings)

    assert check.passed is True
    assert check.evidence == (
        "category=SCHEMA_REJECTION; status=400; provider_request_id=req_round2_400"
    )
    assert "round2-provider-prose" not in check.evidence


@pytest.mark.asyncio
async def test_invalid_key_probe_does_not_treat_an_unrelated_400_as_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = _status_error(
        openai.BadRequestError,
        400,
        {"error": {"type": "invalid_request_error", "code": "unrelated_parameter"}},
    )
    monkeypatch.setattr(
        compatibility.openai,
        "AsyncOpenAI",
        lambda **_kwargs: _RejectingClient(failure),
    )

    check = await compatibility._live_invalid_key_rejection()

    assert check.passed is False
    assert "round2-provider-prose" not in check.evidence
