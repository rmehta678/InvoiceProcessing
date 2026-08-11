"""Live-contract evidence remains complete without exposing provider payloads."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from invoice_agents import compatibility
from invoice_agents.config import Settings
from invoice_agents.errors import InvoiceAgentsError

SECRET = "sk-proj-live-contract-secret"
REQUEST_URL = "https://api.x.ai/v1/chat/completions"


def _settings() -> Settings:
    return Settings(xai_api_key="xai-not-used-by-fake-client")


class _Completions:
    def __init__(self, *, result: object | None = None, error: BaseException | None = None) -> None:
        self._result = result
        self._error = error

    async def create(self, **_kwargs: object) -> object:
        if self._error is not None:
            raise self._error
        return self._result


class _DirectClient:
    def __init__(self, completions: _Completions) -> None:
        self.chat = SimpleNamespace(completions=completions)

    async def close(self) -> None:
        return None


def _status_error(
    *,
    message: str,
    status_code: int = 400,
    request_id: str = "req-contract-safe",
    error_fields: dict[str, str] | None = None,
) -> openai.BadRequestError:
    request = httpx.Request("POST", REQUEST_URL)
    response = httpx.Response(
        status_code,
        request=request,
        headers={"x-request-id": request_id},
    )
    return openai.BadRequestError(
        message,
        response=response,
        body={
            "error": {
                "message": message,
                "credential": SECRET,
                **(error_fields or {}),
            }
        },
    )


def _install_direct_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: object | None = None,
    error: BaseException | None = None,
) -> None:
    client = _DirectClient(_Completions(result=result, error=error))
    monkeypatch.setattr(
        compatibility.openai,
        "AsyncOpenAI",
        lambda **_kwargs: client,
    )


@pytest.mark.asyncio
async def test_invalid_key_contract_evidence_excludes_exception_and_provider_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _status_error(
        message=f"Incorrect API key provided: api_key={SECRET}",
        error_fields={
            "type": "authentication_error",
            "code": "incorrect_api_key",
        },
    )
    _install_direct_client(monkeypatch, error=error)

    check = await compatibility._live_invalid_key_rejection()

    assert check.passed is True
    assert check.evidence == (
        "category=AUTHENTICATION_REJECTION; status=400; provider_request_id=req-contract-safe"
    )
    assert SECRET not in check.evidence
    assert "Incorrect API key" not in check.evidence
    assert "body=" not in check.evidence


@pytest.mark.asyncio
async def test_invalid_key_prose_without_structured_authentication_fields_is_not_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _status_error(message=f"Incorrect API key provided: api_key={SECRET}")
    _install_direct_client(monkeypatch, error=error)

    check = await compatibility._live_invalid_key_rejection()

    assert check.passed is False
    assert check.evidence == (
        "exception_type=BadRequestError; status=400; category=PROVIDER; "
        "stop_reason=PROVIDER_REQUEST_FAILED; provider_request_id=req-contract-safe"
    )
    assert SECRET not in check.evidence
    assert "Incorrect API key" not in check.evidence


@pytest.mark.asyncio
async def test_provider_metadata_sanitizer_failure_is_explicit_and_suppresses_raw_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _status_error(message=f"Incorrect API key provided: api_key={SECRET}")
    _install_direct_client(monkeypatch, error=error)

    def fail_sanitizer(_value: str) -> None:
        raise RuntimeError(f"sanitizer crashed around api_key={SECRET}")

    monkeypatch.setattr(compatibility, "sanitize_text", fail_sanitizer)

    with pytest.raises(InvoiceAgentsError) as caught:
        await compatibility._live_invalid_key_rejection()

    assert caught.value.stop_reason == "SANITIZATION_FAILED"
    assert caught.value.message == "credential sanitization failed closed"
    assert caught.value.__suppress_context__ is True
    assert SECRET not in str(caught.value)


@pytest.mark.asyncio
async def test_structured_rejection_evidence_excludes_exception_and_provider_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _status_error(
        message=f"Schema validation failed: api_key={SECRET}",
        error_fields={
            "type": "invalid_request_error",
            "code": "invalid_json_schema",
            "param": "response_format.json_schema",
        },
    )
    _install_direct_client(monkeypatch, error=error)

    check = await compatibility._structured_output_rejection_live(_settings())

    assert check.passed is True
    assert check.evidence == (
        "category=SCHEMA_REJECTION; status=400; provider_request_id=req-contract-safe"
    )
    assert SECRET not in check.evidence
    assert "Schema validation failed" not in check.evidence
    assert "body=" not in check.evidence


@pytest.mark.asyncio
async def test_schema_rejection_prose_without_exact_machine_fields_is_not_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _status_error(message=f"Schema validation failed: api_key={SECRET}")
    _install_direct_client(monkeypatch, error=error)

    check = await compatibility._structured_output_rejection_live(_settings())

    assert check.passed is False
    assert check.evidence == (
        "exception_type=BadRequestError; status=400; category=PROVIDER; "
        "stop_reason=PROVIDER_REQUEST_FAILED; provider_request_id=req-contract-safe"
    )
    assert SECRET not in check.evidence
    assert "Schema validation failed" not in check.evidence


@pytest.mark.asyncio
async def test_structured_rejection_does_not_turn_transport_failure_into_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", REQUEST_URL)
    error = openai.APIConnectionError(
        message=f"transport failed api_key={SECRET}",
        request=request,
    )
    _install_direct_client(monkeypatch, error=error)

    check = await compatibility._structured_output_rejection_live(_settings())

    assert check.passed is False
    assert check.evidence == (
        "exception_type=APIConnectionError; status=<absent>; category=PROVIDER; "
        "stop_reason=PROVIDER_REQUEST_FAILED; provider_request_id=<absent>"
    )
    assert SECRET not in check.evidence
    assert "transport failed" not in check.evidence


@pytest.mark.asyncio
async def test_structured_acceptance_does_not_echo_provider_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=f"raw provider body {SECRET}"))]
    )
    _install_direct_client(monkeypatch, result=completion)

    check = await compatibility._structured_output_rejection_live(_settings())

    assert check.passed is False
    assert check.evidence == (
        "category=PROVIDER; status=UNEXPECTED_ACCEPT; provider_request_id=<absent>"
    )
    assert SECRET not in check.evidence
    assert "raw provider body" not in check.evidence
