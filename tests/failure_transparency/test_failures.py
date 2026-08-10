"""Failures remain FAILED/ERROR and never synthesize approval or payment."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import openai
import pytest

from invoice_agents.config import Settings
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseStatus, ToolStatus
from invoice_agents.orchestration import (
    _error_record,
    _failed_result,
    _write_result,
    process_invoice,
)
from invoice_agents.tools.comparison import InventoryReader


def test_missing_key_fails_before_case_or_model(
    invoice_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Preflight failures now write artifacts/results JSON (G11); keep it in tmp.
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        xai_api_key=None,
        inventory_db=tmp_path / "missing-inventory.db",
        workflow_db=tmp_path / "missing-workflow.db",
    )
    result = asyncio.run(process_invoice(invoice_dir / "invoice_1001.txt", settings))
    assert result.status is CaseStatus.FAILED
    assert result.stop_reason == "PROVIDER_PREFLIGHT_FAILED"
    assert result.final_decision is None
    assert result.payment is None


def test_missing_sqlite_lookup_is_error_not_not_found(tmp_path: Path) -> None:
    result = InventoryReader(tmp_path / "missing.db").lookup_inventory_exact("NeverFound")
    assert result.status is ToolStatus.ERROR
    assert result.status is not ToolStatus.NOT_FOUND


def test_provider_error_categories_and_request_ids_remain_distinct() -> None:
    request = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    unauthorized_response = httpx.Response(
        401, request=request, headers={"x-request-id": "req-auth-sentinel"}
    )
    auth = _error_record(
        openai.AuthenticationError("invalid key", response=unauthorized_response, body=None)
    )
    assert auth.category == "AUTHENTICATION"
    assert auth.stop_reason == "PROVIDER_AUTHENTICATION_FAILED"
    assert auth.provider_request_id == "req-auth-sentinel"

    rate_response = httpx.Response(429, request=request, headers={"x-request-id": "req-rate"})
    rate = _error_record(openai.RateLimitError("exhausted", response=rate_response, body=None))
    assert rate.category == "RATE_LIMIT"
    assert rate.stop_reason == "PROVIDER_RATE_LIMIT_EXHAUSTED"
    assert rate.provider_request_id == "req-rate"

    timeout = _error_record(openai.APITimeoutError(request))
    assert timeout.category == "TIMEOUT"
    assert timeout.stop_reason == "PROVIDER_TIMEOUT"

    network = _error_record(openai.APIConnectionError(request=request))
    assert network.category == "PROVIDER"
    assert network.stop_reason == "PROVIDER_REQUEST_FAILED"


def test_exception_credentials_are_sanitized_before_result_artifact_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_secret = "sk-proj-abcdefgh_12345678"
    key_secret = "plainResultCredential"
    cookie_secret = "session=plainCookieCredential"
    cause = RuntimeError(f"Cookie: {cookie_secret}")
    error = InvoiceAgentsError(
        ErrorCategory.PROVIDER,
        f"provider exception api_key={key_secret}; token={token_secret}",
        case_id="case_sensitive_result",
        stop_reason="PROVIDER_REQUEST_FAILED",
        provider_request_id="req_safe_result_123",
        details={
            "exception_chain": cause,
            "nested": {"Authorization": f"Bearer {token_secret}"},
        },
    )
    error.__cause__ = cause
    result = _failed_result(
        "case_sensitive_result",
        "src_sensitive_result",
        datetime.now(UTC),
        error,
    )
    monkeypatch.chdir(tmp_path)

    target = _write_result(result)

    raw = target.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["status"] == "FAILED"
    assert payload["errors"][0]["provider_request_id"] == "req_safe_result_123"
    assert "[REDACTED]" in raw
    for secret in (token_secret, key_secret, cookie_secret):
        assert secret not in raw
    assert payload["final_decision"] is None
    assert payload["payment"] is None


def test_provider_response_body_and_credential_are_not_copied_into_error_record() -> None:
    secret = "xai-abcdefgh_12345678"
    request = httpx.Request(
        "POST",
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {secret}"},
    )
    response = httpx.Response(
        400,
        request=request,
        headers={"x-request-id": "req_provider_body_safe"},
        json={"error": {"message": f"invalid api_key={secret}"}},
    )
    error = openai.BadRequestError(
        f"provider body contained api_key={secret}",
        response=response,
        body={"error": {"message": f"invalid api_key={secret}"}},
    )

    record = _error_record(error)
    encoded = record.model_dump_json()

    assert record.category == "PROVIDER"
    assert record.stop_reason == "PROVIDER_REQUEST_FAILED"
    assert record.provider_request_id == "req_provider_body_safe"
    assert record.message == "provider request failed"
    assert secret not in encoded
    assert "body" not in record.details
