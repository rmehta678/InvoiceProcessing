"""Failures remain FAILED/ERROR and never synthesize approval or payment."""

import asyncio
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from rich.console import Console

from invoice_agents import cli
from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import (
    CaseStatus,
    DecisionKind,
    FinalDecision,
    PaymentStatus,
    ToolStatus,
    UsageSummary,
)
from invoice_agents.orchestration import (
    _error_record,
    _failed_result,
    _result_from_stop,
    _write_result,
    prepare_case,
    process_invoice,
)
from invoice_agents.payment import service as payment_service
from invoice_agents.tools.comparison import InventoryReader
from invoice_agents.ui.server import create_app


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


def test_round1_payment_errors_share_one_sanitized_db_artifact_ui_cli_contract(
    invoice_dir: Path,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_a = "round1-payment-cookie-a"
    marker_b = "round1-payment-cookie-b"
    unsafe_error = f"Cookie: session={marker_a}; preference={marker_b}"
    prepared = prepare_case(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at = prepared
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    invoice = store.promote_predecessor_extraction(claim)

    def unsafe_authorization(*_args: object, **_kwargs: object) -> object:
        raise payment_service._AuthorizationSnapshotError(unsafe_error)

    monkeypatch.setattr(payment_service, "_load_authorization_snapshot", unsafe_authorization)
    payment = payment_service.mock_payment(
        case_id,
        invoice,
        store,
        settings.workflow_db,
        claim,
    )
    assert payment.status is PaymentStatus.NOT_ELIGIBLE
    assert payment.error == "Cookie: [REDACTED]"

    final = FinalDecision(
        decision=DecisionKind.APPROVE,
        reasons=["synthetic local payment-failure terminalization"],
        critic_disposition=DecisionKind.APPROVE,
        payment_eligible=True,
    )
    context = SimpleNamespace(
        case_id=case_id,
        store=SimpleNamespace(
            load_current_review=lambda _claim: None,
            load_current_final_decision=lambda _claim: final,
        ),
        claim=claim,
        payment_result=payment,
        tool_failures=[],
        invoice=lambda: invoice,
    )
    result = _result_from_stop(
        context,
        SimpleNamespace(stop_reason="payment stopped"),
        started_at,
        UsageSummary(),
    )
    assert result.errors[-1].message == "Cookie: [REDACTED]"

    original_payment_facts = payment.model_dump(exclude={"error"})
    store.finish_case(result, claim)
    monkeypatch.chdir(tmp_path)
    artifact = _write_result(result)
    with connect_database(settings.workflow_db, read_only=True) as connection:
        database_raw = str(
            connection.execute(
                "SELECT result_json FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()["result_json"]
        )
    artifact_raw = artifact.read_text(encoding="utf-8")
    for raw in (database_raw, artifact_raw):
        assert marker_a not in raw
        assert marker_b not in raw
        payload = json.loads(raw)
        assert payload["payment"]["error"] == "Cookie: [REDACTED]"
        assert payload["errors"][-1]["message"] == "Cookie: [REDACTED]"

    legacy_payment = result.payment.model_copy(update={"error": unsafe_error}, deep=True)
    legacy_error = result.errors[-1].model_copy(update={"message": unsafe_error}, deep=True)
    legacy = result.model_copy(
        update={"payment": legacy_payment, "errors": [legacy_error]}, deep=True
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (legacy.model_dump_json(), case_id),
        )
        connection.commit()

    loaded = store.load_result(case_id)
    assert loaded is not None and loaded.payment is not None
    assert loaded.payment.error == "Cookie: [REDACTED]"
    assert loaded.errors[-1].message == "Cookie: [REDACTED]"
    assert loaded.payment.model_dump(exclude={"error"}) == original_payment_facts

    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        response = client.get(f"/cases/{case_id}")
    assert response.status_code == 200
    assert "Cookie: [REDACTED]" in response.text
    assert marker_a not in response.text and marker_b not in response.text

    output = io.StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=160),
    )
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    cli.case_status(case_id)
    rendered = output.getvalue()
    assert "[REDACTED]" in rendered
    assert marker_a not in rendered and marker_b not in rendered
