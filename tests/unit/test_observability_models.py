"""Secret redaction, normalized tool events, and strict output validation."""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_core import FunctionCall
from autogen_core.models import FunctionExecutionResult
from pydantic import ValidationError

from invoice_agents import orchestration
from invoice_agents.db.core import connect_database
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.models import DecisionKind, FinalDecision, UsageSummary
from invoice_agents.observability import audit as audit_module
from invoice_agents.observability.audit import (
    AuditRecorder,
    ProviderRetryAuditHandler,
    RedactingFilter,
    bind_audit_recorder,
    redact,
)
from invoice_agents.ui import queries


class _SecretRepresentation:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"opaque cookie={self.value}"


def _sanitize_text(value: str) -> str:
    sanitizer = getattr(audit_module, "sanitize_text", None)
    assert callable(sanitizer), "Task 13 requires the public sanitize_text interface"
    return sanitizer(value)


def _record_stream_event(event: object, context: Any, usage: UsageSummary) -> None:
    recorder = getattr(orchestration, "_record_stream_event", None)
    assert callable(recorder), "Task 13 requires the _record_stream_event interface"
    recorder(event, context, usage)


def _event_rows(workflow_db: Any) -> list[Any]:
    with connect_database(workflow_db, read_only=True) as connection:
        return connection.execute(
            "SELECT event_type, tool_call_id, provider_request_id, payload_json "
            "FROM events ORDER BY rowid"
        ).fetchall()


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


@pytest.mark.parametrize(
    ("value", "secret"),
    [
        ("upstream said Bearer bearer-secret-value", "bearer-secret-value"),
        ("xAI rejected xai-abcdefgh_12345678", "xai-abcdefgh_12345678"),
        ("provider rejected sk-abcdefgh_12345678", "sk-abcdefgh_12345678"),
        ("provider rejected sk-proj-abcdefgh_12345678", "sk-proj-abcdefgh_12345678"),
        ("request api_key=plainCredentialValue", "plainCredentialValue"),
        ("request CoOkIe: session=plainCookieValue", "session=plainCookieValue"),
        ("request AUTHORIZATION=BasicCredentialValue", "BasicCredentialValue"),
        (
            "request aUtHoRiZaTiOn:\nBasic basicSchemeCredentialValue",
            "basicSchemeCredentialValue",
        ),
    ],
)
def test_sanitize_text_redacts_embedded_credential_patterns(value: str, secret: str) -> None:
    cleaned = _sanitize_text(value)
    assert "[REDACTED]" in cleaned
    assert secret not in cleaned


def test_sanitize_text_is_bounded_control_safe_and_preserves_ordinary_invoice_text() -> None:
    ordinary = "Invoice sk-123 for Whiskey Skate Co.; authorization fee; cookie cutters"
    assert _sanitize_text(ordinary) == ordinary
    assert _sanitize_text("") == ""
    assert redact(None) is None

    split_secret = "s\x00k-proj-\nabcdefgh_12345678"
    cleaned = _sanitize_text(f"provider exception: {split_secret}")
    assert "[REDACTED]" in cleaned
    assert "abcdefgh_12345678" not in cleaned

    long_value = "api_key=plainCredentialValue " + ("invoice-context-" * 10_000)
    bounded = _sanitize_text(long_value)
    assert bounded == _sanitize_text(long_value), "bounding must be deterministic"
    assert len(bounded) <= 4096
    assert "plainCredentialValue" not in bounded
    assert "[REDACTED]" in bounded


def test_round1_sanitizer_redacts_complete_cookie_headers_and_neutralizes_controls() -> None:
    value = (
        "Invoice 1042\r\n"
        "\tOrdinary multiline description\r\n"
        "  cOoKiE: session=round1-cookie-a; preference=round1-cookie-b\r\n"
        "\tSeT-CoOkIe: first=round1-set-cookie-a; Path=/, "
        "second=round1-set-cookie-b; Secure\r"
        "ANSI \x1b[31mround1-color-marker\x1b[0m NUL\x00 ESC\x1b "
        "C1\x85 zero\u200bwidth"
    )

    cleaned = _sanitize_text(value)

    assert cleaned.startswith("Invoice 1042\n\tOrdinary multiline description\n")
    assert "cOoKiE: [REDACTED]" in cleaned
    assert "SeT-CoOkIe: [REDACTED]" in cleaned
    for marker in (
        "round1-cookie-a",
        "round1-cookie-b",
        "round1-set-cookie-a",
        "round1-set-cookie-b",
    ):
        assert marker not in cleaned
    for unsafe in ("\r", "\x00", "\x1b", "\x85", "\u200b", "[31m", "[0m"):
        assert unsafe not in cleaned
    assert "round1-color-marker" in cleaned
    assert _sanitize_text("Line one\r\n\tLine two\nLine three") == (
        "Line one\n\tLine two\nLine three"
    )
    assert redact({"prompt_tokens": 47, "invoice_text": "blue cookie cutters"}) == {
        "prompt_tokens": 47,
        "invoice_text": "blue cookie cutters",
    }
    assert redact({"api\u200b_key": "round1-control-key-marker"}) == {"api_key": "[REDACTED]"}


def test_recursive_redaction_sanitizes_bytes_objects_and_not_usage_or_invoice_data() -> None:
    xai_secret = "xai-abcdefgh_12345678"
    cookie_secret = "plainCookieCredential"
    cleaned = redact(
        {
            "safe_invoice": {
                "description": "sk-123 blue cookie cutters",
                "prompt_tokens": 47,
            },
            "nested": [{"XAI_API_KEY": xai_secret}],
            "bytes": f"api_key={cookie_secret}".encode(),
            "opaque": _SecretRepresentation(cookie_secret),
        }
    )
    encoded = json.dumps(cleaned, default=str, sort_keys=True)
    assert cleaned["safe_invoice"] == {
        "description": "sk-123 blue cookie cutters",
        "prompt_tokens": 47,
    }
    assert cleaned["nested"][0]["XAI_API_KEY"] == "[REDACTED]"
    assert xai_secret not in encoded
    assert cookie_secret not in encoded
    assert encoded.count("[REDACTED]") >= 3


def test_logging_filter_sanitizes_formatted_arguments_and_exception_chain() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("invoice_agents.tests.task13")
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    try:
        try:
            try:
                raise ValueError("Authorization: nestedCredentialValue")
            except ValueError as cause:
                raise RuntimeError("cookie=outerCredentialValue") from cause
        except RuntimeError:
            logger.exception("retry api_key=%s", "argumentCredentialValue")
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    output = stream.getvalue()
    assert "[REDACTED]" in output
    for secret in (
        "nestedCredentialValue",
        "outerCredentialValue",
        "argumentCredentialValue",
    ):
        assert secret not in output


def test_audit_rows_sanitize_nested_representation_boundaries_and_provider_ids(
    workflow_db: Any,
) -> None:
    secret = "plainAuditCredential"
    recorder = AuditRecorder(workflow_db, "case_task13_audit")
    recorder.record(
        "test.sensitive",
        {
            "safe_invoice_number": "INV-sk-123",
            "nested": [{"message": f"api_key={secret}"}],
            "bytes": f"Cookie: session={secret}".encode(),
            "opaque": _SecretRepresentation(secret),
        },
        provider_request_id=f"Authorization:{secret}",
    )
    recorder.record(
        "test.safe-request-id",
        {"status": "failed"},
        provider_request_id="req_safe.123:abc",
    )

    rows = _event_rows(workflow_db)
    sensitive_payload = rows[0]["payload_json"]
    assert "[REDACTED]" in sensitive_payload
    assert secret not in sensitive_payload
    assert "INV-sk-123" in sensitive_payload
    assert rows[0]["provider_request_id"] is None
    assert rows[1]["provider_request_id"] == "req_safe.123:abc"


def test_provider_retry_event_persists_sanitized_formatted_message(workflow_db: Any) -> None:
    secret = "plainRetryCredential"
    recorder = AuditRecorder(workflow_db, "case_task13_retry")
    logger = logging.getLogger("openai.tests.task13")
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers = [ProviderRetryAuditHandler()]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        with bind_audit_recorder(recorder):
            logger.info("Retrying provider request api_key=%s", secret)
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    rows = _event_rows(workflow_db)
    assert len(rows) == 1
    assert rows[0]["event_type"] == "provider.retry"
    assert "[REDACTED]" in rows[0]["payload_json"]
    assert secret not in rows[0]["payload_json"]


def test_stream_tool_events_persist_one_safe_correlated_row_per_item(
    workflow_db: Any,
) -> None:
    secret = "plainToolArgumentCredential"
    context = SimpleNamespace(
        case_id="case_task13_calls",
        audit=AuditRecorder(workflow_db, "case_task13_calls"),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_task13")),
    )
    usage = UsageSummary()
    request = ToolCallRequestEvent(
        source="coordinator",
        metadata={"request_id": "req_task13_safe"},
        content=[
            FunctionCall(
                id="call/vendor=1",
                name="lookup_first",
                arguments=json.dumps({"invoice_number": "INV-42", "api_key": secret}),
            ),
            FunctionCall(
                id="call_two",
                name="lookup_second",
                arguments=json.dumps({"invoice_number": "INV-43"}),
            ),
        ],
    )
    execution = ToolCallExecutionEvent(
        source="coordinator",
        metadata={"request_id": "req_task13_safe"},
        content=[
            FunctionExecutionResult(
                call_id="call_two",
                name="lookup_second",
                content="inventory unavailable Authorization: secondCredentialValue",
                is_error=True,
            ),
            FunctionExecutionResult(
                call_id="call/vendor=1",
                name="lookup_first",
                content="INV-42 matched",
                is_error=False,
            ),
        ],
    )

    _record_stream_event(request, context, usage)
    _record_stream_event(execution, context, usage)

    rows = _event_rows(workflow_db)
    assert [row["tool_call_id"] for row in rows] == [
        "call/vendor=1",
        "call_two",
        "call_two",
        "call/vendor=1",
    ]
    assert all(row["provider_request_id"] == "req_task13_safe" for row in rows)
    assert usage.tool_calls == 2, "execution events must not double-count tool requests"
    assert len(context.tool_failures) == 1
    assert "[REDACTED]" in context.tool_failures[0]
    assert "secondCredentialValue" not in context.tool_failures[0]

    payloads = [json.loads(row["payload_json"]) for row in rows]
    assert all(len(payload["content"]) == 1 for payload in payloads)
    assert [payload["content"][0]["name"] for payload in payloads] == [
        "lookup_first",
        "lookup_second",
        "lookup_second",
        "lookup_first",
    ]
    encoded = json.dumps(payloads)
    assert secret not in encoded
    assert "secondCredentialValue" not in encoded
    assert "INV-42" in encoded and "INV-43" in encoded


def test_round1_tool_json_strings_and_legacy_rows_are_structurally_redacted(
    workflow_db: Any,
) -> None:
    context = SimpleNamespace(
        case_id="case_round1_json_fields",
        audit=AuditRecorder(workflow_db, "case_round1_json_fields"),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_round1_json")),
    )
    usage = UsageSummary()
    request = ToolCallRequestEvent(
        source="coordinator",
        content=[
            FunctionCall(
                id="call_round1_valid",
                name="lookup_valid",
                arguments=json.dumps(
                    {
                        "invoice_number": "INV-1042",
                        "nested": {"Authorization": "round1-argument-marker"},
                        "prompt_tokens": 47,
                    }
                ),
            ),
            FunctionCall(
                id="call_round1_malformed",
                name="lookup_malformed",
                arguments='{"api_key":"round1-malformed-marker"',
            ),
        ],
    )
    execution = ToolCallExecutionEvent(
        source="coordinator",
        content=[
            FunctionExecutionResult(
                call_id="call_round1_valid",
                name="lookup_valid",
                content=json.dumps(
                    {
                        "count": 3,
                        "nested": {"Cookie": "round1-result-marker"},
                        "safe_id": "row_1042",
                    }
                ),
                is_error=False,
            )
        ],
    )

    _record_stream_event(request, context, usage)
    _record_stream_event(execution, context, usage)

    payloads = [json.loads(row["payload_json"]) for row in _event_rows(workflow_db)]
    valid_arguments = payloads[0]["content"][0]["arguments"]
    assert valid_arguments == (
        '{"invoice_number":"INV-1042","nested":{"Authorization":"[REDACTED]"},"prompt_tokens":47}'
    )
    malformed_arguments = payloads[1]["content"][0]["arguments"]
    assert isinstance(malformed_arguments, str)
    assert len(malformed_arguments) <= 4096
    assert "round1-malformed-marker" not in malformed_arguments
    result_content = payloads[2]["content"][0]["content"]
    assert result_content == ('{"count":3,"nested":{"Cookie":"[REDACTED]"},"safe_id":"row_1042"}')

    legacy_payload = json.dumps(
        {
            "type": "ToolCallRequestEvent",
            "source": "legacy_agent",
            "metadata": {"headers": "Cookie: round1-legacy-header-marker"},
            "content": [
                {
                    "id": "call_round1_legacy",
                    "name": "legacy_lookup",
                    "arguments": json.dumps(
                        {
                            "invoice_number": "INV-LEGACY",
                            "nested": {"api_key": "round1-legacy-json-marker"},
                        }
                    ),
                }
            ],
        }
    )
    with connect_database(workflow_db) as connection:
        connection.execute(
            "INSERT INTO events(event_id, case_id, source_id, event_type, payload_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "evt_round1_legacy",
                "case_round1_json_fields",
                "src_round1_json",
                "autogen.ToolCallRequestEvent",
                legacy_payload,
                "2026-08-10T00:00:00+00:00",
            ),
        )
        connection.commit()
    presented = queries.events_after(workflow_db, "case_round1_json_fields")[-1]
    assert "round1-legacy-header-marker" not in presented.payload_json
    assert "round1-legacy-json-marker" not in presented.payload_json
    presented_payload = json.loads(presented.payload_json)
    assert set(presented_payload) == {"content", "source", "type"}
    assert json.loads(presented_payload["content"][0]["arguments"])["nested"] == {
        "api_key": "[REDACTED]"
    }


def test_round1_stream_rows_whitelist_tool_and_non_tool_fields(workflow_db: Any) -> None:
    context = SimpleNamespace(
        case_id="case_round1_whitelist",
        audit=AuditRecorder(workflow_db, "case_round1_whitelist"),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_round1_whitelist")),
    )
    usage = UsageSummary()
    tool_event = ToolCallRequestEvent(
        source="coordinator",
        models_usage={"prompt_tokens": 11, "completion_tokens": 7},
        metadata={
            "request_id": "req_round1_safe",
            "headers": "Cookie: round1-tool-header-marker",
            "raw_body": "round1-tool-body-marker",
        },
        content=[
            FunctionCall(
                id="call_round1_safe",
                name="lookup_invoice",
                arguments='{"invoice_number":"INV-1042"}',
            )
        ],
    )
    text_event = TextMessage(
        content="ordinary model summary",
        source="coordinator",
        models_usage={"prompt_tokens": 13, "completion_tokens": 5},
        metadata={
            "request_id": "Cookie: round1-unsafe-request-id",
            "headers": "Set-Cookie: round1-text-header-marker",
            "raw_response": "round1-text-response-marker",
        },
    )

    _record_stream_event(tool_event, context, usage)
    _record_stream_event(text_event, context, usage)

    rows = _event_rows(workflow_db)
    tool_payload = json.loads(rows[0]["payload_json"])
    text_payload = json.loads(rows[1]["payload_json"])
    assert set(tool_payload) == {"content", "models_usage", "source", "type"}
    assert set(tool_payload["content"][0]) == {"arguments", "id", "name"}
    assert tool_payload["content"][0]["id"] == "call_round1_safe"
    assert tool_payload["models_usage"] == {"completion_tokens": 7, "prompt_tokens": 11}
    assert rows[0]["provider_request_id"] == "req_round1_safe"
    assert set(text_payload) == {"content", "models_usage", "source", "type"}
    assert text_payload["content"] == "ordinary model summary"
    assert text_payload["models_usage"] == {"completion_tokens": 5, "prompt_tokens": 13}
    assert rows[1]["provider_request_id"] is None
    assert usage.prompt_tokens == 24
    assert usage.completion_tokens == 12
    assert usage.model_calls == 2
    assert usage.tool_calls == 1
    encoded_rows = json.dumps([dict(row) for row in rows], sort_keys=True)
    for marker in (
        "round1-tool-header-marker",
        "round1-tool-body-marker",
        "round1-unsafe-request-id",
        "round1-text-header-marker",
        "round1-text-response-marker",
    ):
        assert marker not in encoded_rows


def test_stream_tool_events_preserve_duplicate_items_without_deduplicating_usage(
    workflow_db: Any,
) -> None:
    context = SimpleNamespace(
        case_id="case_task13_duplicates",
        audit=AuditRecorder(workflow_db, "case_task13_duplicates"),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_task13")),
    )
    usage = UsageSummary()
    request = ToolCallRequestEvent(
        source="coordinator",
        content=[
            FunctionCall(id="call_duplicate", name="first", arguments="{}"),
            FunctionCall(id="call_duplicate", name="second", arguments="{}"),
        ],
    )

    _record_stream_event(request, context, usage)

    rows = _event_rows(workflow_db)
    assert [row["tool_call_id"] for row in rows] == ["call_duplicate", "call_duplicate"]
    assert [json.loads(row["payload_json"])["content"][0]["name"] for row in rows] == [
        "first",
        "second",
    ]
    assert usage.tool_calls == 2


@pytest.mark.parametrize(
    ("event", "expected_tool_calls"),
    [
        (
            ToolCallRequestEvent(
                source="coordinator",
                content=[
                    FunctionCall(id="call_valid", name="valid", arguments="{}"),
                    FunctionCall(id="", name="missing", arguments="{}"),
                ],
            ),
            2,
        ),
        (
            ToolCallExecutionEvent(
                source="coordinator",
                content=[
                    FunctionExecutionResult(
                        call_id="", name="missing", content="failed", is_error=True
                    )
                ],
            ),
            0,
        ),
    ],
)
def test_stream_tool_events_fail_closed_before_writing_missing_correlation_ids(
    workflow_db: Any, event: object, expected_tool_calls: int
) -> None:
    context = SimpleNamespace(
        case_id="case_task13_missing",
        audit=AuditRecorder(workflow_db, "case_task13_missing"),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_task13")),
    )
    usage = UsageSummary()

    with pytest.raises(InvoiceAgentsError) as failure:
        _record_stream_event(event, context, usage)

    assert failure.value.stop_reason == "TOOL_CALL_ID_MISSING"
    assert _event_rows(workflow_db) == []
    assert usage.tool_calls == expected_tool_calls
    assert context.tool_failures == []


def test_stream_tool_event_rejects_credential_shaped_control_id_before_persistence(
    workflow_db: Any,
) -> None:
    context = SimpleNamespace(
        case_id="case_task13_invalid",
        audit=AuditRecorder(workflow_db, "case_task13_invalid"),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_task13")),
    )
    usage = UsageSummary()
    event = ToolCallRequestEvent(
        source="coordinator",
        content=[
            FunctionCall(
                id="call_\nxai-abcdefgh_12345678",
                name="invalid",
                arguments="{}",
            )
        ],
    )

    with pytest.raises(InvoiceAgentsError) as failure:
        _record_stream_event(event, context, usage)

    assert failure.value.stop_reason == "TOOL_CALL_ID_INVALID"
    assert _event_rows(workflow_db) == []
    assert usage.tool_calls == 1


def test_stream_non_tool_event_is_persisted_once_without_tool_correlation(
    workflow_db: Any,
) -> None:
    context = SimpleNamespace(
        case_id="case_task13_text",
        audit=AuditRecorder(workflow_db, "case_task13_text"),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_task13")),
    )
    usage = UsageSummary()

    _record_stream_event(
        TextMessage(content="ordinary invoice status", source="coordinator"), context, usage
    )

    rows = _event_rows(workflow_db)
    assert len(rows) == 1
    assert rows[0]["event_type"] == "autogen.TextMessage"
    assert rows[0]["tool_call_id"] is None
    assert usage.tool_calls == 0


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
