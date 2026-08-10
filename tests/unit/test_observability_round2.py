"""Round-two regressions for fail-closed observability sanitization."""

from __future__ import annotations

import io
import json
import logging

import pytest

from invoice_agents import orchestration
from invoice_agents.db.core import connect_database
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.observability.audit import (
    AuditRecorder,
    RedactingFilter,
    redact,
    sanitize_json_text,
    sanitize_stored_event_payload,
    sanitize_text,
)

DEFAULT_IGNORABLES = (
    "\u2061",
    "\u2062",
    "\u2063",
    "\u2064",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
    "\u180e",
    "\u061c",
    "\u202e",
    "\u034f",
)
REJECTED_JSON = '{"error":"JSON_PAYLOAD_REJECTED","original":"[REDACTED]"}'
REJECTED_EVENT = "EVENT_PAYLOAD_REJECTED"


@pytest.mark.parametrize("invisible", DEFAULT_IGNORABLES)
def test_default_ignorables_cannot_split_credentials_or_tool_correlation_ids(
    invisible: str,
) -> None:
    marker = f"sk-{invisible}abcdefgh_12345678"

    cleaned = sanitize_text(f"provider rejected {marker}")

    assert cleaned == "provider rejected [REDACTED]"
    with pytest.raises(InvoiceAgentsError) as failure:
        orchestration._validated_tool_call_ids([f"call_{marker}"], "case_round2")
    assert failure.value.stop_reason == "TOOL_CALL_ID_INVALID"


def test_text_and_mapping_redaction_use_complete_values_and_exact_credential_keys() -> None:
    cookie_a = "round2-cookie-primary"
    cookie_b = "round2-cookie-continuation"
    folded_auth = "round2-folded-authorization"
    cleaned = sanitize_text(
        f"cookie=session={cookie_a}; preference={cookie_b}\r\n"
        f"Authorization: Bearer abc\r{folded_auth}"
    )

    assert cleaned == "cookie=[REDACTED]\nAuthorization: [REDACTED]"
    for marker in (cookie_a, cookie_b, folded_auth):
        assert marker not in cleaned

    ordinary = "Résumé हिंदी العربية 日本語 cafe\u0301"
    values = {
        "prompt_tokens": 47,
        "completion_tokens": 12,
        "token_count": 59,
        "secretary": "Ada",
        "authorization_fee": "15.00",
        "ordinary": ordinary,
        "api\u2061_key": "round2-map-credential",
    }
    assert redact(values) == {
        "prompt_tokens": 47,
        "completion_tokens": 12,
        "token_count": 59,
        "secretary": "Ada",
        "authorization_fee": "15.00",
        "ordinary": ordinary,
        "api_key": "[REDACTED]",
    }


def test_exception_logging_and_audit_rows_apply_the_same_complete_redaction(
    workflow_db: object,
) -> None:
    cookie_a = "round2-log-cookie-a"
    cookie_b = "round2-log-cookie-b"
    split_key = "sk-abcd\u2063efgh_12345678"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("invoice_agents.tests.observability_round2")
    previous = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers = [handler]
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    try:
        try:
            raise ValueError(f"cookie=session={cookie_a}; preference={cookie_b}")
        except ValueError as cause:
            try:
                raise RuntimeError(f"provider rejected {split_key}") from cause
            except RuntimeError:
                logger.exception("request failed")
    finally:
        logger.handlers, logger.level, logger.propagate = previous

    recorder = AuditRecorder(workflow_db, "case_round2_audit")
    recorder.record(
        "provider.failure",
        {
            "message": f"cookie=session={cookie_a}; preference={cookie_b}",
            "exception": f"provider rejected {split_key}",
            "prompt_tokens": 47,
            "authorization_fee": "15.00",
        },
    )
    with connect_database(workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT payload_json FROM events WHERE case_id = ?",
            ("case_round2_audit",),
        ).fetchone()
    assert row is not None
    combined = f"{stream.getvalue()} {row['payload_json']}"
    for marker in (cookie_a, cookie_b, "abcdefgh_12345678"):
        assert marker not in combined
    payload = json.loads(row["payload_json"])
    assert payload["prompt_tokens"] == 47
    assert payload["authorization_fee"] == "15.00"


@pytest.mark.parametrize(
    "raw",
    [
        '{"api\\u005fkey":"round2-duplicate-a","api_key":"round2-duplicate-b"}',
        '{"api_key" : "round2 unterminated marker',
        '["round2-list-marker"]',
        '"round2-string-marker"',
        json.dumps({"nested": [[[[[[[[[[[[[[[[[[[[[["round2-depth-marker"]]]]]]]]]]]]]]]]]]]]]]}),
        json.dumps({"safe": "x" * 65_537}),
    ],
)
def test_tool_json_parse_shape_depth_and_size_fail_to_one_opaque_record(raw: str) -> None:
    cleaned = sanitize_json_text(raw)

    assert cleaned == REJECTED_JSON
    assert "round2" not in cleaned


def test_valid_bounded_tool_json_is_structurally_sanitized_without_losing_usage() -> None:
    raw = json.dumps(
        {
            "api\u2062_key": "round2-json-credential",
            "prompt_tokens": 47,
            "token_count": 59,
            "authorization_fee": "15.00",
        }
    )

    assert json.loads(sanitize_json_text(raw)) == {
        "api_key": "[REDACTED]",
        "prompt_tokens": 47,
        "token_count": 59,
        "authorization_fee": "15.00",
    }


@pytest.mark.parametrize(
    "raw",
    [
        (
            '{"type":"ToolCallRequestEvent","response\\u005fbody":"round2-body-a",'
            '"response_body":"round2-body-b"}'
        ),
        '[{"response_body":"round2-list-body"}]',
        '"round2-string-body"',
        '{"content":[[[[[[[[[[[[[[[[[[[[[["round2-deep-body"]]]]]]]]]]]]]]]]]]]]]]}',
    ],
)
def test_legacy_autogen_invalid_payloads_emit_safe_minimal_unavailable_event(raw: str) -> None:
    cleaned = sanitize_stored_event_payload("autogen.ToolCallRequestEvent", raw)
    payload = json.loads(cleaned)

    assert payload == {
        "error": REJECTED_EVENT,
        "type": "ToolCallRequestEvent",
    }
    assert "round2" not in cleaned


def test_live_autogen_wrong_shape_is_not_copied_to_an_audit_row(workflow_db: object) -> None:
    recorder = AuditRecorder(workflow_db, "case_round2_wrong_shape")

    recorder.record("autogen.ToolCallRequestEvent", ["round2-live-body"])

    with connect_database(workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT payload_json FROM events WHERE case_id = ?",
            ("case_round2_wrong_shape",),
        ).fetchone()
    assert row is not None
    assert json.loads(row["payload_json"]) == {
        "error": REJECTED_EVENT,
        "type": "ToolCallRequestEvent",
    }
    assert "round2-live-body" not in row["payload_json"]
