"""Round-two regressions for fail-closed observability sanitization."""

from __future__ import annotations

import io
import json
import logging
import math

import pytest

from invoice_agents import orchestration
from invoice_agents.db.core import connect_database
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.observability import audit as audit_module
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
REJECTED_VALUE = "[VALUE_REJECTED]"


class _ExplodingRepresentation:
    def __str__(self) -> str:
        raise RuntimeError("round4 representation canary")


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


@pytest.mark.parametrize("mark", ["\u0301", "\u0903", "\u20dd"])
def test_combining_marks_cannot_split_credential_markers_or_sensitive_keys(mark: str) -> None:
    token_marker = "abcdefgh_12345678"
    text = sanitize_text(f"provider rejected s{mark}k-{token_marker}")

    assert text == "provider rejected [REDACTED]"
    assert token_marker not in text
    assert redact({f"api{mark}_key": "round3-map-canary"}) == {f"api{mark}_key": "[REDACTED]"}
    with pytest.raises(InvoiceAgentsError) as failure:
        orchestration._validated_tool_call_ids([f"call_s{mark}k-{token_marker}"], "case_round3")
    assert failure.value.stop_reason == "TOOL_CALL_ID_INVALID"


def test_control_split_and_complete_quoted_credential_values_are_redacted() -> None:
    markers = (
        "round3-control-canary",
        "round3 double quoted canary",
        "round3 single quoted canary",
        "round3 unterminated canary",
    )
    cleaned = sanitize_text(
        f"api\n_key={markers[0]}\n"
        f'api_key="{markers[1]}"\n'
        f"client_secret='{markers[2]}'\n"
        f'auth_token="{markers[3]}'
    )

    for marker in markers:
        assert marker not in cleaned
    assert cleaned.count("[REDACTED]") == len(markers)


def test_unrelated_decomposed_unicode_is_preserved_exactly() -> None:
    ordinary = "Résumé हिंदी العربية 日本語 cafe\u0301"

    assert sanitize_text(ordinary) == ordinary


@pytest.mark.parametrize(
    "raw",
    [
        '{"safe":1e309}',
        '{"safe":-0}',
        '{"safe":1e0}',
        ' {"safe":1}',
        '{"safe":1} ',
    ],
)
def test_nonfinite_or_noncanonical_json_numbers_and_outer_space_are_opaque(raw: str) -> None:
    assert sanitize_json_text(raw) == REJECTED_JSON


def test_valid_finite_json_numbers_remain_structured() -> None:
    assert json.loads(sanitize_json_text('{"count":1,"ratio":1.5}')) == {
        "count": 1,
        "ratio": 1.5,
    }


def test_overdeep_json_is_rejected_before_calling_json_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '{"safe":' + ("[" * 21) + "0" + ("]" * 21) + "}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("overdeep JSON reached json.loads")

    monkeypatch.setattr(audit_module.json, "loads", forbidden_loads)

    assert sanitize_json_text(raw) == REJECTED_JSON


def test_nonfinite_python_values_never_serialize_as_json_constants(workflow_db: object) -> None:
    assert redact({"safe": math.inf}) == {"safe": "[VALUE_REJECTED]"}

    recorder = AuditRecorder(workflow_db, "case_round3_nonfinite")
    recorder.record("autogen.ToolCallRequestEvent", {"safe": math.inf})

    with connect_database(workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT payload_json FROM events WHERE case_id = ?",
            ("case_round3_nonfinite",),
        ).fetchone()
    assert row is not None
    assert json.loads(row["payload_json"]) == {
        "error": REJECTED_EVENT,
        "type": "ToolCallRequestEvent",
    }
    assert "Infinity" not in row["payload_json"]


@pytest.mark.parametrize(
    "credential",
    [
        "ápi_key=round4-nfc-key-canary",
        "provider rejected sk-abcd efgh_12345678",
        "provider rejected sk-abcd\u0435fgh_12345678",
    ],
)
def test_normalized_whitespace_and_confusable_credentials_are_redacted(
    credential: str,
) -> None:
    cleaned = sanitize_text(credential)

    assert cleaned.count("[REDACTED]") == 1
    assert "round4" not in cleaned
    assert "abcd" not in cleaned


def test_nfc_equivalent_sensitive_mapping_key_redacts_without_changing_the_key() -> None:
    assert redact({"ápi_key": "round4-nfc-map-canary", "prompt_tokens": 47}) == {
        "ápi_key": "[REDACTED]",
        "prompt_tokens": 47,
    }


@pytest.mark.parametrize(
    "tool_call_id",
    [
        "call_sk-abcd efgh_12345678",
        "call_sk-abcd\u0435fgh_12345678",
    ],
)
def test_split_and_confusable_credential_tool_ids_are_rejected_everywhere(
    workflow_db: object,
    tool_call_id: str,
) -> None:
    with pytest.raises(InvoiceAgentsError) as failure:
        orchestration._validated_tool_call_ids([tool_call_id], "case_round4")
    assert failure.value.stop_reason == "TOOL_CALL_ID_INVALID"

    recorder = AuditRecorder(workflow_db, "case_round4_raw_tool_id")
    with pytest.raises(ValueError, match="tool call"):
        recorder.record("test.raw-tool-id", {"safe": True}, tool_call_id=tool_call_id)
    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT tool_call_id FROM events WHERE case_id = ?",
            ("case_round4_raw_tool_id",),
        ).fetchall()
    assert rows == []


def test_quoted_credentials_span_lines_and_escapes_without_deleting_evidence() -> None:
    marker_a = "round4-multiline-a"
    marker_b = "round4-multiline-b"
    cleaned = sanitize_text(
        f'api_key="{marker_a}\ncontinued \\"quoted\\" {marker_b}" prompt_tokens=47 token_count=59'
    )

    assert cleaned == 'api_key="[REDACTED]" prompt_tokens=47 token_count=59'
    assert marker_a not in cleaned
    assert marker_b not in cleaned


def test_unquoted_credential_stops_at_lexical_whitespace_and_preserves_evidence() -> None:
    cleaned = sanitize_text("api_key=round4-unquoted-canary prompt_tokens=47 token_count=59")

    assert cleaned == "api_key=[REDACTED] prompt_tokens=47 token_count=59"


def test_sanitizer_bounds_input_work_without_leaking_a_boundary_spanning_secret() -> None:
    prefix = "invoice context " * 245
    value = prefix + ' api_key="round4-boundary-' + ("secret" * 20_000) + '" trailing'

    cleaned = sanitize_text(value)

    assert len(cleaned) <= 4_096
    assert cleaned.endswith("…[TRUNCATED]")
    assert "round4-boundary" not in cleaned
    assert cleaned == sanitize_text(value)


@pytest.mark.parametrize(
    "raw",
    [
        '{"bad":"\\ud800"}',
        '{"bad":"\\udc00"}',
        '{"bad":"\ud800"}',
    ],
)
def test_unpaired_json_surrogates_are_rejected_opaquely(raw: str) -> None:
    assert sanitize_json_text(raw) == REJECTED_JSON


def test_paired_json_surrogates_remain_valid_unicode() -> None:
    assert json.loads(sanitize_json_text('{"emoji":"\\ud83d\\ude00"}')) == {"emoji": "😀"}


@pytest.mark.parametrize(
    "raw",
    [
        '{"huge":' + ("9" * 4_000) + "}",
        '{"huge":1e' + ("9" * 1_000) + "}",
    ],
)
def test_long_numeric_lexemes_are_rejected_before_integer_or_float_conversion(raw: str) -> None:
    assert sanitize_json_text(raw) == REJECTED_JSON


def test_python_object_budget_rejects_huge_integers_cycles_and_serialization_failures(
    workflow_db: object,
) -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    huge_integer = 10**5_000
    recorder = AuditRecorder(workflow_db, "case_round4_object_budget")

    assert redact({"huge": huge_integer}) == REJECTED_VALUE
    assert redact(cyclic) == REJECTED_VALUE
    assert redact({"opaque": _ExplodingRepresentation()}) == REJECTED_VALUE
    for payload in ({"huge": huge_integer}, cyclic, {"opaque": _ExplodingRepresentation()}):
        recorder.record("test.invalid-python-payload", payload)

    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM events WHERE case_id = ? ORDER BY rowid",
            ("case_round4_object_budget",),
        ).fetchall()
    assert [json.loads(row["payload_json"]) for row in rows] == [
        REJECTED_VALUE,
        REJECTED_VALUE,
        REJECTED_VALUE,
    ]


def test_python_object_budget_allows_shared_acyclic_references() -> None:
    shared = {"invoice_number": "INV-42", "prompt_tokens": 47}

    assert redact({"first": shared, "second": shared}) == {
        "first": {"invoice_number": "INV-42", "prompt_tokens": 47},
        "second": {"invoice_number": "INV-42", "prompt_tokens": 47},
    }
