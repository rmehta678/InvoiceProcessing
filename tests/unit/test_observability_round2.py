"""Round-two regressions for fail-closed observability sanitization."""

from __future__ import annotations

import io
import json
import logging
import math
from types import SimpleNamespace

import pytest
from autogen_agentchat.messages import ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_core import FunctionCall
from autogen_core.models import FunctionExecutionResult

from invoice_agents import orchestration
from invoice_agents.db.core import connect_database
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.models import UsageSummary
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
FULLY_CYRILLIC_API_KEY = "\u0430\u0440\u0456_\u043a\u0435\u0443"
FULLY_CYRILLIC_SK = "\u0455\u043a"
FULLY_CYRILLIC_XAI = "\u0445\u0430\u0456"


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


@pytest.mark.parametrize(
    ("credential", "expected"),
    [
        ("provider sκ-abcdefgh_12345678", "provider [REDACTED]"),
        ("provider ѕκ-abcdefgh_12345678", "provider [REDACTED]"),
        ("\u03b1\u03c1i_key=round5-canary", "\u03b1\u03c1i_key=[REDACTED]"),
        ("api_κey=round5-canary", "api_κey=[REDACTED]"),
        ("t\u03bfken=round5-canary", "t\u03bfken=[REDACTED]"),
    ],
)
def test_unknown_mixed_script_letters_cannot_evade_credential_boundaries(
    credential: str,
    expected: str,
) -> None:
    assert sanitize_text(credential) == expected


@pytest.mark.parametrize(
    "credential_key",
    ["\u03b1\u03c1i_key", "api_κey", "t\u03bfken"],
)
def test_mixed_script_sensitive_mapping_keys_redact_without_changing_the_key(
    credential_key: str,
) -> None:
    assert redact({credential_key: "round5-map-canary", "prompt_tokens": 47}) == {
        credential_key: "[REDACTED]",
        "prompt_tokens": 47,
    }


@pytest.mark.parametrize(
    "tool_call_id",
    [
        "call_sκ-abcdefgh_12345678",
        "call_ѕκ-abcdefgh_12345678",
        "call_\u03b1\u03c1i_key=round5-canary",
        "call_api_κey=round5-canary",
        "call_t\u03bfken=round5-canary",
    ],
)
def test_mixed_script_credential_tool_ids_are_rejected_at_both_persistence_boundaries(
    workflow_db: object,
    tool_call_id: str,
) -> None:
    with pytest.raises(InvoiceAgentsError) as failure:
        orchestration._validated_tool_call_ids([tool_call_id], "case_round5")
    assert failure.value.stop_reason == "TOOL_CALL_ID_INVALID"

    recorder = AuditRecorder(workflow_db, "case_round5_raw_tool_id")
    with pytest.raises(ValueError, match="tool call"):
        recorder.record("test.raw-tool-id", {"safe": True}, tool_call_id=tool_call_id)
    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT tool_call_id FROM events WHERE case_id = ?",
            ("case_round5_raw_tool_id",),
        ).fetchall()
    assert rows == []


def test_mixed_script_credentials_are_removed_from_exception_logs_and_audit_rows(
    workflow_db: object,
) -> None:
    marker = "round5-exception-audit-canary"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("invoice_agents.tests.observability_round5")
    previous = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers = [handler]
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    try:
        try:
            raise RuntimeError(f"api_κey={marker}")
        except RuntimeError:
            logger.exception("mixed-script provider failure")
    finally:
        logger.handlers, logger.level, logger.propagate = previous

    recorder = AuditRecorder(workflow_db, "case_round5_audit")
    recorder.record(
        "provider.failure",
        {
            "message": f"provider sκ-abcdefgh_{marker}",
            "prompt_tokens": 47,
        },
    )
    with connect_database(workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT payload_json FROM events WHERE case_id = ?",
            ("case_round5_audit",),
        ).fetchone()
    assert row is not None
    combined = f"{stream.getvalue()} {row['payload_json']}"
    assert marker not in combined
    assert "abcdefgh" not in combined
    assert "[REDACTED]" in combined
    assert json.loads(row["payload_json"])["prompt_tokens"] == 47


@pytest.mark.parametrize(
    "ordinary",
    [
        "résumé=Alice",
        "κόστος=15",
        "日本語=ordinary",
        "المدينة=Chicago",
    ],
)
def test_unrelated_unicode_assignments_are_preserved_without_false_redaction(
    ordinary: str,
) -> None:
    assert sanitize_text(ordinary) == ordinary
    assert redact({ordinary.split("=", 1)[0]: ordinary.split("=", 1)[1]}) == {
        ordinary.split("=", 1)[0]: ordinary.split("=", 1)[1]
    }


@pytest.mark.parametrize(
    ("credential", "expected"),
    [
        ("provider σκ-abcdefgh_12345678", "provider [REDACTED]"),
        ("απι_κεγ=round6-secret", "απι_κεγ=[REDACTED]"),
    ],
)
def test_fully_substituted_spoof_script_credentials_fail_closed(
    credential: str,
    expected: str,
) -> None:
    assert sanitize_text(credential) == expected


def test_fully_substituted_sensitive_mapping_key_redacts_its_value() -> None:
    assert redact({"απι_κεγ": "round6-map-secret", "prompt_tokens": 47}) == {
        "απι_κεγ": "[REDACTED]",
        "prompt_tokens": 47,
    }


def test_fully_cyrillic_provider_and_key_credentials_are_redacted() -> None:
    assert (
        sanitize_text(
            f"providers {FULLY_CYRILLIC_SK}-abcdefgh_12345678 "
            f"and {FULLY_CYRILLIC_XAI}-abcdefgh_12345678"
        )
        == "providers [REDACTED] and [REDACTED]"
    )
    assert (
        sanitize_text(f"{FULLY_CYRILLIC_API_KEY}=round6-cyrillic-key-secret")
        == f"{FULLY_CYRILLIC_API_KEY}=[REDACTED]"
    )


def test_recursive_redaction_handles_fully_cyrillic_credentials_without_wildcards() -> None:
    ordinary_cyrillic = "смета_города"
    ordinary_japanese = "日本語_請求書"

    assert redact(
        {
            "nested": {
                FULLY_CYRILLIC_API_KEY: "round6-cyrillic-map-secret",
                "provider_error": f"provider {FULLY_CYRILLIC_SK}-abcdefgh_12345678",
                ordinary_cyrillic: "ordinary",
                ordinary_japanese: "ordinary",
            }
        }
    ) == {
        "nested": {
            FULLY_CYRILLIC_API_KEY: "[REDACTED]",
            "provider_error": "provider [REDACTED]",
            ordinary_cyrillic: "ordinary",
            ordinary_japanese: "ordinary",
        }
    }


def test_fully_cyrillic_credentials_cannot_persist_as_tool_request_or_execution_ids(
    workflow_db: object,
) -> None:
    request_case_id = "case_round6_cyrillic_request_id"
    request_context = SimpleNamespace(
        case_id=request_case_id,
        audit=AuditRecorder(workflow_db, request_case_id),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_round6_request")),
    )
    request = ToolCallRequestEvent(
        source="coordinator",
        content=[
            FunctionCall(
                id=f"call_{FULLY_CYRILLIC_API_KEY}=round6-request-id-secret",
                name="lookup_invoice",
                arguments='{"invoice_number":"INV-42"}',
            )
        ],
    )
    with pytest.raises(InvoiceAgentsError) as request_failure:
        orchestration._record_stream_event(request, request_context, UsageSummary())
    assert request_failure.value.stop_reason == "TOOL_CALL_ID_INVALID"

    execution_case_id = "case_round6_cyrillic_execution_id"
    execution_context = SimpleNamespace(
        case_id=execution_case_id,
        audit=AuditRecorder(workflow_db, execution_case_id),
        tool_failures=[],
        invoice=lambda: SimpleNamespace(source=SimpleNamespace(source_id="src_round6_execution")),
    )
    execution = ToolCallExecutionEvent(
        source="coordinator",
        content=[
            FunctionExecutionResult(
                call_id=f"call_{FULLY_CYRILLIC_SK}-abcdefgh_12345678",
                name="lookup_invoice",
                content="not executed",
                is_error=False,
            )
        ],
    )
    with pytest.raises(InvoiceAgentsError) as execution_failure:
        orchestration._record_stream_event(execution, execution_context, UsageSummary())
    assert execution_failure.value.stop_reason == "TOOL_CALL_ID_INVALID"

    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT case_id, tool_call_id FROM events WHERE case_id IN (?, ?)",
            (request_case_id, execution_case_id),
        ).fetchall()
    assert rows == []


def test_audit_recorder_redacts_fully_cyrillic_payload_and_error_surfaces(
    workflow_db: object,
) -> None:
    case_id = "case_round6_cyrillic_audit"
    fullwidth_key = "\uff41\uff50\uff49\uff3f\uff4b\uff45\uff59"
    ordinary_cyrillic = "смета_города"
    ordinary_japanese = "日本語_請求書"
    AuditRecorder(workflow_db, case_id).record(
        "provider.failure",
        {
            "error": f"{FULLY_CYRILLIC_API_KEY}=round6-cyrillic-error-secret",
            "details": {
                FULLY_CYRILLIC_API_KEY: "round6-cyrillic-map-secret",
                "provider": f"{FULLY_CYRILLIC_XAI}-abcdefgh_12345678",
            },
            ordinary_cyrillic: "ordinary",
            ordinary_japanese: "ordinary",
            "fullwidth": f"{fullwidth_key}=round6-fullwidth-secret",
        },
    )

    with connect_database(workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT payload_json FROM events WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    assert row is not None
    assert json.loads(row["payload_json"]) == {
        "details": {
            FULLY_CYRILLIC_API_KEY: "[REDACTED]",
            "provider": "[REDACTED]",
        },
        "error": f"{FULLY_CYRILLIC_API_KEY}=[REDACTED]",
        ordinary_cyrillic: "ordinary",
        ordinary_japanese: "ordinary",
        "fullwidth": f"{fullwidth_key}=[REDACTED]",
    }


def test_compatibility_width_credential_skeleton_remains_fail_closed() -> None:
    fullwidth_key = "\uff41\uff50\uff49\uff3f\uff4b\uff45\uff59"
    credential = f"{fullwidth_key}=round6-fullwidth-secret"
    tool_call_id = f"call_{credential}"

    assert sanitize_text(credential) == f"{fullwidth_key}=[REDACTED]"
    with pytest.raises(InvoiceAgentsError) as failure:
        orchestration._validated_tool_call_ids([tool_call_id], "case_round6_fullwidth")
    assert failure.value.stop_reason == "TOOL_CALL_ID_INVALID"


@pytest.mark.parametrize(
    "tool_call_id",
    [
        "call_σκ-abcdefgh_12345678",
        "call_απι_κεγ=round6-secret",
    ],
)
def test_fully_substituted_credential_tool_ids_are_rejected_at_both_boundaries(
    workflow_db: object,
    tool_call_id: str,
) -> None:
    with pytest.raises(InvoiceAgentsError) as failure:
        orchestration._validated_tool_call_ids([tool_call_id], "case_round6")
    assert failure.value.stop_reason == "TOOL_CALL_ID_INVALID"

    recorder = AuditRecorder(workflow_db, "case_round6_raw_tool_id")
    with pytest.raises(ValueError, match="tool call"):
        recorder.record("test.raw-tool-id", {"safe": True}, tool_call_id=tool_call_id)
    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT tool_call_id FROM events WHERE case_id = ?",
            ("case_round6_raw_tool_id",),
        ).fetchall()
    assert rows == []


@pytest.mark.parametrize(
    "identifier",
    [
        "a日本_語文字",
        "σχέδιο_πόλης",
        "смета_города",
    ],
)
def test_unrelated_multilingual_identifier_remains_intact_at_both_boundaries(
    workflow_db: object,
    identifier: str,
) -> None:
    assignment = f"{identifier}=ordinary"
    tool_call_id = f"call_{assignment}"

    assert sanitize_text(assignment) == assignment
    assert redact({identifier: "ordinary"}) == {identifier: "ordinary"}
    assert orchestration._validated_tool_call_ids([tool_call_id], "case_round6") == [tool_call_id]

    recorder = AuditRecorder(workflow_db, "case_round6_multilingual_tool_id")
    recorder.record("test.raw-tool-id", {"safe": True}, tool_call_id=tool_call_id)
    with connect_database(workflow_db, read_only=True) as connection:
        stored = connection.execute(
            "SELECT tool_call_id FROM events WHERE case_id = ?",
            ("case_round6_multilingual_tool_id",),
        ).fetchone()
    assert stored is not None
    assert stored["tool_call_id"] == tool_call_id


@pytest.mark.parametrize(
    "redacted",
    [
        "api_key=[REDACTED]",
        "api_κey=[REDACTED]",
        "t\u03bfken=[REDACTED]",
    ],
)
def test_opaque_redaction_marker_is_a_sanitizer_fixed_point(redacted: str) -> None:
    assert sanitize_text(redacted) == redacted
