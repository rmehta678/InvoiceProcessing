"""Audit-safe event recording with recursive credential redaction."""

from __future__ import annotations

import json
import logging
import re
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from invoice_agents.db.core import connect_database
from invoice_agents.models import CaseResult

SENSITIVE_KEY = re.compile(r"(authorization|api[_-]?key|secret|token|cookie)", re.IGNORECASE)
BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
AUTHORIZATION_SCHEME_VALUE = re.compile(
    r"(?i)\bauthorization[\"']?\s*[:=]\s*[\"']?"
    r"(?:bearer|basic|digest|token)\s+[A-Za-z0-9._~+/=-]+[\"']?"
)
PROVIDER_CREDENTIAL_VALUE = re.compile(r"(?i)\b(?:xai|sk)(?:-proj)?-[A-Za-z0-9_-]{8,}\b")
KEYED_CREDENTIAL_VALUE = re.compile(
    r"(?i)(?P<prefix>\b[A-Za-z0-9_-]*"
    r"(?:api[_-]?key|authorization|secret|token(?!s(?:$|[_-])))"
    r"[A-Za-z0-9_-]*[\"']?\s*[:=]\s*[\"']?)"
    r"(?P<value>\[REDACTED\]|[^\s\"',}\]]+)(?P<suffix>[\"']?)"
)
COOKIE_HEADER_VALUE = re.compile(
    r"(?im)(?P<prefix>(?<![\w-])(?:set-cookie|cookie)[ \t]*:[ \t]*)"
    r"[^\n]*(?:\n[ \t]+(?![A-Za-z0-9-]+[ \t]*:)[^\n]*)*"
)
COOKIE_ASSIGNMENT_VALUE = re.compile(r"(?i)\bcookie[\"']?\s*=\s*[\"']?[^\s\"',}\]]+[\"']?")
DOUBLE_QUOTED_COOKIE_FIELD_VALUE = re.compile(
    r'(?i)(?P<prefix>"(?:set-cookie|cookie)"\s*:\s*")'
    r'(?P<value>(?:\\.|[^"\\\r\n])*)(?P<suffix>"?)'
)
SINGLE_QUOTED_COOKIE_FIELD_VALUE = re.compile(
    r"(?i)(?P<prefix>'(?:set-cookie|cookie)'\s*:\s*')"
    r"(?P<value>(?:\\.|[^'\\\r\n])*)(?P<suffix>'?)"
)
SPLIT_PROVIDER_CREDENTIAL = re.compile(
    r"(?i)\b(?:x[\t\n]*a[\t\n]*i|s[\t\n]*k)"
    r"(?:[\t\n]*-[\t\n]*p[\t\n]*r[\t\n]*o[\t\n]*j)?"
    r"[\t\n]*-[\t\n]*(?:[A-Za-z0-9_-][\t\n]*){8,}"
)
ANSI_ESCAPE = re.compile(
    r"(?:\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])|"
    r"\x9b[0-?]*[ -/]*[@-~])"
)
UNSAFE_CONTROL_CHARACTER = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2060\ufeff]"
)
SAFE_PROVIDER_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SANITIZED_TEXT_MAX_CHARS = 4096
TRUNCATION_MARKER = "…[TRUNCATED]"
NUMERIC_USAGE_TOKEN_KEYS = frozenset({"prompt_tokens", "completion_tokens"})
_CURRENT_AUDIT: ContextVar[AuditRecorder | None] = ContextVar(
    "invoice_agents_current_audit", default=None
)


def _redact_text_patterns(value: str) -> str:
    double_cookie_safe = DOUBLE_QUOTED_COOKIE_FIELD_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]{match.group('suffix')}",
        value,
    )
    quoted_cookie_safe = SINGLE_QUOTED_COOKIE_FIELD_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]{match.group('suffix')}",
        double_cookie_safe,
    )
    cookie_safe = COOKIE_HEADER_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]",
        quoted_cookie_safe,
    )
    provider_safe = SPLIT_PROVIDER_CREDENTIAL.sub("[REDACTED]", cookie_safe)
    authorization_safe = AUTHORIZATION_SCHEME_VALUE.sub("[REDACTED]", provider_safe)
    bearer_safe = BEARER_VALUE.sub("Bearer [REDACTED]", authorization_safe)
    keyed_safe = KEYED_CREDENTIAL_VALUE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]{match.group('suffix')}",
        PROVIDER_CREDENTIAL_VALUE.sub("[REDACTED]", bearer_safe),
    )
    return COOKIE_ASSIGNMENT_VALUE.sub("[REDACTED]", keyed_safe)


def _neutralize_controls(value: str) -> str:
    without_ansi = ANSI_ESCAPE.sub("", value)
    normalized_lines = without_ansi.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = normalized_lines.replace("\u2028", "\n").replace("\u2029", "\n")
    return UNSAFE_CONTROL_CHARACTER.sub("", normalized_lines)


def sanitize_text(value: str) -> str:
    """Redact credential-like free text and apply one deterministic size ceiling."""

    sanitized = _redact_text_patterns(_neutralize_controls(value))
    if len(sanitized) <= SANITIZED_TEXT_MAX_CHARS:
        return sanitized
    visible_chars = SANITIZED_TEXT_MAX_CHARS - len(TRUNCATION_MARKER)
    return f"{sanitized[:visible_chars]}{TRUNCATION_MARKER}"


def safe_provider_request_id(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or SAFE_PROVIDER_REQUEST_ID.fullmatch(value) is None
        or sanitize_text(value) != value
    ):
        return None
    return value


def redact(value: Any) -> Any:
    """Redact known secret keys and sanitize every serializable text boundary."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            raw_key = str(key)
            safe_key = sanitize_text(raw_key)
            if raw_key in NUMERIC_USAGE_TOKEN_KEYS and type(item) is int and item >= 0:
                cleaned[safe_key] = item
            elif SENSITIVE_KEY.search(raw_key) or SENSITIVE_KEY.search(safe_key):
                cleaned[safe_key] = "[REDACTED]"
            else:
                cleaned[safe_key] = redact(item)
        return cleaned
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return sanitize_text(bytes(value).decode("utf-8", errors="replace"))
    if isinstance(value, Sequence):
        return [redact(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(str(value))


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key")
        payload[key] = value
    return payload


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON number")


def sanitize_json_text(value: str) -> str:
    """Redact one JSON string structurally or return one bounded text fallback."""

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return sanitize_text(value)
    return json.dumps(
        redact(parsed),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _safe_usage(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    usage: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens"):
        count = value.get(field)
        if type(count) is int and count >= 0:
            usage[field] = count
    return usage or None


def _safe_text_field(value: object) -> str | None:
    return sanitize_text(value) if isinstance(value, str) else None


def _normalized_tool_items(event_name: str, value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    request = event_name == "ToolCallRequestEvent"
    for raw_item in value:
        if not isinstance(raw_item, Mapping):
            continue
        item: dict[str, Any] = {}
        id_field = "id" if request else "call_id"
        identifier = _safe_text_field(raw_item.get(id_field))
        name = _safe_text_field(raw_item.get("name"))
        if identifier is not None:
            item[id_field] = identifier
        if name is not None:
            item["name"] = name
        json_field = "arguments" if request else "content"
        json_value = raw_item.get(json_field)
        if isinstance(json_value, str):
            item[json_field] = sanitize_json_text(json_value)
        elif json_value is not None:
            item[json_field] = redact(json_value)
        if not request and type(raw_item.get("is_error")) is bool:
            item["is_error"] = raw_item["is_error"]
        normalized.append(item)
    return normalized


def normalize_autogen_event_payload(
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Select only the semantic AutoGen fields allowed in an audit payload."""

    event_name = event_type.removeprefix("autogen.")
    normalized: dict[str, Any] = {"type": event_name}
    source = _safe_text_field(payload.get("source"))
    if source is not None:
        normalized["source"] = source
    usage = _safe_usage(payload.get("models_usage"))
    if usage is not None:
        normalized["models_usage"] = usage
    if event_name in {"ToolCallRequestEvent", "ToolCallExecutionEvent"}:
        normalized["content"] = _normalized_tool_items(event_name, payload.get("content"))
    elif "content" in payload:
        normalized["content"] = redact(payload["content"])
    if event_name == "HandoffMessage":
        target = _safe_text_field(payload.get("target"))
        if target is not None:
            normalized["target"] = target
    return normalized


def sanitize_stored_event_payload(event_type: str, value: str) -> str:
    """Present current or legacy event JSON without exposing historical secrets."""

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return sanitize_text(value)
    if event_type.startswith("autogen.") and isinstance(parsed, Mapping):
        safe_payload: Any = normalize_autogen_event_payload(event_type, parsed)
    else:
        safe_payload = redact(parsed)
    return json.dumps(
        safe_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sanitize_case_result(result: CaseResult) -> CaseResult:
    """Normalize only terminal free text before storage or artifact publication."""

    payment = result.payment
    if payment is not None and payment.error is not None:
        payment = payment.model_copy(
            update={"error": sanitize_text(payment.error)},
            deep=True,
        )
    errors = [
        error.model_copy(
            update={
                "category": sanitize_text(error.category),
                "message": sanitize_text(error.message),
                "stop_reason": (
                    sanitize_text(error.stop_reason) if error.stop_reason is not None else None
                ),
                "provider_request_id": safe_provider_request_id(error.provider_request_id),
                "details": cast(dict[str, Any], redact(error.details)),
            },
            deep=True,
        )
        for error in result.errors
    ]
    return result.model_copy(
        update={"payment": payment, "errors": errors},
        deep=True,
    )


class RedactingFilter(logging.Filter):
    """Prevent credentials from reaching console or local logging handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = sanitize_text(record.getMessage())
        record.args = ()
        if record.exc_info is not None:
            record.exc_text = sanitize_text("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = sanitize_text(record.exc_text)
        if record.stack_info:
            record.stack_info = sanitize_text(record.stack_info)
        return True


class ProviderRetryAuditHandler(logging.Handler):
    """Persist OpenAI SDK retry attempts against the current async case context."""

    def emit(self, record: logging.LogRecord) -> None:
        message = sanitize_text(record.getMessage())
        if "retry" not in message.casefold():
            return
        recorder = _CURRENT_AUDIT.get()
        if recorder is not None:
            recorder.record(
                "provider.retry",
                {"logger": record.name, "level": record.levelname, "message": message},
            )


def configure_logging(level: str = "INFO") -> None:
    """Configure concise local logging and an application-wide redaction filter."""

    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[handler],
        force=True,
    )
    # Structured AutoGen events are persisted from run_stream. Avoid duplicating
    # full prompts/responses to the human console at ordinary INFO verbosity.
    logging.getLogger("autogen_core").setLevel(logging.WARNING)
    logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    openai_logger = logging.getLogger("openai")
    openai_logger.setLevel(logging.INFO)
    if not any(isinstance(item, ProviderRetryAuditHandler) for item in openai_logger.handlers):
        openai_logger.addHandler(ProviderRetryAuditHandler())
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())


class AuditRecorder:
    """Append correlated, redacted events to the mutable workflow database."""

    def __init__(self, workflow_db: Path, case_id: str | None = None) -> None:
        self.workflow_db = workflow_db
        self.case_id = case_id
        self.tracer = trace.get_tracer("invoice_agents")

    def record(
        self,
        event_type: str,
        payload: Any,
        *,
        source_id: str | None = None,
        agent_name: str | None = None,
        tool_call_id: str | None = None,
        db_evidence_id: str | None = None,
        review_id: str | None = None,
        payment_id: str | None = None,
        provider_request_id: str | None = None,
    ) -> str:
        """Persist one event and mirror its timing/correlation in a local span."""

        event_id = f"evt_{uuid4().hex}"
        safe_payload = (
            normalize_autogen_event_payload(event_type, payload)
            if event_type.startswith("autogen.") and isinstance(payload, Mapping)
            else redact(payload)
        )
        encoded = json.dumps(safe_payload, default=str, ensure_ascii=False, sort_keys=True)
        sanitized_provider_request_id = safe_provider_request_id(provider_request_id)
        created_at = datetime.now(UTC).isoformat()
        with self.tracer.start_as_current_span(event_type) as span:
            if self.case_id:
                span.set_attribute("invoice.case_id", self.case_id)
            if agent_name:
                span.set_attribute("invoice.agent", agent_name)
            with connect_database(self.workflow_db) as connection:
                connection.execute(
                    "INSERT INTO events("
                    "event_id, case_id, source_id, event_type, agent_name, tool_call_id, "
                    "db_evidence_id, review_id, payment_id, provider_request_id, payload_json, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_id,
                        self.case_id,
                        source_id,
                        event_type,
                        agent_name,
                        tool_call_id,
                        db_evidence_id,
                        review_id,
                        payment_id,
                        sanitized_provider_request_id,
                        encoded,
                        created_at,
                    ),
                )
                connection.commit()
        return event_id


@contextmanager
def bind_audit_recorder(recorder: AuditRecorder) -> Iterator[None]:
    """Bind SDK retry logging to one async case without global case state."""

    token = _CURRENT_AUDIT.set(recorder)
    try:
        yield
    finally:
        _CURRENT_AUDIT.reset(token)
