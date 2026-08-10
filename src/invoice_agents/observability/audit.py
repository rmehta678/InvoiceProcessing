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
from typing import Any
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from invoice_agents.db.core import connect_database

SENSITIVE_KEY = re.compile(r"(authorization|api[_-]?key|secret|token|cookie)", re.IGNORECASE)
BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
AUTHORIZATION_SCHEME_VALUE = re.compile(
    r"(?i)\bauthorization[\"']?\s*[:=]\s*[\"']?"
    r"(?:bearer|basic|digest|token)\s+[A-Za-z0-9._~+/=-]+[\"']?"
)
CREDENTIAL_VALUE = re.compile(
    r"(?i)\b(?:xai|sk)(?:-proj)?-[A-Za-z0-9_-]{8,}\b|"
    r"\b(?:api[_-]?key|authorization|cookie)[\"']?\s*[:=]\s*"
    r"[\"']?[^\s\"',}\]]+[\"']?"
)
CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028\u2029\u2060\ufeff]")
SAFE_PROVIDER_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SANITIZED_TEXT_MAX_CHARS = 4096
TRUNCATION_MARKER = "…[TRUNCATED]"
NUMERIC_USAGE_TOKEN_KEYS = frozenset({"prompt_tokens", "completion_tokens"})
_CURRENT_AUDIT: ContextVar[AuditRecorder | None] = ContextVar(
    "invoice_agents_current_audit", default=None
)


def _redact_text_patterns(value: str) -> str:
    return CREDENTIAL_VALUE.sub(
        "[REDACTED]",
        BEARER_VALUE.sub(
            "Bearer [REDACTED]",
            AUTHORIZATION_SCHEME_VALUE.sub("[REDACTED]", value),
        ),
    )


def sanitize_text(value: str) -> str:
    """Redact credential-like free text and apply one deterministic size ceiling."""

    sanitized = _redact_text_patterns(value)
    control_free = CONTROL_CHARACTER.sub("", value)
    if control_free != value and _redact_text_patterns(control_free) != control_free:
        # Controls can split a token across log lines or invisible characters. Use
        # the canonical control-free text only when that canonicalization exposes a
        # credential; ordinary multiline invoice text otherwise stays unchanged.
        sanitized = _redact_text_patterns(control_free)
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
            elif SENSITIVE_KEY.search(raw_key):
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
        safe_payload = redact(payload)
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
