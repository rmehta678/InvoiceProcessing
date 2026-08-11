"""Audit-safe event recording with recursive credential redaction."""

from __future__ import annotations

import json
import logging
import re
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
BEARER_VALUE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
NUMERIC_USAGE_TOKEN_KEYS = frozenset({"prompt_tokens", "completion_tokens"})
_CURRENT_AUDIT: ContextVar[AuditRecorder | None] = ContextVar(
    "invoice_agents_current_audit", default=None
)


def redact(value: Any) -> Any:
    """Redact known secret keys and bearer values recursively."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in NUMERIC_USAGE_TOKEN_KEYS and type(item) is int and item >= 0:
                cleaned[normalized_key] = item
            elif SENSITIVE_KEY.search(normalized_key):
                cleaned[normalized_key] = "[REDACTED]"
            else:
                cleaned[normalized_key] = redact(item)
        return cleaned
    if isinstance(value, str):
        return BEARER_VALUE.sub("Bearer [REDACTED]", value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [redact(item) for item in value]
    return value


class RedactingFilter(logging.Filter):
    """Prevent credentials from reaching console or local logging handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(redact(item) for item in record.args)
        return True


class ProviderRetryAuditHandler(logging.Handler):
    """Persist OpenAI SDK retry attempts against the current async case context."""

    def emit(self, record: logging.LogRecord) -> None:
        message = str(redact(record.getMessage()))
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
                        provider_request_id,
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
