"""Strict, environment-independent settings contract for local worker protocols."""

from __future__ import annotations

import math
import os
from datetime import date
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from invoice_agents.config import Settings


class WireSettings(BaseModel):
    """The exact redacted native-scalar settings payload accepted by child workers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    xai_api_key: None
    inventory_db: str
    workflow_db: str
    sqlite_journal_mode: str
    source_archive_dir: str
    source_max_bytes: int
    pdf_max_pages: int
    pdf_parse_timeout_seconds: float
    pdf_worker_cpu_seconds: int
    pdf_worker_memory_bytes: int
    pdf_worker_result_max_bytes: int
    review_threshold_amount: str
    review_threshold_currency: str
    review_threshold_effective_date: str
    due_date_tolerance_days: int
    review_age_amber_hours: int
    max_messages: int
    model_timeout_seconds: float
    transient_retries: int
    case_concurrency: int
    log_level: str

    @field_validator(
        "source_max_bytes",
        "pdf_max_pages",
        "pdf_worker_cpu_seconds",
        "pdf_worker_memory_bytes",
        "pdf_worker_result_max_bytes",
        "due_date_tolerance_days",
        "review_age_amber_hours",
        "max_messages",
        "transient_retries",
        "case_concurrency",
        mode="before",
    )
    @classmethod
    def require_native_integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("wire integer setting is not a native integer")
        return value

    @field_validator(
        "pdf_parse_timeout_seconds",
        "model_timeout_seconds",
        mode="before",
    )
    @classmethod
    def require_finite_native_float(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("wire float setting is not a finite native float")
        return value

    @field_validator(
        "inventory_db",
        "workflow_db",
        "sqlite_journal_mode",
        "source_archive_dir",
        "review_threshold_amount",
        "review_threshold_currency",
        "review_threshold_effective_date",
        "log_level",
        mode="before",
    )
    @classmethod
    def require_native_string(cls, value: object) -> object:
        if type(value) is not str or not value or value != value.strip():
            raise ValueError("wire string setting is not an exact nonempty native string")
        return value

    @model_validator(mode="after")
    def require_canonical_serialized_scalars(self) -> WireSettings:
        try:
            amount = Decimal(self.review_threshold_amount)
        except InvalidOperation as exc:
            raise ValueError("wire decimal setting is invalid") from exc
        if not amount.is_finite() or str(amount) != self.review_threshold_amount:
            raise ValueError("wire decimal setting is not canonical and finite")
        try:
            effective_date = date.fromisoformat(self.review_threshold_effective_date)
        except ValueError as exc:
            raise ValueError("wire date setting is invalid") from exc
        if effective_date.isoformat() != self.review_threshold_effective_date:
            raise ValueError("wire date setting is not canonical")
        return self


WIRE_SETTINGS_FIELDS = frozenset(WireSettings.model_fields)


def serialize_wire_settings(settings: Settings) -> dict[str, object]:
    """Serialize every runtime setting exactly once, replacing the credential with null."""

    return {
        "xai_api_key": None,
        "inventory_db": os.fspath(settings.inventory_db),
        "workflow_db": os.fspath(settings.workflow_db),
        "sqlite_journal_mode": settings.sqlite_journal_mode,
        "source_archive_dir": os.fspath(settings.source_archive_dir),
        "source_max_bytes": settings.source_max_bytes,
        "pdf_max_pages": settings.pdf_max_pages,
        "pdf_parse_timeout_seconds": settings.pdf_parse_timeout_seconds,
        "pdf_worker_cpu_seconds": settings.pdf_worker_cpu_seconds,
        "pdf_worker_memory_bytes": settings.pdf_worker_memory_bytes,
        "pdf_worker_result_max_bytes": settings.pdf_worker_result_max_bytes,
        "review_threshold_amount": str(settings.review_threshold_amount),
        "review_threshold_currency": settings.review_threshold_currency,
        "review_threshold_effective_date": settings.review_threshold_effective_date.isoformat(),
        "due_date_tolerance_days": settings.due_date_tolerance_days,
        "review_age_amber_hours": settings.review_age_amber_hours,
        "max_messages": settings.max_messages,
        "model_timeout_seconds": settings.model_timeout_seconds,
        "transient_retries": settings.transient_retries,
        "case_concurrency": settings.case_concurrency,
        "log_level": settings.log_level,
    }


def decode_wire_settings(payload: object) -> Settings:
    """Validate strict wire data before constructing environment-backed runtime settings."""

    wire = WireSettings.model_validate(payload, strict=True)
    exact_payload = wire.model_dump()
    settings = Settings(**exact_payload)
    if serialize_wire_settings(settings) != exact_payload:
        raise ValueError("runtime settings changed during strict wire construction")
    return settings
