"""Validated runtime configuration with secrets kept as secret values."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from invoice_agents.errors import ErrorCategory, InvoiceAgentsError

XAI_MODEL = "grok-4.5"
XAI_BASE_URL = "https://api.x.ai/v1"


class Settings(BaseSettings):
    """Environment-backed settings; model/provider identity cannot silently drift."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="INVOICE_",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    xai_api_key: SecretStr | None = Field(default=None, validation_alias="XAI_API_KEY")
    inventory_db: Path = Path("inventory.db")
    workflow_db: Path = Path("workflow.db")
    source_archive_dir: Path = Path("artifacts/sources")
    source_max_bytes: int = Field(default=10_485_760, gt=0)
    pdf_max_pages: int = Field(default=100, ge=1, le=1_000)
    pdf_parse_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    pdf_worker_cpu_seconds: int = Field(default=10, ge=1, le=60)
    # Additional virtual-address headroom above the spawned worker's measured baseline.
    pdf_worker_memory_bytes: int = Field(
        default=536_870_912,
        ge=134_217_728,
        le=1_073_741_824,
    )
    pdf_worker_result_max_bytes: int = Field(
        default=4_194_304,
        ge=65_536,
        le=16_777_216,
    )
    review_threshold_amount: Decimal = Decimal("10000.00")
    review_threshold_currency: str = "USD"
    review_threshold_effective_date: date = date(2026, 8, 6)
    # Business policy, not an operational knob: the allowed calendar-day distance
    # between the stated due date and invoice_date + Net-N terms before review.
    due_date_tolerance_days: int = Field(default=3, ge=0, le=10)
    # Console display only: queue age (hours) after which a pending review is
    # visually flagged as aging; it never changes review behavior.
    review_age_amber_hours: int = Field(default=24, ge=1, le=24 * 14)
    max_messages: int = Field(default=40, ge=8, le=200)
    model_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    transient_retries: int = Field(default=2, ge=0, le=5)
    case_concurrency: int = Field(default=2, ge=1, le=8)
    log_level: str = "INFO"

    @field_validator("review_threshold_currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.upper().strip()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("review threshold currency must be a three-letter code")
        return normalized

    def provider_key(self) -> str:
        """Return the key only at the client construction boundary."""

        if self.xai_api_key is None or not self.xai_api_key.get_secret_value().strip():
            raise InvoiceAgentsError(
                ErrorCategory.CONFIGURATION,
                "XAI_API_KEY is missing or empty; no model fallback is permitted",
                stop_reason="PROVIDER_PREFLIGHT_FAILED",
            )
        return self.xai_api_key.get_secret_value()
