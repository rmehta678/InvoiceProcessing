"""Validated runtime configuration with secrets kept as secret values."""

from __future__ import annotations

import math
import re
from datetime import date
from decimal import Decimal
from ipaddress import IPv6Address, ip_address
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from invoice_agents.errors import ErrorCategory, InvoiceAgentsError

XAI_MODEL = "grok-4.5"
XAI_BASE_URL = "https://api.x.ai/v1"
_UI_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class PdfPolicy(BaseModel):
    """Immutable, credential-free PDF limits safe to cross worker boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    pdf_max_pages: int = Field(ge=1, le=1_000)
    pdf_parse_timeout_seconds: float = Field(gt=0, le=120)
    pdf_worker_cpu_seconds: int = Field(ge=1, le=60)
    pdf_worker_memory_bytes: int = Field(ge=134_217_728, le=1_073_741_824)
    pdf_worker_result_max_bytes: int = Field(ge=65_536, le=16_777_216)

    @field_validator(
        "pdf_max_pages",
        "pdf_worker_cpu_seconds",
        "pdf_worker_memory_bytes",
        "pdf_worker_result_max_bytes",
        mode="before",
    )
    @classmethod
    def require_native_integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("PDF policy integer is not a native integer")
        return value

    @field_validator("pdf_parse_timeout_seconds", mode="before")
    @classmethod
    def require_finite_native_float(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("PDF policy timeout is not a finite native float")
        return value


def normalize_ui_host(value: str) -> str:
    """Return one canonical bare IP address or case-folded ASCII DNS host."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 253
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value)
        or any(delimiter in value for delimiter in ("*", "/", "\\", "@", "[", "]", "%"))
    ):
        raise ValueError(f"invalid UI allowed host: {value!r}")
    try:
        address = ip_address(value)
    except ValueError:
        host = value.lower()
        labels = host.split(".")
        if (
            ":" in host
            or all(character.isdigit() or character == "." for character in host)
            or any(not _UI_HOST_LABEL.fullmatch(label) for label in labels)
        ):
            raise ValueError(f"invalid UI allowed host: {value!r}") from None
        return host
    return address.compressed


def is_ui_loopback_host(value: str) -> bool:
    """Return whether a validated UI bind host is local loopback."""

    host = normalize_ui_host(value)
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def serialize_ui_origin(scheme: str, host: str, port: int) -> str:
    """Serialize a validated UI origin, including required IPv6 brackets."""

    if scheme not in {"http", "https"}:
        raise ValueError(f"invalid UI origin scheme: {scheme!r}")
    if not 1 <= port <= 65_535:
        raise ValueError(f"invalid UI origin port: {port!r}")
    normalized = normalize_ui_host(host)
    try:
        authority_host = (
            f"[{normalized}]" if isinstance(ip_address(normalized), IPv6Address) else normalized
        )
    except ValueError:
        authority_host = normalized
    return f"{scheme}://{authority_host}:{port}"


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
    sqlite_journal_mode: str = "DELETE"
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
    ui_allowed_hosts: tuple[str, ...] = ()
    ui_session_secret: SecretStr | None = None
    log_level: str = "INFO"

    @field_validator("review_threshold_currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.upper().strip()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("review threshold currency must be a three-letter code")
        return normalized

    @field_validator("sqlite_journal_mode", mode="before")
    @classmethod
    def normalize_delete_journal_mode(cls, value: object) -> str:
        """Fail before database access when two-file atomicity cannot be guaranteed."""

        if not isinstance(value, str) or value.strip().upper() != "DELETE":
            raise ValueError("SQLite journal mode must be DELETE")
        return "DELETE"

    @field_validator("ui_allowed_hosts")
    @classmethod
    def validate_ui_allowed_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for raw_host in value:
            host = normalize_ui_host(raw_host)
            if host not in normalized:
                normalized.append(host)
        return tuple(normalized)

    @field_validator("ui_session_secret", mode="before")
    @classmethod
    def empty_ui_session_secret_is_unconfigured(cls, value: object) -> object:
        if value == "" or (isinstance(value, SecretStr) and value.get_secret_value() == ""):
            return None
        return value

    @field_validator("ui_session_secret")
    @classmethod
    def validate_ui_session_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if (
            len(secret.encode("utf-8")) < 32
            or secret != secret.strip()
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in secret)
        ):
            raise ValueError(
                "UI session secret must contain at least 32 bytes without surrounding "
                "whitespace or control characters"
            )
        return value

    def configured_ui_session_secret(self) -> bytes | None:
        """Return explicit shared session key bytes only at app construction."""

        if self.ui_session_secret is None:
            return None
        return self.ui_session_secret.get_secret_value().encode("utf-8")

    def pdf_policy(self) -> PdfPolicy:
        """Project only active PDF limits into an immutable worker-safe policy."""

        return PdfPolicy(
            pdf_max_pages=self.pdf_max_pages,
            pdf_parse_timeout_seconds=self.pdf_parse_timeout_seconds,
            pdf_worker_cpu_seconds=self.pdf_worker_cpu_seconds,
            pdf_worker_memory_bytes=self.pdf_worker_memory_bytes,
            pdf_worker_result_max_bytes=self.pdf_worker_result_max_bytes,
        )

    def assert_delete_journal_mode(self) -> None:
        """Defend production boundaries even if model validation was explicitly bypassed."""

        if self.sqlite_journal_mode != "DELETE":
            raise InvoiceAgentsError(
                ErrorCategory.CONFIGURATION,
                f"SQLite journal mode must be DELETE, not {self.sqlite_journal_mode!r}",
                stop_reason="SQLITE_JOURNAL_MODE_UNSUPPORTED",
            )

    def provider_key(self) -> str:
        """Return the key only at the client construction boundary."""

        if self.xai_api_key is None or not self.xai_api_key.get_secret_value().strip():
            raise InvoiceAgentsError(
                ErrorCategory.CONFIGURATION,
                "XAI_API_KEY is missing or empty; no model fallback is permitted",
                stop_reason="PROVIDER_PREFLIGHT_FAILED",
            )
        return self.xai_api_key.get_secret_value()
