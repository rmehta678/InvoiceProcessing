"""Small, strict Pydantic contracts shared by tools, agents, persistence, and CLI."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """Reject undeclared fields so provider/schema drift is observable."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ToolStatus(StrEnum):
    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID_INPUT = "INVALID_INPUT"
    ERROR = "ERROR"


class CaseStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


class DecisionKind(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    HOLD = "HOLD"
    FAILED = "FAILED"


class HumanDecisionKind(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_CORRECTION = "REQUEST_CORRECTION"
    ESTABLISH_MAPPING = "ESTABLISH_MAPPING"
    SUPERSEDE_REVISION = "SUPERSEDE_REVISION"


class PaymentStatus(StrEnum):
    PAID = "PAID"
    DUPLICATE = "DUPLICATE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    FAILED = "FAILED"


class InventoryStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    EXCEEDS_STOCK = "EXCEEDS_STOCK"
    OUT_OF_STOCK = "OUT_OF_STOCK"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    ERROR = "ERROR"


class IdentityRelationship(StrEnum):
    EXACT_ARTIFACT = "EXACT_ARTIFACT"
    DUPLICATE_REPRESENTATION = "DUPLICATE_REPRESENTATION"
    POSSIBLE_REVISION = "POSSIBLE_REVISION"
    CONFLICT = "CONFLICT"


CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


class EvidenceRef(StrictModel):
    """Precise location and raw value supporting one fact."""

    source_id: str
    locator_type: Literal["line", "row", "json_path", "xpath", "page", "file"]
    locator: str
    raw_value: str | None = None
    excerpt: str | None = None


class SourceArtifact(StrictModel):
    """Immutable identity for a submitted invoice file."""

    source_id: str
    canonical_path: Path
    sha256: str
    source_format: Literal["txt", "json", "csv", "xml", "pdf"]
    size_bytes: int = Field(ge=0)
    modified_at: datetime
    page_count: int | None = Field(default=None, ge=1)
    row_count: int | None = Field(default=None, ge=0)


class EvidenceValue(StrictModel):
    """Raw and normalized forms of a field with its transformation recorded."""

    raw_value: str | None
    normalized_value: str | None
    normalization: str = "none"
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguity: str | None = None


class Money(StrictModel):
    amount: Decimal
    currency: CurrencyCode


class InvoiceLine(StrictModel):
    """One source line; mappings remain explicit and never fuzzy-autoaccepted."""

    line_id: str
    raw_item: str
    normalized_item: str
    candidate_skus: list[str] = Field(default_factory=list)
    canonical_sku: str | None = None
    raw_quantity: str
    quantity: Decimal
    raw_unit_price: str
    unit_price: Decimal
    raw_declared_line_total: str | None = None
    declared_line_total: Decimal | None = None
    calculated_line_total: Decimal
    evidence: list[EvidenceRef]
    ambiguity: list[str] = Field(default_factory=list)


class ExtractedInvoice(StrictModel):
    """Format-neutral invoice evidence with all material omissions and conflicts."""

    source: SourceArtifact
    invoice_number: EvidenceValue
    revision: EvidenceValue | None = None
    vendor: EvidenceValue
    invoice_date: EvidenceValue
    due_date: EvidenceValue
    payment_terms: EvidenceValue
    currency: EvidenceValue
    lines: list[InvoiceLine]
    declared_subtotal: Decimal | None = None
    declared_tax_rate: Decimal | None = None
    declared_tax_amount: Decimal | None = None
    declared_fees: Decimal = Decimal("0")
    declared_total: Decimal | None = None
    missing_fields: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)


class InventoryRow(StrictModel):
    sku: str
    item_name: str
    available_stock: int = Field(ge=0)


class InventoryLookupResult(StrictModel):
    status: ToolStatus
    query: str
    row: InventoryRow | None = None
    candidates: list[InventoryRow] = Field(default_factory=list)
    alias_provenance: dict[str, str | None] | None = None
    error: str | None = None


class CanonicalMapping(StrictModel):
    raw_item: str
    sku: str
    basis: Literal["exact_item_name", "approved_alias", "human_decision"]
    evidence: list[EvidenceRef] = Field(default_factory=list)


class AggregatedQuantity(StrictModel):
    sku: str | None
    raw_items: list[str]
    requested_quantity: Decimal
    mapping_basis: list[str]


class InventoryComparison(StrictModel):
    sku: str | None
    raw_items: list[str]
    requested_quantity: Decimal
    available_stock: int | None
    status: InventoryStatus
    queried_row: InventoryRow | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    explanation: str


class FinancialComparison(StrictModel):
    calculated_subtotal: Decimal
    declared_subtotal: Decimal | None
    subtotal_delta: Decimal | None
    calculated_tax: Decimal
    declared_tax: Decimal | None
    tax_delta: Decimal | None
    tax_recomputable: bool
    tax_basis: str
    calculated_fees: Decimal
    calculated_total: Decimal
    declared_total: Decimal | None
    total_delta: Decimal | None
    line_deltas: dict[str, Decimal]
    exact: bool


class DateAssessment(StrictModel):
    field: str
    raw_value: str | None
    parsed_date: date | None
    status: Literal["EXACT", "AMBIGUOUS", "RELATIVE", "INVALID", "MISSING"]
    explanation: str


class RiskPolicy(StrictModel):
    """Configured business-policy inputs that produced one risk assessment."""

    review_threshold_amount: Decimal
    review_threshold_currency: CurrencyCode
    review_threshold_effective_date: date
    due_date_tolerance_days: int = Field(ge=0, le=10)


class IdentityCandidate(StrictModel):
    case_id: str
    source_id: str
    invoice_number: str | None
    vendor: str | None
    source_hash: str
    revision: str | None
    source_format: str
    relationship: IdentityRelationship
    explanation: str


class RiskAssessment(StrictModel):
    policy: RiskPolicy
    financial: FinancialComparison
    dates: list[DateAssessment]
    inventory: list[InventoryComparison]
    identity_candidates: list[IdentityCandidate]
    suspicious_signals: list[str]
    unavailable_reconciliations: list[str]
    policy_review_reasons: list[str]


class EvidenceBlocker(StrictModel):
    """Stable identity plus human-readable context for approval-blocking evidence."""

    blocker_id: str
    kind: Literal["inventory", "financial"]
    evidence_id: str
    description: str


class Critique(StrictModel):
    supported_findings: list[str]
    challenged_findings: list[str]
    missing_evidence: list[str]
    requested_follow_up: list[str]
    recommended_disposition: DecisionKind
    rationale: list[str]


class HumanDecision(StrictModel):
    review_id: str
    reviewer: str = Field(min_length=1)
    decision: HumanDecisionKind
    reason: str = Field(min_length=1)
    decided_at: datetime
    mappings: list[CanonicalMapping] = Field(default_factory=list)
    superseded_case_id: str | None = None
    addressed_blocker_ids: list[str] = Field(default_factory=list)


class ReviewRequest(StrictModel):
    review_id: str
    case_id: str
    # Review cycles are ordered per case; pre-schema-v2 payloads default to 1.
    sequence: int = Field(default=1, ge=1)
    status: Literal["PENDING", "RESOLVED"]
    reasons: list[str]
    amount: Money | None
    source: SourceArtifact
    evidence_bundle: dict[str, Any]
    agent_recommendation: DecisionKind
    agent_rationale: list[str]
    critic: Critique
    questions: list[str]
    created_at: datetime
    human_decision: HumanDecision | None = None


class FinalDecision(StrictModel):
    decision: DecisionKind
    reasons: list[str] = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    critic_disposition: DecisionKind
    human_outcome: HumanDecision | None = None
    payment_eligible: bool

    @field_validator("payment_eligible")
    @classmethod
    def approved_only(cls, value: bool, info: Any) -> bool:
        decision = info.data.get("decision")
        if value and decision != DecisionKind.APPROVE:
            raise ValueError("only APPROVE can be payment eligible")
        return value


class PaymentResult(StrictModel):
    payment_id: str | None
    case_id: str
    idempotency_key: str
    status: PaymentStatus
    vendor: str | None
    amount: Money | None
    processed_at: datetime | None
    duplicate_of: str | None = None
    error: str | None = None


class PersistedPaymentRow(StrictModel):
    """Strict storage boundary for one immutable payment-ledger row."""

    model_config = ConfigDict(extra="forbid", strict=True)

    payment_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    vendor: str = Field(min_length=1)
    amount: str = Field(min_length=1)
    currency: CurrencyCode
    status: Literal["PAID", "FAILED"]
    error: str | None
    created_at: str = Field(min_length=1)
    decision_generation: int = Field(ge=1)
    evidence_snapshot_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: str = Field(min_length=1)
    invoice_number: str | None
    review_id: str | None

    @field_validator("amount")
    @classmethod
    def canonical_positive_amount(cls, value: str) -> str:
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("payment amount is not a canonical decimal") from exc
        if not parsed.is_finite() or parsed <= 0 or str(parsed) != value:
            raise ValueError("payment amount is not a canonical positive decimal")
        return value

    @field_validator("created_at")
    @classmethod
    def canonical_utc_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("payment timestamp is not canonical UTC") from exc
        offset = parsed.utcoffset()
        if (
            parsed.tzinfo is None
            or offset is None
            or offset.total_seconds() != 0
            or parsed.isoformat() != value
        ):
            raise ValueError("payment timestamp is not canonical UTC")
        return value

    @model_validator(mode="after")
    def status_error_pair_is_exact(self) -> PersistedPaymentRow:
        if self.status == "PAID" and self.error is not None:
            raise ValueError("paid payment cannot contain an error")
        if self.status == "FAILED" and (self.error is None or not self.error.strip()):
            raise ValueError("failed payment requires a nonempty error")
        return self


class ErrorRecord(StrictModel):
    category: str
    message: str
    case_id: str | None = None
    stop_reason: str | None = None
    provider_request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class UsageSummary(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class CaseResult(StrictModel):
    case_id: str
    source_id: str | None
    status: CaseStatus
    stop_reason: str
    final_decision: FinalDecision | None = None
    review_request: ReviewRequest | None = None
    payment: PaymentResult | None = None
    errors: list[ErrorRecord] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    started_at: datetime
    finished_at: datetime


class ToolEnvelope(StrictModel):
    """Uniform boundary for agent-callable tools."""

    status: ToolStatus
    result: dict[str, Any] | list[Any] | None = None
    error_category: str | None = None
    error: str | None = None
