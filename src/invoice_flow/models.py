"""Data contracts shared by every agent in the pipeline.

The extraction models are deliberately permissive: bad source data must survive
ingestion as *data* so the validation agent can flag it. A schema that rejected a
negative quantity would turn invoice 1009 into a crash instead of a finding.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import BASE_CURRENCY


def normalize(name: str) -> str:
    """Casefold and strip everything that is not a letter or digit.

    Turns the OCR-spaced ``Widget A`` into ``widgeta`` so it matches the
    catalogue's ``WidgetA``.
    """
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def money(amount: float | None, currency: str | None = None) -> str:
    """Format an amount for display. A non-base currency keeps its code."""
    if amount is None:
        return "-"
    suffix = f" {currency}" if currency and currency != BASE_CURRENCY else ""
    return f"${amount:,.2f}{suffix}"


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


class Severity(str, Enum):
    """How much a finding should weigh on the approval decision."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FindingCode(str, Enum):
    """Stable identifiers for every check the system performs.

    Tests assert on these rather than on message text, so wording can change
    without breaking the golden suite.
    """

    # Extraction quality
    EXTRACTION_LOW_CONFIDENCE = "EXTRACTION_LOW_CONFIDENCE"

    # Arithmetic reconciliation
    LINE_AMOUNT_MISMATCH = "LINE_AMOUNT_MISMATCH"
    SUBTOTAL_MISMATCH = "SUBTOTAL_MISMATCH"
    TAX_MISMATCH = "TAX_MISMATCH"
    TOTAL_MISMATCH = "TOTAL_MISMATCH"

    # Inventory
    ITEM_UNKNOWN = "ITEM_UNKNOWN"
    ITEM_OUT_OF_STOCK = "ITEM_OUT_OF_STOCK"
    STOCK_SHORTFALL = "STOCK_SHORTFALL"
    ITEM_NAME_SUGGESTION = "ITEM_NAME_SUGGESTION"

    # Data integrity
    QUANTITY_INVALID = "QUANTITY_INVALID"
    QUANTITY_NON_INTEGER = "QUANTITY_NON_INTEGER"
    UNIT_PRICE_INVALID = "UNIT_PRICE_INVALID"
    TOTAL_NON_POSITIVE = "TOTAL_NON_POSITIVE"
    TOTAL_MISSING = "TOTAL_MISSING"
    TOTAL_RECONSTRUCTED = "TOTAL_RECONSTRUCTED"
    VENDOR_MISSING = "VENDOR_MISSING"
    INVOICE_NUMBER_MISSING = "INVOICE_NUMBER_MISSING"
    NO_LINE_ITEMS = "NO_LINE_ITEMS"

    # Ledger
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    DUPLICATE_INVOICE_CONFLICT = "DUPLICATE_INVOICE_CONFLICT"
    DUPLICATE_INVOICE_REVISION = "DUPLICATE_INVOICE_REVISION"

    # Dates
    DUE_DATE_MISSING = "DUE_DATE_MISSING"
    DUE_DATE_UNPARSEABLE = "DUE_DATE_UNPARSEABLE"
    DUE_DATE_BEFORE_INVOICE_DATE = "DUE_DATE_BEFORE_INVOICE_DATE"
    DUE_DATE_NOT_AFTER_INVOICE_DATE = "DUE_DATE_NOT_AFTER_INVOICE_DATE"

    # Currency
    CURRENCY_NON_BASE = "CURRENCY_NON_BASE"

    # Risk / fraud signals
    URGENCY_PRESSURE = "URGENCY_PRESSURE"
    WIRE_TRANSFER_REQUEST = "WIRE_TRANSFER_REQUEST"
    AMOUNT_JUST_UNDER_THRESHOLD = "AMOUNT_JUST_UNDER_THRESHOLD"
    AMOUNT_OVER_THRESHOLD = "AMOUNT_OVER_THRESHOLD"


class Finding(BaseModel):
    """A single observation about an invoice, produced by any stage."""

    code: FindingCode
    severity: Severity
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    source: str = "validation"

    def render(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code.value}: {self.message}"


# --------------------------------------------------------------------------
# Documents and extraction
# --------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """The raw invoice as loaded from disk, before any LLM sees it."""

    path: str
    file_format: str
    text: str

    @property
    def name(self) -> str:
        return Path(self.path).name


class LineItem(BaseModel):
    """One billed line. `name` is kept exactly as written on the invoice."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="Item name exactly as it appears on the invoice")
    quantity: float | None = Field(default=None, description="Quantity billed")
    unit_price: float | None = Field(default=None, description="Price per unit")
    amount: float | None = Field(default=None, description="Line total if stated")
    note: str | None = Field(default=None, description="Any per-line annotation")

    @property
    def normalized_name(self) -> str:
        """Casefolded, punctuation- and space-stripped form for DB matching."""
        return normalize(self.name)


class ExtractedInvoice(BaseModel):
    """The schema the LLM fills in.

    Every field is optional: a missing vendor is a finding, not a parse error.
    Derived fields (parsed dates, confidence) live on `InvoiceDraft` below and
    are computed by our own code -- there is no reason to let an LLM guess at
    something a date parser answers exactly.
    """

    model_config = ConfigDict(extra="ignore")

    invoice_number: str | None = Field(default=None, description="Invoice identifier, e.g. INV-1001")
    # Corroboration for a human, never grounds to prefer one document over
    # another: a revision marker is vendor-supplied text, and "INV-1004 rev R2"
    # for $50,000 is the exact shape of the fraud the ledger check exists to
    # catch. See `_duplicate_findings` in agents/validation.py.
    revision: str | None = Field(
        default=None, description="Revision or version marker if the document states one"
    )
    vendor_name: str | None = Field(default=None, description="Name of the company billing Acme")
    vendor_address: str | None = Field(default=None, description="Vendor postal address if stated")
    invoice_date_raw: str | None = Field(
        default=None, description="Invoice issue date, copied verbatim as written"
    )
    due_date_raw: str | None = Field(
        default=None, description="Payment due date, copied verbatim as written"
    )
    line_items: list[LineItem] = Field(
        default_factory=list, description="Every billed line, in document order"
    )
    subtotal: float | None = Field(default=None, description="Stated subtotal before tax")
    tax_rate: float | None = Field(default=None, description="Tax rate as a decimal, e.g. 0.07")
    tax_amount: float | None = Field(default=None, description="Stated tax amount")
    shipping: float | None = Field(default=None, description="Shipping or freight charge")
    total: float | None = Field(default=None, description="Stated grand total")
    currency: str | None = Field(default=None, description="ISO currency code, e.g. USD")
    payment_terms: str | None = Field(default=None, description="Terms such as 'Net 30'")
    notes: str | None = Field(default=None, description="Free-text notes or instructions")

    def to_draft(self) -> "InvoiceDraft":
        return InvoiceDraft(**self.model_dump())


class InvoiceDraft(ExtractedInvoice):
    """An extraction plus the fields our own code derives from it."""

    invoice_date: date | None = None
    due_date: date | None = None
    extraction_confidence: float = 1.0
    # True when the total was derived from line items rather than stated. The
    # derived figure always agrees with itself, so the cross-check that catches
    # a vendor's arithmetic error is unavailable for this invoice.
    total_reconstructed: bool = False

    def aggregated_quantities(self) -> dict[str, float]:
        """Total quantity per normalized item name.

        Invoices 1010 and 1013 bill the same product on several lines; stock
        must be checked against the sum, not each line independently.
        """
        totals: dict[str, float] = {}
        for item in self.line_items:
            if item.quantity is None:
                continue
            totals[item.normalized_name] = totals.get(item.normalized_name, 0.0) + item.quantity
        return totals

    def display_name(self, normalized: str) -> str:
        """Best original spelling for a normalized item key."""
        for item in self.line_items:
            if item.normalized_name == normalized:
                return item.name
        return normalized


class ExtractionAttempt(BaseModel):
    """One pass of the ingestion repair loop, kept for the audit trail."""

    round_index: int
    draft: InvoiceDraft
    issues: list[str] = Field(default_factory=list)
    accepted: bool = False


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

# Item-check status -> human wording. Shared so the terminal and the HTML
# report cannot drift into describing the same status differently.
ITEM_STATUS_LABEL = {
    "ok": "ok",
    "unknown": "not in catalogue",
    "out_of_stock": "zero stock",
    "shortfall": "insufficient stock",
}


class ItemCheck(BaseModel):
    """Result of checking one aggregated item against inventory."""

    invoice_name: str
    normalized_name: str
    quantity_requested: float
    matched_item: str | None = None
    stock_available: int | None = None
    status: str = "ok"  # ok | unknown | out_of_stock | shortfall


class ValidationReport(BaseModel):
    """Everything the validation stage learned about an invoice."""

    findings: list[Finding] = Field(default_factory=list)
    item_checks: list[ItemCheck] = Field(default_factory=list)
    agent_summary: str | None = None

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.CRITICAL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    def codes(self) -> set[FindingCode]:
        return {f.code for f in self.findings}


# --------------------------------------------------------------------------
# Approval
# --------------------------------------------------------------------------


class Decision(str, Enum):
    """Three outcomes. A binary approve/reject has nowhere to put an invoice
    that is neither clean nor clearly wrong -- e.g. the EUR-denominated 1014."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


class ApprovalDraft(BaseModel):
    """A VP decision before the critic has had a say."""

    decision: Decision
    rationale: str
    key_factors: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """The critic agent's challenge to a draft decision."""

    agrees: bool
    objections: list[str] = Field(default_factory=list)
    suggested_decision: Decision | None = None
    reasoning: str = ""


class ReflectionRound(BaseModel):
    """One draft/critique exchange, preserved so the reasoning is auditable."""

    round_index: int
    draft: ApprovalDraft
    critique: Critique | None = None
    revised: bool = False
    # What the model actually proposed, when policy overrode it. An auditor
    # needs to see that the guard rail fired, not just its result.
    overridden_from: Decision | None = None


class ApprovalDecision(BaseModel):
    """Final approval outcome plus the reasoning that produced it."""

    decision: Decision
    rationale: str
    key_factors: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    policy_reasons: list[str] = Field(default_factory=list)
    scrutiny_level: str = "standard"  # standard | heightened
    rounds: list[ReflectionRound] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Payment
# --------------------------------------------------------------------------


class PaymentReceipt(BaseModel):
    """Result of the mock payment call, or the reason it was skipped."""

    status: str  # success | skipped | duplicate | failed
    vendor: str | None = None
    amount: float | None = None
    currency: str | None = None
    reference: str | None = None
    message: str = ""
    paid_at: datetime | None = None


# --------------------------------------------------------------------------
# Run state
# --------------------------------------------------------------------------


class RunState(BaseModel):
    """The object threaded through every node of the LangGraph pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    invoice_path: str
    document: SourceDocument | None = None
    draft: InvoiceDraft | None = None
    extraction_attempts: list[ExtractionAttempt] = Field(default_factory=list)
    validation: ValidationReport | None = None
    approval: ApprovalDecision | None = None
    payment: PaymentReceipt | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Working state for the graph's two cycles. Carried on the state object so
    # each loop iteration is a real node transition rather than a hidden `for`.
    pending_draft: InvoiceDraft | None = None
    approval_draft: ApprovalDraft | None = None
    approval_overridden_from: Decision | None = None
    approval_rounds: list[ReflectionRound] = Field(default_factory=list)
    approval_notes: list[str] = Field(default_factory=list)
    last_critique: Critique | None = None
    approval_round_index: int = 0

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    @property
    def all_findings(self) -> list[Finding]:
        return list(self.validation.findings) if self.validation else []
