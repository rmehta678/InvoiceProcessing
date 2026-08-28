"""Deterministic reconciliation of an invoice's own numbers.

This module is the backbone of the ingestion self-correction loop. Critiquing an
extraction with a second LLM opinion tends to drift -- two models can disagree
forever. Arithmetic cannot be argued with: either the line items sum to the
stated subtotal or they do not. Grounding the critique here is what makes the
repair loop converge.

It also catches real defects in the source data. Invoice 1013 states a total
$50 higher than its own subtotal plus tax, and invoice 1009's line items sum to
-$250 against a stated subtotal of $1,000.
"""

from __future__ import annotations

from ..config import ARITHMETIC_TOLERANCE
from ..models import Finding, FindingCode, InvoiceDraft, Severity, money


def _close(a: float, b: float, tolerance: float = ARITHMETIC_TOLERANCE) -> bool:
    return abs(a - b) <= tolerance


def computed_subtotal(draft: InvoiceDraft) -> float | None:
    """Sum the line items, preferring a stated amount over qty x unit_price."""
    total = 0.0
    contributed = False
    for item in draft.line_items:
        if item.amount is not None:
            total += item.amount
            contributed = True
        elif item.quantity is not None and item.unit_price is not None:
            total += item.quantity * item.unit_price
            contributed = True
    return total if contributed else None


def reconstruct_total(draft: InvoiceDraft) -> bool:
    """Fill in a missing total from the line items, tax, and shipping.

    Returns True if a total was reconstructed.

    Must run only *after* the extraction repair loop has settled. Filling the
    total mid-loop would leave `check_arithmetic` comparing a derived figure
    against itself -- which always reconciles, and would mask exactly the
    discrepancies the loop exists to surface ($110 on invoice 1007, $50 on
    1013). Deriving a total is a fallback for an incomplete document, never a
    substitute for one the vendor stated.
    """
    if draft.total is not None:
        return False

    base = draft.subtotal if draft.subtotal is not None else computed_subtotal(draft)
    if base is None:
        return False

    draft.total = base + (draft.tax_amount or 0.0) + (draft.shipping or 0.0)
    draft.total_reconstructed = True
    return True


def check_arithmetic(draft: InvoiceDraft) -> list[Finding]:
    """Reconcile line amounts, subtotal, tax, and total against each other.

    Returns findings describing every inconsistency found. An invoice with no
    numbers to check produces no findings -- absence is handled by the
    completeness checks in `check_data_integrity`.
    """
    findings: list[Finding] = []

    # -- per-line: quantity x unit price should equal the stated amount ----
    for index, item in enumerate(draft.line_items):
        if item.amount is None or item.quantity is None or item.unit_price is None:
            continue
        expected = item.quantity * item.unit_price
        if not _close(expected, item.amount):
            findings.append(
                Finding(
                    code=FindingCode.LINE_AMOUNT_MISMATCH,
                    severity=Severity.WARNING,
                    message=(
                        f"Line {index + 1} ({item.name}): {item.quantity:g} x "
                        f"{money(item.unit_price)} = {money(expected)}, "
                        f"but the invoice states {money(item.amount)}."
                    ),
                    detail={
                        "item": item.name,
                        "expected": round(expected, 2),
                        "stated": round(item.amount, 2),
                        "difference": round(item.amount - expected, 2),
                    },
                    source="arithmetic",
                )
            )

    # -- subtotal ----------------------------------------------------------
    derived = computed_subtotal(draft)
    if derived is not None and draft.subtotal is not None and not _close(derived, draft.subtotal):
        findings.append(
            Finding(
                code=FindingCode.SUBTOTAL_MISMATCH,
                severity=Severity.CRITICAL,
                message=(
                    f"Line items sum to {money(derived)} but the stated subtotal "
                    f"is {money(draft.subtotal)} "
                    f"(difference {money(draft.subtotal - derived)})."
                ),
                detail={
                    "computed": round(derived, 2),
                    "stated": round(draft.subtotal, 2),
                    "difference": round(draft.subtotal - derived, 2),
                },
                source="arithmetic",
            )
        )

    subtotal = draft.subtotal if draft.subtotal is not None else derived

    # -- tax ---------------------------------------------------------------
    if subtotal is not None and draft.tax_rate is not None and draft.tax_amount is not None:
        expected_tax = subtotal * draft.tax_rate
        # Percentage tax on a large subtotal rounds; scale the tolerance a
        # little rather than flagging half a cent.
        tax_tolerance = max(ARITHMETIC_TOLERANCE, abs(subtotal) * 0.0001)
        if not _close(expected_tax, draft.tax_amount, tax_tolerance):
            findings.append(
                Finding(
                    code=FindingCode.TAX_MISMATCH,
                    severity=Severity.WARNING,
                    message=(
                        f"Tax of {draft.tax_rate:.2%} on {money(subtotal)} should be "
                        f"{money(expected_tax)}, but the invoice states "
                        f"{money(draft.tax_amount)}."
                    ),
                    detail={
                        "expected": round(expected_tax, 2),
                        "stated": round(draft.tax_amount, 2),
                        "tax_rate": draft.tax_rate,
                    },
                    source="arithmetic",
                )
            )

    # -- total -------------------------------------------------------------
    if subtotal is not None and draft.total is not None:
        expected_total = subtotal + (draft.tax_amount or 0.0) + (draft.shipping or 0.0)
        if not _close(expected_total, draft.total):
            findings.append(
                Finding(
                    code=FindingCode.TOTAL_MISMATCH,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Subtotal {money(subtotal)} + tax "
                        f"{money(draft.tax_amount or 0.0)} + shipping "
                        f"{money(draft.shipping or 0.0)} = {money(expected_total)}, "
                        f"but the stated total is {money(draft.total)} "
                        f"(unexplained {money(draft.total - expected_total)})."
                    ),
                    detail={
                        "computed": round(expected_total, 2),
                        "stated": round(draft.total, 2),
                        "difference": round(draft.total - expected_total, 2),
                    },
                    source="arithmetic",
                )
            )

    return findings


def check_data_integrity(draft: InvoiceDraft) -> list[Finding]:
    """Structural sanity checks that do not need the inventory database."""
    findings: list[Finding] = []

    if not draft.line_items:
        findings.append(
            Finding(
                code=FindingCode.NO_LINE_ITEMS,
                severity=Severity.CRITICAL,
                message="No line items could be extracted from this document.",
                source="integrity",
            )
        )

    if not (draft.vendor_name or "").strip():
        findings.append(
            Finding(
                code=FindingCode.VENDOR_MISSING,
                severity=Severity.CRITICAL,
                message="Vendor name is missing; payment cannot be directed to a payee.",
                source="integrity",
            )
        )

    if not (draft.invoice_number or "").strip():
        findings.append(
            Finding(
                code=FindingCode.INVOICE_NUMBER_MISSING,
                severity=Severity.WARNING,
                message="Invoice number is missing; duplicate payments cannot be detected.",
                source="integrity",
            )
        )

    for index, item in enumerate(draft.line_items):
        if item.quantity is None:
            findings.append(
                Finding(
                    code=FindingCode.QUANTITY_INVALID,
                    severity=Severity.CRITICAL,
                    message=f"Line {index + 1} ({item.name}) has no quantity.",
                    detail={"item": item.name},
                    source="integrity",
                )
            )
        elif item.quantity <= 0:
            findings.append(
                Finding(
                    code=FindingCode.QUANTITY_INVALID,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Line {index + 1} ({item.name}) has a non-positive quantity "
                        f"of {item.quantity:g}. A purchase invoice cannot bill for "
                        "zero or negative units."
                    ),
                    detail={"item": item.name, "quantity": item.quantity},
                    source="integrity",
                )
            )
        elif item.quantity != int(item.quantity):
            findings.append(
                Finding(
                    code=FindingCode.QUANTITY_NON_INTEGER,
                    severity=Severity.WARNING,
                    message=(
                        f"Line {index + 1} ({item.name}) has a fractional quantity "
                        f"of {item.quantity:g}; inventory is tracked in whole units."
                    ),
                    detail={"item": item.name, "quantity": item.quantity},
                    source="integrity",
                )
            )

        if item.unit_price is not None and item.unit_price < 0:
            findings.append(
                Finding(
                    code=FindingCode.UNIT_PRICE_INVALID,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Line {index + 1} ({item.name}) has a negative unit price "
                        f"of {money(item.unit_price)}."
                    ),
                    detail={"item": item.name, "unit_price": item.unit_price},
                    source="integrity",
                )
            )

    if draft.total is None and computed_subtotal(draft) is None and draft.subtotal is None:
        findings.append(
            Finding(
                code=FindingCode.TOTAL_MISSING,
                severity=Severity.CRITICAL,
                message=(
                    "No total is stated and none can be derived: the invoice has no "
                    "subtotal and no priced line items. There is no amount to pay."
                ),
                source="integrity",
            )
        )

    if draft.total is not None and draft.total <= 0:
        findings.append(
            Finding(
                code=FindingCode.TOTAL_NON_POSITIVE,
                severity=Severity.CRITICAL,
                message=(
                    f"Invoice total is {money(draft.total)}. A payable invoice must "
                    "have a positive total; this may be a credit note filed as an invoice."
                ),
                detail={"total": draft.total},
                source="integrity",
            )
        )

    return findings


def summarise_for_repair(findings: list[Finding]) -> list[str]:
    """Phrase arithmetic findings as instructions for a re-extraction attempt.

    Only issues an extractor could plausibly have caused are worth repeating --
    a genuinely wrong invoice should not send the loop spinning.
    """
    repairable = {
        FindingCode.LINE_AMOUNT_MISMATCH,
        FindingCode.SUBTOTAL_MISMATCH,
        FindingCode.TAX_MISMATCH,
        FindingCode.TOTAL_MISMATCH,
        FindingCode.NO_LINE_ITEMS,
        FindingCode.QUANTITY_INVALID,
    }
    return [f.message for f in findings if f.code in repairable]
