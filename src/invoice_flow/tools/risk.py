"""Risk signals that inform how hard the approval agent should look.

These are not by themselves grounds for rejection. They set the *scrutiny
level*: an invoice carrying urgency pressure and an unparseable due date
deserves a closer read than a routine one, even if every number reconciles.
"""

from __future__ import annotations

import re

from ..config import APPROVAL_THRESHOLD, BASE_CURRENCY, THRESHOLD_PROXIMITY_RATIO
from ..models import Finding, FindingCode, InvoiceDraft, Severity

# Phrases that manufacture time pressure. Legitimate vendors state terms; they
# do not demand immediate payment in capitals.
_URGENCY_PATTERNS = (
    r"\burgent\b",
    r"\bimmediate(?:ly)?\b",
    r"\bpay\s+(?:now|at\s+once|today)\b",
    r"\bavoid\s+penalt",
    r"\bfinal\s+notice\b",
    r"\bact\s+now\b",
    r"!!!",
)

# Payment-rail changes are the classic vector for business email compromise.
_WIRE_PATTERNS = (
    r"\bwire\s+transfer\b",
    r"\bwire\s+the\s+funds\b",
    r"\bupdated?\s+bank(?:ing)?\s+(?:details|information)\b",
    r"\bnew\s+account\s+(?:number|details)\b",
)


def _search(patterns: tuple[str, ...], text: str) -> list[str]:
    hits = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hits.append(match.group(0))
    return hits


def check_currency(draft: InvoiceDraft) -> list[Finding]:
    """Flag any invoice not denominated in the base currency.

    Invoice 1014 is billed in EUR. Neither paying it at face value nor
    rejecting it is right -- it needs a human with an FX rate, which is exactly
    what the ESCALATED outcome exists for.
    """
    currency = (draft.currency or "").strip().upper()
    if currency and currency != BASE_CURRENCY:
        return [
            Finding(
                code=FindingCode.CURRENCY_NON_BASE,
                severity=Severity.WARNING,
                message=(
                    f"Invoice is denominated in {currency}, not {BASE_CURRENCY}. "
                    "Settlement requires an FX rate and a treasury decision that "
                    "this system is not authorised to make."
                ),
                detail={"currency": currency, "base_currency": BASE_CURRENCY},
                source="risk",
            )
        ]
    return []


def check_threshold(draft: InvoiceDraft) -> list[Finding]:
    """Position the invoice total against the VP approval threshold.

    Both sides matter. Over the threshold means heightened scrutiny. *Just
    under* it is its own signal -- $9,900 and $9,975 in the sample set sit
    within 1% of the $10,000 line, the shape invoice-splitting takes.
    """
    total = draft.total
    if total is None or total <= 0:
        return []

    if total > APPROVAL_THRESHOLD:
        return [
            Finding(
                code=FindingCode.AMOUNT_OVER_THRESHOLD,
                severity=Severity.WARNING,
                message=(
                    f"Total of ${total:,.2f} exceeds the ${APPROVAL_THRESHOLD:,.0f} "
                    "VP approval threshold and requires heightened scrutiny."
                ),
                detail={"total": total, "threshold": APPROVAL_THRESHOLD},
                source="risk",
            )
        ]

    margin = APPROVAL_THRESHOLD - total
    if margin <= APPROVAL_THRESHOLD * THRESHOLD_PROXIMITY_RATIO:
        return [
            Finding(
                code=FindingCode.AMOUNT_JUST_UNDER_THRESHOLD,
                severity=Severity.WARNING,
                message=(
                    f"Total of ${total:,.2f} falls just ${margin:,.2f} below the "
                    f"${APPROVAL_THRESHOLD:,.0f} approval threshold. Amounts clustering "
                    "beneath an approval limit can indicate deliberate splitting to "
                    "avoid review."
                ),
                detail={
                    "total": total,
                    "threshold": APPROVAL_THRESHOLD,
                    "margin": round(margin, 2),
                },
                source="risk",
            )
        ]
    return []


def check_pressure_language(draft: InvoiceDraft, document_text: str) -> list[Finding]:
    """Detect urgency pressure and payment-rail redirection in the document."""
    findings: list[Finding] = []
    haystack = "\n".join(filter(None, [draft.notes, draft.payment_terms, document_text]))

    urgency = _search(_URGENCY_PATTERNS, haystack)
    if urgency:
        findings.append(
            Finding(
                code=FindingCode.URGENCY_PRESSURE,
                severity=Severity.WARNING,
                message=(
                    "Invoice uses urgency pressure language "
                    f"({', '.join(repr(h) for h in urgency)}). Manufactured time "
                    "pressure is a standard tactic for rushing payments past review."
                ),
                detail={"matches": urgency},
                source="risk",
            )
        )

    wire = _search(_WIRE_PATTERNS, haystack)
    if wire:
        findings.append(
            Finding(
                code=FindingCode.WIRE_TRANSFER_REQUEST,
                severity=Severity.WARNING,
                message=(
                    "Invoice requests or prefers wire transfer "
                    f"({', '.join(repr(h) for h in wire)}). Wire payments are "
                    "irreversible and are the usual target of payment fraud."
                ),
                detail={"matches": wire},
                source="risk",
            )
        )

    return findings


def assess(draft: InvoiceDraft, document_text: str = "") -> list[Finding]:
    """Run every risk check and return the combined findings."""
    return [
        *check_currency(draft),
        *check_threshold(draft),
        *check_pressure_language(draft, document_text),
    ]
