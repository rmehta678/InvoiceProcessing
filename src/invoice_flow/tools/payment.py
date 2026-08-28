"""Mock payment rail.

Stands in for the banking API described in the case. It is deliberately the
only place in the codebase that "spends money", so the guard rails around it
are easy to audit: it refuses non-positive amounts and refuses to pay a payee
it cannot name.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..config import BASE_CURRENCY
from ..models import PaymentReceipt, money

if TYPE_CHECKING:
    from ..models import InvoiceDraft


def payable_fingerprint(draft: "InvoiceDraft") -> str:
    """Stable hash of the facts that determine what gets paid.

    Covers vendor, aggregated quantities per item, and total. Deliberately
    excludes file format, notes, addresses, and line-item ordering, so the same
    invoice arriving as a PDF and as a text body hashes identically -- that is
    one document seen twice, not a conflict.

    An absent currency is read as the base currency for the same reason. Most
    sample documents show only "$" and no ISO code, so the extractor returns
    "USD" on one pass and null on the next -- both faithful to the document.
    Hashing those differently made a second read of INV-1001 look like a
    conflicting revision and escalated a clean invoice. Optional-field presence
    is not a payable fact; a genuinely different currency still is.

    A *different* hash under the same invoice number means the payable facts
    changed, which is a revision and needs a human.
    """
    quantities = sorted(draft.aggregated_quantities().items())
    seed = "|".join(
        [
            (draft.vendor_name or "").strip().casefold(),
            f"{draft.total:.2f}" if draft.total is not None else "none",
            (draft.currency or "").strip().upper() or BASE_CURRENCY,
            ";".join(f"{name}={qty:g}" for name, qty in quantities),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def payment_reference(invoice_number: str | None, vendor: str | None, amount: float) -> str:
    """Deterministic reference derived from the invoice's identity.

    Because it is a pure function of (invoice number, vendor, amount), the same
    invoice always produces the same reference -- which is what makes
    duplicate payments detectable in the ledger.
    """
    seed = f"{invoice_number or 'UNKNOWN'}|{vendor or 'UNKNOWN'}|{amount:.2f}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12].upper()
    return f"PAY-{digest}"


def execute_payment(
    vendor: str | None,
    amount: float | None,
    currency: str | None = "USD",
    invoice_number: str | None = None,
) -> PaymentReceipt:
    """Attempt a payment, returning a receipt describing what happened."""
    if not (vendor or "").strip():
        return PaymentReceipt(
            status="failed",
            vendor=vendor,
            amount=amount,
            currency=currency,
            message="Refused: no payee name. Funds cannot be directed to an unnamed vendor.",
        )

    if amount is None or amount <= 0:
        return PaymentReceipt(
            status="failed",
            vendor=vendor,
            amount=amount,
            currency=currency,
            message=f"Refused: invalid payment amount ({amount}). Must be positive.",
        )

    # The mock rail: nothing to call, so the payment simply succeeds.
    return PaymentReceipt(
        status="success",
        vendor=vendor,
        amount=amount,
        currency=currency,
        reference=payment_reference(invoice_number, vendor, amount),
        message=f"Paid {money(amount, currency)} to {vendor}",
        paid_at=datetime.now(timezone.utc),
    )
