"""Payment agent: execute or decline, and record either way.

Every outcome writes to the ledger, not just the payments. A rejection nobody
recorded is a rejection that gets re-submitted next week and paid on the second
attempt.

The duplicate guard is the highest-value control here. The sample set ships
invoice 1004 twice -- once at $1,890 and once revised to $5,940, under the same
invoice number. Paying both is the exact failure a five-day manual queue
produces, and it costs real money.
"""

from __future__ import annotations

from typing import Any

from ..models import ApprovalDecision, Decision, InvoiceDraft, PaymentReceipt
from ..tools.inventory import InventoryRepository
from ..tools.payment import execute_payment, payable_fingerprint


def run_payment(
    draft: InvoiceDraft,
    approval: ApprovalDecision,
    repo: InventoryRepository,
    run_id: str,
    source_path: str | None = None,
    tracer: Any = None,
) -> PaymentReceipt:
    """Settle an approved invoice, or record why it was not settled."""

    if approval.decision is not Decision.APPROVED:
        reason = (
            "Rejected: not authorised for payment."
            if approval.decision is Decision.REJECTED
            else "Escalated: held pending human review."
        )
        receipt = PaymentReceipt(
            status="skipped",
            vendor=draft.vendor_name,
            amount=draft.total,
            currency=draft.currency,
            message=f"{reason} {approval.rationale}".strip(),
        )
        if tracer is not None:
            tracer.emit(
                "payment.skipped",
                decision=approval.decision.value,
                invoice_number=draft.invoice_number,
                amount=draft.total,
                rationale=approval.rationale,
            )
        _record(repo, run_id, draft, approval, receipt, source_path)
        return receipt

    # Approved -- but check the ledger before moving money.
    prior = repo.prior_payments(draft.invoice_number)
    if prior:
        previous = prior[-1]
        receipt = PaymentReceipt(
            status="duplicate",
            vendor=draft.vendor_name,
            amount=draft.total,
            currency=draft.currency,
            reference=previous.get("payment_reference"),
            message=(
                f"Payment blocked: invoice {draft.invoice_number} was already paid on "
                f"{previous.get('processed_at')} "
                f"(${previous.get('amount') or 0:,.2f}, ref {previous.get('payment_reference')}). "
                f"{len(prior)} prior payment(s) on record."
            ),
        )
        if tracer is not None:
            tracer.emit(
                "payment.duplicate_blocked",
                invoice_number=draft.invoice_number,
                prior_payments=len(prior),
                prior_reference=previous.get("payment_reference"),
                amount=draft.total,
            )
        _record(repo, run_id, draft, approval, receipt, source_path)
        return receipt

    receipt = execute_payment(
        vendor=draft.vendor_name,
        amount=draft.total,
        currency=draft.currency,
        invoice_number=draft.invoice_number,
    )
    if tracer is not None:
        tracer.emit(
            "payment.executed",
            status=receipt.status,
            invoice_number=draft.invoice_number,
            vendor=receipt.vendor,
            amount=receipt.amount,
            reference=receipt.reference,
        )
    _record(repo, run_id, draft, approval, receipt, source_path)
    return receipt


def _record(
    repo: InventoryRepository,
    run_id: str,
    draft: InvoiceDraft,
    approval: ApprovalDecision,
    receipt: PaymentReceipt,
    source_path: str | None,
) -> None:
    repo.record_ledger_entry(
        run_id=run_id,
        invoice_number=draft.invoice_number,
        vendor=draft.vendor_name,
        amount=draft.total,
        currency=draft.currency,
        decision=approval.decision.value,
        payment_status=receipt.status,
        payment_reference=receipt.reference,
        # Recorded for every outcome, not just payments: a rejected invoice
        # still establishes what was seen under that number, which is what makes
        # a later conflicting version detectable.
        content_hash=payable_fingerprint(draft),
        source_path=source_path,
    )
