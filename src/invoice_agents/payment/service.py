"""Auditable local payment simulator.

The idempotency identity intentionally excludes source format, hash, amount, and
revision: representations and revisions of the same vendor invoice must not produce
two payments without a separately reviewed adjustment design.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import (
    DecisionKind,
    ExtractedInvoice,
    FinalDecision,
    Money,
    PaymentResult,
    PaymentStatus,
    ReviewRequest,
    RiskAssessment,
)


def payment_idempotency_key(invoice: ExtractedInvoice) -> str:
    """Build a stable identity across duplicate formats and revisions."""

    material = "|".join(
        [
            (invoice.vendor.normalized_value or "").casefold().strip(),
            (invoice.invoice_number.normalized_value or "").casefold().strip(),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _from_row(
    row: sqlite3.Row,
    *,
    duplicate: bool = False,
    attempted_case_id: str | None = None,
) -> PaymentResult:
    stored_status = PaymentStatus(str(row["status"]))
    status = (
        PaymentStatus.DUPLICATE
        if duplicate and stored_status is PaymentStatus.PAID
        else stored_status
    )
    return PaymentResult(
        payment_id=str(row["payment_id"]),
        case_id=attempted_case_id or str(row["case_id"]),
        idempotency_key=str(row["idempotency_key"]),
        status=status,
        vendor=str(row["vendor"]),
        amount=Money(amount=Decimal(str(row["amount"])), currency=str(row["currency"])),
        processed_at=datetime.fromisoformat(str(row["created_at"])),
        duplicate_of=str(row["payment_id"]) if duplicate else None,
        error=str(row["error"]) if row["error"] is not None else None,
    )


def mock_payment(
    case_id: str,
    invoice: ExtractedInvoice,
    store: WorkflowStore,
    workflow_db: Path,
    claim: ExecutionClaim,
    *,
    inject_failure: bool = False,
) -> PaymentResult:
    """Authorize and record payment from one transaction-local evidence snapshot."""

    # Local import avoids payment.service -> agents package -> team -> payment.service.
    from invoice_agents.agents.decision_rules import (
        AUTHORIZING_HUMAN_DECISIONS,
        unaddressed_blockers,
    )

    if store.path != workflow_db.resolve():
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "payment store and workflow database paths do not match",
            case_id=case_id,
            stop_reason="PAYMENT_DATABASE_MISMATCH",
        )
    if claim.case_id != case_id:
        raise InvoiceAgentsError(
            ErrorCategory.PAYMENT,
            "payment execution claim does not belong to this case",
            case_id=case_id,
            stop_reason="STALE_EXECUTION_CLAIM",
        )

    def not_eligible(
        key: str, vendor: str | None, error: str
    ) -> PaymentResult:
        return PaymentResult(
            payment_id=None,
            case_id=case_id,
            idempotency_key=key,
            status=PaymentStatus.NOT_ELIGIBLE,
            vendor=vendor,
            amount=None,
            processed_at=None,
            error=error,
        )

    with connect_database(workflow_db) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            authorization_time = datetime.now(UTC)
            case_row = connection.execute(
                "SELECT c.execution_token, c.execution_generation, c.execution_state, "
                "c.lease_expires_at FROM cases c WHERE c.case_id = ?",
                (case_id,),
            ).fetchone()
            current_claim = (
                case_row is not None
                and str(case_row["execution_token"]) == claim.token
                and int(case_row["execution_generation"]) == claim.generation
                and str(case_row["execution_state"]) == "RUNNING"
                and case_row["lease_expires_at"] is not None
                and datetime.fromisoformat(str(case_row["lease_expires_at"]))
                > authorization_time
            )
            if not current_claim:
                raise InvoiceAgentsError(
                    ErrorCategory.PAYMENT,
                    "payment requires the current unexpired execution claim",
                    case_id=case_id,
                    stop_reason="STALE_EXECUTION_CLAIM",
                    details={"execution_generation": claim.generation},
                )

            extraction_row = connection.execute(
                "SELECT payload_json FROM extractions WHERE case_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (case_id,),
            ).fetchone()
            if extraction_row is None:
                key = payment_idempotency_key(invoice)
                connection.rollback()
                return not_eligible(key, invoice.vendor.normalized_value, "invoice evidence is missing")
            persisted_invoice = ExtractedInvoice.model_validate_json(extraction_row["payload_json"])
            if persisted_invoice != invoice:
                key = payment_idempotency_key(persisted_invoice)
                connection.rollback()
                return not_eligible(
                    key,
                    persisted_invoice.vendor.normalized_value,
                    "payment invoice does not match the latest persisted extraction",
                )
            key = payment_idempotency_key(persisted_invoice)

            decision_row = connection.execute(
                "SELECT payload_json, decision_generation FROM final_decisions "
                "WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if decision_row is None or int(decision_row["decision_generation"]) != claim.generation:
                connection.rollback()
                return not_eligible(
                    key,
                    persisted_invoice.vendor.normalized_value,
                    "final decision is missing or stale for the current execution generation",
                )
            decision = FinalDecision.model_validate_json(decision_row["payload_json"])
            if (
                decision.decision is not DecisionKind.APPROVE
                or not decision.payment_eligible
            ):
                connection.rollback()
                return not_eligible(
                    key,
                    persisted_invoice.vendor.normalized_value,
                    "case lacks an APPROVE decision with payment_eligible=true",
                )

            risk_row = connection.execute(
                "SELECT payload_json FROM comparison_results "
                "WHERE case_id = ? AND comparison_type = 'risk' "
                "ORDER BY created_at DESC LIMIT 1",
                (case_id,),
            ).fetchone()
            if risk_row is None:
                connection.rollback()
                return not_eligible(
                    key,
                    persisted_invoice.vendor.normalized_value,
                    "latest risk assessment is missing",
                )
            risk = RiskAssessment.model_validate_json(risk_row["payload_json"])

            review_row = connection.execute(
                "SELECT payload_json FROM review_requests WHERE case_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (case_id,),
            ).fetchone()
            review = (
                ReviewRequest.model_validate_json(review_row["payload_json"])
                if review_row is not None
                else None
            )
            human = review.human_decision if review is not None else None
            if review is not None and (
                review.status != "RESOLVED"
                or human is None
                or human.decision not in AUTHORIZING_HUMAN_DECISIONS
            ):
                connection.rollback()
                return not_eligible(
                    key,
                    persisted_invoice.vendor.normalized_value,
                    "human review is unresolved or does not authorize approval",
                )
            if decision.human_outcome != human:
                connection.rollback()
                return not_eligible(
                    key,
                    persisted_invoice.vendor.normalized_value,
                    "final decision does not match the latest human review decision",
                )
            remaining_blockers = unaddressed_blockers(risk, human)
            if remaining_blockers:
                connection.rollback()
                return not_eligible(
                    key,
                    persisted_invoice.vendor.normalized_value,
                    "current blocking evidence is not explicitly authorized: "
                    f"{[blocker.blocker_id for blocker in remaining_blockers]}",
                )

            vendor = persisted_invoice.vendor.normalized_value
            currency = persisted_invoice.currency.normalized_value
            amount = persisted_invoice.declared_total
            if not vendor or not currency or amount is None or amount <= 0:
                connection.rollback()
                return not_eligible(
                    key,
                    vendor,
                    "payment requires a vendor, currency, and positive declared total",
                )
            prior = connection.execute(
                "SELECT * FROM payments WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if prior is not None:
                connection.rollback()
                return _from_row(prior, duplicate=True, attempted_case_id=case_id)
            payment_id = f"pay_{uuid4().hex}"
            created_at = datetime.now(UTC)
            status = PaymentStatus.FAILED if inject_failure else PaymentStatus.PAID
            error = "injected mock-payment failure" if inject_failure else None
            connection.execute(
                "INSERT INTO payments("
                "payment_id, case_id, idempotency_key, vendor, amount, currency, status, error, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    payment_id,
                    case_id,
                    key,
                    vendor,
                    str(amount),
                    currency,
                    status,
                    error,
                    created_at.isoformat(),
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return PaymentResult(
        payment_id=payment_id,
        case_id=case_id,
        idempotency_key=key,
        status=status,
        vendor=vendor,
        amount=Money(amount=amount, currency=currency),
        processed_at=created_at,
        error=error,
    )
