"""Auditable local payment simulator.

The idempotency identity intentionally excludes source format, hash, amount, and
revision: representations and revisions of the same vendor invoice must not produce
two payments without a separately reviewed adjustment design.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import (
    Critique,
    DecisionKind,
    ExtractedInvoice,
    FinalDecision,
    Money,
    PaymentResult,
    PaymentStatus,
    ReviewRequest,
    RiskAssessment,
)


@dataclass(frozen=True, slots=True)
class _AuthorizationSnapshot:
    invoice: ExtractedInvoice
    decision: FinalDecision
    risk: RiskAssessment
    review: ReviewRequest | None


class _AuthorizationSnapshotError(Exception):
    pass


def _load_authorization_snapshot(
    connection: sqlite3.Connection, case_id: str, generation: int
) -> _AuthorizationSnapshot:
    """Load and validate one complete generation-bound approval snapshot."""

    from invoice_agents.agents.decision_rules import validate_final_decision

    extraction_row = connection.execute(
        "SELECT payload_json, execution_generation FROM extractions WHERE case_id = ? "
        "ORDER BY version DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    if extraction_row is None or int(extraction_row["execution_generation"]) != generation:
        raise _AuthorizationSnapshotError("latest extraction is missing or stale")
    invoice = ExtractedInvoice.model_validate_json(extraction_row["payload_json"])

    for table, extra_predicate, label in (
        ("identity_results", "", "identity evidence"),
        (
            "comparison_results",
            "AND comparison_type = 'inventory'",
            "inventory comparison",
        ),
        ("critique_results", "", "critique"),
    ):
        row = connection.execute(
            f"SELECT execution_generation FROM {table} WHERE case_id = ? "
            f"{extra_predicate} ORDER BY created_at DESC LIMIT 1",
            (case_id,),
        ).fetchone()
        if row is None or int(row["execution_generation"]) != generation:
            raise _AuthorizationSnapshotError(f"latest {label} is missing or stale")

    risk_row = connection.execute(
        "SELECT payload_json, execution_generation FROM comparison_results "
        "WHERE case_id = ? AND comparison_type = 'risk' ORDER BY created_at DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    if risk_row is None or int(risk_row["execution_generation"]) != generation:
        raise _AuthorizationSnapshotError("latest risk assessment is missing or stale")
    risk = RiskAssessment.model_validate_json(risk_row["payload_json"])

    critique_row = connection.execute(
        "SELECT payload_json FROM critique_results WHERE case_id = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    if critique_row is None:
        raise _AuthorizationSnapshotError("latest critique is missing")
    critique = Critique.model_validate_json(critique_row["payload_json"])

    review_row = connection.execute(
        "SELECT payload_json, execution_generation FROM review_requests WHERE case_id = ? "
        "ORDER BY sequence DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    if review_row is not None and int(review_row["execution_generation"]) != generation:
        raise _AuthorizationSnapshotError("latest review is stale")
    review = (
        ReviewRequest.model_validate_json(review_row["payload_json"])
        if review_row is not None
        else None
    )

    decision_row = connection.execute(
        "SELECT payload_json, decision_generation FROM final_decisions WHERE case_id = ?",
        (case_id,),
    ).fetchone()
    if decision_row is None or int(decision_row["decision_generation"]) != generation:
        raise _AuthorizationSnapshotError("final decision is missing or stale")
    decision = FinalDecision.model_validate_json(decision_row["payload_json"])
    if decision.decision is not DecisionKind.APPROVE or not decision.payment_eligible:
        raise _AuthorizationSnapshotError(
            "case lacks an APPROVE decision with payment_eligible=true"
        )
    try:
        validate_final_decision(
            decision.decision,
            decision.payment_eligible,
            risk,
            critique,
            review,
            case_id=case_id,
        )
    except InvoiceAgentsError as exc:
        raise _AuthorizationSnapshotError(exc.message) from exc
    human = review.human_decision if review is not None else None
    if decision.human_outcome != human:
        raise _AuthorizationSnapshotError(
            "final decision does not match the latest human review decision"
        )
    if decision.critic_disposition is not critique.recommended_disposition:
        raise _AuthorizationSnapshotError(
            "final decision does not match the latest independent critique"
        )
    return _AuthorizationSnapshot(invoice, decision, risk, review)


def _validate_paid_ledger_source(connection: sqlite3.Connection, row: sqlite3.Row) -> None:
    """Reject a PAID idempotency row whose source authorization has drifted."""

    case_id = str(row["case_id"])
    generation = int(row["decision_generation"])
    try:
        snapshot = _load_authorization_snapshot(connection, case_id, generation)
    except (InvoiceAgentsError, _AuthorizationSnapshotError, ValueError) as exc:
        raise InvoiceAgentsError(
            ErrorCategory.PAYMENT,
            f"paid ledger source snapshot is inconsistent for case {case_id}",
            case_id=case_id,
            stop_reason="PAYMENT_LEDGER_INCONSISTENT",
            details={"reason": str(exc)},
        ) from exc
    invoice = snapshot.invoice
    valid = (
        payment_idempotency_key(invoice) == str(row["idempotency_key"])
        and invoice.vendor.normalized_value == str(row["vendor"])
        and invoice.currency.normalized_value == str(row["currency"])
        and invoice.declared_total is not None
        and invoice.declared_total == Decimal(str(row["amount"]))
    )
    if not valid:
        raise InvoiceAgentsError(
            ErrorCategory.PAYMENT,
            f"paid ledger identity does not match source case {case_id}",
            case_id=case_id,
            stop_reason="PAYMENT_LEDGER_INCONSISTENT",
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

    def not_eligible(key: str, vendor: str | None, error: str) -> PaymentResult:
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
                and datetime.fromisoformat(str(case_row["lease_expires_at"])) > authorization_time
            )
            if not current_claim:
                raise InvoiceAgentsError(
                    ErrorCategory.PAYMENT,
                    "payment requires the current unexpired execution claim",
                    case_id=case_id,
                    stop_reason="STALE_EXECUTION_CLAIM",
                    details={"execution_generation": claim.generation},
                )

            try:
                snapshot = _load_authorization_snapshot(connection, case_id, claim.generation)
            except _AuthorizationSnapshotError as exc:
                key = payment_idempotency_key(invoice)
                connection.rollback()
                return not_eligible(key, invoice.vendor.normalized_value, str(exc))
            persisted_invoice = snapshot.invoice
            if persisted_invoice != invoice:
                key = payment_idempotency_key(persisted_invoice)
                connection.rollback()
                return not_eligible(
                    key,
                    persisted_invoice.vendor.normalized_value,
                    "payment invoice does not match the latest persisted extraction",
                )
            key = payment_idempotency_key(persisted_invoice)

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
                if PaymentStatus(str(prior["status"])) is PaymentStatus.PAID:
                    _validate_paid_ledger_source(connection, prior)
                connection.rollback()
                return _from_row(
                    prior,
                    duplicate=PaymentStatus(str(prior["status"])) is PaymentStatus.PAID,
                    attempted_case_id=case_id,
                )
            payment_id = f"pay_{uuid4().hex}"
            created_at = datetime.now(UTC)
            status = PaymentStatus.FAILED if inject_failure else PaymentStatus.PAID
            error = "injected mock-payment failure" if inject_failure else None
            connection.execute(
                "INSERT INTO payments("
                "payment_id, case_id, idempotency_key, vendor, amount, currency, status, error, "
                "created_at, decision_generation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    claim.generation,
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
