"""Auditable local payment simulator.

The idempotency identity intentionally excludes source format, hash, amount, and
revision: representations and revisions of the same vendor invoice must not produce
two payments without a separately reviewed adjustment design.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import (
    ExecutionClaim,
    WorkflowStore,
    load_generation_evidence_snapshot,
    parse_canonical_utc,
)
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.evidence_snapshot import EvidenceSnapshotError, validate_review_snapshot
from invoice_agents.models import (
    DecisionKind,
    ExtractedInvoice,
    FinalDecision,
    HumanDecision,
    Money,
    PaymentResult,
    PaymentStatus,
    ReviewRequest,
    RiskAssessment,
)
from invoice_agents.payment.identity import payment_identity_key


@dataclass(frozen=True, slots=True)
class _AuthorizationSnapshot:
    invoice: ExtractedInvoice
    decision: FinalDecision
    risk: RiskAssessment
    review: ReviewRequest | None
    evidence_snapshot_digest: str


class _AuthorizationSnapshotError(Exception):
    pass


def _reconcile_review_authorization(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    case_id: str,
) -> ReviewRequest:
    """Reconcile embedded review JSON with authoritative relational HITL rows."""

    from invoice_agents.agents.decision_rules import AUTHORIZING_HUMAN_DECISIONS

    inconsistent = _AuthorizationSnapshotError("review authorization records are inconsistent")
    try:
        review = ReviewRequest.model_validate_json(row["payload_json"])
    except ValueError as exc:
        raise inconsistent from exc
    human = review.human_decision
    if (
        review.case_id != case_id
        or review.review_id != row["review_id"]
        or row["case_id"] != case_id
        or row["status"] != "RESOLVED"
        or review.status != "RESOLVED"
        or row["resolved_at"] is None
        or human is None
    ):
        raise inconsistent
    human_rows = connection.execute(
        "SELECT review_id, reviewer, decision, reason, payload_json, decided_at "
        "FROM human_decisions WHERE review_id = ?",
        (review.review_id,),
    ).fetchall()
    if len(human_rows) != 1:
        raise inconsistent
    human_row = human_rows[0]
    try:
        relational_human = HumanDecision.model_validate_json(human_row["payload_json"])
    except ValueError as exc:
        raise inconsistent from exc
    relational_columns_match = (
        human_row["review_id"] == review.review_id
        and human_row["reviewer"] == relational_human.reviewer
        and human_row["decision"] == relational_human.decision
        and human_row["reason"] == relational_human.reason
        and human_row["decided_at"] == relational_human.decided_at.isoformat()
        and row["resolved_at"] == relational_human.decided_at.isoformat()
    )
    raw_blockers = review.evidence_bundle.get("blocking_evidence")
    if not isinstance(raw_blockers, list):
        raise inconsistent
    package_blocker_ids = [
        entry.get("blocker_id") if isinstance(entry, dict) else None for entry in raw_blockers
    ]
    blocker_linkage_valid = (
        all(isinstance(blocker_id, str) and blocker_id for blocker_id in package_blocker_ids)
        and len(set(package_blocker_ids)) == len(package_blocker_ids)
        and len(set(human.addressed_blocker_ids)) == len(human.addressed_blocker_ids)
        and set(human.addressed_blocker_ids).issubset(set(package_blocker_ids))
        and (not human.addressed_blocker_ids or human.decision in AUTHORIZING_HUMAN_DECISIONS)
    )
    if (
        relational_human != human
        or relational_human.review_id != review.review_id
        or not relational_columns_match
        or not blocker_linkage_valid
    ):
        raise inconsistent
    return review


def _load_authorization_snapshot(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
    settings: Settings,
) -> _AuthorizationSnapshot:
    """Load and validate one complete generation-bound approval snapshot."""

    from invoice_agents.agents.decision_rules import validate_final_decision

    try:
        evidence = load_generation_evidence_snapshot(
            connection,
            case_id,
            generation,
            settings,
        )
    except EvidenceSnapshotError as exc:
        raise _AuthorizationSnapshotError(f"evidence snapshot is invalid: {exc}") from exc
    invoice = evidence.invoice
    risk = evidence.risk
    critique = evidence.critique

    review_row = connection.execute(
        "SELECT review_id, case_id, status, payload_json, resolved_at, "
        "execution_generation, evidence_snapshot_digest FROM review_requests WHERE case_id = ? "
        "ORDER BY sequence DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    if review_row is not None and int(review_row["execution_generation"]) != generation:
        raise _AuthorizationSnapshotError("latest review is stale")
    review = (
        _reconcile_review_authorization(connection, review_row, case_id)
        if review_row is not None
        else None
    )
    if review is not None:
        try:
            validate_review_snapshot(review, evidence)
        except EvidenceSnapshotError as exc:
            raise _AuthorizationSnapshotError(str(exc)) from exc
        if review_row["evidence_snapshot_digest"] != evidence.digest:
            raise _AuthorizationSnapshotError("review snapshot digest does not match evidence")

    decision_row = connection.execute(
        "SELECT payload_json, decision_generation, evidence_snapshot_digest, source_id, "
        "invoice_number, vendor, authorized_amount, authorized_currency, "
        "payment_idempotency_key, review_id "
        "FROM final_decisions WHERE case_id = ?",
        (case_id,),
    ).fetchone()
    if decision_row is None or int(decision_row["decision_generation"]) != generation:
        raise _AuthorizationSnapshotError("final decision is missing or stale")
    if decision_row["evidence_snapshot_digest"] != evidence.digest:
        raise _AuthorizationSnapshotError("final decision snapshot digest does not match evidence")
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
    expected_review_id = review.review_id if review is not None else None
    exact_authorization_columns = (
        decision_row["source_id"] == invoice.source.source_id
        and decision_row["invoice_number"] == invoice.invoice_number.normalized_value
        and decision_row["vendor"] == invoice.vendor.normalized_value
        and decision_row["authorized_amount"]
        == (str(invoice.declared_total) if invoice.declared_total is not None else None)
        and decision_row["authorized_currency"] == invoice.currency.normalized_value
        and decision_row["payment_idempotency_key"] == payment_idempotency_key(invoice)
        and decision_row["review_id"] == expected_review_id
    )
    if not exact_authorization_columns:
        raise _AuthorizationSnapshotError(
            "final decision relational authorization fields do not match evidence"
        )
    return _AuthorizationSnapshot(invoice, decision, risk, review, evidence.digest)


def _validate_paid_ledger_source(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    settings: Settings,
) -> None:
    """Reject a PAID idempotency row whose source authorization has drifted."""

    case_id = str(row["case_id"])
    generation = int(row["decision_generation"])
    try:
        snapshot = _load_authorization_snapshot(connection, case_id, generation, settings)
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
        snapshot.evidence_snapshot_digest == row["evidence_snapshot_digest"]
        and isinstance(row["evidence_snapshot_digest"], str)
        and payment_idempotency_key(invoice) == str(row["idempotency_key"])
        and invoice.vendor.normalized_value == str(row["vendor"])
        and invoice.currency.normalized_value == str(row["currency"])
        and invoice.declared_total is not None
        and invoice.declared_total == Decimal(str(row["amount"]))
        and row["source_id"] == invoice.source.source_id
        and row["invoice_number"] == invoice.invoice_number.normalized_value
        and row["review_id"]
        == (snapshot.review.review_id if snapshot.review is not None else None)
        and str(row["status"]) in {PaymentStatus.PAID, PaymentStatus.FAILED}
        and (
            (str(row["status"]) == PaymentStatus.PAID and row["error"] is None)
            or (str(row["status"]) == PaymentStatus.FAILED and row["error"] is not None)
        )
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

    return payment_identity_key(
        invoice.vendor.normalized_value,
        invoice.invoice_number.normalized_value,
    )


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
    snapshot_settings = store._snapshot_settings()

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
            lease = (
                parse_canonical_utc(case_row["lease_expires_at"]) if case_row is not None else None
            )
            current_claim = (
                case_row is not None
                and str(case_row["execution_token"]) == claim.token
                and int(case_row["execution_generation"]) == claim.generation
                and str(case_row["execution_state"]) == "RUNNING"
                and lease is not None
                and lease > authorization_time
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
                snapshot = _load_authorization_snapshot(
                    connection,
                    case_id,
                    claim.generation,
                    snapshot_settings,
                )
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
            for existing_case_row in connection.execute(
                "SELECT * FROM payments WHERE case_id = ?",
                (case_id,),
            ).fetchall():
                _validate_paid_ledger_source(
                    connection,
                    existing_case_row,
                    snapshot_settings,
                )
            prior = connection.execute(
                "SELECT * FROM payments WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if prior is not None:
                if PaymentStatus(str(prior["status"])) is PaymentStatus.PAID:
                    _validate_paid_ledger_source(connection, prior, snapshot_settings)
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
                "created_at, decision_generation, evidence_snapshot_digest, source_id, "
                "invoice_number, review_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    snapshot.evidence_snapshot_digest,
                    persisted_invoice.source.source_id,
                    persisted_invoice.invoice_number.normalized_value,
                    snapshot.review.review_id if snapshot.review is not None else None,
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
