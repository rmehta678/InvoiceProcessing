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
    execution_claim_expiry_iso,
    load_authoritative_review_authorization,
    load_authorization_evidence_snapshot,
    parse_canonical_utc,
    validate_execution_claim,
    validated_evidence_facts,
)
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.evidence_snapshot import (
    EvidenceSnapshotError,
    validate_final_decision_snapshot,
)
from invoice_agents.models import (
    DecisionKind,
    ExtractedInvoice,
    FinalDecision,
    Money,
    PaymentResult,
    PaymentStatus,
    PersistedPaymentRow,
    ReviewRequest,
    RiskAssessment,
)
from invoice_agents.observability.audit import sanitize_text
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


def _strict_payment_row(row: sqlite3.Row) -> PersistedPaymentRow:
    try:
        return PersistedPaymentRow.model_validate(dict(row), strict=True)
    except ValueError as exc:
        raise InvoiceAgentsError(
            ErrorCategory.PAYMENT,
            "payment ledger row has an invalid storage shape",
            stop_reason="PAYMENT_LEDGER_INCONSISTENT",
        ) from exc


def _load_authorization_snapshot(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
    settings: Settings,
) -> _AuthorizationSnapshot:
    """Load and validate one complete generation-bound approval snapshot."""

    from invoice_agents.agents.decision_rules import validate_final_decision

    try:
        review_authorization = load_authoritative_review_authorization(
            connection,
            case_id,
            generation,
        )
    except EvidenceSnapshotError as exc:
        raise _AuthorizationSnapshotError(str(exc)) from exc
    try:
        evidence = load_authorization_evidence_snapshot(
            connection,
            case_id,
            generation,
            settings,
            review_authorization,
        )
    except EvidenceSnapshotError as exc:
        raise _AuthorizationSnapshotError(f"evidence snapshot is invalid: {exc}") from exc
    invoice = evidence.invoice
    risk = evidence.risk
    critique = evidence.critique

    review = review_authorization.review if review_authorization is not None else None
    decision_row = connection.execute(
        "SELECT payload_json, decision_generation, evidence_snapshot_digest, source_id, "
        "invoice_number, vendor, authorized_amount, authorized_currency, "
        "payment_idempotency_key, review_id "
        "FROM final_decisions WHERE case_id = ?",
        (case_id,),
    ).fetchone()
    if decision_row is None or int(decision_row["decision_generation"]) != generation:
        raise _AuthorizationSnapshotError("final decision is missing or stale")
    facts = validated_evidence_facts(evidence, review_authorization)
    anchor = connection.execute(
        "SELECT evidence_snapshot_digest, policy_review_required, unresolved_blocker_count, "
        "critique_disposition, review_id, review_snapshot_digest, validated_at "
        "FROM validated_evidence_snapshots WHERE case_id = ? AND execution_generation = ?",
        (case_id, generation),
    ).fetchone()
    anchor_is_exact = (
        anchor is not None
        and anchor["evidence_snapshot_digest"] == evidence.digest
        and int(anchor["policy_review_required"]) == facts.policy_review_required
        and int(anchor["unresolved_blocker_count"]) == facts.unresolved_blocker_count
        and anchor["critique_disposition"] == facts.critique_disposition
        and anchor["review_id"] == facts.review_id
        and anchor["review_snapshot_digest"] == facts.review_snapshot_digest
        and parse_canonical_utc(anchor["validated_at"]) is not None
    )
    if not anchor_is_exact:
        raise _AuthorizationSnapshotError(
            "validated evidence snapshot anchor does not match current evidence"
        )

    if decision_row["evidence_snapshot_digest"] != evidence.digest:
        raise _AuthorizationSnapshotError("final decision snapshot digest does not match evidence")
    try:
        decision = FinalDecision.model_validate_json(decision_row["payload_json"], strict=True)
    except ValueError as exc:
        raise _AuthorizationSnapshotError("final decision payload is invalid") from exc
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
    try:
        validate_final_decision_snapshot(decision, evidence, review)
    except EvidenceSnapshotError as exc:
        raise _AuthorizationSnapshotError(str(exc)) from exc
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

    payment = _strict_payment_row(row)
    case_id = payment.case_id
    generation = payment.decision_generation
    try:
        snapshot = _load_authorization_snapshot(connection, case_id, generation, settings)
    except (InvoiceAgentsError, _AuthorizationSnapshotError, ValueError) as exc:
        raise InvoiceAgentsError(
            ErrorCategory.PAYMENT,
            f"paid ledger source snapshot is inconsistent for case {case_id}",
            case_id=case_id,
            stop_reason="PAYMENT_LEDGER_INCONSISTENT",
        ) from exc
    invoice = snapshot.invoice
    valid = (
        snapshot.evidence_snapshot_digest == payment.evidence_snapshot_digest
        and payment_idempotency_key(invoice) == payment.idempotency_key
        and invoice.vendor.normalized_value == payment.vendor
        and invoice.currency.normalized_value == payment.currency
        and invoice.declared_total is not None
        and invoice.declared_total == Decimal(payment.amount)
        and payment.source_id == invoice.source.source_id
        and payment.invoice_number == invoice.invoice_number.normalized_value
        and payment.review_id
        == (snapshot.review.review_id if snapshot.review is not None else None)
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
    payment = _strict_payment_row(row)
    stored_status = PaymentStatus(payment.status)
    status = (
        PaymentStatus.DUPLICATE
        if duplicate and stored_status is PaymentStatus.PAID
        else stored_status
    )
    return PaymentResult(
        payment_id=payment.payment_id,
        case_id=attempted_case_id or payment.case_id,
        idempotency_key=payment.idempotency_key,
        status=status,
        vendor=payment.vendor,
        amount=Money(amount=Decimal(payment.amount), currency=payment.currency),
        processed_at=datetime.fromisoformat(payment.created_at),
        duplicate_of=payment.payment_id if duplicate else None,
        error=sanitize_text(payment.error) if payment.error is not None else None,
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

    claim = validate_execution_claim(claim, expected_case_id=case_id)
    if store.path != workflow_db.resolve():
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "payment store and workflow database paths do not match",
            case_id=case_id,
            stop_reason="PAYMENT_DATABASE_MISMATCH",
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
            error=sanitize_text(error),
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
            claim_lease = execution_claim_expiry_iso(claim)
            current_claim = (
                case_row is not None
                and claim_lease is not None
                and str(case_row["execution_token"]) == claim.token
                and int(case_row["execution_generation"]) == claim.generation
                and str(case_row["execution_state"]) == "RUNNING"
                and lease is not None
                and case_row["lease_expires_at"] == claim_lease
                and claim.expires_at > authorization_time
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
                prior_payment = _strict_payment_row(prior)
                if prior_payment.status == "PAID":
                    _validate_paid_ledger_source(connection, prior, snapshot_settings)
                connection.rollback()
                return _from_row(
                    prior,
                    duplicate=prior_payment.status == "PAID",
                    attempted_case_id=case_id,
                )
            payment_id = f"pay_{uuid4().hex}"
            created_at = datetime.now(UTC)
            status = PaymentStatus.FAILED if inject_failure else PaymentStatus.PAID
            error = sanitize_text("injected mock-payment failure") if inject_failure else None
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
