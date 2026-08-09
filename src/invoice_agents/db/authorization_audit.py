"""Authoritative read-only audit of every workflow authorization record."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pydantic import ValidationError

from invoice_agents.agents.decision_rules import validate_final_decision
from invoice_agents.config import Settings
from invoice_agents.db.store import (
    ReviewAuthorization,
    _reconcile_review_authorization,
    load_authoritative_review_authorization,
    load_authorization_evidence_snapshot,
    parse_canonical_utc,
    validate_review_authorization_snapshot,
    validated_evidence_facts,
)
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.evidence_snapshot import (
    EvidenceSnapshot,
    EvidenceSnapshotError,
    validate_final_decision_snapshot,
)
from invoice_agents.models import FinalDecision, HumanDecision, PersistedPaymentRow
from invoice_agents.payment.identity import payment_identity_key

AUDIT_COUNT_KEYS = (
    "invalid_review_count",
    "invalid_human_decision_count",
    "invalid_snapshot_count",
    "invalid_final_decision_count",
    "invalid_payment_count",
    "invalid_quarantine_count",
)

_AUDIT_ERRORS = (
    EvidenceSnapshotError,
    InvoiceAgentsError,
    KeyError,
    TypeError,
    ValueError,
    ValidationError,
)


@dataclass(frozen=True, slots=True)
class _AuditedAnchor:
    snapshot: EvidenceSnapshot
    review_authorization: ReviewAuthorization | None


def _lower_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_positive_generation(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("authorization generation is not a strict positive integer")
    return value


def _audit_review_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    settings: Settings,
    inventory_schema: str,
) -> ReviewAuthorization:
    if type(row["sequence"]) is not int or int(row["sequence"]) < 1:
        raise ValueError("review sequence is invalid")
    if parse_canonical_utc(row["created_at"]) is None or (
        row["resolved_at"] is not None and parse_canonical_utc(row["resolved_at"]) is None
    ):
        raise ValueError("review timestamps are invalid")
    generation = _strict_positive_generation(row["execution_generation"])
    if not _lower_hex_digest(row["evidence_snapshot_digest"]):
        raise ValueError("review snapshot digest is invalid")
    authorization = _reconcile_review_authorization(
        connection,
        row,
        str(row["case_id"]),
    )
    if authorization.execution_generation != generation:
        raise ValueError("review generation changed during reconciliation")
    validate_review_authorization_snapshot(
        connection,
        authorization,
        settings,
        inventory_connection=connection,
        inventory_schema=inventory_schema,
    )
    return authorization


def _audit_human_row(connection: sqlite3.Connection, row: sqlite3.Row) -> HumanDecision:
    if not isinstance(row["decision_id"], str) or not row["decision_id"]:
        raise ValueError("human decision id is invalid")
    human = HumanDecision.model_validate_json(row["payload_json"], strict=True)
    if parse_canonical_utc(row["decided_at"]) is None:
        raise ValueError("human decision timestamp is invalid")
    review_rows = connection.execute(
        "SELECT payload_json FROM review_requests WHERE review_id = ?",
        (row["review_id"],),
    ).fetchall()
    sibling_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM human_decisions WHERE review_id = ?",
            (row["review_id"],),
        ).fetchone()[0]
    )
    if len(review_rows) != 1 or sibling_count != 1:
        raise ValueError("human decision does not have one exact review relationship")
    from invoice_agents.models import ReviewRequest

    review = ReviewRequest.model_validate_json(review_rows[0]["payload_json"], strict=True)
    exact = (
        human.review_id == row["review_id"]
        and human.reviewer == row["reviewer"]
        and human.decision == row["decision"]
        and human.reason == row["reason"]
        and human.decided_at.isoformat() == row["decided_at"]
        and review.status == "RESOLVED"
        and review.human_decision == human
    )
    if not exact:
        raise ValueError("human decision relational and embedded values do not match")
    return human


def _audit_anchor_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    settings: Settings,
    inventory_schema: str,
) -> _AuditedAnchor:
    case_id = str(row["case_id"])
    generation = _strict_positive_generation(row["execution_generation"])
    if not _lower_hex_digest(row["evidence_snapshot_digest"]):
        raise ValueError("validated snapshot digest is invalid")
    if type(row["policy_review_required"]) is not int or row["policy_review_required"] not in (
        0,
        1,
    ):
        raise ValueError("validated policy flag is invalid")
    if type(row["unresolved_blocker_count"]) is not int or row["unresolved_blocker_count"] < 0:
        raise ValueError("validated blocker count is invalid")
    if parse_canonical_utc(row["validated_at"]) is None:
        raise ValueError("validated snapshot timestamp is invalid")
    review_authorization = load_authoritative_review_authorization(
        connection,
        case_id,
        generation,
    )
    snapshot = load_authorization_evidence_snapshot(
        connection,
        case_id,
        generation,
        settings,
        review_authorization,
        inventory_connection=connection,
        inventory_schema=inventory_schema,
    )
    facts = validated_evidence_facts(snapshot, review_authorization)
    if (
        row["evidence_snapshot_digest"] != snapshot.digest
        or row["policy_review_required"] != facts.policy_review_required
        or row["unresolved_blocker_count"] != facts.unresolved_blocker_count
        or row["critique_disposition"] != facts.critique_disposition
        or row["review_id"] != facts.review_id
        or row["review_snapshot_digest"] != facts.review_snapshot_digest
    ):
        raise ValueError("validated snapshot columns do not match authoritative evidence")
    return _AuditedAnchor(snapshot=snapshot, review_authorization=review_authorization)


def _anchor_for_case(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
    settings: Settings,
    inventory_schema: str,
) -> _AuditedAnchor:
    rows = connection.execute(
        "SELECT case_id, execution_generation, evidence_snapshot_digest, "
        "policy_review_required, unresolved_blocker_count, critique_disposition, review_id, "
        "review_snapshot_digest, validated_at FROM validated_evidence_snapshots "
        "WHERE case_id = ? AND execution_generation = ?",
        (case_id, generation),
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("final/payment does not have one exact validated snapshot")
    return _audit_anchor_row(connection, rows[0], settings, inventory_schema)


def _audit_final_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    settings: Settings,
    inventory_schema: str,
) -> FinalDecision:
    if not isinstance(row["decision_id"], str) or not row["decision_id"]:
        raise ValueError("final decision id is invalid")
    case_id = str(row["case_id"])
    generation = _strict_positive_generation(row["decision_generation"])
    if parse_canonical_utc(row["created_at"]) is None:
        raise ValueError("final decision timestamp is invalid")
    if not _lower_hex_digest(row["evidence_snapshot_digest"]):
        raise ValueError("final decision digest is invalid")
    decision = FinalDecision.model_validate_json(row["payload_json"], strict=True)
    audited = _anchor_for_case(
        connection,
        case_id,
        generation,
        settings,
        inventory_schema,
    )
    snapshot = audited.snapshot
    review = (
        audited.review_authorization.review if audited.review_authorization is not None else None
    )
    validate_final_decision(
        decision.decision,
        decision.payment_eligible,
        snapshot.risk,
        snapshot.critique,
        review,
        case_id=case_id,
    )
    validate_final_decision_snapshot(decision, snapshot, review)
    invoice = snapshot.invoice
    exact = (
        row["evidence_snapshot_digest"] == snapshot.digest
        and row["source_id"] == invoice.source.source_id
        and row["invoice_number"] == invoice.invoice_number.normalized_value
        and row["vendor"] == invoice.vendor.normalized_value
        and row["authorized_amount"]
        == (str(invoice.declared_total) if invoice.declared_total is not None else None)
        and row["authorized_currency"] == invoice.currency.normalized_value
        and row["payment_idempotency_key"]
        == payment_identity_key(
            invoice.vendor.normalized_value,
            invoice.invoice_number.normalized_value,
        )
        and row["review_id"] == (review.review_id if review is not None else None)
    )
    if not exact:
        raise ValueError("final decision does not match its authoritative anchor")
    return decision


def _audit_payment_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    settings: Settings,
    inventory_schema: str,
) -> PersistedPaymentRow:
    payment = PersistedPaymentRow.model_validate(dict(row), strict=True)
    final_rows = connection.execute(
        "SELECT decision_id, case_id, payload_json, created_at, decision_generation, "
        "evidence_snapshot_digest, source_id, invoice_number, vendor, authorized_amount, "
        "authorized_currency, payment_idempotency_key, review_id FROM final_decisions "
        "WHERE case_id = ?",
        (payment.case_id,),
    ).fetchall()
    if len(final_rows) != 1:
        raise ValueError("payment does not have one exact final decision")
    final = _audit_final_row(connection, final_rows[0], settings, inventory_schema)
    audited = _anchor_for_case(
        connection,
        payment.case_id,
        payment.decision_generation,
        settings,
        inventory_schema,
    )
    invoice = audited.snapshot.invoice
    review = (
        audited.review_authorization.review if audited.review_authorization is not None else None
    )
    exact = (
        final.decision.value == "APPROVE"
        and final.payment_eligible
        and payment.evidence_snapshot_digest == audited.snapshot.digest
        and payment.source_id == invoice.source.source_id
        and payment.invoice_number == invoice.invoice_number.normalized_value
        and payment.vendor == invoice.vendor.normalized_value
        and payment.amount == str(invoice.declared_total)
        and payment.currency == invoice.currency.normalized_value
        and payment.idempotency_key
        == payment_identity_key(
            invoice.vendor.normalized_value,
            invoice.invoice_number.normalized_value,
        )
        and payment.review_id == (review.review_id if review is not None else None)
    )
    if not exact:
        raise ValueError("payment does not match its authoritative final decision")
    return payment


def audit_workflow_authorization(
    connection: sqlite3.Connection,
    settings: Settings,
    *,
    inventory_schema: str,
) -> dict[str, int]:
    """Enumerate and revalidate every active authorization row without mutation."""

    counts = {key: 0 for key in AUDIT_COUNT_KEYS}
    review_rows = connection.execute(
        "SELECT review_id, case_id, sequence, status, payload_json, created_at, resolved_at, "
        "execution_generation, evidence_snapshot_digest FROM review_requests "
        "ORDER BY review_id"
    ).fetchall()
    for row in review_rows:
        try:
            _audit_review_row(connection, row, settings, inventory_schema)
        except _AUDIT_ERRORS:
            counts["invalid_review_count"] += 1

    human_rows = connection.execute(
        "SELECT decision_id, review_id, reviewer, decision, reason, payload_json, decided_at "
        "FROM human_decisions ORDER BY decision_id"
    ).fetchall()
    for row in human_rows:
        try:
            _audit_human_row(connection, row)
        except _AUDIT_ERRORS:
            counts["invalid_human_decision_count"] += 1

    anchor_rows = connection.execute(
        "SELECT case_id, execution_generation, evidence_snapshot_digest, "
        "policy_review_required, unresolved_blocker_count, critique_disposition, review_id, "
        "review_snapshot_digest, validated_at FROM validated_evidence_snapshots "
        "ORDER BY case_id, execution_generation"
    ).fetchall()
    for row in anchor_rows:
        try:
            _audit_anchor_row(connection, row, settings, inventory_schema)
        except _AUDIT_ERRORS:
            counts["invalid_snapshot_count"] += 1

    final_rows = connection.execute(
        "SELECT decision_id, case_id, payload_json, created_at, decision_generation, "
        "evidence_snapshot_digest, source_id, invoice_number, vendor, authorized_amount, "
        "authorized_currency, payment_idempotency_key, review_id FROM final_decisions "
        "ORDER BY decision_id"
    ).fetchall()
    for row in final_rows:
        try:
            _audit_final_row(connection, row, settings, inventory_schema)
        except _AUDIT_ERRORS:
            counts["invalid_final_decision_count"] += 1

    payment_rows = connection.execute(
        "SELECT payment_id, case_id, idempotency_key, vendor, amount, currency, status, error, "
        "created_at, decision_generation, evidence_snapshot_digest, source_id, invoice_number, "
        "review_id FROM payments ORDER BY payment_id"
    ).fetchall()
    for row in payment_rows:
        try:
            _audit_payment_row(connection, row, settings, inventory_schema)
        except _AUDIT_ERRORS:
            counts["invalid_payment_count"] += 1

    from invoice_agents.db.legacy_archive import audit_legacy_authorization_archives

    counts["invalid_quarantine_count"] = audit_legacy_authorization_archives(connection)

    return counts
