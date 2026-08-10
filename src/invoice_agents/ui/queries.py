"""Read-only console queries; every mutation stays in the existing services.

These queries surface stored values verbatim (``json_extract`` over persisted
payloads); nothing here recomputes, softens, or summarizes a status.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from invoice_agents.db.core import connect_database
from invoice_agents.models import InventoryRow
from invoice_agents.observability.audit import (
    safe_provider_request_id,
    sanitize_stored_event_payload,
    sanitize_text,
)

SOURCE_FORMATS = ("txt", "json", "csv", "xml", "pdf")

_LATEST_EXTRACTION_FIELD = (
    "(SELECT json_extract(e.payload_json, '{json_path}') FROM extractions e "
    "WHERE e.case_id = c.case_id ORDER BY e.version DESC LIMIT 1)"
)


class CaseListRow(BaseModel):
    """One dashboard row; every value is read from storage as persisted."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    invoice_number: str | None
    vendor: str | None
    source_format: str | None
    declared_total: str | None
    currency: str | None
    status: str
    stop_reason: str | None
    decision: str | None
    payment_status: str | None
    started_at: datetime
    finished_at: datetime | None


class CaseHeaderRow(BaseModel):
    """The ``cases`` table row for one case, as stored."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    source_id: str | None
    invoice_number: str | None
    vendor: str | None
    revision: str | None
    status: str
    stop_reason: str | None
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None
    has_result: bool


class EventRow(BaseModel):
    """One persisted audit event; ``seq`` is the SQLite rowid used as tail cursor."""

    model_config = ConfigDict(frozen=True)

    seq: int
    event_id: str
    event_type: str
    agent_name: str | None
    tool_call_id: str | None
    db_evidence_id: str | None
    review_id: str | None
    payment_id: str | None
    provider_request_id: str | None
    payload_json: str
    created_at: str


class PriorCaseRow(BaseModel):
    """A prior case of the same invoice number, for SUPERSEDE_REVISION selection."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    status: str
    revision: str | None
    declared_total: str | None
    currency: str | None
    payment_status: str | None
    started_at: datetime


def _parse_dt(value: Any) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value else None


def list_cases(
    workflow_db: Path,
    *,
    status: str | None = None,
    decision: str | None = None,
    source_format: str | None = None,
    search: str | None = None,
) -> list[CaseListRow]:
    """List cases newest first with optional stored-value filters."""

    declared_total = _LATEST_EXTRACTION_FIELD.format(json_path="$.declared_total")
    currency = _LATEST_EXTRACTION_FIELD.format(json_path="$.currency.normalized_value")
    sql = (
        "SELECT c.case_id, c.invoice_number, c.vendor, c.status, c.stop_reason, "
        "c.started_at, c.finished_at, s.source_format, "
        "json_extract(c.result_json, '$.final_decision.decision') AS decision, "
        "json_extract(c.result_json, '$.payment.status') AS payment_status, "
        f"{declared_total} AS declared_total, {currency} AS currency "
        "FROM cases c LEFT JOIN source_artifacts s ON s.source_id = c.source_id"
    )
    clauses: list[str] = []
    params: list[str] = []
    if status:
        clauses.append("c.status = ?")
        params.append(status)
    if decision:
        clauses.append("json_extract(c.result_json, '$.final_decision.decision') = ?")
        params.append(decision)
    if source_format:
        clauses.append("s.source_format = ?")
        params.append(source_format)
    if search:
        needle = f"%{search.casefold()}%"
        clauses.append(
            "(LOWER(IFNULL(c.invoice_number,'')) LIKE ? OR LOWER(IFNULL(c.vendor,'')) LIKE ?)"
        )
        params.extend([needle, needle])
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY c.started_at DESC, c.case_id"
    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [
        CaseListRow(
            case_id=str(row["case_id"]),
            invoice_number=row["invoice_number"],
            vendor=row["vendor"],
            source_format=row["source_format"],
            declared_total=None if row["declared_total"] is None else str(row["declared_total"]),
            currency=row["currency"],
            status=str(row["status"]),
            stop_reason=row["stop_reason"],
            decision=row["decision"],
            payment_status=row["payment_status"],
            started_at=datetime.fromisoformat(str(row["started_at"])),
            finished_at=_parse_dt(row["finished_at"]),
        )
        for row in rows
    ]


def status_counts(workflow_db: Path) -> dict[str, int]:
    """Stored case counts by status."""

    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS n FROM cases GROUP BY status"
        ).fetchall()
    return {str(row["status"]): int(row["n"]) for row in rows}


def pending_review_count(workflow_db: Path) -> int:
    with connect_database(workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS n FROM review_requests WHERE status = 'PENDING'"
        ).fetchone()
    return int(row["n"])


def case_header(workflow_db: Path, case_id: str) -> CaseHeaderRow | None:
    """The stored case row, or None when the case does not exist."""

    with connect_database(workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT case_id, source_id, invoice_number, vendor, revision, status, stop_reason, "
            "started_at, updated_at, finished_at, result_json IS NOT NULL AS has_result "
            "FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    if row is None:
        return None
    return CaseHeaderRow(
        case_id=str(row["case_id"]),
        source_id=row["source_id"],
        invoice_number=row["invoice_number"],
        vendor=row["vendor"],
        revision=row["revision"],
        status=str(row["status"]),
        stop_reason=row["stop_reason"],
        started_at=datetime.fromisoformat(str(row["started_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        finished_at=_parse_dt(row["finished_at"]),
        has_result=bool(row["has_result"]),
    )


def events_after(workflow_db: Path, case_id: str, after_seq: int = 0) -> list[EventRow]:
    """Tail persisted events for a case strictly after the given rowid cursor."""

    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT rowid AS seq, event_id, event_type, agent_name, tool_call_id, "
            "db_evidence_id, review_id, payment_id, provider_request_id, payload_json, "
            "created_at FROM events WHERE case_id = ? AND rowid > ? ORDER BY rowid",
            (case_id, after_seq),
        ).fetchall()
    return [
        EventRow(
            seq=int(row["seq"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            agent_name=(
                sanitize_text(str(row["agent_name"])) if row["agent_name"] is not None else None
            ),
            tool_call_id=(
                sanitize_text(str(row["tool_call_id"])) if row["tool_call_id"] is not None else None
            ),
            db_evidence_id=row["db_evidence_id"],
            review_id=row["review_id"],
            payment_id=row["payment_id"],
            provider_request_id=safe_provider_request_id(row["provider_request_id"]),
            payload_json=sanitize_stored_event_payload(
                str(row["event_type"]), str(row["payload_json"])
            ),
            created_at=str(row["created_at"]),
        )
        for row in rows
    ]


def prior_cases_for_invoice(
    workflow_db: Path, invoice_number: str | None, exclude_case_id: str
) -> list[PriorCaseRow]:
    """Prior cases of the same stored invoice number, with amount and paid state."""

    if not invoice_number:
        return []
    declared_total = _LATEST_EXTRACTION_FIELD.format(json_path="$.declared_total")
    currency = _LATEST_EXTRACTION_FIELD.format(json_path="$.currency.normalized_value")
    with connect_database(workflow_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT c.case_id, c.status, c.revision, c.started_at, "
            f"{declared_total} AS declared_total, {currency} AS currency, "
            "(SELECT p.status FROM payments p WHERE p.case_id = c.case_id "
            " ORDER BY p.created_at DESC LIMIT 1) AS payment_status "
            "FROM cases c WHERE c.invoice_number = ? AND c.case_id <> ? "
            "ORDER BY c.started_at",
            (invoice_number, exclude_case_id),
        ).fetchall()
    return [
        PriorCaseRow(
            case_id=str(row["case_id"]),
            status=str(row["status"]),
            revision=row["revision"],
            declared_total=None if row["declared_total"] is None else str(row["declared_total"]),
            currency=row["currency"],
            payment_status=row["payment_status"],
            started_at=datetime.fromisoformat(str(row["started_at"])),
        )
        for row in rows
    ]


def payment_case(workflow_db: Path, payment_id: str) -> str | None:
    """The case a stored payment belongs to, for duplicate_of links."""

    with connect_database(workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT case_id FROM payments WHERE payment_id = ?", (payment_id,)
        ).fetchone()
    return str(row["case_id"]) if row else None


def list_inventory(inventory_db: Path) -> list[InventoryRow]:
    """All authoritative inventory rows for the mapping SKU dropdown."""

    with connect_database(inventory_db, read_only=True) as connection:
        rows = connection.execute(
            "SELECT sku, item_name, available_stock FROM inventory ORDER BY sku"
        ).fetchall()
    return [
        InventoryRow(
            sku=str(row["sku"]),
            item_name=str(row["item_name"]),
            available_stock=int(row["available_stock"]),
        )
        for row in rows
    ]
