"""Typed persistence for case state, evidence, review, decisions, and results."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Never, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel

from invoice_agents.db.core import connect_database
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.evidence_snapshot import (
    EvidenceSnapshot,
    EvidenceSnapshotError,
    build_evidence_snapshot,
    validate_review_snapshot,
)
from invoice_agents.models import (
    CaseResult,
    CaseStatus,
    Critique,
    ExtractedInvoice,
    FinalDecision,
    HumanDecision,
    HumanDecisionKind,
    ReviewRequest,
    SourceArtifact,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

ROLLBACK_JOURNAL_MODES = frozenset({"delete", "persist", "truncate"})


@dataclass(frozen=True, slots=True)
class ExecutionClaim:
    """Database-issued fencing authority for one case execution generation."""

    case_id: str
    token: str
    generation: int
    expires_at: datetime


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_canonical_utc(value: object) -> datetime | None:
    """Parse only Python's canonical, timezone-aware UTC ISO representation."""

    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed if parsed.isoformat() == value else None


def _authoritative_source_for_case(connection: sqlite3.Connection, case_id: str) -> SourceArtifact:
    row = connection.execute(
        "SELECT c.source_id AS case_source_id, s.source_id, s.canonical_path, "
        "s.source_hash, s.source_format, s.size_bytes, s.modified_at, s.metadata_json "
        "FROM cases c JOIN source_artifacts s ON s.source_id = c.source_id "
        "WHERE c.case_id = ?",
        (case_id,),
    ).fetchone()
    if row is None:
        raise EvidenceSnapshotError("case has no authoritative source_artifact")
    try:
        source = SourceArtifact.model_validate_json(row["metadata_json"])
    except ValueError as exc:
        raise EvidenceSnapshotError(f"source_artifacts metadata is invalid: {exc}") from exc
    relationally_coherent = (
        row["case_source_id"] == source.source_id
        and row["source_id"] == source.source_id
        and row["canonical_path"] == str(source.canonical_path)
        and row["source_hash"] == source.sha256
        and row["source_format"] == source.source_format
        and int(row["size_bytes"]) == source.size_bytes
        and row["modified_at"] == source.modified_at.isoformat()
    )
    if not relationally_coherent:
        raise EvidenceSnapshotError(
            "source_artifacts relational columns do not match their metadata_json"
        )
    return source


def load_generation_evidence_snapshot(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
    *,
    require_latest: bool = True,
) -> EvidenceSnapshot:
    """Load one generation's latest component rows and validate them as one snapshot."""

    specifications = {
        "extraction": (
            "SELECT payload_json FROM extractions WHERE case_id = ? "
            "AND execution_generation = ? ORDER BY version DESC LIMIT 1",
            "extractions",
            "execution_generation",
            "",
        ),
        "identity": (
            "SELECT payload_json FROM identity_results WHERE case_id = ? "
            "AND execution_generation = ? ORDER BY rowid DESC LIMIT 1",
            "identity_results",
            "execution_generation",
            "",
        ),
        "inventory": (
            "SELECT payload_json FROM comparison_results WHERE case_id = ? "
            "AND execution_generation = ? AND comparison_type = 'inventory' "
            "ORDER BY rowid DESC LIMIT 1",
            "comparison_results",
            "execution_generation",
            "AND comparison_type = 'inventory'",
        ),
        "risk": (
            "SELECT payload_json FROM comparison_results WHERE case_id = ? "
            "AND execution_generation = ? AND comparison_type = 'risk' "
            "ORDER BY rowid DESC LIMIT 1",
            "comparison_results",
            "execution_generation",
            "AND comparison_type = 'risk'",
        ),
        "critique": (
            "SELECT payload_json FROM critique_results WHERE case_id = ? "
            "AND execution_generation = ? ORDER BY rowid DESC LIMIT 1",
            "critique_results",
            "execution_generation",
            "",
        ),
    }
    payloads: dict[str, str] = {}
    missing: list[str] = []
    stale: list[str] = []
    for name, (sql, table, column, predicate) in specifications.items():
        row = connection.execute(sql, (case_id, generation)).fetchone()
        if row is None:
            missing.append(name)
            continue
        payloads[name] = str(row["payload_json"])
        if require_latest:
            latest = connection.execute(
                f"SELECT MAX({column}) AS generation FROM {table} WHERE case_id = ? {predicate}",
                (case_id,),
            ).fetchone()["generation"]
            if latest is None or int(latest) != generation:
                stale.append(f"{name}:{latest}")
    if missing or stale:
        raise EvidenceSnapshotError(
            f"generation {generation} evidence is missing={sorted(missing)} stale={sorted(stale)}"
        )
    return build_evidence_snapshot(
        case_id,
        _authoritative_source_for_case(connection, case_id),
        payloads["extraction"],
        payloads["identity"],
        payloads["inventory"],
        payloads["risk"],
        payloads["critique"],
    )


def encode(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


def _normalize_alias(value: str) -> str:
    # Local import avoids the comparison module's WorkflowStore import during startup.
    from invoice_agents.tools.comparison import normalize_alias

    return normalize_alias(value)


def _decision_semantics(decision: HumanDecision) -> tuple[object, ...]:
    """Stable equality for idempotent human-decision replay.

    ``decided_at`` is deliberately excluded because each service attempt creates it.
    Mapping and blocker presentation order is not decision evidence.
    """

    mappings = tuple(
        sorted(
            json.dumps(
                mapping.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for mapping in decision.mappings
        )
    )
    return (
        decision.reviewer.strip(),
        decision.decision,
        " ".join(decision.reason.split()),
        mappings,
        decision.superseded_case_id.strip() if decision.superseded_case_id else None,
        tuple(sorted(decision.addressed_blocker_ids)),
    )


def _review_from_row(row: sqlite3.Row | None, review_id: str) -> ReviewRequest:
    if row is None:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            f"review request does not exist: {review_id}",
            stop_reason="REVIEW_NOT_FOUND",
        )
    return ReviewRequest.model_validate_json(row["payload_json"])


def _resolved_human_decision_replay(
    review: ReviewRequest, decision: HumanDecision | None
) -> ReviewRequest:
    """Return an exact resolved replay or classify every difference consistently."""

    existing = review.human_decision
    if (
        decision is not None
        and existing is not None
        and _decision_semantics(existing) == _decision_semantics(decision)
    ):
        return review
    raise InvoiceAgentsError(
        ErrorCategory.DATABASE,
        f"review {review.review_id} is already resolved",
        case_id=review.case_id,
        stop_reason="REVIEW_ALREADY_RESOLVED",
    )


def _normalized_invoice_field(invoice: dict[str, Any], name: str) -> str | None:
    raw = invoice.get(name)
    if not isinstance(raw, dict):
        return None
    normalized = raw.get("normalized_value")
    return str(normalized) if normalized is not None else None


def _validate_supersession(
    connection: sqlite3.Connection, review: ReviewRequest, superseded_case_id: str
) -> None:
    """Require one distinct, earlier persisted POSSIBLE_REVISION candidate."""

    candidate: dict[str, Any] | None = None
    for raw_candidate in review.evidence_bundle.get("identity_candidates", []):
        if isinstance(raw_candidate, dict) and raw_candidate.get("case_id") == superseded_case_id:
            candidate = raw_candidate
            break
    raw_invoice = review.evidence_bundle.get("invoice")
    invoice = raw_invoice if isinstance(raw_invoice, dict) else {}
    invoice_number = _normalized_invoice_field(invoice, "invoice_number")
    vendor = _normalized_invoice_field(invoice, "vendor")
    rows = connection.execute(
        "SELECT case_id, invoice_number, vendor, started_at FROM cases WHERE case_id IN (?, ?)",
        (review.case_id, superseded_case_id),
    ).fetchall()
    cases = {str(row["case_id"]): row for row in rows}
    current = cases.get(review.case_id)
    prior = cases.get(superseded_case_id)
    valid = (
        superseded_case_id != review.case_id
        and candidate is not None
        and candidate.get("relationship") == "POSSIBLE_REVISION"
        and invoice_number is not None
        and vendor is not None
        and candidate.get("invoice_number") == invoice_number
        and candidate.get("vendor") == vendor
        and current is not None
        and prior is not None
        and current["invoice_number"] == invoice_number
        and current["vendor"] == vendor
        and prior["invoice_number"] == invoice_number
        and prior["vendor"] == vendor
        and datetime.fromisoformat(str(prior["started_at"]))
        < datetime.fromisoformat(str(current["started_at"]))
    )
    if not valid:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "superseded case must be a distinct, earlier POSSIBLE_REVISION candidate "
            "for this invoice and vendor",
            case_id=review.case_id,
            stop_reason="SUPERSEDED_CASE_INVALID",
        )


def _validate_human_decision(
    connection: sqlite3.Connection, review: ReviewRequest, decision: HumanDecision
) -> list[tuple[str, str]]:
    """Validate every authorizing input against the transaction-local review evidence."""

    from invoice_agents.agents.decision_rules import AUTHORIZING_HUMAN_DECISIONS

    mappings = decision.mappings
    if mappings and decision.decision is not HumanDecisionKind.ESTABLISH_MAPPING:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "mappings are permitted only for ESTABLISH_MAPPING",
            case_id=review.case_id,
            stop_reason="HUMAN_MAPPING_INVALID",
        )
    if decision.decision is HumanDecisionKind.ESTABLISH_MAPPING and not mappings:
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "ESTABLISH_MAPPING requires at least one explicit mapping",
            case_id=review.case_id,
            stop_reason="HUMAN_MAPPING_MISSING",
        )

    unresolved_aliases: set[str] = set()
    for entry in review.evidence_bundle.get("inventory", []):
        if not isinstance(entry, dict) or entry.get("sku"):
            continue
        raw_items = entry.get("raw_items")
        if isinstance(raw_items, list):
            unresolved_aliases.update(
                normalized
                for raw_item in raw_items
                if isinstance(raw_item, str) and (normalized := _normalize_alias(raw_item))
            )
    validated_mappings: list[tuple[str, str]] = []
    targets_by_alias: dict[str, str] = {}
    for mapping in mappings:
        normalized = _normalize_alias(mapping.raw_item)
        if not normalized:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                "mapping alias is empty after normalization",
                case_id=review.case_id,
                stop_reason="MAPPING_ALIAS_INVALID",
            )
        if normalized not in unresolved_aliases:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                f"mapping alias is not unresolved inventory evidence in this review: "
                f"{mapping.raw_item}",
                case_id=review.case_id,
                stop_reason="MAPPING_ALIAS_NOT_IN_REVIEW",
            )
        sku = mapping.sku.strip()
        existing_target = targets_by_alias.get(normalized)
        if existing_target is not None and existing_target != sku:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                f"mapping alias has conflicting target SKUs: {mapping.raw_item}",
                case_id=review.case_id,
                stop_reason="HUMAN_MAPPING_INVALID",
            )
        targets_by_alias[normalized] = sku
        row = connection.execute(
            "SELECT sku FROM inventory_db.inventory WHERE sku = ?", (sku,)
        ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"mapping target SKU does not exist: {sku}",
                case_id=review.case_id,
                stop_reason="MAPPING_SKU_NOT_FOUND",
            )
        validated_mappings.append((normalized, sku))

    if (
        decision.superseded_case_id
        and decision.decision is not HumanDecisionKind.SUPERSEDE_REVISION
    ):
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "superseded_case_id is permitted only for SUPERSEDE_REVISION",
            case_id=review.case_id,
            stop_reason="SUPERSEDED_CASE_INVALID",
        )
    if decision.decision is HumanDecisionKind.SUPERSEDE_REVISION:
        if not decision.superseded_case_id:
            raise InvoiceAgentsError(
                ErrorCategory.TOOL,
                "SUPERSEDE_REVISION requires a superseded case ID",
                case_id=review.case_id,
                stop_reason="SUPERSEDED_CASE_MISSING",
            )
        _validate_supersession(connection, review, decision.superseded_case_id)

    package_blocker_ids = {
        str(entry["blocker_id"])
        for entry in review.evidence_bundle.get("blocking_evidence", [])
        if isinstance(entry, dict) and isinstance(entry.get("blocker_id"), str)
    }
    unknown_blocker_ids = set(decision.addressed_blocker_ids) - package_blocker_ids
    if decision.addressed_blocker_ids and (
        decision.decision not in AUTHORIZING_HUMAN_DECISIONS or unknown_blocker_ids
    ):
        raise InvoiceAgentsError(
            ErrorCategory.TOOL,
            "blocker authorization is permitted only for authorizing decisions and IDs "
            f"in this review package; unknown IDs: {sorted(unknown_blocker_ids)}",
            case_id=review.case_id,
            stop_reason="BLOCKER_AUTHORIZATION_INVALID",
        )
    return validated_mappings


class WorkflowStore:
    """Own all mutation of the workflow database; inventory remains separate."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def register_source(self, source: SourceArtifact) -> None:
        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT canonical_path, source_hash, source_format, size_bytes, modified_at, "
                    "metadata_json FROM source_artifacts WHERE source_id = ?",
                    (source.source_id,),
                ).fetchone()
                expected = (
                    str(source.canonical_path),
                    source.sha256,
                    source.source_format,
                    source.size_bytes,
                    source.modified_at.isoformat(),
                    source.model_dump_json(),
                )
                if existing is not None:
                    actual = tuple(existing[index] for index in range(6))
                    if actual != expected:
                        raise InvoiceAgentsError(
                            ErrorCategory.DATABASE,
                            f"source artifact {source.source_id} is immutable and conflicts",
                            stop_reason="SOURCE_ARTIFACT_IMMUTABLE",
                        )
                    connection.commit()
                    return
                connection.execute(
                    "INSERT INTO source_artifacts("
                    "source_id, canonical_path, source_hash, source_format, size_bytes, "
                    "modified_at, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (source.source_id, *expected, now_iso()),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def create_case(self, case_id: str, source: SourceArtifact, started_at: datetime) -> None:
        with connect_database(self.path) as connection:
            connection.execute(
                "INSERT INTO cases(case_id, source_id, status, stop_reason, started_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    source.source_id,
                    CaseStatus.INCOMPLETE,
                    "CASE_CREATED",
                    started_at.isoformat(),
                    now_iso(),
                ),
            )
            connection.commit()

    def claim_case_execution(
        self,
        case_id: str,
        expected_statuses: frozenset[CaseStatus],
        lease_seconds: int,
    ) -> ExecutionClaim:
        """Atomically claim one fresh or resumable case and return its fencing token."""

        if not expected_statuses:
            raise ValueError("expected_statuses must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        token = f"exec_{uuid4().hex}"
        statuses = tuple(str(status) for status in sorted(expected_statuses))
        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                claimed_at = datetime.now(UTC)
                expires_at = claimed_at + timedelta(seconds=lease_seconds)
                row = connection.execute(
                    "SELECT status, execution_token, execution_generation, "
                    "execution_state, lease_expires_at FROM cases WHERE case_id = ?",
                    (case_id,),
                ).fetchone()
                if row is None:
                    raise InvoiceAgentsError(
                        ErrorCategory.DATABASE,
                        f"case does not exist: {case_id}",
                        case_id=case_id,
                        stop_reason="CASE_NOT_FOUND",
                    )
                if not self._authority_tuple_is_valid(row):
                    raise InvoiceAgentsError(
                        ErrorCategory.DATABASE,
                        f"case {case_id} has a contradictory execution authority tuple",
                        case_id=case_id,
                        stop_reason="EXECUTION_AUTHORITY_CORRUPT",
                    )
                if str(row["status"]) not in statuses:
                    raise InvoiceAgentsError(
                        ErrorCategory.ORCHESTRATION,
                        f"case {case_id} has status {row['status']} and is not claimable",
                        case_id=case_id,
                        stop_reason="CASE_STATUS_NOT_CLAIMABLE",
                    )
                previous_state = str(row["execution_state"])
                previous_token = row["execution_token"]
                previous_generation = int(row["execution_generation"])
                previous_lease = row["lease_expires_at"]
                parsed_lease = parse_canonical_utc(previous_lease)
                claimable = previous_state in {"IDLE", "FINISHED"} or (
                    previous_state == "RUNNING"
                    and parsed_lease is not None
                    and parsed_lease <= claimed_at
                )
                if not claimable:
                    raise InvoiceAgentsError(
                        ErrorCategory.ORCHESTRATION,
                        f"case {case_id} already has an active execution claim",
                        case_id=case_id,
                        stop_reason="CASE_ALREADY_CLAIMED",
                    )
                generation = previous_generation + 1
                updated = connection.execute(
                    "UPDATE cases SET execution_token = ?, "
                    "execution_generation = ?, "
                    "execution_state = 'RUNNING', lease_expires_at = ?, updated_at = ? "
                    "WHERE case_id = ? AND status = ? AND execution_state = ? "
                    "AND execution_generation = ? AND execution_token IS ? "
                    "AND lease_expires_at IS ?",
                    (
                        token,
                        generation,
                        expires_at.isoformat(),
                        claimed_at.isoformat(),
                        case_id,
                        row["status"],
                        previous_state,
                        previous_generation,
                        previous_token,
                        previous_lease,
                    ),
                )
                if updated.rowcount != 1:
                    raise InvoiceAgentsError(
                        ErrorCategory.ORCHESTRATION,
                        f"case {case_id} execution authority changed during claim",
                        case_id=case_id,
                        stop_reason="CASE_ALREADY_CLAIMED",
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return ExecutionClaim(case_id, token, generation, expires_at)

    def renew_case_execution(self, claim: ExecutionClaim, lease_seconds: int) -> ExecutionClaim:
        """Renew only the still-current, unexpired execution claim."""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            renewed_at = datetime.now(UTC)
            expires_at = renewed_at + timedelta(seconds=lease_seconds)
            self._require_current_claim(connection, claim, renewed_at)
            updated = connection.execute(
                "UPDATE cases SET lease_expires_at = ?, updated_at = ? "
                "WHERE case_id = ? AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?",
                (
                    expires_at.isoformat(),
                    renewed_at.isoformat(),
                    claim.case_id,
                    claim.token,
                    claim.generation,
                    renewed_at.isoformat(),
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                self._raise_stale_execution_claim(claim)
            connection.commit()
        return ExecutionClaim(claim.case_id, claim.token, claim.generation, expires_at)

    def release_case_execution(self, claim: ExecutionClaim) -> None:
        """Release a preparation-only claim without fabricating a terminal result."""

        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            released_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, released_at)
            updated = connection.execute(
                "UPDATE cases SET execution_token = NULL, execution_state = 'IDLE', "
                "lease_expires_at = NULL, updated_at = ? WHERE case_id = ? "
                "AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?",
                (
                    released_at.isoformat(),
                    claim.case_id,
                    claim.token,
                    claim.generation,
                    released_at.isoformat(),
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                self._raise_stale_execution_claim(claim)
            connection.commit()

    @staticmethod
    def _authority_tuple_is_valid(row: sqlite3.Row) -> bool:
        state = row["execution_state"]
        token = row["execution_token"]
        raw_generation = row["execution_generation"]
        lease = row["lease_expires_at"]
        if not isinstance(state, str) or not isinstance(raw_generation, int):
            return False
        generation = raw_generation
        valid_token = isinstance(token, str) and bool(token)
        valid_lease = parse_canonical_utc(lease) is not None
        return (
            (state == "IDLE" and token is None and lease is None and generation >= 0)
            or (state == "RUNNING" and valid_token and valid_lease and generation >= 1)
            or (state == "FINISHED" and valid_token and lease is None and generation >= 1)
        )

    def _require_current_claim(
        self,
        connection: sqlite3.Connection,
        claim: ExecutionClaim,
        checked_at: datetime,
    ) -> None:
        row = connection.execute(
            "SELECT execution_token, execution_generation, execution_state, "
            "lease_expires_at FROM cases WHERE case_id = ?",
            (claim.case_id,),
        ).fetchone()
        if row is not None and not self._authority_tuple_is_valid(row):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"case {claim.case_id} has a contradictory execution authority tuple",
                case_id=claim.case_id,
                stop_reason="EXECUTION_AUTHORITY_CORRUPT",
            )
        lease = parse_canonical_utc(row["lease_expires_at"]) if row is not None else None
        if (
            row is None
            or row["execution_token"] != claim.token
            or int(row["execution_generation"]) != claim.generation
            or row["execution_state"] != "RUNNING"
            or lease is None
            or lease <= checked_at
        ):
            self._raise_stale_execution_claim(claim)

    @staticmethod
    def _assert_evidence_generation_mutable(
        connection: sqlite3.Connection, case_id: str, generation: int
    ) -> None:
        locked = connection.execute(
            "SELECT 'final' AS reason FROM final_decisions WHERE case_id = ? "
            "AND decision_generation = ? UNION ALL "
            "SELECT 'paid' FROM payments WHERE case_id = ? AND status = 'PAID' LIMIT 1",
            (case_id, generation, case_id),
        ).fetchone()
        if locked is not None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"authorization evidence is immutable after {locked['reason']} for case {case_id}",
                case_id=case_id,
                stop_reason="AUTHORIZATION_EVIDENCE_IMMUTABLE",
                details={"execution_generation": generation},
            )

    @staticmethod
    def _raise_snapshot_invalid(case_id: str, generation: int, reason: str) -> Never:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            f"authorization evidence snapshot is invalid for case {case_id}: {reason}",
            case_id=case_id,
            stop_reason="EVIDENCE_SNAPSHOT_INVALID",
            details={"execution_generation": generation, "reason": reason},
        )

    @staticmethod
    def _raise_stale_execution_claim(claim: ExecutionClaim) -> None:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            f"execution claim is stale for case {claim.case_id}",
            case_id=claim.case_id,
            stop_reason="STALE_EXECUTION_CLAIM",
            details={"execution_generation": claim.generation},
        )

    def save_extraction(
        self, case_id: str, invoice: ExtractedInvoice, claim: ExecutionClaim
    ) -> str:
        if claim.case_id != case_id:
            self._raise_stale_execution_claim(claim)
        extraction_id = f"ext_{uuid4().hex}"
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, written_at)
            self._assert_evidence_generation_mutable(connection, case_id, claim.generation)
            version_row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM extractions WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            version = int(version_row["version"]) + 1
            connection.execute(
                "INSERT INTO extractions(extraction_id, case_id, version, payload_json, "
                "created_at, execution_generation) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    extraction_id,
                    case_id,
                    version,
                    invoice.model_dump_json(),
                    written_at.isoformat(),
                    claim.generation,
                ),
            )
            updated = connection.execute(
                "UPDATE cases SET invoice_number = ?, vendor = ?, revision = ?, updated_at = ? "
                "WHERE case_id = ? AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?",
                (
                    invoice.invoice_number.normalized_value,
                    invoice.vendor.normalized_value,
                    invoice.revision.normalized_value if invoice.revision else None,
                    written_at.isoformat(),
                    case_id,
                    claim.token,
                    claim.generation,
                    written_at.isoformat(),
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                self._raise_stale_execution_claim(claim)
            connection.commit()
        return extraction_id

    def load_extraction(self, case_id: str) -> ExtractedInvoice:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM extractions WHERE case_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"no extraction exists for case {case_id}",
                case_id=case_id,
                stop_reason="EXTRACTION_NOT_FOUND",
            )
        return ExtractedInvoice.model_validate_json(row["payload_json"])

    def load_current_extraction(self, claim: ExecutionClaim) -> ExtractedInvoice:
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            row = connection.execute(
                "SELECT payload_json FROM extractions WHERE case_id = ? "
                "AND execution_generation = ? ORDER BY version DESC LIMIT 1",
                (claim.case_id, claim.generation),
            ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"current execution has no extraction for case {claim.case_id}",
                case_id=claim.case_id,
                stop_reason="EXTRACTION_GENERATION_MISMATCH",
            )
        return ExtractedInvoice.model_validate_json(row["payload_json"])

    def save_identity(
        self, case_id: str, payload: list[dict[str, Any]], claim: ExecutionClaim
    ) -> str:
        identity_id = f"ident_{uuid4().hex}"
        self._insert_payload(
            "identity_results", "identity_id", identity_id, case_id, payload, claim
        )
        return identity_id

    def load_identity(self, case_id: str) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._load_latest_payload("identity_results", case_id))

    def load_current_identity(self, claim: ExecutionClaim) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._load_current_payload("identity_results", claim),
        )

    def save_comparison(self, case_id: str, kind: str, payload: Any, claim: ExecutionClaim) -> str:
        if claim.case_id != case_id:
            self._raise_stale_execution_claim(claim)
        comparison_id = f"cmp_{uuid4().hex}"
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, written_at)
            self._assert_evidence_generation_mutable(connection, case_id, claim.generation)
            connection.execute(
                "INSERT INTO comparison_results("
                "comparison_id, case_id, comparison_type, payload_json, created_at, "
                "execution_generation) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    comparison_id,
                    case_id,
                    kind,
                    encode(payload),
                    written_at.isoformat(),
                    claim.generation,
                ),
            )
            connection.commit()
        return comparison_id

    def load_comparison(self, case_id: str, kind: str) -> Any:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM comparison_results "
                "WHERE case_id = ? AND comparison_type = ? ORDER BY created_at DESC LIMIT 1",
                (case_id, kind),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def load_current_comparison(self, claim: ExecutionClaim, kind: str) -> Any:
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            row = connection.execute(
                "SELECT payload_json FROM comparison_results WHERE case_id = ? "
                "AND comparison_type = ? AND execution_generation = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (claim.case_id, kind, claim.generation),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def save_critique(self, case_id: str, critique: Critique, claim: ExecutionClaim) -> str:
        critique_id = f"crit_{uuid4().hex}"
        self._insert_payload(
            "critique_results", "critique_id", critique_id, case_id, critique, claim
        )
        return critique_id

    def load_critique(self, case_id: str) -> Critique:
        payload = self._load_latest_payload("critique_results", case_id)
        if not payload:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"critic has not recorded a result for case {case_id}",
                case_id=case_id,
                stop_reason="CRITIQUE_MISSING",
            )
        return Critique.model_validate(payload)

    def load_current_critique(self, claim: ExecutionClaim) -> Critique:
        payload = self._load_current_payload("critique_results", claim)
        if not payload:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"current execution has no critique for case {claim.case_id}",
                case_id=claim.case_id,
                stop_reason="CRITIQUE_GENERATION_MISMATCH",
            )
        return Critique.model_validate(payload)

    def adopt_latest_evidence(self, claim: ExecutionClaim) -> None:
        """Promote one complete, coherent immediate-predecessor snapshot."""

        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                adopted_at = datetime.now(UTC)
                self._require_current_claim(connection, claim, adopted_at)
                self._assert_evidence_generation_mutable(
                    connection, claim.case_id, claim.generation
                )
                predecessor = claim.generation - 1
                if predecessor < 1:
                    self._raise_evidence_provenance(claim, "no predecessor generation exists")
                self._reject_future_evidence(connection, claim)
                try:
                    snapshot = load_generation_evidence_snapshot(
                        connection, claim.case_id, predecessor
                    )
                except EvidenceSnapshotError as exc:
                    self._raise_evidence_provenance(claim, str(exc))
                next_version = (
                    int(
                        connection.execute(
                            "SELECT COALESCE(MAX(version), 0) FROM extractions WHERE case_id = ?",
                            (claim.case_id,),
                        ).fetchone()[0]
                    )
                    + 1
                )
                connection.execute(
                    "INSERT INTO extractions(extraction_id, case_id, version, payload_json, "
                    "created_at, execution_generation) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        f"ext_{uuid4().hex}",
                        claim.case_id,
                        next_version,
                        snapshot.invoice.model_dump_json(),
                        adopted_at.isoformat(),
                        claim.generation,
                    ),
                )
                for table, id_column, prefix, payload_json in (
                    (
                        "identity_results",
                        "identity_id",
                        "ident",
                        encode([item.model_dump(mode="json") for item in snapshot.identity]),
                    ),
                    (
                        "critique_results",
                        "critique_id",
                        "crit",
                        snapshot.critique.model_dump_json(),
                    ),
                ):
                    connection.execute(
                        f"INSERT INTO {table}({id_column}, case_id, payload_json, created_at, "
                        "execution_generation) VALUES (?, ?, ?, ?, ?)",
                        (
                            f"{prefix}_{uuid4().hex}",
                            claim.case_id,
                            payload_json,
                            adopted_at.isoformat(),
                            claim.generation,
                        ),
                    )
                for comparison_type, payload_json in (
                    ("inventory", encode(snapshot.inventory_payload)),
                    ("risk", snapshot.risk.model_dump_json()),
                ):
                    connection.execute(
                        "INSERT INTO comparison_results(comparison_id, case_id, "
                        "comparison_type, payload_json, created_at, execution_generation) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            f"cmp_{uuid4().hex}",
                            claim.case_id,
                            comparison_type,
                            payload_json,
                            adopted_at.isoformat(),
                            claim.generation,
                        ),
                    )
                review = connection.execute(
                    "SELECT review_id, payload_json, evidence_snapshot_digest "
                    "FROM review_requests WHERE case_id = ? "
                    "AND execution_generation = ? ORDER BY sequence DESC LIMIT 1",
                    (claim.case_id, predecessor),
                ).fetchone()
                if review is not None:
                    try:
                        predecessor_review = ReviewRequest.model_validate_json(
                            review["payload_json"]
                        )
                        validate_review_snapshot(predecessor_review, snapshot)
                    except (EvidenceSnapshotError, ValueError) as exc:
                        self._raise_evidence_provenance(claim, str(exc))
                    if review["evidence_snapshot_digest"] != snapshot.digest:
                        self._raise_evidence_provenance(
                            claim, "predecessor review snapshot digest does not match evidence"
                        )
                    updated = connection.execute(
                        "UPDATE review_requests SET execution_generation = ? "
                        "WHERE review_id = ? AND execution_generation = ?",
                        (claim.generation, review["review_id"], predecessor),
                    )
                    if updated.rowcount != 1:
                        self._raise_evidence_provenance(
                            claim, "predecessor review changed during promotion"
                        )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def promote_predecessor_extraction(self, claim: ExecutionClaim) -> ExtractedInvoice:
        """Promote only preparation's immediate-predecessor extraction into a fresh run."""

        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                promoted_at = datetime.now(UTC)
                self._require_current_claim(connection, claim, promoted_at)
                self._assert_evidence_generation_mutable(
                    connection, claim.case_id, claim.generation
                )
                predecessor = claim.generation - 1
                if predecessor < 1:
                    self._raise_evidence_provenance(claim, "no predecessor generation exists")
                self._reject_future_evidence(connection, claim)
                row = connection.execute(
                    "SELECT payload_json FROM extractions WHERE case_id = ? "
                    "AND execution_generation = ? ORDER BY version DESC LIMIT 1",
                    (claim.case_id, predecessor),
                ).fetchone()
                if row is None:
                    self._raise_evidence_provenance(
                        claim, f"predecessor generation {predecessor} has no extraction"
                    )
                invoice = self._validate_case_invoice(connection, claim, row["payload_json"])
                next_version = (
                    int(
                        connection.execute(
                            "SELECT COALESCE(MAX(version), 0) FROM extractions WHERE case_id = ?",
                            (claim.case_id,),
                        ).fetchone()[0]
                    )
                    + 1
                )
                connection.execute(
                    "INSERT INTO extractions(extraction_id, case_id, version, payload_json, "
                    "created_at, execution_generation) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        f"ext_{uuid4().hex}",
                        claim.case_id,
                        next_version,
                        invoice.model_dump_json(),
                        promoted_at.isoformat(),
                        claim.generation,
                    ),
                )
                connection.commit()
                return invoice
            except BaseException:
                connection.rollback()
                raise

    def _reject_future_evidence(
        self, connection: sqlite3.Connection, claim: ExecutionClaim
    ) -> None:
        generation_columns = (
            ("extractions", "execution_generation"),
            ("identity_results", "execution_generation"),
            ("comparison_results", "execution_generation"),
            ("critique_results", "execution_generation"),
            ("review_requests", "execution_generation"),
            ("final_decisions", "decision_generation"),
            ("payments", "decision_generation"),
        )
        invalid = []
        for table, column in generation_columns:
            row = connection.execute(
                f"SELECT MAX({column}) AS generation FROM {table} WHERE case_id = ?",
                (claim.case_id,),
            ).fetchone()
            if row["generation"] is not None and int(row["generation"]) >= claim.generation:
                invalid.append(f"{table}:{row['generation']}")
        if invalid:
            self._raise_evidence_provenance(
                claim, f"current or future evidence generations exist: {sorted(invalid)}"
            )

    def _validate_case_invoice(
        self,
        connection: sqlite3.Connection,
        claim: ExecutionClaim,
        payload_json: str,
    ) -> ExtractedInvoice:
        try:
            invoice = ExtractedInvoice.model_validate_json(payload_json)
        except ValueError as exc:
            self._raise_evidence_provenance(claim, f"invalid predecessor extraction: {exc}")
        try:
            authoritative_source = _authoritative_source_for_case(connection, claim.case_id)
        except EvidenceSnapshotError as exc:
            self._raise_evidence_provenance(claim, str(exc))
        row = connection.execute(
            "SELECT source_id, invoice_number, vendor, revision FROM cases WHERE case_id = ?",
            (claim.case_id,),
        ).fetchone()
        revision = invoice.revision.normalized_value if invoice.revision else None
        if row is None or (
            invoice.source != authoritative_source
            or invoice.source.source_id != row["source_id"]
            or invoice.invoice_number.normalized_value != row["invoice_number"]
            or invoice.vendor.normalized_value != row["vendor"]
            or revision != row["revision"]
        ):
            self._raise_evidence_provenance(
                claim, "predecessor extraction does not match the case identity"
            )
        return invoice

    @staticmethod
    def _raise_evidence_provenance(claim: ExecutionClaim, reason: str) -> Never:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            f"cannot promote evidence for case {claim.case_id}: {reason}",
            case_id=claim.case_id,
            stop_reason="EVIDENCE_PROVENANCE_INVALID",
            details={"execution_generation": claim.generation, "reason": reason},
        )

    def save_review(self, review: ReviewRequest, claim: ExecutionClaim) -> ReviewRequest:
        """Persist the next review cycle for the case and return it with its sequence.

        The UNIQUE(case_id, sequence) index turns a concurrent double-insert into a
        visible IntegrityError instead of a silently reordered queue.
        """

        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, written_at)
            self._assert_evidence_generation_mutable(connection, review.case_id, claim.generation)
            if claim.case_id != review.case_id:
                self._raise_stale_execution_claim(claim)
            try:
                snapshot = load_generation_evidence_snapshot(
                    connection, review.case_id, claim.generation
                )
                validate_review_snapshot(review, snapshot)
            except EvidenceSnapshotError as exc:
                self._raise_snapshot_invalid(review.case_id, claim.generation, str(exc))
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM review_requests "
                "WHERE case_id = ?",
                (review.case_id,),
            ).fetchone()
            sequenced = review.model_copy(update={"sequence": int(row["sequence"]) + 1}, deep=True)
            connection.execute(
                "INSERT INTO review_requests("
                "review_id, case_id, sequence, status, payload_json, created_at, "
                "execution_generation, evidence_snapshot_digest) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sequenced.review_id,
                    sequenced.case_id,
                    sequenced.sequence,
                    sequenced.status,
                    sequenced.model_dump_json(),
                    sequenced.created_at.isoformat(),
                    claim.generation,
                    snapshot.digest,
                ),
            )
            connection.commit()
        return sequenced

    def load_review(self, review_id: str) -> ReviewRequest:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM review_requests WHERE review_id = ?", (review_id,)
            ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"review request does not exist: {review_id}",
                stop_reason="REVIEW_NOT_FOUND",
            )
        return ReviewRequest.model_validate_json(row["payload_json"])

    def load_case_review(self, case_id: str) -> ReviewRequest | None:
        """Return the latest review cycle for the case, by sequence."""

        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM review_requests WHERE case_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        return ReviewRequest.model_validate_json(row["payload_json"]) if row else None

    def load_current_review(self, claim: ExecutionClaim) -> ReviewRequest | None:
        """Return only the latest review owned by the current unexpired generation."""

        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            row = connection.execute(
                "SELECT payload_json FROM review_requests WHERE case_id = ? "
                "AND execution_generation = ? ORDER BY sequence DESC LIMIT 1",
                (claim.case_id, claim.generation),
            ).fetchone()
        return ReviewRequest.model_validate_json(row["payload_json"]) if row else None

    def list_reviews(self, pending_only: bool = True) -> list[ReviewRequest]:
        sql = "SELECT payload_json FROM review_requests"
        params: tuple[str, ...] = ()
        if pending_only:
            sql += " WHERE status = ?"
            params = ("PENDING",)
        sql += " ORDER BY created_at, sequence"
        with connect_database(self.path, read_only=True) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [ReviewRequest.model_validate_json(row["payload_json"]) for row in rows]

    def classify_human_decision_replay(
        self, review_id: str, decision: HumanDecision | None
    ) -> ReviewRequest | None:
        """Classify persisted resolved state without opening or touching inventory.

        ``None`` means the workflow review was pending at this preliminary read. The
        attached write transaction must reload it before validation or mutation.
        """

        with connect_database(self.path, read_only=True) as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT case_id, payload_json, execution_generation, "
                "evidence_snapshot_digest FROM review_requests WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            review = _review_from_row(row, review_id)
            if review.status == "PENDING":
                return None
            generation = int(row["execution_generation"])
            try:
                snapshot = load_generation_evidence_snapshot(
                    connection, str(row["case_id"]), generation
                )
                validate_review_snapshot(review, snapshot)
            except EvidenceSnapshotError as exc:
                self._raise_snapshot_invalid(str(row["case_id"]), generation, str(exc))
            if row["evidence_snapshot_digest"] != snapshot.digest:
                self._raise_snapshot_invalid(
                    str(row["case_id"]),
                    generation,
                    "resolved review digest does not match evidence",
                )
            return _resolved_human_decision_replay(review, decision)

    def save_human_decision(self, decision: HumanDecision, inventory_db: Path) -> ReviewRequest:
        """Atomically commit a validated decision and its aliases across both DB files."""

        with connect_database(self.path) as connection:
            # ATTACH accepts a bound filename expression; never interpolate a path into SQL.
            connection.execute("ATTACH DATABASE ? AS inventory_db", (str(inventory_db.resolve()),))
            modes = {
                "workflow": str(connection.execute("PRAGMA main.journal_mode").fetchone()[0]),
                "inventory": str(
                    connection.execute("PRAGMA inventory_db.journal_mode").fetchone()[0]
                ),
            }
            incompatible = {
                name: mode
                for name, mode in modes.items()
                if mode.casefold() not in ROLLBACK_JOURNAL_MODES
            }
            if incompatible:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "atomic human decisions require rollback-journal mode for workflow and "
                    f"inventory databases; incompatible modes: {incompatible}",
                    stop_reason="ATOMIC_JOURNAL_MODE_REQUIRED",
                )
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT case_id, payload_json, execution_generation, "
                    "evidence_snapshot_digest FROM review_requests WHERE review_id = ?",
                    (decision.review_id,),
                ).fetchone()
                review = _review_from_row(row, decision.review_id)
                case_id = str(row["case_id"])
                generation = int(row["execution_generation"])
                self._assert_evidence_generation_mutable(connection, case_id, generation)
                try:
                    snapshot = load_generation_evidence_snapshot(connection, case_id, generation)
                    validate_review_snapshot(review, snapshot)
                except EvidenceSnapshotError as exc:
                    self._raise_snapshot_invalid(case_id, generation, str(exc))
                if row["evidence_snapshot_digest"] != snapshot.digest:
                    self._raise_snapshot_invalid(
                        case_id, generation, "review digest does not match current evidence"
                    )
                if review.status == "RESOLVED":
                    replay = _resolved_human_decision_replay(review, decision)
                    connection.rollback()
                    return replay

                mappings = _validate_human_decision(connection, review, decision)
                resolved = review.model_copy(
                    update={"status": "RESOLVED", "human_decision": decision}, deep=True
                )
                decided_at = decision.decided_at.isoformat()
                for normalized, sku in mappings:
                    connection.execute(
                        "INSERT INTO inventory_db.item_aliases("
                        "alias_normalized, sku, source, approved_by, approved_at) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(alias_normalized) DO UPDATE SET "
                        "sku=excluded.sku, source=excluded.source, "
                        "approved_by=excluded.approved_by, approved_at=excluded.approved_at",
                        (
                            normalized,
                            sku,
                            f"human_review:{decision.review_id}",
                            decision.reviewer,
                            decided_at,
                        ),
                    )
                connection.execute(
                    "INSERT INTO human_decisions("
                    "decision_id, review_id, reviewer, decision, reason, payload_json, decided_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"hdec_{uuid4().hex}",
                        decision.review_id,
                        decision.reviewer,
                        decision.decision,
                        decision.reason,
                        decision.model_dump_json(),
                        decided_at,
                    ),
                )
                updated = connection.execute(
                    "UPDATE review_requests SET status = 'RESOLVED', payload_json = ?, "
                    "resolved_at = ? WHERE review_id = ? AND status = 'PENDING'",
                    (resolved.model_dump_json(), decided_at, decision.review_id),
                )
                if updated.rowcount != 1:
                    raise InvoiceAgentsError(
                        ErrorCategory.DATABASE,
                        f"review {decision.review_id} changed while recording its decision",
                        case_id=review.case_id,
                        stop_reason="REVIEW_ALREADY_RESOLVED",
                    )
                connection.commit()
                return resolved
            except BaseException:
                connection.rollback()
                raise

    def save_final_decision(
        self, case_id: str, decision: FinalDecision, claim: ExecutionClaim
    ) -> None:
        if claim.case_id != case_id:
            self._raise_stale_execution_claim(claim)
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC).isoformat()
            self._require_current_claim(connection, claim, datetime.fromisoformat(written_at))
            paid = connection.execute(
                "SELECT 1 FROM payments WHERE case_id = ? AND status = 'PAID' LIMIT 1",
                (case_id,),
            ).fetchone()
            if paid is not None:
                connection.rollback()
                raise InvoiceAgentsError(
                    ErrorCategory.PAYMENT,
                    f"paid case {case_id} has an immutable final decision",
                    case_id=case_id,
                    stop_reason="PAID_FINAL_DECISION_IMMUTABLE",
                )
            try:
                snapshot = load_generation_evidence_snapshot(connection, case_id, claim.generation)
            except EvidenceSnapshotError as exc:
                self._raise_snapshot_invalid(case_id, claim.generation, str(exc))
            review_row = connection.execute(
                "SELECT payload_json, execution_generation, evidence_snapshot_digest "
                "FROM review_requests WHERE case_id = ? ORDER BY sequence DESC LIMIT 1",
                (case_id,),
            ).fetchone()
            review = None
            if review_row is not None:
                if int(review_row["execution_generation"]) != claim.generation:
                    self._raise_snapshot_invalid(
                        case_id, claim.generation, "latest review belongs to another generation"
                    )
                try:
                    review = ReviewRequest.model_validate_json(review_row["payload_json"])
                    validate_review_snapshot(review, snapshot)
                except (EvidenceSnapshotError, ValueError) as exc:
                    self._raise_snapshot_invalid(case_id, claim.generation, str(exc))
                if review_row["evidence_snapshot_digest"] != snapshot.digest:
                    self._raise_snapshot_invalid(
                        case_id, claim.generation, "review digest does not match current evidence"
                    )
            from invoice_agents.agents.decision_rules import validate_final_decision

            validate_final_decision(
                decision.decision,
                decision.payment_eligible,
                snapshot.risk,
                snapshot.critique,
                review,
                case_id=case_id,
            )
            human = review.human_decision if review is not None else None
            if (
                decision.human_outcome != human
                or decision.critic_disposition is not snapshot.critique.recommended_disposition
            ):
                self._raise_snapshot_invalid(
                    case_id,
                    claim.generation,
                    "final decision does not match review or critique snapshot",
                )
            updated = connection.execute(
                "INSERT INTO final_decisions("
                "decision_id, case_id, payload_json, created_at, decision_generation, "
                "evidence_snapshot_digest) "
                "SELECT ?, case_id, ?, ?, ?, ? FROM cases WHERE case_id = ? "
                "AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ? "
                "ON CONFLICT(case_id) DO UPDATE SET payload_json=excluded.payload_json, "
                "created_at=excluded.created_at, "
                "decision_generation=excluded.decision_generation, "
                "evidence_snapshot_digest=excluded.evidence_snapshot_digest WHERE EXISTS ("
                "SELECT 1 FROM cases WHERE cases.case_id = excluded.case_id "
                "AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?)",
                (
                    f"fdec_{uuid4().hex}",
                    decision.model_dump_json(),
                    written_at,
                    claim.generation,
                    snapshot.digest,
                    case_id,
                    claim.token,
                    claim.generation,
                    written_at,
                    claim.token,
                    claim.generation,
                    written_at,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                self._raise_stale_execution_claim(claim)
            connection.commit()

    def load_final_decision(self, case_id: str) -> FinalDecision | None:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM final_decisions WHERE case_id = ?", (case_id,)
            ).fetchone()
        return FinalDecision.model_validate_json(row["payload_json"]) if row else None

    def load_current_final_decision(self, claim: ExecutionClaim) -> FinalDecision | None:
        """Return only a final decision owned by the current unexpired generation."""

        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            row = connection.execute(
                "SELECT payload_json FROM final_decisions WHERE case_id = ? "
                "AND decision_generation = ?",
                (claim.case_id, claim.generation),
            ).fetchone()
        return FinalDecision.model_validate_json(row["payload_json"]) if row else None

    def save_team_state(self, case_id: str, state: dict[str, Any], claim: ExecutionClaim) -> None:
        if claim.case_id != case_id:
            self._raise_stale_execution_claim(claim)
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC).isoformat()
            self._require_current_claim(connection, claim, datetime.fromisoformat(written_at))
            updated = connection.execute(
                "UPDATE cases SET team_state_json = ?, updated_at = ? WHERE case_id = ? "
                "AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?",
                (
                    encode(state),
                    written_at,
                    case_id,
                    claim.token,
                    claim.generation,
                    written_at,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                self._raise_stale_execution_claim(claim)
            connection.commit()

    def load_team_state(self, case_id: str) -> dict[str, Any] | None:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT team_state_json FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"case does not exist: {case_id}",
                case_id=case_id,
                stop_reason="CASE_NOT_FOUND",
            )
        return json.loads(row["team_state_json"]) if row["team_state_json"] else None

    def finish_case(self, result: CaseResult, claim: ExecutionClaim) -> None:
        if claim.case_id != result.case_id:
            self._raise_stale_execution_claim(claim)
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC).isoformat()
            self._require_current_claim(connection, claim, datetime.fromisoformat(written_at))
            updated = connection.execute(
                "UPDATE cases SET status = ?, stop_reason = ?, result_json = ?, updated_at = ?, "
                "finished_at = ?, execution_state = 'FINISHED', lease_expires_at = NULL "
                "WHERE case_id = ? AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?",
                (
                    result.status,
                    result.stop_reason,
                    result.model_dump_json(),
                    written_at,
                    result.finished_at.isoformat(),
                    result.case_id,
                    claim.token,
                    claim.generation,
                    written_at,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                self._raise_stale_execution_claim(claim)
            connection.commit()

    def load_result(self, case_id: str) -> CaseResult | None:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT result_json FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"case does not exist: {case_id}",
                case_id=case_id,
                stop_reason="CASE_NOT_FOUND",
            )
        return CaseResult.model_validate_json(row["result_json"]) if row["result_json"] else None

    def count_events(self, case_id: str, event_type: str) -> int:
        """Count persisted audit events of one type; retries use 'provider.retry'."""

        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS event_count FROM events WHERE case_id = ? AND event_type = ?",
                (case_id, event_type),
            ).fetchone()
        return int(row["event_count"])

    def identity_rows(
        self, case_id: str, invoice_number: str | None, vendor: str | None
    ) -> list[Any]:
        with connect_database(self.path, read_only=True) as connection:
            return connection.execute(
                "SELECT c.case_id, c.invoice_number, c.vendor, c.revision, "
                "s.source_id, s.source_hash, s.source_format "
                "FROM cases c JOIN source_artifacts s ON s.source_id = c.source_id "
                "WHERE c.case_id <> ? AND (c.invoice_number = ? OR c.vendor = ?)",
                (case_id, invoice_number, vendor),
            ).fetchall()

    def _insert_payload(
        self,
        table: str,
        id_column: str,
        record_id: str,
        case_id: str,
        payload: Any,
        claim: ExecutionClaim,
    ) -> None:
        allowed = {"identity_results", "critique_results"}
        if table not in allowed:
            raise ValueError(f"unsupported payload table: {table}")
        if claim.case_id != case_id:
            self._raise_stale_execution_claim(claim)
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, written_at)
            self._assert_evidence_generation_mutable(connection, case_id, claim.generation)
            connection.execute(
                f"INSERT INTO {table}("
                f"{id_column}, case_id, payload_json, created_at, execution_generation) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record_id,
                    case_id,
                    encode(payload),
                    written_at.isoformat(),
                    claim.generation,
                ),
            )
            connection.commit()

    def _load_latest_payload(self, table: str, case_id: str) -> Any:
        allowed = {"identity_results", "critique_results"}
        if table not in allowed:
            raise ValueError(f"unsupported payload table: {table}")
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                f"SELECT payload_json FROM {table} WHERE case_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else []

    def _load_current_payload(self, table: str, claim: ExecutionClaim) -> Any:
        allowed = {"identity_results", "critique_results"}
        if table not in allowed:
            raise ValueError(f"unsupported payload table: {table}")
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            row = connection.execute(
                f"SELECT payload_json FROM {table} WHERE case_id = ? "
                "AND execution_generation = ? ORDER BY created_at DESC LIMIT 1",
                (claim.case_id, claim.generation),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else []

    def _begin_current_read(self, connection: sqlite3.Connection, claim: ExecutionClaim) -> None:
        """Pin a read snapshot after proving the claim is current in that snapshot."""

        connection.execute("BEGIN")
        self._require_current_claim(connection, claim, datetime.now(UTC))
