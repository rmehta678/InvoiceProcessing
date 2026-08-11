"""Typed persistence for case state, evidence, review, decisions, and results."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Never, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter

from invoice_agents.config import Settings
from invoice_agents.db.core import _strict_critic_follow_up_payload, connect_database
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.evidence_snapshot import (
    EvidenceSnapshot,
    EvidenceSnapshotError,
    build_evidence_snapshot,
    validate_final_decision_snapshot,
    validate_review_snapshot,
)
from invoice_agents.models import (
    CaseResult,
    CaseStatus,
    Critique,
    CritiqueFollowUpResponse,
    ErrorRecord,
    ExtractedInvoice,
    FinalDecision,
    HumanDecision,
    HumanDecisionKind,
    IdentityCandidate,
    InventoryComparison,
    InventoryLookupResult,
    Money,
    PaymentResult,
    PaymentStatus,
    PersistedPaymentRow,
    ReviewRequest,
    RiskAssessment,
    SourceArtifact,
)
from invoice_agents.observability.audit import sanitize_case_result, sanitize_text

ModelT = TypeVar("ModelT", bound=BaseModel)

REQUIRED_JOURNAL_MODE = "delete"
_DATETIME_WIRE_TYPE = datetime
_DATETIME_ADAPTER = TypeAdapter(datetime)
_REQUESTED_EXECUTION_TOKEN = re.compile(r"^exec_[0-9a-f]{32}$")
_ARTIFACT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUBMISSION_REQUEST_ID = re.compile(r"^submission_[A-Za-z0-9][A-Za-z0-9_-]{0,126}$")
_CASE_LIVE_TARGET = re.compile(r"^/cases/(case_[0-9a-f]{32})/live$")
_BATCH_TARGET = re.compile(r"^/batches/(batch_[0-9a-f]{24})$")
_STORED_FINISHED_AT_OMITTED = object()
_DIRECT_CRITIQUE_FOLLOW_UP_EVENT_TYPES = frozenset(
    {"tool.critic_inventory_recheck", "tool.critic_line_recompute"}
)
_PERSISTED_SPECIALIST_FOLLOW_UP_EVENT_TYPES = frozenset(
    {
        "tool.identity_candidates",
        "tool.inventory_comparison",
        "tool.mapping_evidence_recorded",
        "tool.financial_risk_assessment",
    }
)


def _validated_requested_execution_token(token: object) -> str:
    if type(token) is not str or _REQUESTED_EXECUTION_TOKEN.fullmatch(token) is None:
        raise ValueError("requested execution token is not canonical")
    return token


def _execute_result_artifact_binding(
    connection: sqlite3.Connection,
    statement: str,
    parameters: tuple[object, ...],
) -> sqlite3.Cursor:
    """Execute the one binding mutation through an injectable kernel seam."""

    return connection.execute(statement, parameters)


def _commit_result_artifact_binding(connection: sqlite3.Connection) -> None:
    """Commit the binding transaction through an injectable kernel seam."""

    connection.commit()


def _rollback_result_artifact_binding(connection: sqlite3.Connection) -> None:
    """Rollback the binding transaction through an injectable kernel seam."""

    connection.rollback()


def _binding_failure_precedence(
    primary: BaseException,
    *secondary: BaseException | None,
) -> BaseException:
    """Return the earliest process control, otherwise the primary failure."""

    for failure in (primary, *secondary):
        if failure is not None and not isinstance(failure, Exception):
            return failure
    return primary


def _raise_chainless(error: BaseException) -> Never:
    error.__cause__ = None
    error.__context__ = None
    raise error from None


@dataclass(frozen=True, slots=True)
class ExecutionClaim:
    """Database-issued fencing authority for one case execution generation."""

    case_id: str
    token: str
    generation: int
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class StagedInvoiceAdmission:
    """One fully inspected source payload that has not mutated workflow SQLite."""

    submitted_path: Path
    source: SourceArtifact
    invoice: ExtractedInvoice
    case_id: str
    started_at: datetime
    execution_token: str
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class AdmittedCase:
    """Durable case admission; ``claim`` exists only for newly authorized work."""

    source_id: str
    source_path: Path
    case_id: str
    started_at: datetime
    state: Literal["queued", "running", "done", "failed"]
    claim: ExecutionClaim | None


@dataclass(frozen=True, slots=True)
class SubmissionAdmission:
    """The exact durable redirect and cases bound to one submission request."""

    request_id: str
    created_at: datetime
    kind: Literal["single", "batch"]
    fingerprint: str
    redirect_target: str
    cases: tuple[AdmittedCase, ...]
    batch_id: str | None = None


@dataclass(frozen=True, slots=True)
class PersistedBatchEntry:
    source_id: str
    source_path: Path
    case_id: str
    started_at: datetime
    state: Literal["queued", "running", "done", "failed"]
    result: CaseResult | None


@dataclass(frozen=True, slots=True)
class PersistedBatch:
    batch_id: str
    created_at: datetime
    concurrency: int
    state: Literal["queued", "running", "done", "failed"]
    entries: tuple[PersistedBatchEntry, ...]


@dataclass(frozen=True, slots=True)
class CaseExecutionSnapshot:
    """One authoritative cases-row view for SSE terminal/lease decisions."""

    started_at: datetime
    result: CaseResult | None
    execution_state: str
    has_valid_lease: bool
    execution_token: str | None
    execution_generation: int
    lease_expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class ResultArtifactBinding:
    """Exact durable file identity authorized for one terminal generation."""

    case_id: str
    execution_generation: int
    artifact_sha256: str
    artifact_device: int
    artifact_inode: int
    artifact_file_type: int
    artifact_size_bytes: int


@dataclass(frozen=True, slots=True)
class ReviewAuthorization:
    """Relationally reconciled review state and its review-time snapshot digest."""

    review: ReviewRequest
    evidence_snapshot_digest: str
    execution_generation: int


@dataclass(frozen=True, slots=True)
class ValidatedEvidenceFacts:
    """Values independently derived before sealing one final/payment snapshot."""

    policy_review_required: int
    unresolved_blocker_count: int
    critique_disposition: str
    review_id: str | None
    review_snapshot_digest: str | None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def submission_fingerprint(kind: Literal["single", "batch"], source_ids: tuple[str, ...]) -> str:
    """Hash the exact request kind and ordered immutable source identities."""

    canonical = json.dumps(
        {"kind": kind, "source_ids": list(source_ids)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


def _runtime_utc_datetime(value: object) -> datetime | None:
    """Accept exact datetime values at semantic UTC without tzinfo identity assumptions."""

    if type(value) is not _DATETIME_WIRE_TYPE:
        return None
    offset = value.utcoffset()
    return value if value.tzinfo is not None and offset == timedelta(0) else None


def _canonical_pydantic_datetime_wire(value: datetime) -> str | None:
    """Return Pydantic's one canonical JSON UTC scalar (the ``Z`` form)."""

    if _runtime_utc_datetime(value) is None:
        return None
    try:
        encoded = _DATETIME_ADAPTER.dump_json(value)
        wire = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return wire if type(wire) is str and wire.endswith("Z") else None


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate persisted result key")
        payload[key] = value
    return payload


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite persisted result number")


def _decode_terminal_result_json(raw: object) -> CaseResult:
    """Decode a strict terminal aggregate and require canonical datetime wire scalars."""

    if type(raw) is not str or not raw:
        raise ValueError("terminal result JSON is missing")
    payload = json.loads(
        raw,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if type(payload) is not dict:
        raise ValueError("terminal result JSON is not an object")
    result = sanitize_case_result(CaseResult.model_validate_json(raw, strict=True))
    for field in ("started_at", "finished_at"):
        wire = payload.get(field)
        value = getattr(result, field)
        if type(wire) is not str or wire != _canonical_pydantic_datetime_wire(value):
            raise ValueError(f"terminal result {field} is not canonical UTC")
    return result


def execution_claim_expiry_iso(claim: ExecutionClaim) -> str | None:
    """Return the claim's canonical UTC lease text or reject its runtime shape."""

    if type(claim) is not ExecutionClaim:
        return None
    expires_at = claim.expires_at
    if (
        type(claim.case_id) is not str
        or not claim.case_id.strip()
        or claim.case_id != claim.case_id.strip()
        or type(claim.token) is not str
        or _REQUESTED_EXECUTION_TOKEN.fullmatch(claim.token) is None
        or type(claim.generation) is not int
        or claim.generation <= 0
        or type(expires_at) is not _DATETIME_WIRE_TYPE
        or expires_at.tzinfo is not UTC
    ):
        return None
    encoded = expires_at.isoformat()
    parsed = parse_canonical_utc(encoded)
    return encoded if parsed == expires_at else None


def validate_execution_claim(
    claim: object,
    *,
    expected_case_id: str | None = None,
) -> ExecutionClaim:
    """Return one exact canonical claim or fail before any storage/provider boundary."""

    if type(claim) is ExecutionClaim:
        exact = claim
        valid = execution_claim_expiry_iso(exact) is not None
        if expected_case_id is not None:
            valid = (
                valid
                and type(expected_case_id) is str
                and bool(expected_case_id.strip())
                and exact.case_id == expected_case_id
            )
        if valid:
            return exact
    case_id = (
        claim.case_id
        if type(claim) is ExecutionClaim
        and type(claim.case_id) is str
        and bool(claim.case_id.strip())
        else expected_case_id
        if type(expected_case_id) is str and bool(expected_case_id.strip())
        else None
    )
    raise InvoiceAgentsError(
        ErrorCategory.ORCHESTRATION,
        "execution claim has an invalid runtime shape or case binding",
        case_id=case_id,
        stop_reason="STALE_EXECUTION_CLAIM",
    ) from None


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


def _authoritative_identity_for_scope(
    connection: sqlite3.Connection,
    case_id: str,
    invoice: ExtractedInvoice,
    evaluated_at: datetime,
) -> tuple[IdentityCandidate, ...]:
    """Rebuild the exact identity candidates visible when identity was evaluated."""

    from invoice_agents.tools.comparison import classify_identity_candidate
    from invoice_agents.tools.evidence import extract_invoice_evidence

    candidates: list[IdentityCandidate] = []
    rows = connection.execute(
        "SELECT case_id, started_at FROM cases WHERE case_id <> ? ORDER BY case_id",
        (case_id,),
    ).fetchall()
    for row in rows:
        started_at = parse_canonical_utc(row["started_at"])
        if started_at is None:
            raise EvidenceSnapshotError(
                f"identity scope case {row['case_id']} has a noncanonical started_at"
            )
        if started_at > evaluated_at:
            continue
        prior_case_id = str(row["case_id"])
        try:
            prior = extract_invoice_evidence(
                _authoritative_source_for_case(connection, prior_case_id)
            )
        except InvoiceAgentsError as exc:
            raise EvidenceSnapshotError(
                f"identity scope case {prior_case_id} cannot be re-extracted: {exc}"
            ) from exc
        if not (
            prior.source.sha256 == invoice.source.sha256
            or prior.invoice_number.normalized_value == invoice.invoice_number.normalized_value
            or prior.vendor.normalized_value == invoice.vendor.normalized_value
        ):
            continue
        candidates.append(classify_identity_candidate(prior_case_id, invoice, prior))
    return tuple(candidates)


def load_generation_evidence_snapshot(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
    settings: Settings,
    *,
    require_latest: bool = True,
    excluded_alias_sources: frozenset[str] = frozenset(),
    inventory_connection: sqlite3.Connection | None = None,
    inventory_schema: str = "main",
) -> EvidenceSnapshot:
    """Load one generation's evidence plus the latest durable critique cycle."""

    authoritative_critique = reconcile_critique_follow_up_evidence(
        connection,
        case_id,
        generation,
    )
    specifications = {
        "extraction": (
            "SELECT payload_json FROM extractions WHERE case_id = ? "
            "AND execution_generation = ? ORDER BY version DESC LIMIT 1",
            "extractions",
            "execution_generation",
            "",
            False,
        ),
        "identity": (
            "SELECT payload_json, evaluated_at FROM identity_results WHERE case_id = ? "
            "AND execution_generation = ? ORDER BY rowid DESC LIMIT 1",
            "identity_results",
            "execution_generation",
            "",
            False,
        ),
        "inventory": (
            "SELECT payload_json FROM comparison_results WHERE case_id = ? "
            "AND execution_generation = ? AND comparison_type = 'inventory' "
            "ORDER BY rowid DESC LIMIT 1",
            "comparison_results",
            "execution_generation",
            "AND comparison_type = 'inventory'",
            False,
        ),
        "risk": (
            "SELECT payload_json FROM comparison_results WHERE case_id = ? "
            "AND execution_generation = ? AND comparison_type = 'risk' "
            "ORDER BY rowid DESC LIMIT 1",
            "comparison_results",
            "execution_generation",
            "AND comparison_type = 'risk'",
            False,
        ),
        "critique": (
            "SELECT payload_json FROM critique_results WHERE case_id = ? "
            "AND execution_generation <= ? ORDER BY cycle DESC LIMIT 1",
            "critique_results",
            "execution_generation",
            "",
            True,
        ),
    }
    payloads: dict[str, str] = {}
    missing: list[str] = []
    stale: list[str] = []
    identity_evaluated_at: datetime | None = None
    for name, (sql, table, column, predicate, carry_forward) in specifications.items():
        row = connection.execute(sql, (case_id, generation)).fetchone()
        if row is None:
            missing.append(name)
            continue
        payloads[name] = str(row["payload_json"])
        if name == "identity":
            identity_evaluated_at = parse_canonical_utc(row["evaluated_at"])
            if identity_evaluated_at is None:
                raise EvidenceSnapshotError(
                    "identity result has no canonical UTC evaluation boundary"
                )
        if require_latest:
            latest = connection.execute(
                f"SELECT MAX({column}) AS generation FROM {table} WHERE case_id = ? {predicate}",
                (case_id,),
            ).fetchone()["generation"]
            if latest is None or (
                int(latest) > generation if carry_forward else int(latest) != generation
            ):
                stale.append(f"{name}:{latest}")
    if missing or stale:
        raise EvidenceSnapshotError(
            f"generation {generation} evidence is missing={sorted(missing)} stale={sorted(stale)}"
        )
    if identity_evaluated_at is None:
        raise EvidenceSnapshotError("identity evaluation boundary is missing")
    if authoritative_critique is not None:
        try:
            selected_critique = Critique.model_validate_json(
                payloads["critique"],
                strict=True,
            )
        except ValueError as exc:
            raise EvidenceSnapshotError("critique payload is invalid") from exc
        if selected_critique != authoritative_critique:
            raise EvidenceSnapshotError(
                "critique follow-up evidence does not identify the authoritative cycle"
            )
    try:
        stored_invoice = ExtractedInvoice.model_validate_json(payloads["extraction"])
    except ValueError as exc:
        raise EvidenceSnapshotError(f"extraction payload is invalid: {exc}") from exc
    authoritative_identity = _authoritative_identity_for_scope(
        connection,
        case_id,
        stored_invoice,
        identity_evaluated_at,
    )
    return build_evidence_snapshot(
        case_id,
        _authoritative_source_for_case(connection, case_id),
        payloads["extraction"],
        payloads["identity"],
        payloads["inventory"],
        payloads["risk"],
        payloads["critique"],
        identity_evaluated_at=identity_evaluated_at,
        authoritative_identity=authoritative_identity,
        settings=settings,
        excluded_alias_sources=excluded_alias_sources,
        inventory_connection=inventory_connection,
        inventory_schema=inventory_schema,
    )


def reconcile_critique_follow_up_evidence(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
) -> Critique | None:
    """Reconcile critique payloads, relationships, events, and evidence source rows."""

    if type(generation) is not int or generation < 1:
        raise EvidenceSnapshotError("critique evidence generation is invalid")
    try:
        records = WorkflowStore._critique_records(connection, case_id)
    except InvoiceAgentsError as exc:
        raise EvidenceSnapshotError(
            f"critique follow-up evidence is inconsistent: {exc.message}"
        ) from exc
    authoritative = [
        critique
        for _critique_id, critique, critique_generation in records
        if critique_generation <= generation
    ]
    return authoritative[-1] if authoritative else None


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
    try:
        return ReviewRequest.model_validate_json(row["payload_json"], strict=True)
    except ValueError as exc:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            f"review authorization payload is invalid: {review_id}",
            stop_reason="REVIEW_AUTHORIZATION_INVALID",
        ) from exc


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


def _review_alias_sources(review: ReviewRequest | None) -> frozenset[str]:
    if (
        review is not None
        and review.status == "RESOLVED"
        and review.human_decision is not None
        and review.human_decision.decision is HumanDecisionKind.ESTABLISH_MAPPING
    ):
        return frozenset({f"human_review:{review.review_id}"})
    return frozenset()


def _reconcile_review_authorization(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    case_id: str,
) -> ReviewAuthorization:
    """Reconcile one review JSON document with every authoritative relational column."""

    from invoice_agents.agents.decision_rules import AUTHORIZING_HUMAN_DECISIONS

    inconsistent = EvidenceSnapshotError("review authorization records are inconsistent")
    try:
        review = ReviewRequest.model_validate_json(row["payload_json"], strict=True)
    except ValueError as exc:
        raise inconsistent from exc
    human_rows = connection.execute(
        "SELECT review_id, reviewer, decision, reason, payload_json, decided_at "
        "FROM human_decisions WHERE review_id = ?",
        (review.review_id,),
    ).fetchall()
    base_columns_match = (
        review.case_id == case_id
        and review.review_id == row["review_id"]
        and row["case_id"] == case_id
        and int(row["sequence"]) == review.sequence
        and row["status"] == review.status
        and row["created_at"] == review.created_at.isoformat()
        and int(row["execution_generation"]) >= 1
        and isinstance(row["evidence_snapshot_digest"], str)
    )
    if not base_columns_match:
        raise inconsistent
    if review.status == "PENDING":
        if review.human_decision is not None or row["resolved_at"] is not None or human_rows:
            raise inconsistent
        return ReviewAuthorization(
            review,
            str(row["evidence_snapshot_digest"]),
            int(row["execution_generation"]),
        )

    human = review.human_decision
    if row["resolved_at"] is None or human is None or len(human_rows) != 1:
        raise inconsistent
    human_row = human_rows[0]
    try:
        relational_human = HumanDecision.model_validate_json(human_row["payload_json"], strict=True)
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
    return ReviewAuthorization(
        review,
        str(row["evidence_snapshot_digest"]),
        int(row["execution_generation"]),
    )


def load_authoritative_review_authorization(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
) -> ReviewAuthorization | None:
    """Load the latest review and require it to belong exactly to this generation."""

    row = connection.execute(
        "SELECT review_id, case_id, sequence, status, payload_json, created_at, resolved_at, "
        "execution_generation, evidence_snapshot_digest FROM review_requests WHERE case_id = ? "
        "ORDER BY sequence DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    if row is None:
        return None
    if int(row["execution_generation"]) != generation:
        raise EvidenceSnapshotError("latest review belongs to another generation")
    return _reconcile_review_authorization(connection, row, case_id)


def _mapping_review_authorizations(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
) -> tuple[ReviewAuthorization, ...]:
    rows = connection.execute(
        "SELECT review_id, case_id, sequence, status, payload_json, created_at, resolved_at, "
        "execution_generation, evidence_snapshot_digest FROM review_requests WHERE case_id = ? "
        "AND execution_generation <= ? ORDER BY sequence",
        (case_id, generation),
    ).fetchall()
    authorizations = tuple(
        _reconcile_review_authorization(connection, row, case_id) for row in rows
    )
    return tuple(
        authorization
        for authorization in authorizations
        if authorization.review.status == "RESOLVED"
        and authorization.review.human_decision is not None
        and authorization.review.human_decision.decision is HumanDecisionKind.ESTABLISH_MAPPING
    )


def _validate_mapping_successor(
    snapshot: EvidenceSnapshot,
    mapping_authorizations: tuple[ReviewAuthorization, ...],
    settings: Settings,
    *,
    inventory_connection: sqlite3.Connection | None = None,
    inventory_schema: str = "main",
) -> None:
    """Require exact persisted alias provenance and its exact mapped successor evidence."""

    expected: dict[str, tuple[str, str, str, str]] = {}
    for authorization in mapping_authorizations:
        human = authorization.review.human_decision
        if human is None:
            raise EvidenceSnapshotError("mapping review has no authoritative human decision")
        source = f"human_review:{authorization.review.review_id}"
        for mapping in human.mappings:
            alias = _normalize_alias(mapping.raw_item)
            value = (mapping.sku, source, human.reviewer, human.decided_at.isoformat())
            previous = expected.get(alias)
            if not alias or (previous is not None and previous != value):
                raise EvidenceSnapshotError(
                    "mapping reviews contain conflicting alias authorization"
                )
            expected[alias] = value
    if not expected:
        return
    if not inventory_schema.replace("_", "a").isalnum():
        raise EvidenceSnapshotError("inventory schema name is invalid")
    sql = (
        "SELECT alias_normalized, sku, source, approved_by, approved_at "
        f'FROM "{inventory_schema}"."item_aliases" '
        f"WHERE alias_normalized IN ({', '.join('?' for _ in expected)})"
    )
    parameters = tuple(expected)
    if inventory_connection is None:
        with connect_database(settings.inventory_db, read_only=True) as inventory:
            rows = inventory.execute(sql, parameters).fetchall()
    else:
        rows = inventory_connection.execute(
            sql,
            parameters,
        ).fetchall()
    actual = {
        str(row["alias_normalized"]): (
            str(row["sku"]),
            str(row["source"]),
            str(row["approved_by"]),
            str(row["approved_at"]),
        )
        for row in rows
    }
    if actual != expected:
        raise EvidenceSnapshotError("approved mapping alias provenance is not exact")
    for alias, (sku, _source, _reviewer, _approved_at) in expected.items():
        mapped_lines = [
            line for line in snapshot.invoice.lines if _normalize_alias(line.raw_item) == alias
        ]
        if not mapped_lines or any(line.canonical_sku != sku for line in mapped_lines):
            raise EvidenceSnapshotError(
                "recomputed successor does not include the exact approved mapping"
            )
        if not any(
            comparison.sku == sku
            and any(_normalize_alias(raw_item) == alias for raw_item in comparison.raw_items)
            for comparison in snapshot.inventory
        ):
            raise EvidenceSnapshotError(
                "recomputed inventory successor does not include the exact approved mapping"
            )


def _validate_mapping_review_snapshots(
    connection: sqlite3.Connection,
    case_id: str,
    settings: Settings,
    authorizations: tuple[ReviewAuthorization, ...],
    *,
    inventory_connection: sqlite3.Connection | None = None,
    inventory_schema: str = "main",
) -> None:
    """Bind every persisted mapping ruling to its exact review-time predecessor."""

    for authorization in authorizations:
        predecessor = authorization.execution_generation - 1
        if predecessor < 1:
            raise EvidenceSnapshotError("mapping review has no predecessor evidence generation")
        review_snapshot = load_generation_evidence_snapshot(
            connection,
            case_id,
            predecessor,
            settings,
            require_latest=False,
            excluded_alias_sources=_review_alias_sources(authorization.review),
            inventory_connection=inventory_connection,
            inventory_schema=inventory_schema,
        )
        validate_review_snapshot(authorization.review, review_snapshot)
        if authorization.evidence_snapshot_digest != review_snapshot.digest:
            raise EvidenceSnapshotError("mapping review digest does not match review-time evidence")


def validate_review_authorization_snapshot(
    connection: sqlite3.Connection,
    authorization: ReviewAuthorization,
    settings: Settings,
    *,
    inventory_connection: sqlite3.Connection | None = None,
    inventory_schema: str = "main",
) -> EvidenceSnapshot:
    """Re-derive the one review-time snapshot identified by a review row.

    A resolved mapping review moves forward one generation only when its exact
    mapped successor is adopted.  Before that adoption, its review-time evidence
    is in the same generation.  Enumerate both lineage positions and require one
    unambiguous match; a stored digest is checked only after semantic validation.
    """

    review = authorization.review
    human = review.human_decision
    generations = [authorization.execution_generation]
    if human is not None and human.decision is HumanDecisionKind.ESTABLISH_MAPPING:
        predecessor = authorization.execution_generation - 1
        if predecessor >= 1:
            generations.append(predecessor)
    matches: list[EvidenceSnapshot] = []
    for generation in generations:
        try:
            snapshot = load_generation_evidence_snapshot(
                connection,
                review.case_id,
                generation,
                settings,
                require_latest=False,
                excluded_alias_sources=_review_alias_sources(review),
                inventory_connection=inventory_connection,
                inventory_schema=inventory_schema,
            )
            validate_review_snapshot(review, snapshot)
        except EvidenceSnapshotError:
            continue
        if authorization.evidence_snapshot_digest == snapshot.digest:
            matches.append(snapshot)
    if len(matches) != 1:
        raise EvidenceSnapshotError(
            "review authorization does not identify exactly one authoritative snapshot"
        )
    if human is not None:
        try:
            if inventory_connection is None:
                with connect_database(settings.inventory_db, read_only=True) as inventory:
                    _validate_human_decision_authority(
                        connection,
                        review,
                        human,
                        inventory_connection=inventory,
                        inventory_schema="main",
                        require_persisted_mapping_provenance=True,
                    )
            else:
                _validate_human_decision_authority(
                    connection,
                    review,
                    human,
                    inventory_connection=inventory_connection,
                    inventory_schema=inventory_schema,
                    require_persisted_mapping_provenance=True,
                )
        except InvoiceAgentsError as exc:
            raise EvidenceSnapshotError(str(exc)) from exc
    return matches[0]


def load_authorization_evidence_snapshot(
    connection: sqlite3.Connection,
    case_id: str,
    generation: int,
    settings: Settings,
    review_authorization: ReviewAuthorization | None,
    *,
    inventory_connection: sqlite3.Connection | None = None,
    inventory_schema: str = "main",
) -> EvidenceSnapshot:
    """Revalidate review-time evidence and the exact current authorization successor."""

    mapping_authorizations = _mapping_review_authorizations(connection, case_id, generation)
    _validate_mapping_review_snapshots(
        connection,
        case_id,
        settings,
        mapping_authorizations,
        inventory_connection=inventory_connection,
        inventory_schema=inventory_schema,
    )
    current = load_generation_evidence_snapshot(
        connection,
        case_id,
        generation,
        settings,
        inventory_connection=inventory_connection,
        inventory_schema=inventory_schema,
    )
    mapping_review_ids = {
        authorization.review.review_id for authorization in mapping_authorizations
    }
    if (
        review_authorization is not None
        and review_authorization.review.review_id not in mapping_review_ids
    ):
        validate_review_snapshot(review_authorization.review, current)
        if review_authorization.evidence_snapshot_digest != current.digest:
            raise EvidenceSnapshotError("review digest does not match current evidence")
    _validate_mapping_successor(
        current,
        mapping_authorizations,
        settings,
        inventory_connection=inventory_connection,
        inventory_schema=inventory_schema,
    )
    return current


def validated_evidence_facts(
    snapshot: EvidenceSnapshot,
    review_authorization: ReviewAuthorization | None,
) -> ValidatedEvidenceFacts:
    """Derive the relational facts that final and payment triggers enforce."""

    from invoice_agents.agents.decision_rules import unaddressed_blockers

    review = review_authorization.review if review_authorization is not None else None
    human = review.human_decision if review is not None and review.status == "RESOLVED" else None
    return ValidatedEvidenceFacts(
        policy_review_required=int(bool(snapshot.risk.policy_review_reasons)),
        unresolved_blocker_count=len(unaddressed_blockers(snapshot.risk, human)),
        critique_disposition=str(snapshot.critique.recommended_disposition),
        review_id=review.review_id if review is not None else None,
        review_snapshot_digest=(
            review_authorization.evidence_snapshot_digest
            if review_authorization is not None
            else None
        ),
    )


def _normalized_invoice_field(invoice: dict[str, Any], name: str) -> str | None:
    raw = invoice.get(name)
    if not isinstance(raw, dict):
        return None
    normalized = raw.get("normalized_value")
    return str(normalized) if normalized is not None else None


def _valid_superseded_case_ids(
    connection: sqlite3.Connection,
    review: ReviewRequest,
    requested_case_ids: frozenset[str],
) -> frozenset[str]:
    """Derive exact supersession facts without deciding their applicability."""

    if not requested_case_ids:
        return frozenset()
    candidates = {
        str(raw_candidate["case_id"]): raw_candidate
        for raw_candidate in review.evidence_bundle.get("identity_candidates", [])
        if isinstance(raw_candidate, dict)
        and isinstance(raw_candidate.get("case_id"), str)
        and raw_candidate["case_id"] in requested_case_ids
    }
    raw_invoice = review.evidence_bundle.get("invoice")
    invoice = raw_invoice if isinstance(raw_invoice, dict) else {}
    invoice_number = _normalized_invoice_field(invoice, "invoice_number")
    vendor = _normalized_invoice_field(invoice, "vendor")
    parameters = (review.case_id, *sorted(requested_case_ids))
    rows = connection.execute(
        "SELECT case_id, invoice_number, vendor, started_at FROM cases WHERE case_id IN "
        f"({', '.join('?' for _ in parameters)})",
        parameters,
    ).fetchall()
    cases = {str(row["case_id"]): row for row in rows}
    current = cases.get(review.case_id)
    if invoice_number is None or vendor is None or current is None:
        return frozenset()
    valid: set[str] = set()
    for case_id, candidate in candidates.items():
        prior = cases.get(case_id)
        if (
            case_id != review.case_id
            and candidate.get("relationship") == "POSSIBLE_REVISION"
            and candidate.get("invoice_number") == invoice_number
            and candidate.get("vendor") == vendor
            and prior is not None
            and current["invoice_number"] == invoice_number
            and current["vendor"] == vendor
            and prior["invoice_number"] == invoice_number
            and prior["vendor"] == vendor
            and datetime.fromisoformat(str(prior["started_at"]))
            < datetime.fromisoformat(str(current["started_at"]))
        ):
            valid.add(case_id)
    return frozenset(valid)


def _validate_human_decision_authority(
    connection: sqlite3.Connection,
    review: ReviewRequest,
    decision: HumanDecision,
    *,
    inventory_connection: sqlite3.Connection | None,
    inventory_schema: str,
    require_persisted_mapping_provenance: bool,
) -> tuple[tuple[str, str], ...]:
    """Load explicit authoritative facts and invoke the one pure decision validator."""

    from invoice_agents.agents.decision_rules import validate_human_decision_applicability

    if not inventory_schema.replace("_", "a").isalnum():
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "inventory schema name is invalid",
            case_id=review.case_id,
            stop_reason="HUMAN_MAPPING_PROVENANCE_INVALID",
        )
    normalized_aliases = {
        normalized
        for mapping in decision.mappings
        if (normalized := _normalize_alias(mapping.raw_item))
    }
    requested_skus = {mapping.sku.strip() for mapping in decision.mappings if mapping.sku.strip()}
    inventory_skus: frozenset[str] = frozenset()
    persisted_provenance: dict[str, tuple[str, str, str, str]] | None = None
    if inventory_connection is None and decision.mappings:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "mapping validation requires an authoritative inventory connection",
            case_id=review.case_id,
            stop_reason="HUMAN_MAPPING_PROVENANCE_INVALID",
        )
    if decision.mappings:
        assert inventory_connection is not None
        sku_rows = inventory_connection.execute(
            f'SELECT sku FROM "{inventory_schema}".inventory WHERE sku IN '
            f"({', '.join('?' for _ in requested_skus)})",
            tuple(sorted(requested_skus)),
        ).fetchall()
        inventory_skus = frozenset(str(row["sku"]) for row in sku_rows)
    if require_persisted_mapping_provenance and decision.mappings:
        assert inventory_connection is not None
        source = f"human_review:{review.review_id}"
        alias_clause = (
            f" OR alias_normalized IN ({', '.join('?' for _ in normalized_aliases)})"
            if normalized_aliases
            else ""
        )
        provenance_rows = inventory_connection.execute(
            f"SELECT alias_normalized, sku, source, approved_by, approved_at "
            f'FROM "{inventory_schema}".item_aliases WHERE source = ?{alias_clause}',
            (source, *sorted(normalized_aliases)),
        ).fetchall()
        persisted_provenance = {
            str(row["alias_normalized"]): (
                str(row["sku"]),
                str(row["source"]),
                str(row["approved_by"]),
                str(row["approved_at"]),
            )
            for row in provenance_rows
        }
    elif require_persisted_mapping_provenance:
        persisted_provenance = {}

    superseded = decision.superseded_case_id
    valid_superseded_case_ids = _valid_superseded_case_ids(
        connection,
        review,
        frozenset({superseded}) if superseded is not None else frozenset(),
    )
    return validate_human_decision_applicability(
        review,
        decision,
        inventory_skus=inventory_skus,
        valid_superseded_case_ids=valid_superseded_case_ids,
        persisted_mapping_provenance=persisted_provenance,
    )


def _validate_human_decision(
    connection: sqlite3.Connection, review: ReviewRequest, decision: HumanDecision
) -> list[tuple[str, str]]:
    """Validate every authorizing input against the transaction-local review evidence."""

    return list(
        _validate_human_decision_authority(
            connection,
            review,
            decision,
            inventory_connection=connection,
            inventory_schema="inventory_db",
            require_persisted_mapping_provenance=False,
        )
    )


class WorkflowStore:
    """Own all mutation of the workflow database; inventory remains separate."""

    def __init__(self, path: Path | Settings) -> None:
        if isinstance(path, Settings):
            path.assert_delete_journal_mode()
        self.settings = path if isinstance(path, Settings) else None
        selected_path = path.workflow_db if isinstance(path, Settings) else path
        self.path = selected_path.resolve()

    def _snapshot_settings(self) -> Settings:
        if self.settings is None:
            raise InvoiceAgentsError(
                ErrorCategory.CONFIGURATION,
                "authorization evidence requires explicit inventory and risk-policy settings",
                stop_reason="EVIDENCE_AUTHORITY_MISSING",
            )
        return self.settings

    def require_current_execution_claim(self, claim: ExecutionClaim) -> None:
        """Prove exact, unexpired execution authority without mutating storage."""

        claim = validate_execution_claim(claim)
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)

    def load_authoritative_case_source_id(self, claim: ExecutionClaim) -> str:
        """Validate and return the source identity in the exact claim's read snapshot."""

        claim = validate_execution_claim(claim)
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            try:
                return _authoritative_source_for_case(connection, claim.case_id).source_id
            except EvidenceSnapshotError:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "case source authority is missing or inconsistent",
                    case_id=claim.case_id,
                    stop_reason="PERSISTED_RESULT_INVALID",
                ) from None

    def load_authoritative_case_started_at(self, claim: ExecutionClaim) -> datetime:
        """Return the canonical case start from the exact running claim's snapshot."""

        claim = validate_execution_claim(claim)
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            row = connection.execute(
                "SELECT started_at FROM cases WHERE case_id = ?",
                (claim.case_id,),
            ).fetchone()
        started_at = parse_canonical_utc(row["started_at"] if row is not None else None)
        if started_at is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has a noncanonical persisted start timestamp",
                case_id=claim.case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        return started_at

    @staticmethod
    def _require_terminal_chronology(
        result: CaseResult,
        authoritative_started_at: object,
        *,
        stored_finished_at: object = _STORED_FINISHED_AT_OMITTED,
    ) -> None:
        """Require one exact authoritative, monotonic, canonical UTC terminal clock."""

        authoritative = parse_canonical_utc(authoritative_started_at)
        started_at = _runtime_utc_datetime(result.started_at)
        finished_at = _runtime_utc_datetime(result.finished_at)
        stored_finish_is_required = stored_finished_at is not _STORED_FINISHED_AT_OMITTED
        stored_finish = (
            parse_canonical_utc(stored_finished_at) if stored_finish_is_required else None
        )
        valid = (
            authoritative is not None
            and started_at is not None
            and finished_at is not None
            and started_at == authoritative
            and finished_at >= started_at
            and _canonical_pydantic_datetime_wire(started_at) is not None
            and _canonical_pydantic_datetime_wire(finished_at) is not None
        )
        if stored_finish_is_required:
            valid = valid and stored_finish is not None and finished_at == stored_finish
        if not valid:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "terminal result chronology does not match its authoritative case",
                case_id=result.case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None

    @classmethod
    def _decode_terminal_result_row(cls, row: sqlite3.Row) -> CaseResult:
        """Decode and bind one stored aggregate to the same relational row snapshot."""

        try:
            result = _decode_terminal_result_json(row["result_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has an invalid persisted result",
                case_id=str(row["case_id"]),
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from exc
        if (
            result.case_id != row["case_id"]
            or result.source_id != row["source_id"]
            or str(result.status) != row["status"]
            or result.stop_reason != row["stop_reason"]
        ):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case result does not match its relational authority row",
                case_id=str(row["case_id"]),
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        cls._require_terminal_chronology(
            result,
            row["started_at"],
            stored_finished_at=row["finished_at"],
        )
        return result

    @classmethod
    def _decode_optional_stored_result_row(
        cls,
        row: sqlite3.Row,
        *,
        predecessor: bool,
        require_result: bool = False,
    ) -> CaseResult | None:
        """Validate any aggregate/finish pair before a row can authorize later work."""

        if row["result_json"] is not None:
            return (
                cls._decode_recovery_predecessor_row(row)
                if predecessor
                else cls._decode_terminal_result_row(row)
            )
        if row["finished_at"] is not None or require_result:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has terminal authority without a valid persisted result",
                case_id=str(row["case_id"]),
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        return None

    @classmethod
    def _decode_recovery_predecessor_row(cls, row: sqlite3.Row) -> CaseResult:
        """Decode an older terminal aggregate retained by a resumed authority.

        A resumed/expired row may already have moved its relational status back
        to ``INCOMPLETE`` while retaining the predecessor aggregate.  Recovery
        validates that predecessor's immutable identity and chronology without
        pretending its old status is the current row status.
        """

        try:
            result = _decode_terminal_result_json(row["result_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has an invalid persisted predecessor result",
                case_id=str(row["case_id"]),
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from exc
        if result.case_id != row["case_id"] or result.source_id != row["source_id"]:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case predecessor result does not match its relational identity",
                case_id=str(row["case_id"]),
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        cls._require_terminal_chronology(
            result,
            row["started_at"],
            stored_finished_at=row["finished_at"],
        )
        return result

    @classmethod
    def _encode_terminal_result(
        cls,
        connection: sqlite3.Connection,
        result: CaseResult,
    ) -> str:
        """Validate terminal identity/chronology in the write transaction and encode once."""

        result = sanitize_case_result(result)
        row = connection.execute(
            "SELECT started_at FROM cases WHERE case_id = ?",
            (result.case_id,),
        ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"case does not exist: {result.case_id}",
                case_id=result.case_id,
                stop_reason="CASE_NOT_FOUND",
            ) from None
        cls._require_terminal_chronology(result, row["started_at"])
        encoded = result.model_dump_json()
        try:
            decoded = _decode_terminal_result_json(encoded)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "terminal result cannot be encoded canonically",
                case_id=result.case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from exc
        if decoded != result:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "terminal result changes during canonical encoding",
                case_id=result.case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        return encoded

    def load_recovery_case_source_id(self, claim: ExecutionClaim) -> str:
        """Load source authority for this exact running or finished generation."""

        claim = validate_execution_claim(claim)
        with connect_database(self.path, read_only=True) as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT execution_token, execution_generation, execution_state, "
                "lease_expires_at FROM cases WHERE case_id = ?",
                (claim.case_id,),
            ).fetchone()
            if row is not None and not self._authority_tuple_is_valid(row):
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "case has a contradictory execution authority tuple",
                    case_id=claim.case_id,
                    stop_reason="EXECUTION_AUTHORITY_CORRUPT",
                ) from None
            exact_running = (
                row is not None
                and row["execution_state"] == "RUNNING"
                and row["lease_expires_at"] == claim.expires_at.isoformat()
                and claim.expires_at > datetime.now(UTC)
            )
            exact_finished = (
                row is not None
                and row["execution_state"] == "FINISHED"
                and row["lease_expires_at"] is None
            )
            if (
                row is None
                or row["execution_token"] != claim.token
                or type(row["execution_generation"]) is not int
                or row["execution_generation"] != claim.generation
                or not (exact_running or exact_finished)
            ):
                self._raise_stale_execution_claim(claim)
            try:
                return _authoritative_source_for_case(connection, claim.case_id).source_id
            except EvidenceSnapshotError:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "case source authority is missing or inconsistent",
                    case_id=claim.case_id,
                    stop_reason="PERSISTED_RESULT_INVALID",
                ) from None

    @staticmethod
    def _raise_admission_invalid(message: str, *, case_id: str | None = None) -> Never:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            message,
            case_id=case_id,
            stop_reason="PERSISTED_SUBMISSION_INVALID",
        ) from None

    @classmethod
    def _validate_staged_admission(cls, staged: StagedInvoiceAdmission) -> ExecutionClaim:
        if type(staged) is not StagedInvoiceAdmission:
            raise TypeError("admission requires an exact staged payload")
        claim = validate_execution_claim(
            ExecutionClaim(
                staged.case_id,
                staged.execution_token,
                1,
                staged.lease_expires_at,
            ),
            expected_case_id=staged.case_id,
        )
        encoded_start = staged.started_at.isoformat() if type(staged.started_at) is datetime else ""
        checked_at = datetime.now(UTC)
        if (
            _runtime_utc_datetime(staged.started_at) is None
            or parse_canonical_utc(encoded_start) != staged.started_at
            or staged.started_at > checked_at
            or claim.expires_at <= checked_at
            or claim.expires_at <= staged.started_at
            or not staged.submitted_path.is_absolute()
            or staged.invoice.source != staged.source
        ):
            raise ValueError("staged admission payload is not canonical")
        return claim

    @staticmethod
    def _insert_source_in_transaction(
        connection: sqlite3.Connection,
        source: SourceArtifact,
        created_at: str,
    ) -> None:
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
            if tuple(existing[index] for index in range(6)) != expected:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    f"source artifact {source.source_id} is immutable and conflicts",
                    stop_reason="SOURCE_ARTIFACT_IMMUTABLE",
                )
            return
        connection.execute(
            "INSERT INTO source_artifacts("
            "source_id, canonical_path, source_hash, source_format, size_bytes, "
            "modified_at, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (source.source_id, *expected, created_at),
        )

    @classmethod
    def _stored_case_admission_state(
        cls,
        connection: sqlite3.Connection,
        case_id: str,
        expected_source_id: str,
    ) -> tuple[Literal["running", "done", "failed"], datetime, CaseResult | None]:
        row = connection.execute(
            "SELECT case_id, source_id, status, stop_reason, result_json, started_at, "
            "finished_at, execution_token, execution_generation, execution_state, "
            "lease_expires_at FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None or row["source_id"] != expected_source_id:
            cls._raise_admission_invalid(
                "source run claim does not identify its authoritative case",
                case_id=case_id,
            )
        assert row is not None
        started_at = parse_canonical_utc(row["started_at"])
        if started_at is None or not cls._authority_tuple_is_valid(row):
            cls._raise_admission_invalid(
                "source run claim references invalid case authority",
                case_id=case_id,
            )
        if row["execution_state"] == "RUNNING":
            return "running", started_at, None
        if row["execution_state"] != "FINISHED":
            cls._raise_admission_invalid(
                "source run claim references a non-running nonterminal case",
                case_id=case_id,
            )
        result = cls._decode_terminal_result_row(row)
        state: Literal["done", "failed"] = (
            "failed" if result.status in {CaseStatus.FAILED, CaseStatus.INCOMPLETE} else "done"
        )
        return state, started_at, result

    @classmethod
    def _validated_exact_source_run_claim(
        cls,
        connection: sqlite3.Connection,
        source_id: str,
        case_id: str,
        authoritative_state: Literal["running", "done", "failed"],
        started_at: datetime,
        result: CaseResult | None,
    ) -> Literal["queued", "running", "done", "failed"]:
        """Validate the current source authority before honoring an idempotent target."""

        row = connection.execute(
            "SELECT source_id, case_id, state, claimed_at, released_at "
            "FROM source_run_claims WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None or row["source_id"] != source_id or row["case_id"] != case_id:
            cls._raise_admission_invalid(
                "submission source authority is missing or references another case",
                case_id=case_id,
            )
        assert row is not None
        checked_at = datetime.now(UTC)
        claimed_at = parse_canonical_utc(row["claimed_at"])
        if claimed_at is None or claimed_at < started_at or claimed_at > checked_at:
            cls._raise_admission_invalid(
                "submission source claim time is not canonical",
                case_id=case_id,
            )
        claim_state = str(row["state"])
        if authoritative_state == "running":
            if claim_state not in {"queued", "running"} or row["released_at"] is not None:
                cls._raise_admission_invalid(
                    "active submission source claim is not exact",
                    case_id=case_id,
                )
            return cast(Literal["queued", "running"], claim_state)
        released_at = parse_canonical_utc(row["released_at"])
        if (
            claim_state != authoritative_state
            or result is None
            or released_at is None
            or released_at < claimed_at
            or released_at < result.finished_at
            or released_at > checked_at
        ):
            cls._raise_admission_invalid(
                "terminal submission source claim is not exact",
                case_id=case_id,
            )
        return claim_state

    @classmethod
    def _validated_source_run_claim_for_target(
        cls,
        connection: sqlite3.Connection,
        source_id: str,
        target_case_id: str,
        target_state: Literal["running", "done", "failed"],
        target_started_at: datetime,
        target_result: CaseResult | None,
    ) -> tuple[Literal["queued", "running", "done", "failed"], bool]:
        """Validate a current or legitimately superseded submission target.

        The boolean is true only for the explicit Task 9 orphan-recovery state:
        the case is terminal but its still-exact active admission mirror has not
        yet been advanced.  Callers with write authority may reconcile that one
        state; every other disagreement is corruption.
        """

        row = connection.execute(
            "SELECT case_id, state FROM source_run_claims WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            cls._raise_admission_invalid(
                "submission source authority is missing",
                case_id=target_case_id,
            )
        assert row is not None
        current_case_id = str(row["case_id"])
        if current_case_id == target_case_id:
            if (
                target_state in {"done", "failed"}
                and row["state"] in {"queued", "running"}
                and target_result is not None
                and target_result.stop_reason == "ORPHANED_EXECUTION"
            ):
                active_state = cls._validated_exact_source_run_claim(
                    connection,
                    source_id,
                    target_case_id,
                    "running",
                    target_started_at,
                    None,
                )
                return active_state, True
            return (
                cls._validated_exact_source_run_claim(
                    connection,
                    source_id,
                    target_case_id,
                    target_state,
                    target_started_at,
                    target_result,
                ),
                False,
            )

        if target_state not in {"done", "failed"} or target_result is None:
            cls._raise_admission_invalid(
                "active submission target was superseded",
                case_id=target_case_id,
            )
        current_state, current_started_at, current_result = cls._stored_case_admission_state(
            connection,
            current_case_id,
            source_id,
        )
        if current_started_at < target_result.finished_at:
            cls._raise_admission_invalid(
                "source run authority does not follow its historical submission target",
                case_id=target_case_id,
            )
        cls._validated_exact_source_run_claim(
            connection,
            source_id,
            current_case_id,
            current_state,
            current_started_at,
            current_result,
        )
        return target_state, False

    @classmethod
    def _reconcile_recovered_source_admission(
        cls,
        connection: sqlite3.Connection,
        source_id: str,
        case_id: str,
        state: Literal["done", "failed"],
        started_at: datetime,
        result: CaseResult,
    ) -> None:
        """Mirror one exact Task 9 orphan terminal inside admission ownership."""

        released_at = now_iso()
        released_at_clock = parse_canonical_utc(released_at)
        if released_at_clock is None or released_at_clock < result.finished_at:
            cls._raise_admission_invalid(
                "recovered source admission timestamp is not monotonic",
                case_id=case_id,
            )
        claim = connection.execute(
            "SELECT state FROM source_run_claims WHERE source_id = ? AND case_id = ?",
            (source_id, case_id),
        ).fetchone()
        if claim is None or claim["state"] not in {"queued", "running"}:
            cls._raise_admission_invalid(
                "recovered source admission is not exact",
                case_id=case_id,
            )
        batch_ids = cls._require_exact_case_batch_mirrors(
            connection,
            case_id,
            str(claim["state"]),
        )
        updated_claim = connection.execute(
            "UPDATE source_run_claims SET state = ?, released_at = ? "
            "WHERE source_id = ? AND case_id = ? "
            "AND state IN ('queued', 'running') AND released_at IS NULL",
            (state, released_at, source_id, case_id),
        )
        updated_entries = connection.execute(
            "UPDATE batch_entries SET state = ? WHERE case_id = ? "
            "AND state IN ('queued', 'running')",
            (state, case_id),
        )
        if updated_claim.rowcount != 1 or updated_entries.rowcount != len(batch_ids):
            cls._raise_admission_invalid(
                "recovered admission transition was not exact",
                case_id=case_id,
            )
        for batch_id in batch_ids:
            cls._recompute_batch_state(connection, batch_id)
        cls._validated_exact_source_run_claim(
            connection,
            source_id,
            case_id,
            state,
            started_at,
            result,
        )

    @staticmethod
    def _derived_batch_state(
        states: tuple[str, ...],
    ) -> Literal["running", "done", "failed"]:
        if not states or any(
            state not in {"queued", "running", "done", "failed"} for state in states
        ):
            WorkflowStore._raise_admission_invalid("durable batch has invalid entries")
        return (
            "running"
            if any(state in {"queued", "running"} for state in states)
            else "failed"
            if any(state == "failed" for state in states)
            else "done"
        )

    @classmethod
    def _require_exact_case_batch_mirrors(
        cls,
        connection: sqlite3.Connection,
        case_id: str,
        expected_entry_state: str,
    ) -> tuple[str, ...]:
        """Prove every batch mirror before an authorized state transition."""

        rows = connection.execute(
            "SELECT batch_id, state FROM batch_entries WHERE case_id = ? ORDER BY batch_id",
            (case_id,),
        ).fetchall()
        batch_ids = tuple(str(row["batch_id"]) for row in rows)
        if len(set(batch_ids)) != len(batch_ids) or any(
            row["state"] != expected_entry_state for row in rows
        ):
            cls._raise_admission_invalid(
                "batch entry disagrees with its source admission",
                case_id=case_id,
            )
        for batch_id in batch_ids:
            batch_row = connection.execute(
                "SELECT state FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            states = tuple(
                str(row["state"])
                for row in connection.execute(
                    "SELECT state FROM batch_entries WHERE batch_id = ? ORDER BY position",
                    (batch_id,),
                ).fetchall()
            )
            if batch_row is None or batch_row["state"] != cls._derived_batch_state(states):
                cls._raise_admission_invalid(
                    "batch state disagrees with its durable entries",
                    case_id=case_id,
                )
        return batch_ids

    @classmethod
    def _load_existing_submission(
        cls,
        connection: sqlite3.Connection,
        request_id: str,
        kind: Literal["single", "batch"],
        fingerprint: str,
        source_ids: tuple[str, ...],
    ) -> SubmissionAdmission | None:
        row = connection.execute(
            "SELECT created_at, kind, fingerprint, redirect_target FROM submission_requests "
            "WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        if row["kind"] != kind or row["fingerprint"] != fingerprint:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "submission ID is already bound to a different request",
                stop_reason="SUBMISSION_FINGERPRINT_MISMATCH",
            ) from None
        target = str(row["redirect_target"])
        created_at = parse_canonical_utc(row["created_at"])
        if created_at is None:
            cls._raise_admission_invalid("submission creation time is not canonical")
        admitted: list[AdmittedCase] = []
        batch_id: str | None = None
        if kind == "single":
            match = _CASE_LIVE_TARGET.fullmatch(target)
            if match is None or len(source_ids) != 1:
                cls._raise_admission_invalid("single submission redirect is invalid")
            case_id = match.group(1)
            state, started_at, result = cls._stored_case_admission_state(
                connection, case_id, source_ids[0]
            )
            admission_state, _recovery_required = cls._validated_source_run_claim_for_target(
                connection,
                source_ids[0],
                case_id,
                state,
                started_at,
                result,
            )
            admitted.append(
                AdmittedCase(
                    source_id=source_ids[0],
                    source_path=Path("."),
                    case_id=case_id,
                    started_at=started_at,
                    state=admission_state,
                    claim=None,
                )
            )
        else:
            match = _BATCH_TARGET.fullmatch(target)
            if match is None:
                cls._raise_admission_invalid("batch submission redirect is invalid")
            batch_id = match.group(1)
            batch_row = connection.execute(
                "SELECT batch_id FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            entry_rows = connection.execute(
                "SELECT position, source_id, case_id, source_path FROM batch_entries "
                "WHERE batch_id = ? ORDER BY position",
                (batch_id,),
            ).fetchall()
            if (
                batch_row is None
                or tuple(item["position"] for item in entry_rows) != tuple(range(len(entry_rows)))
                or tuple(str(item["source_id"]) for item in entry_rows) != source_ids
            ):
                cls._raise_admission_invalid("batch submission target is inconsistent")
            for entry in entry_rows:
                source_id = str(entry["source_id"])
                case_id = str(entry["case_id"])
                state, started_at, result = cls._stored_case_admission_state(
                    connection, case_id, source_id
                )
                admission_state, _recovery_required = cls._validated_source_run_claim_for_target(
                    connection,
                    source_id,
                    case_id,
                    state,
                    started_at,
                    result,
                )
                admitted.append(
                    AdmittedCase(
                        source_id=source_id,
                        source_path=Path(str(entry["source_path"])),
                        case_id=case_id,
                        started_at=started_at,
                        state=admission_state,
                        claim=None,
                    )
                )
        return SubmissionAdmission(
            request_id=request_id,
            created_at=created_at,
            kind=kind,
            fingerprint=fingerprint,
            redirect_target=target,
            cases=tuple(admitted),
            batch_id=batch_id,
        )

    def load_submission(
        self,
        request_id: str,
        kind: Literal["single", "batch"],
        source_ids: tuple[str, ...],
    ) -> SubmissionAdmission | None:
        """Read and strictly validate an already-persisted request without mutation."""

        if _SUBMISSION_REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("submission request ID is not canonical")
        fingerprint = submission_fingerprint(kind, source_ids)
        with connect_database(self.path, read_only=True) as connection:
            return self._load_existing_submission(
                connection, request_id, kind, fingerprint, source_ids
            )

    def claim_source_run(
        self,
        connection: sqlite3.Connection,
        staged: StagedInvoiceAdmission,
        *,
        claimed_at: str,
        force_reprocess: bool,
    ) -> AdmittedCase:
        """Claim one source inside the caller-owned admission transaction."""

        if not connection.in_transaction:
            raise RuntimeError("source run claims require an active admission transaction")
        claim = self._validate_staged_admission(staged)
        if parse_canonical_utc(claimed_at) is None:
            raise ValueError("source claim timestamp is not canonical UTC")
        self._insert_source_in_transaction(connection, staged.source, claimed_at)
        prior = connection.execute(
            "SELECT source_id, case_id, state, claimed_at, released_at "
            "FROM source_run_claims WHERE source_id = ?",
            (staged.source.source_id,),
        ).fetchone()
        if prior is not None:
            prior_case_id = str(prior["case_id"])
            authoritative_state, prior_started_at, prior_result = self._stored_case_admission_state(
                connection, prior_case_id, staged.source.source_id
            )
            prior_state, recovery_required = self._validated_source_run_claim_for_target(
                connection,
                staged.source.source_id,
                prior_case_id,
                authoritative_state,
                prior_started_at,
                prior_result,
            )
            if recovery_required:
                if authoritative_state == "done":
                    recovered_state: Literal["done", "failed"] = "done"
                elif authoritative_state == "failed":
                    recovered_state = "failed"
                else:
                    self._raise_admission_invalid(
                        "recovered source admission lacks exact terminal evidence",
                        case_id=prior_case_id,
                    )
                if prior_result is None:
                    self._raise_admission_invalid(
                        "recovered source admission lacks its terminal result",
                        case_id=prior_case_id,
                    )
                self._reconcile_recovered_source_admission(
                    connection,
                    staged.source.source_id,
                    prior_case_id,
                    recovered_state,
                    prior_started_at,
                    prior_result,
                )
                prior_state = recovered_state
        if prior is not None and not force_reprocess:
            return AdmittedCase(
                source_id=staged.source.source_id,
                source_path=staged.submitted_path,
                case_id=prior_case_id,
                started_at=prior_started_at,
                state=prior_state,
                claim=None,
            )
        if prior is not None and prior_state not in {"done", "failed"}:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "source run has not reached a durable terminal state",
                case_id=prior_case_id,
                stop_reason="SOURCE_RUN_NOT_TERMINAL",
            ) from None
        connection.execute(
            "INSERT INTO cases("
            "case_id, source_id, status, stop_reason, started_at, updated_at, "
            "execution_token, execution_generation, execution_state, lease_expires_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RUNNING', ?)",
            (
                staged.case_id,
                staged.source.source_id,
                CaseStatus.INCOMPLETE,
                "CASE_CREATED",
                staged.started_at.isoformat(),
                claimed_at,
                claim.token,
                claim.generation,
                claim.expires_at.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO extractions("
            "extraction_id, case_id, version, payload_json, created_at, "
            "execution_generation) VALUES (?, ?, 1, ?, ?, ?)",
            (
                f"ext_{uuid4().hex}",
                staged.case_id,
                staged.invoice.model_dump_json(),
                claimed_at,
                claim.generation,
            ),
        )
        connection.execute(
            "INSERT INTO events(event_id, case_id, source_id, event_type, "
            "payload_json, created_at) VALUES (?, ?, ?, 'case.prepared', ?, ?)",
            (
                f"evt_{uuid4().hex}",
                staged.case_id,
                staged.source.source_id,
                json.dumps(
                    {
                        "source": staged.source.model_dump(mode="json"),
                        "extraction_version": 1,
                        "note": ("pre-model extraction enables complete batch identity visibility"),
                    },
                    default=str,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                claimed_at,
            ),
        )
        if prior is None:
            connection.execute(
                "INSERT INTO source_run_claims("
                "source_id, case_id, state, claimed_at, released_at"
                ") VALUES (?, ?, 'queued', ?, NULL)",
                (staged.source.source_id, staged.case_id, claimed_at),
            )
        else:
            updated = connection.execute(
                "UPDATE source_run_claims SET case_id = ?, state = 'queued', "
                "claimed_at = ?, released_at = NULL WHERE source_id = ? "
                "AND case_id = ? AND state IN ('done', 'failed')",
                (
                    staged.case_id,
                    claimed_at,
                    staged.source.source_id,
                    prior["case_id"],
                ),
            )
            if updated.rowcount != 1:
                raise InvoiceAgentsError(
                    ErrorCategory.ORCHESTRATION,
                    "source run authority changed during forced admission",
                    case_id=staged.case_id,
                    stop_reason="SOURCE_RUN_NOT_TERMINAL",
                )
        return AdmittedCase(
            source_id=staged.source.source_id,
            source_path=staged.submitted_path,
            case_id=staged.case_id,
            started_at=staged.started_at,
            state="queued",
            claim=claim,
        )

    def claim_submission(
        self,
        request_id: str,
        kind: Literal["single", "batch"],
        staged_sources: tuple[StagedInvoiceAdmission, ...],
        *,
        concurrency: int | None = None,
        force_reprocess: bool = False,
    ) -> SubmissionAdmission:
        """Atomically bind source, execution, idempotency, and redirect authority."""

        if _SUBMISSION_REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("submission request ID is not canonical")
        if kind not in {"single", "batch"} or not staged_sources:
            raise ValueError("admission kind and staged sources are required")
        if kind == "single" and len(staged_sources) != 1:
            raise ValueError("single admission requires exactly one source")
        if type(force_reprocess) is not bool:
            raise ValueError("force_reprocess must be an exact boolean")
        if kind == "batch" and (type(concurrency) is not int or not 1 <= concurrency <= 8):
            raise ValueError("batch admission concurrency must be between 1 and 8")
        if kind == "batch" and force_reprocess:
            raise ValueError("batch admission cannot force source reprocessing")
        source_ids = tuple(item.source.source_id for item in staged_sources)
        if len(set(source_ids)) != len(source_ids):
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "one batch cannot contain the same immutable source more than once",
                stop_reason="DUPLICATE_BATCH_SOURCE",
            ) from None
        fingerprint = submission_fingerprint(kind, source_ids)
        created_at_clock = datetime.now(UTC)
        created_at = created_at_clock.isoformat()
        batch_id = f"batch_{uuid4().hex[:24]}" if kind == "batch" else None
        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = self._load_existing_submission(
                    connection, request_id, kind, fingerprint, source_ids
                )
                if existing is not None:
                    connection.commit()
                    return existing
                admitted = [
                    self.claim_source_run(
                        connection,
                        staged,
                        claimed_at=created_at,
                        force_reprocess=force_reprocess,
                    )
                    for staged in staged_sources
                ]
                if kind == "single":
                    redirect_target = f"/cases/{admitted[0].case_id}/live"
                    connection.execute(
                        "INSERT INTO submission_requests("
                        "request_id, created_at, kind, fingerprint, redirect_target"
                        ") VALUES (?, ?, 'single', ?, ?)",
                        (request_id, created_at, fingerprint, redirect_target),
                    )
                else:
                    assert batch_id is not None and concurrency is not None
                    redirect_target = f"/batches/{batch_id}"
                    connection.execute(
                        "INSERT INTO submission_requests("
                        "request_id, created_at, kind, fingerprint, redirect_target"
                        ") VALUES (?, ?, 'batch', ?, ?)",
                        (request_id, created_at, fingerprint, redirect_target),
                    )
                    initial_batch_state = (
                        "done"
                        if all(item.state == "done" for item in admitted)
                        else "failed"
                        if any(item.state == "failed" for item in admitted)
                        and all(item.state in {"done", "failed"} for item in admitted)
                        else "running"
                    )
                    connection.execute(
                        "INSERT INTO batches(batch_id, created_at, concurrency, state) "
                        "VALUES (?, ?, ?, ?)",
                        (batch_id, created_at, concurrency, initial_batch_state),
                    )
                    connection.executemany(
                        "INSERT INTO batch_entries("
                        "batch_id, position, source_id, case_id, source_path, state"
                        ") VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            (
                                batch_id,
                                position,
                                item.source_id,
                                item.case_id,
                                str(item.source_path),
                                item.state,
                            )
                            for position, item in enumerate(admitted)
                        ),
                    )
                result = SubmissionAdmission(
                    request_id=request_id,
                    created_at=created_at_clock,
                    kind=kind,
                    fingerprint=fingerprint,
                    redirect_target=redirect_target,
                    cases=tuple(admitted),
                    batch_id=batch_id,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return result

    def mark_admission_running(self, case_id: str) -> None:
        """Persist the queued-to-running transition before entering model code."""

        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    "SELECT source_id, case_id, state FROM source_run_claims WHERE case_id = ?",
                    (case_id,),
                ).fetchall()
                if len(rows) != 1:
                    self._raise_admission_invalid(
                        "running case does not have exactly one durable source admission",
                        case_id=case_id,
                    )
                row = rows[0]
                authoritative_state, started_at, result = self._stored_case_admission_state(
                    connection,
                    case_id,
                    str(row["source_id"]),
                )
                if authoritative_state != "running" or result is not None:
                    self._raise_admission_invalid(
                        "admission cannot enter running after its case is terminal",
                        case_id=case_id,
                    )
                self._validated_exact_source_run_claim(
                    connection,
                    str(row["source_id"]),
                    case_id,
                    authoritative_state,
                    started_at,
                    result,
                )
                batch_ids = self._require_exact_case_batch_mirrors(
                    connection,
                    case_id,
                    str(row["state"]),
                )
                if row["state"] == "queued":
                    updated = connection.execute(
                        "UPDATE source_run_claims SET state = 'running' "
                        "WHERE source_id = ? AND case_id = ? AND state = 'queued'",
                        (row["source_id"], case_id),
                    )
                    if updated.rowcount != 1:
                        self._raise_admission_invalid(
                            "source admission running transition was not exact",
                            case_id=case_id,
                        )
                elif row["state"] != "running":
                    self._raise_admission_invalid(
                        "terminal source admission cannot re-enter running", case_id=case_id
                    )
                updated_entries = connection.execute(
                    "UPDATE batch_entries SET state = 'running' "
                    "WHERE case_id = ? AND state = 'queued'",
                    (case_id,),
                )
                expected_updates = len(batch_ids) if row["state"] == "queued" else 0
                if updated_entries.rowcount != expected_updates:
                    self._raise_admission_invalid(
                        "batch admission running transition was not exact",
                        case_id=case_id,
                    )
                for batch_id in batch_ids:
                    self._recompute_batch_state(connection, batch_id)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def mark_admission_terminal(self, case_id: str) -> None:
        """Mirror an exact terminal case into source and batch admission state."""

        released_at = now_iso()
        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                source_rows = connection.execute(
                    "SELECT source_id FROM source_run_claims WHERE case_id = ?", (case_id,)
                ).fetchall()
                if len(source_rows) != 1:
                    self._raise_admission_invalid(
                        "terminal case does not have exactly one durable source admission",
                        case_id=case_id,
                    )
                source_row = source_rows[0]
                state, started_at, result = self._stored_case_admission_state(
                    connection, case_id, str(source_row["source_id"])
                )
                released_at_clock = parse_canonical_utc(released_at)
                if (
                    state not in {"done", "failed"}
                    or result is None
                    or released_at_clock is None
                    or released_at_clock < result.finished_at
                ):
                    self._raise_admission_invalid(
                        "admission cannot terminalize before its case", case_id=case_id
                    )
                current = connection.execute(
                    "SELECT state FROM source_run_claims WHERE source_id = ? AND case_id = ?",
                    (source_row["source_id"], case_id),
                ).fetchone()
                if current is None:
                    self._raise_admission_invalid(
                        "terminal case source admission disappeared",
                        case_id=case_id,
                    )
                current_is_active = current["state"] in {"queued", "running"}
                if current_is_active:
                    self._validated_exact_source_run_claim(
                        connection,
                        str(source_row["source_id"]),
                        case_id,
                        "running",
                        started_at,
                        None,
                    )
                else:
                    self._validated_exact_source_run_claim(
                        connection,
                        str(source_row["source_id"]),
                        case_id,
                        state,
                        started_at,
                        result,
                    )
                batch_ids = self._require_exact_case_batch_mirrors(
                    connection,
                    case_id,
                    str(current["state"]),
                )
                updated = connection.execute(
                    "UPDATE source_run_claims SET state = ?, released_at = ? "
                    "WHERE source_id = ? AND case_id = ? AND state IN ('queued', 'running')",
                    (state, released_at, source_row["source_id"], case_id),
                )
                if updated.rowcount != (1 if current_is_active else 0):
                    self._raise_admission_invalid(
                        "source admission terminal transition was not exact",
                        case_id=case_id,
                    )
                updated_entries = connection.execute(
                    "UPDATE batch_entries SET state = ? WHERE case_id = ? "
                    "AND state IN ('queued', 'running')",
                    (state, case_id),
                )
                expected_entry_updates = len(batch_ids) if current_is_active else 0
                if updated_entries.rowcount != expected_entry_updates:
                    self._raise_admission_invalid(
                        "batch admission terminal transition was not exact",
                        case_id=case_id,
                    )
                self._validated_exact_source_run_claim(
                    connection,
                    str(source_row["source_id"]),
                    case_id,
                    state,
                    started_at,
                    result,
                )
                for selected_batch_id in batch_ids:
                    self._recompute_batch_state(connection, selected_batch_id)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _recompute_batch_state(
        connection: sqlite3.Connection,
        batch_id: str,
    ) -> Literal["running", "done", "failed"]:
        states = tuple(
            str(row["state"])
            for row in connection.execute(
                "SELECT state FROM batch_entries WHERE batch_id = ? ORDER BY position",
                (batch_id,),
            ).fetchall()
        )
        state = WorkflowStore._derived_batch_state(states)
        updated = connection.execute(
            "UPDATE batches SET state = ? WHERE batch_id = ?",
            (state, batch_id),
        )
        if updated.rowcount != 1:
            WorkflowStore._raise_admission_invalid("durable batch row is missing")
        return state

    def load_batch(self, batch_id: str) -> PersistedBatch | None:
        """Reconcile one batch from terminal case authority and return its durable view."""

        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                batch_row = connection.execute(
                    "SELECT batch_id, created_at, concurrency, state FROM batches "
                    "WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                if batch_row is None:
                    connection.commit()
                    return None
                if (
                    type(batch_row["concurrency"]) is not int
                    or not 1 <= batch_row["concurrency"] <= 8
                    or batch_row["state"] not in {"queued", "running", "done", "failed"}
                ):
                    self._raise_admission_invalid("durable batch row is invalid")
                rows = connection.execute(
                    "SELECT position, source_id, case_id, source_path, state "
                    "FROM batch_entries WHERE batch_id = ? ORDER BY position",
                    (batch_id,),
                ).fetchall()
                entries: list[PersistedBatchEntry] = []
                reconciled_terminal_entry = False
                for row in rows:
                    source_id = str(row["source_id"])
                    case_id = str(row["case_id"])
                    source_claim = connection.execute(
                        "SELECT case_id, state FROM source_run_claims WHERE source_id = ?",
                        (source_id,),
                    ).fetchone()
                    if source_claim is None:
                        self._raise_admission_invalid(
                            "batch entry has no source run claim", case_id=case_id
                        )
                    state, started_at, result = self._stored_case_admission_state(
                        connection, case_id, source_id
                    )
                    claim_state, recovery_required = self._validated_source_run_claim_for_target(
                        connection,
                        source_id,
                        case_id,
                        state,
                        started_at,
                        result,
                    )
                    entry_state: Literal["queued", "running", "done", "failed"] = claim_state
                    if recovery_required:
                        if source_claim["case_id"] != case_id or result is None:
                            self._raise_admission_invalid(
                                "recovered batch and source run states disagree",
                                case_id=case_id,
                            )
                        if state == "done":
                            recovered_state: Literal["done", "failed"] = "done"
                        elif state == "failed":
                            recovered_state = "failed"
                        else:
                            self._raise_admission_invalid(
                                "recovered batch lacks exact terminal evidence",
                                case_id=case_id,
                            )
                        self._reconcile_recovered_source_admission(
                            connection,
                            source_id,
                            case_id,
                            recovered_state,
                            started_at,
                            result,
                        )
                        entry_state = recovered_state
                        reconciled_terminal_entry = True
                    elif row["state"] != claim_state:
                        self._raise_admission_invalid(
                            "batch entry disagrees with its exact source run claim",
                            case_id=case_id,
                        )
                    entries.append(
                        PersistedBatchEntry(
                            source_id=source_id,
                            source_path=Path(str(row["source_path"])),
                            case_id=case_id,
                            started_at=started_at,
                            state=entry_state,
                            result=result,
                        )
                    )
                batch_state = self._recompute_batch_state(connection, batch_id)
                if batch_row["state"] != batch_state and not (
                    reconciled_terminal_entry and batch_row["state"] in {"queued", "running"}
                ):
                    self._raise_admission_invalid("batch state disagrees with its durable entries")
                created_at = parse_canonical_utc(batch_row["created_at"])
                if created_at is None:
                    self._raise_admission_invalid("batch creation time is not canonical")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return PersistedBatch(
            batch_id=batch_id,
            created_at=created_at,
            concurrency=batch_row["concurrency"],
            state=batch_state,
            entries=tuple(entries),
        )

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
        encoded_start = started_at.isoformat() if type(started_at) is datetime else ""
        if (
            _runtime_utc_datetime(started_at) is None
            or parse_canonical_utc(encoded_start) != started_at
        ):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case start timestamp must be canonical UTC",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        with connect_database(self.path) as connection:
            connection.execute(
                "INSERT INTO cases(case_id, source_id, status, stop_reason, started_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    source.source_id,
                    CaseStatus.INCOMPLETE,
                    "CASE_CREATED",
                    encoded_start,
                    now_iso(),
                ),
            )
            connection.commit()

    def claim_case_execution(
        self,
        case_id: str,
        expected_statuses: frozenset[CaseStatus],
        lease_seconds: int,
        *,
        requested_token: str | None = None,
    ) -> ExecutionClaim:
        """Atomically claim one fresh or resumable case and return its fencing token."""

        if not expected_statuses:
            raise ValueError("expected_statuses must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        token = (
            f"exec_{uuid4().hex}"
            if requested_token is None
            else _validated_requested_execution_token(requested_token)
        )
        statuses = tuple(str(status) for status in sorted(expected_statuses))
        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                claimed_at = datetime.now(UTC)
                expires_at = claimed_at + timedelta(seconds=lease_seconds)
                row = connection.execute(
                    "SELECT case_id, source_id, status, stop_reason, result_json, started_at, "
                    "finished_at, execution_token, execution_generation, execution_state, "
                    "lease_expires_at FROM cases WHERE case_id = ?",
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
                self._decode_optional_stored_result_row(
                    row,
                    predecessor=previous_state != "FINISHED",
                    require_result=previous_state == "FINISHED",
                )
                generation = previous_generation + 1
                connection.execute(
                    "DELETE FROM result_artifact_bindings WHERE case_id = ?",
                    (case_id,),
                )
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

        claim = validate_execution_claim(claim)
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

        claim = validate_execution_claim(claim)
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            released_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, released_at)
            connection.execute(
                "DELETE FROM result_artifact_bindings WHERE case_id = ?",
                (claim.case_id,),
            )
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

    def handoff_case_execution(
        self,
        claim: ExecutionClaim,
        lease_seconds: int,
        *,
        requested_token: str | None = None,
    ) -> ExecutionClaim:
        """Atomically replace preparation authority with the claim used by its run."""

        claim = validate_execution_claim(claim)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        token = (
            f"exec_{uuid4().hex}"
            if requested_token is None
            else _validated_requested_execution_token(requested_token)
        )
        if token == claim.token:
            raise ValueError("handoff execution token must be a distinct authority")
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            handed_off_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, handed_off_at)
            generation = claim.generation + 1
            expires_at = handed_off_at + timedelta(seconds=lease_seconds)
            connection.execute(
                "DELETE FROM result_artifact_bindings WHERE case_id = ?",
                (claim.case_id,),
            )
            updated = connection.execute(
                "UPDATE cases SET execution_token = ?, execution_generation = ?, "
                "lease_expires_at = ?, updated_at = ? WHERE case_id = ? "
                "AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?",
                (
                    token,
                    generation,
                    expires_at.isoformat(),
                    handed_off_at.isoformat(),
                    claim.case_id,
                    claim.token,
                    claim.generation,
                    handed_off_at.isoformat(),
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                self._raise_stale_execution_claim(claim)
            connection.commit()
        return ExecutionClaim(claim.case_id, token, generation, expires_at)

    def has_valid_execution_lease(
        self, case_id: str, *, checked_at: datetime | None = None
    ) -> bool:
        """Return whether storage proves a current, nonexpired RUNNING authority."""

        observed_at = checked_at or datetime.now(UTC)
        self._require_canonical_utc_clock(observed_at)
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT execution_token, execution_generation, execution_state, "
                "lease_expires_at FROM cases WHERE case_id = ?",
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
        lease = parse_canonical_utc(row["lease_expires_at"])
        return row["execution_state"] == "RUNNING" and lease is not None and lease > observed_at

    def load_case_execution_snapshot(
        self,
        case_id: str,
        *,
        checked_at: datetime | None = None,
    ) -> CaseExecutionSnapshot | None:
        """Read result identity and execution authority from one SQLite row snapshot."""

        if checked_at is not None:
            self._require_canonical_utc_clock(checked_at)
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT case_id, source_id, status, stop_reason, result_json, started_at, "
                "finished_at, "
                "execution_token, execution_generation, execution_state, lease_expires_at "
                "FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            observed_at = checked_at or datetime.now(UTC)
        if row is None:
            return None
        if not self._authority_tuple_is_valid(row):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"case {case_id} has a contradictory execution authority tuple",
                case_id=case_id,
                stop_reason="EXECUTION_AUTHORITY_CORRUPT",
            )
        started_at = parse_canonical_utc(row["started_at"])
        if started_at is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has a noncanonical persisted start timestamp",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            )
        result: CaseResult | None = None
        if row["result_json"] is not None:
            result = self._decode_terminal_result_row(row)
        elif row["finished_at"] is not None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has a terminal timestamp without a persisted result",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        lease = parse_canonical_utc(row["lease_expires_at"])
        execution_state = str(row["execution_state"])
        return CaseExecutionSnapshot(
            started_at=started_at,
            result=result,
            execution_state=execution_state,
            has_valid_lease=(
                execution_state == "RUNNING" and lease is not None and lease > observed_at
            ),
            execution_token=cast(str | None, row["execution_token"]),
            execution_generation=int(row["execution_generation"]),
            lease_expires_at=lease,
        )

    def recover_expired_executions(
        self,
        *,
        now: datetime | None = None,
        case_id: str | None = None,
    ) -> list[str]:
        """Atomically terminalize expired RUNNING authorities in case-ID order.

        Recovery advances the execution generation and replaces the abandoned token.
        A worker holding the prior claim therefore cannot overwrite the recovered result.
        The case update and recovery audit event commit in the same SQLite transaction.
        IDLE is intentionally excluded: it has no lease and can be the committed gap
        between case creation and claim acquisition or a deliberately released case.
        """

        recovered_at = now or datetime.now(UTC)
        self._require_canonical_utc_clock(recovered_at)
        if not self.path.is_file():
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"workflow database does not exist: {self.path}",
                stop_reason="DATABASE_MISSING",
            )
        recovered: list[str] = []
        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                sql = (
                    "SELECT case_id, source_id, status, stop_reason, result_json, started_at, "
                    "finished_at, "
                    "execution_token, execution_generation, execution_state, lease_expires_at "
                    "FROM cases WHERE execution_state = 'RUNNING' "
                    "AND status IN ('INCOMPLETE', 'NEEDS_HUMAN')"
                )
                params: tuple[str, ...] = ()
                if case_id is not None:
                    sql += " AND case_id = ?"
                    params = (case_id,)
                sql += " ORDER BY case_id"
                rows = connection.execute(sql, params).fetchall()
                for row in rows:
                    if not self._authority_tuple_is_valid(row):
                        raise InvoiceAgentsError(
                            ErrorCategory.DATABASE,
                            f"case {row['case_id']} has a contradictory execution authority tuple",
                            case_id=str(row["case_id"]),
                            stop_reason="EXECUTION_AUTHORITY_CORRUPT",
                        )
                    lease = parse_canonical_utc(row["lease_expires_at"])
                    if lease is None or lease > recovered_at:
                        continue
                    started_at = parse_canonical_utc(row["started_at"])
                    if started_at is None:
                        raise InvoiceAgentsError(
                            ErrorCategory.DATABASE,
                            "case has a noncanonical persisted start timestamp",
                            case_id=str(row["case_id"]),
                            stop_reason="PERSISTED_RESULT_INVALID",
                        )
                    predecessor_finished_at = parse_canonical_utc(row["finished_at"])
                    if row["finished_at"] is not None and (
                        predecessor_finished_at is None or predecessor_finished_at < started_at
                    ):
                        raise InvoiceAgentsError(
                            ErrorCategory.DATABASE,
                            "case has an invalid predecessor terminal timestamp",
                            case_id=str(row["case_id"]),
                            stop_reason="PERSISTED_RESULT_INVALID",
                        ) from None
                    previous: CaseResult | None = None
                    if row["result_json"] is not None:
                        previous = self._decode_recovery_predecessor_row(row)
                    (
                        recovered_final,
                        recovered_review,
                        recovered_payment,
                    ) = self._load_relational_recovery_evidence(
                        connection,
                        str(row["case_id"]),
                        cast(str | None, row["source_id"]),
                        previous.payment if previous is not None else None,
                    )
                    if previous is not None:
                        self._require_relational_terminal_facts(
                            previous,
                            recovered_final,
                            recovered_review,
                            recovered_payment,
                        )
                    recovery_error = ErrorRecord(
                        category=ErrorCategory.ORCHESTRATION,
                        message="execution lease expired before a terminal result was recorded",
                        case_id=str(row["case_id"]),
                        stop_reason="ORPHANED_EXECUTION",
                        details={
                            "abandoned_execution_generation": int(row["execution_generation"])
                        },
                    )
                    if previous is None:
                        result = CaseResult(
                            case_id=str(row["case_id"]),
                            source_id=row["source_id"],
                            status=CaseStatus.INCOMPLETE,
                            stop_reason="ORPHANED_EXECUTION",
                            final_decision=recovered_final,
                            review_request=recovered_review,
                            payment=recovered_payment,
                            errors=[recovery_error],
                            started_at=started_at,
                            finished_at=recovered_at,
                        )
                    else:
                        result = previous.model_copy(
                            update={
                                "status": CaseStatus.INCOMPLETE,
                                "stop_reason": "ORPHANED_EXECUTION",
                                "final_decision": recovered_final,
                                "review_request": recovered_review,
                                "payment": recovered_payment,
                                "errors": [*previous.errors, recovery_error],
                                "finished_at": recovered_at,
                            },
                            deep=True,
                        )
                    self._require_terminal_chronology(result, row["started_at"])
                    result = sanitize_case_result(result)
                    encoded_result = result.model_dump_json()
                    try:
                        if _decode_terminal_result_json(encoded_result) != result:
                            raise ValueError("recovery result changed during encoding")
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise InvoiceAgentsError(
                            ErrorCategory.DATABASE,
                            "recovery result chronology is not canonical",
                            case_id=str(row["case_id"]),
                            stop_reason="PERSISTED_RESULT_INVALID",
                        ) from exc
                    recovery_generation = int(row["execution_generation"]) + 1
                    recovery_token = f"exec_{uuid4().hex}"
                    connection.execute(
                        "DELETE FROM result_artifact_bindings WHERE case_id = ?",
                        (row["case_id"],),
                    )
                    updated = connection.execute(
                        "UPDATE cases SET status = ?, stop_reason = ?, result_json = ?, "
                        "updated_at = ?, finished_at = ?, execution_token = ?, "
                        "execution_generation = ?, execution_state = 'FINISHED', "
                        "lease_expires_at = NULL WHERE case_id = ? AND status = ? "
                        "AND execution_token IS ? AND execution_generation = ? "
                        "AND execution_state = ? AND lease_expires_at IS ?",
                        (
                            result.status,
                            result.stop_reason,
                            encoded_result,
                            recovered_at.isoformat(),
                            result.finished_at.isoformat(),
                            recovery_token,
                            recovery_generation,
                            row["case_id"],
                            row["status"],
                            row["execution_token"],
                            row["execution_generation"],
                            row["execution_state"],
                            row["lease_expires_at"],
                        ),
                    )
                    if updated.rowcount != 1:
                        raise InvoiceAgentsError(
                            ErrorCategory.ORCHESTRATION,
                            "execution authority changed during recovery",
                            case_id=str(row["case_id"]),
                            stop_reason="EXECUTION_RECOVERY_RACE",
                        )
                    connection.execute(
                        "INSERT INTO events(event_id, case_id, source_id, event_type, "
                        "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            f"evt_{uuid4().hex}",
                            row["case_id"],
                            row["source_id"],
                            "case.execution_recovered",
                            json.dumps(
                                {
                                    "status": str(result.status),
                                    "stop_reason": result.stop_reason,
                                    "abandoned_execution_generation": int(
                                        row["execution_generation"]
                                    ),
                                    "recovery_execution_generation": recovery_generation,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            recovered_at.isoformat(),
                        ),
                    )
                    recovered.append(str(row["case_id"]))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return recovered

    def unrecovered_execution_case_ids(
        self,
        *,
        checked_at: datetime,
    ) -> list[str]:
        """Read expired RUNNING authorities a scan should have finished."""

        self._require_canonical_utc_clock(checked_at)
        with connect_database(self.path, read_only=True) as connection:
            rows = connection.execute(
                "SELECT case_id, status, execution_token, execution_generation, "
                "execution_state, lease_expires_at, updated_at FROM cases "
                "WHERE execution_state = 'RUNNING' "
                "AND status IN ('INCOMPLETE', 'NEEDS_HUMAN') ORDER BY case_id"
            ).fetchall()
        unrecovered: list[str] = []
        for row in rows:
            case_id = str(row["case_id"])
            if not self._authority_tuple_is_valid(row):
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    f"case {case_id} has a contradictory execution authority tuple",
                    case_id=case_id,
                    stop_reason="EXECUTION_AUTHORITY_CORRUPT",
                )
            lease = parse_canonical_utc(row["lease_expires_at"])
            if lease is not None and lease <= checked_at:
                unrecovered.append(case_id)
        return unrecovered

    def _load_relational_recovery_evidence(
        self,
        connection: sqlite3.Connection,
        case_id: str,
        source_id: str | None,
        aggregate_payment: PaymentResult | None = None,
    ) -> tuple[FinalDecision | None, ReviewRequest | None, PaymentResult | None]:
        """Strictly reconstruct durable evidence when the aggregate JSON is absent."""

        final_row = connection.execute(
            "SELECT payload_json, decision_generation, evidence_snapshot_digest, source_id, "
            "invoice_number, vendor, authorized_amount, authorized_currency, "
            "payment_idempotency_key, review_id "
            "FROM final_decisions WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        review_row = connection.execute(
            "SELECT payload_json, execution_generation FROM review_requests "
            "WHERE case_id = ? ORDER BY sequence DESC LIMIT 1",
            (case_id,),
        ).fetchone()
        payment_rows = connection.execute(
            "SELECT payment_id, case_id, idempotency_key, vendor, amount, currency, "
            "status, error, created_at, decision_generation, evidence_snapshot_digest, "
            "source_id, invoice_number, review_id FROM payments WHERE case_id = ? "
            "ORDER BY created_at, payment_id",
            (case_id,),
        ).fetchall()
        try:
            final = (
                FinalDecision.model_validate_json(final_row["payload_json"], strict=True)
                if final_row is not None
                else None
            )
            review = (
                ReviewRequest.model_validate_json(review_row["payload_json"], strict=True)
                if review_row is not None
                else None
            )
            if len(payment_rows) > 1:
                raise ValueError("multiple case-local payment ledger rows are ambiguous")
            persisted_payment = (
                PersistedPaymentRow.model_validate(dict(payment_rows[0]), strict=True)
                if payment_rows
                else None
            )
        except ValueError as exc:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "relational recovery evidence has an invalid storage shape",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from exc
        generations_match = persisted_payment is None or (
            final_row is not None
            and persisted_payment.decision_generation == int(final_row["decision_generation"])
        )
        sources_match = (final_row is None or final_row["source_id"] == source_id) and (
            persisted_payment is None or persisted_payment.source_id == source_id
        )
        payment_authorized = persisted_payment is None or (
            final is not None and final.payment_eligible and str(final.decision) == "APPROVE"
        )
        if not generations_match or not sources_match or not payment_authorized:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "relational recovery evidence does not match execution authority",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            )
        payment = (
            PaymentResult(
                payment_id=persisted_payment.payment_id,
                case_id=persisted_payment.case_id,
                idempotency_key=persisted_payment.idempotency_key,
                status=PaymentStatus(persisted_payment.status),
                vendor=persisted_payment.vendor,
                amount=Money(
                    amount=Decimal(persisted_payment.amount),
                    currency=persisted_payment.currency,
                ),
                processed_at=datetime.fromisoformat(persisted_payment.created_at),
                error=(
                    sanitize_text(persisted_payment.error)
                    if persisted_payment.error is not None
                    else None
                ),
            )
            if persisted_payment is not None
            else None
        )
        if aggregate_payment is not None and aggregate_payment.status is PaymentStatus.DUPLICATE:
            payment = self._reconstruct_duplicate_payment(
                connection,
                case_id,
                source_id,
                aggregate_payment,
                final_row,
                payment_rows,
            )
        return final, review, payment

    def _reconstruct_duplicate_payment(
        self,
        connection: sqlite3.Connection,
        case_id: str,
        source_id: str | None,
        aggregate: PaymentResult,
        final_row: sqlite3.Row | None,
        case_payment_rows: list[sqlite3.Row],
    ) -> PaymentResult:
        """Derive a DUPLICATE attempt from exact current and original authorities."""

        referenced_id = aggregate.duplicate_of
        if (
            aggregate.case_id != case_id
            or aggregate.payment_id is None
            or referenced_id is None
            or aggregate.payment_id != referenced_id
            or final_row is None
        ):
            self._raise_invalid_duplicate(case_id)
        referenced_rows = connection.execute(
            "SELECT payment_id, case_id, idempotency_key, vendor, amount, currency, "
            "status, error, created_at, decision_generation, evidence_snapshot_digest, "
            "source_id, invoice_number, review_id FROM payments WHERE payment_id = ?",
            (referenced_id,),
        ).fetchall()
        if len(referenced_rows) != 1:
            self._raise_invalid_duplicate(case_id)
        referenced_row = referenced_rows[0]
        try:
            original = PersistedPaymentRow.model_validate(dict(referenced_row), strict=True)
        except ValueError:
            self._raise_invalid_duplicate(case_id)
        if original.status != "PAID":
            self._raise_invalid_duplicate(case_id)
        if case_payment_rows and (
            len(case_payment_rows) != 1
            or case_payment_rows[0]["payment_id"] != original.payment_id
            or original.case_id != case_id
        ):
            self._raise_invalid_duplicate(case_id)

        try:
            from invoice_agents.payment.identity import payment_identity_key
            from invoice_agents.payment.service import (
                _AuthorizationSnapshotError,
                _load_authorization_snapshot,
                _validate_paid_ledger_source,
            )

            snapshot_settings = self._snapshot_settings()
            _validate_paid_ledger_source(connection, referenced_row, snapshot_settings)
            current_generation = int(final_row["decision_generation"])
            current = _load_authorization_snapshot(
                connection,
                case_id,
                current_generation,
                snapshot_settings,
            )
            invoice = current.invoice
            expected_key = payment_identity_key(
                invoice.vendor.normalized_value,
                invoice.invoice_number.normalized_value,
            )
            expected_amount = invoice.declared_total
            current_columns_are_exact = (
                final_row["source_id"] == source_id == invoice.source.source_id
                and final_row["invoice_number"] == invoice.invoice_number.normalized_value
                and final_row["vendor"] == invoice.vendor.normalized_value
                and final_row["authorized_amount"]
                == (str(expected_amount) if expected_amount is not None else None)
                and final_row["authorized_currency"] == invoice.currency.normalized_value
                and final_row["payment_idempotency_key"] == expected_key
                and final_row["evidence_snapshot_digest"] == current.evidence_snapshot_digest
                and original.idempotency_key == expected_key == aggregate.idempotency_key
                and original.vendor == invoice.vendor.normalized_value == aggregate.vendor
                and expected_amount is not None
                and Decimal(original.amount) == expected_amount
                and aggregate.amount == Money(amount=expected_amount, currency=original.currency)
                and original.currency == invoice.currency.normalized_value
                and aggregate.processed_at == datetime.fromisoformat(original.created_at)
                and aggregate.error
                == (sanitize_text(original.error) if original.error is not None else None)
            )
        except InvoiceAgentsError as exc:
            if exc.stop_reason == "EVIDENCE_AUTHORITY_MISSING":
                raise exc from None
            self._raise_invalid_duplicate(case_id)
        except (_AuthorizationSnapshotError, ValueError, TypeError):
            self._raise_invalid_duplicate(case_id)
        if not current_columns_are_exact:
            self._raise_invalid_duplicate(case_id)
        return PaymentResult(
            payment_id=original.payment_id,
            case_id=case_id,
            idempotency_key=original.idempotency_key,
            status=PaymentStatus.DUPLICATE,
            vendor=original.vendor,
            amount=Money(amount=Decimal(original.amount), currency=original.currency),
            processed_at=datetime.fromisoformat(original.created_at),
            duplicate_of=original.payment_id,
            error=sanitize_text(original.error) if original.error is not None else None,
        )

    @staticmethod
    def _raise_invalid_duplicate(case_id: str) -> Never:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "duplicate payment evidence does not match immutable payment authority",
            case_id=case_id,
            stop_reason="PERSISTED_RESULT_INVALID",
        ) from None

    @staticmethod
    def _require_relational_terminal_facts(
        aggregate: CaseResult,
        relational_final: FinalDecision | None,
        relational_review: ReviewRequest | None,
        relational_payment: PaymentResult | None,
    ) -> None:
        """Reject aggregate authorization facts whose relational authority vanished."""

        payment_requires_ledger = aggregate.payment is not None
        if (
            (aggregate.final_decision is not None and relational_final is None)
            or (aggregate.review_request is not None and relational_review is None)
            or (payment_requires_ledger and relational_payment is None)
        ):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "aggregate terminal facts are missing relational authority",
                case_id=aggregate.case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            )

    def merge_relational_case_evidence(self, result: CaseResult) -> CaseResult:
        """Overlay strictly validated relational facts on one terminal envelope."""

        with connect_database(self.path, read_only=True) as connection:
            case_row = connection.execute(
                "SELECT case_id, source_id, status, stop_reason, result_json, started_at, "
                "finished_at, execution_state FROM cases WHERE case_id = ?",
                (result.case_id,),
            ).fetchone()
            if case_row is None:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "terminal evidence case does not exist",
                    case_id=result.case_id,
                    stop_reason="CASE_NOT_FOUND",
                )
            self._decode_optional_stored_result_row(
                case_row,
                predecessor=case_row["execution_state"] != "FINISHED",
                require_result=case_row["execution_state"] == "FINISHED",
            )
            source_id = cast(str | None, case_row["source_id"])
            if result.source_id != source_id:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "terminal result identity does not match its relational case",
                    case_id=result.case_id,
                    stop_reason="PERSISTED_RESULT_INVALID",
                )
            self._require_terminal_chronology(result, case_row["started_at"])
            final, review, payment = self._load_relational_recovery_evidence(
                connection,
                result.case_id,
                source_id,
                result.payment,
            )
            self._require_relational_terminal_facts(result, final, review, payment)
        return result.model_copy(
            update={
                "final_decision": final,
                "review_request": review,
                "payment": payment,
            },
            deep=True,
        )

    @staticmethod
    def _require_canonical_utc_clock(value: datetime) -> None:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset != timedelta(0):
            raise ValueError("recovery clock must be timezone-aware UTC")

    @staticmethod
    def _authority_tuple_is_valid(row: sqlite3.Row) -> bool:
        state = row["execution_state"]
        token = row["execution_token"]
        raw_generation = row["execution_generation"]
        lease = row["lease_expires_at"]
        if not isinstance(state, str) or not isinstance(raw_generation, int):
            return False
        generation = raw_generation
        valid_token = type(token) is str and _REQUESTED_EXECUTION_TOKEN.fullmatch(token) is not None
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
        claim = validate_execution_claim(claim)
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
        claim_lease = execution_claim_expiry_iso(claim)
        lease = parse_canonical_utc(row["lease_expires_at"]) if row is not None else None
        if (
            row is None
            or claim_lease is None
            or row["execution_token"] != claim.token
            or int(row["execution_generation"]) != claim.generation
            or row["execution_state"] != "RUNNING"
            or lease is None
            or row["lease_expires_at"] != claim_lease
            or claim.expires_at <= checked_at
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
        claim = validate_execution_claim(claim, expected_case_id=case_id)
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
        claim = validate_execution_claim(claim)
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
        claim = validate_execution_claim(claim, expected_case_id=case_id)
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
        claim = validate_execution_claim(claim)
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
        claim = validate_execution_claim(claim, expected_case_id=case_id)
        critique_id = f"crit_{uuid4().hex}"
        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                written_at = datetime.now(UTC)
                self._require_current_claim(connection, claim, written_at)
                self._assert_evidence_generation_mutable(
                    connection,
                    case_id,
                    claim.generation,
                )
                records = self._critique_records(connection, case_id)
                if len(records) >= 2:
                    self._raise_critique_cycle_error(
                        case_id,
                        "the persisted two-cycle critic limit has been reached",
                        "CRITIQUE_CYCLE_LIMIT",
                    )
                if not records:
                    if (
                        critique.cycle != 1
                        or critique.responds_to_critique_id is not None
                        or critique.follow_up_responses
                    ):
                        self._raise_critique_cycle_error(
                            case_id,
                            "the first critique cannot respond to another critique",
                            "CRITIQUE_RESPONSE_INVALID",
                        )
                else:
                    parent_id, parent, _parent_generation = records[0]
                    if (
                        parent.cycle != 1
                        or critique.cycle != 2
                        or critique.responds_to_critique_id != parent_id
                    ):
                        self._raise_critique_cycle_error(
                            case_id,
                            "the second critique does not identify the exact first cycle",
                            "CRITIQUE_RESPONSE_INVALID",
                        )
                    from invoice_agents.agents.decision_rules import (
                        _assert_critique_sequence_complete,
                    )

                    _assert_critique_sequence_complete(case_id, [parent, critique])
                    parent_row = connection.execute(
                        "SELECT created_at FROM critique_results WHERE critique_id = ?",
                        (parent_id,),
                    ).fetchone()
                    if parent_row is None:
                        self._raise_critique_cycle_error(
                            case_id,
                            "the first critique disappeared during follow-up validation",
                            "CRITIQUE_RESPONSE_INVALID",
                        )
                    for response in critique.follow_up_responses:
                        for event_id in response.evidence_event_ids:
                            if not self._follow_up_event_is_valid(
                                connection,
                                case_id,
                                str(parent_row["created_at"]),
                                claim.generation,
                                event_id,
                            ):
                                self._raise_critique_cycle_error(
                                    case_id,
                                    "critique follow-up evidence is not bound to the exact "
                                    "case, chronology, and evidence-producing tool",
                                    "CRITIQUE_FOLLOW_UP_EVIDENCE_INVALID",
                                )
                connection.execute(
                    "INSERT INTO critique_results("
                    "critique_id, case_id, payload_json, created_at, execution_generation, "
                    "cycle, responds_to_critique_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        critique_id,
                        case_id,
                        critique.model_dump_json(),
                        written_at.isoformat(),
                        claim.generation,
                        critique.cycle,
                        critique.responds_to_critique_id,
                    ),
                )
                for response in critique.follow_up_responses:
                    connection.executemany(
                        "INSERT INTO critique_follow_up_evidence("
                        "critique_id, requested_item, outcome, evidence_event_id) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            (
                                critique_id,
                                response.requested_item,
                                response.outcome.value,
                                event_id,
                            )
                            for event_id in response.evidence_event_ids
                        ),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return critique_id

    @staticmethod
    def _raise_critique_cycle_error(
        case_id: str,
        message: str,
        stop_reason: str,
    ) -> Never:
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            message,
            case_id=case_id,
            stop_reason=stop_reason,
        )

    @staticmethod
    def _follow_up_event_is_valid(
        connection: sqlite3.Connection,
        case_id: str,
        parent_created_at: str,
        child_generation: int,
        event_id: str,
    ) -> bool:
        row = connection.execute(
            "SELECT evidence.event_id, evidence.case_id, evidence.source_id, "
            "evidence.event_type, evidence.agent_name, evidence.tool_call_id, "
            "evidence.db_evidence_id, evidence.review_id, evidence.payment_id, "
            "evidence.provider_request_id, evidence.payload_json, evidence.created_at, "
            "cases.source_id AS case_source_id FROM events evidence "
            "JOIN cases ON cases.case_id = ? WHERE evidence.event_id = ?",
            (case_id, event_id),
        ).fetchone()
        parent_timestamp = parse_canonical_utc(parent_created_at)
        if (
            row is None
            or parent_timestamp is None
            or type(child_generation) is not int
            or child_generation < 1
            or type(event_id) is not str
            or not event_id.startswith("evt_")
            or not event_id[4:]
        ):
            return False
        event_timestamp = parse_canonical_utc(row["created_at"])
        event_type = row["event_type"]
        if (
            event_timestamp is None
            or event_timestamp <= parent_timestamp
            or row["case_id"] != case_id
            or type(event_type) is not str
        ):
            return False
        if event_type in _DIRECT_CRITIQUE_FOLLOW_UP_EVENT_TYPES:
            return WorkflowStore._direct_critic_event_is_valid(row, child_generation)
        if event_type not in _PERSISTED_SPECIALIST_FOLLOW_UP_EVENT_TYPES:
            return False
        return WorkflowStore._specialist_event_is_valid(
            connection,
            row,
            case_id,
            child_generation,
            event_timestamp,
        )

    @staticmethod
    def _strict_event_payload(raw: object) -> object | None:
        if type(raw) is not str:
            return None
        try:
            return json.loads(
                raw,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @classmethod
    def _direct_critic_event_is_valid(
        cls,
        row: sqlite3.Row,
        child_generation: int,
    ) -> bool:
        if (
            row["agent_name"] != "independent_critic_agent"
            or row["source_id"] is not None
            or row["tool_call_id"] is not None
            or row["db_evidence_id"] is not None
            or row["review_id"] is not None
            or row["payment_id"] is not None
            or row["provider_request_id"] is not None
        ):
            return False
        return (
            _strict_critic_follow_up_payload(
                row["event_type"],
                row["payload_json"],
                child_generation,
            )
            == 1
        )

    @classmethod
    def _specialist_event_is_valid(
        cls,
        connection: sqlite3.Connection,
        event: sqlite3.Row,
        case_id: str,
        child_generation: int,
        event_timestamp: datetime,
    ) -> bool:
        event_contracts = {
            "tool.identity_candidates": (
                "identity_results",
                "identity_id",
                None,
                "identity_provenance_agent",
            ),
            "tool.inventory_comparison": (
                "comparison_results",
                "comparison_id",
                "inventory",
                "inventory_comparison_agent",
            ),
            "tool.mapping_evidence_recorded": (
                "extractions",
                "extraction_id",
                None,
                "inventory_comparison_agent",
            ),
            "tool.financial_risk_assessment": (
                "comparison_results",
                "comparison_id",
                "risk",
                "financial_risk_agent",
            ),
        }
        event_type = str(event["event_type"])
        table, id_column, comparison_type, agent_name = event_contracts[event_type]
        evidence_id = event["db_evidence_id"]
        if (
            type(evidence_id) is not str
            or not evidence_id.strip()
            or event["agent_name"] != agent_name
            or event["source_id"] != event["case_source_id"]
            or event["tool_call_id"] is not None
            or event["review_id"] is not None
            or event["payment_id"] is not None
            or event["provider_request_id"] is not None
        ):
            return False
        predicate = " AND comparison_type = ?" if comparison_type is not None else ""
        params: tuple[object, ...] = (
            (evidence_id, case_id, child_generation, comparison_type)
            if comparison_type is not None
            else (evidence_id, case_id, child_generation)
        )
        evidence = connection.execute(
            f"SELECT payload_json, created_at FROM {table} WHERE {id_column} = ? "
            f"AND case_id = ? AND execution_generation = ?{predicate}",
            params,
        ).fetchone()
        if evidence is None:
            return False
        evidence_timestamp = parse_canonical_utc(evidence["created_at"])
        if evidence_timestamp is None or evidence_timestamp > event_timestamp:
            return False
        event_payload = cls._strict_event_payload(event["payload_json"])
        stored_payload = cls._strict_event_payload(evidence["payload_json"])
        if event_type != "tool.mapping_evidence_recorded":
            if event_payload is None or event_payload != stored_payload:
                return False
            try:
                if event_type == "tool.identity_candidates":
                    TypeAdapter(list[IdentityCandidate]).validate_json(
                        str(evidence["payload_json"]),
                        strict=True,
                    )
                elif event_type == "tool.inventory_comparison":
                    if type(stored_payload) is not dict or set(stored_payload) != {
                        "comparisons",
                        "unresolved_candidates",
                    }:
                        return False
                    TypeAdapter(list[InventoryComparison]).validate_json(
                        json.dumps(stored_payload["comparisons"], ensure_ascii=False),
                        strict=True,
                    )
                    unresolved = stored_payload["unresolved_candidates"]
                    if type(unresolved) is not dict:
                        return False
                    for result in unresolved.values():
                        InventoryLookupResult.model_validate_json(
                            json.dumps(result, ensure_ascii=False, sort_keys=True),
                            strict=True,
                        )
                else:
                    RiskAssessment.model_validate_json(
                        str(evidence["payload_json"]),
                        strict=True,
                    )
            except ValueError:
                return False
            return True
        if type(event_payload) is not dict or set(event_payload) != {"extraction_id", "lines"}:
            return False
        try:
            invoice = ExtractedInvoice.model_validate_json(
                str(evidence["payload_json"]),
                strict=True,
            )
        except ValueError:
            return False
        expected_lines = [
            {
                "line_id": line.line_id,
                "canonical_sku": line.canonical_sku,
                "candidate_skus": line.candidate_skus,
            }
            for line in invoice.lines
        ]
        return event_payload == {
            "extraction_id": evidence_id,
            "lines": expected_lines,
        }

    @classmethod
    def _follow_up_rows_match_payload(
        cls,
        connection: sqlite3.Connection,
        case_id: str,
        parent_created_at: str,
        child_generation: int,
        critique_id: str,
        responses: list[CritiqueFollowUpResponse],
    ) -> bool:
        rows = connection.execute(
            "SELECT requested_item, outcome, evidence_event_id "
            "FROM critique_follow_up_evidence WHERE critique_id = ?",
            (critique_id,),
        ).fetchall()
        expected = {
            (response.requested_item, response.outcome.value, event_id)
            for response in responses
            for event_id in response.evidence_event_ids
        }
        actual = {
            (str(row["requested_item"]), str(row["outcome"]), str(row["evidence_event_id"]))
            for row in rows
        }
        return expected == actual and all(
            cls._follow_up_event_is_valid(
                connection,
                case_id,
                parent_created_at,
                child_generation,
                str(row["evidence_event_id"]),
            )
            for row in rows
        )

    @classmethod
    def _critique_records(
        cls,
        connection: sqlite3.Connection,
        case_id: str,
    ) -> list[tuple[str, Critique, int]]:
        rows = connection.execute(
            "SELECT critique_id, payload_json, created_at, execution_generation, cycle, "
            "responds_to_critique_id FROM critique_results WHERE case_id = ? "
            "ORDER BY cycle",
            (case_id,),
        ).fetchall()
        records: list[tuple[str, Critique, int]] = []
        created_at_by_id: dict[str, str] = {}
        for row in rows:
            try:
                critique = Critique.model_validate_json(row["payload_json"], strict=True)
            except (TypeError, ValueError) as exc:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "persisted critique payload is invalid",
                    case_id=case_id,
                    stop_reason="PERSISTED_RESULT_INVALID",
                ) from exc
            relational_cycle = row["cycle"]
            relational_parent = row["responds_to_critique_id"]
            generation = row["execution_generation"]
            created_at = row["created_at"]
            if (
                type(row["critique_id"]) is not str
                or type(relational_cycle) is not int
                or (relational_parent is not None and type(relational_parent) is not str)
                or type(generation) is not int
                or type(created_at) is not str
                or parse_canonical_utc(created_at) is None
                or critique.cycle != relational_cycle
                or critique.responds_to_critique_id != relational_parent
                or (critique.cycle == 1 and bool(critique.follow_up_responses))
            ):
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "persisted critique relationship does not match its payload",
                    case_id=case_id,
                    stop_reason="PERSISTED_RESULT_INVALID",
                )
            created_at_by_id[row["critique_id"]] = created_at
            records.append((row["critique_id"], critique, generation))
        if records:
            first_id, first, first_generation = records[0]
            relationship_is_valid = first.cycle == 1 and first.responds_to_critique_id is None
            if len(records) == 2:
                second_id, second, second_generation = records[1]
                relationship_is_valid = relationship_is_valid and (
                    second.cycle == 2
                    and second.responds_to_critique_id == first_id
                    and second_generation >= first_generation
                    and datetime.fromisoformat(created_at_by_id[second_id])
                    > datetime.fromisoformat(created_at_by_id[first_id])
                )
                follow_up_is_valid = cls._follow_up_rows_match_payload(
                    connection,
                    case_id,
                    created_at_by_id[first_id],
                    second_generation,
                    second_id,
                    second.follow_up_responses,
                )
                if relationship_is_valid and not follow_up_is_valid:
                    raise InvoiceAgentsError(
                        ErrorCategory.DATABASE,
                        "persisted critique follow-up evidence does not match its payload",
                        case_id=case_id,
                        stop_reason="PERSISTED_RESULT_INVALID",
                    )
                relationship_is_valid = relationship_is_valid and follow_up_is_valid
            if len(records) > 2 or not relationship_is_valid:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "persisted critique cycle sequence is invalid",
                    case_id=case_id,
                    stop_reason="PERSISTED_RESULT_INVALID",
                )
        return records

    def list_current_critique_follow_up_events(
        self,
        claim: ExecutionClaim,
        parent_critique_id: str,
    ) -> list[dict[str, Any]]:
        """Return exact post-parent evidence identities eligible for cycle two."""

        claim = validate_execution_claim(claim)
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            parent = connection.execute(
                "SELECT created_at FROM critique_results WHERE critique_id = ? "
                "AND case_id = ? AND cycle = 1",
                (parent_critique_id, claim.case_id),
            ).fetchone()
            if parent is None or parse_canonical_utc(parent["created_at"]) is None:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "critique follow-up parent is absent or invalid",
                    case_id=claim.case_id,
                    stop_reason="PERSISTED_RESULT_INVALID",
                )
            rows = connection.execute(
                "SELECT event_id, event_type, db_evidence_id, created_at FROM events "
                "WHERE case_id = ? AND created_at > ? ORDER BY created_at, event_id",
                (claim.case_id, parent["created_at"]),
            ).fetchall()
            eligible = [
                {
                    "evidence_event_id": str(row["event_id"]),
                    "event_type": str(row["event_type"]),
                    "db_evidence_id": row["db_evidence_id"],
                    "created_at": str(row["created_at"]),
                }
                for row in rows
                if self._follow_up_event_is_valid(
                    connection,
                    claim.case_id,
                    str(parent["created_at"]),
                    claim.generation,
                    str(row["event_id"]),
                )
            ]
        return eligible

    def list_critiques(self, case_id: str) -> list[Critique]:
        with connect_database(self.path, read_only=True) as connection:
            records = self._critique_records(connection, case_id)
        return [critique for _critique_id, critique, _generation in records]

    def list_critique_records(self, case_id: str) -> list[tuple[str, Critique]]:
        """Return persisted critique IDs and payloads in authoritative cycle order."""

        with connect_database(self.path, read_only=True) as connection:
            records = self._critique_records(connection, case_id)
        return [
            (critique_id, critique)
            for critique_id, critique, _generation in records
        ]

    def load_critique(self, case_id: str) -> Critique:
        critiques = self.list_critiques(case_id)
        if not critiques:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"critic has not recorded a result for case {case_id}",
                case_id=case_id,
                stop_reason="CRITIQUE_MISSING",
            )
        return critiques[-1]

    def load_current_critique(self, claim: ExecutionClaim) -> Critique:
        claim = validate_execution_claim(claim)
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            records = self._critique_records(connection, claim.case_id)
        if any(generation > claim.generation for _id, _critique, generation in records):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"future critique evidence exists for case {claim.case_id}",
                case_id=claim.case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            )
        current = [
            critique
            for _critique_id, critique, generation in records
            if generation <= claim.generation
        ]
        if not current:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"current execution has no critique for case {claim.case_id}",
                case_id=claim.case_id,
                stop_reason="CRITIQUE_GENERATION_MISMATCH",
            )
        return current[-1]

    def adopt_latest_evidence(self, claim: ExecutionClaim) -> None:
        """Promote one complete, coherent immediate-predecessor snapshot."""

        claim = validate_execution_claim(claim)
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
                    predecessor_review_authorization = load_authoritative_review_authorization(
                        connection,
                        claim.case_id,
                        predecessor,
                    )
                    predecessor_review = (
                        predecessor_review_authorization.review
                        if predecessor_review_authorization is not None
                        else None
                    )
                    snapshot = load_generation_evidence_snapshot(
                        connection,
                        claim.case_id,
                        predecessor,
                        self._snapshot_settings(),
                        excluded_alias_sources=_review_alias_sources(predecessor_review),
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
                connection.execute(
                    "INSERT INTO identity_results(identity_id, case_id, payload_json, created_at, "
                    "execution_generation, evaluated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        f"ident_{uuid4().hex}",
                        claim.case_id,
                        encode([item.model_dump(mode="json") for item in snapshot.identity]),
                        adopted_at.isoformat(),
                        claim.generation,
                        snapshot.identity_evaluated_at.isoformat(),
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
                if predecessor_review_authorization is not None:
                    try:
                        predecessor_review = predecessor_review_authorization.review
                        validate_review_snapshot(predecessor_review, snapshot)
                    except (EvidenceSnapshotError, ValueError) as exc:
                        self._raise_evidence_provenance(claim, str(exc))
                    if predecessor_review_authorization.evidence_snapshot_digest != snapshot.digest:
                        self._raise_evidence_provenance(
                            claim, "predecessor review snapshot digest does not match evidence"
                        )
                    updated = connection.execute(
                        "UPDATE review_requests SET execution_generation = ? "
                        "WHERE review_id = ? AND execution_generation = ?",
                        (claim.generation, predecessor_review.review_id, predecessor),
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

        claim = validate_execution_claim(claim)
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

        claim = validate_execution_claim(claim, expected_case_id=review.case_id)
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, written_at)
            self._assert_evidence_generation_mutable(connection, review.case_id, claim.generation)
            from invoice_agents.agents.decision_rules import (
                _assert_critique_sequence_complete,
            )

            persisted_critiques = [
                critique
                for _critique_id, critique, _generation in self._critique_records(
                    connection,
                    review.case_id,
                )
            ]
            _assert_critique_sequence_complete(review.case_id, persisted_critiques)
            try:
                snapshot = load_generation_evidence_snapshot(
                    connection,
                    review.case_id,
                    claim.generation,
                    self._snapshot_settings(),
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
        return ReviewRequest.model_validate_json(row["payload_json"], strict=True)

    def load_case_review(self, case_id: str) -> ReviewRequest | None:
        """Return the latest review cycle for the case, by sequence."""

        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM review_requests WHERE case_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        return ReviewRequest.model_validate_json(row["payload_json"], strict=True) if row else None

    def load_current_review(self, claim: ExecutionClaim) -> ReviewRequest | None:
        """Return only the latest review owned by the current unexpired generation."""

        claim = validate_execution_claim(claim)
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            row = connection.execute(
                "SELECT payload_json FROM review_requests WHERE case_id = ? "
                "AND execution_generation = ? ORDER BY sequence DESC LIMIT 1",
                (claim.case_id, claim.generation),
            ).fetchone()
        return ReviewRequest.model_validate_json(row["payload_json"], strict=True) if row else None

    def list_reviews(self, pending_only: bool = True) -> list[ReviewRequest]:
        sql = "SELECT payload_json FROM review_requests"
        params: tuple[str, ...] = ()
        if pending_only:
            sql += " WHERE status = ?"
            params = ("PENDING",)
        sql += " ORDER BY created_at, sequence"
        with connect_database(self.path, read_only=True) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [ReviewRequest.model_validate_json(row["payload_json"], strict=True) for row in rows]

    def classify_human_decision_replay(
        self,
        review_id: str,
        decision: HumanDecision | None,
        inventory_db: Path,
    ) -> ReviewRequest | None:
        """Classify persisted resolved state against exact workflow and mapping facts.

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
                    connection,
                    str(row["case_id"]),
                    generation,
                    self._snapshot_settings(),
                    excluded_alias_sources=_review_alias_sources(review),
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
            existing = review.human_decision
            if existing is None:
                self._raise_snapshot_invalid(
                    str(row["case_id"]),
                    generation,
                    "resolved review has no human decision",
                )
            if existing.mappings:
                with connect_database(inventory_db, read_only=True) as inventory:
                    _validate_human_decision_authority(
                        connection,
                        review,
                        existing,
                        inventory_connection=inventory,
                        inventory_schema="main",
                        require_persisted_mapping_provenance=True,
                    )
            else:
                _validate_human_decision_authority(
                    connection,
                    review,
                    existing,
                    inventory_connection=None,
                    inventory_schema="main",
                    require_persisted_mapping_provenance=True,
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
                if mode.casefold() != REQUIRED_JOURNAL_MODE
            }
            if incompatible:
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    "atomic human decisions require DELETE journal mode for workflow and "
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
                    snapshot = load_generation_evidence_snapshot(
                        connection,
                        case_id,
                        generation,
                        self._snapshot_settings(),
                        excluded_alias_sources=_review_alias_sources(review),
                    )
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
        claim = validate_execution_claim(claim, expected_case_id=case_id)
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
            existing_final = connection.execute(
                "SELECT 1 FROM final_decisions WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if existing_final is not None:
                connection.rollback()
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    f"case {case_id} already has an immutable final decision",
                    case_id=case_id,
                    stop_reason="FINAL_DECISION_IMMUTABLE",
                )
            from invoice_agents.agents.decision_rules import (
                _assert_critique_sequence_complete,
            )

            persisted_critiques = [
                critique
                for _critique_id, critique, _generation in self._critique_records(
                    connection,
                    case_id,
                )
            ]
            _assert_critique_sequence_complete(case_id, persisted_critiques)
            try:
                review_authorization = load_authoritative_review_authorization(
                    connection,
                    case_id,
                    claim.generation,
                )
                snapshot = load_authorization_evidence_snapshot(
                    connection,
                    case_id,
                    claim.generation,
                    self._snapshot_settings(),
                    review_authorization,
                )
            except EvidenceSnapshotError as exc:
                self._raise_snapshot_invalid(case_id, claim.generation, str(exc))
            review = review_authorization.review if review_authorization is not None else None
            from invoice_agents.agents.decision_rules import validate_final_decision

            validate_final_decision(
                decision.decision,
                decision.payment_eligible,
                snapshot.risk,
                snapshot.critique,
                review,
                case_id=case_id,
            )
            try:
                validate_final_decision_snapshot(decision, snapshot, review)
            except EvidenceSnapshotError as exc:
                self._raise_snapshot_invalid(
                    case_id,
                    claim.generation,
                    str(exc),
                )
            from invoice_agents.payment.identity import payment_identity_key

            invoice = snapshot.invoice
            invoice_number = invoice.invoice_number.normalized_value
            vendor = invoice.vendor.normalized_value
            authorized_amount = (
                str(invoice.declared_total) if invoice.declared_total is not None else None
            )
            authorized_currency = invoice.currency.normalized_value
            idempotency_key = payment_identity_key(vendor, invoice_number)
            facts = validated_evidence_facts(snapshot, review_authorization)
            existing_anchor = connection.execute(
                "SELECT 1 FROM validated_evidence_snapshots WHERE case_id = ? "
                "AND execution_generation = ?",
                (case_id, claim.generation),
            ).fetchone()
            if existing_anchor is not None:
                self._raise_snapshot_invalid(
                    case_id,
                    claim.generation,
                    "validated evidence snapshot anchor already exists",
                )
            connection.execute(
                "INSERT INTO validated_evidence_snapshots("
                "case_id, execution_generation, evidence_snapshot_digest, "
                "policy_review_required, unresolved_blocker_count, critique_disposition, "
                "review_id, review_snapshot_digest, validated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    claim.generation,
                    snapshot.digest,
                    facts.policy_review_required,
                    facts.unresolved_blocker_count,
                    facts.critique_disposition,
                    facts.review_id,
                    facts.review_snapshot_digest,
                    written_at,
                ),
            )
            updated = connection.execute(
                "INSERT INTO final_decisions("
                "decision_id, case_id, payload_json, created_at, decision_generation, "
                "evidence_snapshot_digest, source_id, invoice_number, vendor, "
                "authorized_amount, authorized_currency, payment_idempotency_key, review_id) "
                "SELECT ?, case_id, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? FROM cases "
                "WHERE case_id = ? "
                "AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?",
                (
                    f"fdec_{uuid4().hex}",
                    decision.model_dump_json(),
                    written_at,
                    claim.generation,
                    snapshot.digest,
                    invoice.source.source_id,
                    invoice_number,
                    vendor,
                    authorized_amount,
                    authorized_currency,
                    idempotency_key,
                    review.review_id if review is not None else None,
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

    def load_final_decision(self, case_id: str) -> FinalDecision | None:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM final_decisions WHERE case_id = ?", (case_id,)
            ).fetchone()
        return FinalDecision.model_validate_json(row["payload_json"], strict=True) if row else None

    def load_current_final_decision(self, claim: ExecutionClaim) -> FinalDecision | None:
        """Return only a final decision owned by the current unexpired generation."""

        claim = validate_execution_claim(claim)
        with connect_database(self.path, read_only=True) as connection:
            self._begin_current_read(connection, claim)
            row = connection.execute(
                "SELECT payload_json FROM final_decisions WHERE case_id = ? "
                "AND decision_generation = ?",
                (claim.case_id, claim.generation),
            ).fetchone()
        return FinalDecision.model_validate_json(row["payload_json"], strict=True) if row else None

    def save_team_state(self, case_id: str, state: dict[str, Any], claim: ExecutionClaim) -> None:
        claim = validate_execution_claim(claim, expected_case_id=case_id)
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
        claim = validate_execution_claim(claim, expected_case_id=result.case_id)
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC).isoformat()
            self._require_current_claim(connection, claim, datetime.fromisoformat(written_at))
            self._require_terminal_result_identity(connection, result)
            encoded_result = self._encode_terminal_result(connection, result)
            connection.execute(
                "DELETE FROM result_artifact_bindings WHERE case_id = ?",
                (result.case_id,),
            )
            updated = connection.execute(
                "UPDATE cases SET status = ?, stop_reason = ?, result_json = ?, updated_at = ?, "
                "finished_at = ?, execution_state = 'FINISHED', lease_expires_at = NULL "
                "WHERE case_id = ? AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'RUNNING' AND lease_expires_at > ?",
                (
                    result.status,
                    result.stop_reason,
                    encoded_result,
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

    def update_finished_case_result(self, result: CaseResult, claim: ExecutionClaim) -> None:
        """Replace only this generation's terminal envelope after a secondary fault."""

        claim = validate_execution_claim(claim, expected_case_id=result.case_id)
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC).isoformat()
            current = connection.execute(
                "SELECT case_id, source_id, status, stop_reason, result_json, started_at, "
                "finished_at, execution_token, execution_generation, execution_state, "
                "lease_expires_at FROM cases WHERE case_id = ?",
                (result.case_id,),
            ).fetchone()
            if current is not None and not self._authority_tuple_is_valid(current):
                raise InvoiceAgentsError(
                    ErrorCategory.DATABASE,
                    f"case {result.case_id} has a contradictory execution authority tuple",
                    case_id=result.case_id,
                    stop_reason="EXECUTION_AUTHORITY_CORRUPT",
                )
            if (
                current is None
                or current["execution_token"] != claim.token
                or int(current["execution_generation"]) != claim.generation
                or current["execution_state"] != "FINISHED"
                or current["lease_expires_at"] is not None
            ):
                self._raise_stale_execution_claim(claim)
            self._decode_optional_stored_result_row(
                current,
                predecessor=False,
                require_result=True,
            )
            self._require_terminal_result_identity(connection, result)
            encoded_result = self._encode_terminal_result(connection, result)
            connection.execute(
                "DELETE FROM result_artifact_bindings WHERE case_id = ?",
                (result.case_id,),
            )
            updated = connection.execute(
                "UPDATE cases SET status = ?, stop_reason = ?, result_json = ?, "
                "updated_at = ?, finished_at = ? WHERE case_id = ? "
                "AND execution_token = ? AND execution_generation = ? "
                "AND execution_state = 'FINISHED' AND lease_expires_at IS NULL",
                (
                    result.status,
                    result.stop_reason,
                    encoded_result,
                    written_at,
                    result.finished_at.isoformat(),
                    result.case_id,
                    claim.token,
                    claim.generation,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                self._raise_stale_execution_claim(claim)
            connection.commit()

    @staticmethod
    def _require_terminal_result_identity(
        connection: sqlite3.Connection,
        result: CaseResult,
    ) -> None:
        """Validate aggregate source identity against authoritative rows in this write tx."""

        try:
            source = _authoritative_source_for_case(connection, result.case_id)
        except EvidenceSnapshotError:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "terminal result source authority is missing or inconsistent",
                case_id=result.case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        if result.source_id != source.source_id:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "terminal result identity does not match its authoritative source",
                case_id=result.case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None

    def load_case_source_id(self, case_id: str) -> str | None:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT source_id FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"case does not exist: {case_id}",
                case_id=case_id,
                stop_reason="CASE_NOT_FOUND",
            )
        return cast(str | None, row["source_id"])

    def load_result(self, case_id: str) -> CaseResult | None:
        with connect_database(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT case_id, source_id, status, stop_reason, result_json, "
                "started_at, finished_at FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"case does not exist: {case_id}",
                case_id=case_id,
                stop_reason="CASE_NOT_FOUND",
            )
        if row["result_json"] is not None:
            return self._decode_terminal_result_row(row)
        if row["finished_at"] is not None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has a terminal timestamp without a persisted result",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        return None

    @staticmethod
    def _artifact_binding_is_valid(binding: object) -> bool:
        return (
            type(binding) is ResultArtifactBinding
            and type(binding.case_id) is str
            and bool(binding.case_id)
            and type(binding.execution_generation) is int
            and binding.execution_generation >= 1
            and type(binding.artifact_sha256) is str
            and _ARTIFACT_SHA256.fullmatch(binding.artifact_sha256) is not None
            and type(binding.artifact_device) is int
            and binding.artifact_device >= 0
            and type(binding.artifact_inode) is int
            and binding.artifact_inode > 0
            and type(binding.artifact_file_type) is int
            and binding.artifact_file_type == stat.S_IFREG
            and type(binding.artifact_size_bytes) is int
            and binding.artifact_size_bytes >= 0
        )

    @classmethod
    def _decode_result_artifact_binding_row(
        cls,
        row: sqlite3.Row,
    ) -> ResultArtifactBinding | None:
        if row["binding_case_id"] is None:
            return None
        binding = ResultArtifactBinding(
            case_id=row["binding_case_id"],
            execution_generation=row["binding_execution_generation"],
            artifact_sha256=row["artifact_sha256"],
            artifact_device=row["artifact_device"],
            artifact_inode=row["artifact_inode"],
            artifact_file_type=row["artifact_file_type"],
            artifact_size_bytes=row["artifact_size_bytes"],
        )
        if (
            not cls._artifact_binding_is_valid(binding)
            or binding.case_id != row["case_id"]
            or binding.execution_generation != row["execution_generation"]
        ):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has an invalid persisted result-artifact binding",
                case_id=str(row["case_id"]),
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        return binding

    def load_result_with_artifact_binding(
        self,
        case_id: str,
    ) -> tuple[CaseResult | None, int, ResultArtifactBinding | None]:
        """Read terminal result, current generation, and binding in one snapshot."""

        with connect_database(self.path, read_only=True) as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT c.case_id, c.source_id, c.status, c.stop_reason, c.result_json, "
                "c.started_at, c.finished_at, c.execution_token, c.execution_generation, "
                "c.execution_state, c.lease_expires_at, "
                "b.case_id AS binding_case_id, "
                "b.execution_generation AS binding_execution_generation, "
                "b.artifact_sha256, b.artifact_device, b.artifact_inode, "
                "b.artifact_file_type, b.artifact_size_bytes "
                "FROM cases c LEFT JOIN result_artifact_bindings b ON b.case_id = c.case_id "
                "WHERE c.case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                f"case does not exist: {case_id}",
                case_id=case_id,
                stop_reason="CASE_NOT_FOUND",
            ) from None
        if not self._authority_tuple_is_valid(row):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has an invalid execution authority for result-artifact binding",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        terminal_parent = row["execution_state"] == "FINISHED"
        result = self._decode_optional_stored_result_row(
            row,
            predecessor=not terminal_parent,
            require_result=terminal_parent,
        )
        generation = row["execution_generation"]
        if type(generation) is not int or generation < 0:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has an invalid execution generation",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        binding = self._decode_result_artifact_binding_row(row)
        if binding is not None and (not terminal_parent or result is None):
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "case has a result-artifact binding without exact terminal authority",
                case_id=case_id,
                stop_reason="PERSISTED_RESULT_INVALID",
            ) from None
        return result, generation, binding

    def load_result_artifact_binding(self, case_id: str) -> ResultArtifactBinding | None:
        """Load only the binding currently authorized by the case generation."""

        _result, _generation, binding = self.load_result_with_artifact_binding(case_id)
        return binding

    def save_result_artifact_binding(
        self,
        binding: ResultArtifactBinding,
        result: CaseResult,
    ) -> None:
        """Bind an already-durable exact file to the still-current terminal result."""

        if not self._artifact_binding_is_valid(binding) or binding.case_id != result.case_id:
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "result-artifact binding has an invalid runtime shape",
                case_id=result.case_id,
                stop_reason="RESULT_ARTIFACT_BINDING_INVALID",
            ) from None
        primary_failure: BaseException | None = None
        rollback_failure: BaseException | None = None
        with connect_database(self.path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT case_id, source_id, status, stop_reason, result_json, started_at, "
                    "finished_at, execution_token, execution_generation, execution_state, "
                    "lease_expires_at FROM cases WHERE case_id = ?",
                    (binding.case_id,),
                ).fetchone()
                if (
                    row is None
                    or not self._authority_tuple_is_valid(row)
                    or row["execution_state"] != "FINISHED"
                    or row["lease_expires_at"] is not None
                    or row["execution_generation"] != binding.execution_generation
                ):
                    raise InvoiceAgentsError(
                        ErrorCategory.DATABASE,
                        "result-artifact binding authority is no longer current",
                        case_id=binding.case_id,
                        stop_reason="RESULT_ARTIFACT_BINDING_STALE",
                    )
                stored = self._decode_optional_stored_result_row(
                    row,
                    predecessor=False,
                    require_result=True,
                )
                if stored != result:
                    raise InvoiceAgentsError(
                        ErrorCategory.DATABASE,
                        "result-artifact binding does not match the current terminal result",
                        case_id=binding.case_id,
                        stop_reason="RESULT_ARTIFACT_BINDING_STALE",
                    )
                _execute_result_artifact_binding(
                    connection,
                    "INSERT INTO result_artifact_bindings("
                    "case_id, execution_generation, artifact_sha256, artifact_device, "
                    "artifact_inode, artifact_file_type, artifact_size_bytes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(case_id) DO UPDATE SET "
                    "execution_generation = excluded.execution_generation, "
                    "artifact_sha256 = excluded.artifact_sha256, "
                    "artifact_device = excluded.artifact_device, "
                    "artifact_inode = excluded.artifact_inode, "
                    "artifact_file_type = excluded.artifact_file_type, "
                    "artifact_size_bytes = excluded.artifact_size_bytes",
                    (
                        binding.case_id,
                        binding.execution_generation,
                        binding.artifact_sha256,
                        binding.artifact_device,
                        binding.artifact_inode,
                        binding.artifact_file_type,
                        binding.artifact_size_bytes,
                    ),
                )
                _commit_result_artifact_binding(connection)
                return
            except BaseException as exc:
                primary_failure = exc
                try:
                    _rollback_result_artifact_binding(connection)
                except BaseException as rollback_exc:
                    rollback_failure = rollback_exc

        assert primary_failure is not None
        selected_failure = _binding_failure_precedence(primary_failure, rollback_failure)
        try:
            observed_result, observed_generation, observed_binding = (
                self.load_result_with_artifact_binding(binding.case_id)
            )
        except BaseException as readback_failure:
            unresolved_failure = _binding_failure_precedence(
                primary_failure,
                rollback_failure,
                readback_failure,
            )
            if not isinstance(unresolved_failure, Exception):
                _raise_chainless(unresolved_failure)
            raise InvoiceAgentsError(
                ErrorCategory.DATABASE,
                "result-artifact binding transaction outcome could not be proven",
                case_id=binding.case_id,
                stop_reason="RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED",
            ) from None
        exact_parent = (
            observed_result == result
            and observed_generation == binding.execution_generation
        )
        if exact_parent and observed_binding == binding:
            if not isinstance(selected_failure, Exception):
                _raise_chainless(selected_failure)
            return
        if exact_parent and observed_binding is None:
            _raise_chainless(selected_failure)
        if not isinstance(selected_failure, Exception):
            _raise_chainless(selected_failure)
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "result-artifact binding transaction resolved to a contradictory state",
            case_id=binding.case_id,
            stop_reason="RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED",
        ) from None

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
                "WHERE c.case_id <> ? AND (c.invoice_number = ? OR c.vendor = ?) "
                "ORDER BY c.case_id",
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
        claim = validate_execution_claim(claim, expected_case_id=case_id)
        allowed = {"identity_results", "critique_results"}
        if table not in allowed:
            raise ValueError(f"unsupported payload table: {table}")
        with connect_database(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            written_at = datetime.now(UTC)
            self._require_current_claim(connection, claim, written_at)
            self._assert_evidence_generation_mutable(connection, case_id, claim.generation)
            if table == "identity_results":
                connection.execute(
                    f"INSERT INTO {table}("
                    f"{id_column}, case_id, payload_json, created_at, execution_generation, "
                    "evaluated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record_id,
                        case_id,
                        encode(payload),
                        written_at.isoformat(),
                        claim.generation,
                        written_at.isoformat(),
                    ),
                )
            else:
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
        claim = validate_execution_claim(claim)
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

        claim = validate_execution_claim(claim)
        connection.execute("BEGIN")
        self._require_current_claim(connection, claim, datetime.now(UTC))
