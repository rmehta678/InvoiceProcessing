"""Console routes: reads render stored state, mutations delegate to services.

Every mutation goes through ``record_human_decision``, ``prepare_invoice`` /
``run_prepared_case`` (the ``process_invoice`` seam), or ``resume_case``. The UI
composes no SQL of its own beyond the read-only queries in :mod:`queries`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, unquote

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from sse_starlette.sse import EventSourceResponse

from invoice_agents.agents.decision_rules import AUTHORIZING_HUMAN_DECISIONS
from invoice_agents.config import XAI_BASE_URL, XAI_MODEL, Settings
from invoice_agents.db.core import DatabaseKind, verify_database
from invoice_agents.db.store import ResultArtifactBinding, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError, SourceEvidenceError
from invoice_agents.hitl.service import record_human_decision
from invoice_agents.models import (
    CanonicalMapping,
    CaseResult,
    CaseStatus,
    Critique,
    ExtractedInvoice,
    HumanDecisionKind,
    ReviewRequest,
)
from invoice_agents.observability.audit import sanitize_case_result, sanitize_text
from invoice_agents.orchestration import validate_case_concurrency
from invoice_agents.ui import queries
from invoice_agents.ui.preflight import key_present, run_preflight
from invoice_agents.ui.recovery import RecoveryCoordinator
from invoice_agents.ui.runs import RunRegistry
from invoice_agents.ui.security import secure_cookie
from invoice_agents.ui.sse import case_event_stream

router = APIRouter()

INVOICE_DIR = Path("data/invoices")
UPLOAD_DIR_NAME = "uploads"
SUPPORTED_SUFFIXES = {".txt", ".json", ".csv", ".xml", ".pdf"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
RESULT_ARTIFACT_MAX_BYTES = 1_048_576
RESULT_ARTIFACT_MAX_DEPTH = 64

# One-line consequence per HumanDecisionKind, matching decision_rules exactly:
# an authorizing decision permits APPROVE (or HOLD when blocking evidence remains),
# REJECT forces REJECT, REQUEST_CORRECTION forces HOLD.
DECISION_CONSEQUENCES: dict[HumanDecisionKind, str] = {
    HumanDecisionKind.APPROVE: (
        "Authorizes approval: the resumed team may finalize APPROVE and record the mock "
        "payment - or HOLD if blocking evidence (stock, total delta) remains."
    ),
    HumanDecisionKind.REJECT: (
        "Forces final decision REJECT; the workflow completes as SUCCEEDED with no payment."
    ),
    HumanDecisionKind.REQUEST_CORRECTION: (
        "Forces final decision HOLD; no payment. Correct the source file and reprocess."
    ),
    HumanDecisionKind.ESTABLISH_MAPPING: (
        "Records the raw item - SKU aliases below, deterministically recomputes inventory, "
        "financial, and risk evidence, then authorizes like APPROVE."
    ),
    HumanDecisionKind.SUPERSEDE_REVISION: (
        "Marks the selected prior case as superseded by this revision and authorizes like APPROVE."
    ),
}


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def _registry(request: Request) -> RunRegistry:
    registry: RunRegistry = request.app.state.registry
    return registry


def _recovery_coordinator(request: Request) -> RecoveryCoordinator:
    coordinator: RecoveryCoordinator = request.app.state.recovery_coordinator
    return coordinator


def _store(request: Request) -> WorkflowStore:
    return WorkflowStore(_settings(request))


def _render(
    request: Request, name: str, context: dict[str, Any], status_code: int = 200
) -> Response:
    templates = request.app.state.templates
    response: Response = templates.TemplateResponse(request, name, context, status_code=status_code)
    return response


def _not_found(request: Request, message: str, stop_reason: str) -> Response:
    error = InvoiceAgentsError(
        ErrorCategory.DATABASE,
        sanitize_text(message),
        stop_reason=sanitize_text(stop_reason),
    )
    return _render(request, "error.html", {"nav": None, "error": error}, status_code=404)


def _artifact_binding_conflict(result: CaseResult) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "case_id": result.case_id,
            "status": result.status.value,
            "stop_reason": "RESULT_ARTIFACT_BINDING_UNRESOLVED",
        },
    )


def _artifact_binding_durability_conflict(case_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "case_id": case_id,
            "status": CaseStatus.INCOMPLETE.value,
            "stop_reason": "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED",
        },
    )


def _read_exact_bound_result_artifact(
    case_id: str,
    binding: ResultArtifactBinding,
) -> bytes | None:
    """Read only the exact no-follow file identity authorized by ``binding``."""

    directory_descriptor: int | None = None
    validation_directory_descriptor: int | None = None
    artifact_descriptor: int | None = None
    payload: bytes | None = None
    cleanup_failed = False
    target_name = f"{case_id}.json"
    expected_identity = (
        binding.artifact_device,
        binding.artifact_inode,
        binding.artifact_file_type,
        binding.artifact_size_bytes,
    )
    if not 0 < binding.artifact_size_bytes <= RESULT_ARTIFACT_MAX_BYTES:
        return None
    try:
        directory_path = Path.cwd() / "artifacts" / "results"
        classified_directory = os.lstat(directory_path)
        if not stat.S_ISDIR(classified_directory.st_mode):
            return None
        expected_directory_identity = (
            classified_directory.st_dev,
            classified_directory.st_ino,
        )
        directory_descriptor = os.open(
            directory_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        opened_directory = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or (opened_directory.st_dev, opened_directory.st_ino)
            != expected_directory_identity
        ):
            return None
        namespace_identity = os.lstat(target_name, dir_fd=directory_descriptor)
        observed_namespace = (
            namespace_identity.st_dev,
            namespace_identity.st_ino,
            stat.S_IFMT(namespace_identity.st_mode),
            namespace_identity.st_size,
        )
        if observed_namespace != expected_identity or not stat.S_ISREG(
            namespace_identity.st_mode
        ):
            return None
        artifact_descriptor = os.open(
            target_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_descriptor,
        )
        opened_identity = os.fstat(artifact_descriptor)
        observed_opened = (
            opened_identity.st_dev,
            opened_identity.st_ino,
            stat.S_IFMT(opened_identity.st_mode),
            opened_identity.st_size,
        )
        if observed_opened != expected_identity or not stat.S_ISREG(opened_identity.st_mode):
            return None
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(artifact_descriptor, 65_536)
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > binding.artifact_size_bytes:
                return None
            digest.update(chunk)
            chunks.append(chunk)
        final_opened_identity = os.fstat(artifact_descriptor)
        final_namespace_identity = os.lstat(target_name, dir_fd=directory_descriptor)
        final_opened = (
            final_opened_identity.st_dev,
            final_opened_identity.st_ino,
            stat.S_IFMT(final_opened_identity.st_mode),
            final_opened_identity.st_size,
        )
        final_namespace = (
            final_namespace_identity.st_dev,
            final_namespace_identity.st_ino,
            stat.S_IFMT(final_namespace_identity.st_mode),
            final_namespace_identity.st_size,
        )
        if (
            observed_size != binding.artifact_size_bytes
            or digest.hexdigest() != binding.artifact_sha256
            or final_opened != expected_identity
            or final_namespace != expected_identity
        ):
            return None
        validation_directory_descriptor = os.open(
            directory_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        validation_directory = os.fstat(validation_directory_descriptor)
        if (
            not stat.S_ISDIR(validation_directory.st_mode)
            or (validation_directory.st_dev, validation_directory.st_ino)
            != expected_directory_identity
        ):
            return None
        payload = b"".join(chunks)
    except OSError:
        payload = None
    finally:
        if artifact_descriptor is not None:
            try:
                os.close(artifact_descriptor)
            except OSError:
                cleanup_failed = True
        if validation_directory_descriptor is not None:
            try:
                os.close(validation_directory_descriptor)
            except OSError:
                cleanup_failed = True
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                cleanup_failed = True
    return None if cleanup_failed else payload


def _invalid_result_artifact(request: Request, case_id: str) -> Response:
    error = InvoiceAgentsError(
        ErrorCategory.DATABASE,
        "result artifact failed validation against the authoritative case",
        case_id=case_id,
        stop_reason="RESULT_ARTIFACT_INVALID",
    )
    return _render(request, "error.html", {"nav": None, "error": error}, status_code=409)


def _strict_result_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate result artifact key")
        payload[key] = value
    return payload


def _reject_result_constant(_value: str) -> object:
    raise ValueError("non-finite result artifact number")


def _finite_result_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite result artifact number")
    return parsed


def _reject_excessive_result_nesting(value: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > RESULT_ARTIFACT_MAX_DEPTH:
                raise ValueError("result artifact nesting exceeds the parser boundary")
        elif character in "}]":
            depth -= 1


def _decode_canonical_result_artifact(raw: bytes) -> CaseResult:
    text = raw.decode("utf-8", errors="strict")
    _reject_excessive_result_nesting(text)
    payload = json.loads(
        text,
        object_pairs_hook=_strict_result_object,
        parse_constant=_reject_result_constant,
        parse_float=_finite_result_float,
    )
    if type(payload) is not dict:
        raise ValueError("result artifact is not an object")
    result = CaseResult.model_validate_json(text, strict=True)
    canonical_payload = json.loads(
        result.model_dump_json(),
        object_pairs_hook=_strict_result_object,
        parse_constant=_reject_result_constant,
    )
    if payload != canonical_payload:
        raise ValueError("result artifact is not canonically encoded")
    return sanitize_case_result(result)


def _invoice_files() -> list[Path]:
    """Top-level supported files, exactly the CLI batch selection."""

    if not INVOICE_DIR.is_dir():
        return []
    return sorted(
        path
        for path in INVOICE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


# --------------------------------------------------------------------------- dashboard


@router.get("/")
async def dashboard(
    request: Request,
    status: str | None = None,
    decision: str | None = None,
    fmt: str | None = None,
    q: str | None = None,
) -> Response:
    settings = _settings(request)
    # A broken workflow DB must still render the dashboard: the preflight strip
    # carries the exact stop reason and the run actions are disabled.
    db_error: str | None = None
    rows: list[queries.CaseListRow] = []
    counts: dict[str, int] = {}
    pending_reviews = 0
    try:
        rows = queries.list_cases(
            settings.workflow_db,
            status=status or None,
            decision=decision or None,
            source_format=fmt or None,
            search=q or None,
        )
        counts = queries.status_counts(settings.workflow_db)
        pending_reviews = queries.pending_review_count(settings.workflow_db)
    except sqlite3.Error as exc:
        db_error = sanitize_text(str(exc))
    table_context = {
        "rows": rows,
        "db_error": db_error,
        "filters": {
            "status": status or "",
            "decision": decision or "",
            "fmt": fmt or "",
            "q": q or "",
        },
    }
    if request.headers.get("HX-Request"):
        return _render(request, "_case_table.html", table_context)
    context = {
        "nav": "dashboard",
        "preflight": run_preflight(settings),
        "counts": counts,
        "pending_reviews": pending_reviews,
        "statuses": [status.value for status in CaseStatus],
        "formats": list(queries.SOURCE_FORMATS),
        **table_context,
    }
    return _render(request, "dashboard.html", context)


# --------------------------------------------------------------------------- case detail


def _line_deltas(risk: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(risk, dict):
        financial = risk.get("financial")
        if isinstance(financial, dict):
            deltas = financial.get("line_deltas")
            if isinstance(deltas, dict):
                return deltas
    return {}


@router.get("/cases/{case_id}")
async def case_detail(request: Request, case_id: str) -> Response:
    settings = _settings(request)
    registry = _registry(request)
    store = _store(request)
    header = queries.case_header(settings.workflow_db, case_id)
    if header is None:
        return _not_found(request, f"case does not exist: {case_id}", "CASE_NOT_FOUND")
    result = store.load_result(case_id)
    invoice: ExtractedInvoice | None
    try:
        invoice = store.load_extraction(case_id)
    except InvoiceAgentsError:
        invoice = None
    critique: Critique | None
    try:
        critique = store.load_critique(case_id)
    except InvoiceAgentsError:
        critique = None
    risk = store.load_comparison(case_id, "risk")
    inventory_cmp = store.load_comparison(case_id, "inventory") or {}
    duplicate_case: str | None = None
    if result and result.payment and result.payment.duplicate_of:
        duplicate_case = queries.payment_case(settings.workflow_db, result.payment.duplicate_of)
    context = {
        "nav": "dashboard",
        "header": header,
        "result": result,
        "invoice": invoice,
        "identity": store.load_identity(case_id),
        "inventory_cmp": inventory_cmp,
        "risk": risk,
        "line_deltas": _line_deltas(risk),
        "critique": critique,
        "review": store.load_case_review(case_id),
        "events": queries.events_after(settings.workflow_db, case_id, 0),
        "running": registry.is_running(case_id),
        "run_state": registry.run_state(case_id),
        "run_error": registry.run_error(case_id),
        "duplicate_case": duplicate_case,
        "result_file": Path("artifacts/results") / f"{case_id}.json",
    }
    return _render(request, "case_detail.html", context)


@router.get("/cases/{case_id}/result.json")
async def case_result_json(request: Request, case_id: str) -> Response:
    if SAFE_ID.fullmatch(case_id) is None:
        return _not_found(request, f"case does not exist: {case_id}", "CASE_NOT_FOUND")
    try:
        result, generation, binding = _store(request).load_result_with_artifact_binding(case_id)
    except InvoiceAgentsError as exc:
        if exc.stop_reason == "CASE_NOT_FOUND":
            return _not_found(request, exc.message, exc.stop_reason)
        if exc.stop_reason == "RESULT_ARTIFACT_BINDING_DURABILITY_UNRESOLVED":
            return _artifact_binding_durability_conflict(case_id)
        return _invalid_result_artifact(request, case_id)
    if result is not None and result.stop_reason == "RESULT_ARTIFACT_DURABILITY_UNRESOLVED":
        return JSONResponse(
            status_code=409,
            content={
                "case_id": case_id,
                "status": result.status.value,
                "stop_reason": result.stop_reason,
            },
        )
    if result is None:
        return _not_found(
            request,
            f"no terminal result exists for case {case_id}",
            "RESULT_ARTIFACT_MISSING",
        )
    if binding is None or binding.execution_generation != generation:
        return _artifact_binding_conflict(result)
    payload = _read_exact_bound_result_artifact(case_id, binding)
    if payload is None:
        return _artifact_binding_conflict(result)
    try:
        artifact = _decode_canonical_result_artifact(payload)
    except (RecursionError, TypeError, ValueError):
        return _invalid_result_artifact(request, case_id)
    authoritative = sanitize_case_result(result)
    if artifact.case_id != case_id or artifact != authoritative:
        return _invalid_result_artifact(request, case_id)
    return Response(content=artifact.model_dump_json(), media_type="application/json")


@router.get("/cases/{case_id}/live")
async def case_live(request: Request, case_id: str) -> Response:
    settings = _settings(request)
    header = queries.case_header(settings.workflow_db, case_id)
    if header is None:
        return _not_found(request, f"case does not exist: {case_id}", "CASE_NOT_FOUND")
    context = {
        "nav": "dashboard",
        "header": header,
        "running": _registry(request).is_running(case_id),
    }
    return _render(request, "live.html", context)


@router.get("/cases/{case_id}/events")
async def case_events(request: Request, case_id: str, after: int = 0) -> EventSourceResponse:
    settings = _settings(request)
    return EventSourceResponse(
        case_event_stream(
            settings.workflow_db,
            case_id,
            _registry(request),
            after,
            settings=settings,
            recovery_coordinator=_recovery_coordinator(request),
        )
    )


@router.post("/cases/{case_id}/resume")
async def case_resume(request: Request, case_id: str) -> Response:
    settings = _settings(request)
    registry = _registry(request)
    store = _store(request)
    if registry.is_running(case_id):
        return RedirectResponse("/reviews", status_code=303)
    header = queries.case_header(settings.workflow_db, case_id)
    if header is None:
        return _not_found(request, f"case does not exist: {case_id}", "CASE_NOT_FOUND")
    result = store.load_result(case_id)
    if result is None or result.status is not CaseStatus.NEEDS_HUMAN:
        error = InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            f"case {case_id} is not waiting for human review",
            case_id=case_id,
            stop_reason="CASE_NOT_RESUMABLE",
        )
        return _render(request, "error.html", {"nav": None, "error": error}, status_code=409)
    review = store.load_case_review(case_id)
    if review is None or review.status != "RESOLVED" or review.human_decision is None:
        error = InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "a persisted human decision is required before resume",
            case_id=case_id,
            stop_reason="HUMAN_DECISION_MISSING",
        )
        return _render(request, "error.html", {"nav": None, "error": error}, status_code=409)
    await registry.start_resume(case_id, settings)
    return RedirectResponse("/reviews", status_code=303)


# --------------------------------------------------------------------------- reviews


def _unresolved_items(invoice: dict[str, Any]) -> list[str]:
    """Raw items of stored lines without a canonical SKU, in evidence order."""

    items: list[str] = []
    for line in invoice.get("lines") or []:
        if isinstance(line, dict) and not line.get("canonical_sku"):
            raw_item = str(line.get("raw_item") or "")
            if raw_item and raw_item not in items:
                items.append(raw_item)
    return items


def _review_context(request: Request, review: ReviewRequest) -> dict[str, Any]:
    settings = _settings(request)
    store = _store(request)
    bundle = review.evidence_bundle
    raw_invoice = bundle.get("invoice")
    invoice: dict[str, Any] = raw_invoice if isinstance(raw_invoice, dict) else {}
    inventory_cmp = store.load_comparison(review.case_id, "inventory") or {}
    inventory_rows: list[Any] = []
    inventory_error: str | None = None
    try:
        inventory_rows = list(queries.list_inventory(settings.inventory_db))
    except Exception as exc:
        inventory_error = sanitize_text(str(exc))
    invoice_number = None
    number_field = invoice.get("invoice_number")
    if isinstance(number_field, dict):
        invoice_number = number_field.get("normalized_value")
    raw_financial = bundle.get("financial")
    financial: dict[str, Any] | None = raw_financial if isinstance(raw_financial, dict) else None
    header = queries.case_header(settings.workflow_db, review.case_id)
    return {
        "nav": "reviews",
        "review": review,
        "case_header": header,
        "invoice": invoice,
        "financial": financial,
        "line_deltas": (financial or {}).get("line_deltas") or {},
        "inventory_list": bundle.get("inventory") or [],
        "unresolved": inventory_cmp.get("unresolved_candidates") or {},
        "identity": bundle.get("identity_candidates") or [],
        "dates": bundle.get("dates") or [],
        "suspicious": bundle.get("suspicious_signals") or [],
        "unavailable": bundle.get("unavailable_reconciliations") or [],
        "blocking_evidence": bundle.get("blocking_evidence") or [],
        "rendered_pages": bundle.get("rendered_pages") or [],
        "critique": review.critic,
        "decision_kinds": list(HumanDecisionKind),
        "blocker_authorizing_decision_kinds": AUTHORIZING_HUMAN_DECISIONS,
        "consequences": DECISION_CONSEQUENCES,
        "unresolved_items": _unresolved_items(invoice),
        "inventory_rows": inventory_rows,
        "inventory_error": inventory_error,
        "prior_cases": queries.prior_cases_for_invoice(
            settings.workflow_db, invoice_number, review.case_id
        ),
        "reviewer_prefill": unquote(request.cookies.get("ui_reviewer", "")),
        "running": _registry(request).is_running(review.case_id),
        "form_error": None,
        "form_values": {},
    }


@router.get("/reviews")
async def review_queue(request: Request, all: int = 0) -> Response:
    settings = _settings(request)
    store = _store(request)
    reviews = store.list_reviews(pending_only=not all)
    context = {
        "nav": "reviews",
        "reviews": reviews,
        "show_all": bool(all),
        "age_amber_hours": settings.review_age_amber_hours,
        "now_utc": datetime.now(UTC),
    }
    return _render(request, "reviews.html", context)


@router.get("/reviews/{review_id}")
async def review_detail(request: Request, review_id: str, decided: int = 0) -> Response:
    store = _store(request)
    try:
        review = store.load_review(review_id)
    except InvoiceAgentsError as exc:
        return _not_found(request, exc.message, exc.stop_reason or "REVIEW_NOT_FOUND")
    context = _review_context(request, review)
    context["decided"] = bool(decided)
    return _render(request, "review_detail.html", context)


@router.get("/reviews/{review_id}/pages/{page}")
async def review_page_image(request: Request, review_id: str, page: int) -> Response:
    store = _store(request)
    try:
        review = store.load_review(review_id)
    except InvoiceAgentsError as exc:
        return _not_found(request, exc.message, exc.stop_reason or "REVIEW_NOT_FOUND")
    for entry in review.evidence_bundle.get("rendered_pages") or []:
        if isinstance(entry, dict) and entry.get("page") == page:
            target = Path(str(entry.get("path")))
            if target.is_file():
                return FileResponse(target, media_type="image/png")
    return _not_found(
        request, f"no rendered page {page} for review {review_id}", "RENDER_PAGE_INVALID"
    )


@router.post("/reviews/{review_id}/decision")
async def review_decide(
    request: Request,
    review_id: str,
    reviewer: Annotated[str | None, Form()] = None,
    decision: Annotated[str | None, Form()] = None,
    reason: Annotated[str | None, Form()] = None,
    mapping_raw: Annotated[list[str] | None, Form()] = None,
    mapping_sku: Annotated[list[str] | None, Form()] = None,
    superseded_case_id: Annotated[str | None, Form()] = None,
    addressed_blocker_ids_approve: Annotated[list[str] | None, Form()] = None,
    addressed_blocker_ids_establish_mapping: Annotated[list[str] | None, Form()] = None,
    addressed_blocker_ids_supersede_revision: Annotated[list[str] | None, Form()] = None,
) -> Response:
    settings = _settings(request)
    store = _store(request)
    try:
        review = store.load_review(review_id)
    except InvoiceAgentsError as exc:
        return _not_found(request, exc.message, exc.stop_reason or "REVIEW_NOT_FOUND")

    def rerender(message: str) -> Response:
        context = _review_context(request, review)
        context["decided"] = False
        context["form_error"] = sanitize_text(message)
        context["form_values"] = {
            "reviewer": reviewer or "",
            "decision": decision or "",
            "reason": reason or "",
            "superseded_case_id": superseded_case_id or "",
            "addressed_blocker_ids_by_decision": {
                HumanDecisionKind.APPROVE: addressed_blocker_ids_approve or [],
                HumanDecisionKind.ESTABLISH_MAPPING: (
                    addressed_blocker_ids_establish_mapping or []
                ),
                HumanDecisionKind.SUPERSEDE_REVISION: (
                    addressed_blocker_ids_supersede_revision or []
                ),
            },
        }
        return _render(request, "review_detail.html", context, status_code=400)

    if not decision:
        return rerender("Select a decision; none is preselected by design.")
    try:
        selected = HumanDecisionKind(decision)
    except ValueError:
        return rerender(f"decision must be one of {[kind.value for kind in HumanDecisionKind]}")
    addressed_blocker_ids_by_decision = {
        HumanDecisionKind.APPROVE: addressed_blocker_ids_approve or [],
        HumanDecisionKind.ESTABLISH_MAPPING: addressed_blocker_ids_establish_mapping or [],
        HumanDecisionKind.SUPERSEDE_REVISION: addressed_blocker_ids_supersede_revision or [],
    }
    mappings = [
        CanonicalMapping(raw_item=raw.strip(), sku=sku.strip(), basis="human_decision")
        for raw, sku in zip(mapping_raw or [], mapping_sku or [], strict=False)
        if raw.strip() and sku.strip()
    ]
    try:
        record_human_decision(
            review_id,
            (reviewer or "").strip(),
            selected,
            (reason or "").strip(),
            store,
            settings.inventory_db,
            mappings=mappings,
            superseded_case_id=(superseded_case_id or "").strip() or None,
            addressed_blocker_ids=addressed_blocker_ids_by_decision.get(selected, []),
        )
    except InvoiceAgentsError as exc:
        return rerender(sanitize_text(str(exc)))
    response = RedirectResponse(f"/reviews/{review_id}?decided=1", status_code=303)
    if reviewer and reviewer.strip():
        # URL-encoded so addresses with @ survive the cookie layer unquoted.
        response.set_cookie(
            "ui_reviewer",
            quote(reviewer.strip()),
            max_age=180 * 24 * 3600,
            samesite="lax",
            secure=secure_cookie(request),
        )
    return response


# --------------------------------------------------------------------------- submit & batch


async def write_bounded_upload(upload: UploadFile, target: Path, max_bytes: int) -> int:
    """Stream an upload in fixed chunks and remove every incomplete target."""

    written = 0
    created = False
    try:
        with target.open("xb") as handle:
            created = True
            while chunk := await upload.read(65_536):
                written += len(chunk)
                if written > max_bytes:
                    raise SourceEvidenceError(
                        ErrorCategory.SOURCE,
                        f"invoice source exceeds the {max_bytes}-byte ceiling",
                        stop_reason="SOURCE_TOO_LARGE",
                    )
                handle.write(chunk)
        return written
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


@router.get("/submit")
async def submit_page(request: Request) -> Response:
    settings = _settings(request)
    context = {
        "nav": "submit",
        "preflight": run_preflight(settings),
        "files": _invoice_files(),
        "invoice_dir": INVOICE_DIR,
        "upload_dir": INVOICE_DIR / UPLOAD_DIR_NAME,
        "default_concurrency": settings.case_concurrency,
        "concurrency_options": list(range(1, 9)),
    }
    return _render(request, "submit.html", context)


async def _resolve_submission(
    request: Request,
    existing: list[str],
    upload: UploadFile | None,
) -> list[Path] | Response:
    if upload is not None and upload.filename:
        name = Path(upload.filename).name
        if not SAFE_ID.match(name) or Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            error = InvoiceAgentsError(
                ErrorCategory.SOURCE,
                f"unsupported or unsafe upload name: {name!r}; "
                f"supported formats are {sorted(SUPPORTED_SUFFIXES)}",
                stop_reason="SOURCE_FORMAT_UNSUPPORTED",
            )
            return _render(request, "error.html", {"nav": None, "error": error}, status_code=400)
        upload_dir = INVOICE_DIR / UPLOAD_DIR_NAME
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / name
        counter = 1
        while target.exists():
            target = upload_dir / f"{Path(name).stem}-{counter}{Path(name).suffix}"
            counter += 1
        try:
            await write_bounded_upload(upload, target, _settings(request).source_max_bytes)
        except InvoiceAgentsError as exc:
            return _render(
                request,
                "error.html",
                {"nav": None, "error": exc},
                status_code=400,
            )
        return [target]
    if existing:
        invoice_root = INVOICE_DIR.resolve()
        resolved: list[Path] = []
        # dict.fromkeys: a crafted form may repeat names; keep first occurrence order.
        for name in dict.fromkeys(existing):
            candidate = (INVOICE_DIR / Path(name).name).resolve()
            if not (candidate.is_file() and candidate.parent == invoice_root):
                error = InvoiceAgentsError(
                    ErrorCategory.SOURCE,
                    f"invoice source does not exist under {INVOICE_DIR}: {name!r}",
                    stop_reason="SOURCE_NOT_FOUND",
                )
                return _render(
                    request, "error.html", {"nav": None, "error": error}, status_code=404
                )
            resolved.append(candidate)
        return resolved
    error = InvoiceAgentsError(
        ErrorCategory.SOURCE,
        "choose at least one invoice file or upload one",
        stop_reason="SOURCE_NOT_FOUND",
    )
    return _render(request, "error.html", {"nav": None, "error": error}, status_code=400)


@router.post("/submit")
async def submit_invoice(
    request: Request,
    existing: Annotated[list[str] | None, Form()] = None,
    upload: UploadFile | None = None,
) -> Response:
    settings = _settings(request)
    registry = _registry(request)
    resolved = await _resolve_submission(request, existing or [], upload)
    if isinstance(resolved, Response):
        return resolved
    if len(resolved) > 1:
        batch = await registry.start_batch(resolved, settings, None)
        return RedirectResponse(f"/batches/{batch.batch_id}", status_code=303)
    source = resolved[0]
    running = registry.running_case_for_source(source)
    if running is not None:
        return RedirectResponse(f"/cases/{running}/live", status_code=303)
    outcome = await registry.start_process(source, settings)
    if isinstance(outcome, CaseResult):
        context = {
            "nav": "submit",
            "result": outcome,
            "source_path": source,
            "case_exists": queries.case_header(settings.workflow_db, outcome.case_id) is not None,
        }
        return _render(request, "submit_failed.html", context, status_code=422)
    return RedirectResponse(f"/cases/{outcome}/live", status_code=303)


@router.post("/batch")
async def submit_batch(
    request: Request,
    concurrency: Annotated[int | None, Form()] = None,
) -> Response:
    settings = _settings(request)
    registry = _registry(request)
    paths = _invoice_files()
    if not paths:
        error = InvoiceAgentsError(
            ErrorCategory.SOURCE,
            f"no supported invoice files found in {INVOICE_DIR}",
            stop_reason="SOURCE_NOT_FOUND",
        )
        return _render(request, "error.html", {"nav": None, "error": error}, status_code=400)
    try:
        selected_concurrency = validate_case_concurrency(concurrency, settings.case_concurrency)
    except ValueError:
        error = InvoiceAgentsError(
            ErrorCategory.CONFIGURATION,
            "batch concurrency must be a positive integer",
            stop_reason="INVALID_CONCURRENCY",
        )
        return _render(request, "error.html", {"nav": None, "error": error}, status_code=400)
    batch = await registry.start_batch(paths, settings, selected_concurrency)
    return RedirectResponse(f"/batches/{batch.batch_id}", status_code=303)


def _batch_rows_context(request: Request, batch_id: str) -> dict[str, Any] | None:
    settings = _settings(request)
    registry = _registry(request)
    batch = registry.batch(batch_id)
    if batch is None:
        return None
    rows = []
    for entry in batch.entries:
        header = queries.case_header(settings.workflow_db, entry.case_id)
        rows.append(
            {
                "entry": entry,
                "header": header,
                "run_state": registry.run_state(entry.case_id),
                "run_error": registry.run_error(entry.case_id),
            }
        )
    return {"batch": batch, "rows": rows}


@router.get("/batches/{batch_id}")
async def batch_view(request: Request, batch_id: str) -> Response:
    context = _batch_rows_context(request, batch_id)
    if context is None:
        return _not_found(request, f"batch does not exist: {batch_id}", "BATCH_NOT_FOUND")
    return _render(request, "batch.html", {"nav": "submit", **context})


@router.get("/batches/{batch_id}/rows")
async def batch_rows(request: Request, batch_id: str) -> Response:
    context = _batch_rows_context(request, batch_id)
    if context is None:
        return _not_found(request, f"batch does not exist: {batch_id}", "BATCH_NOT_FOUND")
    # HTTP 286 tells htmx to stop polling once every run has reached storage.
    batch = context["batch"]
    status_code = 200 if batch.running else 286
    return _render(request, "_batch_rows.html", context, status_code=status_code)


# --------------------------------------------------------------------------- system


@router.get("/system")
async def system_page(request: Request) -> Response:
    settings = _settings(request)
    checks: list[dict[str, Any]] = []
    for name, path, kind in (
        ("Inventory database", settings.inventory_db, DatabaseKind.INVENTORY),
        ("Workflow database", settings.workflow_db, DatabaseKind.WORKFLOW),
    ):
        try:
            checks.append(
                {
                    "name": name,
                    "info": verify_database(path, kind, settings=settings),
                    "error": None,
                }
            )
        except InvoiceAgentsError as exc:
            checks.append({"name": name, "info": None, "error": exc})
    context = {
        "nav": "system",
        "checks": checks,
        "settings": settings,
        "model": XAI_MODEL,
        "base_url": XAI_BASE_URL,
        "key_present": key_present(settings),
        "test_command": 'uv run pytest -m "not live"',
    }
    return _render(request, "system.html", context)
