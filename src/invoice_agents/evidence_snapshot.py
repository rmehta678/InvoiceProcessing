"""Canonical, semantically validated authorization-evidence snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from invoice_agents.config import Settings
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.models import (
    Critique,
    ExtractedInvoice,
    FinalDecision,
    HumanDecision,
    IdentityCandidate,
    InventoryComparison,
    InventoryStatus,
    Money,
    ReviewRequest,
    RiskAssessment,
    SourceArtifact,
    ToolStatus,
    critic_disagreement_reason,
)

SNAPSHOT_DIGEST_DOMAIN = b"galatiq.invoice-agents/evidence-snapshot\x00"
SNAPSHOT_DIGEST_VERSION = 1


class EvidenceSnapshotError(ValueError):
    """Stored evidence cannot form one coherent authorization snapshot."""


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    case_id: str
    invoice: ExtractedInvoice
    identity: tuple[IdentityCandidate, ...]
    identity_evaluated_at: datetime
    inventory_payload: dict[str, Any]
    inventory: tuple[InventoryComparison, ...]
    risk: RiskAssessment
    critique: Critique
    digest: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_snapshot_envelope(envelope: dict[str, Any]) -> str:
    payload = _canonical_json(envelope).encode("utf-8")
    preimage = (
        SNAPSHOT_DIGEST_DOMAIN
        + SNAPSHOT_DIGEST_VERSION.to_bytes(4, "big")
        + len(payload).to_bytes(8, "big")
        + payload
    )
    return hashlib.sha256(preimage).hexdigest()


def stored_evidence_snapshot_digest(
    case_id: object,
    extraction_json: object,
    identity_json: object,
    identity_evaluated_at: object,
    inventory_json: object,
    risk_json: object,
    critique_json: object,
) -> str:
    """Digest exact stored components for SQLite's relational provenance checks."""

    if not all(
        isinstance(value, str)
        for value in (
            case_id,
            extraction_json,
            identity_json,
            identity_evaluated_at,
            inventory_json,
            risk_json,
            critique_json,
        )
    ):
        raise ValueError("stored evidence snapshot inputs must all be text")
    assert isinstance(case_id, str)
    assert isinstance(extraction_json, str)
    assert isinstance(identity_json, str)
    assert isinstance(identity_evaluated_at, str)
    assert isinstance(inventory_json, str)
    assert isinstance(risk_json, str)
    assert isinstance(critique_json, str)
    invoice = ExtractedInvoice.model_validate_json(extraction_json)
    raw_identity = json.loads(identity_json)
    raw_inventory = json.loads(inventory_json)
    if not isinstance(raw_identity, list) or not isinstance(raw_inventory, dict):
        raise ValueError("stored identity or inventory evidence has the wrong shape")
    identity = tuple(IdentityCandidate.model_validate(item) for item in raw_identity)
    risk = RiskAssessment.model_validate_json(risk_json)
    critique = Critique.model_validate_json(critique_json)
    evaluated_at = datetime.fromisoformat(identity_evaluated_at)
    if evaluated_at.isoformat() != identity_evaluated_at:
        raise ValueError("identity evaluation boundary is not canonical")
    envelope = {
        "case_id": case_id,
        "invoice": invoice.model_dump(mode="json"),
        "identity": [item.model_dump(mode="json") for item in identity],
        "identity_evaluated_at": identity_evaluated_at,
        "inventory": raw_inventory,
        "risk": risk.model_dump(mode="json"),
        "critique": critique.model_dump(mode="json"),
    }
    return _digest_snapshot_envelope(envelope)


def stored_unresolved_blocker_count(risk_json: object, human_json: object) -> int:
    """Derive the exact remaining blocker count for SQLite authorization triggers."""

    from invoice_agents.agents.decision_rules import unaddressed_blockers

    if not isinstance(risk_json, str):
        raise ValueError("stored risk evidence must be text")
    risk = RiskAssessment.model_validate_json(risk_json)
    if human_json is None:
        human = None
    elif isinstance(human_json, str):
        human = HumanDecision.model_validate_json(human_json, strict=True)
    else:
        raise ValueError("stored human decision must be text or null")
    return len(unaddressed_blockers(risk, human))


def _without_mapping_results(invoice: ExtractedInvoice) -> ExtractedInvoice:
    source_derived = invoice.model_copy(deep=True)
    for line in source_derived.lines:
        line.candidate_skus = []
        line.canonical_sku = None
    return source_derived


def _validate_inventory_relationships(
    invoice: ExtractedInvoice, inventory: tuple[InventoryComparison, ...]
) -> None:
    invoice_names = list(dict.fromkeys(line.raw_item for line in invoice.lines))
    compared_names = [raw for comparison in inventory for raw in comparison.raw_items]
    if len(compared_names) != len(set(compared_names)) or set(compared_names) != set(invoice_names):
        raise EvidenceSnapshotError(
            "inventory comparisons do not partition the exact extraction line identities"
        )
    for comparison in inventory:
        lines = [line for line in invoice.lines if line.raw_item in comparison.raw_items]
        requested = sum((line.quantity for line in lines), start=Decimal("0"))
        expected_evidence = [reference for line in lines for reference in line.evidence]
        if requested != comparison.requested_quantity or expected_evidence != comparison.evidence:
            raise EvidenceSnapshotError(
                "inventory quantities or evidence do not derive from the exact extraction"
            )
        canonical_skus = {line.canonical_sku for line in lines if line.canonical_sku is not None}
        if canonical_skus and any(line.canonical_sku is None for line in lines):
            raise EvidenceSnapshotError(
                "inventory lines with the same identity have inconsistent SKU mappings"
            )
        if canonical_skus and canonical_skus != {comparison.sku}:
            raise EvidenceSnapshotError(
                "inventory SKU mapping does not derive from the exact extraction"
            )
        if requested <= 0:
            if (
                comparison.status is not InventoryStatus.INVALID_QUANTITY
                or comparison.available_stock is not None
                or comparison.queried_row is not None
            ):
                raise EvidenceSnapshotError("nonpositive inventory quantity is not marked invalid")
            continue
        if comparison.queried_row is not None:
            row = comparison.queried_row
            if (
                comparison.sku != row.sku
                or comparison.available_stock != row.available_stock
                or (
                    not canonical_skus and any(raw != row.item_name for raw in comparison.raw_items)
                )
            ):
                raise EvidenceSnapshotError(
                    "inventory comparison does not match its exact queried row"
                )
            expected_status = (
                InventoryStatus.OUT_OF_STOCK
                if row.available_stock == 0
                else InventoryStatus.EXCEEDS_STOCK
                if requested > row.available_stock
                else InventoryStatus.AVAILABLE
            )
            if comparison.status is not expected_status:
                raise EvidenceSnapshotError(
                    "inventory status does not derive from quantity and queried stock"
                )
        elif (
            comparison.available_stock is not None
            or (comparison.sku is not None and comparison.status is not InventoryStatus.ERROR)
            or (
                comparison.sku is None
                and comparison.status
                not in {
                    InventoryStatus.UNKNOWN,
                    InventoryStatus.AMBIGUOUS,
                    InventoryStatus.ERROR,
                }
            )
        ):
            raise EvidenceSnapshotError(
                "inventory status or stock is unsupported without an exact queried row"
            )


def build_evidence_snapshot(
    case_id: str,
    authoritative_source: SourceArtifact,
    extraction_json: str,
    identity_json: str,
    inventory_json: str,
    risk_json: str,
    critique_json: str,
    *,
    identity_evaluated_at: datetime,
    authoritative_identity: tuple[IdentityCandidate, ...],
    settings: Settings,
    excluded_alias_sources: frozenset[str] = frozenset(),
    inventory_connection: sqlite3.Connection | None = None,
    inventory_schema: str = "main",
) -> EvidenceSnapshot:
    """Parse, validate, and digest one exact generation's latest evidence rows."""

    from invoice_agents.tools.comparison import (
        InventoryReader,
        apply_mapping_evidence,
        build_risk_assessment,
        compare_inventory_evidence,
        compute_invoice_totals,
    )
    from invoice_agents.tools.evidence import extract_invoice_evidence

    try:
        invoice = ExtractedInvoice.model_validate_json(extraction_json)
        raw_identity = json.loads(identity_json)
        raw_inventory = json.loads(inventory_json)
        risk = RiskAssessment.model_validate_json(risk_json)
        critique = Critique.model_validate_json(critique_json)
        if not isinstance(raw_identity, list) or not isinstance(raw_inventory, dict):
            raise ValueError("identity or inventory payload has the wrong shape")
        raw_comparisons = raw_inventory.get("comparisons")
        if not isinstance(raw_comparisons, list):
            raise ValueError("inventory payload has no comparisons list")
        identity = tuple(IdentityCandidate.model_validate(item) for item in raw_identity)
        inventory = tuple(InventoryComparison.model_validate(item) for item in raw_comparisons)
    except (TypeError, ValueError) as exc:
        raise EvidenceSnapshotError(f"evidence payload is invalid: {exc}") from exc
    if invoice.source != authoritative_source:
        raise EvidenceSnapshotError(
            "embedded extraction source metadata does not match source_artifacts"
        )
    try:
        source_invoice = extract_invoice_evidence(authoritative_source)
    except InvoiceAgentsError as exc:
        raise EvidenceSnapshotError(f"authoritative source cannot be re-extracted: {exc}") from exc
    if _without_mapping_results(invoice) != source_invoice:
        raise EvidenceSnapshotError(
            "stored extraction facts or evidence references do not derive from archived source bytes"
        )
    if identity != authoritative_identity:
        raise EvidenceSnapshotError(
            "identity candidates do not match the authoritative evaluated case scope"
        )
    reader = InventoryReader(
        settings.inventory_db,
        excluded_alias_sources=excluded_alias_sources,
        connection=inventory_connection,
        schema=inventory_schema,
    )
    mappings, expected_inventory, unresolved = compare_inventory_evidence(source_invoice, reader)
    inventory_errors = [
        result.error for result in unresolved.values() if result.status is ToolStatus.ERROR
    ]
    if inventory_errors:
        raise EvidenceSnapshotError(
            f"authoritative inventory cannot be re-read: {inventory_errors}"
        )
    expected_invoice = apply_mapping_evidence(source_invoice, mappings, unresolved)
    if invoice != expected_invoice:
        raise EvidenceSnapshotError(
            "stored mapping outcomes do not derive from source and authoritative inventory"
        )
    expected_inventory_payload = {
        "comparisons": [item.model_dump(mode="json") for item in expected_inventory],
        "unresolved_candidates": {
            item: result.model_dump(mode="json") for item, result in unresolved.items()
        },
    }
    if raw_inventory != expected_inventory_payload or inventory != tuple(expected_inventory):
        raise EvidenceSnapshotError(
            "inventory evidence does not derive from the configured authoritative database"
        )
    expected_risk = build_risk_assessment(
        invoice,
        expected_inventory,
        list(authoritative_identity),
        compute_invoice_totals(invoice),
        settings,
    )
    if risk != expected_risk:
        raise EvidenceSnapshotError(
            "risk policy, dates, identity, inventory, or financial facts were not re-derived"
        )
    if list(identity) != risk.identity_candidates or list(inventory) != risk.inventory:
        raise EvidenceSnapshotError(
            "risk identity or inventory evidence does not match its component rows"
        )
    if compute_invoice_totals(invoice) != risk.financial:
        raise EvidenceSnapshotError(
            "risk financial evidence does not derive from the exact extraction"
        )
    _validate_inventory_relationships(invoice, inventory)
    envelope = {
        "case_id": case_id,
        "invoice": invoice.model_dump(mode="json"),
        "identity": [item.model_dump(mode="json") for item in identity],
        "identity_evaluated_at": identity_evaluated_at.isoformat(),
        "inventory": raw_inventory,
        "risk": risk.model_dump(mode="json"),
        "critique": critique.model_dump(mode="json"),
    }
    digest = _digest_snapshot_envelope(envelope)
    return EvidenceSnapshot(
        case_id=case_id,
        invoice=invoice,
        identity=identity,
        identity_evaluated_at=identity_evaluated_at,
        inventory_payload=raw_inventory,
        inventory=inventory,
        risk=risk,
        critique=critique,
        digest=digest,
    )


def validate_review_snapshot(review: ReviewRequest, snapshot: EvidenceSnapshot) -> None:
    """Require a review package to describe exactly the snapshot its digest binds."""

    from invoice_agents.agents.decision_rules import blocking_evidence

    invoice = snapshot.invoice
    expected_amount = (
        Money(amount=invoice.declared_total, currency=invoice.currency.normalized_value)
        if invoice.declared_total is not None and invoice.currency.normalized_value is not None
        else None
    )
    bundle = review.evidence_bundle
    expected_bundle = {
        "invoice": invoice.model_dump(mode="json"),
        "financial": snapshot.risk.financial.model_dump(mode="json"),
        "inventory": [item.model_dump(mode="json") for item in snapshot.inventory],
        "identity_candidates": [item.model_dump(mode="json") for item in snapshot.identity],
        "dates": [item.model_dump(mode="json") for item in snapshot.risk.dates],
        "suspicious_signals": snapshot.risk.suspicious_signals,
        "unavailable_reconciliations": snapshot.risk.unavailable_reconciliations,
        "blocking_evidence": [
            item.model_dump(mode="json") for item in blocking_evidence(snapshot.risk)
        ],
    }
    expected_keys = {*expected_bundle, "rendered_pages"}
    policy_reasons = snapshot.risk.policy_review_reasons
    missing_policy_reasons = [reason for reason in policy_reasons if reason not in review.reasons]
    expected_disagreement = critic_disagreement_reason(
        review.agent_recommendation,
        snapshot.critique.recommended_disposition,
    )
    raw_pages = bundle.get("rendered_pages")
    expected_page_numbers = (
        list(
            range(
                1,
                (invoice.source.page_count or 1) + 1,
            )
        )
        if invoice.source.source_format == "pdf" and (invoice.source.page_count or 1) <= 3
        else [1]
        if invoice.source.source_format == "pdf"
        else []
    )
    if isinstance(raw_pages, list) and len(raw_pages) == len(expected_page_numbers):
        rendered_pages_valid = True
        for raw_page, page_number in zip(raw_pages, expected_page_numbers, strict=True):
            if not isinstance(raw_page, dict):
                rendered_pages_valid = False
                break
            path = raw_page.get("path")
            rendered_path = Path(path) if isinstance(path, str) else None
            if (
                set(raw_page) != {"path", "page", "sha256", "renderer"}
                or type(raw_page.get("page")) is not int
                or raw_page.get("page") != page_number
                or not isinstance(raw_page.get("renderer"), str)
                or not str(raw_page.get("renderer")).strip()
                or not isinstance(raw_page.get("sha256"), str)
                or len(str(raw_page.get("sha256"))) != 64
                or any(
                    character not in "0123456789abcdef" for character in str(raw_page.get("sha256"))
                )
                or rendered_path is None
                or not rendered_path.is_absolute()
                or rendered_path.name != f"{invoice.source.source_id}-page-{page_number}.png"
                or rendered_path.parent.name != review.review_id
                or rendered_path.parent.parent.name != "reviews"
            ):
                rendered_pages_valid = False
                break
    else:
        rendered_pages_valid = False
    if (
        review.case_id != snapshot.case_id
        or review.source != invoice.source
        or review.amount != expected_amount
        or review.critic != snapshot.critique
        or missing_policy_reasons
        or review.critic_disagreement_reason != expected_disagreement
        or set(bundle) != expected_keys
        or not rendered_pages_valid
        or any(bundle.get(key) != value for key, value in expected_bundle.items())
    ):
        raise EvidenceSnapshotError(
            "review package does not match its bound authorization evidence snapshot"
        )


def validate_final_decision_snapshot(
    decision: FinalDecision,
    snapshot: EvidenceSnapshot,
    review: ReviewRequest | None,
) -> None:
    """Require every final payload field that cites evidence/state to match its anchor."""

    expected_evidence = [
        reference for line in snapshot.invoice.lines for reference in line.evidence[:1]
    ]
    human = review.human_decision if review is not None else None
    if (
        any(not reason.strip() for reason in decision.reasons)
        or decision.evidence != expected_evidence
        or decision.critic_disposition is not snapshot.critique.recommended_disposition
        or decision.human_outcome != human
    ):
        raise EvidenceSnapshotError(
            "final reasons, evidence, critique, or human outcome do not match the anchor"
        )
