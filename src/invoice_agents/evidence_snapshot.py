"""Canonical, semantically validated authorization-evidence snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from invoice_agents.models import (
    Critique,
    ExtractedInvoice,
    IdentityCandidate,
    InventoryComparison,
    InventoryStatus,
    Money,
    ReviewRequest,
    RiskAssessment,
    SourceArtifact,
)


class EvidenceSnapshotError(ValueError):
    """Stored evidence cannot form one coherent authorization snapshot."""


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    case_id: str
    invoice: ExtractedInvoice
    identity: tuple[IdentityCandidate, ...]
    inventory_payload: dict[str, Any]
    inventory: tuple[InventoryComparison, ...]
    risk: RiskAssessment
    critique: Critique
    digest: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
) -> EvidenceSnapshot:
    """Parse, validate, and digest one exact generation's latest evidence rows."""

    from invoice_agents.tools.comparison import compute_invoice_totals

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
        "inventory": raw_inventory,
        "risk": risk.model_dump(mode="json"),
        "critique": critique.model_dump(mode="json"),
    }
    digest = hashlib.sha256(_canonical_json(envelope).encode("utf-8")).hexdigest()
    return EvidenceSnapshot(
        case_id=case_id,
        invoice=invoice,
        identity=identity,
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
    if (
        review.case_id != snapshot.case_id
        or review.source != invoice.source
        or review.amount != expected_amount
        or review.critic != snapshot.critique
        or any(bundle.get(key) != value for key, value in expected_bundle.items())
    ):
        raise EvidenceSnapshotError(
            "review package does not match its bound authorization evidence snapshot"
        )
