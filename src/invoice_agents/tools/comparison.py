"""Exact SQLite, identity, quantity, money, date, and policy evidence tools."""

from __future__ import annotations

import difflib
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dateutil import parser as date_parser

from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import WorkflowStore
from invoice_agents.models import (
    AggregatedQuantity,
    CanonicalMapping,
    DateAssessment,
    ExtractedInvoice,
    FinancialComparison,
    IdentityCandidate,
    IdentityRelationship,
    InventoryComparison,
    InventoryLookupResult,
    InventoryRow,
    InventoryStatus,
    RiskAssessment,
    RiskPolicy,
    ToolStatus,
)


def normalize_alias(value: str) -> str:
    """Normalize alias keys for explicit lookup, never for implicit acceptance."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


class InventoryReader:
    """Read-only, parameterized access to the authoritative inventory database."""

    def __init__(
        self,
        path: Path,
        *,
        excluded_alias_sources: frozenset[str] = frozenset(),
    ) -> None:
        self.path = path.resolve()
        self.excluded_alias_sources = excluded_alias_sources

    @staticmethod
    def _row(row: sqlite3.Row) -> InventoryRow:
        return InventoryRow(
            sku=str(row["sku"]),
            item_name=str(row["item_name"]),
            available_stock=int(row["available_stock"]),
        )

    def lookup_inventory_exact(self, item_name: str) -> InventoryLookupResult:
        """Look up an exact authoritative item name; SQL errors remain ERROR."""

        if not item_name.strip():
            return InventoryLookupResult(
                status=ToolStatus.INVALID_INPUT,
                query=item_name,
                error="item_name is empty",
            )
        try:
            with connect_database(self.path, read_only=True) as connection:
                rows = connection.execute(
                    "SELECT sku, item_name, available_stock FROM inventory WHERE item_name = ?",
                    (item_name.strip(),),
                ).fetchall()
        except sqlite3.Error as exc:
            return InventoryLookupResult(
                status=ToolStatus.ERROR,
                query=item_name,
                error=f"SQLite exact lookup failed: {exc}",
            )
        if not rows:
            return InventoryLookupResult(status=ToolStatus.NOT_FOUND, query=item_name)
        if len(rows) > 1:
            return InventoryLookupResult(
                status=ToolStatus.AMBIGUOUS,
                query=item_name,
                candidates=[self._row(row) for row in rows],
            )
        return InventoryLookupResult(status=ToolStatus.OK, query=item_name, row=self._row(rows[0]))

    def lookup_item_alias(self, alias: str) -> InventoryLookupResult:
        """Resolve only a persisted, human-approved alias and return its provenance."""

        normalized = normalize_alias(alias)
        if not normalized:
            return InventoryLookupResult(
                status=ToolStatus.INVALID_INPUT,
                query=alias,
                error="alias is empty after normalization",
            )
        try:
            with connect_database(self.path, read_only=True) as connection:
                rows = connection.execute(
                    "SELECT i.sku, i.item_name, i.available_stock, a.source, a.approved_by, "
                    "a.approved_at FROM item_aliases a JOIN inventory i ON i.sku = a.sku "
                    "WHERE a.alias_normalized = ?",
                    (normalized,),
                ).fetchall()
        except sqlite3.Error as exc:
            return InventoryLookupResult(
                status=ToolStatus.ERROR,
                query=alias,
                error=f"SQLite alias lookup failed: {exc}",
            )
        rows = [
            row for row in rows if str(row["source"]) not in self.excluded_alias_sources
        ]
        if not rows:
            return InventoryLookupResult(status=ToolStatus.NOT_FOUND, query=alias)
        if len(rows) > 1:
            return InventoryLookupResult(
                status=ToolStatus.AMBIGUOUS,
                query=alias,
                candidates=[self._row(row) for row in rows],
            )
        row = rows[0]
        return InventoryLookupResult(
            status=ToolStatus.OK,
            query=alias,
            row=self._row(row),
            alias_provenance={
                "source": str(row["source"]),
                "approved_by": str(row["approved_by"]),
                "approved_at": str(row["approved_at"]),
            },
        )

    def search_inventory_candidates(self, raw_item: str, limit: int = 5) -> InventoryLookupResult:
        """Surface fuzzy candidates for review without accepting a mapping."""

        if not raw_item.strip() or limit < 1:
            return InventoryLookupResult(
                status=ToolStatus.INVALID_INPUT,
                query=raw_item,
                error="raw_item must be non-empty and limit positive",
            )
        try:
            with connect_database(self.path, read_only=True) as connection:
                rows = connection.execute(
                    "SELECT sku, item_name, available_stock FROM inventory ORDER BY item_name"
                ).fetchall()
        except sqlite3.Error as exc:
            return InventoryLookupResult(
                status=ToolStatus.ERROR,
                query=raw_item,
                error=f"SQLite candidate search failed: {exc}",
            )
        normalized_query = normalize_alias(raw_item)
        scored = sorted(
            rows,
            key=lambda row: difflib.SequenceMatcher(
                None, normalized_query, normalize_alias(str(row["item_name"]))
            ).ratio(),
            reverse=True,
        )
        candidates = [
            self._row(row)
            for row in scored[:limit]
            if difflib.SequenceMatcher(
                None, normalized_query, normalize_alias(str(row["item_name"]))
            ).ratio()
            >= 0.45
        ]
        return InventoryLookupResult(
            status=ToolStatus.AMBIGUOUS if candidates else ToolStatus.NOT_FOUND,
            query=raw_item,
            candidates=candidates,
        )

    def row_by_sku(self, sku: str) -> InventoryLookupResult:
        try:
            with connect_database(self.path, read_only=True) as connection:
                rows = connection.execute(
                    "SELECT sku, item_name, available_stock FROM inventory WHERE sku = ?", (sku,)
                ).fetchall()
        except sqlite3.Error as exc:
            return InventoryLookupResult(
                status=ToolStatus.ERROR,
                query=sku,
                error=f"SQLite SKU lookup failed: {exc}",
            )
        return (
            InventoryLookupResult(status=ToolStatus.OK, query=sku, row=self._row(rows[0]))
            if rows
            else InventoryLookupResult(status=ToolStatus.NOT_FOUND, query=sku)
        )


def resolve_mappings(
    invoice: ExtractedInvoice, reader: InventoryReader
) -> tuple[list[CanonicalMapping], dict[str, InventoryLookupResult]]:
    """Accept exact item names and approved aliases; candidates remain unresolved."""

    mappings: list[CanonicalMapping] = []
    unresolved: dict[str, InventoryLookupResult] = {}
    for raw_item in dict.fromkeys(line.raw_item for line in invoice.lines):
        exact = reader.lookup_inventory_exact(raw_item)
        if exact.status is ToolStatus.ERROR:
            unresolved[raw_item] = exact
            continue
        if exact.status is ToolStatus.OK and exact.row:
            mappings.append(
                CanonicalMapping(
                    raw_item=raw_item,
                    sku=exact.row.sku,
                    basis="exact_item_name",
                    evidence=[],
                )
            )
            continue
        alias = reader.lookup_item_alias(raw_item)
        if alias.status is ToolStatus.ERROR:
            unresolved[raw_item] = alias
            continue
        if alias.status is ToolStatus.OK and alias.row:
            mappings.append(
                CanonicalMapping(
                    raw_item=raw_item,
                    sku=alias.row.sku,
                    basis="approved_alias",
                    evidence=[],
                )
            )
            continue
        unresolved[raw_item] = reader.search_inventory_candidates(raw_item)
    return mappings, unresolved


def aggregate_quantities(
    invoice: ExtractedInvoice,
    mappings: Iterable[CanonicalMapping],
) -> list[AggregatedQuantity]:
    """Aggregate every repeated line after—and only after—explicit canonical mapping."""

    mapping_by_raw = {mapping.raw_item: mapping for mapping in mappings}
    quantities: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    raw_names: dict[str, list[str]] = defaultdict(list)
    bases: dict[str, list[str]] = defaultdict(list)
    for line in invoice.lines:
        mapping = mapping_by_raw.get(line.raw_item)
        key = mapping.sku if mapping else f"UNRESOLVED:{line.raw_item}"
        quantities[key] += line.quantity
        if line.raw_item not in raw_names[key]:
            raw_names[key].append(line.raw_item)
        if mapping and mapping.basis not in bases[key]:
            bases[key].append(mapping.basis)
    return [
        AggregatedQuantity(
            sku=None if key.startswith("UNRESOLVED:") else key,
            raw_items=raw_names[key],
            requested_quantity=quantity,
            mapping_basis=bases[key],
        )
        for key, quantity in quantities.items()
    ]


def compare_inventory_evidence(
    invoice: ExtractedInvoice,
    reader: InventoryReader,
) -> tuple[list[CanonicalMapping], list[InventoryComparison], dict[str, InventoryLookupResult]]:
    """Resolve, aggregate, and compare, returning the explicit mappings as evidence."""

    mappings, unresolved = resolve_mappings(invoice, reader)
    aggregates = aggregate_quantities(invoice, mappings)
    comparisons: list[InventoryComparison] = []
    for aggregate in aggregates:
        evidence = [
            ref
            for line in invoice.lines
            if line.raw_item in aggregate.raw_items
            for ref in line.evidence
        ]
        if aggregate.requested_quantity <= 0:
            comparisons.append(
                InventoryComparison(
                    sku=aggregate.sku,
                    raw_items=aggregate.raw_items,
                    requested_quantity=aggregate.requested_quantity,
                    available_stock=None,
                    status=InventoryStatus.INVALID_QUANTITY,
                    evidence=evidence,
                    explanation="aggregate requested quantity must be greater than zero",
                )
            )
            continue
        if aggregate.sku is None:
            lookup = unresolved[aggregate.raw_items[0]]
            raw_normalized = normalize_alias(aggregate.raw_items[0])
            structurally_plausible = any(
                raw_normalized == normalize_alias(candidate.item_name)
                or raw_normalized.startswith(normalize_alias(candidate.item_name))
                for candidate in lookup.candidates
            )
            if lookup.status is ToolStatus.ERROR:
                status = InventoryStatus.ERROR
            elif structurally_plausible:
                status = InventoryStatus.AMBIGUOUS
            else:
                status = InventoryStatus.UNKNOWN
            comparisons.append(
                InventoryComparison(
                    sku=None,
                    raw_items=aggregate.raw_items,
                    requested_quantity=aggregate.requested_quantity,
                    available_stock=None,
                    status=status,
                    evidence=evidence,
                    explanation=lookup.error
                    or "no exact item or approved alias establishes a canonical SKU",
                )
            )
            continue
        lookup = reader.row_by_sku(aggregate.sku)
        if lookup.status is ToolStatus.ERROR or lookup.row is None:
            comparisons.append(
                InventoryComparison(
                    sku=aggregate.sku,
                    raw_items=aggregate.raw_items,
                    requested_quantity=aggregate.requested_quantity,
                    available_stock=None,
                    status=InventoryStatus.ERROR,
                    evidence=evidence,
                    explanation=lookup.error or "mapped SKU is absent from authoritative inventory",
                )
            )
            continue
        stock = lookup.row.available_stock
        if stock == 0:
            status = InventoryStatus.OUT_OF_STOCK
        elif aggregate.requested_quantity > stock:
            status = InventoryStatus.EXCEEDS_STOCK
        else:
            status = InventoryStatus.AVAILABLE
        comparisons.append(
            InventoryComparison(
                sku=aggregate.sku,
                raw_items=aggregate.raw_items,
                requested_quantity=aggregate.requested_quantity,
                available_stock=stock,
                status=status,
                queried_row=lookup.row,
                evidence=evidence,
                explanation=(
                    f"aggregate quantity {aggregate.requested_quantity} compared with exact row "
                    f"{lookup.row.model_dump()}"
                ),
            )
        )
    return mappings, comparisons, unresolved


def compare_inventory(
    invoice: ExtractedInvoice,
    reader: InventoryReader,
) -> tuple[list[InventoryComparison], dict[str, InventoryLookupResult]]:
    """Resolve, aggregate, then compare requested quantities with exact queried rows."""

    _, comparisons, unresolved = compare_inventory_evidence(invoice, reader)
    return comparisons, unresolved


def apply_mapping_evidence(
    invoice: ExtractedInvoice,
    mappings: Iterable[CanonicalMapping],
    unresolved: Mapping[str, InventoryLookupResult],
) -> ExtractedInvoice:
    """Carry explicit mapping outcomes onto a new invoice version.

    Only explicit bases (exact_item_name, approved_alias, human_decision) may populate
    canonical_sku; unresolved lines keep canonical_sku=None and expose the fuzzy
    candidates for review. The caller persists the returned copy as a new extraction
    version so the raw v1 evidence remains immutable.
    """

    mapping_by_raw = {mapping.raw_item: mapping for mapping in mappings}
    updated = invoice.model_copy(deep=True)
    for line in updated.lines:
        mapping = mapping_by_raw.get(line.raw_item)
        if mapping is not None:
            line.canonical_sku = mapping.sku
            line.candidate_skus = []
            continue
        line.canonical_sku = None
        lookup = unresolved.get(line.raw_item)
        line.candidate_skus = (
            [candidate.sku for candidate in lookup.candidates] if lookup is not None else []
        )
    return updated


def recompute_line_extension(quantity: str, unit_price: str) -> dict[str, str]:
    """Multiply exact Decimals for the critic; invalid input raises, never guesses."""

    try:
        parsed_quantity = Decimal(quantity.strip())
        parsed_price = Decimal(unit_price.strip())
    except InvalidOperation as exc:
        raise ValueError(
            f"quantity {quantity!r} and unit_price {unit_price!r} must be exact decimal strings"
        ) from exc
    return {
        "quantity": str(parsed_quantity),
        "unit_price": str(parsed_price),
        "extended_total": str(parsed_quantity * parsed_price),
    }


def compute_invoice_totals(invoice: ExtractedInvoice) -> FinancialComparison:
    """Recompute exact Decimal line extensions, subtotal, tax when possible, fees, and total."""

    calculated_subtotal = sum(
        (line.calculated_line_total for line in invoice.lines), start=Decimal("0")
    )
    line_deltas = {
        line.line_id: line.calculated_line_total - line.declared_line_total
        for line in invoice.lines
        if line.declared_line_total is not None
    }
    if invoice.declared_tax_rate is not None:
        calculated_tax = calculated_subtotal * invoice.declared_tax_rate
        tax_recomputable = True
        tax_basis = "calculated subtotal multiplied by declared tax rate"
    elif invoice.declared_tax_amount is None:
        calculated_tax = Decimal("0")
        tax_recomputable = True
        tax_basis = (
            "no tax rate or amount declared; treated as zero with missing-tax evidence retained"
        )
    else:
        calculated_tax = invoice.declared_tax_amount
        tax_recomputable = False
        tax_basis = "declared tax amount carried into total because source provides no tax rate"
    calculated_total = calculated_subtotal + calculated_tax + invoice.declared_fees
    subtotal_delta = (
        calculated_subtotal - invoice.declared_subtotal
        if invoice.declared_subtotal is not None
        else None
    )
    tax_delta = (
        calculated_tax - invoice.declared_tax_amount
        if invoice.declared_tax_amount is not None and tax_recomputable
        else None
    )
    total_delta = (
        calculated_total - invoice.declared_total if invoice.declared_total is not None else None
    )
    exact = (
        all(delta == 0 for delta in line_deltas.values())
        and subtotal_delta in (None, Decimal("0"))
        and tax_delta in (None, Decimal("0"))
        and total_delta in (None, Decimal("0"))
        and tax_recomputable
    )
    return FinancialComparison(
        calculated_subtotal=calculated_subtotal,
        declared_subtotal=invoice.declared_subtotal,
        subtotal_delta=subtotal_delta,
        calculated_tax=calculated_tax,
        declared_tax=invoice.declared_tax_amount,
        tax_delta=tax_delta,
        tax_recomputable=tax_recomputable,
        tax_basis=tax_basis,
        calculated_fees=invoice.declared_fees,
        calculated_total=calculated_total,
        declared_total=invoice.declared_total,
        total_delta=total_delta,
        line_deltas=line_deltas,
        exact=exact,
    )


def assess_date(field: str, raw_value: str | None, normalized: str | None) -> DateAssessment:
    if not raw_value:
        return DateAssessment(
            field=field,
            raw_value=raw_value,
            parsed_date=None,
            status="MISSING",
            explanation="date is missing",
        )
    if re.search(r"\b(today|tomorrow|yesterday|next|last)\b", raw_value, re.I):
        return DateAssessment(
            field=field,
            raw_value=raw_value,
            parsed_date=None,
            status="RELATIVE",
            explanation="relative date lacks an explicit source reference date",
        )
    if normalized is None:
        return DateAssessment(
            field=field,
            raw_value=raw_value,
            parsed_date=None,
            status="INVALID",
            explanation="normalization did not produce an ISO date",
        )
    try:
        parsed = date_parser.isoparse(normalized).date()
    except (ValueError, OverflowError):
        return DateAssessment(
            field=field,
            raw_value=raw_value,
            parsed_date=None,
            status="INVALID",
            explanation="normalized value is not an ISO date",
        )
    ambiguous = "O" in raw_value.upper() and any(char.isdigit() for char in raw_value)
    return DateAssessment(
        field=field,
        raw_value=raw_value,
        parsed_date=parsed,
        status="AMBIGUOUS" if ambiguous else "EXACT",
        explanation="OCR-like character normalization requires review"
        if ambiguous
        else "parsed exactly",
    )


def classify_identity_candidate(
    candidate_case_id: str,
    invoice: ExtractedInvoice,
    prior: ExtractedInvoice,
) -> IdentityCandidate:
    """Classify one authoritative prior extraction against the current invoice."""

    same_hash = prior.source.sha256 == invoice.source.sha256
    same_number = (
        prior.invoice_number.normalized_value == invoice.invoice_number.normalized_value
    )
    same_vendor = prior.vendor.normalized_value == invoice.vendor.normalized_value
    prior_revision = prior.revision.normalized_value if prior.revision else None
    current_revision = invoice.revision.normalized_value if invoice.revision else None
    if same_hash:
        relationship = IdentityRelationship.EXACT_ARTIFACT
        explanation = "source SHA-256 matches a prior artifact"
    elif (
        same_number
        and same_vendor
        and prior_revision != current_revision
        and (prior_revision or current_revision)
    ):
        relationship = IdentityRelationship.POSSIBLE_REVISION
        explanation = "invoice/vendor match while revision evidence differs"
    elif same_number and same_vendor:
        relationship = (
            IdentityRelationship.DUPLICATE_REPRESENTATION
            if prior.declared_total == invoice.declared_total
            else IdentityRelationship.CONFLICT
        )
        explanation = (
            "invoice/vendor/amount match across distinct source representations"
            if relationship is IdentityRelationship.DUPLICATE_REPRESENTATION
            else "invoice/vendor match but declared amounts conflict"
        )
    else:
        relationship = IdentityRelationship.CONFLICT
        explanation = "partial identity match has conflicting invoice or vendor evidence"
    return IdentityCandidate(
        case_id=candidate_case_id,
        source_id=prior.source.source_id,
        invoice_number=prior.invoice_number.normalized_value,
        vendor=prior.vendor.normalized_value,
        source_hash=prior.source.sha256,
        revision=prior_revision,
        source_format=prior.source.source_format,
        relationship=relationship,
        explanation=explanation,
    )


def find_prior_invoice_candidates(
    case_id: str,
    invoice: ExtractedInvoice,
    store: WorkflowStore,
) -> list[IdentityCandidate]:
    """Classify prior artifact/representation/revision candidates without picking a winner."""

    candidates: list[IdentityCandidate] = []
    rows = store.identity_rows(
        case_id, invoice.invoice_number.normalized_value, invoice.vendor.normalized_value
    )
    for row in rows:
        prior = store.load_extraction(str(row["case_id"]))
        candidates.append(classify_identity_candidate(str(row["case_id"]), invoice, prior))
    return candidates


def build_risk_assessment(
    invoice: ExtractedInvoice,
    inventory: list[InventoryComparison],
    identity: list[IdentityCandidate],
    financial: FinancialComparison,
    settings: Settings,
) -> RiskAssessment:
    """Apply explicit review policy to evidence without synthesizing an approval."""

    policy = RiskPolicy(
        review_threshold_amount=settings.review_threshold_amount,
        review_threshold_currency=settings.review_threshold_currency,
        review_threshold_effective_date=settings.review_threshold_effective_date,
        due_date_tolerance_days=settings.due_date_tolerance_days,
    )
    dates = [
        assess_date(
            "invoice_date", invoice.invoice_date.raw_value, invoice.invoice_date.normalized_value
        ),
        assess_date("due_date", invoice.due_date.raw_value, invoice.due_date.normalized_value),
    ]
    suspicious: list[str] = []
    combined_notes = " ".join(invoice.extraction_notes)
    if re.search(r"urgent|wire|avoid penalties|immediate", combined_notes, re.I):
        suspicious.append("urgent, immediate, penalty, or wire-payment language appears in source")
    if any(line.quantity <= 0 for line in invoice.lines):
        suspicious.append("one or more quantities are zero or negative")
    review: list[str] = []
    currency = invoice.currency.normalized_value
    amount = invoice.declared_total
    if currency != policy.review_threshold_currency:
        review.append(
            f"currency {currency or '<missing>'} has no approved FX/threshold policy against "
            f"{policy.review_threshold_currency}"
        )
    elif amount is not None and amount >= policy.review_threshold_amount:
        review.append(
            f"declared amount {amount} {currency} is at or above policy threshold "
            f"{policy.review_threshold_amount} {policy.review_threshold_currency} effective "
            f"{policy.review_threshold_effective_date.isoformat()}"
        )
    if invoice.missing_fields:
        review.append(f"required fields are missing: {', '.join(invoice.missing_fields)}")
    if invoice.conflicts:
        review.append(f"extraction conflicts exist: {', '.join(invoice.conflicts)}")
    if invoice.currency.ambiguity:
        review.append(invoice.currency.ambiguity)
    ambiguous_notes = sorted(
        {
            note
            for note in invoice.extraction_notes
            if re.search(
                r"ambigu|missing|OCR|relative|urgent|wire|requires review|conflict",
                note,
                re.I,
            )
        }
    )
    if ambiguous_notes:
        review.extend(f"extraction note: {note}" for note in ambiguous_notes)
    inventory_exceptions = [
        item for item in inventory if item.status is not InventoryStatus.AVAILABLE
    ]
    if inventory_exceptions:
        review.extend(
            f"inventory {item.status}: {item.raw_items} requested={item.requested_quantity} "
            f"stock={item.available_stock}"
            for item in inventory_exceptions
        )
    if not financial.exact:
        review.append(
            "financial evidence is incomplete or conflicts: "
            f"subtotal_delta={financial.subtotal_delta}, tax_delta={financial.tax_delta}, "
            f"total_delta={financial.total_delta}, line_deltas={financial.line_deltas}, "
            f"tax_recomputable={financial.tax_recomputable}"
        )
    if identity:
        review.extend(
            f"identity {candidate.relationship}: prior case {candidate.case_id}"
            for candidate in identity
        )
    review.extend(
        f"{assessment.field} is {assessment.status}: {assessment.explanation}"
        for assessment in dates
        if assessment.status != "EXACT"
    )
    if (
        dates[0].parsed_date
        and dates[1].parsed_date
        and dates[1].parsed_date <= dates[0].parsed_date
    ):
        review.append("due date is on or before invoice date")
    # Terms consistency is configured policy (§9), never a tolerance embedded in a prompt.
    net_terms = (
        re.search(r"\bnet\s*(\d{1,4})\b", invoice.payment_terms.normalized_value, re.I)
        if invoice.payment_terms.normalized_value
        else None
    )
    if net_terms and dates[0].parsed_date and dates[1].parsed_date:
        expected_due = dates[0].parsed_date + timedelta(days=int(net_terms.group(1)))
        delta_days = (dates[1].parsed_date - expected_due).days
        if abs(delta_days) > policy.due_date_tolerance_days:
            review.append(
                f"stated due date {dates[1].parsed_date.isoformat()} deviates from the "
                f"Net {int(net_terms.group(1))} expectation {expected_due.isoformat()} by "
                f"{delta_days:+d} days, beyond the configured tolerance of "
                f"{policy.due_date_tolerance_days} days"
            )
    review.extend(suspicious)
    # Database scope is stated every time so no agent can imply unsupported reconciliation.
    unavailable = [
        "vendor master reconciliation unavailable: no authoritative vendor table",
        "purchase-order reconciliation unavailable: no authoritative PO tables",
        "price validation unavailable: no authoritative price catalog",
        "tax-policy validation unavailable: no authoritative tax table",
        "bank-account validation unavailable: no authoritative payment master",
    ]
    return RiskAssessment(
        policy=policy,
        financial=financial,
        dates=dates,
        inventory=inventory,
        identity_candidates=identity,
        suspicious_signals=suspicious,
        unavailable_reconciliations=unavailable,
        policy_review_reasons=list(dict.fromkeys(review)),
    )
