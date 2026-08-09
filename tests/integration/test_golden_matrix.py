"""Integration golden matrix: the deterministic pipeline over every fixture (remediation §4/G9).

Runs prepare -> extraction -> identity -> comparison -> totals -> risk for all 20
invoice artifacts against freshly migrated temporary databases, sharing ONE workflow
store in a fixed order so sequenced pairs (duplicate representations and the revision)
can see their earlier cases. Every assertion is the recorded expected treatment; no
model or network call is involved. This is the local half of the Phase 8
reconciliation: the same matrix a live batch run is reconciled against.
"""

from decimal import Decimal
from pathlib import Path
from typing import NamedTuple

import pytest

from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind, migrate_database, seed_inventory
from invoice_agents.db.store import WorkflowStore
from invoice_agents.models import (
    CaseResult,
    CaseStatus,
    ExtractedInvoice,
    FinancialComparison,
    IdentityCandidate,
    IdentityRelationship,
    InventoryComparison,
    InventoryStatus,
    RiskAssessment,
)
from invoice_agents.orchestration import prepare_case
from invoice_agents.tools.comparison import (
    InventoryReader,
    build_risk_assessment,
    compare_inventory,
    compute_invoice_totals,
    find_prior_invoice_candidates,
)

pytestmark = pytest.mark.integration

INVOICE_DIR = Path(__file__).resolve().parents[2] / "data" / "invoices"
THRESHOLD_REASON = "at or above policy threshold"

# The fixed order matters: later artifacts (PDF twins, the 1004 revision, and 1016's
# vendor overlap with 1001) depend on earlier cases already existing in the store.
PROCESSING_ORDER = (
    "invoice_1001.txt",
    "invoice_1002.txt",
    "invoice_1003.txt",
    "invoice_1004.json",
    "invoice_1004_revised.json",
    "invoice_1005.json",
    "invoice_1006.csv",
    "invoice_1007.csv",
    "invoice_1008.txt",
    "invoice_1009.json",
    "invoice_1010.txt",
    "invoice_1011.txt",
    "invoice_1011.pdf",
    "invoice_1012.txt",
    "invoice_1012.pdf",
    "invoice_1013.json",
    "invoice_1013.pdf",
    "invoice_1014.xml",
    "invoice_1015.csv",
    "invoice_1016.json",
)


class CaseEvidence(NamedTuple):
    """One artifact's deterministic pipeline outputs plus its workflow case id."""

    case_id: str
    invoice: ExtractedInvoice
    identity: list[IdentityCandidate]
    comparisons: list[InventoryComparison]
    financial: FinancialComparison
    risk: RiskAssessment


@pytest.fixture(scope="module")
def golden(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CaseEvidence]:
    """Migrate temp databases once and process all 20 artifacts in the fixed order."""

    tmp = tmp_path_factory.mktemp("golden-matrix")
    inventory_db = tmp / "inventory.db"
    workflow_db = tmp / "workflow.db"
    migrate_database(inventory_db, DatabaseKind.INVENTORY)
    seed_inventory(inventory_db)
    migrate_database(workflow_db, DatabaseKind.WORKFLOW)
    settings = Settings(
        xai_api_key="golden-matrix-not-used",
        inventory_db=inventory_db,
        workflow_db=workflow_db,
        source_archive_dir=tmp / "sources",
    )
    store = WorkflowStore(settings.workflow_db)
    reader = InventoryReader(settings.inventory_db)
    processed: dict[str, CaseEvidence] = {}
    for name in PROCESSING_ORDER:
        prepared = prepare_case(INVOICE_DIR / name, settings)
        if isinstance(prepared, CaseResult):
            # The missing key surfaces in test_no_fixture_prepare_failures.
            continue
        case_id, _started_at = prepared
        invoice = store.load_extraction(case_id)
        identity = find_prior_invoice_candidates(case_id, invoice, store)
        claim = store.claim_case_execution(
            case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
        )
        store.promote_predecessor_extraction(claim)
        store.save_identity(case_id, [c.model_dump(mode="json") for c in identity], claim)
        store.release_case_execution(claim)
        comparisons, _unresolved = compare_inventory(invoice, reader)
        financial = compute_invoice_totals(invoice)
        risk = build_risk_assessment(invoice, comparisons, identity, financial, settings)
        processed[name] = CaseEvidence(case_id, invoice, identity, comparisons, financial, risk)
    return processed


def assert_reason(reasons: list[str], *substrings: str) -> str:
    """Assert at least one reason contains every substring; return the first match."""

    matches = [reason for reason in reasons if all(part in reason for part in substrings)]
    assert matches, f"no policy reason contains {substrings!r}; reasons={reasons!r}"
    return matches[0]


def assert_no_reason(reasons: list[str], substring: str) -> None:
    offending = [reason for reason in reasons if substring in reason]
    assert not offending, f"unexpected reason(s) containing {substring!r}: {offending!r}"


def comparison_for(comparisons: list[InventoryComparison], raw_item: str) -> InventoryComparison:
    """Return the single comparison referencing raw_item (aggregation makes it unique)."""

    matches = [item for item in comparisons if raw_item in item.raw_items]
    assert len(matches) == 1, f"expected one comparison for {raw_item!r}, got {matches!r}"
    return matches[0]


def assert_available(
    comparisons: list[InventoryComparison], raw_item: str, requested: str, stock: int
) -> None:
    item = comparison_for(comparisons, raw_item)
    assert item.status is InventoryStatus.AVAILABLE
    assert item.requested_quantity == Decimal(requested)
    assert item.available_stock == stock


def test_no_fixture_prepare_failures(golden: dict[str, CaseEvidence]) -> None:
    """prepare_case returned (case_id, started_at), never a CaseResult, for all 20."""

    assert sorted(golden) == sorted(PROCESSING_ORDER)
    assert len(golden) == 20


def test_1001_txt_clean_baseline(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1001.txt"]
    assert record.risk.policy_review_reasons == []
    assert len(record.comparisons) == 2
    assert_available(record.comparisons, "WidgetA", "10", 15)
    assert_available(record.comparisons, "WidgetB", "5", 10)
    assert record.financial.total_delta == Decimal("0")
    assert record.financial.exact is True


def test_1002_txt_threshold_stock_and_terms_reasons(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1002.txt"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, THRESHOLD_REASON)
    assert_reason(reasons, "EXCEEDS_STOCK", "'GadgetX'")
    # INV-1002 must carry TWO independent date reasons: the ordering violation and the
    # configured R1 §3.1 Net-30 terms deviation are distinct policy findings.
    ordering = assert_reason(reasons, "due date is on or before invoice date")
    terms = assert_reason(
        reasons,
        "deviates from the Net 30 expectation",
        "beyond the configured tolerance of 3 days",
    )
    assert ordering != terms
    gadget = comparison_for(record.comparisons, "GadgetX")
    assert gadget.status is InventoryStatus.EXCEEDS_STOCK
    assert gadget.requested_quantity == Decimal("20")
    assert gadget.available_stock == 5


def test_1003_txt_out_of_stock_relative_date_and_suspicious_language(
    golden: dict[str, CaseEvidence],
) -> None:
    record = golden["invoice_1003.txt"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, THRESHOLD_REASON)
    assert_reason(reasons, "OUT_OF_STOCK")
    assert_reason(reasons, "relative date")
    assert any("urgent" in reason or "wire" in reason for reason in reasons), reasons
    assert_reason(reasons, "required fields are missing: due_date")
    fake = comparison_for(record.comparisons, "FakeItem")
    assert fake.status is InventoryStatus.OUT_OF_STOCK
    assert fake.requested_quantity == Decimal("100")
    assert fake.available_stock == 0
    due = next(date for date in record.risk.dates if date.field == "due_date")
    assert due.status == "RELATIVE"


def test_1004_json_clean_original(golden: dict[str, CaseEvidence]) -> None:
    assert golden["invoice_1004.json"].risk.policy_review_reasons == []


def test_1004_revised_json_possible_revision_identity(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1004_revised.json"]
    assert len(record.identity) == 1
    candidate = record.identity[0]
    assert candidate.relationship is IdentityRelationship.POSSIBLE_REVISION
    assert candidate.case_id == golden["invoice_1004.json"].case_id
    assert_reason(record.risk.policy_review_reasons, "POSSIBLE_REVISION")


def test_1005_json_exceeds_stock_with_boundary_available(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1005.json"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, THRESHOLD_REASON)
    assert_reason(reasons, "EXCEEDS_STOCK", "'GadgetX'")
    gadget = comparison_for(record.comparisons, "GadgetX")
    assert gadget.status is InventoryStatus.EXCEEDS_STOCK
    assert gadget.requested_quantity == Decimal("8")
    assert gadget.available_stock == 5
    # Boundary: requested == stock is availability, not an exceedance.
    assert_available(record.comparisons, "WidgetB", "10", 10)


def test_1006_csv_vertical_repeated_keys_recorded_actual(golden: dict[str, CaseEvidence]) -> None:
    # The original remediation plan labeled 1006 'record actual'; this IS the recorded
    # actual: NOT clean under current policy. A declared 0.00 tax amount without a rate
    # is carried, not recomputed (tax_recomputable False), and the omitted currency
    # falls back to the reviewed USD convention.
    record = golden["invoice_1006.csv"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, "USD convention requires review")
    assert_reason(reasons, "financial evidence is incomplete")
    assert record.financial.tax_recomputable is False
    assert record.financial.exact is False
    assert_no_reason(reasons, THRESHOLD_REASON)
    assert_no_reason(reasons, "inventory")
    # The naive-dict-collapse trap did NOT collapse: repeated vertical field,value keys
    # parse as two distinct lines.
    assert len(record.invoice.lines) == 2
    assert len(record.comparisons) == 2
    assert_available(record.comparisons, "WidgetA", "5", 15)
    assert_available(record.comparisons, "WidgetB", "3", 10)


def test_1007_csv_exceeds_stock_and_total_delta(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1007.csv"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, THRESHOLD_REASON)
    assert_reason(reasons, "EXCEEDS_STOCK", "'WidgetA'")
    assert_reason(reasons, "EXCEEDS_STOCK", "'WidgetB'")
    assert_reason(reasons, "required fields are missing: payment_terms")
    assert_reason(reasons, "USD convention requires review")
    assert_reason(reasons, "financial evidence is incomplete")
    widget_a = comparison_for(record.comparisons, "WidgetA")
    assert widget_a.status is InventoryStatus.EXCEEDS_STOCK
    assert widget_a.requested_quantity == Decimal("20")
    assert widget_a.available_stock == 15
    widget_b = comparison_for(record.comparisons, "WidgetB")
    assert widget_b.status is InventoryStatus.EXCEEDS_STOCK
    assert widget_b.requested_quantity == Decimal("15")
    assert widget_b.available_stock == 10
    assert_available(record.comparisons, "GadgetX", "3", 5)
    assert record.financial.total_delta == Decimal("110.0000")


def test_1008_txt_unknown_items_below_threshold_boundary(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1008.txt"]
    reasons = record.risk.policy_review_reasons
    unknown_reasons = [reason for reason in reasons if "inventory UNKNOWN" in reason]
    assert len(unknown_reasons) == 2, reasons
    assert_reason(reasons, "inventory UNKNOWN", "SuperGizmo")
    assert_reason(reasons, "inventory UNKNOWN", "MegaSprocket")
    assert_reason(reasons, "required fields are missing: payment_terms")
    # Boundary: 9,900.00 USD sits below the 10,000.00 threshold, so no threshold reason.
    assert record.invoice.declared_total == Decimal("9900.00")
    assert_no_reason(reasons, THRESHOLD_REASON)


def test_1009_json_missing_fields_and_invalid_quantity(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1009.json"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, "required fields are missing:", "vendor", "due_date", "payment_terms")
    assert_reason(reasons, "INVALID_QUANTITY")
    assert_reason(reasons, "financial evidence is incomplete")
    assert_reason(reasons, "one or more quantities are zero or negative")
    widget_a = comparison_for(record.comparisons, "WidgetA")
    assert widget_a.status is InventoryStatus.INVALID_QUANTITY
    assert widget_a.requested_quantity == Decimal("-5")
    assert record.financial.subtotal_delta == Decimal("-1250")
    due = next(date for date in record.risk.dates if date.field == "due_date")
    assert due.status == "MISSING"


def test_1010_txt_rush_order_ambiguous(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1010.txt"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, "inventory AMBIGUOUS", "WidgetA (rush order)")
    assert_reason(reasons, "financial evidence is incomplete")
    assert_no_reason(reasons, THRESHOLD_REASON)
    rush = comparison_for(record.comparisons, "WidgetA (rush order)")
    assert rush.status is InventoryStatus.AMBIGUOUS
    assert rush.requested_quantity == Decimal("4")
    assert rush.available_stock is None
    assert_available(record.comparisons, "WidgetA", "8", 15)
    assert_available(record.comparisons, "WidgetB", "4", 10)
    assert_available(record.comparisons, "GadgetX", "2", 5)
    assert record.financial.total_delta == Decimal("0")


def test_1011_txt_clean_first_seen(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1011.txt"]
    assert record.risk.policy_review_reasons == []
    assert record.identity == []


def test_1011_pdf_duplicate_representation(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1011.pdf"]
    assert len(record.identity) == 1
    candidate = record.identity[0]
    assert candidate.relationship is IdentityRelationship.DUPLICATE_REPRESENTATION
    assert candidate.case_id == golden["invoice_1011.txt"].case_id
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, "DUPLICATE_REPRESENTATION")
    # The PDF omits the terms row its text twin carries.
    assert_reason(reasons, "required fields are missing: payment_terms")
    assert record.invoice.declared_total == golden["invoice_1011.txt"].invoice.declared_total
    assert record.financial.total_delta == Decimal("0")


def assert_1012_shared_shape(record: CaseEvidence) -> None:
    """Both 1012 representations show the same OCR, alias, and date treatment."""

    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, "OCR-like O normalized to 0")
    assert_reason(reasons, "inventory AMBIGUOUS", "Widget A")
    assert_reason(reasons, "inventory AMBIGUOUS", "Gadget X")
    assert_reason(reasons, "invoice_date is AMBIGUOUS")
    # Spaced aliases never auto-map to a canonical SKU.
    widget_a = comparison_for(record.comparisons, "Widget A")
    assert widget_a.status is InventoryStatus.AMBIGUOUS
    assert widget_a.sku is None
    assert widget_a.requested_quantity == Decimal("12")
    gadget_x = comparison_for(record.comparisons, "Gadget X")
    assert gadget_x.status is InventoryStatus.AMBIGUOUS
    assert gadget_x.sku is None
    assert gadget_x.requested_quantity == Decimal("4")
    assert_available(record.comparisons, "WidgetB", "7", 10)


def test_1012_txt_ocr_and_spaced_aliases(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1012.txt"]
    assert_1012_shared_shape(record)
    assert record.identity == []


def test_1012_pdf_duplicate_representation_of_txt(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1012.pdf"]
    assert_1012_shared_shape(record)
    assert len(record.identity) == 1
    candidate = record.identity[0]
    assert candidate.relationship is IdentityRelationship.DUPLICATE_REPRESENTATION
    assert candidate.case_id == golden["invoice_1012.txt"].case_id


def assert_1013_aggregated_exceeds(record: CaseEvidence) -> None:
    """Repeated lines aggregate BEFORE the stock check in both representations."""

    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, THRESHOLD_REASON)
    assert_reason(reasons, "EXCEEDS_STOCK", "'WidgetA'")
    assert_reason(reasons, "EXCEEDS_STOCK", "'WidgetB'")
    assert_reason(reasons, "EXCEEDS_STOCK", "'GadgetX'")
    assert_reason(reasons, "financial evidence is incomplete")
    widget_a = comparison_for(record.comparisons, "WidgetA")
    assert widget_a.status is InventoryStatus.EXCEEDS_STOCK
    assert widget_a.requested_quantity == Decimal("22")
    assert widget_a.available_stock == 15
    widget_b = comparison_for(record.comparisons, "WidgetB")
    assert widget_b.status is InventoryStatus.EXCEEDS_STOCK
    assert widget_b.requested_quantity == Decimal("18")
    assert widget_b.available_stock == 10
    gadget_x = comparison_for(record.comparisons, "GadgetX")
    assert gadget_x.status is InventoryStatus.EXCEEDS_STOCK
    assert gadget_x.requested_quantity == Decimal("9")
    assert gadget_x.available_stock == 5
    # Declared 22562.80 vs calculated 22512.80; calculated - declared = -50.
    assert record.financial.total_delta == Decimal("-50.000")


def test_1013_json_aggregated_exceeds_and_negative_delta(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1013.json"]
    assert_1013_aggregated_exceeds(record)
    assert record.identity == []


def test_1013_pdf_duplicate_of_json_with_same_aggregates(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1013.pdf"]
    assert_1013_aggregated_exceeds(record)
    assert len(record.identity) == 1
    candidate = record.identity[0]
    assert candidate.relationship is IdentityRelationship.DUPLICATE_REPRESENTATION
    assert candidate.case_id == golden["invoice_1013.json"].case_id


def test_1014_xml_eur_short_circuits_usd_threshold(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1014.xml"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, "currency EUR has no approved FX/threshold policy against USD")
    # EUR short-circuits the USD threshold comparison entirely.
    assert_no_reason(reasons, THRESHOLD_REASON)
    assert {item.status for item in record.comparisons} == {InventoryStatus.AVAILABLE}
    assert record.financial.total_delta == Decimal("0")


def test_1015_csv_conservative_review_recorded_actual(golden: dict[str, CaseEvidence]) -> None:
    # The remediation plan's 'clean' label for 1015 does not hold: row-oriented CSV
    # genuinely lacks payment terms and currency evidence, so conservative review is
    # the correct, recorded actual.
    record = golden["invoice_1015.csv"]
    reasons = record.risk.policy_review_reasons
    assert len(reasons) == 3, reasons
    assert_reason(reasons, "required fields are missing: payment_terms")
    assert_reason(reasons, "USD convention requires review")
    assert_reason(reasons, "extraction note:", "USD convention requires review")
    assert_no_reason(reasons, THRESHOLD_REASON)
    assert_no_reason(reasons, "inventory")
    assert_no_reason(reasons, "identity")
    assert_no_reason(reasons, "financial evidence")
    assert {item.status for item in record.comparisons} == {InventoryStatus.AVAILABLE}
    assert record.financial.exact is True


def test_1016_json_unknown_item_and_vendor_conflict(golden: dict[str, CaseEvidence]) -> None:
    record = golden["invoice_1016.json"]
    reasons = record.risk.policy_review_reasons
    assert_reason(reasons, "inventory UNKNOWN", "WidgetC")
    assert_reason(reasons, "identity CONFLICT")
    widget_c = comparison_for(record.comparisons, "WidgetC")
    assert widget_c.status is InventoryStatus.UNKNOWN
    assert widget_c.requested_quantity == Decimal("3")
    assert widget_c.available_stock is None
    assert_available(record.comparisons, "WidgetA", "4", 15)
    assert_available(record.comparisons, "WidgetB", "2", 10)
    # Same vendor as 1001 ('Widgets Inc.') with a different invoice number is a
    # partial identity match, classified as CONFLICT against 1001's case.
    assert len(record.identity) == 1
    candidate = record.identity[0]
    assert candidate.relationship is IdentityRelationship.CONFLICT
    assert candidate.case_id == golden["invoice_1001.txt"].case_id
    assert candidate.vendor == "Widgets Inc."


def test_sequenced_pairs_are_ordered(golden: dict[str, CaseEvidence]) -> None:
    """Later representations and the revision reference the correct earlier case ids."""

    pairs = (
        ("invoice_1011.pdf", "invoice_1011.txt", IdentityRelationship.DUPLICATE_REPRESENTATION),
        ("invoice_1012.pdf", "invoice_1012.txt", IdentityRelationship.DUPLICATE_REPRESENTATION),
        ("invoice_1013.pdf", "invoice_1013.json", IdentityRelationship.DUPLICATE_REPRESENTATION),
        ("invoice_1004_revised.json", "invoice_1004.json", IdentityRelationship.POSSIBLE_REVISION),
    )
    for later, earlier, relationship in pairs:
        candidates = [c for c in golden[later].identity if c.relationship is relationship]
        assert len(candidates) == 1, (later, golden[later].identity)
        assert candidates[0].case_id == golden[earlier].case_id, (later, earlier)
        # The earlier, first-seen representation saw no prior candidate for itself.
        assert all(
            candidate.case_id != golden[later].case_id for candidate in golden[earlier].identity
        )
