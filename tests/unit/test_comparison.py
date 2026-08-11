"""Inventory aggregation, exact alias policy, arithmetic, identity, and risk tests."""

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import (
    CaseStatus,
    ExtractedInvoice,
    IdentityRelationship,
    InventoryStatus,
    ToolStatus,
)
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.comparison import (
    InventoryReader,
    build_risk_assessment,
    compare_inventory,
    compute_invoice_totals,
    find_prior_invoice_candidates,
)
from invoice_agents.tools.evidence import extract_invoice_evidence
from tests.support.pdf_policy import TEST_PDF_POLICY


def invoice(invoice_dir: Path, name: str, archive: Path):  # type: ignore[no-untyped-def]
    source = snapshot_source(
        invoice_dir / name,
        archive,
        max_bytes=10_485_760,
        pdf_policy=TEST_PDF_POLICY,
    )
    return extract_invoice_evidence(source, TEST_PDF_POLICY)


def persist_extraction(store: WorkflowStore, case_id: str, extracted: ExtractedInvoice) -> None:
    store.register_source(extracted.source)
    store.create_case(case_id, extracted.source, datetime.now(UTC))
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.save_extraction(case_id, extracted, claim)
    store.release_case_execution(claim)


def test_stock_excess_unknown_and_negative(
    invoice_dir: Path, inventory_db: Path, tmp_path: Path
) -> None:
    reader = InventoryReader(inventory_db)
    excess, _ = compare_inventory(
        invoice(invoice_dir, "invoice_1002.txt", tmp_path / "sources"), reader
    )
    assert excess[0].status is InventoryStatus.EXCEEDS_STOCK
    assert excess[0].queried_row is not None
    unknown, _ = compare_inventory(
        invoice(invoice_dir, "invoice_1016.json", tmp_path / "sources"), reader
    )
    assert unknown[-1].status is InventoryStatus.UNKNOWN
    negative, _ = compare_inventory(
        invoice(invoice_dir, "invoice_1009.json", tmp_path / "sources"), reader
    )
    assert negative[0].status is InventoryStatus.INVALID_QUANTITY


def test_repeated_skus_are_aggregated_before_stock(
    invoice_dir: Path, inventory_db: Path, tmp_path: Path
) -> None:
    comparisons, _ = compare_inventory(
        invoice(invoice_dir, "invoice_1013.json", tmp_path / "sources"),
        InventoryReader(inventory_db),
    )
    quantities = {item.sku: item.requested_quantity for item in comparisons}
    assert quantities == {
        "SKU-WIDGET-A": Decimal("22"),
        "SKU-WIDGET-B": Decimal("18"),
        "SKU-GADGET-X": Decimal("9"),
    }
    assert all(item.status is InventoryStatus.EXCEEDS_STOCK for item in comparisons)


def test_candidates_never_become_implicit_aliases(
    invoice_dir: Path, inventory_db: Path, tmp_path: Path
) -> None:
    comparisons, unresolved = compare_inventory(
        invoice(invoice_dir, "invoice_1010.txt", tmp_path / "sources"),
        InventoryReader(inventory_db),
    )
    rush = comparisons[-1]
    assert rush.sku is None
    assert rush.status is InventoryStatus.AMBIGUOUS
    assert unresolved["WidgetA (rush order)"].candidates


def test_sql_error_is_not_not_found(tmp_path: Path) -> None:
    result = InventoryReader(tmp_path / "missing.db").lookup_inventory_exact("WidgetA")
    assert result.status is ToolStatus.ERROR
    assert result.error


def test_financial_discrepancies_match_known_fixtures(invoice_dir: Path, tmp_path: Path) -> None:
    inv_1007 = compute_invoice_totals(
        invoice(invoice_dir, "invoice_1007.csv", tmp_path / "sources")
    )
    assert inv_1007.calculated_total == Decimal("15635.00")
    assert inv_1007.total_delta == Decimal("110.00")
    inv_1013 = compute_invoice_totals(
        invoice(invoice_dir, "invoice_1013.json", tmp_path / "sources")
    )
    assert inv_1013.calculated_total == Decimal("22512.80")
    assert inv_1013.total_delta == Decimal("-50.00")
    inv_1009 = compute_invoice_totals(
        invoice(invoice_dir, "invoice_1009.json", tmp_path / "sources")
    )
    assert inv_1009.subtotal_delta == Decimal("-1250.00")


def test_identity_representation_and_revision(
    invoice_dir: Path, workflow_db: Path, tmp_path: Path
) -> None:
    store = WorkflowStore(workflow_db)
    first = invoice(invoice_dir, "invoice_1011.txt", tmp_path / "sources")
    persist_extraction(store, "case_first", first)
    second = invoice(invoice_dir, "invoice_1011.pdf", tmp_path / "sources")
    persist_extraction(store, "case_second", second)
    candidates = find_prior_invoice_candidates("case_second", second, store)
    assert candidates[0].relationship is IdentityRelationship.DUPLICATE_REPRESENTATION

    original = invoice(invoice_dir, "invoice_1004.json", tmp_path / "sources")
    persist_extraction(store, "case_original", original)
    revised = invoice(invoice_dir, "invoice_1004_revised.json", tmp_path / "sources")
    persist_extraction(store, "case_revised", revised)
    revision_candidates = find_prior_invoice_candidates("case_revised", revised, store)
    assert any(
        candidate.relationship is IdentityRelationship.POSSIBLE_REVISION
        for candidate in revision_candidates
    )


def test_qualifying_prior_with_deleted_extraction_fails_loudly(
    invoice_dir: Path, workflow_db: Path, tmp_path: Path
) -> None:
    store = WorkflowStore(workflow_db)
    prior = invoice(invoice_dir, "invoice_1011.txt", tmp_path / "sources")
    persist_extraction(store, "case_prior", prior)
    current = invoice(invoice_dir, "invoice_1011.pdf", tmp_path / "sources")

    identity_rows = store.identity_rows(
        "case_current",
        current.invoice_number.normalized_value,
        current.vendor.normalized_value,
    )
    assert [str(row["case_id"]) for row in identity_rows] == ["case_prior"]
    uncorrupted = find_prior_invoice_candidates("case_current", current, store)
    assert [candidate.relationship for candidate in uncorrupted] == [
        IdentityRelationship.DUPLICATE_REPRESENTATION
    ]
    with connect_database(workflow_db) as connection:
        connection.execute("DELETE FROM extractions WHERE case_id = ?", ("case_prior",))
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as direct_error:
        store.load_extraction("case_prior")
    assert direct_error.value.category is ErrorCategory.DATABASE
    assert direct_error.value.stop_reason == "EXTRACTION_NOT_FOUND"
    assert direct_error.value.case_id == "case_prior"

    with pytest.raises(InvoiceAgentsError) as candidate_error:
        find_prior_invoice_candidates("case_current", current, store)
    assert candidate_error.value.category is ErrorCategory.DATABASE
    assert candidate_error.value.stop_reason == "EXTRACTION_NOT_FOUND"
    assert candidate_error.value.case_id == "case_prior"
    assert candidate_error.value.message == direct_error.value.message


def test_qualifying_prior_with_corrupt_extraction_preserves_validation_error(
    invoice_dir: Path, workflow_db: Path, tmp_path: Path
) -> None:
    store = WorkflowStore(workflow_db)
    prior = invoice(invoice_dir, "invoice_1011.txt", tmp_path / "sources")
    persist_extraction(store, "case_prior", prior)
    current = invoice(invoice_dir, "invoice_1011.pdf", tmp_path / "sources")

    identity_rows = store.identity_rows(
        "case_current",
        current.invoice_number.normalized_value,
        current.vendor.normalized_value,
    )
    assert [str(row["case_id"]) for row in identity_rows] == ["case_prior"]
    uncorrupted = find_prior_invoice_candidates("case_current", current, store)
    assert [candidate.relationship for candidate in uncorrupted] == [
        IdentityRelationship.DUPLICATE_REPRESENTATION
    ]
    with connect_database(workflow_db) as connection:
        connection.execute(
            "UPDATE extractions SET payload_json = ? WHERE case_id = ?",
            ("{}", "case_prior"),
        )
        connection.commit()

    with pytest.raises(ValidationError) as direct_error:
        store.load_extraction("case_prior")
    with pytest.raises(ValidationError) as candidate_error:
        find_prior_invoice_candidates("case_current", current, store)
    assert type(candidate_error.value) is ValidationError
    assert candidate_error.value.title == direct_error.value.title
    assert candidate_error.value.errors(include_url=False) == direct_error.value.errors(
        include_url=False
    )


def test_qualifying_prior_with_malformed_json_preserves_validation_error(
    invoice_dir: Path, workflow_db: Path, tmp_path: Path
) -> None:
    store = WorkflowStore(workflow_db)
    prior = invoice(invoice_dir, "invoice_1011.txt", tmp_path / "sources")
    persist_extraction(store, "case_prior", prior)
    current = invoice(invoice_dir, "invoice_1011.pdf", tmp_path / "sources")

    identity_rows = store.identity_rows(
        "case_current",
        current.invoice_number.normalized_value,
        current.vendor.normalized_value,
    )
    assert [str(row["case_id"]) for row in identity_rows] == ["case_prior"]
    uncorrupted = find_prior_invoice_candidates("case_current", current, store)
    assert [candidate.relationship for candidate in uncorrupted] == [
        IdentityRelationship.DUPLICATE_REPRESENTATION
    ]
    with connect_database(workflow_db) as connection:
        connection.execute(
            "UPDATE extractions SET payload_json = ? WHERE case_id = ?",
            ("{", "case_prior"),
        )
        connection.commit()

    with pytest.raises(ValidationError) as direct_error:
        store.load_extraction("case_prior")
    with pytest.raises(ValidationError) as candidate_error:
        find_prior_invoice_candidates("case_current", current, store)
    assert type(candidate_error.value) is ValidationError
    assert candidate_error.value.title == direct_error.value.title
    assert candidate_error.value.errors(include_url=False) == direct_error.value.errors(
        include_url=False
    )


def test_qualifying_prior_preserves_sqlite_load_failure(
    invoice_dir: Path, workflow_db: Path, tmp_path: Path
) -> None:
    store = WorkflowStore(workflow_db)
    prior = invoice(invoice_dir, "invoice_1011.txt", tmp_path / "sources")
    persist_extraction(store, "case_prior", prior)
    current = invoice(invoice_dir, "invoice_1011.pdf", tmp_path / "sources")

    identity_rows = store.identity_rows(
        "case_current",
        current.invoice_number.normalized_value,
        current.vendor.normalized_value,
    )
    assert [str(row["case_id"]) for row in identity_rows] == ["case_prior"]
    uncorrupted = find_prior_invoice_candidates("case_current", current, store)
    assert [candidate.relationship for candidate in uncorrupted] == [
        IdentityRelationship.DUPLICATE_REPRESENTATION
    ]
    with connect_database(workflow_db) as connection:
        connection.execute("ALTER TABLE extractions RENAME TO extractions_corrupt")
        connection.commit()

    with pytest.raises(sqlite3.OperationalError) as direct_error:
        store.load_extraction("case_prior")
    with pytest.raises(sqlite3.OperationalError) as candidate_error:
        find_prior_invoice_candidates("case_current", current, store)
    assert type(candidate_error.value) is sqlite3.OperationalError
    assert candidate_error.value.args == direct_error.value.args


def test_no_matching_prior_returns_no_identity_candidates(
    invoice_dir: Path, workflow_db: Path, tmp_path: Path
) -> None:
    store = WorkflowStore(workflow_db)
    unrelated = invoice(invoice_dir, "invoice_1014.xml", tmp_path / "sources")
    persist_extraction(store, "case_unrelated", unrelated)
    current = invoice(invoice_dir, "invoice_1004.json", tmp_path / "sources")

    identity_rows = store.identity_rows(
        "case_current",
        current.invoice_number.normalized_value,
        current.vendor.normalized_value,
    )
    assert identity_rows == []
    assert find_prior_invoice_candidates("case_current", current, store) == []


def test_policy_triggers_high_dollar_non_usd_and_clean(
    invoice_dir: Path, inventory_db: Path, settings: Settings, tmp_path: Path
) -> None:
    reader = InventoryReader(inventory_db)
    high = invoice(invoice_dir, "invoice_1002.txt", tmp_path / "sources")
    high_inventory, _ = compare_inventory(high, reader)
    high_risk = build_risk_assessment(
        high, high_inventory, [], compute_invoice_totals(high), settings
    )
    assert any("policy threshold" in reason for reason in high_risk.policy_review_reasons)
    eur = invoice(invoice_dir, "invoice_1014.xml", tmp_path / "sources")
    eur_inventory, _ = compare_inventory(eur, reader)
    eur_risk = build_risk_assessment(eur, eur_inventory, [], compute_invoice_totals(eur), settings)
    assert any("no approved FX" in reason for reason in eur_risk.policy_review_reasons)
    clean = invoice(invoice_dir, "invoice_1001.txt", tmp_path / "sources")
    clean_inventory, _ = compare_inventory(clean, reader)
    clean_risk = build_risk_assessment(
        clean, clean_inventory, [], compute_invoice_totals(clean), settings
    )
    assert clean_risk.policy_review_reasons == []
