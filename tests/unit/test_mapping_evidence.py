"""Mapping evidence (3.2): explicit outcomes become a new extraction version, never a mutation."""

from pathlib import Path

from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import WorkflowStore
from invoice_agents.models import CaseStatus
from invoice_agents.orchestration import prepare_case
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.comparison import (
    InventoryReader,
    apply_mapping_evidence,
    compare_inventory_evidence,
)
from invoice_agents.tools.evidence import extract_invoice_evidence
from tests.support.pdf_policy import TEST_PDF_POLICY


def prepare(invoice_dir: Path, name: str, settings: Settings) -> str:
    prepared = prepare_case(invoice_dir / name, settings)
    assert isinstance(prepared, tuple), f"prepare_case failed: {prepared}"
    return prepared[0]


def enrich_and_persist(case_id: str, settings: Settings, store: WorkflowStore) -> None:
    invoice = store.load_extraction(case_id)
    reader = InventoryReader(settings.inventory_db)
    mappings, _comparisons, unresolved = compare_inventory_evidence(invoice, reader)
    enriched = apply_mapping_evidence(invoice, mappings, unresolved)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.save_extraction(case_id, enriched, claim)
    store.release_case_execution(claim)


def extraction_count(settings: Settings, case_id: str) -> int:
    with connect_database(settings.workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS extraction_count FROM extractions WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    return int(row["extraction_count"])


def test_exact_mappings_are_persisted_as_second_extraction_version(
    invoice_dir: Path, settings: Settings
) -> None:
    case_id = prepare(invoice_dir, "invoice_1001.txt", settings)
    store = WorkflowStore(settings.workflow_db)
    enrich_and_persist(case_id, settings, store)
    loaded = store.load_extraction(case_id)
    by_item = {line.raw_item: line for line in loaded.lines}
    assert by_item["WidgetA"].canonical_sku == "SKU-WIDGET-A"
    assert by_item["WidgetB"].canonical_sku == "SKU-WIDGET-B"
    assert all(line.candidate_skus == [] for line in loaded.lines)
    assert extraction_count(settings, case_id) == 2


def test_unresolved_line_keeps_null_sku_and_exposes_candidates(
    invoice_dir: Path, settings: Settings
) -> None:
    case_id = prepare(invoice_dir, "invoice_1010.txt", settings)
    store = WorkflowStore(settings.workflow_db)
    enrich_and_persist(case_id, settings, store)
    loaded = store.load_extraction(case_id)
    by_item = {line.raw_item: line for line in loaded.lines}
    rush = by_item["WidgetA (rush order)"]
    assert rush.canonical_sku is None
    assert rush.candidate_skus
    assert "SKU-WIDGET-A" in rush.candidate_skus
    assert by_item["WidgetA"].canonical_sku == "SKU-WIDGET-A"
    assert by_item["WidgetB"].canonical_sku == "SKU-WIDGET-B"
    assert by_item["GadgetX"].canonical_sku == "SKU-GADGET-X"
    assert extraction_count(settings, case_id) == 2


def test_apply_mapping_evidence_leaves_input_invoice_unmodified(
    invoice_dir: Path, inventory_db: Path, tmp_path: Path
) -> None:
    source = snapshot_source(
        invoice_dir / "invoice_1001.txt",
        tmp_path / "sources",
        max_bytes=10_485_760,
        pdf_policy=TEST_PDF_POLICY,
    )
    invoice = extract_invoice_evidence(source, TEST_PDF_POLICY)
    reader = InventoryReader(inventory_db)
    mappings, _comparisons, unresolved = compare_inventory_evidence(invoice, reader)
    enriched = apply_mapping_evidence(invoice, mappings, unresolved)
    assert enriched is not invoice
    assert all(line.canonical_sku is None for line in invoice.lines)
    assert all(line.canonical_sku is not None for line in enriched.lines)
