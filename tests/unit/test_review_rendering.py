"""G10a: PDF review requests render original-layout pages with verifiable hashes."""

import hashlib
from pathlib import Path

import pytest

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.hitl.service import create_review_request
from invoice_agents.models import Critique, DecisionKind, ReviewRequest
from invoice_agents.orchestration import prepare_case
from invoice_agents.tools.comparison import (
    InventoryReader,
    build_risk_assessment,
    compare_inventory,
    compute_invoice_totals,
)
from invoice_agents.tools.evidence import render_pdf_page

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "invoices"


def make_critique() -> Critique:
    return Critique(
        supported_findings=["deterministic evidence reviewed"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=DecisionKind.HOLD,
        rationale=["policy triggers require review"],
    )


def build_review(name: str, settings: Settings) -> ReviewRequest:
    prepared = prepare_case(DATA_DIR / name, settings)
    assert isinstance(prepared, tuple), f"prepare_case failed: {prepared}"
    case_id = prepared[0]
    store = WorkflowStore(settings.workflow_db)
    invoice = store.load_extraction(case_id)
    comparisons, _unresolved = compare_inventory(invoice, InventoryReader(settings.inventory_db))
    risk = build_risk_assessment(
        invoice, comparisons, [], compute_invoice_totals(invoice), settings
    )
    assert risk.policy_review_reasons
    return create_review_request(
        case_id,
        invoice,
        risk,
        make_critique(),
        DecisionKind.HOLD,
        ["deterministic review rationale"],
        store,
    )


def test_pdf_review_renders_page_one_with_verifiable_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.chdir(tmp_path)
    review = build_review("invoice_1011.pdf", settings)
    pages = review.evidence_bundle["rendered_pages"]
    assert len(pages) == 1
    entry = pages[0]
    assert set(entry) == {"path", "page", "sha256", "renderer"}
    assert entry["page"] == 1
    assert entry["renderer"] == "PyMuPDF"
    rendered = Path(entry["path"])
    assert rendered.is_file()
    assert rendered.suffix == ".png"
    expected_dir = (tmp_path / "artifacts" / "reviews" / review.review_id).resolve()
    assert rendered.parent == expected_dir
    assert hashlib.sha256(rendered.read_bytes()).hexdigest() == entry["sha256"]


def test_non_pdf_review_has_no_rendered_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.chdir(tmp_path)
    review = build_review("invoice_1002.txt", settings)
    assert review.evidence_bundle["rendered_pages"] == []


def test_pdf_render_uses_snapshot_after_submitted_path_is_replaced(
    tmp_path: Path, settings: Settings
) -> None:
    submitted = tmp_path / "submitted.pdf"
    submitted.write_bytes((DATA_DIR / "invoice_1011.pdf").read_bytes())
    prepared = prepare_case(submitted, settings)
    assert isinstance(prepared, tuple), f"prepare_case failed: {prepared}"
    source = WorkflowStore(settings.workflow_db).load_extraction(prepared[0]).source
    first = render_pdf_page(source, 1, tmp_path / "first")

    submitted.write_bytes((DATA_DIR / "invoice_1012.pdf").read_bytes())
    second = render_pdf_page(source, 1, tmp_path / "second")

    assert source.canonical_path != submitted.resolve()
    assert first["sha256"] == second["sha256"]
