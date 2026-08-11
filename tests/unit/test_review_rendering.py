"""G10a: PDF review requests render original-layout pages with verifiable hashes."""

import hashlib
import stat
from pathlib import Path

import pytest

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.hitl.service import create_review_request
from invoice_agents.models import CaseStatus, Critique, DecisionKind, ReviewRequest
from invoice_agents.orchestration import prepare_case
from invoice_agents.review_artifact import ReviewPageBinding, ReviewPageEvidenceError
from invoice_agents.tools.comparison import (
    InventoryReader,
    apply_mapping_evidence,
    build_risk_assessment,
    compare_inventory_evidence,
    compute_invoice_totals,
    find_prior_invoice_candidates,
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
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    invoice = store.promote_predecessor_extraction(claim)
    mappings, comparisons, unresolved = compare_inventory_evidence(
        invoice, InventoryReader(settings.inventory_db)
    )
    invoice = apply_mapping_evidence(invoice, mappings, unresolved)
    store.save_extraction(case_id, invoice, claim)
    identity = find_prior_invoice_candidates(case_id, invoice, store)
    risk = build_risk_assessment(
        invoice, comparisons, identity, compute_invoice_totals(invoice), settings
    )
    assert risk.policy_review_reasons
    case_critique = make_critique()
    store.save_identity(
        case_id,
        [candidate.model_dump(mode="json") for candidate in identity],
        claim,
    )
    store.save_comparison(
        case_id,
        "inventory",
        {
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        },
        claim,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    store.save_critique(case_id, case_critique, claim)
    review = create_review_request(
        case_id,
        invoice,
        risk,
        case_critique,
        DecisionKind.HOLD,
        ["deterministic review rationale"],
        store,
        claim,
        pdf_policy=settings.pdf_policy(),
    )
    store.release_case_execution(claim)
    return review


def test_pdf_review_renders_page_one_with_verifiable_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.chdir(tmp_path)
    review = build_review("invoice_1011.pdf", settings)
    pages = review.evidence_bundle["rendered_pages"]
    assert len(pages) == 1
    entry = pages[0]
    assert set(entry) == {
        "path",
        "page",
        "sha256",
        "renderer",
        "device",
        "inode",
        "file_type",
        "size_bytes",
    }
    assert entry["page"] == 1
    assert entry["renderer"] == "PyMuPDF"
    rendered = Path(entry["path"])
    assert rendered.is_file()
    assert rendered.suffix == ".png"
    expected_dir = (tmp_path / "artifacts" / "reviews" / review.review_id).resolve()
    assert rendered.parent == expected_dir
    assert hashlib.sha256(rendered.read_bytes()).hexdigest() == entry["sha256"]
    identity = rendered.stat()
    assert entry["device"] == identity.st_dev
    assert entry["inode"] == identity.st_ino
    assert entry["file_type"] == stat.S_IFMT(identity.st_mode) == stat.S_IFREG
    assert entry["size_bytes"] == identity.st_size
    assert identity.st_nlink == 1


def test_non_pdf_review_has_no_rendered_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.chdir(tmp_path)
    review = build_review("invoice_1002.txt", settings)
    assert review.evidence_bundle["rendered_pages"] == []


def test_review_page_binding_rejects_forged_renderer_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    monkeypatch.chdir(tmp_path)
    review = build_review("invoice_1011.pdf", settings)
    entry = review.evidence_bundle["rendered_pages"][0]
    assert isinstance(entry, dict)
    forged = {**entry, "renderer": "forged-renderer"}

    with pytest.raises(ReviewPageEvidenceError):
        ReviewPageBinding.from_payload(
            forged,
            review_id=review.review_id,
            source_id=review.source.source_id,
            expected_page=1,
        )


def test_pdf_render_uses_snapshot_after_submitted_path_is_replaced(
    tmp_path: Path, settings: Settings
) -> None:
    submitted = tmp_path / "submitted.pdf"
    submitted.write_bytes((DATA_DIR / "invoice_1011.pdf").read_bytes())
    prepared = prepare_case(submitted, settings)
    assert isinstance(prepared, tuple), f"prepare_case failed: {prepared}"
    source = WorkflowStore(settings.workflow_db).load_extraction(prepared[0]).source
    first = render_pdf_page(source, 1, tmp_path / "first", settings.pdf_policy())

    submitted.write_bytes((DATA_DIR / "invoice_1012.pdf").read_bytes())
    second = render_pdf_page(source, 1, tmp_path / "second", settings.pdf_policy())

    assert source.canonical_path != submitted.resolve()
    assert first["sha256"] == second["sha256"]
