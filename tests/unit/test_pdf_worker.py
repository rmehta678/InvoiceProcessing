"""Real-process tests for bounded, fail-closed PDF operations."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from datetime import UTC, datetime
from importlib import import_module
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter

from invoice_agents.errors import SourceEvidenceError
from invoice_agents.models import SourceArtifact

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "invoices"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def pdf_worker() -> Any:
    """Load the intended boundary at runtime so its absence is a genuine RED failure."""

    return import_module("invoice_agents.pdf_worker")


def content_addressed_pdf(path: Path, archive: Path, page_count: int | None) -> SourceArtifact:
    """Build a verified artifact without parsing untrusted PDF bytes in this test process."""

    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    archive.mkdir(parents=True, exist_ok=True)
    target = (archive / f"{digest}.pdf").resolve()
    target.write_bytes(content)
    return SourceArtifact(
        source_id=f"src_{digest}",
        canonical_path=target,
        sha256=digest,
        source_format="pdf",
        size_bytes=len(content),
        modified_at=datetime.now(UTC),
        page_count=page_count,
    )


def child_pids() -> set[int]:
    return {child.pid for child in multiprocessing.active_children() if child.pid is not None}


def attempt_over_limit_allocation(connection: Connection, allowance_bytes: int) -> None:
    """Apply the production limits, then pressure the real address-space ceiling."""

    try:
        pdf_worker()._apply_resource_limits(10, allowance_bytes)
        try:
            bytearray(allowance_bytes * 2)
        except (MemoryError, OSError):
            connection.send("blocked")
        else:
            connection.send("allocated")
    except Exception as exc:
        connection.send(f"limit-failed:{type(exc).__name__}:{exc}")
    finally:
        connection.close()


def test_worker_address_space_limit_blocks_allocation_beyond_allowance() -> None:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    allowance_bytes = 32 * 1024 * 1024
    process = context.Process(
        target=attempt_over_limit_allocation,
        args=(child_connection, allowance_bytes),
    )
    process.start()
    child_connection.close()
    assert parent_connection.poll(5.0)
    assert parent_connection.recv() == "blocked"
    process.join(1.0)
    assert process.exitcode == 0
    assert not process.is_alive()
    parent_connection.close()


def test_pdf_timeout_terminates_and_joins_spawned_worker(tmp_path: Path) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker().extract_pdf_in_worker(source, timeout_seconds=0.000_001)

    assert excinfo.value.stop_reason == "PDF_PARSE_TIMEOUT"
    assert child_pids() == before


def test_pdf_parser_crash_fails_closed_and_reaps_worker(tmp_path: Path) -> None:
    source = content_addressed_pdf(FIXTURE_DIR / "corrupt.pdf", tmp_path / "sources", None)
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker().extract_pdf_in_worker(source, timeout_seconds=5.0)

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert child_pids() == before


def test_pdf_page_limit_is_enforced_by_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = tmp_path / "two-pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with submitted.open("wb") as handle:
        writer.write(handle)
    source = content_addressed_pdf(submitted, tmp_path / "sources", 2)
    monkeypatch.setenv("INVOICE_PDF_MAX_PAGES", "1")

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker().extract_pdf_in_worker(source, timeout_seconds=5.0)

    assert excinfo.value.stop_reason == "PDF_PAGE_LIMIT_EXCEEDED"
    assert excinfo.value.details == {"page_count": 2, "max_pages": 1}


def test_pdf_worker_extracts_json_safe_page_text(tmp_path: Path) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)

    result = pdf_worker().extract_pdf_in_worker(source, timeout_seconds=5.0)

    assert result["extractor"] == "pypdf"
    assert result["page_count"] == 1
    assert result["pages"] == [
        {
            "page": 1,
            "text": (
                "INVOICE\nInvoice Number: INV-1011\nVendor: Summit Manufacturing Co.\n"
                "Date: 2026-01-20\nDue Date: 2026-02-20\nItem Qty Unit Price Amount\n"
                "WidgetA 6 $250.00 $1,500.00\nWidgetB 3 $500.00 $1,500.00\n"
                "Total: $3,000.00"
            ),
        }
    ]
    json.dumps(result)


def test_pdf_worker_renders_page_to_real_png(tmp_path: Path) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    target = tmp_path / "rendered.png"

    result = pdf_worker().render_pdf_page_in_worker(
        source, page=1, target=target, timeout_seconds=5.0
    )

    assert result == {
        "path": str(target.resolve()),
        "page": 1,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "renderer": "PyMuPDF",
    }
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".pdf-render-")]
