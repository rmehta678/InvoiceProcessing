"""Real-process tests for bounded, fail-closed PDF operations."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import stat
import struct
import time
import tracemalloc
import warnings
from collections import Counter
from datetime import UTC, datetime
from importlib import import_module
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from invoice_agents.config import Settings
from invoice_agents.errors import SourceEvidenceError
from invoice_agents.models import SourceArtifact
from invoice_agents.tools.evidence import render_pdf_page

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


def pdf_policy(
    *,
    max_pages: int = 100,
    timeout_seconds: float = 5.0,
    result_max_bytes: int = 4_194_304,
) -> Any:
    """Build explicit limits so worker tests never inherit ambient PDF settings."""

    return Settings(
        pdf_max_pages=max_pages,
        pdf_parse_timeout_seconds=timeout_seconds,
        pdf_worker_cpu_seconds=10,
        pdf_worker_memory_bytes=536_870_912,
        pdf_worker_result_max_bytes=result_max_bytes,
    ).pdf_policy()


def write_compressed_text_pdf(path: Path, text_bytes: int) -> None:
    """Create a tiny Flate-compressed PDF whose real extracted text is much larger."""

    write_compressed_pages_pdf(path, [text_bytes])


def write_compressed_pages_pdf(path: Path, page_text_bytes: list[int]) -> None:
    """Create Flate-compressed pages whose extracted text sizes are controlled."""

    writer = PdfWriter()
    for text_bytes in page_text_bytes:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 1 Tf 20 700 Td (" + b"A" * text_bytes + b") Tj ET")
        page[NameObject("/Contents")] = stream.flate_encode()
    with path.open("wb") as handle:
        writer.write(handle)


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


def write_slow_wire_message(connection: Connection, payload: bytes, delay: float) -> None:
    """Send one framed result in pieces to pressure the parent's complete-receipt deadline."""

    try:
        os.write(connection.fileno(), struct.pack("!I", len(payload)) + payload[:1])
        time.sleep(delay)
        os.write(connection.fileno(), payload[1:])
    finally:
        connection.close()


def stall_worker_after_result_header(connection: Connection, *_args: object) -> None:
    """Begin a valid-sized frame, then stall until the production parent kills this process."""

    try:
        os.write(connection.fileno(), struct.pack("!I", 128) + b"{")
        time.sleep(5.0)
    finally:
        connection.close()


def probe_real_page_extraction(connection: Connection, *worker_args: object) -> None:
    """Record real pypdf page extractions and child allocations around production worker code."""

    from pypdf._page import PageObject

    worker_module = pdf_worker()
    extraction_path = Path(os.environ["INVOICE_TEST_EXTRACTION_PROBE_PATH"])
    allocation_path = Path(os.environ["INVOICE_TEST_ALLOCATION_PROBE_PATH"])
    original_extract_text = PageObject.extract_text
    original_send = worker_module._send
    extraction_count = 0

    def recording_extract_text(page: PageObject, *args: Any, **kwargs: Any) -> str:
        nonlocal extraction_count
        extraction_count += 1
        extraction_path.write_text(str(extraction_count), encoding="ascii")
        return original_extract_text(page, *args, **kwargs)

    def record_peak_before_failure_is_published(
        child_connection: Connection,
        payload: dict[str, object],
        max_bytes: int,
    ) -> None:
        if payload.get("stop_reason") == "PDF_WORKER_FAILED":
            _, peak = tracemalloc.get_traced_memory()
            allocation_path.write_text(str(peak), encoding="ascii")
        original_send(child_connection, payload, max_bytes)

    PageObject.extract_text = recording_extract_text
    worker_module._send = record_peak_before_failure_is_published
    tracemalloc.start()
    try:
        worker_module._worker_main(connection, *worker_args)
    finally:
        tracemalloc.stop()


class TrackedPageText(str):
    """Expose when the production extraction loop releases one raw page string."""

    release_path: Path

    def __new__(cls, value: str, release_path: Path) -> TrackedPageText:
        instance = super().__new__(cls, value)
        instance.release_path = release_path
        return instance

    def __del__(self) -> None:
        self.release_path.write_text("released", encoding="ascii")


def probe_page_text_release_during_next_extraction(
    connection: Connection,
    *worker_args: object,
) -> None:
    """Observe page-one liveness from inside page two's real parser call."""

    from pypdf._page import PageObject

    release_path = Path(os.environ["INVOICE_TEST_PAGE_RELEASE_PATH"])
    liveness_path = Path(os.environ["INVOICE_TEST_PAGE_LIVENESS_PATH"])
    original_extract_text = PageObject.extract_text
    extraction_count = 0

    def tracking_extract_text(page: PageObject, *args: Any, **kwargs: Any) -> str:
        nonlocal extraction_count
        extraction_count += 1
        if extraction_count == 2:
            state = "released" if release_path.exists() else "retained"
            liveness_path.write_text(state, encoding="ascii")
        extracted = original_extract_text(page, *args, **kwargs)
        if extraction_count == 1:
            return TrackedPageText(extracted, release_path)
        return extracted

    PageObject.extract_text = tracking_extract_text
    pdf_worker()._worker_main(connection, *worker_args)


def probe_page_text_release_after_append_overflow(
    connection: Connection,
    *worker_args: object,
) -> None:
    """Observe oversized raw-page liveness before the generic failure is sent."""

    from pypdf._page import PageObject

    release_path = Path(os.environ["INVOICE_TEST_PAGE_RELEASE_PATH"])
    liveness_path = Path(os.environ["INVOICE_TEST_PAGE_LIVENESS_PATH"])
    worker_module = pdf_worker()
    original_extract_text = PageObject.extract_text
    original_send = worker_module._send

    def tracking_extract_text(page: PageObject, *args: Any, **kwargs: Any) -> str:
        extracted = original_extract_text(page, *args, **kwargs)
        return TrackedPageText(extracted, release_path)

    def observing_send(
        child_connection: Connection,
        payload: dict[str, object],
        max_bytes: int,
    ) -> None:
        if payload.get("stop_reason") == "PDF_WORKER_FAILED":
            state = "released" if release_path.exists() else "retained"
            liveness_path.write_text(state, encoding="ascii")
        original_send(child_connection, payload, max_bytes)

    PageObject.extract_text = tracking_extract_text
    worker_module._send = observing_send
    worker_module._worker_main(connection, *worker_args)


def send_page_limit_error_then_stall(connection: Connection, *_args: object) -> None:
    """Send a valid page-limit frame and stay alive for parent cleanup verification."""

    payload = json.dumps(
        {
            "ok": False,
            "category": "PARSE",
            "message": "PDF page limit exceeded",
            "stop_reason": "PDF_PAGE_LIMIT_EXCEEDED",
            "details": {"page_count": 101, "max_pages": 100},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    wire_payload = struct.pack("!I", len(payload)) + payload
    try:
        os.write(connection.fileno(), struct.pack("!I", len(wire_payload)) + wire_payload)
        time.sleep(5.0)
    finally:
        connection.close()


def swap_pdf_when_child_parser_opens(
    connection: Connection,
    *worker_args: object,
) -> None:
    """Replace the archive exactly when the real child parser opens its input."""

    operation = worker_args[0]
    source_payload = worker_args[1]
    assert isinstance(operation, str)
    assert isinstance(source_payload, dict)
    source_path = Path(str(source_payload["canonical_path"]))
    swapped = Path(os.environ["INVOICE_TEST_PDF_SWAP_PATH"]).read_bytes()
    if operation in {"inspect", "extract"}:
        import pypdf

        real_pdf_reader = pypdf.PdfReader

        def swapping_pdf_reader(*args: Any, **kwargs: Any) -> Any:
            source_path.write_bytes(swapped)
            return real_pdf_reader(*args, **kwargs)

        pypdf.PdfReader = swapping_pdf_reader  # type: ignore[misc]
    else:
        import fitz

        real_fitz_open = fitz.open

        def swapping_fitz_open(*args: Any, **kwargs: Any) -> Any:
            source_path.write_bytes(swapped)
            return real_fitz_open(*args, **kwargs)

        fitz.open = swapping_fitz_open  # type: ignore[assignment]
    pdf_worker()._worker_main(connection, *worker_args)


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
        pdf_worker().extract_pdf_in_worker(source, pdf_policy(timeout_seconds=0.000_001))

    assert excinfo.value.stop_reason == "PDF_PARSE_TIMEOUT"
    assert child_pids() == before


def test_bounded_receive_deadline_covers_complete_message() -> None:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=write_slow_wire_message,
        args=(child_connection, b'{"ok":true,"result":{}}', 0.25),
    )
    process.start()
    child_connection.close()
    try:
        with pytest.raises(SourceEvidenceError) as excinfo:
            pdf_worker()._receive_bounded_message(
                parent_connection,
                max_bytes=1_024,
                deadline=time.monotonic() + 0.05,
            )
        assert excinfo.value.stop_reason == "PDF_PARSE_TIMEOUT"
    finally:
        if process.is_alive():
            process.terminate()
        process.join(1.0)
        parent_connection.close()
    assert not process.is_alive()


def test_run_worker_automatically_reaps_child_that_stalls_mid_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    worker_module = pdf_worker()
    monkeypatch.setattr(worker_module, "_worker_main", stall_worker_after_result_header)
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module._run_worker("extract", source, pdf_policy(timeout_seconds=0.05))

    assert excinfo.value.stop_reason == "PDF_PARSE_TIMEOUT"
    assert child_pids() == before


def test_error_validation_crossing_deadline_times_out_and_reaps_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    worker_module = pdf_worker()
    original_is_int = worker_module._is_int
    delayed = False

    def delayed_is_int(value: object) -> bool:
        nonlocal delayed
        if not delayed:
            delayed = True
            time.sleep(1.1)
        return original_is_int(value)

    monkeypatch.setattr(worker_module, "_worker_main", send_page_limit_error_then_stall)
    monkeypatch.setattr(worker_module, "_is_int", delayed_is_int)
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module._run_worker("extract", source, pdf_policy(timeout_seconds=1.0))

    assert delayed
    assert excinfo.value.stop_reason == "PDF_PARSE_TIMEOUT"
    assert child_pids() == before


def test_bounded_result_encoding_does_not_allocate_a_second_full_text_copy() -> None:
    worker_module = pdf_worker()
    large_text = "A" * 8_000_000
    payload = {
        "ok": True,
        "result": {
            "pages": [{"page": 1, "text": large_text}],
            "extractor": "pypdf",
            "page_count": 1,
        },
    }
    tracemalloc.start()
    try:
        encoded = worker_module._encode_bounded_message(payload, max_bytes=65_536)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert encoded.text == bytearray()
    assert b"PDF_WORKER_FAILED" in encoded.metadata
    assert peak < 1_000_000


def test_bounded_result_encoding_does_not_allocate_oversized_metadata() -> None:
    worker_module = pdf_worker()
    payload = {
        "ok": False,
        "category": "TOOL",
        "message": "A" * 8_000_000,
        "stop_reason": "PDF_WORKER_FAILED",
    }
    tracemalloc.start()
    try:
        encoded = worker_module._encode_bounded_message(payload, max_bytes=65_536)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert encoded.text == bytearray()
    assert b"PDF_WORKER_FAILED" in encoded.metadata
    assert peak < 1_000_000


def test_write_all_uses_views_without_recopying_remaining_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_module = pdf_worker()
    content = b"A" * 4_000_000

    def partial_write(_file_descriptor: int, remaining: bytes) -> int:
        return min(len(remaining), 4_096)

    monkeypatch.setattr(os, "write", partial_write)
    tracemalloc.start()
    try:
        worker_module._write_all(999, content)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000


def test_oversized_extraction_result_fails_generically_and_reaps_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = tmp_path / "compressed-large-text.pdf"
    write_compressed_text_pdf(submitted, 200_000)
    assert submitted.stat().st_size < 65_536
    source = content_addressed_pdf(submitted, tmp_path / "sources", 1)
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker().extract_pdf_in_worker(source, pdf_policy(result_max_bytes=65_536))

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert excinfo.value.message == "PDF worker failed"
    assert excinfo.value.details is None
    assert child_pids() == before


def test_oversized_first_page_stops_real_extraction_before_later_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = tmp_path / "multi-page-compressed-large-text.pdf"
    write_compressed_pages_pdf(submitted, [200_000, 2_000_000])
    assert submitted.stat().st_size < 65_536
    source = content_addressed_pdf(submitted, tmp_path / "sources", 2)
    extraction_path = tmp_path / "extraction-count.txt"
    allocation_path = tmp_path / "allocation-peak.txt"
    monkeypatch.setenv("INVOICE_TEST_EXTRACTION_PROBE_PATH", str(extraction_path))
    monkeypatch.setenv("INVOICE_TEST_ALLOCATION_PROBE_PATH", str(allocation_path))
    worker_module = pdf_worker()
    monkeypatch.setattr(worker_module, "_worker_main", probe_real_page_extraction)
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module._run_worker(
            "extract",
            source,
            pdf_policy(timeout_seconds=10.0, result_max_bytes=65_536),
        )

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert extraction_path.read_text(encoding="ascii") == "1"
    assert int(allocation_path.read_text(encoding="ascii")) < 4_000_000
    assert child_pids() == before


def test_previous_raw_page_text_is_released_during_next_real_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = tmp_path / "two-fitting-pages.pdf"
    write_compressed_pages_pdf(submitted, [128, 128])
    source = content_addressed_pdf(submitted, tmp_path / "sources", 2)
    release_path = tmp_path / "page-one-released.txt"
    liveness_path = tmp_path / "page-two-observation.txt"
    monkeypatch.setenv("INVOICE_TEST_PAGE_RELEASE_PATH", str(release_path))
    monkeypatch.setenv("INVOICE_TEST_PAGE_LIVENESS_PATH", str(liveness_path))
    worker_module = pdf_worker()
    monkeypatch.setattr(
        worker_module,
        "_worker_main",
        probe_page_text_release_during_next_extraction,
    )
    before = child_pids()

    result = worker_module._run_worker("extract", source, pdf_policy())

    assert result["page_count"] == 2
    assert liveness_path.read_text(encoding="ascii") == "released"
    assert release_path.read_text(encoding="ascii") == "released"
    assert child_pids() == before


def test_raw_page_text_is_released_before_append_overflow_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = tmp_path / "oversized-raw-page.pdf"
    write_compressed_text_pdf(submitted, 40_000)
    assert submitted.stat().st_size < 65_536
    source = content_addressed_pdf(submitted, tmp_path / "sources", 1)
    release_path = tmp_path / "oversized-page-released.txt"
    liveness_path = tmp_path / "failure-send-observation.txt"
    monkeypatch.setenv("INVOICE_TEST_PAGE_RELEASE_PATH", str(release_path))
    monkeypatch.setenv("INVOICE_TEST_PAGE_LIVENESS_PATH", str(liveness_path))
    worker_module = pdf_worker()
    monkeypatch.setattr(
        worker_module,
        "_worker_main",
        probe_page_text_release_after_append_overflow,
    )
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module._run_worker(
            "extract",
            source,
            pdf_policy(result_max_bytes=65_536),
        )

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert liveness_path.read_text(encoding="ascii") == "released"
    assert release_path.read_text(encoding="ascii") == "released"
    assert child_pids() == before


def test_near_ceiling_result_drains_without_pipe_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = tmp_path / "near-ceiling.pdf"
    write_compressed_text_pdf(submitted, 200_000)
    source = content_addressed_pdf(submitted, tmp_path / "sources", 1)
    result = pdf_worker().extract_pdf_in_worker(
        source,
        pdf_policy(result_max_bytes=262_144),
    )

    pages = result["pages"]
    assert isinstance(pages, list)
    assert len(pages[0]["text"]) == 200_000


def test_pdf_parser_crash_is_silent_generic_and_reaps_worker(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    submitted = tmp_path / "corrupt-secret.pdf"
    submitted.write_bytes(FIXTURE_DIR.joinpath("corrupt.pdf").read_bytes() + b"SECRET_SENTINEL")
    source = content_addressed_pdf(submitted, tmp_path / "sources", None)
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker().extract_pdf_in_worker(source, pdf_policy())

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert excinfo.value.message == "PDF worker failed"
    captured = capfd.readouterr()
    visible = captured.out + captured.err + str(excinfo.value)
    assert "SECRET_SENTINEL" not in visible
    assert "Traceback" not in visible
    assert "incorrect startxref" not in visible
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
    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker().extract_pdf_in_worker(source, pdf_policy(max_pages=1))

    assert excinfo.value.stop_reason == "PDF_PAGE_LIMIT_EXCEEDED"
    assert excinfo.value.details == {"page_count": 2, "max_pages": 1}


def test_pdf_worker_extracts_json_safe_page_text(tmp_path: Path) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)

    result = pdf_worker().extract_pdf_in_worker(source, pdf_policy())

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


def test_pdf_child_inspects_the_same_bytes_that_passed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    swapped = tmp_path / "two-pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with swapped.open("wb") as handle:
        writer.write(handle)
    monkeypatch.setenv("INVOICE_TEST_PDF_SWAP_PATH", str(swapped))
    worker_module = pdf_worker()
    monkeypatch.setattr(worker_module, "_worker_main", swap_pdf_when_child_parser_opens)

    page_count = worker_module.inspect_pdf_in_worker(source, pdf_policy())

    assert page_count == 1


def test_pdf_child_extracts_the_same_bytes_that_passed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    monkeypatch.setenv("INVOICE_TEST_PDF_SWAP_PATH", str(DATA_DIR / "invoice_1012.pdf"))
    worker_module = pdf_worker()
    monkeypatch.setattr(worker_module, "_worker_main", swap_pdf_when_child_parser_opens)

    result = worker_module.extract_pdf_in_worker(source, pdf_policy())

    pages = result["pages"]
    assert isinstance(pages, list)
    text = str(pages[0]["text"])
    assert "INV-1011" in text
    assert "INV-1012" not in text


@pytest.mark.parametrize("mutation", ["changed", "missing"])
def test_pdf_child_reports_snapshot_identity_failure_explicitly(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    if mutation == "changed":
        source.canonical_path.write_bytes((DATA_DIR / "invoice_1012.pdf").read_bytes())
    else:
        source.canonical_path.unlink()

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker().extract_pdf_in_worker(source, pdf_policy())

    assert excinfo.value.stop_reason == "SOURCE_HASH_MISMATCH"
    assert excinfo.value.message == "source snapshot identity mismatch"


def test_pdf_worker_renders_page_to_real_png(tmp_path: Path) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    target = tmp_path / "rendered.png"

    result = pdf_worker().render_pdf_page_in_worker(
        source,
        page=1,
        target=target,
        pdf_policy=pdf_policy(),
    )

    assert set(result) == {
        "path",
        "page",
        "sha256",
        "renderer",
        "device",
        "inode",
        "file_type",
        "size_bytes",
    }
    assert result["path"] == str(target.resolve())
    assert result["page"] == 1
    assert result["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert result["renderer"] == "PyMuPDF"
    identity = target.stat()
    assert result["device"] == identity.st_dev
    assert result["inode"] == identity.st_ino
    assert result["file_type"] == stat.S_IFMT(identity.st_mode) == stat.S_IFREG
    assert result["size_bytes"] == identity.st_size
    assert identity.st_nlink == 1
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".pdf-render-")]


def test_pdf_parent_rejects_worker_bytes_that_do_not_match_child_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the parent's descriptor digest check must publish the forged bytes."""

    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    target = tmp_path / "rendered.png"
    child_bytes = b"\x89PNG\r\n\x1a\nchild-verified"
    forged_bytes = b"\x89PNG\r\n\x1a\nforged-after-child-digest"
    worker_module = pdf_worker()

    def return_mismatched_worker_bytes(
        operation: str,
        _source: SourceArtifact,
        _pdf_policy: object,
        *,
        page: int | None = None,
    ) -> dict[str, object]:
        assert operation == "render"
        assert page == 1
        child_digest = hashlib.sha256(child_bytes).hexdigest()
        return {
            "page": page,
            "sha256": child_digest,
            "renderer": "PyMuPDF",
            "png_bytes": forged_bytes,
        }

    monkeypatch.setattr(worker_module, "_run_worker", return_mismatched_worker_bytes)

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module.render_pdf_page_in_worker(
            source,
            page=1,
            target=target,
            pdf_policy=pdf_policy(),
        )

    assert excinfo.value.stop_reason == "REVIEW_PAGE_PUBLICATION_INVALID"
    assert excinfo.value.message == "rendered review page publication failed validation"
    assert not target.exists()


def test_pdf_review_publication_rejects_symlinked_parent_namespace(tmp_path: Path) -> None:
    """Resolving the target before publication must follow and authorize this symlink."""

    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    outside = tmp_path / "outside"
    (outside / "reviews").mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SourceEvidenceError) as excinfo:
        render_pdf_page(
            source,
            1,
            artifacts / "reviews" / "rev_symlinked_parent",
            pdf_policy(),
        )

    assert excinfo.value.stop_reason == "REVIEW_PAGE_PUBLICATION_INVALID"
    assert not list((outside / "reviews" / "rev_symlinked_parent").glob("*.png"))


def test_pdf_parent_rejects_render_candidate_over_configured_byte_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    target = tmp_path / "oversized.png"
    worker_module = pdf_worker()
    oversized = b"P" * 65_537

    def render_oversized_candidate(
        operation: str,
        _source: SourceArtifact,
        _pdf_policy: object,
        *,
        page: int | None = None,
    ) -> dict[str, object]:
        assert operation == "render"
        assert page == 1
        return {
            "page": page,
            "sha256": hashlib.sha256(oversized).hexdigest(),
            "renderer": "PyMuPDF",
            "png_bytes": oversized,
        }

    monkeypatch.setattr(worker_module, "_run_worker", render_oversized_candidate)

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module.render_pdf_page_in_worker(
            source,
            page=1,
            target=target,
            pdf_policy=pdf_policy(result_max_bytes=65_536),
        )

    assert excinfo.value.stop_reason == "REVIEW_PAGE_PUBLICATION_INVALID"
    assert not target.exists()


def test_pdf_parent_rejects_render_directory_swap_after_child_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    review_dir = tmp_path / "artifacts" / "reviews" / "rev_parent_swap"
    target = review_dir / "rendered.png"
    displaced = review_dir.with_name("displaced-review")
    worker_module = pdf_worker()
    child_bytes = b"\x89PNG\r\n\x1a\nparent-swap"

    def swap_parent_after_child_digest(
        operation: str,
        _source: SourceArtifact,
        _pdf_policy: object,
        *,
        page: int | None = None,
    ) -> dict[str, object]:
        assert operation == "render"
        assert page == 1
        digest = hashlib.sha256(child_bytes).hexdigest()
        review_dir.rename(displaced)
        review_dir.mkdir()
        return {
            "page": page,
            "sha256": digest,
            "renderer": "PyMuPDF",
            "png_bytes": child_bytes,
        }

    monkeypatch.setattr(worker_module, "_run_worker", swap_parent_after_child_digest)

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module.render_pdf_page_in_worker(
            source,
            page=1,
            target=target,
            pdf_policy=pdf_policy(),
        )

    assert excinfo.value.stop_reason == "REVIEW_PAGE_PUBLICATION_INVALID"
    assert not target.exists()
    assert not (displaced / target.name).exists()


def test_pdf_child_has_no_path_capability_when_parent_is_swapped_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    review_dir = tmp_path / "artifacts" / "reviews" / "rev_prewrite_parent_swap"
    target = review_dir / "rendered.png"
    displaced = review_dir.with_name("displaced-prewrite-review")
    worker_module = pdf_worker()
    child_bytes = b"\x89PNG\r\n\x1a\ncapability-owned-render"

    def swap_before_child_write(
        operation: str,
        _source: SourceArtifact,
        _pdf_policy: object,
        *,
        page: int | None = None,
        **legacy_path_capability: object,
    ) -> dict[str, object]:
        assert operation == "render"
        assert page == 1
        review_dir.rename(displaced)
        review_dir.mkdir()
        render_target = legacy_path_capability.get("render_target")
        if render_target is not None:
            Path(str(render_target)).write_bytes(child_bytes)
        return {
            "page": page,
            "sha256": hashlib.sha256(child_bytes).hexdigest(),
            "renderer": "PyMuPDF",
            "png_bytes": child_bytes,
        }

    monkeypatch.setattr(worker_module, "_run_worker", swap_before_child_write)

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module.render_pdf_page_in_worker(
            source,
            page=1,
            target=target,
            pdf_policy=pdf_policy(),
        )

    assert excinfo.value.stop_reason == "REVIEW_PAGE_PUBLICATION_INVALID"
    for directory in (review_dir, displaced):
        assert not (directory / target.name).exists()
        assert not list(directory.glob(".pdf-render-*"))


def test_render_cleanup_preserves_primary_control_and_attempts_every_owned_resource_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    target = tmp_path / "reviews" / "rev_cleanup" / "rendered.png"
    worker_module = pdf_worker()
    child_bytes = b"\x89PNG\r\n\x1a\ncleanup-owned-render"
    opened: list[int] = []
    close_attempts: list[int] = []
    unlink_attempts: list[str] = []
    installed_open = os.open
    installed_close = os.close
    installed_unlink = os.unlink
    injected = KeyboardInterrupt("render publication interrupted after candidate creation")

    def render_bytes(
        operation: str,
        _source: SourceArtifact,
        _pdf_policy: object,
        *,
        page: int | None = None,
        **legacy_path_capability: object,
    ) -> dict[str, object]:
        assert operation == "render"
        assert page == 1
        render_target = legacy_path_capability.get("render_target")
        if render_target is not None:
            Path(str(render_target)).write_bytes(child_bytes)
        return {
            "page": page,
            "sha256": hashlib.sha256(child_bytes).hexdigest(),
            "renderer": "PyMuPDF",
            "png_bytes": child_bytes,
        }

    def track_open(*args: object, **kwargs: object) -> int:
        descriptor = installed_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(descriptor)
        return descriptor

    close_fault_injected = False

    def fault_one_close(descriptor: int) -> None:
        nonlocal close_fault_injected
        if descriptor not in opened:
            installed_close(descriptor)
            return
        close_attempts.append(descriptor)
        installed_close(descriptor)
        if not close_fault_injected:
            close_fault_injected = True
            raise OSError("injected owned descriptor close failure")

    unlink_fault_injected = False

    def fault_one_unlink(path: object, *args: object, **kwargs: object) -> None:
        nonlocal unlink_fault_injected
        name = os.fspath(path)
        if isinstance(name, bytes):
            name = os.fsdecode(name)
        if not name.startswith(".pdf-render-"):
            installed_unlink(path, *args, **kwargs)  # type: ignore[arg-type]
            return
        unlink_attempts.append(name)
        installed_unlink(path, *args, **kwargs)  # type: ignore[arg-type]
        if not unlink_fault_injected:
            unlink_fault_injected = True
            raise OSError("injected owned candidate unlink failure")

    monkeypatch.setattr(worker_module, "_run_worker", render_bytes)
    monkeypatch.setattr(worker_module.os, "open", track_open)
    monkeypatch.setattr(worker_module.os, "close", fault_one_close)
    monkeypatch.setattr(worker_module.os, "unlink", fault_one_unlink)

    def interrupt_link(*_args: object, **_kwargs: object) -> None:
        raise injected

    monkeypatch.setattr(worker_module.os, "link", interrupt_link)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        worker_module.render_pdf_page_in_worker(
            source,
            page=1,
            target=target,
            pdf_policy=pdf_policy(),
        )

    assert excinfo.value is injected
    assert Counter(close_attempts) == Counter({descriptor: 1 for descriptor in opened})
    assert len(unlink_attempts) == 1
    assert not target.exists()
    assert not list(target.parent.glob(".pdf-render-*"))


@pytest.mark.parametrize("fault_ordinal", [0, 1], ids=["artifact", "directory"])
def test_render_close_fault_rolls_back_published_name_and_closes_every_descriptor_once(
    fault_ordinal: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    target = tmp_path / "reviews" / "rev_close_fault" / "rendered.png"
    worker_module = pdf_worker()
    png_bytes = b"\x89PNG\r\n\x1a\nclose-fault-render"
    opened: list[int] = []
    close_attempts: list[int] = []
    installed_open = os.open
    installed_close = os.close

    def render_bytes(
        operation: str,
        _source: SourceArtifact,
        _pdf_policy: object,
        *,
        page: int | None = None,
    ) -> dict[str, object]:
        assert operation == "render"
        return {
            "page": page,
            "sha256": hashlib.sha256(png_bytes).hexdigest(),
            "renderer": "PyMuPDF",
            "png_bytes": png_bytes,
        }

    def track_open(*args: object, **kwargs: object) -> int:
        descriptor = installed_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(descriptor)
        return descriptor

    tracked_close_ordinal = 0

    def fault_selected_close(descriptor: int) -> None:
        nonlocal tracked_close_ordinal
        if descriptor not in opened:
            installed_close(descriptor)
            return
        close_attempts.append(descriptor)
        installed_close(descriptor)
        current = tracked_close_ordinal
        tracked_close_ordinal += 1
        if current == fault_ordinal:
            raise OSError(f"injected close failure {fault_ordinal}")

    monkeypatch.setattr(worker_module, "_run_worker", render_bytes)
    monkeypatch.setattr(worker_module.os, "open", track_open)
    monkeypatch.setattr(worker_module.os, "close", fault_selected_close)

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module.render_pdf_page_in_worker(
            source,
            page=1,
            target=target,
            pdf_policy=pdf_policy(),
        )

    assert excinfo.value.stop_reason == "REVIEW_PAGE_PUBLICATION_INVALID"
    assert Counter(close_attempts) == Counter({descriptor: 1 for descriptor in opened})
    assert not target.exists()
    assert not list(target.parent.glob(".pdf-render-*"))


def test_parent_close_that_succeeds_then_raises_uses_live_rollback_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_module = pdf_worker()
    target = tmp_path / "reviews" / "rev_parent_close" / "rendered.png"
    target.parent.mkdir(parents=True)
    descriptors, relationships = worker_module._open_render_parent(target)
    parent_owner = descriptors[-1]
    owned_descriptors = list(descriptors)
    close_attempts: list[int] = []
    png_bytes = b"\x89PNG\r\n\x1a\nparent-close-render"
    installed_open = os.open
    installed_close = os.close
    installed_dup = os.dup
    injected = OSError("parent close completed before reporting failure")

    def track_candidate_open(*args: object, **kwargs: object) -> int:
        descriptor = installed_open(*args, **kwargs)  # type: ignore[arg-type]
        owned_descriptors.append(descriptor)
        return descriptor

    def track_rollback_dup(descriptor: int) -> int:
        duplicate = installed_dup(descriptor)
        owned_descriptors.append(duplicate)
        return duplicate

    def close_parent_then_raise(descriptor: int) -> None:
        if descriptor in owned_descriptors:
            close_attempts.append(descriptor)
        installed_close(descriptor)
        if descriptor == parent_owner:
            raise injected

    monkeypatch.setattr(worker_module.os, "open", track_candidate_open)
    monkeypatch.setattr(worker_module.os, "dup", track_rollback_dup)
    monkeypatch.setattr(worker_module.os, "close", close_parent_then_raise)

    with pytest.raises(BaseException) as excinfo:
        worker_module._publish_verified_render(
            target,
            ".pdf-render-parent-close.png",
            png_bytes,
            hashlib.sha256(png_bytes).hexdigest(),
            max_bytes=4_194_304,
            descriptors=descriptors,
            relationships=relationships,
        )

    assert excinfo.value is injected
    assert Counter(close_attempts) == Counter({descriptor: 1 for descriptor in owned_descriptors})
    assert not target.exists()
    assert not list(target.parent.glob(".pdf-render-*"))


def test_rollback_capability_close_fault_cannot_mask_primary_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_module = pdf_worker()
    target = tmp_path / "reviews" / "rev_rollback_close" / "rendered.png"
    target.parent.mkdir(parents=True)
    descriptors, relationships = worker_module._open_render_parent(target)
    parent_owner = descriptors[-1]
    owned_descriptors = list(descriptors)
    close_attempts: list[int] = []
    png_bytes = b"\x89PNG\r\n\x1a\nrollback-close-render"
    installed_open = os.open
    installed_close = os.close
    installed_dup = os.dup
    installed_fsync = os.fsync
    primary = KeyboardInterrupt("publication interrupted after binding")
    rollback_close = OSError("rollback capability close failed after closing")
    rollback_descriptor: int | None = None
    rollback_close_injected = False

    def track_candidate_open(*args: object, **kwargs: object) -> int:
        descriptor = installed_open(*args, **kwargs)  # type: ignore[arg-type]
        owned_descriptors.append(descriptor)
        return descriptor

    def track_rollback_dup(descriptor: int) -> int:
        nonlocal rollback_descriptor
        rollback_descriptor = installed_dup(descriptor)
        owned_descriptors.append(rollback_descriptor)
        return rollback_descriptor

    def close_rollback_then_raise(descriptor: int) -> None:
        nonlocal rollback_close_injected
        if descriptor in owned_descriptors:
            close_attempts.append(descriptor)
        installed_close(descriptor)
        if descriptor == rollback_descriptor:
            rollback_close_injected = True
            raise rollback_close

    def interrupt_parent_fsync(descriptor: int) -> None:
        if descriptor == parent_owner:
            raise primary
        installed_fsync(descriptor)

    monkeypatch.setattr(worker_module.os, "open", track_candidate_open)
    monkeypatch.setattr(worker_module.os, "dup", track_rollback_dup)
    monkeypatch.setattr(worker_module.os, "close", close_rollback_then_raise)
    monkeypatch.setattr(worker_module.os, "fsync", interrupt_parent_fsync)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        worker_module._publish_verified_render(
            target,
            ".pdf-render-rollback-close.png",
            png_bytes,
            hashlib.sha256(png_bytes).hexdigest(),
            max_bytes=4_194_304,
            descriptors=descriptors,
            relationships=relationships,
        )

    assert excinfo.value is primary
    assert rollback_close_injected
    assert Counter(close_attempts) == Counter({descriptor: 1 for descriptor in owned_descriptors})
    assert not target.exists()
    assert not list(target.parent.glob(".pdf-render-*"))


def test_final_capability_close_after_close_returns_bound_artifact_with_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_module = pdf_worker()
    target = tmp_path / "reviews" / "rev_final_close" / "rendered.png"
    target.parent.mkdir(parents=True)
    descriptors, relationships = worker_module._open_render_parent(target)
    owned_descriptors = list(descriptors)
    close_attempts: list[int] = []
    png_bytes = b"\x89PNG\r\n\x1a\nfinal-capability-close-render"
    installed_open = os.open
    installed_close = os.close
    installed_dup = os.dup
    injected = OSError("final rollback capability closed before reporting failure")
    rollback_descriptor: int | None = None

    def track_candidate_open(*args: object, **kwargs: object) -> int:
        descriptor = installed_open(*args, **kwargs)  # type: ignore[arg-type]
        owned_descriptors.append(descriptor)
        return descriptor

    def track_rollback_dup(descriptor: int) -> int:
        nonlocal rollback_descriptor
        rollback_descriptor = installed_dup(descriptor)
        owned_descriptors.append(rollback_descriptor)
        return rollback_descriptor

    def close_final_capability_then_raise(descriptor: int) -> None:
        if descriptor in owned_descriptors:
            close_attempts.append(descriptor)
        installed_close(descriptor)
        if descriptor == rollback_descriptor:
            raise injected

    monkeypatch.setattr(worker_module.os, "open", track_candidate_open)
    monkeypatch.setattr(worker_module.os, "dup", track_rollback_dup)
    monkeypatch.setattr(worker_module.os, "close", close_final_capability_then_raise)

    publication = worker_module._publish_verified_render(
        target,
        ".pdf-render-final-close.png",
        png_bytes,
        hashlib.sha256(png_bytes).hexdigest(),
        max_bytes=4_194_304,
        descriptors=descriptors,
        relationships=relationships,
    )

    assert publication.close_diagnostic is injected
    target_identity = target.stat()
    assert publication.identity == (
        target_identity.st_dev,
        target_identity.st_ino,
        stat.S_IFMT(target_identity.st_mode),
        target_identity.st_size,
    )
    assert target.read_bytes() == png_bytes
    assert target_identity.st_nlink == 1
    assert Counter(close_attempts) == Counter({descriptor: 1 for descriptor in owned_descriptors})
    assert not list(target.parent.glob(".pdf-render-*"))


@pytest.mark.parametrize(
    "warning_action",
    ["always", "error"],
    ids=["normal-policy", "warnings-as-errors"],
)
def test_public_render_reports_final_close_diagnostic_without_unbinding_artifact(
    warning_action: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = content_addressed_pdf(DATA_DIR / "invoice_1011.pdf", tmp_path / "sources", 1)
    target = tmp_path / "reviews" / "rev_public_final_close" / "rendered.png"
    worker_module = pdf_worker()
    png_bytes = b"\x89PNG\r\n\x1a\npublic-final-close-render"
    installed_close = os.close
    installed_dup = os.dup
    injected = OSError("public final capability close diagnostic")
    rollback_descriptor: int | None = None

    def render_bytes(
        operation: str,
        _source: SourceArtifact,
        _pdf_policy: object,
        *,
        page: int | None = None,
    ) -> dict[str, object]:
        assert operation == "render"
        return {
            "page": page,
            "sha256": hashlib.sha256(png_bytes).hexdigest(),
            "renderer": "PyMuPDF",
            "png_bytes": png_bytes,
        }

    def track_rollback_dup(descriptor: int) -> int:
        nonlocal rollback_descriptor
        rollback_descriptor = installed_dup(descriptor)
        return rollback_descriptor

    def close_final_capability_then_raise(descriptor: int) -> None:
        installed_close(descriptor)
        if descriptor == rollback_descriptor:
            raise injected

    monkeypatch.setattr(worker_module, "_run_worker", render_bytes)
    monkeypatch.setattr(worker_module.os, "dup", track_rollback_dup)
    monkeypatch.setattr(worker_module.os, "close", close_final_capability_then_raise)
    caplog.set_level(logging.WARNING, logger=worker_module.__name__)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter(warning_action, worker_module.RenderPublicationCloseWarning)
        result = worker_module.render_pdf_page_in_worker(
            source,
            page=1,
            target=target,
            pdf_policy=pdf_policy(),
        )

    if warning_action == "always":
        assert len(captured) == 1
        assert captured[0].message.diagnostic is injected
    else:
        assert captured == []
    diagnostic_records = [
        record
        for record in caplog.records
        if record.name == worker_module.__name__
        and record.getMessage().startswith("render_publication_close_diagnostic")
    ]
    assert [record.getMessage() for record in diagnostic_records] == [
        "render_publication_close_diagnostic diagnostic_type=OSError"
    ]
    assert str(injected) not in caplog.text
    assert isinstance(result, worker_module.RenderedPageResult)
    assert result.publication_diagnostic is injected
    assert set(result) == {
        "path",
        "page",
        "sha256",
        "renderer",
        "device",
        "inode",
        "file_type",
        "size_bytes",
    }
    assert result["inode"] == target.stat().st_ino
    assert result["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not list(target.parent.glob(".pdf-render-*"))


def test_pdf_child_renders_the_same_bytes_that_passed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_source = content_addressed_pdf(
        DATA_DIR / "invoice_1011.pdf",
        tmp_path / "baseline-sources",
        1,
    )
    baseline_target = tmp_path / "baseline.png"
    baseline = pdf_worker().render_pdf_page_in_worker(
        baseline_source,
        page=1,
        target=baseline_target,
        pdf_policy=pdf_policy(),
    )
    source = content_addressed_pdf(
        DATA_DIR / "invoice_1011.pdf",
        tmp_path / "mutated-sources",
        1,
    )
    monkeypatch.setenv("INVOICE_TEST_PDF_SWAP_PATH", str(DATA_DIR / "invoice_1012.pdf"))
    worker_module = pdf_worker()
    monkeypatch.setattr(worker_module, "_worker_main", swap_pdf_when_child_parser_opens)
    target = tmp_path / "mutated.png"

    rendered = worker_module.render_pdf_page_in_worker(
        source,
        page=1,
        target=target,
        pdf_policy=pdf_policy(),
    )

    assert rendered["sha256"] == baseline["sha256"]


def test_pdf_worker_memory_allowance_rejects_excessive_configuration() -> None:
    with pytest.raises(ValidationError):
        Settings(pdf_worker_memory_bytes=1_073_741_825)


@pytest.mark.parametrize(
    ("operation", "payload", "expected_page"),
    [
        ("inspect", {"ok": True, "result": {"page_count": "1"}}, None),
        (
            "extract",
            {
                "ok": True,
                "result": {"pages": "not-a-list", "extractor": "pypdf", "page_count": 1},
            },
            None,
        ),
        (
            "render",
            {
                "ok": True,
                "result": {"page": 1, "sha256": None, "renderer": "PyMuPDF"},
            },
            1,
        ),
    ],
)
def test_invalid_operation_result_envelopes_fail_generically(
    operation: str,
    payload: dict[str, object],
    expected_page: int | None,
) -> None:
    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker()._decode_worker_message(
            operation,
            json.dumps(payload).encode("utf-8"),
            expected_page=expected_page,
        )

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert excinfo.value.message == "PDF worker failed"
    assert excinfo.value.details is None


def test_unknown_error_category_fails_generically() -> None:
    payload = {
        "ok": False,
        "category": "NOT_A_CATEGORY",
        "message": "unsafe child detail",
        "stop_reason": "UNSAFE_REASON",
    }

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker()._decode_worker_message(
            "extract", json.dumps(payload).encode("utf-8"), expected_page=None
        )

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert excinfo.value.message == "PDF worker failed"
    assert "unsafe child detail" not in str(excinfo.value)


def test_known_failure_category_cannot_smuggle_raw_child_message() -> None:
    payload = {
        "ok": False,
        "category": "TOOL",
        "message": "SECRET_SENTINEL raw parser exception",
        "stop_reason": "PDF_WORKER_FAILED",
    }

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker()._decode_worker_message(
            "extract", json.dumps(payload).encode("utf-8"), expected_page=None
        )

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert excinfo.value.message == "PDF worker failed"
    assert "SECRET_SENTINEL" not in str(excinfo.value)


def test_error_envelope_rejects_disallowed_empty_details() -> None:
    payload = {
        "ok": False,
        "category": "PARSE",
        "message": "PDF contains no extractable text",
        "stop_reason": "PDF_TEXT_EMPTY",
        "details": {},
    }

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker()._decode_worker_message(
            "extract", json.dumps(payload).encode("utf-8"), expected_page=None
        )

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert excinfo.value.message == "PDF worker failed"


@pytest.mark.parametrize("page_number", [True, 1.0])
def test_extract_page_number_requires_a_strict_integer(page_number: object) -> None:
    payload = {
        "ok": True,
        "result": {
            "pages": [{"page": page_number, "text_bytes": 7}],
            "extractor": "pypdf",
            "page_count": 1,
        },
    }

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker()._decode_worker_message(
            "extract",
            json.dumps(payload).encode("utf-8"),
            expected_page=None,
            text_payload=b"invoice",
        )

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"


@pytest.mark.parametrize(
    ("operation", "result"),
    [
        ("inspect", {"page_count": 2}),
        (
            "extract",
            {
                "pages": [
                    {"page": 1, "text_bytes": 1},
                    {"page": 2, "text_bytes": 1},
                ],
                "extractor": "pypdf",
                "page_count": 2,
            },
        ),
    ],
)
def test_success_envelope_cannot_exceed_configured_page_limit(
    operation: str,
    result: dict[str, object],
) -> None:
    payload = {"ok": True, "result": result}

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker()._decode_worker_message(
            operation,
            json.dumps(payload).encode("utf-8"),
            expected_page=None,
            max_pages=1,
        )

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"


def test_recursive_metadata_decode_failure_is_generic() -> None:
    recursive_json = b"[" * 10_000 + b"]" * 10_000

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker()._decode_worker_message("extract", recursive_json, expected_page=None)

    assert excinfo.value.stop_reason == "PDF_WORKER_FAILED"
    assert excinfo.value.message == "PDF worker failed"


def test_incremental_text_decode_enforces_deadline_during_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        "ok": True,
        "result": {
            "pages": [{"page": 1, "text_bytes": 4_000_000}],
            "extractor": "pypdf",
            "page_count": 1,
        },
    }

    monotonic_values = iter((0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic_values, 2.0))

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker()._decode_worker_message(
            "extract",
            json.dumps(metadata).encode("utf-8"),
            expected_page=None,
            max_pages=100,
            text_payload=b"A" * 4_000_000,
            deadline=1.0,
        )

    assert excinfo.value.stop_reason == "PDF_PARSE_TIMEOUT"
