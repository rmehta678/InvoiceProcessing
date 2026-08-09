"""Real-process tests for bounded, fail-closed PDF operations."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import struct
import time
import tracemalloc
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

    extraction_path = Path(os.environ["INVOICE_TEST_EXTRACTION_PROBE_PATH"])
    allocation_path = Path(os.environ["INVOICE_TEST_ALLOCATION_PROBE_PATH"])
    original_extract_text = PageObject.extract_text
    extraction_count = 0

    def recording_extract_text(page: PageObject, *args: Any, **kwargs: Any) -> str:
        nonlocal extraction_count
        extraction_count += 1
        extraction_path.write_text(str(extraction_count), encoding="ascii")
        return original_extract_text(page, *args, **kwargs)

    PageObject.extract_text = recording_extract_text
    tracemalloc.start()
    try:
        pdf_worker()._worker_main(connection, *worker_args)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        allocation_path.write_text(str(peak), encoding="ascii")


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
        worker_module._run_worker("extract", source, timeout_seconds=0.05)

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
        worker_module._run_worker("extract", source, timeout_seconds=1.0)

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
    monkeypatch.setenv("INVOICE_PDF_WORKER_RESULT_MAX_BYTES", "65536")
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        pdf_worker().extract_pdf_in_worker(source, timeout_seconds=5.0)

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
    monkeypatch.setenv("INVOICE_PDF_WORKER_RESULT_MAX_BYTES", "65536")
    monkeypatch.setenv("INVOICE_TEST_EXTRACTION_PROBE_PATH", str(extraction_path))
    monkeypatch.setenv("INVOICE_TEST_ALLOCATION_PROBE_PATH", str(allocation_path))
    worker_module = pdf_worker()
    monkeypatch.setattr(worker_module, "_worker_main", probe_real_page_extraction)
    before = child_pids()

    with pytest.raises(SourceEvidenceError) as excinfo:
        worker_module._run_worker("extract", source, timeout_seconds=10.0)

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

    result = worker_module._run_worker("extract", source, timeout_seconds=5.0)

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
    monkeypatch.setenv("INVOICE_PDF_WORKER_RESULT_MAX_BYTES", "65536")
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
        worker_module._run_worker("extract", source, timeout_seconds=5.0)

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
    monkeypatch.setenv("INVOICE_PDF_WORKER_RESULT_MAX_BYTES", "262144")

    result = pdf_worker().extract_pdf_in_worker(source, timeout_seconds=5.0)

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
        pdf_worker().extract_pdf_in_worker(source, timeout_seconds=5.0)

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
        pdf_worker()._decode_worker_message(
            "extract", recursive_json, expected_page=None
        )

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
