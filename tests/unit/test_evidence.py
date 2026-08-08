"""Golden format extraction for every supplied artifact."""

from decimal import Decimal
from pathlib import Path

import pytest

from invoice_agents.errors import SourceEvidenceError
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.evidence import extract_invoice_evidence


def snapshot(path: Path, archive: Path):  # type: ignore[no-untyped-def]
    return snapshot_source(path, archive, max_bytes=10_485_760)


EXPECTED = {
    "invoice_1001.txt": ("INV-1001", 2, Decimal("5000.00")),
    "invoice_1002.txt": ("INV-1002", 1, Decimal("15000.00")),
    "invoice_1003.txt": ("INV-1003", 1, Decimal("100000.00")),
    "invoice_1004.json": ("INV-1004", 2, Decimal("1890.00")),
    "invoice_1004_revised.json": ("INV-1004", 3, Decimal("5940.00")),
    "invoice_1005.json": ("INV-1005", 3, Decimal("15225.00")),
    "invoice_1006.csv": ("INV-1006", 2, Decimal("2750.00")),
    "invoice_1007.csv": ("INV-1007", 3, Decimal("15525.00")),
    "invoice_1008.txt": ("INV-1008", 2, Decimal("9900.00")),
    "invoice_1009.json": ("INV-1009", 2, Decimal("-250.00")),
    "invoice_1010.txt": ("INV-1010", 4, Decimal("7185.00")),
    "invoice_1011.pdf": ("INV-1011", 2, Decimal("3000.00")),
    "invoice_1011.txt": ("INV-1011", 2, Decimal("3000.00")),
    "invoice_1012.pdf": ("INV-1012", 3, Decimal("9975.00")),
    "invoice_1012.txt": ("INV-1012", 3, Decimal("9975.00")),
    "invoice_1013.json": ("INV-1013", 8, Decimal("22562.80")),
    "invoice_1013.pdf": ("INV-1013", 8, Decimal("22562.80")),
    "invoice_1014.xml": ("INV-1014", 2, Decimal("4125.00")),
    "invoice_1015.csv": ("INV-1015", 3, Decimal("6500.00")),
    "invoice_1016.json": ("INV-1016", 3, Decimal("3233.00")),
}


@pytest.mark.parametrize(("filename", "expected"), EXPECTED.items())
def test_all_twenty_artifacts_extract(
    invoice_dir: Path, tmp_path: Path, filename: str, expected: tuple[str, int, Decimal]
) -> None:
    invoice = extract_invoice_evidence(snapshot(invoice_dir / filename, tmp_path / "sources"))
    assert invoice.invoice_number.normalized_value == expected[0]
    assert len(invoice.lines) == expected[1]
    assert invoice.declared_total == expected[2]
    assert invoice.source.sha256
    assert all(line.evidence for line in invoice.lines)
    assert all(
        line.raw_item and line.raw_quantity and line.raw_unit_price for line in invoice.lines
    )


def test_special_raw_normalization_and_missing_fields(invoice_dir: Path, tmp_path: Path) -> None:
    ocr = extract_invoice_evidence(snapshot(invoice_dir / "invoice_1012.txt", tmp_path / "sources"))
    assert ocr.invoice_date.raw_value == "26-Jan-2O26"
    assert ocr.invoice_date.normalized_value == "2026-01-26"
    assert ocr.invoice_date.ambiguity is not None
    invalid = extract_invoice_evidence(
        snapshot(invoice_dir / "invoice_1009.json", tmp_path / "sources")
    )
    assert {"vendor", "due_date", "payment_terms"}.issubset(invalid.missing_fields)
    assert invalid.lines[0].quantity == Decimal("-5")


def test_email_vendor_and_pdf_columns_are_not_conflated(invoice_dir: Path, tmp_path: Path) -> None:
    email = extract_invoice_evidence(
        snapshot(invoice_dir / "invoice_1008.txt", tmp_path / "sources")
    )
    assert email.vendor.normalized_value == "NoProd Industries"
    pdf = extract_invoice_evidence(snapshot(invoice_dir / "invoice_1013.pdf", tmp_path / "sources"))
    assert pdf.vendor.normalized_value == "Atlas Industrial Supply"


def test_unknown_corrupt_and_empty_sources_fail(tmp_path: Path) -> None:
    unsupported = tmp_path / "invoice.bin"
    unsupported.write_bytes(b"x")
    with pytest.raises(SourceEvidenceError, match="unsupported"):
        snapshot(unsupported, tmp_path / "sources")
    broken = tmp_path / "invoice.json"
    broken.write_text("{broken", encoding="utf-8")
    with pytest.raises(SourceEvidenceError, match="JSON parse failed"):
        extract_invoice_evidence(snapshot(broken, tmp_path / "sources"))
    empty = tmp_path / "invoice.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SourceEvidenceError, match="empty"):
        extract_invoice_evidence(snapshot(empty, tmp_path / "sources"))
