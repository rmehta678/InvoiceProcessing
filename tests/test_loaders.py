"""Loader tests: every sample file must reach the agents as readable text."""

from __future__ import annotations

from pathlib import Path

import pytest

from invoice_flow.tools.loaders import (
    DocumentLoadError,
    UnsupportedFormatError,
    load_document,
)


def test_discovers_every_sample_invoice(all_invoice_files: list[Path]) -> None:
    assert len(all_invoice_files) == 20  # 18 invoices, two of which also ship as PDFs


def test_all_samples_load_to_non_empty_text(all_invoice_files: list[Path]) -> None:
    for path in all_invoice_files:
        doc = load_document(path)
        assert doc.text.strip(), f"{path.name} produced empty text"
        assert doc.file_format == path.suffix.lstrip(".")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "nope.txt")


def test_unsupported_format_raises(tmp_path: Path) -> None:
    bad = tmp_path / "invoice.docx"
    bad.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        load_document(bad)


def test_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "invoice.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_document(empty)


def test_csv_preserves_repeated_keys(invoice_dir: Path) -> None:
    """1006 repeats the `item` key; csv.DictReader would keep only the last."""
    text = load_document(invoice_dir / "invoice_1006.csv").text
    assert text.count("item") >= 2
    assert "WidgetA" in text and "WidgetB" in text


def test_csv_keeps_summary_rows(invoice_dir: Path) -> None:
    """1007 puts subtotal/tax/total in trailing rows with empty leading cells."""
    text = load_document(invoice_dir / "invoice_1007.csv").text
    assert "Subtotal:" in text and "Total:" in text
    for item in ("WidgetA", "WidgetB", "GadgetX"):
        assert item in text


def test_pdf_text_extraction_preserves_ocr_artifacts(invoice_dir: Path) -> None:
    """The messy PDF's artifacts must survive to the extractor, not be silently
    cleaned up -- the agent has to demonstrate it can cope with them."""
    text = load_document(invoice_dir / "invoice_1012.pdf").text
    assert "2O26" in text  # letter O standing in for a zero
    assert "$3,500.O0" in text
    assert "Widget A" in text  # spaced item name


def test_pdf_and_txt_twins_carry_the_same_line_items(invoice_dir: Path) -> None:
    pdf = load_document(invoice_dir / "invoice_1011.pdf").text
    txt = load_document(invoice_dir / "invoice_1011.txt").text
    for token in ("INV-1011", "WidgetA", "WidgetB"):
        assert token in pdf and token in txt


def test_xml_reaches_the_agent_as_text(invoice_dir: Path) -> None:
    text = load_document(invoice_dir / "invoice_1014.xml").text
    assert text.count("<item>") == 2
    assert "EUR" in text


def test_malformed_json_still_loads_as_text(tmp_path: Path) -> None:
    """A truncated file is an invoice problem, not a reason to refuse it."""
    broken = tmp_path / "invoice.json"
    broken.write_text('{"invoice_number": "INV-9", "total":', encoding="utf-8")
    assert "INV-9" in load_document(broken).text
