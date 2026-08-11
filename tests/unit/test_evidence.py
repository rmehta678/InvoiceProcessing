"""Golden format extraction for every supplied artifact."""

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_agents.errors import SourceEvidenceError
from invoice_agents.models import SourceArtifact
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


def write_text_invoice(
    tmp_path: Path,
    *,
    quantity: str = "2",
    unit_price: str = "$50.00",
    declared_line_total: str = "$100.00",
    total: str = "$100.00",
    notes_column: bool = False,
    note: str = "",
    document_fields: tuple[str, ...] = (),
) -> SourceArtifact:
    """Create a complete, immutable text invoice with one controllable numeric field."""

    path = tmp_path / "numeric-evidence.txt"
    lines = [
        "INVOICE",
        "Vendor: Numeric Supplies",
        "Invoice Number: INV-4242",
        "Date: 2026-01-15",
        "Due Date: 2026-02-01",
        "",
    ]
    if notes_column:
        lines.append("Item Qty Unit Price Amount Notes")
    lines.extend((f"WidgetA qty: {quantity} unit price: {unit_price} {declared_line_total}{note}",))
    lines.extend(document_fields)
    lines.append(f"Total: {total}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return snapshot_source(path, tmp_path / "sources", 10_485_760)


def write_row_oriented_tax_csv(
    tmp_path: Path, tax_rows: tuple[tuple[str, str], ...]
) -> SourceArtifact:
    """Create a row-oriented CSV with real tax labels and independently controlled amounts."""

    path = tmp_path / "tax-labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Invoice Number",
                "Vendor",
                "Date",
                "Due Date",
                "Item",
                "Qty",
                "Unit Price",
                "Line Total",
            ]
        )
        writer.writerow(
            [
                "INV-4242",
                "Numeric Supplies",
                "2026-01-15",
                "2026-02-01",
                "WidgetA",
                "2",
                "$50.00",
                "$100.00",
            ]
        )
        for label, amount in tax_rows:
            writer.writerow(["", "", "", "", "", "", label, amount])
        writer.writerow(["", "", "", "", "", "", "Total", "$110.00"])
    return snapshot_source(path, tmp_path / "sources", 10_485_760)


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


@pytest.mark.parametrize("raw", ["100BAD", "$5,000.00oops", "12X34", "100.00 USD extra"])
def test_declared_total_rejects_trailing_content(raw: str, tmp_path: Path) -> None:
    source = write_text_invoice(tmp_path, total=raw)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.category == "PARSE"
    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared total"
    assert excinfo.value.details["locator"] == "line:8"
    assert excinfo.value.details["source_id"] == source.source_id
    assert excinfo.value.details["raw_value"] == raw


@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("quantity", "12X34"),
        ("unit price", "$5,000.00oops"),
        ("declared line total", "100.00 USD extra"),
    ],
)
def test_line_numeric_field_rejects_trailing_content(field: str, raw: str, tmp_path: Path) -> None:
    values = {
        "quantity": "2",
        "unit_price": "$50.00",
        "declared_line_total": "$100.00",
    }
    values[field.replace(" ", "_")] = raw
    source = write_text_invoice(tmp_path, **values)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.category == "PARSE"
    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == field
    assert excinfo.value.details["locator"] == "line:7"
    assert excinfo.value.details["raw_value"] == raw


@pytest.mark.parametrize(
    ("label", "field"),
    [
        ("Subtotal", "declared subtotal"),
        ("Tax (0%)", "declared tax"),
        ("Shipping", "declared fee"),
        ("Total", "declared total"),
    ],
)
def test_document_money_field_rejects_trailing_content(
    label: str, field: str, tmp_path: Path
) -> None:
    raw = "$5,000.00oops"
    kwargs = {"total": raw} if label == "Total" else {"document_fields": (f"{label}: {raw}",)}
    source = write_text_invoice(tmp_path, **kwargs)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == field
    assert excinfo.value.details["raw_value"] == raw


def test_line_note_suffix_requires_a_declared_notes_column(tmp_path: Path) -> None:
    no_notes_column = write_text_invoice(
        tmp_path / "no-notes-column",
        declared_line_total="$100.00Custom promotion",
    )
    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(no_notes_column)
    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared line total"

    notes_column = write_text_invoice(
        tmp_path / "notes-column",
        declared_line_total="$100.00",
        notes_column=True,
        note="Custom promotion",
    )
    invoice = extract_invoice_evidence(notes_column)
    assert invoice.lines[0].declared_line_total == Decimal("100.00")


def test_notes_column_does_not_apply_before_its_table_header(tmp_path: Path) -> None:
    path = tmp_path / "notes-scope.txt"
    path.write_text(
        "\n".join(
            (
                "INVOICE",
                "Vendor: Numeric Supplies",
                "Invoice Number: INV-4242",
                "Date: 2026-01-15",
                "Due Date: 2026-02-01",
                "WidgetA qty: 2 unit price: $50.00 $100.00MALFORMED",
                "",
                "Item Qty Unit Price Amount Notes",
                "WidgetB 2 $50.00 $100.00Arbitrary current-table note",
                "Total: $200.00",
            )
        ),
        encoding="utf-8",
    )
    source = snapshot_source(path, tmp_path / "sources", 10_485_760)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared line total"
    assert excinfo.value.details["locator"] == "line:6"


def test_notes_column_ends_at_the_next_table_header(tmp_path: Path) -> None:
    path = tmp_path / "notes-boundary.txt"
    path.write_text(
        "\n".join(
            (
                "INVOICE",
                "Vendor: Numeric Supplies",
                "Invoice Number: INV-4242",
                "Date: 2026-01-15",
                "Due Date: 2026-02-01",
                "Item Qty Unit Price Amount Notes",
                "WidgetA 2 $50.00 $100.00Arbitrary first-table note",
                "",
                "Item Qty Unit Price Amount",
                "WidgetB 2 $50.00 $100.00MALFORMED",
                "Total: $200.00",
            )
        ),
        encoding="utf-8",
    )
    source = snapshot_source(path, tmp_path / "sources", 10_485_760)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared line total"
    assert excinfo.value.details["locator"] == "line:10"


def test_notes_column_does_not_turn_malformed_comma_grouping_into_a_note(tmp_path: Path) -> None:
    path = tmp_path / "notes-comma.txt"
    path.write_text(
        "\n".join(
            (
                "INVOICE",
                "Vendor: Numeric Supplies",
                "Invoice Number: INV-4242",
                "Date: 2026-01-15",
                "Due Date: 2026-02-01",
                "Item Qty Unit Price Amount Notes",
                "WidgetA 2 $50.00 $1,,00.00Arbitrary note",
                "Total: $100.00",
            )
        ),
        encoding="utf-8",
    )
    source = snapshot_source(path, tmp_path / "sources", 10_485_760)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared line total"
    assert excinfo.value.details["raw_value"] == "$1,,00.00Arbitrary note"


@pytest.mark.parametrize(
    ("values", "expected_quantity", "expected_total"),
    [
        (
            {
                "quantity": "-2",
                "unit_price": "€1,250.00",
                "declared_line_total": "-€2,500.00",
                "total": "-€2,500.00",
            },
            Decimal("-2"),
            Decimal("-2500.00"),
        ),
        (
            {
                "quantity": "1O",
                "unit_price": "$1,2O0.00",
                "declared_line_total": "$12,000.O0",
                "total": "$12,000.O0",
            },
            Decimal("10"),
            Decimal("12000.00"),
        ),
    ],
)
def test_complete_numeric_fields_preserve_supported_formats(
    values: dict[str, str], expected_quantity: Decimal, expected_total: Decimal, tmp_path: Path
) -> None:
    invoice = extract_invoice_evidence(write_text_invoice(tmp_path, **values))

    assert invoice.lines[0].quantity == expected_quantity
    assert invoice.lines[0].declared_line_total == expected_total
    assert invoice.declared_total == expected_total


@pytest.mark.parametrize("raw", ["$1,,00.00", "$100,", "$12,34.00"])
def test_declared_total_rejects_malformed_comma_grouping(raw: str, tmp_path: Path) -> None:
    source = write_text_invoice(tmp_path, total=raw)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared total"
    assert excinfo.value.details["raw_value"] == raw


def test_text_tax_rate_rejects_malformed_content(tmp_path: Path) -> None:
    source = write_text_invoice(
        tmp_path,
        document_fields=("Tax (0BAD%): $0.00",),
    )

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared tax rate"
    assert excinfo.value.details["raw_value"] == "0BAD"


@pytest.mark.parametrize("label", ["Tax (10%oops)", "Tax (10%%)"])
def test_text_tax_rate_rejects_malformed_wrapper(label: str, tmp_path: Path) -> None:
    source = write_text_invoice(
        tmp_path,
        document_fields=(f"{label}: $10.00",),
    )

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared tax rate"
    assert excinfo.value.details["raw_value"] == label


def test_text_tax_rate_rejects_malformed_wrapper_without_amount(tmp_path: Path) -> None:
    source = write_text_invoice(
        tmp_path,
        document_fields=("Tax (10%oops):",),
    )

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared tax rate"
    assert excinfo.value.details["raw_value"] == "Tax (10%oops)"


def test_text_tax_rate_validates_later_malformed_label_after_bare_tax(tmp_path: Path) -> None:
    source = write_text_invoice(
        tmp_path,
        document_fields=("Tax: $0.00", "Tax (10%oops): $10.00"),
    )

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared tax rate"
    assert excinfo.value.details["raw_value"] == "Tax (10%oops)"


def test_text_tax_rate_retains_later_valid_label_after_bare_tax(tmp_path: Path) -> None:
    source = write_text_invoice(
        tmp_path,
        document_fields=("Tax: $0.00", "Tax (10%): $10.00"),
    )

    invoice = extract_invoice_evidence(source)

    assert invoice.declared_tax_rate == Decimal("0.1")


def test_row_oriented_csv_tax_rate_rejects_malformed_content(tmp_path: Path) -> None:
    path = tmp_path / "malformed-tax-rate.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Invoice Number",
                "Vendor",
                "Date",
                "Due Date",
                "Item",
                "Qty",
                "Unit Price",
                "Line Total",
            ]
        )
        writer.writerow(
            [
                "INV-4242",
                "Numeric Supplies",
                "2026-01-15",
                "2026-02-01",
                "WidgetA",
                "2",
                "$50.00",
                "$100.00",
            ]
        )
        writer.writerow(["", "", "", "", "", "", "Tax (0BAD%)", "$0.00"])
        writer.writerow(["", "", "", "", "", "", "Total", "$100.00"])
    source = snapshot_source(path, tmp_path / "sources", 10_485_760)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared tax rate"
    assert excinfo.value.details["locator"] == "row:3"
    assert excinfo.value.details["raw_value"] == "0BAD"


def test_row_oriented_csv_tax_rate_rejects_malformed_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "malformed-tax-wrapper.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Invoice Number",
                "Vendor",
                "Date",
                "Due Date",
                "Item",
                "Qty",
                "Unit Price",
                "Line Total",
            ]
        )
        writer.writerow(
            [
                "INV-4242",
                "Numeric Supplies",
                "2026-01-15",
                "2026-02-01",
                "WidgetA",
                "2",
                "$50.00",
                "$100.00",
            ]
        )
        writer.writerow(["", "", "", "", "", "", "Tax (10%oops)", "$10.00"])
        writer.writerow(["", "", "", "", "", "", "Total", "$110.00"])
    source = snapshot_source(path, tmp_path / "sources", 10_485_760)

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared tax rate"
    assert excinfo.value.details["locator"] == "row:3"
    assert excinfo.value.details["raw_value"] == "Tax (10%oops)"


def test_row_oriented_csv_tax_rate_preserves_unparenthesized_label(tmp_path: Path) -> None:
    path = tmp_path / "unparenthesized-tax-rate.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Invoice Number",
                "Vendor",
                "Date",
                "Due Date",
                "Item",
                "Qty",
                "Unit Price",
                "Line Total",
            ]
        )
        writer.writerow(
            [
                "INV-4242",
                "Numeric Supplies",
                "2026-01-15",
                "2026-02-01",
                "WidgetA",
                "2",
                "$50.00",
                "$100.00",
            ]
        )
        writer.writerow(["", "", "", "", "", "", "Tax 10%", "$10.00"])
        writer.writerow(["", "", "", "", "", "", "Total", "$110.00"])
    source = snapshot_source(path, tmp_path / "sources", 10_485_760)

    invoice = extract_invoice_evidence(source)

    assert invoice.declared_tax_rate == Decimal("0.1")


def test_row_oriented_csv_tax_rate_rejects_malformed_wrapper_without_amount(
    tmp_path: Path,
) -> None:
    source = write_row_oriented_tax_csv(tmp_path, (("Tax (10%oops)", ""),))

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_invoice_evidence(source)

    assert excinfo.value.stop_reason == "MALFORMED_MONEY_FIELD"
    assert excinfo.value.details is not None
    assert excinfo.value.details["field"] == "declared tax rate"
    assert excinfo.value.details["locator"] == "row:3"
    assert excinfo.value.details["raw_value"] == "Tax (10%oops)"


def test_row_oriented_csv_tax_rate_does_not_overwrite_valid_rate_with_bare_tax(
    tmp_path: Path,
) -> None:
    source = write_row_oriented_tax_csv(
        tmp_path,
        (("Tax 10%", "$10.00"), ("Tax", "$0.00")),
    )

    invoice = extract_invoice_evidence(source)

    assert invoice.declared_tax_rate == Decimal("0.1")


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
