"""Invoice source readers and format-specific evidence extraction.

Readers preserve source locations and raw values. Normalization is recorded, and
unreadable or unsupported content raises an explicit source/parse error instead of
returning an empty invoice.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import fitz
from dateutil import parser as date_parser
from pypdf import PdfReader

from invoice_agents.errors import ErrorCategory, SourceEvidenceError
from invoice_agents.models import (
    EvidenceRef,
    EvidenceValue,
    ExtractedInvoice,
    InvoiceLine,
    SourceArtifact,
)
from invoice_agents.source_store import verified_source_path

RELATIVE_DATE = re.compile(r"\b(today|tomorrow|yesterday|next\s+\w+|last\s+\w+)\b", re.I)
MONEY_TOKEN = re.compile(r"[-+]?\$?\s*[0-9O][0-9O,]*(?:\.[0-9O]{1,2})?")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text_invoice(source: SourceArtifact) -> dict[str, Any]:
    """Read UTF-8 text with stable one-based line references."""

    path = verified_source_path(source)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise SourceEvidenceError(
            ErrorCategory.SOURCE,
            f"text source could not be read as UTF-8: {exc}",
            stop_reason="SOURCE_READ_FAILED",
        ) from exc
    if not raw.strip():
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            "text source is empty",
            stop_reason="SOURCE_EMPTY",
        )
    return {
        "raw_text": raw,
        "lines": [{"line": index, "text": line} for index, line in enumerate(raw.splitlines(), 1)],
    }


def read_json_invoice(source: SourceArtifact) -> dict[str, Any]:
    """Parse one JSON object; malformed or non-object roots are explicit failures."""

    path = verified_source_path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            f"JSON parse failed: {exc}",
            stop_reason="JSON_PARSE_FAILED",
        ) from exc
    if not isinstance(value, dict):
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            "JSON invoice root must be an object",
            stop_reason="JSON_ROOT_INVALID",
        )
    return value


def read_csv_invoice(source: SourceArtifact) -> dict[str, Any]:
    """Read CSV rows without assuming vertical or row-oriented layout."""

    path = verified_source_path(source)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            f"CSV parse failed: {exc}",
            stop_reason="CSV_PARSE_FAILED",
        ) from exc
    if len(rows) < 2:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            "CSV invoice must contain a header and data",
            stop_reason="CSV_EMPTY",
        )
    return {"rows": [{"row": index, "values": row} for index, row in enumerate(rows, 1)]}


def read_xml_invoice(source: SourceArtifact) -> dict[str, Any]:
    """Parse XML into a transparent path/value representation."""

    path = verified_source_path(source)
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            f"XML parse failed: {exc}",
            stop_reason="XML_PARSE_FAILED",
        ) from exc

    def walk(node: ElementTree.Element, path: str) -> list[dict[str, str | None]]:
        current = f"{path}/{node.tag}"
        output = [
            {
                "path": current,
                "value": node.text.strip() if node.text and node.text.strip() else None,
            }
        ]
        for child in node:
            output.extend(walk(child, current))
        return output

    return {"root": root, "nodes": walk(root, "")}


def extract_pdf_text(source: SourceArtifact) -> dict[str, Any]:
    """Extract text by page while retaining that extraction is not visual proof."""

    path = verified_source_path(source)
    try:
        reader = PdfReader(path)
        pages = []
        for index, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append({"page": index, "text": text})
    except Exception as exc:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            f"PDF text extraction failed: {exc}",
            stop_reason="PDF_EXTRACTION_FAILED",
        ) from exc
    if not any(str(page["text"]).strip() for page in pages):
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            "PDF contains no extractable text; visual/OCR review is required",
            stop_reason="PDF_TEXT_EMPTY",
        )
    return {"pages": pages, "extractor": "pypdf"}


def render_pdf_page(source: SourceArtifact, page: int, output_dir: Path) -> dict[str, Any]:
    """Render one page to PNG for layout evidence; page numbers are one-based."""

    if source.source_format != "pdf":
        raise SourceEvidenceError(
            ErrorCategory.SOURCE,
            "render_pdf_page only accepts PDF sources",
            stop_reason="RENDER_FORMAT_INVALID",
        )
    if page < 1 or (source.page_count is not None and page > source.page_count):
        raise SourceEvidenceError(
            ErrorCategory.SOURCE,
            f"PDF page {page} is out of range",
            stop_reason="RENDER_PAGE_INVALID",
        )
    source_path = verified_source_path(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = (output_dir / f"{source.source_id}-page-{page}.png").resolve()
    try:
        document = fitz.open(source_path)
        try:
            pixmap = document[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pixmap.save(target)
        finally:
            document.close()
    except Exception as exc:
        raise SourceEvidenceError(
            ErrorCategory.TOOL,
            f"PDF page render failed: {exc}",
            stop_reason="PDF_RENDER_FAILED",
        ) from exc
    return {"path": str(target), "page": page, "sha256": _sha256(target), "renderer": "PyMuPDF"}


def _ref(
    source: SourceArtifact,
    locator_type: Literal["line", "row", "json_path", "xpath", "page", "file"],
    locator: str,
    raw: Any,
) -> EvidenceRef:
    return EvidenceRef(
        source_id=source.source_id,
        locator_type=locator_type,
        locator=locator,
        raw_value=None if raw is None else str(raw),
    )


def _value(
    source: SourceArtifact,
    raw: Any,
    normalized: Any,
    locator_type: Literal["line", "row", "json_path", "xpath", "page", "file"],
    locator: str,
    *,
    normalization: str = "none",
    confidence: float = 1.0,
    ambiguity: str | None = None,
) -> EvidenceValue:
    return EvidenceValue(
        raw_value=None if raw is None else str(raw),
        normalized_value=None if normalized is None else str(normalized),
        normalization=normalization,
        evidence=[] if raw is None else [_ref(source, locator_type, locator, raw)],
        confidence=confidence,
        ambiguity=ambiguity,
    )


def _decimal(raw: Any, *, allow_ocr: bool = False) -> tuple[Decimal, str | None]:
    if isinstance(raw, bool) or raw is None:
        raise ValueError("numeric value is missing")
    value = str(raw).strip().replace("$", "").replace("€", "").replace(",", "").replace(" ", "")
    note: str | None = None
    if "O" in value.upper():
        if not allow_ocr:
            raise ValueError(f"invalid numeric value {raw!r}")
        value = value.upper().replace("O", "0")
        note = f"OCR-like O normalized to 0 in {raw!r}"
    try:
        return Decimal(value), note
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal value {raw!r}") from exc


def _invoice_number(raw: Any) -> tuple[str | None, str, str | None]:
    if raw is None or not str(raw).strip():
        return None, "none", None
    text = str(raw).strip().upper()
    compact = re.sub(r"\s+", "-", text)
    compact = re.sub(r"^INV[-#:\s]*", "INV-", compact)
    if re.fullmatch(r"\d{4,}", compact):
        compact = f"INV-{compact}"
    ambiguity = None
    if compact != text:
        ambiguity = "invoice number punctuation/prefix normalized"
    return compact, "uppercase; canonical INV- prefix and separator", ambiguity


def _date_value(
    source: SourceArtifact,
    raw: Any,
    locator_type: Literal["line", "row", "json_path", "xpath", "page", "file"],
    locator: str,
) -> EvidenceValue:
    if raw is None or not str(raw).strip():
        return _value(
            source, None, None, locator_type, locator, ambiguity="date is missing", confidence=0
        )
    text = str(raw).strip()
    if RELATIVE_DATE.search(text):
        return _value(
            source,
            text,
            None,
            locator_type,
            locator,
            confidence=0,
            ambiguity="relative date cannot be normalized without an explicit reference date",
        )
    normalized_input = text
    ocr_note = None
    if re.search(r"(?<=\d)O|O(?=\d)", normalized_input, re.I):
        normalized_input = normalized_input.upper().replace("O", "0")
        ocr_note = "OCR-like O normalized to 0; human review required"
    try:
        parsed = date_parser.parse(normalized_input, fuzzy=False).date().isoformat()
    except (ValueError, OverflowError) as exc:
        return _value(
            source,
            text,
            None,
            locator_type,
            locator,
            confidence=0,
            ambiguity=f"date parse failed: {exc}",
        )
    return _value(
        source,
        text,
        parsed,
        locator_type,
        locator,
        normalization="dateutil parse to ISO-8601" + ("; O->0" if ocr_note else ""),
        confidence=0.7 if ocr_note else 1.0,
        ambiguity=ocr_note,
    )


def _line(
    source: SourceArtifact,
    line_id: str,
    raw_item: Any,
    raw_quantity: Any,
    raw_price: Any,
    locator_type: Literal["line", "row", "json_path", "xpath", "page", "file"],
    locator: str,
    raw_total: Any = None,
) -> InvoiceLine:
    quantity, quantity_note = _decimal(raw_quantity, allow_ocr=True)
    price, price_note = _decimal(raw_price, allow_ocr=True)
    declared: Decimal | None = None
    total_note: str | None = None
    if raw_total not in (None, ""):
        declared, total_note = _decimal(raw_total, allow_ocr=True)
    ambiguity = [note for note in (quantity_note, price_note, total_note) if note]
    item = str(raw_item).strip().lstrip("- ")
    return InvoiceLine(
        line_id=line_id,
        raw_item=item,
        normalized_item=re.sub(r"\s+", " ", item).strip(),
        raw_quantity=str(raw_quantity),
        quantity=quantity,
        raw_unit_price=str(raw_price),
        unit_price=price,
        raw_declared_line_total=None if raw_total is None else str(raw_total),
        declared_line_total=declared,
        calculated_line_total=quantity * price,
        evidence=[
            _ref(source, locator_type, locator, f"{item}|{raw_quantity}|{raw_price}|{raw_total}")
        ],
        ambiguity=ambiguity,
    )


def _currency_value(
    source: SourceArtifact,
    raw: Any,
    locator_type: Literal["line", "row", "json_path", "xpath", "page", "file"],
    locator: str,
    *,
    text: str = "",
) -> EvidenceValue:
    if raw and str(raw).strip():
        normalized = str(raw).strip().upper()
        return _value(
            source, raw, normalized, locator_type, locator, normalization="uppercase ISO code"
        )
    if "$" in text:
        return _value(
            source,
            "$",
            "USD",
            locator_type,
            locator,
            normalization="currency inferred from dollar symbol",
        )
    return _value(
        source,
        None,
        "USD",
        locator_type,
        locator,
        normalization="prototype operating-currency convention",
        confidence=0.6,
        ambiguity="source omits an explicit currency; USD convention requires review if material",
    )


def _finish_invoice(invoice: ExtractedInvoice) -> ExtractedInvoice:
    required = {
        "invoice_number": invoice.invoice_number.normalized_value,
        "vendor": invoice.vendor.normalized_value,
        "invoice_date": invoice.invoice_date.normalized_value,
        "due_date": invoice.due_date.normalized_value,
        "payment_terms": invoice.payment_terms.normalized_value,
        "currency": invoice.currency.normalized_value,
        "total": invoice.declared_total,
    }
    invoice.missing_fields = [key for key, value in required.items() if value in (None, "")]
    if not invoice.lines:
        invoice.missing_fields.append("lines")
    if invoice.invoice_number.ambiguity:
        invoice.extraction_notes.append(invoice.invoice_number.ambiguity)
    for field in (
        invoice.vendor,
        invoice.invoice_date,
        invoice.due_date,
        invoice.payment_terms,
        invoice.currency,
    ):
        if field.ambiguity:
            invoice.extraction_notes.append(field.ambiguity)
    for line in invoice.lines:
        invoice.extraction_notes.extend(line.ambiguity)
    return invoice


def _extract_json(source: SourceArtifact) -> ExtractedInvoice:
    obj = read_json_invoice(source)
    raw_number = obj.get("invoice_number")
    number, normalization, number_ambiguity = _invoice_number(raw_number)
    vendor_obj = obj.get("vendor")
    raw_vendor = vendor_obj.get("name") if isinstance(vendor_obj, dict) else vendor_obj
    raw_currency = obj.get("currency")
    currency = str(raw_currency or "USD").upper()
    raw_lines = obj.get("line_items")
    if not isinstance(raw_lines, list):
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            "JSON line_items must be an array",
            stop_reason="JSON_LINES_INVALID",
        )
    lines: list[InvoiceLine] = []
    for index, item in enumerate(raw_lines):
        if not isinstance(item, dict):
            raise SourceEvidenceError(
                ErrorCategory.PARSE,
                f"JSON line_items[{index}] must be an object",
                stop_reason="JSON_LINE_INVALID",
            )
        try:
            lines.append(
                _line(
                    source,
                    f"{source.source_id}:json:{index}",
                    item.get("item"),
                    item.get("quantity"),
                    item.get("unit_price"),
                    "json_path",
                    f"$.line_items[{index}]",
                    item.get("amount"),
                )
            )
        except ValueError as exc:
            raise SourceEvidenceError(
                ErrorCategory.PARSE,
                f"invalid JSON line_items[{index}]: {exc}",
                stop_reason="JSON_LINE_VALUE_INVALID",
            ) from exc
    try:
        subtotal = _decimal(obj["subtotal"])[0] if obj.get("subtotal") is not None else None
        tax_rate = _decimal(obj["tax_rate"])[0] if obj.get("tax_rate") is not None else None
        tax_amount = _decimal(obj["tax_amount"])[0] if obj.get("tax_amount") is not None else None
        total = _decimal(obj["total"])[0] if obj.get("total") is not None else None
    except ValueError as exc:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            f"invalid JSON total field: {exc}",
            stop_reason="JSON_TOTAL_INVALID",
        ) from exc
    return _finish_invoice(
        ExtractedInvoice(
            source=source,
            invoice_number=_value(
                source,
                raw_number,
                number,
                "json_path",
                "$.invoice_number",
                normalization=normalization,
                ambiguity=number_ambiguity,
            ),
            revision=(
                _value(source, obj.get("revision"), obj.get("revision"), "json_path", "$.revision")
                if obj.get("revision") is not None
                else None
            ),
            vendor=_value(source, raw_vendor, raw_vendor, "json_path", "$.vendor.name"),
            invoice_date=_date_value(source, obj.get("date"), "json_path", "$.date"),
            due_date=_date_value(source, obj.get("due_date"), "json_path", "$.due_date"),
            payment_terms=_value(
                source,
                obj.get("payment_terms"),
                obj.get("payment_terms"),
                "json_path",
                "$.payment_terms",
            ),
            currency=_currency_value(source, raw_currency, "json_path", "$.currency"),
            lines=lines,
            declared_subtotal=subtotal,
            declared_tax_rate=tax_rate,
            declared_tax_amount=tax_amount,
            declared_total=total,
            extraction_notes=[f"currency for numeric amounts: {currency}"],
        )
    )


def _find_labeled(lines: list[str], patterns: list[str]) -> tuple[str | None, int | None]:
    # Pattern precedence matters: an embedded Vendor field must beat an email From header.
    for pattern in patterns:
        for number, line in enumerate(lines, 1):
            match = re.search(pattern, line, re.I)
            if match:
                return match.group("value").strip(), number
    return None, None


def _extract_textual(
    source: SourceArtifact, raw_text: str, locator_prefix: str = "line"
) -> ExtractedInvoice:
    lines = raw_text.splitlines()
    raw_number, number_line = _find_labeled(
        lines,
        [
            r"(?:invoice\s+number|inv\s*#|invoice\s*#|inv\s+no|invoice)\s*[:#]?\s*(?P<value>INV[\s-]*\d+|\d{4,})",
        ],
    )
    number, number_norm, number_ambiguity = _invoice_number(raw_number)
    raw_vendor, vendor_line = _find_labeled(
        lines,
        [
            r"\b(?:vendor|vndr)\s*:\s*(?P<value>.+?)(?:\s{2,}(?:Due|Date)\s*:|$)",
            r"^\s*FROM\s*:\s*(?P<value>.+)$",
        ],
    )
    raw_date, date_line = _find_labeled(
        lines, [r"^\s*(?:date|dt)\s*:\s*(?P<value>.+)$", r"\bDate\s*:\s*(?P<value>[^|]+)"]
    )
    raw_due, due_line = _find_labeled(
        lines, [r"\b(?:due\s+date|due\s+dt|due)\s*:\s*(?P<value>.+)$"]
    )
    raw_terms, terms_line = _find_labeled(
        lines, [r"\b(?:payment\s+terms|pymnt\s+terms|terms)\s*:\s*(?P<value>.+)$"]
    )

    extracted_lines: list[InvoiceLine] = []
    table_pattern = re.compile(
        r"^\s*-?\s*(?P<item>[A-Za-z][A-Za-z0-9 ]*?(?:\s*\([^)]*\))?)\s+"
        r"(?:(?:qty\s*:?\s*|x)(?P<q1>-?\d+(?:\.\d+)?)|(?P<q2>-?\d+(?:\.\d+)?))\s+"
        r"(?:(?:unit\s+price\s*:?\s*|@\s*)?)\$?(?P<price>[0-9O][0-9O,]*(?:\.[0-9O]{1,2})?)"
        r"(?:\s*(?:ea(?:ch)?\b)?)"
        r"(?:\s+\$?(?P<total>[0-9O][0-9O,]*(?:\.[0-9O]{1,2})?))?",
        re.I,
    )
    excluded = {"subtotal", "total", "tax", "sales tax", "shipping", "amount"}
    for index, text_line in enumerate(lines, 1):
        match = table_pattern.match(text_line)
        if not match:
            continue
        item = match.group("item").strip()
        if item.lower() in excluded or item.lower().startswith(("invoice", "date", "due")):
            continue
        quantity = match.group("q1") or match.group("q2")
        try:
            extracted_lines.append(
                _line(
                    source,
                    f"{source.source_id}:{locator_prefix}:{index}",
                    item,
                    quantity,
                    match.group("price"),
                    "page" if locator_prefix.startswith("page") else "line",
                    f"{locator_prefix}:{index}",
                    match.group("total"),
                )
            )
        except ValueError as exc:
            raise SourceEvidenceError(
                ErrorCategory.PARSE,
                f"invalid line at {locator_prefix}:{index}: {exc}",
                stop_reason="TEXT_LINE_INVALID",
            ) from exc

    def total_value(pattern: str) -> tuple[Decimal | None, str | None]:
        raw, _ = _find_labeled(lines, [pattern])
        if raw is None:
            return None, None
        money = MONEY_TOKEN.search(raw)
        if money is None:
            return None, None
        try:
            return _decimal(money.group(), allow_ocr=True)
        except ValueError:
            return None, None

    subtotal, subtotal_note = total_value(r"^\s*Subtotal\s*:\s*(?P<value>.+)$")
    tax, tax_note = total_value(r"^\s*(?:Tax(?:\s*\([^)]*\))?|Sales\s+Tax)\s*:\s*(?P<value>.+)$")
    shipping, shipping_note = total_value(r"^\s*Shipping\s*:\s*(?P<value>.+)$")
    total, total_note = total_value(
        r"^\s*(?:Total\s+Amount|Grand\s+Total|TOTAL|Total|Amt)\s*:\s*(?P<value>.+)$"
    )
    tax_rate_raw, _ = _find_labeled(lines, [r"Tax\s*\((?P<value>\d+(?:\.\d+)?)%\)"])
    tax_rate = Decimal(tax_rate_raw) / 100 if tax_rate_raw else None
    notes = [note for note in (subtotal_note, tax_note, shipping_note, total_note) if note]
    full_locator: Literal["page", "line"] = "page" if locator_prefix.startswith("page") else "line"
    invoice = ExtractedInvoice(
        source=source,
        invoice_number=_value(
            source,
            raw_number,
            number,
            full_locator,
            f"{locator_prefix}:{number_line or '?'}",
            normalization=number_norm,
            ambiguity=number_ambiguity,
        ),
        vendor=_value(
            source, raw_vendor, raw_vendor, full_locator, f"{locator_prefix}:{vendor_line or '?'}"
        ),
        invoice_date=_date_value(
            source, raw_date, full_locator, f"{locator_prefix}:{date_line or '?'}"
        ),
        due_date=_date_value(source, raw_due, full_locator, f"{locator_prefix}:{due_line or '?'}"),
        payment_terms=_value(
            source, raw_terms, raw_terms, full_locator, f"{locator_prefix}:{terms_line or '?'}"
        ),
        currency=_currency_value(source, None, full_locator, locator_prefix, text=raw_text),
        lines=extracted_lines,
        declared_subtotal=subtotal,
        declared_tax_rate=tax_rate,
        declared_tax_amount=tax,
        declared_fees=shipping or Decimal("0"),
        declared_total=total,
        extraction_notes=notes,
    )
    if re.search(r"urgent|wire\s+transfer|avoid\s+penalt", raw_text, re.I):
        invoice.extraction_notes.append("source contains urgent or wire-payment wording")
    return _finish_invoice(invoice)


def _extract_csv(source: SourceArtifact) -> ExtractedInvoice:
    evidence = read_csv_invoice(source)
    rows = [entry["values"] for entry in evidence["rows"]]
    header = [cell.strip() for cell in rows[0]]
    if [cell.lower() for cell in header[:2]] == ["field", "value"]:
        values: dict[str, list[tuple[str, int]]] = {}
        for row_index, row in enumerate(rows[1:], 2):
            if len(row) < 2:
                continue
            values.setdefault(row[0].strip().lower(), []).append((row[1].strip(), row_index))
        raw_items = values.get("item", [])
        quantities = values.get("quantity", [])
        prices = values.get("unit_price", [])
        if not (len(raw_items) == len(quantities) == len(prices)):
            raise SourceEvidenceError(
                ErrorCategory.PARSE,
                "vertical CSV item/quantity/unit_price counts differ",
                stop_reason="CSV_VERTICAL_GROUP_INVALID",
            )
        lines = [
            _line(
                source,
                f"{source.source_id}:row:{raw_items[index][1]}",
                item[0],
                quantities[index][0],
                prices[index][0],
                "row",
                f"row:{item[1]}-{prices[index][1]}",
            )
            for index, item in enumerate(raw_items)
        ]

        def first(key: str) -> tuple[str | None, int | None]:
            return values.get(key, [(None, None)])[0]

        raw_number, number_row = first("invoice_number")
        number, number_norm, number_ambiguity = _invoice_number(raw_number)
        raw_vendor, vendor_row = first("vendor")
        raw_date, date_row = first("date")
        raw_due, due_row = first("due_date")
        raw_terms, terms_row = first("payment_terms")

        def dec(key: str) -> Decimal | None:
            raw, _ = first(key)
            return _decimal(raw)[0] if raw is not None else None

        return _finish_invoice(
            ExtractedInvoice(
                source=source,
                invoice_number=_value(
                    source,
                    raw_number,
                    number,
                    "row",
                    f"row:{number_row}",
                    normalization=number_norm,
                    ambiguity=number_ambiguity,
                ),
                vendor=_value(source, raw_vendor, raw_vendor, "row", f"row:{vendor_row}"),
                invoice_date=_date_value(source, raw_date, "row", f"row:{date_row}"),
                due_date=_date_value(source, raw_due, "row", f"row:{due_row}"),
                payment_terms=_value(source, raw_terms, raw_terms, "row", f"row:{terms_row}"),
                currency=_currency_value(source, None, "file", "CSV", text=""),
                lines=lines,
                declared_subtotal=dec("subtotal"),
                declared_tax_amount=dec("tax"),
                declared_total=dec("total"),
            )
        )

    normalized_header = {name.strip().lower(): index for index, name in enumerate(header)}
    required = {
        "invoice number",
        "vendor",
        "date",
        "due date",
        "item",
        "qty",
        "unit price",
        "line total",
    }
    if not required.issubset(normalized_header):
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            f"row-oriented CSV is missing columns: {sorted(required - normalized_header.keys())}",
            stop_reason="CSV_COLUMNS_MISSING",
        )
    data_rows: list[tuple[list[str], int]] = []
    totals: dict[str, Decimal] = {}
    tax_rate: Decimal | None = None
    for row_index, row in enumerate(rows[1:], 2):
        padded = row + [""] * (len(header) - len(row))
        if padded[normalized_header["invoice number"]].strip():
            data_rows.append((padded, row_index))
            continue
        label = padded[normalized_header["unit price"]].strip().rstrip(":").lower()
        raw_value = padded[normalized_header["line total"]].strip()
        if label and raw_value:
            amount = _decimal(raw_value, allow_ocr=True)[0]
            if label.startswith("subtotal"):
                totals["subtotal"] = amount
            elif label.startswith("tax"):
                totals["tax"] = amount
                rate_match = re.search(r"(\d+(?:\.\d+)?)%", label)
                if rate_match:
                    tax_rate = Decimal(rate_match.group(1)) / 100
            elif label.startswith("total"):
                totals["total"] = amount
    if not data_rows:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            "row-oriented CSV contains no invoice line rows",
            stop_reason="CSV_DATA_MISSING",
        )
    lines = [
        _line(
            source,
            f"{source.source_id}:row:{row_index}",
            row[normalized_header["item"]],
            row[normalized_header["qty"]],
            row[normalized_header["unit price"]],
            "row",
            f"row:{row_index}",
            row[normalized_header["line total"]],
        )
        for row, row_index in data_rows
    ]
    first_row, first_index = data_rows[0]
    raw_number = first_row[normalized_header["invoice number"]]
    number, number_norm, number_ambiguity = _invoice_number(raw_number)
    return _finish_invoice(
        ExtractedInvoice(
            source=source,
            invoice_number=_value(
                source,
                raw_number,
                number,
                "row",
                f"row:{first_index}",
                normalization=number_norm,
                ambiguity=number_ambiguity,
            ),
            vendor=_value(
                source,
                first_row[normalized_header["vendor"]],
                first_row[normalized_header["vendor"]],
                "row",
                f"row:{first_index}",
            ),
            invoice_date=_date_value(
                source, first_row[normalized_header["date"]], "row", f"row:{first_index}"
            ),
            due_date=_date_value(
                source, first_row[normalized_header["due date"]], "row", f"row:{first_index}"
            ),
            payment_terms=_value(
                source,
                None,
                None,
                "file",
                "CSV",
                confidence=0,
                ambiguity="row-oriented CSV does not include payment terms",
            ),
            currency=_currency_value(source, None, "file", "CSV", text=""),
            lines=lines,
            declared_subtotal=totals.get("subtotal"),
            declared_tax_rate=tax_rate,
            declared_tax_amount=totals.get("tax"),
            declared_total=totals.get("total"),
        )
    )


def _extract_xml(source: SourceArtifact) -> ExtractedInvoice:
    parsed = read_xml_invoice(source)
    root: ElementTree.Element = parsed["root"]

    def text(path: str) -> str | None:
        node = root.find(path)
        return node.text.strip() if node is not None and node.text and node.text.strip() else None

    raw_number = text("./header/invoice_number")
    number, number_norm, number_ambiguity = _invoice_number(raw_number)
    lines: list[InvoiceLine] = []
    for index, node in enumerate(root.findall("./line_items/item"), 1):
        raw_item = node.findtext("name")
        raw_quantity = node.findtext("quantity")
        raw_price = node.findtext("unit_price")
        try:
            lines.append(
                _line(
                    source,
                    f"{source.source_id}:xpath:{index}",
                    raw_item,
                    raw_quantity,
                    raw_price,
                    "xpath",
                    f"/invoice/line_items/item[{index}]",
                )
            )
        except ValueError as exc:
            raise SourceEvidenceError(
                ErrorCategory.PARSE,
                f"invalid XML item[{index}]: {exc}",
                stop_reason="XML_LINE_INVALID",
            ) from exc

    def dec(path: str) -> Decimal | None:
        raw = text(path)
        return _decimal(raw)[0] if raw is not None else None

    return _finish_invoice(
        ExtractedInvoice(
            source=source,
            invoice_number=_value(
                source,
                raw_number,
                number,
                "xpath",
                "/invoice/header/invoice_number",
                normalization=number_norm,
                ambiguity=number_ambiguity,
            ),
            vendor=_value(
                source,
                text("./header/vendor"),
                text("./header/vendor"),
                "xpath",
                "/invoice/header/vendor",
            ),
            invoice_date=_date_value(
                source, text("./header/date"), "xpath", "/invoice/header/date"
            ),
            due_date=_date_value(
                source, text("./header/due_date"), "xpath", "/invoice/header/due_date"
            ),
            payment_terms=_value(
                source,
                text("./payment_terms"),
                text("./payment_terms"),
                "xpath",
                "/invoice/payment_terms",
            ),
            currency=_currency_value(
                source, text("./header/currency"), "xpath", "/invoice/header/currency"
            ),
            lines=lines,
            declared_subtotal=dec("./totals/subtotal"),
            declared_tax_rate=dec("./totals/tax_rate"),
            declared_tax_amount=dec("./totals/tax_amount"),
            declared_total=dec("./totals/total"),
        )
    )


def extract_invoice_evidence(source: SourceArtifact) -> ExtractedInvoice:
    """Dispatch to the declared format parser; no cross-format or canned fallback exists."""

    if source.source_format == "json":
        return _extract_json(source)
    if source.source_format == "csv":
        return _extract_csv(source)
    if source.source_format == "xml":
        return _extract_xml(source)
    if source.source_format == "txt":
        return _extract_textual(source, read_text_invoice(source)["raw_text"])
    if source.source_format == "pdf":
        pages = extract_pdf_text(source)["pages"]
        combined = "\n".join(f"[PAGE {page['page']}]\n{page['text']}" for page in pages)
        invoice = _extract_textual(source, combined, "page")
        invoice.extraction_notes.append(
            "PDF fields are based on pypdf text extraction; render pages for layout-sensitive review"
        )
        return invoice
    raise SourceEvidenceError(
        ErrorCategory.SOURCE,
        f"unsupported declared source format: {source.source_format}",
        stop_reason="SOURCE_FORMAT_UNSUPPORTED",
    )
