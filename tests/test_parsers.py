from __future__ import annotations

from decimal import Decimal

from parsers import parse_date, parse_json


def test_parse_json_nested_vendor(invoices):
    invoice = parse_json(invoices / "invoice_1004.json")
    assert invoice.vendor == "Precision Parts Ltd."
    assert invoice.amount == Decimal("1890.00")
    assert [(item.name, item.quantity) for item in invoice.items] == [("WidgetA", 3), ("WidgetB", 2)]
    assert str(invoice.due_date) == "2026-02-22"


def test_parse_json_invalid_amounts(invoices):
    invoice = parse_json(invoices / "invoice_1009.json")
    assert invoice.vendor == ""
    assert invoice.amount == Decimal("-250.0")
    assert invoice.items[0].quantity == -5


def test_parse_date_formats():
    assert parse_date("2026-02-01").isoformat() == "2026-02-01"
    assert parse_date("01/28/2026").isoformat() == "2026-01-28"
    assert parse_date("not a date") is None
    assert parse_date(None) is None
