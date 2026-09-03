from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from models import Invoice, LineItem


class ParseError(Exception):
    pass


def parse_json(path: Path) -> Invoice:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    vendor = data.get("vendor", "")
    if isinstance(vendor, dict):
        vendor = vendor.get("name", "")
    vendor = (vendor or "").strip()

    if "total" in data and data["total"] is not None:
        raw_amount = data["total"]
    elif "amount" in data and data["amount"] is not None:
        raw_amount = data["amount"]
    else:
        raw_amount = 0

    items: list[LineItem] = []
    for item in data.get("line_items") or data.get("items") or []:
        name = item.get("item") or item.get("name") or ""
        qty = item.get("quantity")
        if name and qty is not None:
            items.append(LineItem(name=str(name), quantity=int(qty)))

    due = data.get("due_date")
    return Invoice(
        vendor=vendor,
        amount=_decimal(raw_amount),
        items=items,
        due_date=parse_date(due) if isinstance(due, str) else None,
    )


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _decimal(value: object) -> Decimal:
    try:
        return Decimal(str(value).replace(",", "").replace("$", "").strip() or "0")
    except (InvalidOperation, ValueError) as e:
        raise ParseError(f"Invalid amount: {value}") from e
