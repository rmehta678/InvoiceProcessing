"""Inventory database access, exposed both as Python helpers and LLM tools.

Matching policy, in order:

1. Exact match on the catalogue name.
2. Normalised match -- casefold and strip non-alphanumerics, which resolves the
   OCR-spaced ``Widget A`` to ``WidgetA``.
3. Close-but-not-equal names are surfaced as a *suggestion only*.

Step 3 never substitutes automatically, and that restraint is deliberate.
``WidgetC`` on invoice 1016 is ~0.86 similar to ``WidgetA``; auto-correcting it
would authorise payment for a product the company does not stock. A near-miss
is a question for a human, not a decision for a matcher.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config import DB_PATH, FUZZY_SUGGEST_THRESHOLD
from ..models import Finding, FindingCode, InvoiceDraft, ItemCheck, Severity, normalize


@dataclass
class CatalogueEntry:
    item: str
    stock: int
    unit_price: float | None
    category: str | None


class InventoryRepository:
    """Read access to the inventory catalogue plus ledger writes."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Inventory database not found at {self.db_path}. "
                "Run `python scripts/init_db.py` first."
            )
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._catalogue: dict[str, CatalogueEntry] | None = None

    # -- catalogue ---------------------------------------------------------

    def catalogue(self) -> dict[str, CatalogueEntry]:
        """The full catalogue keyed by normalised name, cached per instance."""
        if self._catalogue is None:
            rows = self._conn.execute(
                "SELECT item, stock, unit_price, category FROM inventory"
            ).fetchall()
            self._catalogue = {
                normalize(row["item"]): CatalogueEntry(
                    item=row["item"],
                    stock=row["stock"],
                    unit_price=row["unit_price"],
                    category=row["category"],
                )
                for row in rows
            }
        return self._catalogue

    def lookup_item(self, name: str) -> dict[str, Any]:
        """Find an item by name. Tool-callable by the validation agent.

        Returns a dict with ``found``, and either the catalogue record or a
        ``suggestion`` naming the closest catalogue entry.
        """
        catalogue = self.catalogue()
        key = normalize(name)
        entry = catalogue.get(key)

        if entry is not None:
            return {
                "found": True,
                "query": name,
                "item": entry.item,
                "stock": entry.stock,
                "unit_price": entry.unit_price,
                "category": entry.category,
                "exact_name_match": entry.item == name,
            }

        best_name, best_score = None, 0.0
        for cat_key, cat_entry in catalogue.items():
            score = SequenceMatcher(None, key, cat_key).ratio()
            if score > best_score:
                best_name, best_score = cat_entry.item, score

        result: dict[str, Any] = {
            "found": False,
            "query": name,
            "message": f"'{name}' is not in the inventory catalogue.",
        }
        if best_name is not None and best_score >= FUZZY_SUGGEST_THRESHOLD:
            result["suggestion"] = best_name
            result["similarity"] = round(best_score, 3)
            result["note"] = (
                "Similar catalogue name found. This is a suggestion for human "
                "review only -- do not treat it as a match."
            )
        elif best_name is not None:
            result["closest_catalogue_name"] = best_name
            result["similarity"] = round(best_score, 3)
        return result

    def check_stock(self, name: str, quantity: float) -> dict[str, Any]:
        """Check whether `quantity` of `name` can be fulfilled from stock."""
        record = self.lookup_item(name)
        if not record["found"]:
            return {**record, "sufficient": False, "reason": "item_not_found"}

        stock = record["stock"]
        sufficient = quantity <= stock
        return {
            **record,
            "quantity_requested": quantity,
            "sufficient": sufficient,
            "shortfall": max(0.0, quantity - stock),
            "reason": None if sufficient else ("out_of_stock" if stock == 0 else "insufficient"),
        }

    def list_catalog(self) -> list[dict[str, Any]]:
        """Every stocked item. Tool-callable, for grounding the agent."""
        return [
            {
                "item": entry.item,
                "stock": entry.stock,
                "unit_price": entry.unit_price,
                "category": entry.category,
            }
            for entry in sorted(self.catalogue().values(), key=lambda e: e.item)
        ]

    # -- ledger ------------------------------------------------------------

    def record_ledger_entry(
        self,
        run_id: str,
        invoice_number: str | None,
        vendor: str | None,
        amount: float | None,
        currency: str | None,
        decision: str,
        payment_status: str | None = None,
        payment_reference: str | None = None,
        content_hash: str | None = None,
        source_path: str | None = None,
    ) -> int:
        """Append to the audit ledger. Returns the new row id."""
        cursor = self._conn.execute(
            "INSERT INTO invoice_ledger (run_id, invoice_number, vendor, amount, currency,"
            " decision, payment_status, payment_reference, content_hash, source_path,"
            " processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                invoice_number,
                vendor,
                amount,
                currency,
                decision,
                payment_status,
                payment_reference,
                content_hash,
                source_path,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def prior_invoices(self, invoice_number: str | None) -> list[dict[str, Any]]:
        """Every prior ledger entry for this invoice number, whatever its outcome.

        Broader than `prior_payments`: a rejected or escalated invoice still
        establishes what was previously seen under that number, which is what
        makes a later conflicting version detectable.
        """
        if not invoice_number:
            return []
        rows = self._conn.execute(
            "SELECT run_id, invoice_number, vendor, amount, currency, decision,"
            " payment_status, payment_reference, content_hash, source_path, processed_at"
            " FROM invoice_ledger WHERE invoice_number = ? ORDER BY processed_at",
            (invoice_number,),
        ).fetchall()
        return [dict(row) for row in rows]

    def prior_payments(self, invoice_number: str | None) -> list[dict[str, Any]]:
        """Previous successful payments against this invoice number."""
        if not invoice_number:
            return []
        rows = self._conn.execute(
            "SELECT run_id, invoice_number, vendor, amount, currency, payment_reference,"
            " processed_at FROM invoice_ledger"
            " WHERE invoice_number = ? AND payment_status = 'success'"
            " ORDER BY processed_at",
            (invoice_number,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------
# Deterministic validation pass
# --------------------------------------------------------------------------


def check_inventory(draft: InvoiceDraft, repo: InventoryRepository) -> tuple[list[ItemCheck], list[Finding]]:
    """Check every billed item against stock, aggregating duplicate lines first.

    Aggregation is essential: invoice 1013 bills WidgetA on three separate
    lines totalling 22 units against stock of 15. Checking each line alone
    (15, 5, 2) passes all three and approves an unfulfillable order.
    """
    checks: list[ItemCheck] = []
    findings: list[Finding] = []

    for normalized, quantity in draft.aggregated_quantities().items():
        display = draft.display_name(normalized)
        line_count = sum(1 for i in draft.line_items if i.normalized_name == normalized)
        record = repo.check_stock(display, quantity)

        if not record["found"]:
            check = ItemCheck(
                invoice_name=display,
                normalized_name=normalized,
                quantity_requested=quantity,
                status="unknown",
            )
            hint = ""
            if record.get("suggestion"):
                hint = (
                    f" The closest catalogue entry is '{record['suggestion']}' "
                    f"(similarity {record['similarity']}), but the names are not "
                    "identical and must not be assumed equivalent."
                )
            findings.append(
                Finding(
                    code=FindingCode.ITEM_UNKNOWN,
                    severity=Severity.CRITICAL,
                    message=(
                        f"'{display}' does not exist in the inventory catalogue. "
                        f"Acme cannot have received {quantity:g} units of a product "
                        f"it does not stock.{hint}"
                    ),
                    detail={
                        "item": display,
                        "quantity": quantity,
                        "suggestion": record.get("suggestion"),
                    },
                    source="inventory",
                )
            )
        else:
            stock = record["stock"]
            status = "ok"
            if stock == 0:
                status = "out_of_stock"
                findings.append(
                    Finding(
                        code=FindingCode.ITEM_OUT_OF_STOCK,
                        severity=Severity.CRITICAL,
                        message=(
                            f"'{record['item']}' is carried at zero stock, yet the invoice "
                            f"bills for {quantity:g} units. Nothing could have been delivered."
                        ),
                        detail={"item": record["item"], "quantity": quantity, "stock": stock},
                        source="inventory",
                    )
                )
            elif quantity > stock:
                status = "shortfall"
                across = (
                    f" across {line_count} line items" if line_count > 1 else ""
                )
                findings.append(
                    Finding(
                        code=FindingCode.STOCK_SHORTFALL,
                        severity=Severity.CRITICAL,
                        message=(
                            f"'{record['item']}': invoice bills {quantity:g} units{across} "
                            f"but only {stock} are in stock "
                            f"(shortfall of {quantity - stock:g})."
                        ),
                        detail={
                            "item": record["item"],
                            "quantity_requested": quantity,
                            "stock_available": stock,
                            "shortfall": quantity - stock,
                            "line_items_aggregated": line_count,
                        },
                        source="inventory",
                    )
                )

            check = ItemCheck(
                invoice_name=display,
                normalized_name=normalized,
                quantity_requested=quantity,
                matched_item=record["item"],
                stock_available=stock,
                status=status,
            )

            if not record.get("exact_name_match", True) and status == "ok":
                findings.append(
                    Finding(
                        code=FindingCode.ITEM_NAME_SUGGESTION,
                        severity=Severity.INFO,
                        message=(
                            f"Invoice spells the item '{display}'; matched to catalogue "
                            f"entry '{record['item']}' after normalisation."
                        ),
                        detail={"invoice_name": display, "catalogue_name": record["item"]},
                        source="inventory",
                    )
                )

        checks.append(check)

    return checks, findings


# --------------------------------------------------------------------------
# LLM tool schemas
# --------------------------------------------------------------------------

INVENTORY_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_item",
            "description": (
                "Look up an item in the inventory catalogue by name. Returns stock "
                "level and unit price if found. If not found, may return a "
                "'suggestion' for a similarly-named catalogue entry -- a suggestion "
                "is NOT a match and must not be treated as one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Item name exactly as written on the invoice.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": (
                "Check whether a given quantity of an item can be fulfilled from "
                "current stock. Pass the TOTAL quantity across all line items for "
                "that product, not a single line's quantity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Item name."},
                    "quantity": {
                        "type": "number",
                        "description": "Total quantity billed for this item.",
                    },
                },
                "required": ["name", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_catalog",
            "description": (
                "List every item Acme stocks, with stock levels and unit prices. "
                "Use this to ground judgements about whether an item is legitimate."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
