"""Create and seed the mock inventory database.

Safe to re-run: uses INSERT OR IGNORE, unlike the plain INSERT in the case
README which fails on the second run against the PRIMARY KEY.

Usage: python scripts/init_db.py [--reset]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_flow.config import DB_PATH  # noqa: E402

# item, stock, unit_price, category
# Stock levels are the README's required seed. Unit prices mirror the rates on
# the sample invoices so price-variance checks have a reference to compare to.
SEED_INVENTORY = [
    ("WidgetA", 15, 250.00, "widgets"),
    ("WidgetB", 10, 500.00, "widgets"),
    ("GadgetX", 5, 750.00, "gadgets"),
    ("FakeItem", 0, 0.00, "unclassified"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS inventory (
    item        TEXT PRIMARY KEY,
    stock       INTEGER NOT NULL DEFAULT 0,
    unit_price  REAL,
    category    TEXT
);

-- Every processed invoice lands here: the audit trail, and the basis for
-- duplicate-payment detection.
CREATE TABLE IF NOT EXISTS invoice_ledger (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    invoice_number    TEXT,
    vendor            TEXT,
    amount            REAL,
    currency          TEXT,
    decision          TEXT NOT NULL,
    payment_status    TEXT,
    payment_reference TEXT,
    content_hash      TEXT,
    source_path       TEXT,
    processed_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_invoice_number
    ON invoice_ledger (invoice_number);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the inventory database with row access by column name."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialise(db_path: Path = DB_PATH, reset: bool = False) -> None:
    if reset and db_path.exists():
        db_path.unlink()
        print(f"Removed existing database at {db_path}")

    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT OR IGNORE INTO inventory (item, stock, unit_price, category)"
            " VALUES (?, ?, ?, ?)",
            SEED_INVENTORY,
        )
        conn.commit()

        rows = conn.execute("SELECT item, stock, unit_price FROM inventory ORDER BY item").fetchall()
        print(f"Inventory database ready at {db_path}")
        print(f"{'ITEM':<12} {'STOCK':>6} {'UNIT PRICE':>12}")
        for row in rows:
            price = f"${row['unit_price']:,.2f}" if row["unit_price"] is not None else "-"
            print(f"{row['item']:<12} {row['stock']:>6} {price:>12}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise the mock inventory database.")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the database.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path.")
    args = parser.parse_args()
    initialise(args.db, reset=args.reset)


if __name__ == "__main__":
    main()
