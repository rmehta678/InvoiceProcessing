from __future__ import annotations

import sqlite3
from pathlib import Path

from config import DB_PATH

SEED_DATA = [
    ("WidgetA", 15),
    ("WidgetB", 10),
    ("GadgetX", 5),
    ("FakeItem", 0),
]


def init_db(db_path: Path | None = None) -> None:
    path = DB_PATH if db_path is None else db_path
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS inventory (item TEXT PRIMARY KEY, stock INTEGER NOT NULL)"
        )
        count = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
        if count == 0:
            conn.executemany("INSERT INTO inventory (item, stock) VALUES (?, ?)", SEED_DATA)
        conn.commit()


def get_stock(item: str, db_path: Path | None = None) -> int | None:
    path = DB_PATH if db_path is None else db_path
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT stock FROM inventory WHERE item = ? COLLATE NOCASE",
            (item,),
        ).fetchone()
    return row[0] if row else None


def list_inventory(db_path: Path | None = None) -> list[tuple[str, int]]:
    path = DB_PATH if db_path is None else db_path
    with sqlite3.connect(path) as conn:
        return list(conn.execute("SELECT item, stock FROM inventory ORDER BY item"))


def lookup_stock(name: str, db_path: Path | None = None) -> int | None:
    stock = get_stock(name, db_path)
    if stock is not None:
        return stock
    compact = "".join(name.split())
    if compact != name:
        return get_stock(compact, db_path)
    return None
