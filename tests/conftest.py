from __future__ import annotations

from pathlib import Path

import pytest

from db import init_db

@pytest.fixture
def invoices():
    return Path(__file__).resolve().parents[1] / "data" / "invoices"


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    db_path = tmp_path / "inventory.db"
    monkeypatch.setattr("db.DB_PATH", db_path)
    init_db(db_path)
    return db_path
