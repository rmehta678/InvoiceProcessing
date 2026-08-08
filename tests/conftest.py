"""Shared explicit on-disk SQLite fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind, migrate_database, seed_inventory


@pytest.fixture
def invoice_dir() -> Path:
    return Path("data/invoices").resolve()


@pytest.fixture
def inventory_db(tmp_path: Path) -> Path:
    path = tmp_path / "inventory.db"
    migrate_database(path, DatabaseKind.INVENTORY)
    seed_inventory(path)
    return path


@pytest.fixture
def workflow_db(tmp_path: Path) -> Path:
    path = tmp_path / "workflow.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    return path


@pytest.fixture
def settings(inventory_db: Path, workflow_db: Path) -> Settings:
    return Settings(
        xai_api_key="test-only-not-a-real-key",
        inventory_db=inventory_db,
        workflow_db=workflow_db,
        source_archive_dir=workflow_db.parent / "sources",
    )
