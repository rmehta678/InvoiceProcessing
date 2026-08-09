"""Database setup and failure-transparency tests."""

from pathlib import Path

import pytest

from invoice_agents.db.core import (
    DatabaseKind,
    connect_database,
    ensure_databases,
    migrate_database,
    seed_inventory,
    verify_database,
)
from invoice_agents.errors import DatabaseVerificationError


def test_ensure_databases_creates_seeds_and_is_repeatable(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.db"
    workflow = tmp_path / "workflow.db"
    first = ensure_databases(inventory, workflow)
    assert first == {"inventory": [1], "workflow": [1, 2, 3]}
    assert verify_database(inventory, DatabaseKind.INVENTORY)["integrity"] == "ok"
    assert verify_database(workflow, DatabaseKind.WORKFLOW)["integrity"] == "ok"
    second = ensure_databases(inventory, workflow)
    assert second == {"inventory": [], "workflow": []}


def test_migrate_seed_verify_is_repeatable(tmp_path: Path) -> None:
    path = tmp_path / "inventory.db"
    assert migrate_database(path, DatabaseKind.INVENTORY) == [1]
    assert migrate_database(path, DatabaseKind.INVENTORY) == []
    assert seed_inventory(path) == 4
    assert seed_inventory(path) == 4
    result = verify_database(path, DatabaseKind.INVENTORY)
    assert result["integrity"] == "ok"
    assert result["schema_version"] == 1


def test_missing_and_corrupt_database_fail_visibly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    with pytest.raises(DatabaseVerificationError, match="does not exist"):
        verify_database(missing, DatabaseKind.INVENTORY)
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not sqlite")
    with pytest.raises(DatabaseVerificationError, match="not a SQLite"):
        verify_database(corrupt, DatabaseKind.INVENTORY)


def test_wrong_version_and_missing_seed_fail(tmp_path: Path) -> None:
    path = tmp_path / "inventory.db"
    migrate_database(path, DatabaseKind.INVENTORY)
    with pytest.raises(DatabaseVerificationError, match="missing expected seed"):
        verify_database(path, DatabaseKind.INVENTORY)
    seed_inventory(path)
    with connect_database(path) as connection:
        connection.execute("UPDATE schema_version SET version = 99")
        connection.commit()
    with pytest.raises(DatabaseVerificationError, match="does not match"):
        verify_database(path, DatabaseKind.INVENTORY)


def test_read_only_connection_rejects_mutation(inventory_db: Path) -> None:
    with (
        connect_database(inventory_db, read_only=True) as connection,
        pytest.raises(Exception, match="readonly"),
    ):
        connection.execute("UPDATE inventory SET available_stock = 1")
