"""Database setup and failure-transparency tests."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from invoice_agents.config import Settings
from invoice_agents.db import cli as database_cli
from invoice_agents.db.core import (
    DatabaseKind,
    _normalized_sql,
    connect_database,
    ensure_databases,
    migrate_database,
    seed_inventory,
    verify_database,
)
from invoice_agents.errors import DatabaseVerificationError


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("status = 'PAID'", "status = 'paid'"),
        ("payload = X'CAFE'", "payload = X'cafe'"),
        ('SELECT "PaidColumn"', 'SELECT "paidcolumn"'),
        ("SELECT `PaidColumn`", "SELECT `paidcolumn`"),
        ("SELECT [PaidColumn]", "SELECT [paidcolumn]"),
    ],
    ids=[
        "single-quoted-literal",
        "blob-literal",
        "double-quoted-identifier",
        "backtick-quoted-identifier",
        "bracket-quoted-identifier",
    ],
)
def test_sql_canonicalization_preserves_literal_and_quoted_identifier_bytes(
    left: str,
    right: str,
) -> None:
    assert _normalized_sql(left) != _normalized_sql(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("SELECT payments", "SELECT payment\u017f"),
        ("SELECT kelvin_k", "SELECT kelvin_\u212a"),
        ("SELECT strasse", "SELECT stra\u00dfe"),
        ("SELECT \u03c3", "SELECT \u03c2"),
        ("SELECT 'payment''s'", "SELECT 'payment''\u017f'"),
        ('SELECT "payment""s"', 'SELECT "payment""\u017f"'),
        ("SELECT `payment``s`", "SELECT `payment``\u017f`"),
    ],
    ids=[
        "long-s",
        "kelvin-sign",
        "sharp-s-expansion",
        "final-sigma",
        "escaped-string-literal",
        "escaped-double-quoted-identifier",
        "escaped-backtick-identifier",
    ],
)
def test_sql_canonicalization_preserves_non_ascii_codepoints(
    left: str,
    right: str,
) -> None:
    assert _normalized_sql(left) != _normalized_sql(right)


def test_sql_canonicalization_ignores_only_comments_spacing_keyword_case_and_creation_guard() -> (
    None
):
    packaged = """
        CREATE TRIGGER trg_paid BEFORE INSERT ON payments
        WHEN NEW.status = 'PAID'
        BEGIN
            SELECT RAISE(ABORT, 'PAYMENT_INVALID');
        END;
    """
    harmless_variant = """
        /* deployment guard and formatting are insignificant */
        create trigger if not exists trg_paid
        before insert on payments -- object semantics are unchanged
        when new.status='PAID'
        begin select raise ( abort , 'PAYMENT_INVALID' ) ; end
    """

    assert _normalized_sql(packaged) == _normalized_sql(harmless_variant)


def test_workflow_manifest_rejects_unicode_lookalike_trigger_redirect(
    settings: Settings,
) -> None:
    trigger_name = "trg_final_decisions_no_insert_after_paid"
    with connect_database(settings.workflow_db) as connection:
        original = str(
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                (trigger_name,),
            ).fetchone()[0]
        )
        connection.execute(
            "CREATE TABLE payment\u017f(case_id TEXT NOT NULL, status TEXT NOT NULL)"
        )
        redirected = original.replace("FROM payments", "FROM payment\u017f", 1)
        assert redirected != original
        connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.execute(redirected)
        connection.commit()
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('payments', 'payment\u017f')"
            )
        }
    assert names == {"payments", "payment\u017f"}

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )

    assert excinfo.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"


def test_workflow_manifest_accepts_harmless_trigger_formatting(settings: Settings) -> None:
    trigger_name = "trg_final_decisions_no_insert_after_paid"
    with connect_database(settings.workflow_db) as connection:
        original = str(
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                (trigger_name,),
            ).fetchone()[0]
        )
        reformatted = original.replace(
            f"CREATE TRIGGER {trigger_name}",
            f"create trigger if not exists {trigger_name}",
            1,
        ).replace(
            "BEFORE INSERT ON final_decisions",
            "before /* harmless formatting */ insert on final_decisions",
            1,
        )
        assert reformatted != original
        connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.execute(reformatted)
        connection.commit()

    assert (
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )["schema_version"]
        == 3
    )


def test_ensure_databases_creates_seeds_and_is_repeatable(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.db"
    workflow = tmp_path / "workflow.db"
    settings = Settings(inventory_db=inventory, workflow_db=workflow)
    first = ensure_databases(settings)
    assert first == {"inventory": [1], "workflow": [1, 2, 3]}
    assert verify_database(inventory, DatabaseKind.INVENTORY)["integrity"] == "ok"
    assert verify_database(workflow, DatabaseKind.WORKFLOW, settings=settings)["integrity"] == "ok"
    second = ensure_databases(settings)
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


def test_workflow_verify_cli_requires_explicit_inventory_context(tmp_path: Path) -> None:
    settings = Settings(
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
    )
    ensure_databases(settings)
    base = [
        "verify",
        "--db",
        str(settings.workflow_db),
        "--kind",
        "workflow",
    ]

    missing_context = CliRunner().invoke(database_cli.app, base)
    verified = CliRunner().invoke(
        database_cli.app,
        [*base, "--inventory-db", str(settings.inventory_db)],
    )

    assert missing_context.exit_code == 2
    assert "--inventory-db is required" in missing_context.stderr
    assert verified.exit_code == 0
    assert "'schema_version': 3" in verified.stdout


def test_workflow_migrate_cli_binds_legacy_v3_retrofit_to_inventory_context(
    tmp_path: Path,
) -> None:
    settings = Settings(
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
    )
    ensure_databases(settings)
    with connect_database(settings.workflow_db) as connection:
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'trg_schema_migration_history_%'"
        ).fetchall():
            connection.execute(f'DROP TRIGGER "{row["name"]}"')
        connection.execute("DROP TABLE schema_migration_history")
        connection.commit()
    base = [
        "migrate",
        "--db",
        str(settings.workflow_db),
        "--kind",
        "workflow",
    ]

    missing_context = CliRunner().invoke(database_cli.app, base)
    migrated = CliRunner().invoke(
        database_cli.app,
        [*base, "--inventory-db", str(settings.inventory_db)],
    )

    assert missing_context.exit_code == 1
    assert "error_code=DATABASE_AUTHORIZATION_CONTEXT_REQUIRED" in missing_context.stderr
    assert migrated.exit_code == 0
    assert "applied=[]" in migrated.stdout


def test_workflow_verification_validates_attached_inventory_schema_in_same_snapshot(
    tmp_path: Path,
) -> None:
    settings = Settings(
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
    )
    ensure_databases(settings)
    with connect_database(settings.inventory_db) as connection:
        connection.execute("DROP INDEX idx_item_aliases_sku")
        connection.commit()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )

    assert excinfo.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"


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
