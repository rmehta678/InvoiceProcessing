"""Database setup and failure-transparency tests."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

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


def _directory_file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def _directory_file_state(directory: Path) -> dict[str, tuple[int, int, int, int, int, str]]:
    state: dict[str, tuple[int, int, int, int, int, str]] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        metadata = path.stat()
        state[path.name] = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return state


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


@pytest.mark.parametrize(
    "separator",
    ["\u000b", "\u0085", "\u00a0", "\u1680", "\u2003", "\u2028", "\u202f", "\u3000"],
    ids=[
        "vertical-tab",
        "next-line",
        "no-break-space",
        "ogham-space-mark",
        "em-space",
        "line-separator",
        "narrow-no-break-space",
        "ideographic-space",
    ],
)
def test_sql_canonicalization_preserves_non_sqlite_whitespace_codepoints(
    separator: str,
) -> None:
    normalized = _normalized_sql(f"SELECT{separator}payments")

    assert normalized != _normalized_sql("SELECT payments")
    assert any(separator in value for _kind, value in json.loads(normalized))


def test_sql_canonicalization_ignores_exact_sqlite_ascii_whitespace_bytes() -> None:
    assert _normalized_sql("SELECT\u0020\u0009\u000d\u000a\u000cpayments") == _normalized_sql(
        "SELECT payments"
    )


def test_sql_canonicalization_ignores_unicode_inside_comments_only() -> None:
    assert _normalized_sql(
        "SELECT/* comment contains \u00a0 \u2003 \u2028 */payments"
    ) == _normalized_sql("SELECT payments")
    assert _normalized_sql("SELECT -- comment contains \u00a0 \u2003\npayments") == _normalized_sql(
        "SELECT payments"
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("SELECT '\u00a0'", "SELECT ' '"),
        ('SELECT "paid\u2003status"', 'SELECT "paid status"'),
        ("SELECT `paid\u202fstatus`", "SELECT `paid status`"),
        ("SELECT [paid\u3000status]", "SELECT [paid status]"),
    ],
    ids=["literal", "double-quoted", "backtick-quoted", "bracket-quoted"],
)
def test_sql_canonicalization_preserves_unicode_whitespace_in_protected_tokens(
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


@pytest.mark.parametrize(
    "separator",
    ["\u00a0", "\u2003", "\u202f"],
    ids=["no-break-space", "em-space", "narrow-no-break-space"],
)
def test_workflow_manifest_rejects_required_trigger_with_unicode_separator(
    settings: Settings,
    separator: str,
) -> None:
    trigger_name = "trg_final_decisions_no_insert_after_paid"
    with connect_database(settings.workflow_db) as connection:
        original = str(
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                (trigger_name,),
            ).fetchone()[0]
        )
        modified = original.replace(
            "SELECT 1 FROM payments",
            f"SELECT 1 {separator} FROM payments",
            1,
        )
        assert modified != original
        connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.execute(modified)
        connection.commit()

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


@pytest.mark.parametrize(
    ("wal_target", "expected_stop_reason"),
    [
        ("workflow", "WORKFLOW_WAL_MODE_UNSUPPORTED"),
        ("inventory", "AUTHORIZATION_INVENTORY_WAL_MODE_UNSUPPORTED"),
    ],
)
def test_authoritative_workflow_verification_rejects_wal_header_without_artifacts(
    settings: Settings,
    wal_target: str,
    expected_stop_reason: str,
) -> None:
    target = settings.workflow_db if wal_target == "workflow" else settings.inventory_db
    with connect_database(target) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    assert target.read_bytes()[18:20] == b"\x02\x02"
    assert not Path(f"{target}-wal").exists()
    assert not Path(f"{target}-shm").exists()
    before = _directory_file_hashes(target.parent)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )

    assert excinfo.value.stop_reason == expected_stop_reason
    assert _directory_file_hashes(target.parent) == before


def test_standalone_inventory_verification_rejects_wal_without_persistent_artifacts(
    inventory_db: Path,
) -> None:
    with connect_database(inventory_db) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    assert not Path(f"{inventory_db}-wal").exists()
    assert not Path(f"{inventory_db}-shm").exists()
    before = _directory_file_hashes(inventory_db.parent)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(inventory_db, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "INVENTORY_WAL_MODE_UNSUPPORTED"
    assert _directory_file_hashes(inventory_db.parent) == before


def test_standalone_inventory_verification_does_not_touch_source_directory(
    inventory_db: Path,
) -> None:
    before = _directory_file_state(inventory_db.parent)

    result = verify_database(inventory_db, DatabaseKind.INVENTORY)

    assert result["path"] == str(inventory_db.resolve())
    assert _directory_file_state(inventory_db.parent) == before


def test_workflow_verification_does_not_touch_either_source_database(
    settings: Settings,
) -> None:
    before = _directory_file_state(settings.workflow_db.parent)

    result = verify_database(
        settings.workflow_db,
        DatabaseKind.WORKFLOW,
        settings=settings,
    )

    assert result["path"] == str(settings.workflow_db.resolve())
    assert _directory_file_state(settings.workflow_db.parent) == before


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_verification_rejects_rollback_database_with_any_sqlite_sidecar_unchanged(
    inventory_db: Path,
    suffix: str,
) -> None:
    sidecar = Path(f"{inventory_db}{suffix}")
    sidecar.write_bytes(f"preexisting{suffix}".encode())
    before = _directory_file_state(inventory_db.parent)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(inventory_db, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "DATABASE_SIDECAR_UNSUPPORTED"
    assert _directory_file_state(inventory_db.parent) == before


def test_authoritative_verification_detects_wal_switch_between_copy_and_temp_audit(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = connect_database
    switched = False

    def switch_source_then_connect(
        target: Path,
        *,
        read_only: bool = False,
    ) -> Any:
        nonlocal switched
        if not switched:
            with sqlite3.connect(settings.workflow_db) as rival:
                assert rival.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
            switched = True
        return real_connect(target, read_only=read_only)

    monkeypatch.setattr("invoice_agents.db.core.connect_database", switch_source_then_connect)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )

    assert switched
    assert excinfo.value.stop_reason == "DATABASE_CHANGED_DURING_VERIFICATION"


def test_authoritative_verification_detects_original_change_during_temp_audit(
    inventory_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = connect_database
    original = inventory_db.resolve()
    changed = False

    def change_source_then_connect(
        target: Path,
        *,
        read_only: bool = False,
    ) -> Any:
        nonlocal changed
        if Path(target).resolve() != original and not changed:
            metadata = original.stat()
            os.utime(
                original,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
            )
            changed = True
        return real_connect(target, read_only=read_only)

    monkeypatch.setattr("invoice_agents.db.core.connect_database", change_source_then_connect)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(inventory_db, DatabaseKind.INVENTORY)

    assert changed
    assert excinfo.value.stop_reason == "DATABASE_CHANGED_DURING_VERIFICATION"


@pytest.mark.parametrize("wal_target", ["inventory", "workflow"])
def test_ensure_databases_guards_every_existing_header_before_any_sqlite_open(
    settings: Settings,
    wal_target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = settings.inventory_db if wal_target == "inventory" else settings.workflow_db
    with sqlite3.connect(target) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    before = _directory_file_state(target.parent)
    real_connect = connect_database
    source_opens: list[Path] = []

    def observe_connect(
        database: Path,
        *,
        read_only: bool = False,
    ) -> Any:
        resolved = Path(database).resolve()
        if resolved in {settings.inventory_db.resolve(), settings.workflow_db.resolve()}:
            source_opens.append(resolved)
        return real_connect(database, read_only=read_only)

    monkeypatch.setattr("invoice_agents.db.core.connect_database", observe_connect)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        ensure_databases(settings)

    assert excinfo.value.stop_reason == (
        "INVENTORY_WAL_MODE_UNSUPPORTED"
        if wal_target == "inventory"
        else "WORKFLOW_WAL_MODE_UNSUPPORTED"
    )
    assert source_opens == []
    assert _directory_file_state(target.parent) == before


def test_twenty_byte_sqlite_lookalike_is_signature_invalid(tmp_path: Path) -> None:
    path = tmp_path / "twenty-byte-lookalike.db"
    path.write_bytes(b"SQLite format 3\x00\x00\x00\x02\x02")

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "DATABASE_SIGNATURE_INVALID"


@pytest.mark.parametrize(
    "corruption",
    [
        "truncated-header",
        "page-size",
        "write-version",
        "read-version",
        "payload-fractions",
        "schema-format",
        "text-encoding",
        "page-count",
        "file-size",
        "reserved-header-bytes",
    ],
)
def test_complete_sqlite_header_contract_rejects_each_invalid_invariant(
    inventory_db: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    data = bytearray(inventory_db.read_bytes())
    if corruption == "truncated-header":
        data = data[:99]
    elif corruption == "page-size":
        data[16:18] = b"\x00\x00"
    elif corruption == "write-version":
        data[18] = 3
    elif corruption == "read-version":
        data[19] = 3
    elif corruption == "payload-fractions":
        data[21:24] = b"\x40\x20\x21"
    elif corruption == "schema-format":
        data[44:48] = (5).to_bytes(4, "big")
    elif corruption == "text-encoding":
        data[56:60] = (4).to_bytes(4, "big")
    elif corruption == "page-count":
        page_count = int.from_bytes(data[28:32], "big")
        data[28:32] = (page_count + 1).to_bytes(4, "big")
    elif corruption == "file-size":
        data.extend(b"x")
    else:
        data[72] = 1
    path = tmp_path / f"invalid-header-{corruption}.db"
    path.write_bytes(data)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "DATABASE_SIGNATURE_INVALID"


def test_migration_accepts_nonexistent_and_zero_length_new_database_paths(
    tmp_path: Path,
) -> None:
    nonexistent = tmp_path / "new-inventory.db"
    zero_length = tmp_path / "zero-length-inventory.db"
    zero_length.touch()

    assert migrate_database(nonexistent, DatabaseKind.INVENTORY) == [1]
    assert migrate_database(zero_length, DatabaseKind.INVENTORY) == [1]
    assert nonexistent.read_bytes()[18:20] == b"\x01\x01"
    assert zero_length.read_bytes()[18:20] == b"\x01\x01"


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
