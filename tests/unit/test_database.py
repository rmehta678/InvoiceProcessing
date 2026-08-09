"""Database setup and failure-transparency tests."""

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import invoice_agents.db.core as core_module
import invoice_agents.db.sqlite_source as sqlite_source_module
from invoice_agents.config import Settings
from invoice_agents.db import cli as database_cli
from invoice_agents.db.core import (
    DatabaseKind,
    _migrate_database_in_process,
    _normalized_sql,
    connect_database,
    ensure_databases,
    migrate_database,
    seed_inventory,
    verify_database,
)
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import DatabaseVerificationError, InvoiceAgentsError


@pytest.mark.parametrize("journal_mode", ["PERSIST", "TRUNCATE", "WAL"])
def test_settings_rejects_non_delete_journal_mode_before_database_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_mode: str,
) -> None:
    sqlite_opens: list[object] = []

    def observe_sqlite_open(*args: object, **kwargs: object) -> sqlite3.Connection:
        sqlite_opens.append((args, kwargs))
        raise AssertionError("invalid configuration must fail before SQLite opens")

    monkeypatch.setattr(sqlite3, "connect", observe_sqlite_open)

    with pytest.raises(ValidationError, match="DELETE"):
        Settings(
            inventory_db=tmp_path / "inventory.db",
            workflow_db=tmp_path / "workflow.db",
            sqlite_journal_mode=journal_mode,
        )

    assert sqlite_opens == []


def test_settings_normalizes_the_only_supported_sqlite_journal_mode() -> None:
    settings = Settings(sqlite_journal_mode=" delete ")

    assert settings.sqlite_journal_mode == "DELETE"


@pytest.mark.parametrize("journal_mode", ["PERSIST", "TRUNCATE", "WAL"])
def test_core_and_store_reject_bypassed_non_delete_config_before_database_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_mode: str,
) -> None:
    settings = Settings(
        inventory_db=tmp_path / "inventory.db",
        workflow_db=tmp_path / "workflow.db",
    ).model_copy(update={"sqlite_journal_mode": journal_mode})
    sqlite_opens: list[object] = []

    def reject_sqlite_open(*args: object, **kwargs: object) -> Any:
        sqlite_opens.append((args, kwargs))
        raise AssertionError("invalid production configuration reached SQLite")

    monkeypatch.setattr(core_module, "connect_database", reject_sqlite_open)

    with pytest.raises(InvoiceAgentsError) as core_error:
        ensure_databases(settings)
    with pytest.raises(InvoiceAgentsError) as store_error:
        WorkflowStore(settings)

    assert core_error.value.stop_reason == "SQLITE_JOURNAL_MODE_UNSUPPORTED"
    assert store_error.value.stop_reason == "SQLITE_JOURNAL_MODE_UNSUPPORTED"
    assert sqlite_opens == []
    assert not settings.inventory_db.exists()
    assert not settings.workflow_db.exists()


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


@pytest.mark.parametrize("journal_mode", ["PERSIST", "TRUNCATE", "WAL"])
@pytest.mark.parametrize(
    "operation",
    ["migrate", "verify", "seed", "reconcile-legacy-authorization"],
)
def test_every_database_cli_operation_rejects_unsupported_journal_mode_before_filesystem_action(
    tmp_path: Path,
    journal_mode: str,
    operation: str,
) -> None:
    target = tmp_path / f"{operation}.db"
    arguments = [operation, "--db", str(target)]
    if operation in {"migrate", "verify"}:
        arguments.extend(("--kind", "inventory"))
    elif operation == "reconcile-legacy-authorization":
        arguments.extend(
            (
                "--reviewer",
                "operator@example.test",
                "--reason",
                "permanent quarantine",
                "--disposition",
                "PERMANENTLY_QUARANTINED_NON_AUTHORIZING",
                "--confirm",
            )
        )
    secret = "sk-proj-database-cli-must-not-leak"

    result = CliRunner().invoke(
        database_cli.app,
        arguments,
        env={
            "INVOICE_SQLITE_JOURNAL_MODE": journal_mode,
            "XAI_API_KEY": secret,
        },
    )

    assert result.exit_code == 1
    assert "error_code=SQLITE_JOURNAL_MODE_UNSUPPORTED" in result.stderr
    assert secret not in result.output
    assert list(tmp_path.iterdir()) == []


def test_database_cli_operations_accept_delete_mode(tmp_path: Path) -> None:
    target = tmp_path / "delete-mode-inventory.db"
    runner = CliRunner()
    environment = {"INVOICE_SQLITE_JOURNAL_MODE": " delete "}

    migrated = runner.invoke(
        database_cli.app,
        ["migrate", "--db", str(target), "--kind", "inventory"],
        env=environment,
    )
    seeded = runner.invoke(database_cli.app, ["seed", "--db", str(target)], env=environment)
    verified = runner.invoke(
        database_cli.app,
        ["verify", "--db", str(target), "--kind", "inventory"],
        env=environment,
    )

    assert migrated.exit_code == 0
    assert "applied=[1]" in migrated.stdout
    assert seeded.exit_code == 0
    assert "seeded_rows=4" in seeded.stdout
    assert verified.exit_code == 0
    assert "'integrity': 'ok'" in verified.stdout


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


@pytest.mark.parametrize("link_kind", ["leaf", "parent"])
def test_authoritative_verification_rejects_symlink_database_path_without_artifacts(
    inventory_db: Path,
    tmp_path: Path,
    link_kind: str,
) -> None:
    before = inventory_db.read_bytes()
    before_names = {candidate.name for candidate in tmp_path.iterdir()}
    if link_kind == "leaf":
        link = tmp_path / "inventory-link.db"
        link.symlink_to(inventory_db.name)
        supplied = link
    else:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        target = real_parent / "inventory.db"
        target.write_bytes(before)
        link = tmp_path / "inventory-parent-link"
        link.symlink_to(real_parent.name, target_is_directory=True)
        supplied = link / target.name
    expected_link = os.readlink(link)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(supplied, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "DATABASE_SYMLINK_UNSUPPORTED"
    assert inventory_db.read_bytes() == before
    assert os.readlink(link) == expected_link
    assert not Path(f"{supplied}-journal").exists()
    assert not Path(f"{supplied}-wal").exists()
    assert not Path(f"{supplied}-shm").exists()
    assert {candidate.name for candidate in tmp_path.iterdir()} == before_names | {link.name} | (
        {"real-parent"} if link_kind == "parent" else set()
    )


def test_authoritative_verification_rejects_parent_replacement_before_leaf_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parent = tmp_path / "authoritative"
    replacement_parent = tmp_path / "replacement"
    moved_original = tmp_path / "moved-authoritative"
    original_parent.mkdir()
    replacement_parent.mkdir()
    supplied = original_parent / "inventory.db"
    replacement = replacement_parent / supplied.name
    migrate_database(supplied, DatabaseKind.INVENTORY)
    migrate_database(replacement, DatabaseKind.INVENTORY)
    seed_inventory(replacement)
    original_bytes = supplied.read_bytes()
    replacement_bytes = replacement.read_bytes()
    real_open = os.open
    replaced = False

    def replace_parent_before_leaf_open(
        target: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        target_text = os.fsdecode(target)
        is_source_leaf = Path(target_text) == supplied or (
            target_text == supplied.name
            and dir_fd is not None
            and not flags & getattr(os, "O_DIRECTORY", 0)
        )
        if is_source_leaf and not replaced:
            original_parent.rename(moved_original)
            replacement_parent.rename(original_parent)
            replaced = True
        return real_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sqlite_source_module.os, "open", replace_parent_before_leaf_open)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(supplied, DatabaseKind.INVENTORY)

    assert replaced
    assert excinfo.value.stop_reason == "DATABASE_CHANGED_DURING_VERIFICATION"
    assert (moved_original / supplied.name).read_bytes() == original_bytes
    assert supplied.read_bytes() == replacement_bytes


def test_workflow_verification_rejects_symlink_inventory_context_without_audit(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory_link = tmp_path / "authorization-inventory-link.db"
    inventory_link.symlink_to(settings.inventory_db.name)
    linked_settings = settings.model_copy(update={"inventory_db": inventory_link})
    workflow_before = settings.workflow_db.read_bytes()
    inventory_before = settings.inventory_db.read_bytes()
    audit_started = False
    real_verify_snapshot = core_module._verify_database_snapshot

    def observe_audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal audit_started
        audit_started = True
        return real_verify_snapshot(*args, **kwargs)

    monkeypatch.setattr(core_module, "_verify_database_snapshot", observe_audit)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=linked_settings,
        )

    assert excinfo.value.stop_reason == "DATABASE_SYMLINK_UNSUPPORTED"
    assert not audit_started
    assert settings.workflow_db.read_bytes() == workflow_before
    assert settings.inventory_db.read_bytes() == inventory_before
    assert inventory_link.is_symlink()


def test_workflow_verification_rejects_symlink_workflow_context_without_audit(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_link = tmp_path / "workflow-context-link.db"
    workflow_link.symlink_to(settings.workflow_db.name)
    linked_settings = settings.model_copy(update={"workflow_db": workflow_link})
    audit_started = False
    real_verify_snapshot = core_module._verify_database_snapshot

    def observe_audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal audit_started
        audit_started = True
        return real_verify_snapshot(*args, **kwargs)

    monkeypatch.setattr(core_module, "_verify_database_snapshot", observe_audit)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=linked_settings,
        )

    assert excinfo.value.stop_reason == "DATABASE_SYMLINK_UNSUPPORTED"
    assert not audit_started
    assert workflow_link.is_symlink()
    assert not Path(f"{workflow_link}-journal").exists()


def test_workflow_verification_rejects_inventory_retarget_before_temp_audit(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moved_inventory = tmp_path / "inventory-before-retarget.db"
    replacement_inventory = tmp_path / "inventory-retarget.db"
    replacement_inventory.write_bytes(settings.inventory_db.read_bytes())
    original_copy = sqlite_source_module._copy_validated_source
    audit_started = False
    retargeted = False

    def copy_then_retarget(
        path: Path,
        destination: Path,
        role: sqlite_source_module.SQLiteSourceRole,
    ) -> sqlite_source_module.SQLiteSourceIdentity:
        nonlocal retargeted
        identity = original_copy(path, destination, role)
        if role.key == "authorization_inventory" and not retargeted:
            settings.inventory_db.rename(moved_inventory)
            settings.inventory_db.symlink_to(replacement_inventory.name)
            retargeted = True
        return identity

    real_verify_snapshot = core_module._verify_database_snapshot

    def observe_audit(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal audit_started
        audit_started = True
        return real_verify_snapshot(*args, **kwargs)

    monkeypatch.setattr(sqlite_source_module, "_copy_validated_source", copy_then_retarget)
    monkeypatch.setattr(core_module, "_verify_database_snapshot", observe_audit)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )

    assert retargeted
    assert not audit_started
    assert excinfo.value.stop_reason == "DATABASE_CHANGED_DURING_VERIFICATION"
    assert settings.inventory_db.is_symlink()
    assert moved_inventory.read_bytes() == replacement_inventory.read_bytes()


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
        "schema-format-zero-only",
        "text-encoding-zero-only",
        "page-count",
        "file-size",
        "first-freelist-page",
        "freelist-page-count",
        "largest-root-page",
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
    elif corruption == "schema-format-zero-only":
        data[44:48] = (0).to_bytes(4, "big")
    elif corruption == "text-encoding-zero-only":
        data[56:60] = (0).to_bytes(4, "big")
    elif corruption == "page-count":
        page_count = int.from_bytes(data[28:32], "big")
        data[28:32] = (page_count + 1).to_bytes(4, "big")
    elif corruption == "file-size":
        data.extend(b"x")
    elif corruption == "first-freelist-page":
        data[32:36] = (2**32 - 1).to_bytes(4, "big")
    elif corruption == "freelist-page-count":
        data[36:40] = (2**32 - 1).to_bytes(4, "big")
    elif corruption == "largest-root-page":
        data[52:56] = (2**32 - 1).to_bytes(4, "big")
    else:
        data[72] = 1
    path = tmp_path / f"invalid-header-{corruption}.db"
    path.write_bytes(data)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "DATABASE_SIGNATURE_INVALID"


def test_user_version_only_pre_schema_database_migrates_from_legal_zero_header_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "user-version-only.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 73")
        connection.commit()
    header = path.read_bytes()[:100]
    assert int.from_bytes(header[44:48], "big") == 0
    assert int.from_bytes(header[56:60], "big") == 0

    assert migrate_database(path, DatabaseKind.INVENTORY) == [1]

    with connect_database(path, read_only=True) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 73
        assert connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1


def test_zero_header_pair_with_user_schema_is_audited_and_rejected_before_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "false-pre-schema.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unauthorized (value BLOB)")
        connection.execute("INSERT INTO unauthorized(value) VALUES (X'CAFE')")
        connection.commit()
    data = bytearray(path.read_bytes())
    data[44:48] = (0).to_bytes(4, "big")
    data[56:60] = (0).to_bytes(4, "big")
    path.write_bytes(data)
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert path.read_bytes() == before
    assert not Path(f"{path}-journal").exists()


@pytest.mark.parametrize(
    "object_name",
    [
        "sqliteXevil",
        "SQLiteXevil",
        "sqlite%evil",
        "sqlite\uff3fevil",
        "sqlite\U0001f600evil",
        "sql\u0131te_evil",
        "\u0455qlite_evil",
    ],
)
def test_zero_header_pair_rejects_like_metacharacter_and_unicode_schema_names_without_mutation(
    tmp_path: Path,
    object_name: str,
) -> None:
    path = tmp_path / "forged-lookalike-schema.db"
    quoted_name = object_name.replace('"', '""')
    with sqlite3.connect(path) as connection:
        connection.execute(f'CREATE TABLE "{quoted_name}" (value BLOB UNIQUE)')
        connection.execute(f"INSERT INTO \"{quoted_name}\"(value) VALUES (X'CAFE')")
        schema_names = {
            str(row[0]) for row in connection.execute("SELECT name FROM sqlite_schema").fetchall()
        }
        connection.commit()
    assert object_name in schema_names
    assert f"sqlite_autoindex_{object_name}_1" in schema_names
    data = bytearray(path.read_bytes())
    data[44:48] = (0).to_bytes(4, "big")
    data[56:60] = (0).to_bytes(4, "big")
    path.write_bytes(data)
    before = _directory_file_state(tmp_path)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert _directory_file_state(tmp_path) == before


@pytest.mark.parametrize(
    ("internal_name", "create_internal", "expected_columns"),
    [
        (
            "sqlite_sequence",
            "CREATE TABLE discarded (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE discarded;",
            ((0, "name", "", 0, None, 0, 0), (1, "seq", "", 0, None, 0, 0)),
        ),
        (
            "sqlite_stat1",
            "ANALYZE;",
            (
                (0, "tbl", "", 0, None, 0, 0),
                (1, "idx", "", 0, None, 0, 0),
                (2, "stat", "", 0, None, 0, 0),
            ),
        ),
    ],
    ids=["create-drop-autoincrement", "analyze-empty-database"],
)
def test_zero_header_pair_with_genuine_empty_sqlite_runtime_table_migrates(
    tmp_path: Path,
    internal_name: str,
    create_internal: str,
    expected_columns: tuple[tuple[object, ...], ...],
) -> None:
    path = tmp_path / f"genuine-{internal_name}.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(create_internal)
        schema_row = connection.execute(
            "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_schema WHERE name = ?",
            (internal_name,),
        ).fetchone()
        columns = tuple(connection.execute(f"PRAGMA table_xinfo({internal_name})").fetchall())
        connection.commit()
    assert schema_row is not None
    assert schema_row[:3] == ("table", internal_name, internal_name)
    assert type(schema_row[3]) is int and schema_row[3] > 0
    assert columns == expected_columns
    data = bytearray(path.read_bytes())
    data[44:48] = (0).to_bytes(4, "big")
    data[56:60] = (0).to_bytes(4, "big")
    path.write_bytes(data)

    assert migrate_database(path, DatabaseKind.INVENTORY) == [1]

    with connect_database(path, read_only=True) as connection:
        assert connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("internal_name", "create_internal", "forgery"),
    [
        (
            "sqlite_sequence",
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE t",
            "type-case",
        ),
        (
            "sqlite_sequence",
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE t",
            "name-case",
        ),
        (
            "sqlite_sequence",
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE t",
            "table-name-case",
        ),
        (
            "sqlite_sequence",
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE t",
            "blob-rootpage",
        ),
        (
            "sqlite_sequence",
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE t",
            "quoted-sql",
        ),
        (
            "sqlite_sequence",
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE t",
            "extra-column",
        ),
        (
            "sqlite_sequence",
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE t",
            "referencing-view",
        ),
        ("sqlite_stat1", "ANALYZE", "type-case"),
        ("sqlite_stat1", "ANALYZE", "name-case"),
        ("sqlite_stat1", "ANALYZE", "table-name-case"),
        ("sqlite_stat1", "ANALYZE", "blob-rootpage"),
        ("sqlite_stat1", "ANALYZE", "quoted-sql"),
        ("sqlite_stat1", "ANALYZE", "extra-column"),
        ("sqlite_stat1", "ANALYZE", "referencing-view"),
    ],
)
def test_zero_header_pair_rejects_forged_sqlite_runtime_table_without_mutation(
    tmp_path: Path,
    internal_name: str,
    create_internal: str,
    forgery: str,
) -> None:
    path = tmp_path / f"forged-{internal_name}-{forgery}.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(create_internal)
        connection.execute("PRAGMA writable_schema = ON")
        if forgery == "type-case":
            connection.execute(
                "UPDATE sqlite_schema SET type = 'TABLE' WHERE name = ?", (internal_name,)
            )
        elif forgery == "name-case":
            connection.execute(
                "UPDATE sqlite_schema SET name = upper(name) WHERE name = ?", (internal_name,)
            )
        elif forgery == "table-name-case":
            connection.execute(
                "UPDATE sqlite_schema SET tbl_name = upper(tbl_name) WHERE name = ?",
                (internal_name,),
            )
        elif forgery == "blob-rootpage":
            connection.execute(
                "UPDATE sqlite_schema SET rootpage = CAST(rootpage AS BLOB) WHERE name = ?",
                (internal_name,),
            )
        elif forgery == "quoted-sql":
            canonical_sql = {
                "sqlite_sequence": 'CREATE TABLE "sqlite_sequence"(name,seq)',
                "sqlite_stat1": 'CREATE TABLE "sqlite_stat1"(tbl,idx,stat)',
            }[internal_name]
            connection.execute(
                "UPDATE sqlite_schema SET sql = ? WHERE name = ?",
                (canonical_sql, internal_name),
            )
        elif forgery == "extra-column":
            decorated_sql = {
                "sqlite_sequence": "CREATE TABLE sqlite_sequence(name,seq,extra)",
                "sqlite_stat1": "CREATE TABLE sqlite_stat1(tbl,idx,stat,extra)",
            }[internal_name]
            connection.execute(
                "UPDATE sqlite_schema SET sql = ? WHERE name = ?",
                (decorated_sql, internal_name),
            )
        else:
            connection.execute(
                "INSERT INTO sqlite_schema(type, name, tbl_name, rootpage, sql) "
                "VALUES ('view', 'runtime_table_reader', 'runtime_table_reader', 0, ?)",
                (f"CREATE VIEW runtime_table_reader AS SELECT * FROM {internal_name}",),
            )
        connection.execute("PRAGMA writable_schema = RESET")
        connection.commit()
    data = bytearray(path.read_bytes())
    data[44:48] = (0).to_bytes(4, "big")
    data[56:60] = (0).to_bytes(4, "big")
    path.write_bytes(data)
    before = _directory_file_state(tmp_path)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert _directory_file_state(tmp_path) == before


def test_schema_classifier_excludes_only_a_proven_sqlite_owned_autoindex(
    tmp_path: Path,
) -> None:
    path = tmp_path / "schema-classification.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE ordinary (value BLOB UNIQUE)")
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "INSERT INTO sqlite_schema(type, name, tbl_name, rootpage, sql) "
            "VALUES ('view', 'sqlite_forged', 'sqlite_forged', 0, "
            "'CREATE VIEW sqlite_forged AS SELECT 1')"
        )
        connection.execute("PRAGMA writable_schema = RESET")
        connection.execute("ANALYZE")
        connection.commit()

    with connect_database(path, read_only=True) as connection:
        objects = core_module._non_internal_schema_objects(connection)

    assert tuple((entry.object_type, entry.name) for entry in objects) == (
        ("table", "ordinary"),
        ("view", "sqlite_forged"),
    )


@pytest.mark.parametrize(
    "create_internal",
    [
        "CREATE TABLE discarded (id INTEGER PRIMARY KEY AUTOINCREMENT); DROP TABLE discarded",
        "ANALYZE",
    ],
    ids=["sqlite-sequence", "sqlite-stat1"],
)
def test_workflow_manifest_accepts_only_genuine_runtime_internal_tables(
    settings: Settings,
    create_internal: str,
) -> None:
    with connect_database(settings.workflow_db) as connection:
        connection.executescript(create_internal)
        connection.commit()

    result = verify_database(
        settings.workflow_db,
        DatabaseKind.WORKFLOW,
        settings=settings,
    )

    assert result["schema_version"] == core_module.SCHEMA_VERSIONS[DatabaseKind.WORKFLOW]


def test_workflow_manifest_rejects_wrong_storage_type_on_runtime_internal_table(
    settings: Settings,
) -> None:
    with connect_database(settings.workflow_db) as connection:
        connection.execute("ANALYZE")
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_schema SET type = CAST(type AS BLOB) WHERE name = 'sqlite_stat1'"
        )
        connection.execute("PRAGMA writable_schema = RESET")
        connection.commit()
    before = settings.workflow_db.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )

    assert excinfo.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"
    assert settings.workflow_db.read_bytes() == before


@pytest.mark.parametrize(
    ("object_name", "object_sql"),
    [
        ("sqlite_forged", 'CREATE VIEW "sqlite_forged" AS SELECT 1'),
        (
            "sqlite_autoindex_forged_1",
            'CREATE VIEW "sqlite_autoindex_forged_1" AS SELECT 1',
        ),
    ],
)
def test_zero_header_pair_rejects_forged_reserved_sqlite_schema_rows_without_mutation(
    tmp_path: Path,
    object_name: str,
    object_sql: str,
) -> None:
    path = tmp_path / "forged-reserved-schema.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE writable_schema_seed (value BLOB)")
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute("DELETE FROM sqlite_schema")
        connection.execute(
            "INSERT INTO sqlite_schema(type, name, tbl_name, rootpage, sql) "
            "VALUES ('view', ?, ?, 0, ?)",
            (object_name, object_name, object_sql),
        )
        connection.execute("PRAGMA writable_schema = RESET")
        connection.commit()
    data = bytearray(path.read_bytes())
    data[44:48] = (0).to_bytes(4, "big")
    data[56:60] = (0).to_bytes(4, "big")
    path.write_bytes(data)
    before = _directory_file_state(tmp_path)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert _directory_file_state(tmp_path) == before


def test_zero_header_pair_with_schema_version_is_rejected_before_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forged-pre-schema-history.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.commit()
    data = bytearray(path.read_bytes())
    data[44:48] = (0).to_bytes(4, "big")
    data[56:60] = (0).to_bytes(4, "big")
    path.write_bytes(data)
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert path.read_bytes() == before
    assert not Path(f"{path}-journal").exists()


@pytest.mark.parametrize("initial_state", ["missing", "zero-length"])
@pytest.mark.parametrize(
    "sidecar_suffix",
    ["-journal", "-wal", "-shm", "-mj 123456789", "-mj123456789"],
)
def test_new_database_migration_rejects_retained_sidecars_without_mutation(
    tmp_path: Path,
    initial_state: str,
    sidecar_suffix: str,
) -> None:
    path = tmp_path / f"{initial_state}.db"
    if initial_state == "zero-length":
        path.touch()
    sidecar = Path(f"{path}{sidecar_suffix}")
    sidecar.write_bytes(b"retained-sidecar-evidence")
    before = _directory_file_state(tmp_path)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "DATABASE_SIDECAR_UNSUPPORTED"
    assert _directory_file_state(tmp_path) == before
    if initial_state == "missing":
        assert not path.exists()
    else:
        assert path.read_bytes() == b""


def test_private_migration_rechecks_sidecars_after_lock_before_sqlite_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "late-sidecar.db"
    path.touch()
    sidecar = Path(f"{path}-journal")
    real_maintenance = core_module.exclusive_database_maintenance
    injected = False

    @contextmanager
    def inject_sidecar_after_lock(
        paths: tuple[Path, ...],
        *,
        create_paths: tuple[Path, ...] = (),
    ) -> Any:
        nonlocal injected
        with real_maintenance(paths, create_paths=create_paths) as locks:
            sidecar.write_bytes(b"appeared-after-lock")
            injected = True
            yield locks

    monkeypatch.setattr(core_module, "exclusive_database_maintenance", inject_sidecar_after_lock)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        _migrate_database_in_process(path, DatabaseKind.INVENTORY)

    assert injected
    assert excinfo.value.stop_reason == "DATABASE_SIDECAR_UNSUPPORTED"
    assert path.read_bytes() == b""
    assert sidecar.read_bytes() == b"appeared-after-lock"


def test_private_migration_does_not_forget_sidecar_removed_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "removed-before-snapshot.db"
    migrate_database(path, DatabaseKind.INVENTORY)
    before = path.read_bytes()
    sidecar = Path(f"{path}-journal")
    sidecar.write_bytes(b"preflight-sidecar")
    real_snapshots = core_module.authoritative_database_snapshots
    removed = False

    @contextmanager
    def remove_sidecar_before_snapshot(sources: Any) -> Any:
        nonlocal removed
        sidecar.unlink()
        removed = True
        with real_snapshots(sources) as snapshots:
            yield snapshots

    monkeypatch.setattr(
        core_module,
        "authoritative_database_snapshots",
        remove_sidecar_before_snapshot,
    )

    with pytest.raises(DatabaseVerificationError) as excinfo:
        _migrate_database_in_process(path, DatabaseKind.INVENTORY)

    assert removed
    assert excinfo.value.stop_reason == "DATABASE_SIDECAR_UNSUPPORTED"
    assert path.read_bytes() == before
    assert not sidecar.exists()


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
    assert not [
        candidate
        for candidate in tmp_path.iterdir()
        if candidate.is_dir() and candidate.name.startswith(".invoice-db-maintenance-")
    ]


def test_migration_rejects_leaf_symlink_before_lock_or_sqlite_artifacts(
    inventory_db: Path,
    tmp_path: Path,
) -> None:
    link = tmp_path / "inventory-migration-link.db"
    link.symlink_to(inventory_db.name)
    before = inventory_db.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(link, DatabaseKind.INVENTORY)

    assert excinfo.value.stop_reason == "DATABASE_SYMLINK_UNSUPPORTED"
    assert link.is_symlink()
    assert os.readlink(link) == inventory_db.name
    assert inventory_db.read_bytes() == before
    assert not Path(f"{link}-journal").exists()
    assert not [
        candidate
        for candidate in tmp_path.iterdir()
        if candidate.is_dir() and candidate.name.startswith(".invoice-db-maintenance-")
    ]


def test_private_hardlink_maintenance_binding_is_0700_and_cleans_up_on_exception(
    inventory_db: Path,
) -> None:
    before = inventory_db.read_bytes()
    maintenance_prefix = ".invoice-db-maintenance-"

    with (
        pytest.raises(RuntimeError, match="injected maintenance failure"),
        sqlite_source_module.exclusive_database_maintenance((inventory_db,)) as locks,
    ):
        sqlite_path = locks.sqlite_path(inventory_db)
        directory = sqlite_path.parent
        source_metadata = os.lstat(inventory_db)
        hardlink_metadata = os.lstat(sqlite_path)

        assert directory.parent == inventory_db.parent
        assert directory.name.startswith(maintenance_prefix)
        assert os.lstat(directory).st_mode & 0o777 == 0o700
        assert not sqlite_path.is_symlink()
        assert (hardlink_metadata.st_dev, hardlink_metadata.st_ino) == (
            source_metadata.st_dev,
            source_metadata.st_ino,
        )
        raise RuntimeError("injected maintenance failure")

    assert inventory_db.read_bytes() == before
    assert not [
        candidate
        for candidate in inventory_db.parent.iterdir()
        if candidate.name.startswith(maintenance_prefix)
    ]
    assert not Path(f"{inventory_db}-journal").exists()


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
