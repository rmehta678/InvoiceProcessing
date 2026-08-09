"""CLI for explicit inventory and workflow database lifecycle operations."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from invoice_agents.config import Settings
from invoice_agents.db.core import (
    DatabaseKind,
    LegacyAuthorizationReconciliationReceipt,
    infer_kind,
    migrate_database,
    reconcile_legacy_authorization,
    seed_inventory,
    verify_database,
)
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError

app = typer.Typer(no_args_is_help=True, help="Explicit SQLite setup and verification.")


def _database_operation_settings(
    db: Path,
    kind: DatabaseKind,
    *,
    inventory_db: Path | None = None,
) -> Settings:
    """Validate fixed journal configuration before any database filesystem action."""

    try:
        if kind is DatabaseKind.WORKFLOW:
            settings = (
                Settings(workflow_db=db)
                if inventory_db is None
                else Settings(workflow_db=db, inventory_db=inventory_db)
            )
        else:
            settings = Settings(inventory_db=db)
    except ValidationError as exc:
        journal_error = any(error.get("loc") == ("sqlite_journal_mode",) for error in exc.errors())
        if not journal_error:
            raise
        raise InvoiceAgentsError(
            ErrorCategory.CONFIGURATION,
            "SQLite journal mode must be DELETE",
            stop_reason="SQLITE_JOURNAL_MODE_UNSUPPORTED",
        ) from None
    settings.assert_delete_journal_mode()
    return settings


def _run_database_operation[OperationResult](
    operation_name: str,
    operation: Callable[[], OperationResult],
) -> OperationResult:
    """Expose only stable, audit-safe error codes at the operator boundary."""

    try:
        return operation()
    except InvoiceAgentsError as exc:
        error_code = exc.stop_reason or "DATABASE_OPERATION_FAILED"
    except Exception:
        error_code = "DATABASE_OPERATION_FAILED"
    typer.echo(
        f"database_operation={operation_name} status=FAILED error_code={error_code}",
        err=True,
    )
    raise typer.Exit(1) from None


@app.command("migrate")
def migrate_command(
    db: Annotated[Path, typer.Option("--db", help="Database file to create or migrate")],
    kind: Annotated[DatabaseKind | None, typer.Option()] = None,
    inventory_db: Annotated[
        str | None,
        typer.Option(
            "--inventory-db",
            help="Authoritative inventory database required for a legacy workflow v3 retrofit",
        ),
    ] = None,
) -> None:
    """Apply versioned migrations without seeding inferred data."""

    def migrate() -> tuple[DatabaseKind, list[int]]:
        selected = kind or infer_kind(db)
        inventory_path = Path(inventory_db) if inventory_db is not None else None
        validated_settings = _database_operation_settings(
            db,
            selected,
            inventory_db=inventory_path,
        )
        migration_settings = (
            validated_settings
            if selected is DatabaseKind.INVENTORY or inventory_path is not None
            else None
        )
        return selected, migrate_database(db, selected, settings=migration_settings)

    selected, applied = _run_database_operation("migrate", migrate)
    typer.echo(f"database={db.resolve()} kind={selected.value} applied={applied}")


@app.command("seed")
def seed_command(
    db: Annotated[Path, typer.Option("--db", help="Migrated inventory database")],
) -> None:
    """Seed the four inventory facts supplied by the challenge README."""

    def seed() -> int:
        _database_operation_settings(db, DatabaseKind.INVENTORY)
        return seed_inventory(db)

    count = _run_database_operation("seed", seed)
    typer.echo(f"database={db.resolve()} seeded_rows={count}")


@app.command("verify")
def verify_command(
    db: Annotated[Path, typer.Option("--db", help="Database file to verify")],
    kind: Annotated[DatabaseKind | None, typer.Option()] = None,
    inventory_db: Annotated[
        str | None,
        typer.Option(
            "--inventory-db",
            help="Authoritative inventory database required for workflow verification",
        ),
    ] = None,
) -> None:
    """Fail loudly if signature, schema, integrity, or seed identity is wrong."""

    selected = kind or infer_kind(db)
    inventory_path = Path(inventory_db) if inventory_db is not None else None
    settings = _run_database_operation(
        "verify",
        lambda: _database_operation_settings(
            db,
            selected,
            inventory_db=inventory_path,
        ),
    )
    if selected is DatabaseKind.WORKFLOW and inventory_db is None:
        raise typer.BadParameter(
            "--inventory-db is required for authoritative workflow verification"
        )

    def verify() -> dict[str, object]:
        return verify_database(db, selected, settings=settings)

    result = _run_database_operation("verify", verify)
    typer.echo(str(result))


@app.command("reconcile-legacy-authorization")
def reconcile_legacy_authorization_command(
    db: Annotated[Path, typer.Option("--db", help="Legacy workflow database")],
    reviewer: Annotated[
        str,
        typer.Option("--reviewer", help="Operator responsible for the permanent disposition"),
    ],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Auditable reason for quarantining legacy authority"),
    ],
    disposition: Annotated[
        str,
        typer.Option(
            "--disposition",
            help="Required permanent non-authorizing disposition",
        ),
    ],
    confirmed: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm verified archival and permanent removal from active authority",
        ),
    ] = False,
) -> None:
    """Archive legacy authority exactly, then remove only its active copies."""

    def reconcile() -> LegacyAuthorizationReconciliationReceipt:
        _database_operation_settings(db, DatabaseKind.WORKFLOW)
        return reconcile_legacy_authorization(
            db,
            reviewer=reviewer,
            reason=reason,
            disposition=disposition,
            confirmed=confirmed,
        )

    receipt = _run_database_operation("reconcile-legacy-authorization", reconcile)
    typer.echo(
        f"database={db.resolve()} reconciliation_id={receipt.reconciliation_id} "
        f"record_count={receipt.record_count} manifest={receipt.record_manifest_hash}"
    )
