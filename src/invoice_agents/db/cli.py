"""CLI for explicit inventory and workflow database lifecycle operations."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from invoice_agents.config import Settings
from invoice_agents.db.core import (
    DatabaseKind,
    infer_kind,
    migrate_database,
    reconcile_legacy_authorization,
    seed_inventory,
    verify_database,
)
from invoice_agents.errors import InvoiceAgentsError

app = typer.Typer(no_args_is_help=True, help="Explicit SQLite setup and verification.")


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
) -> None:
    """Apply versioned migrations without seeding inferred data."""

    selected = kind or infer_kind(db)
    applied = _run_database_operation(
        "migrate",
        lambda: migrate_database(db, selected),
    )
    typer.echo(f"database={db.resolve()} kind={selected.value} applied={applied}")


@app.command("seed")
def seed_command(
    db: Annotated[Path, typer.Option("--db", help="Migrated inventory database")],
) -> None:
    """Seed the four inventory facts supplied by the challenge README."""

    count = _run_database_operation("seed", lambda: seed_inventory(db))
    typer.echo(f"database={db.resolve()} seeded_rows={count}")


@app.command("verify")
def verify_command(
    db: Annotated[Path, typer.Option("--db", help="Database file to verify")],
    kind: Annotated[DatabaseKind | None, typer.Option()] = None,
    inventory_db: Annotated[
        Path | None,
        typer.Option(
            "--inventory-db",
            help="Authoritative inventory database required for workflow verification",
        ),
    ] = None,
) -> None:
    """Fail loudly if signature, schema, integrity, or seed identity is wrong."""

    selected = kind or infer_kind(db)
    if selected is DatabaseKind.WORKFLOW and inventory_db is None:
        raise typer.BadParameter(
            "--inventory-db is required for authoritative workflow verification"
        )

    def verify() -> dict[str, object]:
        settings = (
            Settings(workflow_db=db, inventory_db=inventory_db)
            if selected is DatabaseKind.WORKFLOW
            else None
        )
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

    receipt = _run_database_operation(
        "reconcile-legacy-authorization",
        lambda: reconcile_legacy_authorization(
            db,
            reviewer=reviewer,
            reason=reason,
            disposition=disposition,
            confirmed=confirmed,
        ),
    )
    typer.echo(
        f"database={db.resolve()} reconciliation_id={receipt.reconciliation_id} "
        f"record_count={receipt.record_count} manifest={receipt.record_manifest_hash}"
    )
