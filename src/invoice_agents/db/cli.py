"""CLI for explicit inventory and workflow database lifecycle operations."""

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

app = typer.Typer(no_args_is_help=True, help="Explicit SQLite setup and verification.")


@app.command("migrate")
def migrate_command(
    db: Annotated[Path, typer.Option("--db", help="Database file to create or migrate")],
    kind: Annotated[DatabaseKind | None, typer.Option()] = None,
) -> None:
    """Apply versioned migrations without seeding inferred data."""

    selected = kind or infer_kind(db)
    applied = migrate_database(db, selected)
    typer.echo(f"database={db.resolve()} kind={selected.value} applied={applied}")


@app.command("seed")
def seed_command(
    db: Annotated[Path, typer.Option("--db", help="Migrated inventory database")],
) -> None:
    """Seed the four inventory facts supplied by the challenge README."""

    count = seed_inventory(db)
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
    settings = None
    if selected is DatabaseKind.WORKFLOW:
        if inventory_db is None:
            raise typer.BadParameter(
                "--inventory-db is required for authoritative workflow verification"
            )
        settings = Settings(workflow_db=db, inventory_db=inventory_db)
    result = verify_database(db, selected, settings=settings)
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

    receipt = reconcile_legacy_authorization(
        db,
        reviewer=reviewer,
        reason=reason,
        disposition=disposition,
        confirmed=confirmed,
    )
    typer.echo(
        f"database={db.resolve()} reconciliation_id={receipt.reconciliation_id} "
        f"record_count={receipt.record_count} manifest={receipt.record_manifest_hash}"
    )
