"""Human-oriented CLI with full structured state retained in SQLite and JSON artifacts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from invoice_agents.compatibility import run_live_contracts
from invoice_agents.config import (
    Settings,
    is_ui_loopback_host,
    normalize_ui_host,
    serialize_ui_origin,
)
from invoice_agents.db.cli import app as db_app
from invoice_agents.db.core import ensure_databases
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.hitl.service import record_human_decision
from invoice_agents.models import (
    CanonicalMapping,
    CaseResult,
    CaseStatus,
    HumanDecisionKind,
)
from invoice_agents.observability.audit import configure_logging, redact, sanitize_text
from invoice_agents.orchestration import (
    claim_resumable_case,
    process_batch,
    process_invoice,
    resume_case,
)

console = Console()
app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    help="Auditable AutoGen/Grok invoice processing.",
)
review_app = typer.Typer(no_args_is_help=True, help="Persisted human-review queue.")
case_app = typer.Typer(no_args_is_help=True, help="Inspect persisted case results.")
app.add_typer(db_app, name="db")
app.add_typer(review_app, name="review")
app.add_typer(case_app, name="case")


def _settings() -> Settings:
    settings = Settings()
    configure_logging(settings.log_level)
    return settings


def _safe_cli_text(value: object) -> str:
    return sanitize_text(str(value))


def _safe_json(value: object) -> str:
    return json.dumps(redact(value), ensure_ascii=False, sort_keys=True)


def _print_result(result: CaseResult) -> None:
    console.print(
        _safe_cli_text(
            f"case={result.case_id} status={result.status} stop_reason={result.stop_reason}"
        ),
        markup=False,
    )
    if result.final_decision:
        console.print(
            _safe_cli_text(
                f"decision={result.final_decision.decision} payment_eligible="
                f"{result.final_decision.payment_eligible}"
            ),
            markup=False,
        )
    if result.review_request:
        console.print(
            _safe_cli_text(
                f"review={result.review_request.review_id} "
                f"review_status={result.review_request.status}"
            ),
            markup=False,
        )
    if result.payment:
        console.print(
            _safe_cli_text(
                f"payment={result.payment.status} payment_id={result.payment.payment_id or '-'}"
            ),
            markup=False,
        )
    for error in result.errors:
        console.print(
            _safe_cli_text(f"error[{error.category}]={error.message}"),
            style="red",
            markup=False,
        )
    console.print(
        _safe_cli_text(f"full_result=artifacts/results/{result.case_id}.json"), markup=False
    )


def _exit_for(result: CaseResult) -> None:
    if result.status is CaseStatus.NEEDS_HUMAN:
        raise typer.Exit(2)
    if result.status in {CaseStatus.FAILED, CaseStatus.INCOMPLETE}:
        raise typer.Exit(1)


@app.callback()
def root_callback(
    ctx: typer.Context,
    invoice_path: Annotated[
        Path | None,
        typer.Option(
            "--invoice_path", "--invoice-path", help="README-compatible single source path"
        ),
    ] = None,
) -> None:
    """Support ``python main.py --invoice_path=...`` as well as named commands."""

    if ctx.invoked_subcommand is None and invoice_path is not None:
        result = asyncio.run(process_invoice(invoice_path, _settings()))
        _print_result(result)
        _exit_for(result)


@app.command("process")
def process_command(
    invoice_path: Annotated[Path, typer.Option("--invoice-path", exists=False)],
) -> None:
    """Process one invoice through a fresh AutoGen Swarm."""

    result = asyncio.run(process_invoice(invoice_path, _settings()))
    _print_result(result)
    _exit_for(result)


@app.command("batch")
def batch_command(
    invoice_dir: Annotated[Path, typer.Option("--invoice-dir", file_okay=False)] = Path(
        "data/invoices"
    ),
    concurrency: Annotated[int | None, typer.Option(min=1, max=8)] = None,
) -> None:
    """Process all supported files with per-case status and bounded concurrency."""

    if not invoice_dir.is_dir():
        console.print(
            "SOURCE_DIRECTORY_MISSING: source-repository demo corpus is unavailable; "
            "pass --invoice-dir PATH",
            style="red",
        )
        raise typer.Exit(1)
    paths = sorted(
        path
        for path in invoice_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".json", ".csv", ".xml", ".pdf"}
    )
    results = asyncio.run(process_batch(paths, _settings(), concurrency))
    table = Table("Case", "Source", "Status", "Stop reason", "Decision", "Payment")
    for result in results:
        table.add_row(
            _safe_cli_text(result.case_id),
            _safe_cli_text(result.source_id or "-"),
            _safe_cli_text(result.status),
            _safe_cli_text(result.stop_reason),
            _safe_cli_text(result.final_decision.decision) if result.final_decision else "-",
            _safe_cli_text(result.payment.status) if result.payment else "-",
        )
    console.print(table)
    failed = [
        result for result in results if result.status in {CaseStatus.FAILED, CaseStatus.INCOMPLETE}
    ]
    pending = [result for result in results if result.status is CaseStatus.NEEDS_HUMAN]
    console.print(
        f"total={len(results)} failed_or_incomplete={len(failed)} needs_human={len(pending)}"
    )
    if failed:
        raise typer.Exit(1)
    if pending:
        raise typer.Exit(2)


@case_app.command("status")
def case_status(case_id: str) -> None:
    """Show a persisted structured case result."""

    result = WorkflowStore(_settings().workflow_db).load_result(case_id)
    if result is None:
        console.print(
            _safe_cli_text(f"case {case_id} has no terminal result"),
            style="yellow",
            markup=False,
        )
        raise typer.Exit(1)
    console.print_json(_safe_json(result.model_dump(mode="json")))


@review_app.command("list")
def review_list(all_reviews: Annotated[bool, typer.Option("--all")] = False) -> None:
    """List pending reviews by default."""

    reviews = WorkflowStore(_settings().workflow_db).list_reviews(pending_only=not all_reviews)
    table = Table("Review", "Case", "Status", "Amount", "Recommendation", "Reasons")
    for review in reviews:
        amount = f"{review.amount.amount} {review.amount.currency}" if review.amount else "-"
        table.add_row(
            _safe_cli_text(review.review_id),
            _safe_cli_text(review.case_id),
            _safe_cli_text(review.status),
            _safe_cli_text(amount),
            _safe_cli_text(review.agent_recommendation),
            str(len(review.reasons)),
        )
    console.print(table)


@review_app.command("show")
def review_show(review_id: str) -> None:
    """Show the full review evidence package and exact questions."""

    review = WorkflowStore(_settings().workflow_db).load_review(review_id)
    console.print_json(_safe_json(review.model_dump(mode="json")))


def _parse_mapping(values: list[str]) -> list[CanonicalMapping]:
    mappings: list[CanonicalMapping] = []
    for value in values:
        if "=" not in value:
            raise typer.BadParameter("mapping must use raw item=SKU syntax")
        raw_item, sku = value.split("=", 1)
        mappings.append(
            CanonicalMapping(
                raw_item=raw_item.strip(),
                sku=sku.strip(),
                basis="human_decision",
            )
        )
    return mappings


@review_app.command("decide")
def review_decide(
    review_id: str,
    reviewer: Annotated[str, typer.Option(prompt=True)],
    decision: Annotated[HumanDecisionKind, typer.Option(prompt=True)],
    reason: Annotated[str, typer.Option(prompt=True)],
    mapping: Annotated[list[str] | None, typer.Option("--mapping")] = None,
    superseded_case_id: Annotated[str | None, typer.Option()] = None,
    address_blocker: Annotated[list[str] | None, typer.Option("--address-blocker")] = None,
) -> None:
    """Record an attributable decision; mappings use ``raw item=SKU``."""

    settings = _settings()
    try:
        review = record_human_decision(
            review_id,
            reviewer,
            decision,
            reason,
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=_parse_mapping(mapping or []),
            superseded_case_id=superseded_case_id,
            addressed_blocker_ids=address_blocker or [],
        )
    except InvoiceAgentsError as exc:
        console.print(sanitize_text(str(exc)), style="red", markup=False)
        raise typer.Exit(1) from exc
    console.print(
        _safe_cli_text(
            f"review={review.review_id} status={review.status} decision_recorded={decision}"
        ),
        markup=False,
    )


@review_app.command("resume")
def review_resume(case_id: str) -> None:
    """Resume a stopped team after a persisted human decision."""

    try:
        settings = _settings()
        claim = claim_resumable_case(case_id, settings)
        result = asyncio.run(resume_case(case_id, settings, claim=claim))
    except InvoiceAgentsError as exc:
        console.print(sanitize_text(str(exc)), style="red", markup=False)
        raise typer.Exit(1) from exc
    _print_result(result)
    _exit_for(result)


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@app.command("ui")
def ui_command(
    host: Annotated[str, typer.Option(help="Bind address; loopback only by default")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8787,
    allow_remote: Annotated[
        bool,
        typer.Option(
            "--allow-remote-i-understand",
            help="Required to bind beyond loopback; the console has no auth layer",
        ),
    ] = False,
    init_db: Annotated[
        bool,
        typer.Option(
            "--init-db/--no-init-db",
            help="Migrate, seed, and verify both SQLite databases before serving (idempotent)",
        ),
    ] = True,
) -> None:
    """Bring up the databases (idempotent) and serve the local web console."""

    try:
        bind_host = normalize_ui_host(host)
    except ValueError as exc:
        console.print(f"invalid UI bind host: {host!r}", style="red")
        raise typer.Exit(1) from exc
    is_loopback = is_ui_loopback_host(bind_host)
    if not is_loopback and not allow_remote:
        console.print(
            _safe_cli_text(
                f"refusing to bind {bind_host}: the console has no authentication and is "
                "localhost-only by design. Add auth before any remote exposure, or pass "
                "--allow-remote-i-understand to accept the risk explicitly."
            ),
            style="red",
            markup=False,
        )
        raise typer.Exit(1)
    try:
        settings = _settings()
    except ValidationError as exc:
        if any("ui_allowed_hosts" in error["loc"] for error in exc.errors()):
            console.print(
                "invalid UI allowed host configuration; use exact bare host names without "
                "wildcards, ports, or schemes",
                style="red",
            )
            raise typer.Exit(1) from exc
        if any("ui_session_secret" in error["loc"] for error in exc.errors()):
            console.print(
                "invalid INVOICE_UI_SESSION_SECRET; configure at least 32 random bytes",
                style="red",
            )
            raise typer.Exit(1) from exc
        raise
    if not is_loopback and not settings.ui_allowed_hosts:
        console.print(
            "remote binding also requires an explicit INVOICE_UI_ALLOWED_HOSTS JSON list "
            "of exact browser host names",
            style="red",
        )
        raise typer.Exit(1)
    if not is_loopback and settings.ui_session_secret is None:
        console.print(
            "remote binding requires INVOICE_UI_SESSION_SECRET so every worker and restart "
            "uses the same explicit random session key",
            style="red",
        )
        raise typer.Exit(1)
    configured_hosts = (bind_host,) if is_loopback else settings.ui_allowed_hosts
    configured_origins = tuple(
        serialize_ui_origin("http", allowed_host, port) for allowed_host in configured_hosts
    )
    try:
        import uvicorn

        from invoice_agents.ui.server import create_app
    except ImportError as exc:
        console.print(
            sanitize_text(
                f"the web console requires the 'ui' extra ({exc}); "
                "install it with: uv sync --extra ui"
            ),
            style="red",
            markup=False,
        )
        raise typer.Exit(1) from exc
    if init_db:
        try:
            applied = ensure_databases(settings)
        except InvoiceAgentsError as exc:
            console.print(
                sanitize_text(f"database setup failed [{exc.stop_reason}]: {exc.message}"),
                style="red",
                markup=False,
            )
            raise typer.Exit(1) from exc
        for kind, versions in applied.items():
            state = f"applied migrations {versions}" if versions else "already migrated"
            console.print(
                _safe_cli_text(f"{kind} database ready ({state}), verified"), markup=False
            )
    console.print(
        _safe_cli_text(
            f"Galatiq invoice console on {serialize_ui_origin('http', bind_host, port)} "
            "(Ctrl+C stops it)"
        ),
        markup=False,
    )
    uvicorn.run(
        create_app(
            settings,
            allowed_hosts=configured_hosts,
            allowed_origins=configured_origins,
        ),
        host=bind_host,
        port=port,
        log_level="warning",
    )


@app.command("contract")
def contract_command(
    live: Annotated[bool, typer.Option("--live", help="Acknowledge paid xAI API calls")] = False,
) -> None:
    """Run compatibility contracts; live checks are opt-in and never silently skipped."""

    if not live:
        console.print("live contracts NOT RUN; pass --live to call xAI", style="yellow")
        raise typer.Exit(2)
    checks = asyncio.run(run_live_contracts(_settings()))
    table = Table("Contract", "Result", "Evidence")
    for check in checks:
        table.add_row(
            _safe_cli_text(check.name),
            "PASS" if check.passed else "FAIL",
            _safe_cli_text(check.evidence),
        )
    console.print(table)
    if not all(check.passed for check in checks):
        raise typer.Exit(1)
