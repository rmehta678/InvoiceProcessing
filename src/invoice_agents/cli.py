"""Human-oriented CLI with full structured state retained in SQLite and JSON artifacts."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import traceback
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import uuid4

import typer
from click import ClickException
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from invoice_agents.compatibility import run_live_contracts, validated_live_contract_evidence
from invoice_agents.config import (
    Settings,
    is_ui_loopback_host,
    normalize_ui_host,
    serialize_ui_origin,
)
from invoice_agents.db.cli import app as db_app
from invoice_agents.db.core import ensure_databases
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
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
    process_invoice,
    resume_case,
)
from invoice_agents.ui.runs import RunRegistry

SUPPORTED_INVOICE_SUFFIXES = frozenset({".txt", ".json", ".csv", ".xml", ".pdf"})
OPERATIONAL_MESSAGE_MAX_CHARACTERS = 512
DEBUG_STACK_MAX_FRAMES = 8
DEBUG_STACK_MAX_CHARACTERS = 512
DEBUG_NAME_MAX_CHARACTERS = 128
DEBUG_STACK_SEPARATOR = " -> "
DEBUG_STACK_TRUNCATION_MARKER = "…[TRUNCATED]"
SANITIZATION_FAILURE_LINE = (
    "category=ORCHESTRATION stop_reason=SANITIZATION_FAILED "
    "message=credential sanitization failed closed"
)

_CLI_DEBUG: ContextVar[bool] = ContextVar("invoice_agents_cli_debug", default=False)
_ASSIGNMENT_KEY = re.compile(r"\b[A-Za-z_][A-Za-z0-9_-]*\s*=")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_])/(?:[^/\s]+/)*[^\s]*")
_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\(?:[^\\\s]+\\)*[^\s]*")
_SAFE_DEBUG_NAME_CHARACTER = re.compile(r"[^A-Za-z0-9_.<>-]")
_OPERATIONAL_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


def _sanitized_cli_text(value: str) -> str:
    """Sanitize text or expose that the sanitizer itself failed, never the input."""

    try:
        return sanitize_text(value)
    except Exception:
        error_console.print(SANITIZATION_FAILURE_LINE, markup=False, soft_wrap=True)
        raise typer.Exit(1) from None


def _safe_operational_message(value: str) -> str:
    """Return one bounded, redacted line without paths or nested output fields."""

    safe = _sanitized_cli_text(value)
    safe = _WINDOWS_PATH.sub("[PATH_REDACTED]", safe)
    safe = _POSIX_PATH.sub("[PATH_REDACTED]", safe)
    safe = _ASSIGNMENT_KEY.sub("", safe)
    rendered = " ".join(safe.split()) or "operation failed"
    if len(rendered) <= OPERATIONAL_MESSAGE_MAX_CHARACTERS:
        return rendered
    visible = OPERATIONAL_MESSAGE_MAX_CHARACTERS - len(DEBUG_STACK_TRUNCATION_MARKER)
    return f"{rendered[:visible]}{DEBUG_STACK_TRUNCATION_MARKER}"


def _safe_debug_name(value: str) -> str:
    safe = _SAFE_DEBUG_NAME_CHARACTER.sub("_", _sanitized_cli_text(value))
    return safe[:DEBUG_NAME_MAX_CHARACTERS]


def _debug_stack(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    names = [_safe_debug_name(frame.name) for frame in frames[-DEBUG_STACK_MAX_FRAMES:]]
    rendered = DEBUG_STACK_SEPARATOR.join(names)
    if len(rendered) <= DEBUG_STACK_MAX_CHARACTERS:
        return rendered
    visible = DEBUG_STACK_MAX_CHARACTERS - len(DEBUG_STACK_TRUNCATION_MARKER)
    return f"{rendered[:visible]}{DEBUG_STACK_TRUNCATION_MARKER}"


def _print_operational_error(exc: BaseException) -> NoReturn:
    """Print one safe operational failure and terminate with exit code 1."""

    if not isinstance(exc, Exception):
        raise exc
    if isinstance(exc, InvoiceAgentsError):
        contract_is_valid = (
            type(exc.category) is ErrorCategory
            and type(exc.stop_reason) is str
            and _OPERATIONAL_CODE.fullmatch(exc.stop_reason) is not None
            and type(exc.message) is str
        )
        if contract_is_valid:
            category = exc.category
            stop_reason = exc.stop_reason
            message = _safe_operational_message(exc.message)
            unexpected = False
        else:
            category = ErrorCategory.ORCHESTRATION
            stop_reason = "OPERATIONAL_ERROR_CONTRACT_INVALID"
            message = _safe_operational_message(
                "application error violated the operational error contract"
            )
            unexpected = True
    elif isinstance(exc, sqlite3.Error):
        category = ErrorCategory.DATABASE
        stop_reason = "DATABASE_OPERATION_FAILED"
        message = _safe_operational_message(str(exc))
        unexpected = False
    elif isinstance(exc, OSError):
        category = ErrorCategory.SOURCE
        stop_reason = "FILESYSTEM_OPERATION_FAILED"
        message = _safe_operational_message(str(exc))
        unexpected = False
    else:
        category = ErrorCategory.ORCHESTRATION
        stop_reason = "UNEXPECTED_ERROR"
        message = _safe_operational_message("unexpected application error")
        unexpected = True

    debug_lines: list[str] = []
    if unexpected and _CLI_DEBUG.get():
        debug_lines.extend(
            (
                f"debug_exception_type={_safe_debug_name(type(exc).__name__)}",
                f"debug_exception_message={_safe_operational_message(str(exc))}",
                f"debug_stack={_debug_stack(exc)}",
            )
        )
    error_console.print(
        f"category={category.value} stop_reason={stop_reason} message={message}",
        style="red",
        markup=False,
        soft_wrap=True,
    )
    for debug_line in debug_lines:
        error_console.print(
            debug_line,
            markup=False,
            soft_wrap=True,
        )
    raise typer.Exit(1) from None


class _OperationalBoundaryGroup(TyperGroup):
    """Apply one exception boundary to the callback and every mounted command."""

    def invoke(self, ctx: Any) -> Any:
        debug_token = _CLI_DEBUG.set(bool(ctx.params.get("debug", False)))
        try:
            try:
                return super().invoke(ctx)
            except (typer.Exit, ClickException):
                raise
            except Exception as exc:
                _print_operational_error(exc)
        finally:
            _CLI_DEBUG.reset(debug_token)


console = Console()
error_console = Console(stderr=True)
app = typer.Typer(
    cls=_OperationalBoundaryGroup,
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
    return _sanitized_cli_text(str(value))


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
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Print bounded exception type and stack-frame names for unexpected errors",
        ),
    ] = False,
) -> None:
    """Support ``python main.py --invoice_path=...`` as well as named commands."""

    del debug
    if ctx.invoked_subcommand is None and invoice_path is not None:
        source = _supported_invoice_path(invoice_path)
        result = asyncio.run(process_invoice(source, _settings()))
        _print_result(result)
        _exit_for(result)


@app.command("process")
def process_command(
    invoice_path: Annotated[Path, typer.Option("--invoice-path", exists=False)],
    force_reprocess: Annotated[bool, typer.Option("--force-reprocess")] = False,
) -> None:
    """Process one invoice through a fresh AutoGen Swarm."""

    source = _supported_invoice_path(invoice_path)
    result = asyncio.run(
        process_invoice(
            source,
            _settings(),
            force_reprocess=force_reprocess,
        )
    )
    _print_result(result)
    _exit_for(result)


def _supported_invoice_path(path: Path) -> Path:
    if not path.is_file():
        raise InvoiceAgentsError(
            ErrorCategory.SOURCE,
            "invoice source does not exist or is not a regular file",
            stop_reason="SOURCE_NOT_FOUND",
        )
    return path


def _supported_invoice_paths(directory: Path) -> list[Path]:
    """Return sorted supported regular files or fail before workflow admission."""

    if not directory.is_dir():
        raise InvoiceAgentsError(
            ErrorCategory.SOURCE,
            "source directory is unavailable; pass --invoice-dir PATH",
            stop_reason="SOURCE_DIRECTORY_MISSING",
        )
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_INVOICE_SUFFIXES
    )
    if not paths:
        raise InvoiceAgentsError(
            ErrorCategory.SOURCE,
            "source directory contains no supported invoice files",
            stop_reason="SOURCE_DIRECTORY_EMPTY",
        )
    return paths


@app.command("batch")
def batch_command(
    invoice_dir: Annotated[Path, typer.Option("--invoice-dir", file_okay=False)] = Path(
        "data/invoices"
    ),
    concurrency: Annotated[int | None, typer.Option(min=1, max=8)] = None,
) -> None:
    """Process all supported files with per-case status and bounded concurrency."""

    paths = _supported_invoice_paths(invoice_dir)
    settings = _settings()
    results = asyncio.run(_run_durable_cli_batch(paths, settings, concurrency))
    if not results or len(results) != len(paths):
        raise InvoiceAgentsError(
            ErrorCategory.ORCHESTRATION,
            "batch processing did not return exactly one result per admitted source",
            stop_reason="BATCH_RESULT_CARDINALITY_INVALID",
        )
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


async def _run_durable_cli_batch(
    paths: list[Path],
    settings: Settings,
    concurrency: int | None,
) -> list[CaseResult]:
    """Run one CLI batch through the shared durable admission implementation."""

    registry = RunRegistry(global_limit=settings.case_concurrency)
    batch = await registry.start_batch(
        paths,
        settings,
        concurrency,
        submission_id=f"submission_cli_batch_{uuid4().hex}",
    )
    task = batch.task
    if task is not None:
        await task
    persisted = WorkflowStore(settings).load_batch(batch.batch_id)
    if persisted is None:
        raise InvoiceAgentsError(
            ErrorCategory.DATABASE,
            "durable CLI batch disappeared after admission",
            stop_reason="PERSISTED_SUBMISSION_INVALID",
        ) from None
    results: list[CaseResult] = []
    for entry in persisted.entries:
        if entry.result is None:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "CLI batch reuses a source whose authoritative execution is still active",
                case_id=entry.case_id,
                stop_reason="SOURCE_RUN_ALREADY_ACTIVE",
            ) from None
        results.append(entry.result)
    return results


@case_app.command("status")
def case_status(case_id: str) -> None:
    """Show a persisted structured case result."""

    result = WorkflowStore(_settings().workflow_db).load_result(case_id)
    if result is None:
        raise InvoiceAgentsError(
            ErrorCategory.SOURCE,
            "case has no terminal result",
            stop_reason="CASE_NOT_FOUND",
        )
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
    if review is None:
        raise InvoiceAgentsError(
            ErrorCategory.SOURCE,
            "review does not exist",
            stop_reason="REVIEW_NOT_FOUND",
        )
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
    console.print(
        _safe_cli_text(
            f"review={review.review_id} status={review.status} decision_recorded={decision}"
        ),
        markup=False,
    )


@review_app.command("resume")
def review_resume(case_id: str) -> None:
    """Resume a stopped team after a persisted human decision."""

    settings = _settings()
    claim = claim_resumable_case(case_id, settings)
    result = asyncio.run(resume_case(case_id, settings, claim=claim))
    _print_result(result)
    _exit_for(result)


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
        raise InvoiceAgentsError(
            ErrorCategory.CONFIGURATION,
            "invalid UI bind host; use one exact bare host without a scheme or port",
            stop_reason="UI_BIND_HOST_INVALID",
        ) from exc
    is_loopback = is_ui_loopback_host(bind_host)
    if not is_loopback and not allow_remote:
        raise InvoiceAgentsError(
            ErrorCategory.CONFIGURATION,
            f"refusing to bind {bind_host}: the console has no authentication and is "
            "localhost-only by design; pass --allow-remote-i-understand only after "
            "explicitly accepting the risk",
            stop_reason="UI_REMOTE_BIND_REQUIRES_ACKNOWLEDGEMENT",
        )
    try:
        settings = _settings()
    except ValidationError as exc:
        if any("ui_allowed_hosts" in error["loc"] for error in exc.errors()):
            raise InvoiceAgentsError(
                ErrorCategory.CONFIGURATION,
                "invalid UI allowed host configuration; use exact bare host names without "
                "wildcards, ports, or schemes",
                stop_reason="UI_ALLOWED_HOSTS_INVALID",
            ) from exc
        if any("ui_session_secret" in error["loc"] for error in exc.errors()):
            raise InvoiceAgentsError(
                ErrorCategory.CONFIGURATION,
                "invalid INVOICE_UI_SESSION_SECRET; configure at least 32 random bytes",
                stop_reason="UI_SESSION_SECRET_INVALID",
            ) from exc
        raise
    if not is_loopback and not settings.ui_allowed_hosts:
        raise InvoiceAgentsError(
            ErrorCategory.CONFIGURATION,
            "remote binding also requires an explicit INVOICE_UI_ALLOWED_HOSTS JSON list "
            "of exact browser host names",
            stop_reason="UI_ALLOWED_HOSTS_REQUIRED",
        )
    if not is_loopback and settings.ui_session_secret is None:
        raise InvoiceAgentsError(
            ErrorCategory.CONFIGURATION,
            "remote binding requires INVOICE_UI_SESSION_SECRET so every worker and restart "
            "uses the same explicit random session key",
            stop_reason="UI_SESSION_SECRET_REQUIRED",
        )
    configured_hosts = (bind_host,) if is_loopback else settings.ui_allowed_hosts
    configured_origins = tuple(
        serialize_ui_origin("http", allowed_host, port) for allowed_host in configured_hosts
    )
    try:
        import uvicorn

        from invoice_agents.ui.server import create_app
    except ImportError as exc:
        raise InvoiceAgentsError(
            ErrorCategory.CONFIGURATION,
            f"the web console requires the ui extra ({exc}); install it with uv sync --extra ui",
            stop_reason="UI_DEPENDENCY_MISSING",
        ) from exc
    if init_db:
        applied = ensure_databases(settings)
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
    checks = validated_live_contract_evidence(asyncio.run(run_live_contracts(_settings())))
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
