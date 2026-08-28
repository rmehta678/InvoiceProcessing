"""Command-line entry point for the invoice processing pipeline.

    python main.py --invoice_path=data/invoices/invoice_1001.txt

Every run writes a trace, a structured result, and an HTML report to
``runs/<run_id>/`` so the decision can be audited after the fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from invoice_flow.config import DB_PATH, RUNS_DIR, Settings  # noqa: E402
from invoice_flow.graph import Dependencies, process_invoice  # noqa: E402
from invoice_flow.llm.base import LLMError  # noqa: E402
from invoice_flow.llm.router import AllProvidersUnavailable, build_client  # noqa: E402
from invoice_flow.models import Decision  # noqa: E402
from invoice_flow.observability.trace import Tracer, new_run_id  # noqa: E402
from invoice_flow.reporting import html as html_report  # noqa: E402
from invoice_flow.reporting.console import (  # noqa: E402
    BatchReport,
    make_console,
    progress_echo,
    render_summary,
)
from invoice_flow.tools.inventory import InventoryRepository  # noqa: E402
from invoice_flow.tools.loaders import discover_invoices  # noqa: E402

EXIT_OK = 0
EXIT_NOT_APPROVED = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Process a supplier invoice through the multi-agent pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py --invoice_path=data/invoices/invoice_1001.txt\n"
            "  python main.py --invoice_path=data/invoices/invoice_1014.xml --verbose\n"
            "  python main.py --invoice_path=data/invoices/invoice_1003.txt --json\n"
            "  python main.py --batch=data/invoices/\n"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--invoice_path",
        help="Path to the invoice file (.txt, .json, .csv, .xml, or .pdf).",
    )
    source.add_argument(
        "--batch",
        type=Path,
        help="Process every invoice in a directory and report straight-through rate.",
    )
    parser.add_argument("--model", default=None, help="Override the Grok model to use.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Inventory database path.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the HTML report here instead of the run directory.",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip HTML report generation."
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Print the result as JSON only."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show tool calls as the agents make them."
    )
    return parser


def run_invoice(
    invoice_path: Path,
    settings: Settings,
    repo: InventoryRepository,
    args: Any,
    console: Any,
    echo: bool,
    report_target: Path | None = None,
) -> dict[str, Any]:
    """Process one invoice, writing its trace, result, and report.

    Returns the run state alongside the paths it produced, so both the single
    and batch callers can report on it without re-deriving anything.
    """
    run_id = new_run_id()
    run_dir = RUNS_DIR / run_id
    tracer = Tracer(
        run_id=run_id,
        run_dir=run_dir,
        echo=progress_echo(console, verbose=args.verbose) if echo else None,
    )

    try:
        client = build_client(settings, tracer=tracer)
        if echo:
            console.print(f"[dim]providers: {client.describe_chain()}[/dim]")
        deps = Dependencies(settings=settings, client=client, repo=repo, tracer=tracer)
        state = process_invoice(str(invoice_path), deps)

        usage = tracer.usage.as_dict()
        tracer.write_result(state)

        report_path: Path | None = None
        if not args.no_report:
            target = report_target or (run_dir / "report.html")
            try:
                report_path = html_report.write_report(state, target, usage)
            except Exception as exc:  # noqa: BLE001 - a report failure must not fail the run
                tracer.emit("report.failed", error=str(exc))
                if echo:
                    console.print(f"[yellow]Could not write HTML report: {exc}[/yellow]")
    finally:
        tracer.close()

    return {"state": state, "usage": usage, "run_dir": run_dir, "report": report_path}


def run_single(args: Any, settings: Settings, repo: InventoryRepository, console: Any) -> int:
    invoice_path = Path(args.invoice_path)
    if not invoice_path.exists():
        console.print(f"[red]Invoice file not found:[/red] {invoice_path}")
        return EXIT_ERROR

    if not args.as_json:
        console.print()
        console.print(f"[bold]Processing[/bold] {invoice_path.name}")

    outcome = run_invoice(
        invoice_path, settings, repo, args, console,
        echo=not args.as_json,
        report_target=args.report,
    )
    state = outcome["state"]
    run_dir = outcome["run_dir"]

    if args.as_json:
        payload = state.model_dump(mode="json")
        payload["usage"] = outcome["usage"]
        payload["run_dir"] = str(run_dir)
        if outcome["report"]:
            payload["report"] = str(outcome["report"])
        print(json.dumps(payload, indent=2, default=str))
    else:
        render_summary(state, console, outcome["usage"])
        console.print()
        console.print(f"[dim]trace:  {run_dir / 'trace.jsonl'}[/dim]")
        console.print(f"[dim]result: {run_dir / 'result.json'}[/dim]")
        if outcome["report"]:
            console.print(f"[dim]report: {outcome['report']}[/dim]")
        console.print()

    if state.error:
        return EXIT_ERROR
    if state.approval is None or state.approval.decision is not Decision.APPROVED:
        return EXIT_NOT_APPROVED
    return EXIT_OK


def run_batch(args: Any, settings: Settings, repo: InventoryRepository, console: Any) -> int:
    """Sweep a directory, reporting the straight-through rate over the whole set.

    One repository across the batch, deliberately: the ledger is what makes a
    duplicate invoice detectable, and isolating each invoice would hide exactly
    the control this stage exists to run.
    """
    try:
        paths = discover_invoices(args.batch)
    except NotADirectoryError as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_ERROR

    if not paths:
        console.print(f"[yellow]No loadable invoices found in {args.batch}[/yellow]")
        return EXIT_ERROR

    if not args.as_json:
        console.print()
        console.print(f"[bold]Processing[/bold] {len(paths)} invoices from {args.batch}")

    report = BatchReport(console)
    report.start()

    records: list[dict[str, Any]] = []
    for path in paths:
        outcome = run_invoice(
            path, settings, repo, args, console, echo=args.verbose
        )
        report.row(path.name, outcome["state"])
        records.append(
            {
                "invoice": path.name,
                "run_dir": str(outcome["run_dir"]),
                "decision": (
                    outcome["state"].approval.decision.value
                    if outcome["state"].approval
                    else None
                ),
                "payment_status": (
                    outcome["state"].payment.status if outcome["state"].payment else None
                ),
                "total": outcome["state"].draft.total if outcome["state"].draft else None,
                "error": outcome["state"].error,
                "usage": outcome["usage"],
            }
        )

    summary = report.finish()

    if args.as_json:
        print(json.dumps({"summary": summary, "invoices": records}, indent=2, default=str))

    if summary["failed"]:
        return EXIT_ERROR
    if summary["rejected"] or summary["escalated"]:
        return EXIT_NOT_APPROVED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = make_console(quiet=args.as_json)

    if args.batch and args.report:
        console.print(
            "[red]--report names a single file and cannot be combined with --batch.[/red] "
            "Batch runs write a report into each invoice's own run directory."
        )
        return EXIT_ERROR

    settings = Settings.from_env(model=args.model, db_path=args.db)

    try:
        repo = InventoryRepository(settings.db_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_ERROR

    try:
        if args.batch:
            return run_batch(args, settings, repo, console)
        return run_single(args, settings, repo, console)
    except AllProvidersUnavailable as exc:
        console.print(f"\n[red]{exc}[/red]")
        return EXIT_ERROR
    except LLMError as exc:
        console.print(f"\n[red]LLM error:[/red] {exc}")
        return EXIT_ERROR
    finally:
        repo.close()


if __name__ == "__main__":
    raise SystemExit(main())
