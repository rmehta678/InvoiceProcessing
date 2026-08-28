"""Run the pipeline with scripted LLM responses -- no API key required.

The agents' reasoning is replayed from `tests/fixtures/golden_extractions.json`
rather than generated, so this is not a demonstration of Grok's extraction
quality. It is a demonstration that the orchestration, validation, policy
engine, and reporting work end to end, and it lets anyone see the system run
before setting up a credential.

    python scripts/demo.py                                    # all invoices
    python scripts/demo.py --invoice data/invoices/invoice_1003.txt
    python scripts/demo.py --report out/demo.html

For real agent reasoning, set XAI_API_KEY and use main.py.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from fake_llm import ScriptedGrokClient  # noqa: E402
from init_db import initialise  # noqa: E402

from invoice_flow.config import INVOICE_DIR, Settings  # noqa: E402
from invoice_flow.graph import Dependencies, process_invoice  # noqa: E402
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


def run(path: Path, db: Path, tracer: Tracer):
    repo = InventoryRepository(db)
    try:
        deps = Dependencies(
            settings=Settings.from_env(db_path=db),
            client=ScriptedGrokClient(),
            repo=repo,
            tracer=tracer,
        )
        return process_invoice(str(path), deps)
    finally:
        repo.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invoice", type=Path, default=None, help="Process one invoice in detail.")
    parser.add_argument("--report", type=Path, default=None, help="Write an HTML report here.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    console = make_console()
    console.print()
    console.print("[bold]Invoice pipeline demo[/bold] [dim](scripted responses, no API key)[/dim]")

    workdir = Path(tempfile.mkdtemp(prefix="invoice-demo-"))

    if args.invoice:
        db = workdir / "inventory.db"
        with contextlib.redirect_stdout(io.StringIO()):
            initialise(db)
        console.print()
        tracer = Tracer(run_id=new_run_id(), echo=progress_echo(console, verbose=args.verbose))
        state = run(args.invoice, db, tracer)
        render_summary(state, console, tracer.usage.as_dict())
        if args.report:
            written = html_report.write_report(state, args.report, tracer.usage.as_dict())
            console.print(f"\n[dim]report: {written}[/dim]")
        return 0

    # Sweep every sample invoice against ONE database, so the ledger carries
    # across the way it would in production. That is what makes the duplicate
    # controls visible here: 1011 arrives as PDF then TXT and is paid once,
    # and 1004_revised conflicts with 1004 and escalates instead of double-paying.
    db = workdir / "sweep.db"
    with contextlib.redirect_stdout(io.StringIO()):
        initialise(db)

    report = BatchReport(console)
    report.start()
    for path in discover_invoices(INVOICE_DIR):
        report.row(path.name, run(path, db, Tracer(run_id=new_run_id())))
    report.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
