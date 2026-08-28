"""Rich terminal output.

An AP clerk should be able to read one screen and know what happened and why.
The decision and its single decisive reason come first; supporting detail
follows in order of how much it matters.
"""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..models import ITEM_STATUS_LABEL, Decision, RunState, Severity, money

DECISION_STYLE = {
    Decision.APPROVED: ("green", "APPROVED"),
    Decision.REJECTED: ("red", "REJECTED"),
    Decision.ESCALATED: ("yellow", "ESCALATED - HUMAN REVIEW"),
}

SEVERITY_STYLE = {
    Severity.CRITICAL: "bold red",
    Severity.WARNING: "yellow",
    Severity.INFO: "dim",
}

# Progress lines shown as the graph advances, keyed by trace event.
STAGE_LABELS = {
    "load.start": "Loading document",
    "extract.start": "Extracting structured data",
    "validate.start": "Validating against inventory",
    "vp_decide.start": "VP reviewing",
    "audit_critique.start": "Internal audit challenging the decision",
    "settle.start": "Settling",
}


def make_console(quiet: bool = False) -> Console:
    """Build the console, making it safe against un-encodable characters.

    Windows terminals default to cp1252, which cannot encode most non-ASCII
    text. Without this, printing a vendor name with an accent -- or a provider
    error containing an emoji -- raises `UnicodeEncodeError` from deep inside
    the rendering stack, turning a cosmetic problem into a crash that loses the
    actual message. Degrading an unprintable glyph is always better than
    dropping the line that contains it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Already detached or not reconfigurable; Rich still renders,
                # and the `errors` fallback below covers the common case.
                pass

    return Console(quiet=quiet, highlight=False, safe_box=True)


def progress_echo(console: Console, verbose: bool = False) -> Any:
    """Build a tracer echo callback that narrates the run."""

    def echo(event_type: str, event: dict[str, Any]) -> None:
        label = STAGE_LABELS.get(event_type)
        if label:
            suffix = ""
            if event_type == "extract.start" and event.get("round", 0) > 0:
                suffix = f" (repair attempt {event['round']})"
            if event_type == "vp_decide.start" and event.get("round", 0) > 0:
                suffix = f" (revision {event['round']})"
            console.print(f"  [dim]>[/dim] {label}{suffix}[dim]...[/dim]")
            return

        if event_type == "ingestion.attempt" and event.get("issue_count"):
            console.print(
                f"    [yellow]![/yellow] extraction round {event['round']}: "
                f"{event['issue_count']} inconsistency(ies) found, re-reading"
            )
        elif event_type == "ingestion.unresolved":
            console.print(
                "    [yellow]![/yellow] discrepancies persist across reads - "
                "attributing them to the document, not the extractor"
            )
        elif event_type == "approval.critique" and not event.get("agrees"):
            console.print("    [yellow]![/yellow] audit objected; VP revising")
        elif event_type == "payment.duplicate_blocked":
            console.print("    [red]![/red] duplicate payment blocked")
        elif verbose and event_type == "tool.call":
            console.print(f"    [dim]tool: {event['tool']}({event.get('arguments')})[/dim]")

    return echo


DECISION_COLOUR = {
    Decision.APPROVED: "green",
    Decision.REJECTED: "red",
    Decision.ESCALATED: "yellow",
}


class BatchReport:
    """The sweep table: one row per invoice, then the totals.

    Rows print as each invoice finishes rather than all at the end. A live
    batch against a real model takes minutes, and silence reads as a hang.

    Shared by `main.py --batch` and `scripts/demo.py` so the two cannot drift
    into describing the same run differently.
    """

    HEADER = f"{'INVOICE':<28} {'DECISION':<11} {'PAYMENT':<10} {'TOTAL':>12}  FINDINGS"

    def __init__(self, console: Console) -> None:
        self.console = console
        self.tally = {decision: 0 for decision in Decision}
        self.blocked = 0.0
        self.errors = 0

    def start(self) -> None:
        self.console.print()
        self.console.print(f"[dim]{self.HEADER}[/dim]")
        self.console.print("[dim]" + "-" * len(self.HEADER) + "[/dim]")

    def row(self, name: str, state: RunState) -> None:
        if state.error or state.approval is None:
            self.errors += 1
            self.console.print(f"{name:<28} [red]ERROR[/red]  {state.error}")
            return

        decision = state.approval.decision
        self.tally[decision] += 1
        total = state.draft.total if state.draft else None
        if decision is not Decision.APPROVED and total:
            self.blocked += total

        critical = len(state.validation.critical) if state.validation else 0
        warnings = len(state.validation.warnings) if state.validation else 0
        flags = []
        if critical:
            flags.append(f"[red]{critical} crit[/red]")
        if warnings:
            flags.append(f"[yellow]{warnings} warn[/yellow]")

        if total is None:
            amount = "-"
        elif total < 0:
            amount = f"-${abs(total):,.2f}"
        else:
            amount = f"${total:,.2f}"

        self.console.print(
            f"{name:<28} "
            f"[{DECISION_COLOUR[decision]}]{decision.value:<10}[/{DECISION_COLOUR[decision]}] "
            f"{(state.payment.status if state.payment else '-'):<10} "
            f"{amount:>12}  " + " ".join(flags),
            overflow="ignore",
            crop=False,
        )

    def finish(self) -> dict[str, Any]:
        """Print the totals and return them for JSON output."""
        decided = sum(self.tally.values())
        self.console.print()
        if decided:
            # Straight-through rate: the share needing no human at all. This is
            # the number the business case rests on, so it goes on screen.
            stp = self.tally[Decision.APPROVED] / decided
            self.console.print(
                f"[green]{self.tally[Decision.APPROVED]} approved[/green]  "
                f"[red]{self.tally[Decision.REJECTED]} rejected[/red]  "
                f"[yellow]{self.tally[Decision.ESCALATED]} escalated[/yellow]  "
                + (f"[red]{self.errors} failed[/red]  " if self.errors else "")
                + f"[dim]({stp:.0%} straight-through)[/dim]"
            )
            self.console.print(
                f"[dim]${self.blocked:,.2f} of payments held back for review or rejection.[/dim]"
            )
        elif self.errors:
            self.console.print(f"[red]{self.errors} invoice(s) failed and none completed.[/red]")
        self.console.print()

        return {
            "invoices": decided + self.errors,
            "approved": self.tally[Decision.APPROVED],
            "rejected": self.tally[Decision.REJECTED],
            "escalated": self.tally[Decision.ESCALATED],
            "failed": self.errors,
            "straight_through_rate": round(self.tally[Decision.APPROVED] / decided, 4)
            if decided
            else 0.0,
            "amount_held_back": round(self.blocked, 2),
        }


def render_summary(state: RunState, console: Console, usage: dict[str, int] | None = None) -> None:
    """Print the full result of one invoice run."""
    if state.error:
        console.print(Panel(f"[red]{state.error}[/red]", title="Run failed", border_style="red"))
        return

    draft = state.draft
    approval = state.approval
    report = state.validation

    console.print()

    # -- decision card -----------------------------------------------------
    if approval is not None:
        colour, label = DECISION_STYLE[approval.decision]
        header = Text(label, style=f"bold {colour}")
        body: list[Any] = [header, Text()]
        body.append(Text(approval.rationale))

        if approval.key_factors:
            body.append(Text())
            body.append(Text("Key factors:", style="bold"))
            for factor in approval.key_factors:
                body.append(Text(f"  - {factor}"))

        if approval.conditions:
            body.append(Text())
            body.append(Text("Conditions:", style="bold"))
            for condition in approval.conditions:
                body.append(Text(f"  - {condition}"))

        title = (
            f"{draft.invoice_number or 'unnumbered'} - {draft.vendor_name or 'unknown vendor'}"
            if draft
            else "Decision"
        )
        subtitle = f"scrutiny: {approval.scrutiny_level}"
        console.print(
            Panel(Group(*body), title=title, subtitle=subtitle, border_style=colour, padding=(1, 2))
        )

    # -- extracted data ----------------------------------------------------
    if draft is not None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("Invoice", draft.invoice_number or "[red]missing[/red]")
        table.add_row("Vendor", draft.vendor_name or "[red]missing[/red]")
        table.add_row("Total", money(draft.total, draft.currency))
        table.add_row(
            "Due date",
            (
                draft.due_date.isoformat()
                if draft.due_date
                else f"[yellow]{draft.due_date_raw or 'missing'}[/yellow]"
            ),
        )
        table.add_row("Terms", draft.payment_terms or "-")
        table.add_row("Confidence", f"{draft.extraction_confidence:.0%}")
        console.print(Rule("Extracted", style="dim"))
        console.print(table)

    # -- line items and stock ---------------------------------------------
    if report is not None and report.item_checks:
        console.print(Rule("Inventory check", style="dim"))
        items = Table(box=None, padding=(0, 2))
        items.add_column("Item")
        items.add_column("Billed", justify="right")
        items.add_column("In stock", justify="right")
        items.add_column("Status")
        for check in report.item_checks:
            status = ITEM_STATUS_LABEL.get(check.status, check.status)
            status_colour = "green" if check.status == "ok" else "red"
            items.add_row(
                check.invoice_name,
                f"{check.quantity_requested:g}",
                "-" if check.stock_available is None else str(check.stock_available),
                f"[{status_colour}]{status}[/{status_colour}]",
            )
        console.print(items)

    # -- findings ----------------------------------------------------------
    if report is not None and report.findings:
        console.print(Rule("Findings", style="dim"))
        for finding in report.findings:
            style = SEVERITY_STYLE[finding.severity]
            marker = {
                Severity.CRITICAL: "x",
                Severity.WARNING: "!",
                Severity.INFO: "-",
            }[finding.severity]
            console.print(f"  [{style}]{marker}[/{style}] {finding.message}")

    if report is not None and report.agent_summary:
        console.print(Rule("Validation agent", style="dim"))
        console.print(Panel(report.agent_summary, border_style="dim", padding=(0, 2)))

    # -- reflection --------------------------------------------------------
    if approval is not None and approval.rounds:
        console.print(Rule("Approval reasoning", style="dim"))
        for round_ in approval.rounds:
            console.print(
                f"  [bold]Round {round_.round_index + 1}[/bold] - "
                f"VP proposed [bold]{round_.draft.decision.value}[/bold]"
            )
            if round_.overridden_from is not None:
                console.print(
                    f"    [magenta]policy override:[/magenta] model proposed "
                    f"{round_.overridden_from.value}, not permitted for this invoice"
                )
            if round_.critique is None:
                console.print("    [dim]audit review unavailable[/dim]")
                continue
            if round_.critique.agrees:
                console.print("    [green]audit concurred[/green]")
            else:
                console.print("    [yellow]audit objected:[/yellow]")
                for objection in round_.critique.objections:
                    console.print(f"      - {objection}")
                if round_.critique.suggested_decision:
                    console.print(
                        f"      [dim]audit suggested "
                        f"{round_.critique.suggested_decision.value}[/dim]"
                    )
            if round_.revised:
                console.print("    [cyan]VP revised the decision[/cyan]")

    if approval is not None and approval.policy_reasons:
        console.print(Rule("Policy engine", style="dim"))
        for reason in approval.policy_reasons:
            console.print(f"  [dim]-[/dim] {reason}")

    # -- payment -----------------------------------------------------------
    if state.payment is not None:
        console.print(Rule("Payment", style="dim"))
        style = {
            "success": "green",
            "skipped": "dim",
            "duplicate": "red",
            "failed": "red",
        }.get(state.payment.status, "white")
        console.print(f"  [{style}]{state.payment.status.upper()}[/{style}] {state.payment.message}")
        if state.payment.reference:
            console.print(f"  [dim]reference: {state.payment.reference}[/dim]")

    # -- run stats ---------------------------------------------------------
    parts = []
    if state.duration_seconds is not None:
        seconds = state.duration_seconds
        # A scripted run finishes in milliseconds; "0.0s" reads as a broken
        # timer rather than a fast one.
        parts.append(f"{seconds:.2f}s" if seconds < 1 else f"{seconds:.1f}s")
    # Zero calls means the responses were scripted, not that the model was
    # silent. Saying "0 LLM calls" invites the reader to diagnose a fault.
    if usage and usage.get("calls"):
        parts.append(f"{usage['calls']} LLM calls")
        if usage.get("total_tokens"):
            parts.append(f"{usage['total_tokens']:,} tokens")
    if parts:
        console.print()
        console.print(f"[dim]{' | '.join(parts)}[/dim]")
