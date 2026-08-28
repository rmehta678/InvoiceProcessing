"""Reporting tests: the HTML report must be self-contained and safe."""

from __future__ import annotations

import re
from pathlib import Path

from fake_llm import ScriptedGrokClient
from test_pipeline import run_one

from invoice_flow.reporting import html as html_report


def render(filename: str, invoice_dir: Path, temp_db: Path) -> str:
    state = run_one(invoice_dir / filename, temp_db)
    return html_report.render_report(state, usage={"calls": 4, "total_tokens": 1234})


def test_report_is_self_contained(invoice_dir: Path, temp_db: Path) -> None:
    """A strict no-network rule: the report must open correctly offline."""
    output = render("invoice_1001.txt", invoice_dir, temp_db)
    assert "<style>" in output
    for pattern in (r"src=[\"']https?://", r"href=[\"']https?://", r"@import"):
        assert not re.search(pattern, output), f"external reference found: {pattern}"


def test_report_supports_both_colour_schemes(invoice_dir: Path, temp_db: Path) -> None:
    output = render("invoice_1001.txt", invoice_dir, temp_db)
    assert "prefers-color-scheme: dark" in output
    assert ":root {" in output


def test_report_shows_decision_and_findings(invoice_dir: Path, temp_db: Path) -> None:
    output = render("invoice_1003.txt", invoice_dir, temp_db)
    assert "verdict rejected" in output
    assert "FakeItem" in output
    assert "ITEM_OUT_OF_STOCK" in output
    assert "URGENCY_PRESSURE" in output


def test_report_shows_policy_override(invoice_dir: Path, temp_db: Path) -> None:
    """The guard rail firing must be visible, not just its outcome."""
    output = render("invoice_1003.txt", invoice_dir, temp_db)
    assert "Policy override" in output
    assert "APPROVED" in output  # what the model wanted


def test_report_explains_line_item_aggregation(invoice_dir: Path, temp_db: Path) -> None:
    output = render("invoice_1013.json", invoice_dir, temp_db)
    assert "aggregated into" in output
    assert "8 line items" in output


def test_vendor_text_cannot_inject_markup(tmp_path: Path, temp_db: Path) -> None:
    """Invoice content is untrusted input; autoescaping must hold."""
    hostile = tmp_path / "invoice_evil.json"
    hostile.write_text(
        '{"invoice_number": "INV-1", "vendor": {"name": "<script>alert(1)</script>"},'
        ' "date": "2026-01-01", "due_date": "2026-02-01",'
        ' "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": 1.0}],'
        ' "subtotal": 1.0, "total": 1.0, "currency": "USD"}',
        encoding="utf-8",
    )

    client = ScriptedGrokClient()
    client.extractions["invoice_evil.json"] = {
        "invoice_number": "INV-1",
        "vendor_name": "<script>alert(1)</script>",
        "invoice_date_raw": "2026-01-01",
        "due_date_raw": "2026-02-01",
        "line_items": [{"name": "WidgetA", "quantity": 1, "unit_price": 1.0}],
        "subtotal": 1.0,
        "total": 1.0,
        "currency": "USD",
    }

    state = run_one(hostile, temp_db, client=client)
    output = html_report.render_report(state)
    assert "<script>alert(1)</script>" not in output
    assert "&lt;script&gt;" in output


def test_report_writes_to_disk(invoice_dir: Path, temp_db: Path, tmp_path: Path) -> None:
    state = run_one(invoice_dir / "invoice_1015.csv", temp_db)
    target = tmp_path / "nested" / "report.html"
    written = html_report.write_report(state, target)
    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_batch_report_totals_the_sweep(invoice_dir: Path, temp_db: Path) -> None:
    """The straight-through rate is the number the business case rests on."""
    from invoice_flow.reporting.console import BatchReport, make_console

    report = BatchReport(make_console(quiet=True))
    report.start()
    for name in ("invoice_1001.txt", "invoice_1016.json", "invoice_1014.xml"):
        report.row(name, run_one(invoice_dir / name, temp_db))
    summary = report.finish()

    assert summary["invoices"] == 3
    assert (summary["approved"], summary["rejected"], summary["escalated"]) == (1, 1, 1)
    assert summary["failed"] == 0
    assert summary["straight_through_rate"] == round(1 / 3, 4)
    # Only the invoices that were not approved: $3,233.00 + $4,125.00.
    assert summary["amount_held_back"] == 7358.00


def test_batch_report_counts_a_failed_invoice_without_skewing_the_rate(
    tmp_path: Path, temp_db: Path, invoice_dir: Path
) -> None:
    """A document that cannot be read is a failure, not a rejection."""
    from invoice_flow.reporting.console import BatchReport, make_console

    bad = tmp_path / "invoice.docx"
    bad.write_text("not an invoice", encoding="utf-8")

    report = BatchReport(make_console(quiet=True))
    report.start()
    report.row("invoice_1001.txt", run_one(invoice_dir / "invoice_1001.txt", temp_db))
    report.row(bad.name, run_one(bad, temp_db))
    summary = report.finish()

    assert summary["failed"] == 1
    assert summary["invoices"] == 2
    # One invoice reached a decision and it was approved, so the rate is 100%:
    # a file the loader could not read never entered the decision population.
    assert summary["straight_through_rate"] == 1.0


def test_failed_run_still_renders(tmp_path: Path, temp_db: Path) -> None:
    """A crashed run must produce a report explaining the failure."""
    bad = tmp_path / "invoice.docx"
    bad.write_text("not an invoice", encoding="utf-8")
    state = run_one(bad, temp_db)
    output = html_report.render_report(state)
    assert "Run failed" in output
    assert "UnsupportedFormatError" in output
