"""Tests for the browser front end in `scripts/serve.py`.

Two things are worth guarding here. The command whitelist is a trust boundary:
it is the only thing standing between a POST body and a subprocess. And the
page is assembled by string substitution in Python, which means a Python escape
sequence can silently rewrite the JavaScript -- a `\\n` written into a JS string
literal becomes a real line break, the literal never closes, and the whole
`<script>` fails to parse, disabling every button at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from serve import allowed_invoices, build_command, render_page  # noqa: E402


def script_body(page: str) -> str:
    return page.split("<script>")[1].split("</script>")[0]


# --------------------------------------------------------------------------
# The page has to actually parse
# --------------------------------------------------------------------------


def test_javascript_string_literals_survive_python_substitution() -> None:
    """No JS string literal may be split across a line.

    An unbalanced quote count on a line means a literal ran off the end of it,
    which is a syntax error for the entire script block. Comment lines are
    skipped so ordinary apostrophes in prose do not trip the check.
    """
    offenders = [
        (number, line)
        for number, line in enumerate(script_body(render_page()).splitlines(), start=1)
        if not line.strip().startswith("//") and line.count("'") % 2
    ]
    assert not offenders, f"unterminated JS string literal(s): {offenders}"


def test_every_action_in_the_markup_is_a_command_the_server_accepts() -> None:
    """A button wired to an action the server rejects is a dead button."""
    page = render_page()
    actions = {
        segment.split('"')[0]
        for segment in page.split('data-action="')[1:]
    }
    assert actions, "no action buttons found in the page"

    invoice = next(iter(allowed_invoices()))
    for action in actions:
        build_command(action, invoice)  # raises if the server would refuse it


def test_page_offers_every_loadable_invoice() -> None:
    page = render_page()
    for name in allowed_invoices():
        assert f'value="{name}"' in page


# --------------------------------------------------------------------------
# The command whitelist
# --------------------------------------------------------------------------


def test_commands_match_the_documented_cli() -> None:
    assert build_command("demo_all", None)[1:] == ["scripts/demo.py"]
    assert build_command("reset_db", None)[1:] == ["scripts/init_db.py", "--reset"]
    assert build_command("process", "invoice_1001.txt")[1:] == [
        "main.py",
        "--invoice_path=data/invoices/invoice_1001.txt",
    ]
    assert build_command("demo_one", "invoice_1003.txt")[1:] == [
        "scripts/demo.py",
        "--invoice",
        "data/invoices/invoice_1003.txt",
        "--report",
        "out/report.html",
    ]


@pytest.mark.parametrize(
    "invoice",
    [
        "../../../etc/passwd",
        "../main.py",
        "main.py",
        "data/invoices/invoice_1001.txt",  # a path, not a name
        "invoice_1001.txt; rm -rf /",
        "",
        None,
    ],
)
def test_only_a_known_invoice_name_reaches_the_command_line(invoice: str | None) -> None:
    """The browser sends a name that is looked up; nothing it sends is a path."""
    with pytest.raises(ValueError, match="Not a known invoice"):
        build_command("process", invoice)


@pytest.mark.parametrize("action", ["", "rm -rf /", "PROCESS", "eval", None])
def test_unknown_actions_are_refused(action: str | None) -> None:
    with pytest.raises(ValueError, match="Unknown action"):
        build_command(action, "invoice_1001.txt")
