"""CLI failures that must remain clear outside the source checkout."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from invoice_agents import cli
from invoice_agents.cli import app
from invoice_agents.models import CaseResult, CaseStatus, ErrorRecord

runner = CliRunner()


def test_batch_without_demo_corpus_reports_source_directory_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel must not mistake its absent source-only demo corpus for an empty batch."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["batch"])

    assert result.exit_code == 1
    assert "SOURCE_DIRECTORY_MISSING" in result.output
    assert "--invoice-dir" in result.output


def test_result_operational_output_sanitizes_error_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "plainCliCredential"
    output = io.StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=160),
    )
    now = datetime.now(UTC)
    result = CaseResult(
        case_id="case_cli_sensitive",
        source_id="src_cli_sensitive",
        status=CaseStatus.FAILED,
        stop_reason="PROVIDER_REQUEST_FAILED",
        errors=[
            ErrorRecord(
                category="PROVIDER",
                message=f"provider exception api_key={secret}",
                case_id="case_cli_sensitive",
                stop_reason="PROVIDER_REQUEST_FAILED",
            )
        ],
        started_at=now,
        finished_at=now,
    )

    cli._print_result(result)

    rendered = output.getvalue()
    assert "[REDACTED]" in rendered
    assert secret not in rendered


def test_case_status_json_sanitizes_legacy_error_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "plainLegacyCliCredential"
    output = io.StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=160),
    )
    now = datetime.now(UTC)
    result = CaseResult(
        case_id="case_cli_legacy",
        source_id="src_cli_legacy",
        status=CaseStatus.FAILED,
        stop_reason="PROVIDER_REQUEST_FAILED",
        errors=[
            ErrorRecord(
                category="PROVIDER",
                message=f"provider exception Cookie: session={secret}",
                case_id="case_cli_legacy",
                stop_reason="PROVIDER_REQUEST_FAILED",
            )
        ],
        started_at=now,
        finished_at=now,
    )

    class _Store:
        def __init__(self, _path: Path) -> None:
            pass

        def load_result(self, _case_id: str) -> CaseResult:
            return result

    monkeypatch.setattr(cli, "WorkflowStore", _Store)
    monkeypatch.setattr(cli, "_settings", lambda: type("S", (), {"workflow_db": Path("x")})())

    cli.case_status("case_cli_legacy")

    rendered = output.getvalue()
    assert "[REDACTED]" in rendered
    assert secret not in rendered
