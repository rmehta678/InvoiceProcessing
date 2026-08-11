"""UI and CLI consume the same sanitized terminal aggregate."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

from invoice_agents import cli
from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.models import CaseResult, CaseStatus, ErrorRecord, UsageSummary
from invoice_agents.orchestration import prepare_case
from invoice_agents.ui.server import create_app


def test_unicode_split_and_continued_credentials_never_reach_ui_or_cli(
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_case(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    case_id, started_at = prepared
    store = WorkflowStore(settings)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    invoice = store.promote_predecessor_extraction(claim)
    cookie_a = "round2-surface-cookie-a"
    cookie_b = "round2-surface-cookie-b"
    split_marker = "abcdefgh_12345678"
    mixed_script_marker = "round5-surface-canary"
    result = CaseResult(
        case_id=case_id,
        source_id=invoice.source.source_id,
        status=CaseStatus.FAILED,
        stop_reason="PROVIDER_REQUEST_FAILED",
        errors=[
            ErrorRecord(
                category="PROVIDER",
                message=f"cookie=session={cookie_a}; preference={cookie_b}",
                case_id=case_id,
                stop_reason="PROVIDER_REQUEST_FAILED",
                details={
                    "cause": f"provider rejected sk-abcd\u2064efgh_{split_marker}",
                    "mixed_script": f"api_κey={mixed_script_marker}",
                },
            )
        ],
        usage=UsageSummary(prompt_tokens=47, completion_tokens=12),
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    store.finish_case(result, claim)

    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        ui_output = client.get(f"/cases/{case_id}").text
    cli_stream = io.StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=cli_stream, force_terminal=False, color_system=None, width=160),
    )
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    cli.case_status(case_id)
    cli_output = cli_stream.getvalue()

    for output in (ui_output, cli_output):
        assert cookie_a not in output
        assert cookie_b not in output
        assert split_marker not in output
        assert mixed_script_marker not in output
        assert "[REDACTED]" in output
    assert "47" in ui_output and "12" in ui_output
    assert '"prompt_tokens": 47' in cli_output
    assert '"completion_tokens": 12' in cli_output
