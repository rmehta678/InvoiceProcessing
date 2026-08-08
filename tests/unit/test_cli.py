"""CLI failures that must remain clear outside the source checkout."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from invoice_agents.cli import app

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
