"""Tests for .env loading and settings resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from invoice_flow.config import DEFAULT_MODEL, Settings, load_dotenv


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real credentials out of these assertions."""
    for name in ("XAI_API_KEY", "GROK_API_KEY", "INVOICE_FLOW_MODEL", "SOME_KEY"):
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path: Path, content: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_simple_pair(tmp_path: Path) -> None:
    applied = load_dotenv(write_env(tmp_path, "XAI_API_KEY=xai-abc123\n"))
    assert applied == {"XAI_API_KEY": "xai-abc123"}
    assert os.environ["XAI_API_KEY"] == "xai-abc123"


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_real_environment_wins_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`XAI_API_KEY=... python main.py` must beat whatever is on disk."""
    monkeypatch.setenv("XAI_API_KEY", "from-shell")
    load_dotenv(write_env(tmp_path, "XAI_API_KEY=from-file\n"))
    assert os.environ["XAI_API_KEY"] == "from-shell"


def test_override_flag_forces_file_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "from-shell")
    load_dotenv(write_env(tmp_path, "XAI_API_KEY=from-file\n"), override=True)
    assert os.environ["XAI_API_KEY"] == "from-file"


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    applied = load_dotenv(
        write_env(
            tmp_path,
            "# a comment\n\n  \nXAI_API_KEY=xai-abc\n# trailing comment\n",
        )
    )
    assert applied == {"XAI_API_KEY": "xai-abc"}


def test_export_prefix_is_tolerated(tmp_path: Path) -> None:
    """People paste the line straight out of their shell profile."""
    applied = load_dotenv(write_env(tmp_path, "export XAI_API_KEY=xai-abc\n"))
    assert applied == {"XAI_API_KEY": "xai-abc"}


@pytest.mark.parametrize(
    "line,expected",
    [
        ('XAI_API_KEY="xai-abc"', "xai-abc"),
        ("XAI_API_KEY='xai-abc'", "xai-abc"),
        ("XAI_API_KEY=xai-abc  ", "xai-abc"),
        ('XAI_API_KEY="xai-with#hash"', "xai-with#hash"),
        ("XAI_API_KEY=xai-abc # inline comment", "xai-abc"),
        ('XAI_API_KEY="  padded  "', "  padded  "),
    ],
)
def test_quoting_and_trailing_comments(tmp_path: Path, line: str, expected: str) -> None:
    load_dotenv(write_env(tmp_path, line + "\n"))
    assert os.environ["XAI_API_KEY"] == expected


def test_malformed_lines_are_skipped_not_fatal(tmp_path: Path) -> None:
    """A stray line must not stop the program starting."""
    applied = load_dotenv(
        write_env(tmp_path, "this line has no equals sign\n=novalue\nXAI_API_KEY=xai-ok\n")
    )
    assert applied == {"XAI_API_KEY": "xai-ok"}


def test_settings_picks_up_key_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("invoice_flow.config.ENV_FILE", write_env(tmp_path, "XAI_API_KEY=xai-zzz\n"))
    settings = Settings.from_env()
    assert settings.api_key == "xai-zzz"
    assert settings.model == DEFAULT_MODEL


def test_dotenv_can_set_the_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "invoice_flow.config.ENV_FILE",
        write_env(tmp_path, "XAI_API_KEY=xai-zzz\nINVOICE_FLOW_MODEL=grok-4\n"),
    )
    assert Settings.from_env().model == "grok-4"


def test_grok_api_key_alias_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("invoice_flow.config.ENV_FILE", write_env(tmp_path, "GROK_API_KEY=xai-alt\n"))
    assert Settings.from_env().api_key == "xai-alt"


def test_cli_overrides_beat_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "invoice_flow.config.ENV_FILE",
        write_env(tmp_path, "XAI_API_KEY=xai-zzz\nINVOICE_FLOW_MODEL=grok-4\n"),
    )
    assert Settings.from_env(model="grok-3-mini").model == "grok-3-mini"
