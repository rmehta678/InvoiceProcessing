"""Shared pytest fixtures and import path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from invoice_flow.config import INVOICE_DIR  # noqa: E402


# Provider settings that `Settings.from_env()` reads. Neutralised for every
# test so the suite asserts against the code's defaults rather than whatever a
# developer happens to have in their local .env -- otherwise a pinned
# pinned OPENROUTER_MODEL silently changes what the tests mean.
_PROVIDER_ENV_VARS = (
    "XAI_API_KEY",
    "GROK_API_KEY",
    "INVOICE_FLOW_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_REASONING_EFFORT",
)


@pytest.fixture(autouse=True)
def isolated_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local credentials and model pins out of the test suite."""
    monkeypatch.setattr("invoice_flow.config.ENV_FILE", Path("nonexistent.env"))
    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def invoice_dir() -> Path:
    return INVOICE_DIR


@pytest.fixture(scope="session")
def all_invoice_files() -> list[Path]:
    from invoice_flow.tools.loaders import discover_invoices

    return discover_invoices(INVOICE_DIR)


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    """A freshly seeded inventory database isolated to one test."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from init_db import initialise

    db = tmp_path / "inventory.db"
    initialise(db)
    return db
