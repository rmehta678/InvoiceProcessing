"""Central configuration: paths, policy thresholds, and LLM settings.

Everything tunable lives here so the policy the system enforces can be read in
one place rather than reverse-engineered from the agent prompts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INVOICE_DIR = DATA_DIR / "invoices"
DB_PATH = DATA_DIR / "inventory.db"
RUNS_DIR = PROJECT_ROOT / "runs"

ENV_FILE = PROJECT_ROOT / ".env"

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-3"

# Fallback provider. The case asks for Grok as the reasoning engine, so xAI
# leads; OpenRouter picks up only when xAI cannot serve at all.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Low, not default: this pipeline supplies its own reasoning structure (the
# extraction repair loop and the VP/audit critique cycle), and an unconstrained
# reasoning model spends its whole output budget thinking and truncates
# mid-JSON. Raise it only if you have measured that it helps.
DEFAULT_OPENROUTER_REASONING_EFFORT = "low"

# Deterministic decisions: the same invoice must reach the same verdict twice.
TEMPERATURE = 0.0

# The README's headline rule: invoices over $10K get additional scrutiny.
APPROVAL_THRESHOLD = 10_000.0

# Amounts landing within this fraction below the threshold are treated as
# suspicious rather than clean. Two invoices in the sample set ($9,900 and
# $9,975) sit just underneath $10K, which is a well-known split-invoice tactic.
THRESHOLD_PROXIMITY_RATIO = 0.02

# Currency the business pays in. Anything else needs a human (FX rate, hedging).
BASE_CURRENCY = "USD"

# Tolerance for arithmetic reconciliation, in currency units. Covers ordinary
# rounding on percentage tax without letting real discrepancies through.
ARITHMETIC_TOLERANCE = 0.05

# Item-name similarity at or above this is offered as a "did you mean" hint.
# Never used to substitute automatically -- see tools/inventory.py.
FUZZY_SUGGEST_THRESHOLD = 0.90

MAX_EXTRACTION_REPAIRS = 2
MAX_APPROVAL_REFLECTIONS = 2

# How many call-tools/read-results rounds the validation agent gets before it is
# asked to summarise with no tools attached. Bounds a model that keeps querying
# the catalogue; real runs settle in three or four.
MAX_VALIDATION_TOOL_ROUNDS = 6


def load_dotenv(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Load `KEY=value` pairs from a .env file into the environment.

    A real environment variable wins over the file, matching the usual dotenv
    convention. Returns what was applied. Never raises on a malformed file: a
    bad line in optional config is skipped, not a reason to fail startup.
    """
    path = Path(path) if path is not None else ENV_FILE
    applied: dict[str, str] = {}
    if not path.is_file():
        return applied

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return applied

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        # Strip matching surrounding quotes; leave inner content untouched so a
        # key containing '#' or spaces survives intact.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif "#" in value:
            # Only an unquoted value can carry a trailing comment.
            value = value.split("#", 1)[0].strip()

        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    return applied


@dataclass
class Settings:
    """Runtime settings for a single invocation."""

    model: str = DEFAULT_MODEL
    db_path: Path = DB_PATH
    runs_dir: Path = RUNS_DIR
    api_key: str | None = field(default=None)
    base_url: str = XAI_BASE_URL
    max_extraction_repairs: int = MAX_EXTRACTION_REPAIRS
    max_approval_reflections: int = MAX_APPROVAL_REFLECTIONS

    # Fallback provider, tried only when xAI cannot serve.
    openrouter_api_key: str | None = None
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_reasoning_effort: str = DEFAULT_OPENROUTER_REASONING_EFFORT

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        load_dotenv()
        base = cls(
            api_key=os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"),
            model=os.environ.get("INVOICE_FLOW_MODEL", DEFAULT_MODEL),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            openrouter_reasoning_effort=os.environ.get(
                "OPENROUTER_REASONING_EFFORT", DEFAULT_OPENROUTER_REASONING_EFFORT
            ),
        )
        for name, value in overrides.items():
            if value is not None:
                setattr(base, name, value)
        return base
