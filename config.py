from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "inventory.db"
CHECKPOINT_PATH = BASE_DIR / "checkpoints.db"
OUTPUT_DIR = BASE_DIR / "output"

APPROVAL_THRESHOLD = Decimal(os.getenv("APPROVAL_THRESHOLD", "10000"))
REVIEW_CONFIDENCE_MIN = float(os.getenv("REVIEW_CONFIDENCE_MIN", "0.8"))

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3")


def load_api_key() -> str:
    if XAI_API_KEY:
        return XAI_API_KEY

    api_file = BASE_DIR.parent / "api.md"
    if api_file.exists():
        key = api_file.read_text().strip()
        if key.startswith("xai-"):
            return key

    return ""
