"""Read-only preflight snapshot for the console header; it never repairs anything.

The strip mirrors :func:`invoice_agents.orchestration.preflight` item by item and
reports either OK or the exact stop reason with the verbatim fix command.
"""

from __future__ import annotations

from dataclasses import dataclass

from invoice_agents.config import Settings
from invoice_agents.db.core import DatabaseKind, verify_database
from invoice_agents.errors import InvoiceAgentsError


@dataclass(slots=True)
class PreflightItem:
    name: str
    ok: bool
    detail: str
    stop_reason: str | None = None
    fix_command: str | None = None


@dataclass(slots=True)
class PreflightReport:
    items: list[PreflightItem]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)


def key_present(settings: Settings) -> bool:
    """Whether XAI_API_KEY is configured; the value itself is never surfaced."""

    key = settings.xai_api_key
    return key is not None and bool(key.get_secret_value().strip())


def run_preflight(settings: Settings) -> PreflightReport:
    """Verify both databases and key presence exactly as case preflight would."""

    items: list[PreflightItem] = []
    checks = (
        (
            "inventory DB",
            settings.inventory_db,
            DatabaseKind.INVENTORY,
            f"uv run python -m invoice_agents.db migrate --db {settings.inventory_db}; "
            f"uv run python -m invoice_agents.db seed --db {settings.inventory_db}",
        ),
        (
            "workflow DB",
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            f"uv run python -m invoice_agents.db migrate --db {settings.workflow_db} "
            "--kind workflow",
        ),
    )
    for name, path, kind, fix_command in checks:
        try:
            info = verify_database(path, kind, settings=settings)
            items.append(
                PreflightItem(
                    name=name,
                    ok=True,
                    detail=f"schema v{info['schema_version']}, integrity ok",
                )
            )
        except InvoiceAgentsError as exc:
            items.append(
                PreflightItem(
                    name=name,
                    ok=False,
                    detail=exc.message,
                    stop_reason=exc.stop_reason,
                    fix_command=fix_command,
                )
            )
    if key_present(settings):
        items.append(PreflightItem(name="XAI_API_KEY", ok=True, detail="configured"))
    else:
        items.append(
            PreflightItem(
                name="XAI_API_KEY",
                ok=False,
                detail="XAI_API_KEY is missing or empty; no model fallback is permitted",
                stop_reason="PROVIDER_PREFLIGHT_FAILED",
                fix_command="Set XAI_API_KEY in .env",
            )
        )
    return PreflightReport(items=items)
