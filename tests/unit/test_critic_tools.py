"""Deterministic critic-tool primitives: exact Decimal arithmetic and inventory lookups."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from invoice_agents.db.core import connect_database
from invoice_agents.models import Critique, DecisionKind, ToolStatus
from invoice_agents.tools.comparison import (
    InventoryReader,
    normalize_alias,
    recompute_line_extension,
)


def test_recompute_line_extension_multiplies_exact_decimals() -> None:
    assert recompute_line_extension("4", "300") == {
        "quantity": "4",
        "unit_price": "300",
        "extended_total": "1200",
    }


def test_recompute_line_extension_is_decimal_exact() -> None:
    # 3 * 0.1 must be exactly 0.3, never the float artifact 0.30000000000000004.
    assert recompute_line_extension("3", "0.1")["extended_total"] == "0.3"


def test_recompute_line_extension_rejects_non_decimal_input() -> None:
    with pytest.raises(ValueError, match="exact decimal strings"):
        recompute_line_extension("four", "1")


def test_lookup_inventory_exact_returns_authoritative_row(inventory_db: Path) -> None:
    result = InventoryReader(inventory_db).lookup_inventory_exact("WidgetA")
    assert result.status is ToolStatus.OK
    assert result.row is not None
    assert result.row.sku == "SKU-WIDGET-A"
    assert result.row.item_name == "WidgetA"
    assert result.row.available_stock == 15


def test_lookup_inventory_exact_unknown_item_is_not_found(inventory_db: Path) -> None:
    result = InventoryReader(inventory_db).lookup_inventory_exact("NoSuchItem")
    assert result.status is ToolStatus.NOT_FOUND
    assert result.row is None


def test_lookup_item_alias_resolves_only_persisted_approved_aliases(inventory_db: Path) -> None:
    reader = InventoryReader(inventory_db)
    assert reader.lookup_item_alias("Widget A (rush)").status is ToolStatus.NOT_FOUND
    approved_at = datetime(2026, 8, 1, tzinfo=UTC).isoformat()
    with connect_database(inventory_db) as connection:
        connection.execute(
            "INSERT INTO item_aliases(alias_normalized, sku, source, approved_by, approved_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                normalize_alias("Widget A (rush)"),
                "SKU-WIDGET-A",
                "human_review:rev_test",
                "reviewer@example.com",
                approved_at,
            ),
        )
        connection.commit()
    result = reader.lookup_item_alias("Widget A (rush)")
    assert result.status is ToolStatus.OK
    assert result.row is not None
    assert result.row.sku == "SKU-WIDGET-A"
    assert result.alias_provenance == {
        "source": "human_review:rev_test",
        "approved_by": "reviewer@example.com",
        "approved_at": approved_at,
    }


def test_legacy_critique_payload_defaults_to_the_only_initial_cycle() -> None:
    critique = Critique.model_validate(
        {
            "supported_findings": ["the persisted arithmetic is exact"],
            "challenged_findings": [],
            "missing_evidence": [],
            "requested_follow_up": [],
            "recommended_disposition": "APPROVE",
            "rationale": ["all persisted evidence agrees"],
        }
    )

    assert critique.cycle == 1
    assert critique.responds_to_critique_id is None


def test_critique_model_accepts_an_exact_persisted_cycle_two_relationship() -> None:
    critique = Critique(
        cycle=2,
        responds_to_critique_id="crit_cycle_one",
        supported_findings=["the requested inventory row was rechecked"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=DecisionKind.APPROVE,
        rationale=["the persisted follow-up resolved the challenge"],
    )

    assert critique.cycle == 2
    assert critique.responds_to_critique_id == "crit_cycle_one"


@pytest.mark.parametrize(
    ("cycle", "responds_to_critique_id"),
    [
        (0, None),
        (3, "crit_prior"),
    ],
    ids=["cycle-zero", "cycle-three"],
)
def test_critique_model_rejects_out_of_range_cycle(
    cycle: int,
    responds_to_critique_id: str | None,
) -> None:
    with pytest.raises(ValidationError):
        Critique(
            cycle=cycle,
            responds_to_critique_id=responds_to_critique_id,
            supported_findings=["persisted evidence reviewed"],
            challenged_findings=[],
            missing_evidence=[],
            requested_follow_up=[],
            recommended_disposition=DecisionKind.HOLD,
            rationale=["finite-cycle contract test"],
        )
