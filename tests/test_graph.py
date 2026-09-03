from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from graph import compile_graph
from models import Invoice, LineItem, Review


def _approve(**overrides):
    data = {"recommendation": "approve", "confidence": 0.95, "flags": [], "reason": "Looks legitimate"}
    data.update(overrides)
    return Review(**data)


def _run(path, thread="t"):
    graph = compile_graph(InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke({"invoice_path": str(path)}, config)
    return graph, config, result


def test_invalid_json_rejects_without_llm(inventory, invoices):
    with patch("graph.review_invoice") as review, patch("graph.extract_invoice") as extract:
        graph, _, result = _run(invoices / "invoice_1009.json")
    extract.assert_not_called()
    review.assert_not_called()
    assert result["outcome"] == "rejected"
    codes = {issue.code.value if hasattr(issue.code, "value") else issue.code for issue in result["issues"]}
    assert "missing_vendor" in codes
    assert "invalid_qty" in codes


def test_stock_mismatch_rejects(inventory, invoices):
    with patch("graph.review_invoice") as review:
        _, _, result = _run(invoices / "invoice_1005.json")
    review.assert_not_called()
    assert result["outcome"] == "rejected"
    assert any("Insufficient stock" in issue.detail for issue in result["issues"])


def test_clean_invoice_pays_after_challenge(inventory, invoices):
    with patch("graph.review_invoice", return_value=_approve()), patch(
        "graph.challenge_review", return_value=_approve(reason="No issues")
    ):
        _, _, result = _run(invoices / "invoice_1004.json")
    assert result["outcome"] == "paid"
    assert result["payment"].success is True


def test_challenge_flags_trigger_second_review(inventory, invoices):
    reviews = [
        _approve(reason="First pass"),
        _approve(recommendation="reject", confidence=0.9, flags=["vendor looks fake"], reason="On reflection, reject"),
    ]
    with patch("graph.review_invoice", side_effect=reviews) as review, patch(
        "graph.challenge_review",
        return_value=_approve(recommendation="escalate", flags=["vendor looks fake"], reason="Check the vendor"),
    ):
        _, _, result = _run(invoices / "invoice_1004.json")
    assert review.call_count == 2
    assert result["outcome"] == "rejected"
    assert "On reflection" in result["reason"]


def test_unknown_item_is_corrected_once(inventory, tmp_path):
    invoice_path = tmp_path / "typo.json"
    invoice_path.write_text(
        '{"vendor": "Acme", "total": 250, "line_items": [{"item": "Widget-A", "quantity": 1}], "due_date": "2026-02-01"}'
    )
    remapped = Invoice(vendor="Acme", amount=Decimal("250"), items=[LineItem(name="WidgetA", quantity=1)])
    with patch("graph.remap_items", return_value=remapped) as remap, patch(
        "graph.review_invoice", return_value=_approve()
    ), patch("graph.challenge_review", return_value=_approve(reason="none")):
        _, _, result = _run(invoice_path, thread="typo")
    remap.assert_called_once()
    assert result["outcome"] == "paid"
    assert result["invoice"].items[0].name == "WidgetA"


def test_true_unknown_still_rejects_after_correction(inventory, invoices):
    with patch("graph.remap_items", side_effect=lambda inv, issues, src: inv) as remap, patch(
        "graph.review_invoice"
    ) as review:
        _, _, result = _run(invoices / "invoice_1016.json")
    remap.assert_called_once()
    review.assert_not_called()
    assert result["outcome"] == "rejected"
    assert any(issue.code.value == "unknown_item" or issue.code == "unknown_item" for issue in result["issues"])


def test_human_gate_resume(inventory, invoices):
    graph = compile_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "gate"}}
    with patch("graph.review_invoice", return_value=_approve(recommendation="escalate", reason="Over policy")), patch(
        "graph.challenge_review", return_value=_approve(reason="No extra flags")
    ):
        graph.invoke({"invoice_path": str(invoices / "invoice_1004.json")}, config)
        paused = graph.get_state(config)
        assert paused.next == ("human_gate",) or paused.tasks
        final = graph.invoke(Command(resume="approve"), config)
    assert final["outcome"] == "paid"
