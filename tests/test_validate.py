from __future__ import annotations

from decimal import Decimal

from graph import _mechanical_issues, _unknown_only
from models import Invoice, IssueCode, LineItem


def test_clean_invoice_has_no_issues(inventory):
    issues = _mechanical_issues(
        Invoice(
            vendor="Widgets Inc.",
            amount=Decimal("5000"),
            items=[LineItem(name="WidgetA", quantity=10), LineItem(name="WidgetB", quantity=5)],
        )
    )
    assert issues == []


def test_aggregates_duplicate_lines(inventory):
    issues = _mechanical_issues(
        Invoice(
            vendor="Atlas",
            amount=Decimal("1"),
            items=[
                LineItem(name="WidgetA", quantity=15),
                LineItem(name="WidgetA", quantity=5),
            ],
        )
    )
    assert [issue.code for issue in issues] == [IssueCode.INSUFFICIENT_STOCK]
    assert "requested 20" in issues[0].detail


def test_unknown_and_out_of_stock(inventory):
    issues = _mechanical_issues(
        Invoice(
            vendor="X",
            amount=Decimal("1"),
            items=[LineItem(name="SuperGizmo", quantity=1), LineItem(name="FakeItem", quantity=1)],
        )
    )
    codes = {issue.code for issue in issues}
    assert IssueCode.UNKNOWN_ITEM in codes
    assert IssueCode.OUT_OF_STOCK in codes
    assert not _unknown_only(issues)


def test_unknown_only(inventory):
    issues = _mechanical_issues(
        Invoice(vendor="X", amount=Decimal("1"), items=[LineItem(name="MegaSprocket", quantity=1)])
    )
    assert _unknown_only(issues)


def test_space_normalized_catalog_lookup(inventory):
    issues = _mechanical_issues(
        Invoice(vendor="X", amount=Decimal("1"), items=[LineItem(name="Widget A", quantity=1)])
    )
    assert issues == []
