from __future__ import annotations

import json

from tools import run_tool, tool_list_catalog, tool_lookup_stock


def test_lookup_found(inventory):
    result = tool_lookup_stock("widgeta")
    assert result == {"item": "widgeta", "found": True, "stock": 15}


def test_lookup_missing(inventory):
    result = tool_lookup_stock("SuperGizmo")
    assert result["found"] is False


def test_list_catalog(inventory):
    catalog = tool_list_catalog()["catalog"]
    names = {row["item"] for row in catalog}
    assert names == {"FakeItem", "GadgetX", "WidgetA", "WidgetB"}


def test_run_tool_json(inventory):
    payload = json.loads(run_tool("lookup_stock", {"item": "GadgetX"}))
    assert payload["stock"] == 5
    unknown = json.loads(run_tool("nope", {}))
    assert "error" in unknown
