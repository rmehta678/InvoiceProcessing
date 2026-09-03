from __future__ import annotations

import json
from typing import Any, Callable

from db import list_inventory, lookup_stock

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_catalog",
            "description": "List every item in the inventory catalog with current stock.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_stock",
            "description": "Look up available stock for one catalog item name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "Item name to look up"},
                },
                "required": ["item"],
                "additionalProperties": False,
            },
        },
    },
]


def tool_list_catalog() -> dict[str, Any]:
    return {"catalog": [{"item": name, "stock": stock} for name, stock in list_inventory()]}


def tool_lookup_stock(item: str) -> dict[str, Any]:
    stock = lookup_stock(item)
    if stock is None:
        return {"item": item, "found": False, "stock": None}
    return {"item": item, "found": True, "stock": stock}


_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "list_catalog": lambda: tool_list_catalog(),
    "lookup_stock": lambda item: tool_lookup_stock(item),
}


def run_tool(name: str, arguments: dict[str, Any] | None = None) -> str:
    func = _DISPATCH.get(name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = func(**(arguments or {}))
    except TypeError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(result)
