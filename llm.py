"""Talk to Grok: extract invoices, review them, optionally call inventory tools."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ValidationError

from config import XAI_BASE_URL, XAI_MODEL, load_api_key
from models import ExtractedInvoice, Invoice, Issue, LineItem, RemapResult, Review
from parsers import parse_date
from tools import TOOLS, run_tool

EXTRACT = (
    "Extract invoice data from the raw document. "
    "Normalize item names that clearly match WidgetA, WidgetB, or GadgetX. "
    "Use the document total as amount. If a due date is unparseable, return null."
)
REVIEW = (
    "You are the AP reviewer for a manufacturing firm. "
    "Use list_catalog and lookup_stock when item identity is unclear. "
    "Recommend approve, reject, or escalate. "
    "Reject only for concrete fraud or nonsense (fake vendor names, round-number scam amounts, "
    "threatening/urgent payment language, impossible terms). "
    "Escalate when the invoice is unusual but not clearly fraudulent. "
    "Do not reject for inventory/stock problems — those are handled separately. "
    "Set confidence between 0 and 1. Put concrete red flags in flags; leave flags empty if none."
)
CHALLENGE = (
    "You are a second AP reviewer. Your only job is to find concrete problems the first reviewer missed. "
    "Use list_catalog and lookup_stock if an item name looks wrong. "
    "Recommend reject only with specific evidence, listed in flags. "
    "If the invoice is legitimate, recommend approve with empty flags. Do not invent concerns."
)
REMAP = (
    "You correct OCR/typo item names on invoices. "
    "Use list_catalog and lookup_stock. "
    "Rename an item to a catalog name only when it is clearly the same product "
    "(spacing, hyphens, obvious OCR). Leave true unknowns unchanged."
)


def extract_invoice(text: str) -> Invoice:
    got = _ask(ExtractedInvoice, EXTRACT, text)
    return Invoice(
        vendor=got.vendor.strip(),
        amount=Decimal(str(got.amount)),
        items=[LineItem(name=i.name.strip(), quantity=i.quantity) for i in got.items],
        due_date=parse_date(got.due_date),
    )


def review_invoice(
    invoice: Invoice,
    issues: list[Issue],
    source_text: str = "",
    prior_challenge: Review | None = None,
) -> Review:
    prompt = _describe(invoice, issues, source_text)
    if prior_challenge and prior_challenge.flags:
        prompt += (
            "\nA second reviewer raised: "
            + "; ".join(prior_challenge.flags)
            + f"\nTheir reason: {prior_challenge.reason}\n"
            "Address those points. Change your recommendation if they are concrete.\n"
        )
    return _ask(Review, REVIEW, prompt, use_tools=True)


def challenge_review(invoice: Invoice, first: Review, source_text: str = "") -> Review:
    prompt = _describe(invoice, [], source_text)
    prompt += (
        f"\nFirst reviewer: {first.recommendation} ({first.confidence:.2f}). "
        f"Reason: {first.reason}. Flags: {first.flags or 'none'}.\n"
    )
    return _ask(Review, CHALLENGE, prompt, use_tools=True)


def remap_items(invoice: Invoice, issues: list[Issue], source_text: str = "") -> Invoice:
    unknown = [issue.item for issue in issues if issue.item]
    prompt = _describe(invoice, issues, source_text)
    prompt += "\nUnknown item names: " + ", ".join(unknown) + "\n"
    got = _ask(RemapResult, REMAP, prompt, use_tools=True)
    return invoice.model_copy(
        update={"items": [LineItem(name=i.name.strip(), quantity=i.quantity) for i in got.items]}
    )


def _ask(model: type[BaseModel], system: str, user: str, use_tools: bool = False) -> BaseModel:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if use_tools:
        early = _tool_loop(messages, model)
        if early is not None:
            return early
        messages.append({"role": "user", "content": f"Return the final {model.__name__} now."})
    return _parse(model, messages)


def _tool_loop(messages: list, model: type[BaseModel]) -> BaseModel | None:
    for _ in range(4):
        msg = _post(messages, tools=TOOLS)["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            content = msg.get("content") or ""
            if content:
                try:
                    return model.model_validate_json(content)
                except ValidationError:
                    pass
            return None
        messages.append(msg)
        for call in calls:
            args = json.loads(call["function"]["arguments"] or "{}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": run_tool(call["function"]["name"], args),
                }
            )
    return None


def _parse(model: type[BaseModel], messages: list) -> BaseModel:
    body = _post(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": model.model_json_schema(),
                "strict": True,
            },
        },
    )
    content = body["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError(f"Model returned empty {model.__name__} output")
    return model.model_validate_json(content)


def _post(messages: list, **extra: Any) -> dict:
    key = load_api_key()
    if not key:
        raise RuntimeError("No API key found. Set XAI_API_KEY or add key to api.md")
    req = urllib.request.Request(
        f"{XAI_BASE_URL}/chat/completions",
        data=json.dumps({"model": XAI_MODEL, "messages": messages, **extra}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"xAI request failed ({e.code}): {e.read().decode('utf-8', 'replace')}") from e


def _describe(invoice: Invoice, issues: list[Issue], source_text: str) -> str:
    items = ", ".join(f"{i.name} x{i.quantity}" for i in invoice.items) or "(none)"
    lines = "; ".join(i.detail for i in issues) or "none"
    text = (
        f"Vendor: {invoice.vendor}\n"
        f"Amount: {invoice.amount}\n"
        f"Items: {items}\n"
        f"Due date: {invoice.due_date or 'unknown'}\n"
        f"Mechanical issues already recorded: {lines}\n"
    )
    excerpt = source_text.strip()[:4000]
    if excerpt:
        text += f"\nSource excerpt:\n{excerpt}\n"
    return text
