"""Invoice graph.

  ingest → validate ┬→ correct (unknown names, once) → validate
                    ├→ reject
                    └→ review → challenge ┬→ review (if challenger raised flags)
                                          ├→ human_gate (>$10k / escalate / flags)
                                          ├→ pay
                                          └→ reject
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pdfplumber
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from config import APPROVAL_THRESHOLD, REVIEW_CONFIDENCE_MIN
from db import lookup_stock
from llm import challenge_review, extract_invoice, remap_items, review_invoice
from models import Event, Invoice, Issue, IssueCode, PaymentResult, Review
from parsers import ParseError, parse_json
from payment import mock_payment
from state import InputState, InvoiceState, OutputState


def ingest(state: InvoiceState) -> Command:
    path = Path(state["invoice_path"])
    try:
        invoice, source = _load(path)
    except (FileNotFoundError, ParseError, OSError, ValueError, RuntimeError) as e:
        return _fail("ingest", str(e))

    return _go(
        "ingest",
        "validate",
        f"{invoice.vendor or '(no vendor)'}, {invoice.amount}, {len(invoice.items)} items",
        invoice=invoice,
        source_text=source,
    )


def validate(state: InvoiceState) -> Command:
    invoice = state.get("invoice")
    if invoice is None:
        return _fail("validate", "No invoice in state")

    issues = _mechanical_issues(invoice)
    if not issues:
        return _go("validate", "review", "Mechanical checks passed", issues=[])

    summary = "; ".join(issue.detail for issue in issues)
    # Only unknown SKUs get a remap pass. Stock/qty/vendor failures stop here.
    if _unknown_only(issues) and not state.get("correction_passes"):
        return _go("validate", "correct", f"{summary} — attempting name correction", issues=issues)
    return _go("validate", "reject", summary, issues=issues)


def correct(state: InvoiceState) -> Command:
    updated = remap_items(state["invoice"], state.get("issues", []), state.get("source_text", ""))
    names = ", ".join(f"{item.name} x{item.quantity}" for item in updated.items)
    return _go(
        "correct",
        "validate",
        f"Remapped items: {names}",
        invoice=updated,
        issues=[],
        correction_passes=state.get("correction_passes", 0) + 1,
    )


def review(state: InvoiceState) -> Command:
    invoice = state["invoice"]
    round_no = state.get("review_round", 0) + 1
    result = review_invoice(
        invoice,
        state.get("issues", []),
        state.get("source_text", ""),
        prior_challenge=state.get("challenge"),
    )
    if result.recommendation == "reject":
        nxt = "reject"
    elif round_no >= 2:
        nxt = _after_review(invoice, result)
    else:
        nxt = "challenge"
    return _go(
        "review",
        nxt,
        f"{result.recommendation} ({result.confidence:.2f}): {result.reason}",
        review=result,
        review_round=round_no,
    )


def challenge(state: InvoiceState) -> Command:
    invoice = state["invoice"]
    first = state["review"]
    result = challenge_review(invoice, first, state.get("source_text", ""))
    extra = {"challenge": result}
    if result.recommendation == "reject" and result.flags:
        nxt = "reject"
        extra["review"] = result
    elif result.flags:
        nxt = "review"
    else:
        nxt = _after_review(invoice, first)
    return _go(
        "challenge",
        nxt,
        f"{result.recommendation} ({result.confidence:.2f}): {result.reason}",
        **extra,
    )


def human_gate(state: InvoiceState) -> Command:
    # interrupt() pauses the graph. On resume this node restarts from the top,
    # so don't put side effects above the interrupt call.
    invoice = state["invoice"]
    review_result = state.get("review")
    decision = interrupt(
        {
            "question": f"Pay {invoice.amount} to {invoice.vendor}?",
            "invoice": invoice.model_dump(mode="json"),
            "issues": [issue.model_dump(mode="json") for issue in state.get("issues", [])],
            "review": review_result.model_dump() if review_result else None,
        }
    )
    if str(decision).strip().lower() in {"approve", "approved", "yes", "y"}:
        return _go("human_gate", "pay", "Human approved payment")
    return _go("human_gate", "reject", "Human rejected payment", reason="Rejected by reviewer")


def pay(state: InvoiceState) -> dict:
    result = mock_payment(state["invoice"].vendor, state["invoice"].amount)
    return {
        "payment": result,
        "outcome": "paid" if result.success else "rejected",
        "reason": result.message,
        "events": [Event(node="pay", message=result.message)],
    }


def reject(state: InvoiceState) -> dict:
    review_result = state.get("review")
    if state.get("reason"):
        reason = state["reason"]
    elif review_result and review_result.recommendation == "reject":
        reason = review_result.reason
    elif state.get("issues"):
        reason = "; ".join(issue.detail for issue in state["issues"])
    else:
        reason = "Rejected"
    return {
        "outcome": "rejected",
        "reason": reason,
        "payment": PaymentResult(success=False, message=f"Rejected: {reason}"),
        "events": [Event(node="reject", message=reason)],
    }


def compile_graph(checkpointer):
    g = StateGraph(InvoiceState, input_schema=InputState, output_schema=OutputState)
    for name, fn in [
        ("ingest", ingest),
        ("validate", validate),
        ("correct", correct),
        ("review", review),
        ("challenge", challenge),
        ("human_gate", human_gate),
        ("pay", pay),
        ("reject", reject),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("pay", END)
    g.add_edge("reject", END)
    return g.compile(checkpointer=checkpointer)


def _go(here: str, there: str, message: str, **fields) -> Command:
    return Command(update={**fields, "events": [Event(node=here, message=message)]}, goto=there)


def _fail(node: str, detail: str) -> Command:
    return _go(node, "reject", detail, issues=[Issue(code=IssueCode.UNREADABLE, detail=detail)])


def _after_review(invoice: Invoice, result: Review) -> str:
    if (
        invoice.amount > APPROVAL_THRESHOLD
        or result.recommendation == "escalate"
        or result.flags
        or result.confidence < REVIEW_CONFIDENCE_MIN
    ):
        return "human_gate"
    return "pay"


def _unknown_only(issues: list[Issue]) -> bool:
    return bool(issues) and all(issue.code == IssueCode.UNKNOWN_ITEM for issue in issues)


def _mechanical_issues(invoice: Invoice) -> list[Issue]:
    issues: list[Issue] = []
    if not invoice.vendor.strip():
        issues.append(Issue(code=IssueCode.MISSING_VENDOR, detail="Missing vendor name"))
    if invoice.amount <= 0:
        issues.append(Issue(code=IssueCode.NONPOSITIVE_AMOUNT, detail=f"Invalid amount: {invoice.amount}"))
    if not invoice.items:
        issues.append(Issue(code=IssueCode.EMPTY_ITEMS, detail="No line items"))

    totals: dict[str, int] = defaultdict(int)
    label: dict[str, str] = {}
    for item in invoice.items:
        key = item.name.casefold()
        totals[key] += item.quantity
        label[key] = item.name

    for key, qty in totals.items():
        name = label[key]
        if qty < 0:
            issues.append(Issue(code=IssueCode.INVALID_QTY, item=name, detail=f"Invalid quantity for {name}: {qty}"))
            continue
        stock = lookup_stock(name)
        if stock is None:
            issues.append(Issue(code=IssueCode.UNKNOWN_ITEM, item=name, detail=f"Item not found: {name}"))
        elif stock == 0:
            issues.append(Issue(code=IssueCode.OUT_OF_STOCK, item=name, detail=f"Out of stock: {name}"))
        elif qty > stock:
            issues.append(
                Issue(
                    code=IssueCode.INSUFFICIENT_STOCK,
                    item=name,
                    detail=f"Insufficient stock for {name}: requested {qty}, available {stock}",
                )
            )
    return issues


def _load(path: Path) -> tuple[Invoice, str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # JSON is already structured. CSV/XML/PDF/text go to Grok.
    if path.suffix.lower() == ".json":
        return parse_json(path), path.read_text(errors="replace")
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = path.read_text(errors="replace")
    if not text.strip():
        raise ValueError(f"Empty document {path.name}")
    return extract_invoice(text), text
