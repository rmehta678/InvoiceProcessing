"""Validation agent: reconcile an invoice against inventory and policy.

Two layers, deliberately separated:

* **Deterministic checks** produce the authoritative findings. Whether 22 units
  exceed a stock of 15 is arithmetic, and an LLM adds nothing but variance.
* **The LLM tool-calling agent** queries the same database through
  `lookup_item` / `check_stock` / `list_catalog` and writes the human-readable
  summary that goes to the VP.

The model cannot overturn a deterministic finding. It can only interpret. That
boundary is what keeps a fluent-but-wrong summary from authorising a payment.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import MAX_VALIDATION_TOOL_ROUNDS, Settings
from ..llm.base import LLMClient, LLMError
from ..llm.prompts import VALIDATION_SYSTEM, validation_user_prompt
from ..models import (
    Finding,
    FindingCode,
    InvoiceDraft,
    Severity,
    SourceDocument,
    ValidationReport,
    money,
)
from ..tools import risk
from ..tools.arithmetic import check_arithmetic, check_data_integrity
from ..tools.dates import check_dates, resolve_dates
from ..tools.inventory import INVENTORY_TOOL_SCHEMAS, InventoryRepository, check_inventory
from ..tools.payment import payable_fingerprint


def _confidence_finding(draft: InvoiceDraft) -> list[Finding]:
    """Surface a shaky extraction as a finding the VP can weigh."""
    if draft.extraction_confidence >= 0.8:
        return []
    return [
        Finding(
            code=FindingCode.EXTRACTION_LOW_CONFIDENCE,
            severity=Severity.WARNING,
            message=(
                f"Extraction confidence is {draft.extraction_confidence:.0%}: the document's "
                "figures could not be reconciled across repeated reads. Treat the "
                "extracted values as approximate."
            ),
            detail={"confidence": draft.extraction_confidence},
            source="ingestion",
        )
    ]


def _reconstructed_total_finding(draft: InvoiceDraft) -> list[Finding]:
    """Flag a total we derived rather than read off the document."""
    if not draft.total_reconstructed:
        return []
    return [
        Finding(
            code=FindingCode.TOTAL_RECONSTRUCTED,
            severity=Severity.WARNING,
            message=(
                f"No total was stated; ${draft.total or 0:,.2f} was derived from the line "
                "items, tax, and shipping. The usual cross-check between stated and computed "
                "totals is unavailable for this invoice, so a vendor arithmetic error here "
                "would not be caught."
            ),
            detail={"derived_total": draft.total},
            source="ingestion",
        )
    ]


def _duplicate_findings(draft: InvoiceDraft, repo: InventoryRepository) -> list[Finding]:
    """Compare this invoice against anything already seen under its number.

    Runs here rather than at the payment stage so the outcome can influence the
    decision. Blocking a second payment after a VP has already approved it is a
    backstop, not a control.
    """
    prior = repo.prior_invoices(draft.invoice_number)
    if not prior:
        return []

    fingerprint = payable_fingerprint(draft)
    matching = [row for row in prior if row.get("content_hash") == fingerprint]
    conflicting = [
        row for row in prior if row.get("content_hash") and row.get("content_hash") != fingerprint
    ]

    # Whether a differing version is a crisis or a routine correction turns on
    # one fact: did money move? If an earlier version was paid, this one cannot
    # simply be approved -- that pays the vendor twice. If nothing was paid, a
    # revision is just a corrected invoice and deserves to be judged on its own
    # merits rather than condemned by a version already thrown out.
    paid_conflicts = [row for row in conflicting if row.get("payment_status") == "success"]
    marker = f" (document declares revision {draft.revision})" if draft.revision else ""

    if paid_conflicts:
        previous = paid_conflicts[-1]
        already_paid = previous.get("amount") or 0.0
        delta = (draft.total or 0.0) - already_paid
        return [
            Finding(
                code=FindingCode.DUPLICATE_INVOICE_CONFLICT,
                severity=Severity.CRITICAL,
                message=(
                    f"{money(already_paid)} was already paid under invoice number "
                    f"{draft.invoice_number} on {previous.get('processed_at')}, and this "
                    f"version totals {money(draft.total)}{marker}. Approving it pays the "
                    f"vendor {money((draft.total or 0.0) + already_paid)} in total. If this "
                    f"supersedes the original, the amount outstanding is {money(delta)} -- "
                    "settle the difference or void the first payment. A human must decide "
                    "which; this system cannot reverse a payment it already made."
                ),
                detail={
                    "invoice_number": draft.invoice_number,
                    "previous_amount": already_paid,
                    "current_amount": draft.total,
                    "outstanding_difference": round(delta, 2),
                    "previous_source": previous.get("source_path"),
                    "previous_decision": previous.get("decision"),
                    "previous_payment_status": previous.get("payment_status"),
                    "revision": draft.revision,
                },
                source="ledger",
            )
        ]

    if conflicting:
        previous = conflicting[-1]
        return [
            Finding(
                code=FindingCode.DUPLICATE_INVOICE_REVISION,
                severity=Severity.WARNING,
                message=(
                    f"A different version of invoice {draft.invoice_number} was seen on "
                    f"{previous.get('processed_at')} for "
                    f"{money(previous.get('amount'))} and was not paid "
                    f"(decision {previous.get('decision')}). This version totals "
                    f"{money(draft.total)}{marker}. Nothing was disbursed against the "
                    "earlier one, so this is a corrected submission rather than a conflict "
                    "-- judge it on its own merits."
                ),
                detail={
                    "invoice_number": draft.invoice_number,
                    "previous_amount": previous.get("amount"),
                    "current_amount": draft.total,
                    "previous_source": previous.get("source_path"),
                    "previous_decision": previous.get("decision"),
                    "previous_payment_status": previous.get("payment_status"),
                    "revision": draft.revision,
                },
                source="ledger",
            )
        ]

    if matching:
        previous = matching[-1]
        return [
            Finding(
                code=FindingCode.DUPLICATE_INVOICE,
                severity=Severity.INFO,
                message=(
                    f"This exact invoice was already processed on "
                    f"{previous.get('processed_at')} (decision {previous.get('decision')}, "
                    f"source {previous.get('source_path')}). Same vendor, items, and total -- "
                    "the same document arriving again, not a new charge."
                ),
                detail={
                    "invoice_number": draft.invoice_number,
                    "previous_decision": previous.get("decision"),
                    "previous_source": previous.get("source_path"),
                    "occurrences": len(matching) + 1,
                },
                source="ledger",
            )
        ]

    return []


def run_deterministic_checks(
    draft: InvoiceDraft,
    repo: InventoryRepository,
    document_text: str = "",
) -> ValidationReport:
    """Every rule-based check, with no LLM involvement.

    Exposed separately so the test suite can assert on validation behaviour
    without a model in the loop.
    """
    # Idempotent, and it makes this entry point self-sufficient: `check_dates`
    # reads the resolved fields, so a draft built by hand -- as the tests do --
    # must not be judged as having no due date simply because nobody parsed it.
    resolve_dates(draft)

    item_checks, inventory_findings = check_inventory(draft, repo)

    findings = [
        *check_data_integrity(draft),
        *check_arithmetic(draft),
        *check_dates(draft),
        *inventory_findings,
        *risk.assess(draft, document_text),
        *_confidence_finding(draft),
        *_reconstructed_total_finding(draft),
        *_duplicate_findings(draft, repo),
    ]

    severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    findings.sort(key=lambda f: severity_order[f.severity])

    return ValidationReport(findings=findings, item_checks=item_checks)


def _run_tool_loop(
    client: LLMClient,
    repo: InventoryRepository,
    draft: InvoiceDraft,
    deterministic: list[Finding],
    tracer: Any = None,
) -> str | None:
    """Let the model interrogate the inventory database, then summarise."""
    dispatch = {
        "lookup_item": repo.lookup_item,
        "check_stock": repo.check_stock,
        "list_catalog": repo.list_catalog,
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": VALIDATION_SYSTEM},
        {"role": "user", "content": validation_user_prompt(draft, deterministic)},
    ]

    for round_index in range(MAX_VALIDATION_TOOL_ROUNDS):
        response = client.complete(
            messages,
            agent=f"validation.r{round_index}",
            tools=INVENTORY_TOOL_SCHEMAS,
        )

        if not response.tool_calls:
            return (response.content or "").strip() or None

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in response.tool_calls
                ],
            }
        )

        for call in response.tool_calls:
            name = call["name"]
            try:
                arguments = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}

            handler = dispatch.get(name)
            if handler is None:
                result: Any = {"error": f"Unknown tool '{name}'"}
            else:
                try:
                    result = handler(**arguments)
                except TypeError as exc:
                    result = {"error": f"Invalid arguments for {name}: {exc}"}
                except Exception as exc:  # noqa: BLE001 - report, don't crash the run
                    result = {"error": f"{type(exc).__name__}: {exc}"}

            if tracer is not None:
                tracer.record_tool_call("validation", name, arguments, result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, default=str),
                }
            )

    # Out of tool rounds: ask once for a summary with no tools available.
    messages.append(
        {
            "role": "user",
            "content": "Stop calling tools. Summarise the invoice's condition in two or three sentences.",
        }
    )
    final = client.complete(messages, agent="validation.summary")
    return (final.content or "").strip() or None


def run_validation(
    draft: InvoiceDraft,
    repo: InventoryRepository,
    client: LLMClient,
    settings: Settings,
    document: SourceDocument | None = None,
    tracer: Any = None,
) -> ValidationReport:
    """Produce the full validation report for an invoice."""
    report = run_deterministic_checks(draft, repo, document.text if document else "")

    try:
        report.agent_summary = _run_tool_loop(client, repo, draft, report.findings, tracer=tracer)
    except LLMError as exc:
        # The deterministic findings stand on their own. Losing the narrative
        # summary degrades the report; it does not invalidate the decision.
        if tracer is not None:
            tracer.emit("validation.summary_failed", error=str(exc))
        report.agent_summary = None

    return report
