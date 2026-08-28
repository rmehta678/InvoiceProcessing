"""Render a run as a self-contained HTML report.

One file, no external assets, no network requests -- it can be attached to an
email or filed as the audit record for a payment. Jinja2 autoescaping is on, so
vendor-supplied text from an invoice cannot inject markup into the report.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import ITEM_STATUS_LABEL, Decision, RunState, money

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "template.html"

_DECISION_LABEL = {
    Decision.APPROVED: "Approved for payment",
    Decision.REJECTED: "Rejected",
    Decision.ESCALATED: "Escalated for human review",
}

_PAYMENT_PILL = {
    "success": "ok",
    "skipped": "warn",
    "duplicate": "bad",
    "failed": "bad",
}


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def build_context(state: RunState, usage: dict[str, int] | None = None) -> dict[str, Any]:
    """Flatten a run into template variables."""
    draft = state.draft
    approval = state.approval
    report = state.validation
    document = state.document

    currency = draft.currency if draft else None

    context: dict[str, Any] = {
        "run_id": state.run_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source_name": document.name if document else Path(state.invoice_path).name,
        "source_format": document.file_format.upper() if document else "",
        "source_text": document.text if document else "",
        "invoice_number": (draft.invoice_number if draft and draft.invoice_number else "Unnumbered"),
        "vendor": (draft.vendor_name if draft and draft.vendor_name else "Unknown vendor"),
    }

    # Missing identity is a finding in its own right; show it as such.
    context["invoice_number_html"] = (
        draft.invoice_number if draft and draft.invoice_number else "missing"
    )
    context["vendor_html"] = draft.vendor_name if draft and draft.vendor_name else "missing"

    if draft is not None:
        context.update(
            {
                "total": money(draft.total, currency),
                "invoice_date": (
                    draft.invoice_date.isoformat()
                    if draft.invoice_date
                    else (draft.invoice_date_raw or "missing")
                ),
                "due_date": (
                    draft.due_date.isoformat()
                    if draft.due_date
                    else (draft.due_date_raw or "missing")
                ),
                "terms": draft.payment_terms or "-",
                "confidence": f"{draft.extraction_confidence:.0%}",
                "line_items": [
                    {
                        "name": item.name,
                        "quantity": f"{item.quantity:g}" if item.quantity is not None else "-",
                        "unit_price": money(item.unit_price),
                        "amount": money(item.amount),
                        "note": item.note or "",
                    }
                    for item in draft.line_items
                ],
            }
        )
    else:
        context.update(
            {
                "total": "-",
                "invoice_date": "-",
                "due_date": "-",
                "terms": "-",
                "confidence": "-",
                "line_items": [],
            }
        )

    if approval is not None:
        context.update(
            {
                "decision": approval.decision.value,
                "decision_class": approval.decision.value.lower(),
                "decision_label": _DECISION_LABEL[approval.decision],
                "rationale": approval.rationale,
                "key_factors": approval.key_factors,
                "conditions": approval.conditions,
                "scrutiny": approval.scrutiny_level,
                "policy_reasons": approval.policy_reasons,
                "rounds": [
                    {
                        "index": round_.round_index + 1,
                        "decision": round_.draft.decision.value,
                        "rationale": round_.draft.rationale,
                        "critique_available": round_.critique is not None,
                        "agrees": round_.critique.agrees if round_.critique else False,
                        "objections": round_.critique.objections if round_.critique else [],
                        "suggested": (
                            round_.critique.suggested_decision.value
                            if round_.critique and round_.critique.suggested_decision
                            else None
                        ),
                        "revised": round_.revised,
                        "overridden_from": (
                            round_.overridden_from.value if round_.overridden_from else None
                        ),
                    }
                    for round_ in approval.rounds
                ],
            }
        )
    else:
        context.update(
            {
                "decision": "ERROR",
                "decision_class": "rejected",
                "decision_label": "Run failed",
                "rationale": state.error or "The pipeline did not reach a decision.",
                "key_factors": [],
                "conditions": [],
                "scrutiny": "n/a",
                "policy_reasons": [],
                "rounds": [],
            }
        )

    if report is not None:
        context.update(
            {
                "findings": [
                    {
                        "code": finding.code.value,
                        "severity": finding.severity.value,
                        "message": finding.message,
                    }
                    for finding in report.findings
                ],
                "critical_count": len(report.critical),
                "warning_count": len(report.warnings),
                "agent_summary": report.agent_summary,
                "item_checks": [
                    {
                        "invoice_name": check.invoice_name,
                        "matched": check.matched_item or "-",
                        "billed": f"{check.quantity_requested:g}",
                        "stock": "-" if check.stock_available is None else str(check.stock_available),
                        "status": ITEM_STATUS_LABEL.get(check.status, check.status),
                        "pill": "ok" if check.status == "ok" else "bad",
                    }
                    for check in report.item_checks
                ],
            }
        )
        aggregated = draft.aggregated_quantities() if draft else {}
        if draft and len(aggregated) < len(draft.line_items):
            context["aggregation_note"] = (
                f"{len(draft.line_items)} line items were aggregated into "
                f"{len(aggregated)} distinct products before checking stock. "
                "Checking each line independently would understate the quantities ordered."
            )
    else:
        context.update(
            {
                "findings": [],
                "critical_count": 0,
                "warning_count": 0,
                "agent_summary": None,
                "item_checks": [],
            }
        )

    payment = state.payment
    context.update(
        {
            "payment_status": payment.status.upper() if payment else "NOT ATTEMPTED",
            "payment_pill": _PAYMENT_PILL.get(payment.status, "warn") if payment else "warn",
            "payment_message": payment.message if payment else "The run did not reach the payment stage.",
            "payment_reference": payment.reference if payment else None,
        }
    )

    context["extraction_attempts"] = len(state.extraction_attempts)
    context["attempts"] = [
        {
            "round": attempt.round_index + 1,
            "issue_count": len(attempt.issues),
            "issues": "; ".join(attempt.issues) if attempt.issues else "reconciled",
        }
        for attempt in state.extraction_attempts
    ]

    stats = []
    if state.duration_seconds is not None:
        stats.append(f"{state.duration_seconds:.1f}s")
    if usage:
        stats.append(f"{usage['calls']} LLM calls")
        if usage.get("total_tokens"):
            stats.append(f"{usage['total_tokens']:,} tokens")
    # Plain separator: Jinja autoescaping would render an HTML entity literally.
    context["stats"] = " · ".join(stats) if stats else ""

    return context


def render_report(state: RunState, usage: dict[str, int] | None = None) -> str:
    """Render the report to an HTML string."""
    template = _environment().get_template(TEMPLATE_NAME)
    return template.render(**build_context(state, usage))


def write_report(state: RunState, path: Path, usage: dict[str, int] | None = None) -> Path:
    """Render and write the report, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(state, usage), encoding="utf-8")
    return path
