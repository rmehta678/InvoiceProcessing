"""Ingestion agent: document in, structured draft out.

Exposed as individual steps so the repair loop can live in the graph as a real
cycle (`extract -> verify -> extract`) rather than being buried in a `for`
statement here. The graph is the only orchestration path; these functions are
its vocabulary.

The loop is grounded in arithmetic rather than a second model opinion. After
each extraction the draft is reconciled against itself -- line amounts,
subtotal, tax, total -- and any inconsistency goes back to the model as a
specific, checkable complaint.

Crucially the loop must be able to *stop being wrong about being wrong*.
Invoices 1009 and 1013 genuinely do not add up, and no amount of re-reading
fixes them. Two guards handle that: the repair prompt tells the model to
re-transcribe unchanged figures when the invoice itself is defective, and
identical consecutive drafts end the loop early. Without them a broken invoice
burns every retry and still arrives with the same numbers.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..llm.base import LLMClient
from ..llm.prompts import EXTRACTION_SYSTEM, extraction_repair_prompt, extraction_user_prompt
from ..models import ExtractedInvoice, ExtractionAttempt, InvoiceDraft, SourceDocument
from ..tools.arithmetic import (
    check_arithmetic,
    check_data_integrity,
    reconstruct_total,
    summarise_for_repair,
)
from ..tools.dates import resolve_dates

# Confidence assigned to a draft depending on how much repair it needed.
CONFIDENCE_CLEAN = 1.0
CONFIDENCE_REPAIRED = 0.85
CONFIDENCE_UNRESOLVED = 0.6


def fingerprint(draft: InvoiceDraft) -> str:
    """Comparable signature of the numbers that matter for reconciliation."""
    items = [
        (item.normalized_name, item.quantity, item.unit_price, item.amount)
        for item in draft.line_items
    ]
    return repr((items, draft.subtotal, draft.tax_amount, draft.shipping, draft.total))


def build_messages(
    document: SourceDocument, attempts: list[ExtractionAttempt]
) -> list[dict[str, Any]]:
    """First-pass prompt, or a repair prompt citing the previous attempt's defects."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": extraction_user_prompt(document)},
    ]
    if attempts:
        previous = attempts[-1]
        messages += [
            {"role": "assistant", "content": previous.draft.model_dump_json()},
            {
                "role": "user",
                "content": extraction_repair_prompt(document, previous.issues),
            },
        ]
    return messages


def extract_once(
    document: SourceDocument,
    client: LLMClient,
    attempts: list[ExtractionAttempt],
) -> InvoiceDraft:
    """Run one extraction pass and resolve its dates."""
    extracted = client.complete_structured(
        build_messages(document, attempts),
        ExtractedInvoice,
        agent=f"ingestion.extract.r{len(attempts)}",
    )
    return resolve_dates(extracted.to_draft())


def verify_extraction(draft: InvoiceDraft) -> list[str]:
    """Machine-checkable defects in a draft, phrased as repair instructions."""
    return summarise_for_repair(check_arithmetic(draft) + check_data_integrity(draft))


def assess_attempt(
    draft: InvoiceDraft,
    attempts: list[ExtractionAttempt],
    settings: Settings,
) -> ExtractionAttempt:
    """Decide whether this attempt ends the loop, and record it."""
    issues = verify_extraction(draft)
    round_index = len(attempts)
    converged = bool(attempts) and fingerprint(draft) == fingerprint(attempts[-1].draft)
    accepted = not issues or converged or round_index >= settings.max_extraction_repairs

    return ExtractionAttempt(
        round_index=round_index,
        draft=draft,
        issues=issues,
        accepted=accepted,
    )


def finalise_confidence(attempt: ExtractionAttempt) -> InvoiceDraft:
    """Settle an accepted draft: reconstruct a missing total, set confidence."""
    draft = attempt.draft

    # Only now that the loop has stopped -- see `reconstruct_total` for why the
    # ordering matters.
    reconstruct_total(draft)

    if not attempt.issues:
        draft.extraction_confidence = (
            CONFIDENCE_CLEAN if attempt.round_index == 0 else CONFIDENCE_REPAIRED
        )
    else:
        # Issues survived every repair attempt. That is a statement about the
        # invoice, not only about the extraction -- but it still lowers our
        # confidence in the numbers being passed downstream.
        draft.extraction_confidence = CONFIDENCE_UNRESOLVED
    return draft
