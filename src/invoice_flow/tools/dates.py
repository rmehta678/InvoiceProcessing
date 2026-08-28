"""Tolerant date parsing and due-date sanity rules.

The sample invoices write dates six different ways -- ISO, ``Jan 30 2026``,
``January 27, 2026``, ``01/28/2026``, ``26-Jan-2026``, and the OCR-damaged
``26-Jan-2O26`` where a capital O stands in for a zero. One of them says
``yesterday``, which is not a date at all and should be treated as a signal
rather than quietly coerced into one.

Sanity checks anchor on the *invoice date*, not today's date. The sample set is
historical, so anchoring on today would flag all 18 invoices as overdue and
teach the approval agent nothing.
"""

from __future__ import annotations

import re
from datetime import date

from dateutil import parser as dateutil_parser

from ..models import Finding, FindingCode, InvoiceDraft, Severity

# Characters an OCR pass commonly substitutes for digits.
_OCR_DIGIT_MAP = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})

# A token worth repairing: has at least one real digit, and every other
# character is a known OCR stand-in. "2O26" qualifies; "Jan" does not.
_OCR_TOKEN = re.compile(r"^(?=[^0-9]*[0-9])[0-9OolI|]+$")


def repair_ocr_digits(text: str) -> str:
    """Fix letter-for-digit OCR substitutions inside numeric tokens only.

    Applied token-wise so month names survive: ``26-Jan-2O26`` becomes
    ``26-Jan-2026`` without ``Jan`` being touched.
    """
    parts = re.split(r"([^A-Za-z0-9|]+)", text)
    return "".join(p.translate(_OCR_DIGIT_MAP) if _OCR_TOKEN.match(p) else p for p in parts)


def parse_date(raw: str | None) -> tuple[date | None, str | None]:
    """Parse a date string leniently.

    Returns ``(date, note)``. ``note`` explains why parsing failed, or records
    that an OCR repair was needed to succeed.
    """
    if raw is None:
        return None, "missing"
    text = raw.strip().strip(".,")
    if not text:
        return None, "missing"

    for candidate, note in ((text, None), (repair_ocr_digits(text), "ocr_repaired")):
        try:
            # fuzzy=False so prose like "yesterday" or "on receipt" is rejected
            # rather than silently resolved against the current clock.
            return dateutil_parser.parse(candidate, fuzzy=False).date(), note
        except (ValueError, OverflowError, TypeError):
            continue

    return None, "unparseable"


def resolve_dates(draft: InvoiceDraft) -> InvoiceDraft:
    """Populate `invoice_date` / `due_date` on a draft from their raw strings."""
    draft.invoice_date, _ = parse_date(draft.invoice_date_raw)
    draft.due_date, _ = parse_date(draft.due_date_raw)
    return draft


def check_dates(draft: InvoiceDraft) -> list[Finding]:
    """Apply due-date rules, returning findings for anything questionable.

    Reads the dates `resolve_dates` already put on the draft rather than
    parsing the raw strings a second time -- two parsers on one string is two
    places for the rules to drift apart.
    """
    findings: list[Finding] = []
    raw = (draft.due_date_raw or "").strip()
    due = draft.due_date
    invoice_date = draft.invoice_date

    if not raw:
        findings.append(
            Finding(
                code=FindingCode.DUE_DATE_MISSING,
                severity=Severity.WARNING,
                message="No due date stated; payment timing cannot be scheduled.",
                source="dates",
            )
        )
    elif due is None:
        findings.append(
            Finding(
                code=FindingCode.DUE_DATE_UNPARSEABLE,
                severity=Severity.WARNING,
                message=(
                    f'Due date "{raw}" is not a date. A vendor who cannot state a '
                    "concrete due date is either careless or applying time pressure."
                ),
                detail={"raw": raw, "reason": "unparseable"},
                source="dates",
            )
        )
    elif invoice_date is not None:
        if due < invoice_date:
            findings.append(
                Finding(
                    code=FindingCode.DUE_DATE_BEFORE_INVOICE_DATE,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Due date {due.isoformat()} precedes the invoice date "
                        f"{invoice_date.isoformat()} -- the invoice was overdue "
                        "before it was issued."
                    ),
                    detail={"due_date": due.isoformat(), "invoice_date": invoice_date.isoformat()},
                    source="dates",
                )
            )
        elif due == invoice_date:
            findings.append(
                Finding(
                    code=FindingCode.DUE_DATE_NOT_AFTER_INVOICE_DATE,
                    severity=Severity.WARNING,
                    message=(
                        f"Due date equals the invoice date ({due.isoformat()}), leaving "
                        "no payment window. This contradicts any stated net terms."
                    ),
                    detail={"due_date": due.isoformat()},
                    source="dates",
                )
            )

    return findings
