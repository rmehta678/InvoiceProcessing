from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional

from typing_extensions import NotRequired, TypedDict

from models import Event, Invoice, Issue, PaymentResult, Review


class InputState(TypedDict):
    invoice_path: str


class OutputState(TypedDict, total=False):
    invoice: Optional[Invoice]
    issues: List[Issue]
    review: Optional[Review]
    challenge: Optional[Review]
    payment: Optional[PaymentResult]
    outcome: Literal["paid", "rejected", "needs_review"]
    reason: str


class InvoiceState(TypedDict):
    invoice_path: str
    invoice: NotRequired[Optional[Invoice]]
    source_text: NotRequired[str]
    issues: NotRequired[List[Issue]]
    review: NotRequired[Optional[Review]]
    challenge: NotRequired[Optional[Review]]
    payment: NotRequired[Optional[PaymentResult]]
    outcome: NotRequired[Literal["paid", "rejected", "needs_review"]]
    reason: NotRequired[str]
    correction_passes: NotRequired[int]
    review_round: NotRequired[int]
    events: Annotated[List[Event], operator.add]
