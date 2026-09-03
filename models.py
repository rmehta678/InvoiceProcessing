from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class IssueCode(str, Enum):
    UNREADABLE = "unreadable"
    MISSING_VENDOR = "missing_vendor"
    NONPOSITIVE_AMOUNT = "nonpositive_amount"
    EMPTY_ITEMS = "empty_items"
    UNKNOWN_ITEM = "unknown_item"
    OUT_OF_STOCK = "out_of_stock"
    INSUFFICIENT_STOCK = "insufficient_stock"
    INVALID_QTY = "invalid_qty"


class LineItem(BaseModel):
    name: str
    quantity: int


class Invoice(BaseModel):
    vendor: str
    amount: Decimal
    items: List[LineItem]
    due_date: Optional[date] = None


class Issue(BaseModel):
    code: IssueCode
    detail: str
    item: Optional[str] = None


class Review(BaseModel):
    recommendation: Literal["approve", "reject", "escalate"]
    confidence: float = Field(ge=0, le=1)
    flags: List[str] = Field(default_factory=list)
    reason: str


class PaymentResult(BaseModel):
    success: bool
    message: str


class Event(BaseModel):
    node: str
    message: str


class ExtractedLineItem(BaseModel):
    name: str = Field(description="Line item name, normalized (e.g. WidgetA not Widget A)")
    quantity: int = Field(description="Requested quantity, may be negative if the source is invalid")


class ExtractedInvoice(BaseModel):
    vendor: str = Field(description="Vendor / supplier name")
    amount: float = Field(description="Total amount due")
    items: List[ExtractedLineItem] = Field(description="Line items")
    due_date: Optional[str] = Field(default=None, description="Due date as YYYY-MM-DD, or null if unknown")


class RemapResult(BaseModel):
    items: List[ExtractedLineItem] = Field(description="Line items after name correction")
    notes: str = Field(description="What was remapped, or 'none' if unchanged")
