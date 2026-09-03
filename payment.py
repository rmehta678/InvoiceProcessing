from __future__ import annotations

from decimal import Decimal

from models import PaymentResult


def mock_payment(vendor: str, amount: Decimal) -> PaymentResult:
    if not vendor.strip():
        return PaymentResult(success=False, message="Missing vendor name")
    if amount <= 0:
        return PaymentResult(success=False, message=f"Invalid amount: {amount}")
    print(f"Paid {amount} to {vendor}")
    return PaymentResult(success=True, message=f"Paid {amount} to {vendor}")
