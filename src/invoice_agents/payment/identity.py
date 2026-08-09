"""Shared deterministic payment identity for Python and SQLite constraints."""

from __future__ import annotations

import hashlib


def payment_identity_key(vendor: object, invoice_number: object) -> str:
    material = "|".join(
        [
            str(vendor or "").casefold().strip(),
            str(invoice_number or "").casefold().strip(),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
