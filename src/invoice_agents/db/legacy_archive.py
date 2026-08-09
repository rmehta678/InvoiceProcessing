"""Canonical hashing and verification for permanently non-authorizing legacy rows."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

LEGACY_NON_AUTHORIZING_DISPOSITION = "PERMANENTLY_NON_AUTHORIZING"
LEGACY_RECORD_HASH_DOMAIN = b"galatiq.invoice-agents/legacy-authorization-record/v1\x00"
LEGACY_MANIFEST_HASH_DOMAIN = b"galatiq.invoice-agents/legacy-authorization-manifest/v1\x00"
LEGACY_RECONCILIATION_ID_DOMAIN = b"galatiq.invoice-agents/legacy-reconciliation/v1\x00"
LEGACY_ACTIVE_TABLE_KEYS: dict[str, str] = {
    "review_requests": "review_id",
    "human_decisions": "decision_id",
    "final_decisions": "decision_id",
    "payments": "payment_id",
}


def _parse_canonical_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed if parsed.isoformat() == value else None


def canonical_legacy_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def legacy_record_hash(table: str, record_key: str, original_row_json: str) -> str:
    material = (
        LEGACY_RECORD_HASH_DOMAIN
        + table.encode()
        + b"\x00"
        + record_key.encode()
        + b"\x00"
        + original_row_json.encode()
    )
    return hashlib.sha256(material).hexdigest()


def legacy_manifest_hash(records: Sequence[tuple[str, str, str]]) -> str:
    material = b"\n".join(
        table.encode() + b"\x00" + key.encode() + b"\x00" + record_hash.encode()
        for table, key, record_hash in sorted(records)
    )
    return hashlib.sha256(LEGACY_MANIFEST_HASH_DOMAIN + material).hexdigest()


def legacy_reconciliation_id(
    *,
    reviewer: str,
    reason: str,
    disposition: str,
    source_schema_version: int,
    manifest_hash: str,
) -> str:
    material = canonical_legacy_json(
        {
            "disposition": disposition,
            "manifest": manifest_hash,
            "reason": reason,
            "reviewer": reviewer,
            "source_schema_version": source_schema_version,
        }
    ).encode()
    return "lrec_" + hashlib.sha256(LEGACY_RECONCILIATION_ID_DOMAIN + material).hexdigest()


def _strict_json_object(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("archive JSON is not text")
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or canonical_legacy_json(parsed) != value:
        raise ValueError("archive JSON is not a canonical object")
    return parsed


def audit_legacy_authorization_archives(connection: sqlite3.Connection) -> int:
    """Return one invalid count per reconciliation group; never authorize archived rows."""

    metadata_rows = connection.execute(
        "SELECT reconciliation_id, reviewer, reason, disposition, confirmed_at, "
        "source_schema_version, record_count, record_manifest_hash, table_counts_json, state "
        "FROM legacy_authorization_reconciliations ORDER BY reconciliation_id"
    ).fetchall()
    archive_rows = connection.execute(
        "SELECT archive_id, reconciliation_id, source_table, source_record_key, "
        "original_row_json, record_hash, authorization_state, archived_at "
        "FROM legacy_authorization_quarantine ORDER BY reconciliation_id, source_table, "
        "source_record_key"
    ).fetchall()
    metadata_by_id = {row["reconciliation_id"]: row for row in metadata_rows}
    rows_by_id: dict[object, list[sqlite3.Row]] = {}
    for row in archive_rows:
        rows_by_id.setdefault(row["reconciliation_id"], []).append(row)
    reconciliation_ids = set(metadata_by_id) | set(rows_by_id)
    invalid = 0
    for reconciliation_id in reconciliation_ids:
        metadata = metadata_by_id.get(reconciliation_id)
        rows = rows_by_id.get(reconciliation_id, [])
        try:
            if metadata is None:
                raise ValueError("archive rows have no reconciliation metadata")
            if (
                not isinstance(reconciliation_id, str)
                or not reconciliation_id
                or not isinstance(metadata["reviewer"], str)
                or not str(metadata["reviewer"]).strip()
                or not isinstance(metadata["reason"], str)
                or not str(metadata["reason"]).strip()
                or metadata["disposition"] != LEGACY_NON_AUTHORIZING_DISPOSITION
                or metadata["state"] != "COMPLETED"
                or _parse_canonical_utc(metadata["confirmed_at"]) is None
                or type(metadata["source_schema_version"]) is not int
                or metadata["source_schema_version"] not in (1, 2)
                or type(metadata["record_count"]) is not int
                or metadata["record_count"] != len(rows)
            ):
                raise ValueError("legacy reconciliation metadata is invalid")
            table_counts = _strict_json_object(metadata["table_counts_json"])
            actual_counts = {table: 0 for table in LEGACY_ACTIVE_TABLE_KEYS}
            manifest_records: list[tuple[str, str, str]] = []
            for row in rows:
                table = row["source_table"]
                key = row["source_record_key"]
                if (
                    table not in LEGACY_ACTIVE_TABLE_KEYS
                    or not isinstance(key, str)
                    or not key
                    or row["authorization_state"] != LEGACY_NON_AUTHORIZING_DISPOSITION
                    or _parse_canonical_utc(row["archived_at"]) is None
                    or row["archived_at"] != metadata["confirmed_at"]
                ):
                    raise ValueError("legacy archive row metadata is invalid")
                original = _strict_json_object(row["original_row_json"])
                if str(original.get(LEGACY_ACTIVE_TABLE_KEYS[str(table)])) != key:
                    raise ValueError("legacy archive key does not match original row")
                record_hash = legacy_record_hash(str(table), key, row["original_row_json"])
                if row["record_hash"] != record_hash or row["archive_id"] != f"lqar_{record_hash}":
                    raise ValueError("legacy archive record hash is invalid")
                actual_counts[str(table)] += 1
                manifest_records.append((str(table), key, record_hash))
            if (
                set(table_counts) != set(LEGACY_ACTIVE_TABLE_KEYS)
                or any(type(value) is not int or value < 0 for value in table_counts.values())
                or table_counts != actual_counts
            ):
                raise ValueError("legacy archive table counts are invalid")
            manifest_hash = legacy_manifest_hash(manifest_records)
            if metadata["record_manifest_hash"] != manifest_hash:
                raise ValueError("legacy archive manifest hash is invalid")
            if reconciliation_id != legacy_reconciliation_id(
                reviewer=metadata["reviewer"],
                reason=metadata["reason"],
                disposition=metadata["disposition"],
                source_schema_version=metadata["source_schema_version"],
                manifest_hash=manifest_hash,
            ):
                raise ValueError("legacy reconciliation identity is invalid")
        except (KeyError, TypeError, ValueError):
            invalid += 1
    return invalid
