"""Exact whole-database archival plus supplemental typed legacy-row inspection."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, cast

LEGACY_NON_AUTHORIZING_DISPOSITION = "PERMANENTLY_NON_AUTHORIZING"
LEGACY_SCHEMA_HASH_DOMAIN = b"galatiq.invoice-agents/legacy-authorization-schema/v2\x00"
LEGACY_RECORD_HASH_DOMAIN = b"galatiq.invoice-agents/legacy-authorization-record/v2\x00"
LEGACY_SCHEMA_MANIFEST_HASH_DOMAIN = (
    b"galatiq.invoice-agents/legacy-authorization-schema-manifest/v2\x00"
)
LEGACY_MANIFEST_HASH_DOMAIN = b"galatiq.invoice-agents/legacy-authorization-manifest/v2\x00"
LEGACY_RECONCILIATION_ID_DOMAIN = b"galatiq.invoice-agents/legacy-reconciliation/v3\x00"
LEGACY_COLUMN_MANIFEST_DOMAIN = b"galatiq.invoice-agents/legacy-columns/v2\x00"
LEGACY_TYPED_ROW_DOMAIN = b"galatiq.invoice-agents/legacy-row/v2\x00"
LEGACY_ACTIVE_TABLE_KEYS: dict[str, str] = {
    "review_requests": "review_id",
    "human_decisions": "decision_id",
    "final_decisions": "decision_id",
    "payments": "payment_id",
}

StorageClass = Literal["NULL", "INTEGER", "REAL", "TEXT", "BLOB"]


@dataclass(frozen=True, slots=True)
class LegacyColumnMetadata:
    cid: int
    name: str
    declared_type: str
    not_null: int
    default_sql: str | None
    primary_key_position: int


@dataclass(frozen=True, slots=True)
class LegacyCell:
    storage_class: StorageClass
    encoded: bytes

    def value(self) -> object:
        if self.storage_class == "NULL":
            return None
        if self.storage_class == "INTEGER":
            return int(self.encoded.decode("ascii"))
        if self.storage_class == "REAL":
            return struct.unpack(">d", self.encoded)[0]
        if self.storage_class == "TEXT":
            return self.encoded.decode("utf-8")
        return self.encoded


@dataclass(frozen=True, slots=True)
class LegacyDecodedRow:
    source_row_ordinal: int
    source_rowid: int | None
    cells: tuple[LegacyCell, ...]
    typed_row: bytes


@dataclass(frozen=True, slots=True)
class LegacyDecodedTable:
    source_table_order: int
    source_table: str
    source_table_sql: str
    columns: tuple[LegacyColumnMetadata, ...]
    column_manifest: bytes
    schema_hash: str
    rows: tuple[LegacyDecodedRow, ...]


@dataclass(frozen=True, slots=True)
class LegacyDatabaseSnapshotFacts:
    sha256: str
    size_bytes: int
    source_schema_version: int
    page_size: int
    page_count: int
    active_table_counts: dict[str, int]


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
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: object) -> bytes:
    if not isinstance(value, str):
        raise ValueError("typed archive byte encoding is not text")
    return base64.b64decode(value, validate=True)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _encode_column_manifest(columns: tuple[LegacyColumnMetadata, ...]) -> bytes:
    payload = [
        {
            "cid": column.cid,
            "default_sql_utf8": (
                _encode_bytes(column.default_sql.encode("utf-8"))
                if column.default_sql is not None
                else None
            ),
            "declared_type_utf8": _encode_bytes(column.declared_type.encode("utf-8")),
            "name_utf8": _encode_bytes(column.name.encode("utf-8")),
            "not_null": column.not_null,
            "primary_key_position": column.primary_key_position,
        }
        for column in columns
    ]
    return LEGACY_COLUMN_MANIFEST_DOMAIN + _canonical_bytes(payload)


def _decode_column_manifest(value: bytes) -> tuple[LegacyColumnMetadata, ...]:
    if not value.startswith(LEGACY_COLUMN_MANIFEST_DOMAIN):
        raise ValueError("legacy column manifest domain is invalid")
    encoded = value[len(LEGACY_COLUMN_MANIFEST_DOMAIN) :]
    payload = json.loads(encoded)
    if not isinstance(payload, list) or _canonical_bytes(payload) != encoded:
        raise ValueError("legacy column manifest is not canonical")
    columns: list[LegacyColumnMetadata] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != {
            "cid",
            "default_sql_utf8",
            "declared_type_utf8",
            "name_utf8",
            "not_null",
            "primary_key_position",
        }:
            raise ValueError("legacy column metadata shape is invalid")
        if (
            type(item["cid"]) is not int
            or item["cid"] != index
            or type(item["not_null"]) is not int
            or item["not_null"] not in (0, 1)
            or type(item["primary_key_position"]) is not int
            or item["primary_key_position"] < 0
        ):
            raise ValueError("legacy column metadata values are invalid")
        default_raw = item["default_sql_utf8"]
        columns.append(
            LegacyColumnMetadata(
                cid=item["cid"],
                name=_decode_bytes(item["name_utf8"]).decode("utf-8"),
                declared_type=_decode_bytes(item["declared_type_utf8"]).decode("utf-8"),
                not_null=item["not_null"],
                default_sql=(
                    _decode_bytes(default_raw).decode("utf-8") if default_raw is not None else None
                ),
                primary_key_position=item["primary_key_position"],
            )
        )
    return tuple(columns)


def legacy_schema_hash(
    table: str,
    source_table_sql: bytes,
    column_manifest: bytes,
) -> str:
    material = (
        LEGACY_SCHEMA_HASH_DOMAIN
        + len(table.encode("utf-8")).to_bytes(4, "big")
        + table.encode("utf-8")
        + len(source_table_sql).to_bytes(8, "big")
        + source_table_sql
        + len(column_manifest).to_bytes(8, "big")
        + column_manifest
    )
    return hashlib.sha256(material).hexdigest()


def _cell_from_sqlite(storage: str, value: object) -> LegacyCell:
    if storage == "null":
        return LegacyCell("NULL", b"")
    if storage == "integer" and type(value) is int:
        return LegacyCell("INTEGER", str(value).encode("ascii"))
    if storage == "real" and isinstance(value, float):
        return LegacyCell("REAL", struct.pack(">d", value))
    if storage == "text" and isinstance(value, bytes):
        return LegacyCell("TEXT", value)
    if storage == "blob" and isinstance(value, bytes):
        return LegacyCell("BLOB", value)
    raise ValueError("SQLite value and storage class are inconsistent")


def _encode_typed_row(
    ordinal: int,
    rowid: int | None,
    cells: tuple[LegacyCell, ...],
) -> bytes:
    payload = {
        "cells": [
            {"data": _encode_bytes(cell.encoded), "storage_class": cell.storage_class}
            for cell in cells
        ],
        "source_row_ordinal": ordinal,
        "source_rowid": str(rowid) if rowid is not None else None,
    }
    return LEGACY_TYPED_ROW_DOMAIN + _canonical_bytes(payload)


def _decode_typed_row(value: bytes, column_count: int) -> LegacyDecodedRow:
    if not value.startswith(LEGACY_TYPED_ROW_DOMAIN):
        raise ValueError("legacy typed row domain is invalid")
    encoded = value[len(LEGACY_TYPED_ROW_DOMAIN) :]
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or _canonical_bytes(payload) != encoded:
        raise ValueError("legacy typed row is not canonical")
    if set(payload) != {"cells", "source_row_ordinal", "source_rowid"}:
        raise ValueError("legacy typed row shape is invalid")
    ordinal = payload["source_row_ordinal"]
    raw_rowid = payload["source_rowid"]
    raw_cells = payload["cells"]
    if type(ordinal) is not int or ordinal < 0 or not isinstance(raw_cells, list):
        raise ValueError("legacy typed row identity is invalid")
    if raw_rowid is None:
        rowid = None
    elif isinstance(raw_rowid, str):
        rowid = int(raw_rowid)
        if str(rowid) != raw_rowid:
            raise ValueError("legacy typed rowid is not canonical")
    else:
        raise ValueError("legacy typed rowid has the wrong type")
    cells: list[LegacyCell] = []
    allowed: set[str] = {"NULL", "INTEGER", "REAL", "TEXT", "BLOB"}
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict) or set(raw_cell) != {"data", "storage_class"}:
            raise ValueError("legacy typed cell shape is invalid")
        storage = raw_cell["storage_class"]
        if storage not in allowed:
            raise ValueError("legacy typed cell storage class is invalid")
        data = _decode_bytes(raw_cell["data"])
        cell = LegacyCell(cast(StorageClass, storage), data)
        if storage == "NULL" and data:
            raise ValueError("legacy NULL cell contains bytes")
        if storage == "REAL" and len(data) != 8:
            raise ValueError("legacy REAL cell is not IEEE-754 binary64")
        if storage == "INTEGER":
            decoded = cell.value()
            if type(decoded) is not int or str(decoded).encode() != data:
                raise ValueError("legacy INTEGER cell is not canonical")
        cells.append(cell)
    if len(cells) != column_count:
        raise ValueError("legacy typed row column count is invalid")
    return LegacyDecodedRow(ordinal, rowid, tuple(cells), value)


def _rowid_reference(connection: sqlite3.Connection, table: str, columns: set[str]) -> str | None:
    candidates = ("rowid", "_rowid_", "oid")
    candidate = next((name for name in candidates if name.casefold() not in columns), None)
    if candidate is None:
        return None
    try:
        connection.execute(
            f"SELECT {_quote_identifier(candidate)} FROM {_quote_identifier(table)} LIMIT 0"
        )
    except sqlite3.OperationalError:
        return None
    return candidate


def _capture_table(
    connection: sqlite3.Connection,
    table_order: int,
    table: str,
) -> LegacyDecodedTable:
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if table_row is None or table_row[0] is None or not isinstance(table_row[0], str):
        raise ValueError(f"legacy source table is missing: {table}")
    source_table_sql = str(table_row[0])
    columns = tuple(
        LegacyColumnMetadata(
            cid=int(row[0]),
            name=str(row[1]),
            declared_type=str(row[2]),
            not_null=int(row[3]),
            default_sql=str(row[4]) if row[4] is not None else None,
            primary_key_position=int(row[5]),
        )
        for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    )
    if not columns or tuple(column.cid for column in columns) != tuple(range(len(columns))):
        raise ValueError("legacy source columns are not ordered by cid")
    key_column = LEGACY_ACTIVE_TABLE_KEYS[table]
    if key_column not in {column.name for column in columns}:
        raise ValueError(f"legacy source table has no identity column: {table}")
    column_manifest = _encode_column_manifest(columns)
    schema_hash = legacy_schema_hash(
        table,
        source_table_sql.encode("utf-8"),
        column_manifest,
    )
    rowid_reference = _rowid_reference(
        connection,
        table,
        {column.name.casefold() for column in columns},
    )
    quoted_columns = [_quote_identifier(column.name) for column in columns]
    select_parts: list[str] = []
    if rowid_reference is not None:
        select_parts.append(_quote_identifier(rowid_reference))
    for quoted in quoted_columns:
        select_parts.extend((quoted, f"typeof({quoted})"))
    order_sql = f" ORDER BY {_quote_identifier(rowid_reference)}" if rowid_reference else ""
    previous_text_factory = connection.text_factory
    try:
        connection.text_factory = bytes
        raw_rows = connection.execute(
            f"SELECT {', '.join(select_parts)} FROM {_quote_identifier(table)}{order_sql}"
        ).fetchall()
    finally:
        connection.text_factory = previous_text_factory
    rows: list[LegacyDecodedRow] = []
    for ordinal, raw in enumerate(raw_rows):
        offset = 0
        source_rowid: int | None = None
        if rowid_reference is not None:
            if type(raw[0]) is not int:
                raise ValueError("legacy source rowid is not an integer")
            source_rowid = raw[0]
            offset = 1
        cells: list[LegacyCell] = []
        for index in range(len(columns)):
            value = raw[offset + index * 2]
            raw_storage = raw[offset + index * 2 + 1]
            storage = raw_storage.decode("ascii") if isinstance(raw_storage, bytes) else raw_storage
            if not isinstance(storage, str):
                raise ValueError("legacy SQLite storage class is invalid")
            cells.append(_cell_from_sqlite(storage, value))
        typed_row = _encode_typed_row(ordinal, source_rowid, tuple(cells))
        rows.append(LegacyDecodedRow(ordinal, source_rowid, tuple(cells), typed_row))
    return LegacyDecodedTable(
        source_table_order=table_order,
        source_table=table,
        source_table_sql=source_table_sql,
        columns=columns,
        column_manifest=column_manifest,
        schema_hash=schema_hash,
        rows=tuple(rows),
    )


def capture_legacy_authorization_tables(
    connection: sqlite3.Connection,
) -> tuple[LegacyDecodedTable, ...]:
    """Capture exact source schemas and rows without parsing any application payload."""

    return tuple(
        _capture_table(connection, order, table)
        for order, table in enumerate(LEGACY_ACTIVE_TABLE_KEYS)
    )


def legacy_json_safe_row(
    table: LegacyDecodedTable,
    row: LegacyDecodedRow,
) -> str | None:
    values: dict[str, Any] = {}
    for column, cell in zip(table.columns, row.cells, strict=True):
        if cell.storage_class == "BLOB":
            return None
        try:
            values[column.name] = cell.value()
        except UnicodeDecodeError:
            return None
    try:
        return canonical_legacy_json(values)
    except (TypeError, ValueError):
        return None


def legacy_source_record_key(table: LegacyDecodedTable, row: LegacyDecodedRow) -> str:
    key_name = LEGACY_ACTIVE_TABLE_KEYS[table.source_table]
    key_index = next(index for index, column in enumerate(table.columns) if column.name == key_name)
    cell = row.cells[key_index]
    if cell.storage_class == "TEXT":
        try:
            return cell.encoded.decode("utf-8")
        except UnicodeDecodeError:
            return f"text:{_encode_bytes(cell.encoded)}"
    if cell.storage_class == "INTEGER":
        return cell.encoded.decode("ascii")
    return f"{cell.storage_class.lower()}:{_encode_bytes(cell.encoded)}"


def legacy_record_hash(table: str, schema_hash: str, typed_row: bytes) -> str:
    material = (
        LEGACY_RECORD_HASH_DOMAIN
        + len(table.encode("utf-8")).to_bytes(4, "big")
        + table.encode("utf-8")
        + bytes.fromhex(schema_hash)
        + len(typed_row).to_bytes(8, "big")
        + typed_row
    )
    return hashlib.sha256(material).hexdigest()


def legacy_schema_manifest_hash(tables: Sequence[LegacyDecodedTable]) -> str:
    material = b"".join(
        table.source_table_order.to_bytes(4, "big")
        + len(table.source_table.encode("utf-8")).to_bytes(4, "big")
        + table.source_table.encode("utf-8")
        + bytes.fromhex(table.schema_hash)
        + len(table.rows).to_bytes(8, "big")
        for table in tables
    )
    return hashlib.sha256(LEGACY_SCHEMA_MANIFEST_HASH_DOMAIN + material).hexdigest()


def legacy_manifest_hash(
    schema_manifest_hash: str,
    records: Sequence[tuple[str, int, str]],
) -> str:
    material = bytes.fromhex(schema_manifest_hash) + b"".join(
        len(table.encode("utf-8")).to_bytes(4, "big")
        + table.encode("utf-8")
        + ordinal.to_bytes(8, "big")
        + bytes.fromhex(record_hash)
        for table, ordinal, record_hash in records
    )
    return hashlib.sha256(LEGACY_MANIFEST_HASH_DOMAIN + material).hexdigest()


def legacy_reconciliation_id(
    *,
    reviewer: str,
    reason: str,
    disposition: str,
    source_schema_version: int,
    manifest_hash: str,
    database_snapshot_hash: str,
) -> str:
    material = canonical_legacy_json(
        {
            "disposition": disposition,
            "database_snapshot": database_snapshot_hash,
            "manifest": manifest_hash,
            "reason": reason,
            "reviewer": reviewer,
            "source_schema_version": source_schema_version,
        }
    ).encode("utf-8")
    return "lrec_" + hashlib.sha256(LEGACY_RECONCILIATION_ID_DOMAIN + material).hexdigest()


def _strict_json_object(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("archive JSON is not text")
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or canonical_legacy_json(parsed) != value:
        raise ValueError("archive JSON is not a canonical object")
    return parsed


def _validate_active_table_counts(value: Mapping[str, object]) -> dict[str, int]:
    if set(value) != set(LEGACY_ACTIVE_TABLE_KEYS) or any(
        type(count) is not int or count < 0 for count in value.values()
    ):
        raise ValueError("legacy database snapshot active-table counts are invalid")
    return {table: cast(int, value[table]) for table in LEGACY_ACTIVE_TABLE_KEYS}


def verify_serialized_database_snapshot(
    database_image: bytes,
    *,
    expected_active_table_counts: Mapping[str, object],
    expected_schema_version: int,
) -> LegacyDatabaseSnapshotFacts:
    """Deserialize and verify an exact, self-contained pre-reconciliation image."""

    if (
        not isinstance(database_image, bytes)
        or not database_image.startswith(b"SQLite format 3\x00")
        or type(expected_schema_version) is not int
        or expected_schema_version not in (1, 2)
    ):
        raise ValueError("legacy database snapshot identity is invalid")
    expected_counts = _validate_active_table_counts(expected_active_table_counts)
    snapshot = sqlite3.connect(":memory:")
    try:
        snapshot.deserialize(database_image)
        if snapshot.serialize() != database_image:
            raise ValueError("legacy database snapshot is not byte-exact after deserialization")
        integrity_rows = tuple(str(row[0]) for row in snapshot.execute("PRAGMA integrity_check"))
        if integrity_rows != ("ok",):
            raise ValueError("legacy database snapshot failed integrity verification")
        history = tuple(
            row[0]
            for row in snapshot.execute("SELECT version FROM schema_version ORDER BY version")
        )
        if history != tuple(range(1, expected_schema_version + 1)):
            raise ValueError("legacy database snapshot schema version is invalid")
        actual_counts: dict[str, int] = {}
        for table in LEGACY_ACTIVE_TABLE_KEYS:
            table_exists = snapshot.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if table_exists is None:
                raise ValueError("legacy database snapshot active table is missing")
            actual_counts[table] = int(
                snapshot.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table)}").fetchone()[0]
            )
        if actual_counts != expected_counts:
            raise ValueError("legacy database snapshot active facts do not match")
        page_size = int(snapshot.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(snapshot.execute("PRAGMA page_count").fetchone()[0])
        if page_size <= 0 or page_count <= 0 or page_size * page_count != len(database_image):
            raise ValueError("legacy database snapshot page metadata is invalid")
    except sqlite3.Error as exc:
        raise ValueError("legacy database snapshot is not a valid SQLite database") from exc
    finally:
        snapshot.close()
    return LegacyDatabaseSnapshotFacts(
        sha256=hashlib.sha256(database_image).hexdigest(),
        size_bytes=len(database_image),
        source_schema_version=expected_schema_version,
        page_size=page_size,
        page_count=page_count,
        active_table_counts=actual_counts,
    )


def _load_verified_database_snapshot(
    connection: sqlite3.Connection,
    reconciliation_id: str,
) -> tuple[bytes, LegacyDatabaseSnapshotFacts]:
    row = connection.execute(
        "SELECT snapshot.snapshot_id, snapshot.reconciliation_id, "
        "snapshot.database_image, snapshot.sha256, snapshot.size_bytes, "
        "snapshot.captured_at, snapshot.source_schema_version, snapshot.page_size, "
        "snapshot.page_count, snapshot.active_table_counts_json, "
        "metadata.confirmed_at, metadata.source_schema_version AS metadata_schema_version "
        "FROM legacy_authorization_database_snapshots AS snapshot "
        "JOIN legacy_authorization_reconciliations AS metadata "
        "ON metadata.reconciliation_id = snapshot.reconciliation_id "
        "WHERE snapshot.reconciliation_id = ?",
        (reconciliation_id,),
    ).fetchall()
    if len(row) != 1:
        raise ValueError("legacy database snapshot does not exist exactly once")
    record = row[0]
    image = record["database_image"]
    counts = _strict_json_object(record["active_table_counts_json"])
    if not isinstance(image, bytes):
        raise ValueError("legacy database snapshot image is not a blob")
    facts = verify_serialized_database_snapshot(
        image,
        expected_active_table_counts=counts,
        expected_schema_version=record["source_schema_version"],
    )
    if (
        record["snapshot_id"] != f"ldbs_{facts.sha256}"
        or record["reconciliation_id"] != reconciliation_id
        or record["sha256"] != facts.sha256
        or record["size_bytes"] != facts.size_bytes
        or record["source_schema_version"] != facts.source_schema_version
        or record["page_size"] != facts.page_size
        or record["page_count"] != facts.page_count
        or counts != facts.active_table_counts
        or canonical_legacy_json(counts) != record["active_table_counts_json"]
        or _parse_canonical_utc(record["captured_at"]) is None
        or record["captured_at"] != record["confirmed_at"]
        or record["source_schema_version"] != record["metadata_schema_version"]
    ):
        raise ValueError("legacy database snapshot metadata is invalid")
    return image, facts


def export_legacy_database_snapshot(
    connection: sqlite3.Connection,
    reconciliation_id: str,
) -> bytes:
    """Return one archive image only after byte, hash, and fact verification."""

    image, _facts = _load_verified_database_snapshot(connection, reconciliation_id)
    return image


def restore_legacy_database_snapshot(
    archive_connection: sqlite3.Connection,
    destination_connection: sqlite3.Connection,
    reconciliation_id: str,
) -> bytes:
    """Restore the authoritative whole-database image into an empty connection."""

    if destination_connection.in_transaction:
        raise ValueError("legacy database restore requires an idle destination connection")
    existing_objects = int(
        destination_connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
    )
    if existing_objects:
        raise ValueError("legacy database restore requires an empty destination connection")
    image, facts = _load_verified_database_snapshot(archive_connection, reconciliation_id)
    try:
        destination_connection.deserialize(image)
    except sqlite3.Error as exc:
        raise ValueError("legacy database snapshot could not be restored") from exc
    restored_image = destination_connection.serialize()
    if restored_image != image:
        raise ValueError("restored legacy database differs from its archived image")
    restored_facts = verify_serialized_database_snapshot(
        restored_image,
        expected_active_table_counts=facts.active_table_counts,
        expected_schema_version=facts.source_schema_version,
    )
    if restored_facts != facts:
        raise ValueError("restored legacy database facts differ from their archive")
    return image


def _decode_table_manifest(row: sqlite3.Row) -> LegacyDecodedTable:
    table = row["source_table"]
    table_sql = row["source_table_sql"]
    column_manifest = row["column_manifest"]
    if (
        not isinstance(table, str)
        or table not in LEGACY_ACTIVE_TABLE_KEYS
        or not isinstance(table_sql, bytes)
        or not isinstance(column_manifest, bytes)
    ):
        raise ValueError("legacy table manifest storage types are invalid")
    columns = _decode_column_manifest(column_manifest)
    source_sql = table_sql.decode("utf-8")
    schema_hash = legacy_schema_hash(table, table_sql, column_manifest)
    if row["schema_hash"] != schema_hash:
        raise ValueError("legacy table schema hash is invalid")
    return LegacyDecodedTable(
        source_table_order=int(row["source_table_order"]),
        source_table=table,
        source_table_sql=source_sql,
        columns=columns,
        column_manifest=column_manifest,
        schema_hash=schema_hash,
        rows=(),
    )


def audit_legacy_authorization_archives(connection: sqlite3.Connection) -> int:
    """Return one invalid count per reconciliation group; never authorize archived rows."""

    metadata_rows = connection.execute(
        "SELECT reconciliation_id, reviewer, reason, disposition, confirmed_at, "
        "source_schema_version, record_count, schema_manifest_hash, record_manifest_hash, "
        "table_counts_json, state FROM legacy_authorization_reconciliations "
        "ORDER BY reconciliation_id"
    ).fetchall()
    snapshot_rows = connection.execute(
        "SELECT snapshot_id, reconciliation_id FROM legacy_authorization_database_snapshots "
        "ORDER BY reconciliation_id"
    ).fetchall()
    manifest_rows = connection.execute(
        "SELECT manifest_id, reconciliation_id, source_table_order, source_table, "
        "source_table_sql, column_manifest, schema_hash, original_row_count "
        "FROM legacy_authorization_table_manifests "
        "ORDER BY reconciliation_id, source_table_order"
    ).fetchall()
    archive_rows = connection.execute(
        "SELECT quarantine.archive_id, quarantine.reconciliation_id, "
        "quarantine.source_table, quarantine.source_record_key, "
        "quarantine.source_row_ordinal, quarantine.source_rowid, "
        "quarantine.original_row_json, quarantine.typed_row, quarantine.schema_hash, "
        "quarantine.record_hash, quarantine.authorization_state, quarantine.archived_at "
        "FROM legacy_authorization_quarantine AS quarantine "
        "LEFT JOIN legacy_authorization_table_manifests AS manifest "
        "ON manifest.reconciliation_id = quarantine.reconciliation_id "
        "AND manifest.source_table = quarantine.source_table "
        "ORDER BY quarantine.reconciliation_id, manifest.source_table_order, "
        "quarantine.source_row_ordinal"
    ).fetchall()
    metadata_by_id = {row["reconciliation_id"]: row for row in metadata_rows}
    snapshots_by_id: dict[object, list[sqlite3.Row]] = {}
    manifests_by_id: dict[object, list[sqlite3.Row]] = {}
    rows_by_id: dict[object, list[sqlite3.Row]] = {}
    for row in manifest_rows:
        manifests_by_id.setdefault(row["reconciliation_id"], []).append(row)
    for row in snapshot_rows:
        snapshots_by_id.setdefault(row["reconciliation_id"], []).append(row)
    for row in archive_rows:
        rows_by_id.setdefault(row["reconciliation_id"], []).append(row)
    reconciliation_ids = (
        set(metadata_by_id) | set(snapshots_by_id) | set(manifests_by_id) | set(rows_by_id)
    )
    invalid = 0
    for reconciliation_id in reconciliation_ids:
        metadata = metadata_by_id.get(reconciliation_id)
        raw_snapshots = snapshots_by_id.get(reconciliation_id, [])
        raw_manifests = manifests_by_id.get(reconciliation_id, [])
        raw_rows = rows_by_id.get(reconciliation_id, [])
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
                or metadata["record_count"] != len(raw_rows)
            ):
                raise ValueError("legacy reconciliation metadata is invalid")
            if len(raw_snapshots) != 1:
                raise ValueError("legacy reconciliation database snapshot is missing")
            _snapshot_image, snapshot_facts = _load_verified_database_snapshot(
                connection,
                reconciliation_id,
            )
            decoded_manifests = tuple(_decode_table_manifest(row) for row in raw_manifests)
            expected_tables = tuple(LEGACY_ACTIVE_TABLE_KEYS)
            if (
                len(decoded_manifests) != len(expected_tables)
                or tuple(table.source_table for table in decoded_manifests) != expected_tables
                or tuple(table.source_table_order for table in decoded_manifests)
                != tuple(range(len(expected_tables)))
            ):
                raise ValueError("legacy table manifests are incomplete or out of order")
            manifest_by_table = {table.source_table: table for table in decoded_manifests}
            table_counts = _strict_json_object(metadata["table_counts_json"])
            actual_counts = {table: 0 for table in LEGACY_ACTIVE_TABLE_KEYS}
            decoded_rows: dict[str, list[LegacyDecodedRow]] = {
                table: [] for table in LEGACY_ACTIVE_TABLE_KEYS
            }
            manifest_records: list[tuple[str, int, str]] = []
            for row in raw_rows:
                table_name = row["source_table"]
                if table_name not in LEGACY_ACTIVE_TABLE_KEYS:
                    raise ValueError("legacy archive source table is invalid")
                table = manifest_by_table[str(table_name)]
                typed_row = row["typed_row"]
                if not isinstance(typed_row, bytes):
                    raise ValueError("legacy typed row is not a blob")
                decoded = _decode_typed_row(typed_row, len(table.columns))
                expected_ordinal = len(decoded_rows[str(table_name)])
                record_hash = legacy_record_hash(str(table_name), table.schema_hash, typed_row)
                expected_json = legacy_json_safe_row(table, decoded)
                if (
                    decoded.source_row_ordinal != expected_ordinal
                    or row["source_row_ordinal"] != expected_ordinal
                    or row["source_rowid"] != decoded.source_rowid
                    or row["schema_hash"] != table.schema_hash
                    or row["record_hash"] != record_hash
                    or row["archive_id"] != f"lqar_{record_hash}"
                    or row["source_record_key"] != legacy_source_record_key(table, decoded)
                    or row["original_row_json"] != expected_json
                    or row["authorization_state"] != LEGACY_NON_AUTHORIZING_DISPOSITION
                    or row["archived_at"] != metadata["confirmed_at"]
                ):
                    raise ValueError("legacy archive typed row metadata is invalid")
                decoded_rows[str(table_name)].append(decoded)
                actual_counts[str(table_name)] += 1
                manifest_records.append((str(table_name), expected_ordinal, record_hash))
            for raw_manifest, decoded_manifest in zip(
                raw_manifests, decoded_manifests, strict=True
            ):
                if (
                    raw_manifest["manifest_id"] != f"ltm_{decoded_manifest.schema_hash}"
                    or raw_manifest["original_row_count"]
                    != actual_counts[decoded_manifest.source_table]
                ):
                    raise ValueError("legacy table manifest row count is invalid")
            if (
                set(table_counts) != set(LEGACY_ACTIVE_TABLE_KEYS)
                or any(type(value) is not int or value < 0 for value in table_counts.values())
                or table_counts != actual_counts
            ):
                raise ValueError("legacy archive table counts are invalid")
            schema_manifest_hash = legacy_schema_manifest_hash(
                tuple(
                    table.__class__(
                        table.source_table_order,
                        table.source_table,
                        table.source_table_sql,
                        table.columns,
                        table.column_manifest,
                        table.schema_hash,
                        tuple(decoded_rows[table.source_table]),
                    )
                    for table in decoded_manifests
                )
            )
            if metadata["schema_manifest_hash"] != schema_manifest_hash:
                raise ValueError("legacy schema manifest hash is invalid")
            manifest_hash = legacy_manifest_hash(schema_manifest_hash, manifest_records)
            if metadata["record_manifest_hash"] != manifest_hash:
                raise ValueError("legacy archive manifest hash is invalid")
            if reconciliation_id != legacy_reconciliation_id(
                reviewer=metadata["reviewer"],
                reason=metadata["reason"],
                disposition=metadata["disposition"],
                source_schema_version=metadata["source_schema_version"],
                manifest_hash=manifest_hash,
                database_snapshot_hash=snapshot_facts.sha256,
            ):
                raise ValueError("legacy reconciliation identity is invalid")
        except (KeyError, TypeError, ValueError):
            invalid += 1
    return invalid


def decode_legacy_authorization_archive(
    connection: sqlite3.Connection,
    reconciliation_id: str,
) -> tuple[LegacyDecodedTable, ...]:
    """Decode one verified archive into exact ordered schemas, identities, and cells."""

    if audit_legacy_authorization_archives(connection):
        raise ValueError("legacy authorization archive failed integrity verification")
    manifest_rows = connection.execute(
        "SELECT manifest_id, reconciliation_id, source_table_order, source_table, "
        "source_table_sql, column_manifest, schema_hash, original_row_count "
        "FROM legacy_authorization_table_manifests WHERE reconciliation_id = ? "
        "ORDER BY source_table_order",
        (reconciliation_id,),
    ).fetchall()
    if not manifest_rows:
        raise ValueError("legacy authorization reconciliation does not exist")
    tables: list[LegacyDecodedTable] = []
    for manifest_row in manifest_rows:
        manifest = _decode_table_manifest(manifest_row)
        row_records = connection.execute(
            "SELECT typed_row FROM legacy_authorization_quarantine "
            "WHERE reconciliation_id = ? AND source_table = ? ORDER BY source_row_ordinal",
            (reconciliation_id, manifest.source_table),
        ).fetchall()
        rows = tuple(
            _decode_typed_row(row["typed_row"], len(manifest.columns)) for row in row_records
        )
        tables.append(
            LegacyDecodedTable(
                manifest.source_table_order,
                manifest.source_table,
                manifest.source_table_sql,
                manifest.columns,
                manifest.column_manifest,
                manifest.schema_hash,
                rows,
            )
        )
    return tuple(tables)


def restore_legacy_authorization_archive(
    archive_connection: sqlite3.Connection,
    destination_connection: sqlite3.Connection,
    reconciliation_id: str,
) -> tuple[LegacyDecodedTable, ...]:
    """Restore the supplemental active-table projection for forensic inspection."""

    tables = decode_legacy_authorization_archive(archive_connection, reconciliation_id)
    foreign_keys = int(destination_connection.execute("PRAGMA foreign_keys").fetchone()[0])
    if destination_connection.in_transaction:
        raise ValueError("legacy archive restore requires an idle destination connection")
    if foreign_keys:
        destination_connection.execute("PRAGMA foreign_keys = OFF")
    destination_connection.execute("SAVEPOINT restore_legacy_authorization")
    try:
        for table in tables:
            exists = destination_connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (table.source_table,)
            ).fetchone()
            if exists is not None:
                raise ValueError(f"restore destination already contains {table.source_table}")
            destination_connection.execute(table.source_table_sql)
        for table in tables:
            quoted_table = _quote_identifier(table.source_table)
            quoted_columns = ", ".join(_quote_identifier(column.name) for column in table.columns)
            for row in table.rows:
                values = tuple(
                    cell.encoded if cell.storage_class == "TEXT" else cell.value()
                    for cell in row.cells
                )
                placeholders = ", ".join(
                    "CAST(? AS TEXT)" if cell.storage_class == "TEXT" else "?" for cell in row.cells
                )
                if row.source_rowid is None:
                    destination_connection.execute(
                        f"INSERT INTO {quoted_table}({quoted_columns}) VALUES ({placeholders})",
                        values,
                    )
                else:
                    destination_connection.execute(
                        f"INSERT INTO {quoted_table}(rowid, {quoted_columns}) "
                        f"VALUES (?, {placeholders})",
                        (row.source_rowid, *values),
                    )
        restored = capture_legacy_authorization_tables(destination_connection)
        if restored != tables:
            raise ValueError("restored legacy archive differs from its typed source")
        destination_connection.execute("RELEASE SAVEPOINT restore_legacy_authorization")
    except BaseException:
        destination_connection.execute("ROLLBACK TO SAVEPOINT restore_legacy_authorization")
        destination_connection.execute("RELEASE SAVEPOINT restore_legacy_authorization")
        raise
    finally:
        if foreign_keys:
            destination_connection.execute("PRAGMA foreign_keys = ON")
    return tables
