"""Immutable content-addressed storage for submitted invoice sources."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pypdf import PdfReader

from invoice_agents.errors import ErrorCategory, SourceEvidenceError
from invoice_agents.models import SourceArtifact

SUPPORTED_FORMATS = {".txt": "txt", ".json": "json", ".csv": "csv", ".xml": "xml", ".pdf": "pdf"}
COPY_BLOCK_BYTES = 1024 * 1024


def _identity_error(path: Path, detail: str) -> SourceEvidenceError:
    return SourceEvidenceError(
        ErrorCategory.SOURCE,
        f"source snapshot identity mismatch for {path}: {detail}",
        stop_reason="SOURCE_HASH_MISMATCH",
    )


def _hash_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(COPY_BLOCK_BYTES), b""):
                digest.update(block)
                size_bytes += len(block)
    except (OSError, ValueError) as exc:
        raise _identity_error(path, str(exc)) from exc
    return digest.hexdigest(), size_bytes


def _verify_path(path: Path, expected_hash: str, expected_size: int) -> None:
    if not path.is_absolute():
        raise _identity_error(path, "canonical path is not absolute")
    if path.is_symlink() or not path.is_file():
        raise _identity_error(path, "canonical path is not a regular non-symlink file")
    try:
        if path.resolve(strict=True) != path:
            raise _identity_error(path, "canonical path does not resolve to itself")
    except OSError as exc:
        raise _identity_error(path, str(exc)) from exc
    actual_hash, actual_size = _hash_and_size(path)
    if actual_size != expected_size or actual_hash != expected_hash:
        raise _identity_error(
            path,
            f"expected sha256={expected_hash} size={expected_size}; "
            f"found sha256={actual_hash} size={actual_size}",
        )


def copy_and_hash_atomically(
    path: Path, archive_dir: Path, max_bytes: int
) -> tuple[str, int, Path]:
    """Stream a submitted file into its digest-named archive entry."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise SourceEvidenceError(
            ErrorCategory.SOURCE,
            f"invoice source does not exist or is not a file: {source}",
            stop_reason="SOURCE_NOT_FOUND",
        )
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise SourceEvidenceError(
            ErrorCategory.SOURCE,
            f"unsupported invoice format {source.suffix or '<none>'}",
            stop_reason="SOURCE_FORMAT_UNSUPPORTED",
        )
    if max_bytes < 1:
        raise SourceEvidenceError(
            ErrorCategory.CONFIGURATION,
            "source byte ceiling must be positive",
            stop_reason="SOURCE_MAX_BYTES_INVALID",
        )

    archive = archive_dir.expanduser().resolve()
    archive.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".source-", suffix=".tmp", dir=archive)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with source.open("rb") as submitted, os.fdopen(descriptor, "wb") as snapshot:
            descriptor = -1
            for block in iter(lambda: submitted.read(COPY_BLOCK_BYTES), b""):
                size_bytes += len(block)
                if size_bytes > max_bytes:
                    raise SourceEvidenceError(
                        ErrorCategory.SOURCE,
                        f"invoice source exceeds the {max_bytes}-byte ceiling",
                        stop_reason="SOURCE_TOO_LARGE",
                    )
                digest.update(block)
                snapshot.write(block)
            snapshot.flush()
            os.fsync(snapshot.fileno())

        source_hash = digest.hexdigest()
        target = archive / f"{source_hash}{suffix}"
        try:
            os.link(temporary, target)
        except FileExistsError:
            _verify_path(target, source_hash, size_bytes)
        temporary.unlink()
        directory_fd = os.open(archive, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _verify_path(target, source_hash, size_bytes)
        return source_hash, size_bytes, target
    except SourceEvidenceError:
        raise
    except OSError as exc:
        raise SourceEvidenceError(
            ErrorCategory.SOURCE,
            f"invoice source could not be snapshotted: {exc}",
            stop_reason="SOURCE_READ_FAILED",
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def inspect_snapshot(target: Path, expected_hash: str, size_bytes: int) -> SourceArtifact:
    """Verify a stored snapshot and derive its immutable format metadata."""

    resolved = target.resolve()
    source_format = SUPPORTED_FORMATS.get(resolved.suffix.lower())
    if source_format is None or resolved.name != f"{expected_hash}{resolved.suffix.lower()}":
        raise _identity_error(resolved, "path is not the expected content address")
    _verify_path(resolved, expected_hash, size_bytes)

    page_count: int | None = None
    row_count: int | None = None
    try:
        if source_format == "pdf":
            page_count = len(PdfReader(resolved).pages)
            if page_count < 1:
                raise ValueError("PDF has no pages")
        elif source_format == "csv":
            with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
                row_count = sum(1 for _ in csv.reader(handle))
    except Exception as exc:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            f"could not inspect {source_format} source: {exc}",
            stop_reason="SOURCE_INSPECTION_FAILED",
        ) from exc
    stat = resolved.stat()
    source_id = f"src_{uuid5(NAMESPACE_URL, str(resolved) + ':' + expected_hash).hex}"
    return SourceArtifact(
        source_id=source_id,
        canonical_path=resolved,
        sha256=expected_hash,
        source_format=source_format,
        size_bytes=size_bytes,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        page_count=page_count,
        row_count=row_count,
    )


def snapshot_source(path: Path, archive_dir: Path, max_bytes: int) -> SourceArtifact:
    """Persist and inspect a submitted file before registration or extraction."""

    source_hash, size_bytes, target = copy_and_hash_atomically(path, archive_dir, max_bytes)
    return inspect_snapshot(target, expected_hash=source_hash, size_bytes=size_bytes)


def verify_source_identity(source: SourceArtifact) -> None:
    """Raise unless the persisted artifact still names and contains the recorded bytes."""

    path = source.canonical_path
    expected_suffix = f".{source.source_format}"
    if path.suffix.lower() != expected_suffix or path.name != f"{source.sha256}{expected_suffix}":
        raise _identity_error(path, "persisted path is not the recorded content address")
    _verify_path(path, source.sha256, source.size_bytes)


def verified_source_path(source: SourceArtifact) -> Path:
    """Return the persisted path only after verifying its complete identity."""

    verify_source_identity(source)
    return source.canonical_path
