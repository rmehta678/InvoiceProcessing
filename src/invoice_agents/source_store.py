"""Immutable content-addressed storage for submitted invoice sources."""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from invoice_agents.config import PdfPolicy
from invoice_agents.errors import ErrorCategory, SourceEvidenceError
from invoice_agents.models import SourceArtifact

SUPPORTED_FORMATS = {".txt": "txt", ".json": "json", ".csv": "csv", ".xml": "xml", ".pdf": "pdf"}
COPY_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _VerifiedSourceRead:
    content: bytes
    stat_result: os.stat_result


def _identity_error(path: Path, detail: str) -> SourceEvidenceError:
    return SourceEvidenceError(
        ErrorCategory.SOURCE,
        f"source snapshot identity mismatch for {path}: {detail}",
        stop_reason="SOURCE_HASH_MISMATCH",
    )


def _read_error(path: Path, detail: str) -> SourceEvidenceError:
    return SourceEvidenceError(
        ErrorCategory.SOURCE,
        f"source snapshot could not be read for {path}: {detail}",
        stop_reason="SOURCE_READ_FAILED",
    )


def _validate_canonical_path(path: Path) -> None:
    if not path.is_absolute():
        raise _identity_error(path, "canonical path is not absolute")
    try:
        if path.resolve(strict=True) != path:
            raise _identity_error(path, "canonical path does not resolve to itself")
    except (OSError, RuntimeError) as exc:
        raise _identity_error(path, str(exc)) from exc


def _read_verified_path(
    path: Path,
    expected_hash: str,
    expected_size: int,
) -> _VerifiedSourceRead:
    """Acquire, authenticate, and return the bytes from one owned file handle."""

    _validate_canonical_path(path)
    if expected_size < 0:
        raise _identity_error(path, "recorded size is negative")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise _read_error(path, "the platform cannot open a source with no-follow semantics")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
            raise _identity_error(path, str(exc)) from exc
        raise _read_error(path, str(exc)) from exc

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _identity_error(path, "canonical path is not a regular file")
        if before.st_size != expected_size:
            raise _identity_error(
                path,
                f"expected size={expected_size}; found size={before.st_size}",
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            content = handle.read(expected_size + 1)
            after = os.fstat(handle.fileno())
    except SourceEvidenceError:
        raise
    except (MemoryError, OSError, OverflowError, ValueError) as exc:
        raise _read_error(path, str(exc)) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    before_identity = (
        before.st_dev,
        before.st_ino,
        stat.S_IFMT(before.st_mode),
        before.st_size,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        stat.S_IFMT(after.st_mode),
        after.st_size,
    )
    if after_identity != before_identity:
        raise _identity_error(path, "file identity or size changed while it was read")
    actual_hash = hashlib.sha256(content).hexdigest()
    actual_size = len(content)
    if actual_size != expected_size or actual_hash != expected_hash:
        raise _identity_error(
            path,
            f"expected sha256={expected_hash} size={expected_size}; "
            f"found sha256={actual_hash} size={actual_size}",
        )
    return _VerifiedSourceRead(content=content, stat_result=after)


def _verify_path(path: Path, expected_hash: str, expected_size: int) -> None:
    _read_verified_path(path, expected_hash, expected_size)


def _validate_content_address(
    path: Path,
    expected_hash: str,
    source_format: str,
) -> None:
    expected_suffix = f".{source_format}"
    if (
        source_format not in SUPPORTED_FORMATS.values()
        or path.suffix.lower() != expected_suffix
        or path.name != f"{expected_hash}{expected_suffix}"
    ):
        raise _identity_error(path, "path is not the expected content address")


def read_verified_source_bytes(source: SourceArtifact) -> bytes:
    """Return authenticated bounded bytes read from one non-symlink file handle."""

    path = source.canonical_path
    _validate_content_address(path, source.sha256, source.source_format)
    return _read_verified_path(path, source.sha256, source.size_bytes).content


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


def inspect_snapshot(
    target: Path,
    expected_hash: str,
    size_bytes: int,
    *,
    pdf_policy: PdfPolicy,
) -> SourceArtifact:
    """Verify a stored snapshot and derive its immutable format metadata."""

    resolved = target.expanduser().absolute()
    source_format = SUPPORTED_FORMATS.get(resolved.suffix.lower())
    if source_format is None:
        raise _identity_error(resolved, "path is not the expected content address")
    _validate_content_address(resolved, expected_hash, source_format)
    verified = _read_verified_path(resolved, expected_hash, size_bytes)

    page_count: int | None = None
    row_count: int | None = None
    try:
        if source_format == "csv":
            text = verified.content.decode("utf-8-sig")
            row_count = sum(1 for _ in csv.reader(io.StringIO(text, newline="")))
    except Exception as exc:
        raise SourceEvidenceError(
            ErrorCategory.PARSE,
            f"could not inspect {source_format} source: {exc}",
            stop_reason="SOURCE_INSPECTION_FAILED",
        ) from exc
    source_id = f"src_{uuid5(NAMESPACE_URL, str(resolved) + ':' + expected_hash).hex}"
    source = SourceArtifact(
        source_id=source_id,
        canonical_path=resolved,
        sha256=expected_hash,
        source_format=source_format,
        size_bytes=size_bytes,
        modified_at=datetime.fromtimestamp(verified.stat_result.st_mtime, tz=UTC),
        page_count=page_count,
        row_count=row_count,
    )
    if source_format == "pdf":
        from invoice_agents.pdf_worker import inspect_pdf_in_worker

        page_count = inspect_pdf_in_worker(source, pdf_policy)
        source = source.model_copy(update={"page_count": page_count})
    return source


def snapshot_source(
    path: Path,
    archive_dir: Path,
    max_bytes: int,
    *,
    pdf_policy: PdfPolicy,
) -> SourceArtifact:
    """Persist and inspect a submitted file before registration or extraction."""

    source_hash, size_bytes, target = copy_and_hash_atomically(path, archive_dir, max_bytes)
    return inspect_snapshot(
        target,
        expected_hash=source_hash,
        size_bytes=size_bytes,
        pdf_policy=pdf_policy,
    )


def verify_source_identity(source: SourceArtifact) -> None:
    """Raise unless the persisted artifact still names and contains the recorded bytes."""

    path = source.canonical_path
    _validate_content_address(path, source.sha256, source.source_format)
    _verify_path(path, source.sha256, source.size_bytes)


def verified_source_path(source: SourceArtifact) -> Path:
    """Return the persisted path only after verifying its complete identity."""

    verify_source_identity(source)
    return source.canonical_path
