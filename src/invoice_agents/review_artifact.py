"""Typed bindings and exact-byte validation for persisted review-page evidence."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from invoice_agents.models import ReviewRequest

REVIEW_PAGE_HARD_MAX_BYTES = 16_777_216
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_STAT_SUPPORTS_DIR_FD = os.stat in os.supports_dir_fd
_STAT_SUPPORTS_NOFOLLOW = os.stat in os.supports_follow_symlinks
REVIEW_PAGE_FIELDS = frozenset(
    {
        "path",
        "page",
        "sha256",
        "renderer",
        "device",
        "inode",
        "file_type",
        "size_bytes",
    }
)


class ReviewPageEvidenceError(ValueError):
    """A review-page binding or its exact filesystem evidence is invalid."""


@dataclass(frozen=True, slots=True)
class ReviewPageBinding:
    path: Path
    page: int
    sha256: str
    renderer: str
    device: int
    inode: int
    file_type: int
    size_bytes: int

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return (self.device, self.inode, self.file_type, self.size_bytes)

    @classmethod
    def from_payload(
        cls,
        raw: object,
        *,
        review_id: str,
        source_id: str,
        expected_page: int,
    ) -> ReviewPageBinding:
        if not isinstance(raw, Mapping) or set(raw) != REVIEW_PAGE_FIELDS:
            raise ReviewPageEvidenceError("review page binding has an invalid shape")
        raw_path = raw.get("path")
        raw_page = raw.get("page")
        raw_sha256 = raw.get("sha256")
        raw_renderer = raw.get("renderer")
        raw_device = raw.get("device")
        raw_inode = raw.get("inode")
        raw_file_type = raw.get("file_type")
        raw_size = raw.get("size_bytes")
        if (
            type(raw_path) is not str
            or type(raw_page) is not int
            or type(raw_sha256) is not str
            or type(raw_renderer) is not str
            or type(raw_device) is not int
            or type(raw_inode) is not int
            or type(raw_file_type) is not int
            or type(raw_size) is not int
        ):
            raise ReviewPageEvidenceError("review page binding has invalid field types")
        path = Path(raw_path)
        if (
            raw_page != expected_page
            or not path.is_absolute()
            or path.name != f"{source_id}-page-{expected_page}.png"
            or path.parent.name != review_id
            or path.parent.parent.name != "reviews"
            or raw_renderer != "PyMuPDF"
            or len(raw_sha256) != 64
            or any(character not in "0123456789abcdef" for character in raw_sha256)
            or raw_device < 0
            or raw_inode <= 0
            or raw_file_type != stat.S_IFREG
            or not 0 < raw_size <= REVIEW_PAGE_HARD_MAX_BYTES
        ):
            raise ReviewPageEvidenceError("review page binding is not canonical")
        return cls(
            path=path,
            page=raw_page,
            sha256=raw_sha256,
            renderer=raw_renderer,
            device=raw_device,
            inode=raw_inode,
            file_type=raw_file_type,
            size_bytes=raw_size,
        )


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
    )


def _validate_parent_relationships(
    relationships: list[tuple[int, str, os.stat_result]],
) -> None:
    for parent_descriptor, component, opened in relationships:
        linked = os.stat(component, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            linked.st_dev != opened.st_dev
            or linked.st_ino != opened.st_ino
            or not stat.S_ISDIR(linked.st_mode)
        ):
            raise ReviewPageEvidenceError("review page parent namespace changed")


def _required_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int:
        raise ReviewPageEvidenceError("review page descriptor validation is unavailable")
    return value


def read_verified_review_page(binding: ReviewPageBinding) -> bytes:
    """Read one exact bound page once through no-follow namespace capabilities."""

    if not _OPEN_SUPPORTS_DIR_FD or not _STAT_SUPPORTS_DIR_FD or not _STAT_SUPPORTS_NOFOLLOW:
        raise ReviewPageEvidenceError("review page descriptor validation is unavailable")
    read_only = _required_open_flag("O_RDONLY")
    close_exec = _required_open_flag("O_CLOEXEC")
    no_follow = _required_open_flag("O_NOFOLLOW")
    nonblocking = _required_open_flag("O_NONBLOCK")
    directory = _required_open_flag("O_DIRECTORY")
    directory_flags = read_only | close_exec | no_follow | nonblocking | directory
    file_flags = read_only | close_exec | no_follow | nonblocking
    descriptors: list[int] = []
    relationships: list[tuple[int, str, os.stat_result]] = []
    try:
        root_descriptor = os.open(binding.path.anchor, directory_flags)
        descriptors.append(root_descriptor)
        parent_descriptor = root_descriptor
        for component in binding.path.parent.parts[1:]:
            if not component or component in {".", ".."} or os.sep in component:
                raise ReviewPageEvidenceError("review page parent component is invalid")
            directory_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            descriptors.append(directory_descriptor)
            opened_directory = os.fstat(directory_descriptor)
            if not stat.S_ISDIR(opened_directory.st_mode):
                raise ReviewPageEvidenceError("review page parent is not a directory")
            relationships.append((parent_descriptor, component, opened_directory))
            parent_descriptor = directory_descriptor
        namespace = os.stat(
            binding.path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _identity(namespace) != binding.identity
            or not stat.S_ISREG(namespace.st_mode)
            or namespace.st_nlink != 1
        ):
            raise ReviewPageEvidenceError("review page namespace is not the exact binding")
        artifact_descriptor = os.open(
            binding.path.name,
            file_flags,
            dir_fd=parent_descriptor,
        )
        descriptors.append(artifact_descriptor)
        opened = os.fstat(artifact_descriptor)
        if (
            _identity(opened) != binding.identity
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise ReviewPageEvidenceError("opened review page is not the exact binding")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(artifact_descriptor, 65_536)
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > binding.size_bytes:
                raise ReviewPageEvidenceError("review page exceeded its exact size")
            digest.update(chunk)
            chunks.append(chunk)
        final_opened = os.fstat(artifact_descriptor)
        final_namespace = os.stat(
            binding.path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            observed_size != binding.size_bytes
            or digest.hexdigest() != binding.sha256
            or _identity(final_opened) != binding.identity
            or _identity(final_namespace) != binding.identity
            or final_opened.st_nlink != 1
            or final_namespace.st_nlink != 1
        ):
            raise ReviewPageEvidenceError("review page changed during its exact read")
        _validate_parent_relationships(relationships)
        return b"".join(chunks)
    except ReviewPageEvidenceError:
        raise
    except Exception:
        raise ReviewPageEvidenceError("review page exact-byte validation failed") from None
    finally:
        cleanup_error: BaseException | None = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None:
            raise ReviewPageEvidenceError("review page descriptor cleanup failed") from None


def validate_review_page_evidence(review: ReviewRequest) -> None:
    """Authenticate every rendered page bound into one persisted review package."""

    source = review.source
    expected_pages = (
        list(range(1, (source.page_count or 1) + 1))
        if source.source_format == "pdf" and (source.page_count or 1) <= 3
        else [1]
        if source.source_format == "pdf"
        else []
    )
    raw_pages = review.evidence_bundle.get("rendered_pages")
    if not isinstance(raw_pages, list) or len(raw_pages) != len(expected_pages):
        raise ReviewPageEvidenceError("review page evidence set is incomplete")
    for raw, page in zip(raw_pages, expected_pages, strict=True):
        binding = ReviewPageBinding.from_payload(
            raw,
            review_id=review.review_id,
            source_id=source.source_id,
            expected_page=page,
        )
        read_verified_review_page(binding)
