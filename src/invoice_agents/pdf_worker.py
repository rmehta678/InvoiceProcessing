"""Killable, resource-bounded child-process boundary for untrusted PDFs."""

from __future__ import annotations

import codecs
import contextlib
import hashlib
import json
import multiprocessing
import os
import resource
import selectors
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal, cast

from invoice_agents.config import Settings
from invoice_agents.errors import ErrorCategory, SourceEvidenceError
from invoice_agents.models import SourceArtifact

WorkerOperation = Literal["inspect", "extract", "render"]
JOIN_GRACE_SECONDS = 1.0
WIRE_HEADER_BYTES = 4
WIRE_READ_CHUNK_BYTES = 65_536
METADATA_MAX_BYTES = 32_768
TEXT_ENCODE_CHARS = 4_096
GENERIC_WORKER_MESSAGE = "PDF worker failed"


@dataclass(slots=True)
class _WireMessage:
    metadata: bytearray
    text: bytearray

    @property
    def payload_bytes(self) -> int:
        return WIRE_HEADER_BYTES + len(self.metadata) + len(self.text)


@dataclass(slots=True)
class _ExtractionWireBuilder:
    """Aggregate only capped extraction bytes and compact page descriptors."""

    page_count: int
    max_bytes: int
    text: bytearray = field(default_factory=bytearray)
    page_descriptors: bytearray = field(default_factory=bytearray)
    has_text: bool = False

    def append_page(self, page_number: int, page_text: str) -> None:
        page_start = len(self.text)
        _append_bounded_utf8(self.text, page_text, max_bytes=self.max_bytes)
        text_bytes = len(self.text) - page_start
        descriptor = (
            (b"," if self.page_descriptors else b"")
            + f'{{"page":{page_number},"text_bytes":{text_bytes}}}'.encode("ascii")
        )
        if self._metadata_size_with(descriptor) > METADATA_MAX_BYTES:
            raise _ResultTooLarge
        self.page_descriptors.extend(descriptor)
        self.has_text = self.has_text or (bool(page_text) and not page_text.isspace())

    def build(self) -> _WireMessage:
        metadata = bytearray(b'{"ok":true,"result":{"pages":[')
        metadata.extend(self.page_descriptors)
        metadata.extend(
            f'],"extractor":"pypdf","page_count":{self.page_count}}}}}'.encode("ascii")
        )
        message = _WireMessage(metadata=metadata, text=self.text)
        if len(metadata) > METADATA_MAX_BYTES or message.payload_bytes > self.max_bytes:
            raise _ResultTooLarge
        return message

    def _metadata_size_with(self, descriptor: bytes) -> int:
        prefix_bytes = len(b'{"ok":true,"result":{"pages":[')
        suffix_bytes = len(
            f'],"extractor":"pypdf","page_count":{self.page_count}}}}}'.encode("ascii")
        )
        return prefix_bytes + len(self.page_descriptors) + len(descriptor) + suffix_bytes


class _ResultTooLarge(Exception):
    """Internal control flow for replacing an oversized result with a generic failure."""


def _bounded_limit(resource_name: int, requested: int) -> tuple[int, int]:
    current_soft, current_hard = resource.getrlimit(resource_name)
    hard = current_hard
    soft = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    if current_soft != resource.RLIM_INFINITY:
        soft = min(soft, current_soft)
    return soft, hard


def _apply_resource_limits(cpu_seconds: int, memory_bytes: int) -> None:
    resource.setrlimit(
        resource.RLIMIT_CPU,
        _bounded_limit(resource.RLIMIT_CPU, cpu_seconds),
    )
    baseline_bytes = _current_virtual_memory_bytes()
    address_space_ceiling = baseline_bytes + memory_bytes
    soft, hard = _bounded_limit(resource.RLIMIT_AS, address_space_ceiling)
    if soft <= baseline_bytes:
        raise RuntimeError("existing address-space limit leaves no PDF worker headroom")
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    applied_soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    if applied_soft == resource.RLIM_INFINITY or applied_soft > address_space_ceiling:
        raise RuntimeError("PDF worker address-space limit was not applied")


def _current_virtual_memory_bytes() -> int:
    """Measure this child before parser imports so RLIMIT_AS can bound added memory."""

    if sys.platform == "darwin":
        completed = subprocess.run(
            ["/bin/ps", "-o", "vsz=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        value = int(completed.stdout.strip()) * 1024
    elif sys.platform.startswith("linux"):
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        value = int(fields[0]) * os.sysconf("SC_PAGE_SIZE")
    else:
        raise RuntimeError(f"cannot measure virtual memory on {sys.platform}")
    if value <= 0:
        raise RuntimeError("worker virtual-memory measurement was not positive")
    return value


def _generic_failure_payload() -> dict[str, object]:
    return {
        "ok": False,
        "category": ErrorCategory.TOOL.value,
        "message": GENERIC_WORKER_MESSAGE,
        "stop_reason": "PDF_WORKER_FAILED",
    }


def _write_all(
    file_descriptor: int,
    content: bytes | bytearray | memoryview,
) -> None:
    view = memoryview(content)
    written = 0
    while written < len(view):
        count = os.write(file_descriptor, view[written:])
        if count < 1:
            raise BrokenPipeError("PDF result pipe closed")
        written += count


def _metadata_bytes(payload: dict[str, object]) -> bytearray:
    stack: list[object] = [payload]
    seen_containers: set[int] = set()
    visited_values = 0
    while stack:
        value = stack.pop()
        visited_values += 1
        if visited_values > METADATA_MAX_BYTES:
            raise _ResultTooLarge
        if isinstance(value, str):
            if len(value) > METADATA_MAX_BYTES // 6:
                raise _ResultTooLarge
            continue
        if isinstance(value, dict):
            identity = id(value)
            if identity in seen_containers or len(value) > METADATA_MAX_BYTES:
                raise _ResultTooLarge
            seen_containers.add(identity)
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in seen_containers or len(value) > METADATA_MAX_BYTES:
                raise _ResultTooLarge
            seen_containers.add(identity)
            stack.extend(value)

    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))
    encoded = bytearray()
    for piece in encoder.iterencode(payload):
        encoded_piece = piece.encode("utf-8")
        if len(encoded) + len(encoded_piece) > METADATA_MAX_BYTES:
            raise _ResultTooLarge
        encoded.extend(encoded_piece)
    return encoded


def _append_bounded_utf8(
    destination: bytearray,
    source: str,
    *,
    max_bytes: int,
) -> None:
    for start in range(0, len(source), TEXT_ENCODE_CHARS):
        encoded_chunk = source[start : start + TEXT_ENCODE_CHARS].encode("utf-8")
        if (
            WIRE_HEADER_BYTES
            + METADATA_MAX_BYTES
            + len(destination)
            + len(encoded_chunk)
            > max_bytes
        ):
            raise _ResultTooLarge
        destination.extend(encoded_chunk)


def _encode_bounded_message(
    payload: dict[str, object],
    *,
    max_bytes: int,
) -> _WireMessage:
    """Encode metadata plus page text without constructing a second full text copy."""

    try:
        text = bytearray()
        metadata_payload = payload
        result = payload.get("result")
        if (
            payload.get("ok") is True
            and isinstance(result, dict)
            and result.get("extractor") == "pypdf"
            and isinstance(result.get("pages"), list)
        ):
            page_metadata: list[dict[str, object]] = []
            for page in result["pages"]:
                if not isinstance(page, dict) or not isinstance(page.get("text"), str):
                    raise ValueError("invalid child extraction result")
                page_start = len(text)
                page_text = page["text"]
                _append_bounded_utf8(text, page_text, max_bytes=max_bytes)
                page_metadata.append(
                    {
                        "page": page.get("page"),
                        "text_bytes": len(text) - page_start,
                    }
                )
            metadata_payload = {
                "ok": True,
                "result": {
                    "pages": page_metadata,
                    "extractor": "pypdf",
                    "page_count": result.get("page_count"),
                },
            }
        metadata = _metadata_bytes(metadata_payload)
        message = _WireMessage(metadata=metadata, text=text)
        if message.payload_bytes > max_bytes:
            raise _ResultTooLarge
        return message
    except _ResultTooLarge:
        metadata = _metadata_bytes(_generic_failure_payload())
        message = _WireMessage(metadata=metadata, text=bytearray())
        if message.payload_bytes > max_bytes:
            raise RuntimeError(
                "PDF result ceiling cannot contain the generic failure envelope"
            ) from None
        return message


def _send(
    connection: Connection,
    payload: dict[str, object],
    max_bytes: int,
) -> None:
    """Send one capped metadata/text frame without full-payload concatenation."""

    message = _encode_bounded_message(payload, max_bytes=max_bytes)
    _send_wire_message(connection, message)


def _send_wire_message(connection: Connection, message: _WireMessage) -> None:
    """Write one already-bounded wire message without concatenating its buffers."""

    file_descriptor = connection.fileno()
    _write_all(file_descriptor, struct.pack("!I", message.payload_bytes))
    _write_all(file_descriptor, struct.pack("!I", len(message.metadata)))
    _write_all(file_descriptor, message.metadata)
    _write_all(file_descriptor, message.text)


def _page_limit_error(
    connection: Connection,
    page_count: int,
    max_pages: int,
    result_max_bytes: int,
) -> None:
    _send(
        connection,
        {
            "ok": False,
            "category": ErrorCategory.PARSE.value,
            "message": f"PDF has {page_count} pages; maximum is {max_pages}",
            "stop_reason": "PDF_PAGE_LIMIT_EXCEEDED",
            "details": {"page_count": page_count, "max_pages": max_pages},
        },
        result_max_bytes,
    )


def _silence_child_output() -> None:
    """Prevent parser diagnostics and tracebacks from crossing inherited file descriptors."""

    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        os.close(descriptor)


def _worker_main(
    connection: Connection,
    operation: WorkerOperation,
    source_payload: dict[str, object],
    max_pages: int,
    cpu_seconds: int,
    memory_bytes: int,
    result_max_bytes: int,
    page: int | None,
    render_target: str | None,
) -> None:
    """Run one PDF operation and reduce unexpected failures to a generic envelope."""

    try:
        _silence_child_output()
        _apply_resource_limits(cpu_seconds, memory_bytes)
        source = SourceArtifact.model_validate(source_payload)
        from invoice_agents.source_store import verified_source_path

        source_path = verified_source_path(source)
        if operation in {"inspect", "extract"}:
            from pypdf import PdfReader

            reader = PdfReader(source_path)
            page_count = len(reader.pages)
            if page_count < 1:
                raise ValueError("PDF has no pages")
            if page_count > max_pages:
                _page_limit_error(connection, page_count, max_pages, result_max_bytes)
                return
            if operation == "inspect":
                _send(
                    connection,
                    {"ok": True, "result": {"page_count": page_count}},
                    result_max_bytes,
                )
                return
            extraction = _ExtractionWireBuilder(
                page_count=page_count,
                max_bytes=result_max_bytes,
            )
            for index, pdf_page in enumerate(reader.pages, 1):
                page_text = pdf_page.extract_text() or ""
                page_too_large = False
                try:
                    extraction.append_page(index, page_text)
                except _ResultTooLarge:
                    page_too_large = True
                finally:
                    del page_text
                if page_too_large:
                    _send(connection, _generic_failure_payload(), result_max_bytes)
                    return
            if not extraction.has_text:
                _send(
                    connection,
                    {
                        "ok": False,
                        "category": ErrorCategory.PARSE.value,
                        "message": (
                            "PDF contains no extractable text; visual/OCR review is required"
                        ),
                        "stop_reason": "PDF_TEXT_EMPTY",
                    },
                    result_max_bytes,
                )
                return
            try:
                _send_wire_message(connection, extraction.build())
            except _ResultTooLarge:
                _send(connection, _generic_failure_payload(), result_max_bytes)
            return

        if page is None or render_target is None:
            raise ValueError("render operation requires a page and target")
        import fitz

        document = fitz.open(source_path)
        try:
            page_count = document.page_count
            if page_count > max_pages:
                _page_limit_error(connection, page_count, max_pages, result_max_bytes)
                return
            if page < 1 or page > page_count:
                _send(
                    connection,
                    {
                        "ok": False,
                        "category": ErrorCategory.SOURCE.value,
                        "message": f"PDF page {page} is out of range",
                        "stop_reason": "RENDER_PAGE_INVALID",
                    },
                    result_max_bytes,
                )
                return
            pixmap = document[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pixmap.save(render_target)
        finally:
            document.close()
        rendered = Path(render_target)
        digest = hashlib.sha256(rendered.read_bytes()).hexdigest()
        _send(
            connection,
            {
                "ok": True,
                "result": {
                    "page": page,
                    "sha256": digest,
                    "renderer": "PyMuPDF",
                },
            },
            result_max_bytes,
        )
    except BaseException:
        with contextlib.suppress(BaseException):
            _send(connection, _generic_failure_payload(), result_max_bytes)
    finally:
        connection.close()


def _stop_process(process: BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
    process.join(JOIN_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join()


def _worker_failed() -> SourceEvidenceError:
    return SourceEvidenceError(
        ErrorCategory.TOOL,
        GENERIC_WORKER_MESSAGE,
        stop_reason="PDF_WORKER_FAILED",
    )


def _parse_timeout() -> SourceEvidenceError:
    return SourceEvidenceError(
        ErrorCategory.TIMEOUT,
        "PDF processing exceeded the configured deadline",
        stop_reason="PDF_PARSE_TIMEOUT",
    )


def _receive_bounded_message(
    connection: Connection,
    *,
    max_bytes: int,
    deadline: float,
) -> bytearray:
    """Receive exactly one capped frame without blocking past the absolute deadline."""

    file_descriptor = connection.fileno()
    was_blocking = os.get_blocking(file_descriptor)
    selector = selectors.DefaultSelector()
    selector.register(file_descriptor, selectors.EVENT_READ)
    os.set_blocking(file_descriptor, False)

    def read_exact(length: int) -> bytearray:
        received = bytearray()
        while len(received) < length:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise _parse_timeout()
            if not selector.select(remaining_seconds):
                raise _parse_timeout()
            try:
                chunk = os.read(
                    file_descriptor,
                    min(WIRE_READ_CHUNK_BYTES, length - len(received)),
                )
            except BlockingIOError:
                continue
            if not chunk:
                raise _worker_failed()
            received.extend(chunk)
        return received

    try:
        header = read_exact(WIRE_HEADER_BYTES)
        (message_length,) = struct.unpack("!I", header)
        if message_length < 1 or message_length > max_bytes:
            raise _worker_failed()
        return read_exact(message_length)
    finally:
        selector.close()
        os.set_blocking(file_descriptor, was_blocking)


def _split_wire_message(
    encoded: bytes | bytearray | memoryview,
) -> tuple[memoryview, memoryview]:
    """Split one received frame into bounded metadata and page-text views."""

    view = memoryview(encoded)
    if len(view) <= WIRE_HEADER_BYTES:
        raise _worker_failed()
    (metadata_length,) = struct.unpack("!I", view[:WIRE_HEADER_BYTES])
    metadata_end = WIRE_HEADER_BYTES + metadata_length
    if (
        metadata_length < 1
        or metadata_length > METADATA_MAX_BYTES
        or metadata_end > len(view)
    ):
        raise _worker_failed()
    return view[WIRE_HEADER_BYTES:metadata_end], view[metadata_end:]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decode_worker_message(
    operation: WorkerOperation,
    encoded: bytes | bytearray | memoryview,
    *,
    expected_page: int | None,
    max_pages: int = 100,
    text_payload: bytes | bytearray | memoryview = b"",
    deadline: float = float("inf"),
) -> dict[str, object]:
    """Validate bounded metadata and incrementally decode extraction text."""

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise _parse_timeout()

    try:
        check_deadline()
        if len(encoded) < 1 or len(encoded) > METADATA_MAX_BYTES:
            raise ValueError("invalid metadata size")
        payload = json.loads(bytes(encoded).decode("utf-8"))
        check_deadline()
        if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
            raise ValueError("invalid common envelope")
        if payload["ok"] is False:
            if len(text_payload) != 0:
                raise ValueError("error envelope included text")
            required = {"ok", "category", "message", "stop_reason"}
            if set(payload) not in (required, required | {"details"}):
                raise ValueError("invalid error envelope keys")
            category = payload["category"]
            message = payload["message"]
            stop_reason = payload["stop_reason"]
            details = payload.get("details")
            if not all(isinstance(value, str) for value in (category, message, stop_reason)):
                raise ValueError("invalid error envelope values")
            if details is not None and not isinstance(details, dict):
                raise ValueError("invalid error details")
            if stop_reason == "PDF_WORKER_FAILED":
                if category != ErrorCategory.TOOL.value or details is not None:
                    raise ValueError("invalid generic worker failure")
                raise _worker_failed()
            if stop_reason == "PDF_PAGE_LIMIT_EXCEEDED":
                if (
                    category != ErrorCategory.PARSE.value
                    or not isinstance(details, dict)
                    or set(details) != {"page_count", "max_pages"}
                    or not _is_int(details.get("page_count"))
                    or not _is_int(details.get("max_pages"))
                    or details["page_count"] <= details["max_pages"]
                    or details["max_pages"] < 1
                    or details["max_pages"] != max_pages
                ):
                    raise ValueError("invalid page-limit failure")
                raise SourceEvidenceError(
                    ErrorCategory.PARSE,
                    f"PDF has {details['page_count']} pages; maximum is {details['max_pages']}",
                    stop_reason="PDF_PAGE_LIMIT_EXCEEDED",
                    details=details,
                )
            if stop_reason == "PDF_TEXT_EMPTY":
                if (
                    operation != "extract"
                    or category != ErrorCategory.PARSE.value
                    or details is not None
                ):
                    raise ValueError("invalid empty-text failure")
                raise SourceEvidenceError(
                    ErrorCategory.PARSE,
                    "PDF contains no extractable text; visual/OCR review is required",
                    stop_reason="PDF_TEXT_EMPTY",
                )
            if stop_reason == "RENDER_PAGE_INVALID":
                if (
                    operation != "render"
                    or category != ErrorCategory.SOURCE.value
                    or details is not None
                ):
                    raise ValueError("invalid render-page failure")
                raise SourceEvidenceError(
                    ErrorCategory.SOURCE,
                    f"PDF page {expected_page} is out of range",
                    stop_reason="RENDER_PAGE_INVALID",
                )
            raise ValueError("unknown worker stop reason")
        if set(payload) != {"ok", "result"} or not isinstance(payload["result"], dict):
            raise ValueError("invalid success envelope")
        result = payload["result"]
        if operation == "inspect":
            if len(text_payload) != 0:
                raise ValueError("inspection result included text")
            if set(result) != {"page_count"} or not _is_int(result.get("page_count")):
                raise ValueError("invalid inspection metadata")
            if result["page_count"] < 1 or result["page_count"] > max_pages:
                raise ValueError("invalid inspection page count")
        elif operation == "extract":
            if set(result) != {"pages", "extractor", "page_count"}:
                raise ValueError("invalid extraction metadata")
            pages = result["pages"]
            page_count = result["page_count"]
            if (
                result["extractor"] != "pypdf"
                or not _is_int(page_count)
                or page_count < 1
                or page_count > max_pages
                or not isinstance(pages, list)
                or len(pages) != page_count
                or len(pages) > max_pages
            ):
                raise ValueError("invalid extraction result")
            text_view = memoryview(text_payload)
            text_offset = 0
            decoded_pages: list[dict[str, object]] = []
            for page_number, page_result in enumerate(pages, 1):
                if (
                    not isinstance(page_result, dict)
                    or set(page_result) != {"page", "text_bytes"}
                    or not _is_int(page_result.get("page"))
                    or page_result.get("page") != page_number
                    or not _is_int(page_result.get("text_bytes"))
                    or page_result["text_bytes"] < 0
                ):
                    raise ValueError("invalid extracted page")
                page_end = text_offset + page_result["text_bytes"]
                if page_end > len(text_view):
                    raise ValueError("invalid extracted text length")
                decoder = codecs.getincrementaldecoder("utf-8")("strict")
                decoded_chunks: list[str] = []
                while text_offset < page_end:
                    check_deadline()
                    chunk_end = min(text_offset + WIRE_READ_CHUNK_BYTES, page_end)
                    decoded_chunks.append(
                        decoder.decode(text_view[text_offset:chunk_end], final=False)
                    )
                    text_offset = chunk_end
                    check_deadline()
                decoded_chunks.append(decoder.decode(b"", final=True))
                check_deadline()
                page_text = "".join(decoded_chunks)
                check_deadline()
                decoded_pages.append({"page": page_number, "text": page_text})
            if text_offset != len(text_view):
                raise ValueError("unexpected extracted text")
            result = {
                "pages": decoded_pages,
                "extractor": "pypdf",
                "page_count": page_count,
            }
        elif operation == "render":
            if len(text_payload) != 0:
                raise ValueError("render result included text")
            if set(result) != {"page", "sha256", "renderer"}:
                raise ValueError("invalid render metadata")
            digest = result["sha256"]
            if (
                not _is_int(result["page"])
                or result["page"] != expected_page
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or result["renderer"] != "PyMuPDF"
            ):
                raise ValueError("invalid render result")
        else:
            raise ValueError("unknown worker operation")
        check_deadline()
        return cast(dict[str, object], result)
    except SourceEvidenceError as error:
        if error.stop_reason == "PDF_PARSE_TIMEOUT":
            raise
        check_deadline()
        raise
    except Exception:
        check_deadline()
        raise _worker_failed() from None


def _run_worker(
    operation: WorkerOperation,
    source: SourceArtifact,
    timeout_seconds: float,
    *,
    page: int | None = None,
    render_target: Path | None = None,
) -> dict[str, object]:
    if timeout_seconds <= 0:
        raise SourceEvidenceError(
            ErrorCategory.CONFIGURATION,
            "PDF parse timeout must be positive",
            stop_reason="PDF_TIMEOUT_INVALID",
        )
    if source.source_format != "pdf":
        raise SourceEvidenceError(
            ErrorCategory.SOURCE,
            "PDF worker only accepts PDF sources",
            stop_reason="PDF_FORMAT_INVALID",
        )

    from invoice_agents.source_store import verified_source_path

    verified_source_path(source)
    settings = Settings()
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_main,
        args=(
            child_connection,
            operation,
            cast(dict[str, object], source.model_dump(mode="json")),
            settings.pdf_max_pages,
            settings.pdf_worker_cpu_seconds,
            settings.pdf_worker_memory_bytes,
            settings.pdf_worker_result_max_bytes,
            page,
            str(render_target) if render_target is not None else None,
        ),
        name=f"invoice-pdf-{operation}",
    )
    started = False
    deadline = time.monotonic() + timeout_seconds
    try:
        process.start()
        started = True
        child_connection.close()
        encoded = _receive_bounded_message(
            parent_connection,
            max_bytes=settings.pdf_worker_result_max_bytes,
            deadline=deadline,
        )
        metadata, text_payload = _split_wire_message(encoded)
        result = _decode_worker_message(
            operation,
            metadata,
            expected_page=page,
            max_pages=settings.pdf_max_pages,
            text_payload=text_payload,
            deadline=deadline,
        )
        process.join(max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            _stop_process(process)
            raise _parse_timeout()
        if process.exitcode != 0:
            raise _worker_failed()
        return result
    except BaseException:
        if started:
            _stop_process(process)
        raise
    finally:
        parent_connection.close()
        child_connection.close()


def inspect_pdf_in_worker(source: SourceArtifact, timeout_seconds: float) -> int:
    """Return a verified PDF's page count from a bounded spawned worker."""

    result = _run_worker("inspect", source, timeout_seconds)
    page_count = result.get("page_count")
    if not isinstance(page_count, int):
        raise _worker_failed()
    return page_count


def extract_pdf_in_worker(
    source: SourceArtifact, timeout_seconds: float
) -> dict[str, object]:
    """Extract JSON-safe PDF page text in a bounded spawned worker."""

    return _run_worker("extract", source, timeout_seconds)


def render_pdf_page_in_worker(
    source: SourceArtifact,
    page: int,
    target: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    """Render a page in a child and atomically publish only a completed PNG."""

    resolved = target.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pdf-render-", suffix=".png", dir=resolved.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        result = _run_worker(
            "render",
            source,
            timeout_seconds,
            page=page,
            render_target=temporary,
        )
        if not temporary.is_file():
            raise _worker_failed()
        os.replace(temporary, resolved)
        return {
            "path": str(resolved),
            "page": result["page"],
            "sha256": result["sha256"],
            "renderer": result["renderer"],
        }
    finally:
        temporary.unlink(missing_ok=True)
