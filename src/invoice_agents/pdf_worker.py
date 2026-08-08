"""Killable, resource-bounded child-process boundary for untrusted PDFs."""

from __future__ import annotations

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
GENERIC_WORKER_MESSAGE = "PDF worker failed"


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


def _write_all(file_descriptor: int, content: bytes) -> None:
    written = 0
    while written < len(content):
        count = os.write(file_descriptor, content[written:])
        if count < 1:
            raise BrokenPipeError("PDF result pipe closed")
        written += count


def _send(
    connection: Connection,
    payload: dict[str, object],
    max_bytes: int,
) -> None:
    """Send one size-capped JSON message using a bounded four-byte frame header."""

    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = json.dumps(_generic_failure_payload(), separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        raise RuntimeError("PDF result ceiling cannot contain the generic failure envelope")
    framed = struct.pack("!I", len(encoded)) + encoded
    _write_all(connection.fileno(), framed)


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
            pages: list[dict[str, object]] = []
            for index, pdf_page in enumerate(reader.pages, 1):
                pages.append({"page": index, "text": pdf_page.extract_text() or ""})
            if not any(str(item["text"]).strip() for item in pages):
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
            _send(
                connection,
                {
                    "ok": True,
                    "result": {
                        "pages": pages,
                        "extractor": "pypdf",
                        "page_count": page_count,
                    },
                },
                result_max_bytes,
            )
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
) -> bytes:
    """Receive exactly one capped frame without blocking past the absolute deadline."""

    file_descriptor = connection.fileno()
    was_blocking = os.get_blocking(file_descriptor)
    selector = selectors.DefaultSelector()
    selector.register(file_descriptor, selectors.EVENT_READ)
    os.set_blocking(file_descriptor, False)

    def read_exact(length: int) -> bytes:
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
        return bytes(received)

    try:
        header = read_exact(WIRE_HEADER_BYTES)
        (message_length,) = struct.unpack("!I", header)
        if message_length < 1 or message_length > max_bytes:
            raise _worker_failed()
        return read_exact(message_length)
    finally:
        selector.close()
        os.set_blocking(file_descriptor, was_blocking)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decode_worker_message(
    operation: WorkerOperation,
    encoded: bytes,
    *,
    expected_page: int | None,
) -> dict[str, object]:
    """Validate the complete common and operation-specific JSON envelope."""

    try:
        payload = json.loads(encoded.decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
            raise ValueError("invalid common envelope")
        if payload["ok"] is False:
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
            if set(result) != {"page_count"} or not _is_int(result.get("page_count")):
                raise ValueError("invalid inspection metadata")
            if result["page_count"] < 1:
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
                or not isinstance(pages, list)
                or len(pages) != page_count
            ):
                raise ValueError("invalid extraction result")
            for page_number, page_result in enumerate(pages, 1):
                if (
                    not isinstance(page_result, dict)
                    or set(page_result) != {"page", "text"}
                    or page_result.get("page") != page_number
                    or not isinstance(page_result.get("text"), str)
                ):
                    raise ValueError("invalid extracted page")
        elif operation == "render":
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
        return cast(dict[str, object], result)
    except SourceEvidenceError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
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
        try:
            result = _decode_worker_message(
                operation,
                encoded,
                expected_page=page,
            )
        except SourceEvidenceError:
            if time.monotonic() > deadline:
                raise _parse_timeout() from None
            raise
        if time.monotonic() > deadline:
            raise _parse_timeout()
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
