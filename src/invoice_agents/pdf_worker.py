"""Killable, resource-bounded child-process boundary for untrusted PDFs."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import tempfile
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal, cast

from invoice_agents.config import Settings
from invoice_agents.errors import ErrorCategory, SourceEvidenceError
from invoice_agents.models import SourceArtifact

WorkerOperation = Literal["inspect", "extract", "render"]
JOIN_GRACE_SECONDS = 1.0


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


def _send(connection: Connection, payload: dict[str, object]) -> None:
    """Cross the process boundary as JSON text, never arbitrary pickled results."""

    connection.send(json.dumps(payload))


def _page_limit_error(connection: Connection, page_count: int, max_pages: int) -> None:
    _send(
        connection,
        {
            "ok": False,
            "category": ErrorCategory.PARSE.value,
            "message": f"PDF has {page_count} pages; maximum is {max_pages}",
            "stop_reason": "PDF_PAGE_LIMIT_EXCEEDED",
            "details": {"page_count": page_count, "max_pages": max_pages},
        },
    )


def _worker_main(
    connection: Connection,
    operation: WorkerOperation,
    source_payload: dict[str, object],
    max_pages: int,
    cpu_seconds: int,
    memory_bytes: int,
    page: int | None,
    render_target: str | None,
) -> None:
    """Run exactly one PDF operation; unexpected exceptions crash the worker visibly."""

    try:
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
                _page_limit_error(connection, page_count, max_pages)
                return
            if operation == "inspect":
                _send(connection, {"ok": True, "result": {"page_count": page_count}})
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
            )
            return

        if page is None or render_target is None:
            raise ValueError("render operation requires a page and target")
        import fitz

        document = fitz.open(source_path)
        try:
            page_count = document.page_count
            if page_count > max_pages:
                _page_limit_error(connection, page_count, max_pages)
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
        )
    finally:
        connection.close()


def _stop_process(process: BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
    process.join(JOIN_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join()


def _worker_failed(message: str = "PDF worker exited without a valid result") -> SourceEvidenceError:
    return SourceEvidenceError(
        ErrorCategory.TOOL,
        message,
        stop_reason="PDF_WORKER_FAILED",
    )


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
            page,
            str(render_target) if render_target is not None else None,
        ),
        name=f"invoice-pdf-{operation}",
    )
    started = False
    try:
        process.start()
        started = True
        child_connection.close()
        if not parent_connection.poll(timeout_seconds):
            _stop_process(process)
            raise SourceEvidenceError(
                ErrorCategory.TIMEOUT,
                f"PDF {operation} exceeded the {timeout_seconds:g}-second deadline",
                stop_reason="PDF_PARSE_TIMEOUT",
            )
        try:
            wire_payload = parent_connection.recv()
        except EOFError as exc:
            process.join(JOIN_GRACE_SECONDS)
            raise _worker_failed() from exc
        process.join(JOIN_GRACE_SECONDS)
        if process.is_alive():
            _stop_process(process)
            raise _worker_failed("PDF worker did not exit after returning a result")
        if process.exitcode != 0 or not isinstance(wire_payload, str):
            raise _worker_failed()
        try:
            payload = json.loads(wire_payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise _worker_failed("PDF worker returned malformed JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
            raise _worker_failed("PDF worker returned an invalid result envelope")
        if payload["ok"] is False:
            category = payload.get("category")
            message = payload.get("message")
            stop_reason = payload.get("stop_reason")
            details = payload.get("details")
            if not isinstance(category, str):
                raise _worker_failed("PDF worker returned an invalid error category")
            if not isinstance(message, str) or not isinstance(stop_reason, str):
                raise _worker_failed("PDF worker returned an invalid error envelope")
            if details is not None and not isinstance(details, dict):
                raise _worker_failed("PDF worker returned invalid error details")
            raise SourceEvidenceError(
                ErrorCategory(category),
                message,
                stop_reason=stop_reason,
                details=details,
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise _worker_failed("PDF worker returned invalid result metadata")
        return cast(dict[str, object], result)
    except BaseException:
        if started and process.is_alive():
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
        raise _worker_failed("PDF inspection returned an invalid page count")
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
            raise _worker_failed("PDF render worker returned without a completed image")
        os.replace(temporary, resolved)
        return {
            "path": str(resolved),
            "page": result["page"],
            "sha256": result["sha256"],
            "renderer": result["renderer"],
        }
    finally:
        temporary.unlink(missing_ok=True)
