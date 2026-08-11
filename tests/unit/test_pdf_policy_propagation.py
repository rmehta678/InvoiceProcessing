"""Active PDF policy must cross every parser and worker boundary unchanged."""

from __future__ import annotations

import hashlib
import multiprocessing
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter

from invoice_agents import orchestration
from invoice_agents.config import PdfPolicy, Settings
from invoice_agents.errors import SourceEvidenceError
from invoice_agents.models import CaseResult, SourceArtifact
from invoice_agents.pdf_worker import _decode_worker_message, _receive_bounded_message
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.evidence import extract_pdf_text


@pytest.fixture
def active_pdf_policy() -> PdfPolicy:
    """Return non-default limits that cannot be confused with ambient defaults."""

    return Settings(
        pdf_max_pages=1,
        pdf_parse_timeout_seconds=2.75,
        pdf_worker_cpu_seconds=3,
        pdf_worker_memory_bytes=268_435_456,
        pdf_worker_result_max_bytes=131_072,
    ).pdf_policy()


def write_pdf(path: Path, *, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def content_addressed_pdf(path: Path, archive: Path) -> SourceArtifact:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    archive.mkdir(parents=True)
    target = (archive / f"{digest}.pdf").resolve()
    target.write_bytes(content)
    return SourceArtifact(
        source_id=f"src_{digest}",
        canonical_path=target,
        sha256=digest,
        source_format="pdf",
        size_bytes=len(content),
        modified_at=datetime.now(UTC),
        page_count=None,
    )


def test_pdf_policy_is_exact_typed_and_immutable(active_pdf_policy: PdfPolicy) -> None:
    assert active_pdf_policy.model_dump() == {
        "pdf_max_pages": 1,
        "pdf_parse_timeout_seconds": 2.75,
        "pdf_worker_cpu_seconds": 3,
        "pdf_worker_memory_bytes": 268_435_456,
        "pdf_worker_result_max_bytes": 131_072,
    }
    with pytest.raises(ValidationError):
        active_pdf_policy.pdf_max_pages = 2


def test_snapshot_pdf_inspection_uses_programmatic_policy_not_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_pdf_policy: PdfPolicy,
) -> None:
    submitted = tmp_path / "two-pages.pdf"
    write_pdf(submitted, pages=2)
    monkeypatch.setenv("INVOICE_PDF_MAX_PAGES", "100")

    with pytest.raises(SourceEvidenceError) as excinfo:
        snapshot_source(
            submitted,
            tmp_path / "archive",
            max_bytes=1_048_576,
            pdf_policy=active_pdf_policy,
        )

    assert excinfo.value.stop_reason == "PDF_PAGE_LIMIT_EXCEEDED"
    assert excinfo.value.details == {"page_count": 2, "max_pages": 1}


def test_evidence_pdf_extraction_uses_programmatic_policy_not_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_pdf_policy: PdfPolicy,
) -> None:
    submitted = tmp_path / "two-pages.pdf"
    write_pdf(submitted, pages=2)
    source = content_addressed_pdf(submitted, tmp_path / "archive")
    monkeypatch.setenv("INVOICE_PDF_MAX_PAGES", "100")

    with pytest.raises(SourceEvidenceError) as excinfo:
        extract_pdf_text(source, active_pdf_policy)

    assert excinfo.value.stop_reason == "PDF_PAGE_LIMIT_EXCEEDED"
    assert excinfo.value.details == {"page_count": 2, "max_pages": 1}


def test_child_wire_policy_applies_exact_resource_and_result_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_pdf_policy: PdfPolicy,
) -> None:
    from invoice_agents import pdf_worker

    submitted = tmp_path / "one-page.pdf"
    write_pdf(submitted, pages=1)
    source = content_addressed_pdf(submitted, tmp_path / "archive")
    applied: list[tuple[int, int]] = []
    monkeypatch.setattr(pdf_worker, "_silence_child_output", lambda: None)
    monkeypatch.setattr(
        pdf_worker,
        "_apply_resource_limits",
        lambda cpu, memory: applied.append((cpu, memory)),
    )
    parent, child = multiprocessing.Pipe(duplex=False)
    try:
        pdf_worker._worker_main(
            child,
            "inspect",
            source.model_dump(mode="json"),
            active_pdf_policy.model_dump(),
            None,
            None,
        )
        encoded = _receive_bounded_message(
            parent,
            max_bytes=active_pdf_policy.pdf_worker_result_max_bytes,
            deadline=time.monotonic() + 5.0,
        )
        metadata, text = pdf_worker._split_wire_message(encoded)
        result = _decode_worker_message(
            "inspect",
            metadata,
            expected_page=None,
            max_pages=active_pdf_policy.pdf_max_pages,
            text_payload=text,
        )
    finally:
        parent.close()
        child.close()

    assert applied == [(3, 268_435_456)]
    assert result == {"page_count": 1}


def test_parent_worker_request_contains_only_the_exact_active_pdf_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_pdf_policy: PdfPolicy,
) -> None:
    from invoice_agents import pdf_worker

    submitted = tmp_path / "one-page.pdf"
    write_pdf(submitted, pages=1)
    source = content_addressed_pdf(submitted, tmp_path / "archive")
    captured: dict[str, object] = {}

    def record_request(
        connection: object,
        operation: object,
        source_payload: object,
        policy_payload: object,
        page: object,
        render_target: object,
    ) -> None:
        captured.update(
            {
                "operation": operation,
                "source": source_payload,
                "policy": policy_payload,
                "page": page,
                "render_target": render_target,
            }
        )
        assert isinstance(policy_payload, dict)
        pdf_worker._send(
            connection,
            {"ok": True, "result": {"page_count": 1}},
            policy_payload["pdf_worker_result_max_bytes"],
        )
        connection.close()

    class InlineProcess:
        def __init__(self, *, target: object, args: tuple[object, ...], name: str) -> None:
            self._target = target
            self._args = args
            self.name = name
            self.exitcode: int | None = None

        def start(self) -> None:
            assert callable(self._target)
            self._target(*self._args)
            self.exitcode = 0

        def join(self, _timeout: float | None = None) -> None:
            return None

        def is_alive(self) -> bool:
            return False

    class InlineContext:
        Pipe = staticmethod(multiprocessing.Pipe)
        Process = InlineProcess

    monkeypatch.setattr(pdf_worker, "_worker_main", record_request)
    monkeypatch.setattr(multiprocessing, "get_context", lambda _method: InlineContext())

    result = pdf_worker.inspect_pdf_in_worker(source, active_pdf_policy)

    assert result == 1
    assert captured == {
        "operation": "inspect",
        "source": source.model_dump(mode="json"),
        "policy": active_pdf_policy.model_dump(),
        "page": None,
        "render_target": None,
    }


@pytest.mark.asyncio
async def test_staging_captures_one_immutable_policy_for_inspection_and_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = tmp_path / "two-blank-pages.pdf"
    write_pdf(submitted, pages=2)
    settings = Settings(
        source_archive_dir=tmp_path / "archive",
        pdf_max_pages=2,
        pdf_parse_timeout_seconds=2.75,
        pdf_worker_cpu_seconds=3,
        pdf_worker_memory_bytes=268_435_456,
        pdf_worker_result_max_bytes=131_072,
    )
    real_snapshot_source = snapshot_source

    def snapshot_then_mutate_settings(*args: object, **kwargs: object) -> SourceArtifact:
        source = real_snapshot_source(*args, **kwargs)  # type: ignore[arg-type]
        settings.pdf_max_pages = 1
        return source

    monkeypatch.setattr(orchestration, "preflight", lambda _settings: None)
    monkeypatch.setattr(orchestration, "snapshot_source", snapshot_then_mutate_settings)

    staged = await orchestration.stage_claimed_invoice_async(submitted, settings)

    assert isinstance(staged, CaseResult)
    assert staged.stop_reason == "PDF_TEXT_EMPTY"
