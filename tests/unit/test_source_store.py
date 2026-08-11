"""Immutable content-addressed invoice source storage."""

import hashlib
import os
from importlib import import_module
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import SourceEvidenceError
from invoice_agents.models import CaseResult
from invoice_agents.orchestration import prepare_case
from invoice_agents.tools.evidence import (
    extract_invoice_evidence,
    read_csv_invoice,
    read_json_invoice,
    read_text_invoice,
    read_xml_invoice,
)
from tests.support.pdf_policy import TEST_PDF_POLICY

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "invoices"


def source_store() -> object:
    """Load the intended API at test runtime so a missing module is a genuine RED test."""

    return import_module("invoice_agents.source_store")


def test_snapshot_is_content_addressed_and_reuses_verified_bytes(tmp_path: Path) -> None:
    submitted = tmp_path / "submitted.txt"
    submitted.write_bytes(b"immutable invoice bytes\n")
    archive = tmp_path / "archive"

    first = source_store().snapshot_source(
        submitted,
        archive,
        max_bytes=1024,
        pdf_policy=TEST_PDF_POLICY,
    )
    second = source_store().snapshot_source(
        submitted,
        archive,
        max_bytes=1024,
        pdf_policy=TEST_PDF_POLICY,
    )

    expected_hash = "342f9daff6a356886ecce3a65945bf67845c3397b35f993ea723a7b261b25b48"
    assert first.sha256 == expected_hash
    assert first.canonical_path == (archive / f"{expected_hash}.txt").resolve()
    assert first.canonical_path.read_bytes() == b"immutable invoice bytes\n"
    assert second == first
    assert [path.name for path in archive.iterdir()] == [f"{expected_hash}.txt"]


def test_snapshot_rejects_a_source_over_the_byte_ceiling(tmp_path: Path) -> None:
    submitted = tmp_path / "oversized.json"
    submitted.write_bytes(b"12345")

    with pytest.raises(SourceEvidenceError) as caught:
        source_store().snapshot_source(
            submitted,
            tmp_path / "archive",
            max_bytes=4,
            pdf_policy=TEST_PDF_POLICY,
        )

    assert caught.value.stop_reason == "SOURCE_TOO_LARGE"
    assert not list((tmp_path / "archive").iterdir())


def test_snapshot_refuses_to_reuse_changed_content_address(tmp_path: Path) -> None:
    submitted = tmp_path / "submitted.txt"
    submitted.write_bytes(b"trusted")
    expected_hash = hashlib.sha256(b"trusted").hexdigest()
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / f"{expected_hash}.txt").write_bytes(b"changed")

    with pytest.raises(SourceEvidenceError) as caught:
        source_store().snapshot_source(
            submitted,
            archive,
            max_bytes=1024,
            pdf_policy=TEST_PDF_POLICY,
        )

    assert caught.value.stop_reason == "SOURCE_HASH_MISMATCH"
    assert (archive / f"{expected_hash}.txt").read_bytes() == b"changed"


def test_snapshot_refuses_to_replace_a_broken_symlink_at_content_address(
    tmp_path: Path,
) -> None:
    submitted = tmp_path / "submitted.txt"
    submitted.write_bytes(b"trusted")
    expected_hash = hashlib.sha256(b"trusted").hexdigest()
    archive = tmp_path / "archive"
    archive.mkdir()
    target = archive / f"{expected_hash}.txt"
    target.symlink_to(archive / "missing.txt")

    with pytest.raises(SourceEvidenceError) as caught:
        source_store().snapshot_source(
            submitted,
            archive,
            max_bytes=1024,
            pdf_policy=TEST_PDF_POLICY,
        )

    assert caught.value.stop_reason == "SOURCE_HASH_MISMATCH"
    assert target.is_symlink()


def test_atomic_publication_does_not_clobber_a_conflicting_race_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = tmp_path / "submitted.txt"
    submitted.write_bytes(b"trusted")
    expected_hash = hashlib.sha256(b"trusted").hexdigest()
    archive = tmp_path / "archive"
    target = archive / f"{expected_hash}.txt"
    conflicting_bytes = b"race winner must remain"
    real_replace = os.replace
    real_link = os.link

    def replace_after_race(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(conflicting_bytes)
        real_replace(source, destination)

    def link_after_race(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(conflicting_bytes)
        real_link(source, destination)

    monkeypatch.setattr(os, "replace", replace_after_race)
    monkeypatch.setattr(os, "link", link_after_race)

    with pytest.raises(SourceEvidenceError) as caught:
        source_store().snapshot_source(
            submitted,
            archive,
            max_bytes=1024,
            pdf_policy=TEST_PDF_POLICY,
        )

    assert caught.value.stop_reason == "SOURCE_HASH_MISMATCH"
    assert target.read_bytes() == conflicting_bytes


def test_prepare_case_extracts_archived_bytes_after_submitted_path_changes(
    tmp_path: Path, settings: Settings
) -> None:
    submitted = tmp_path / "invoice.txt"
    original = (DATA_DIR / "invoice_1001.txt").read_bytes()
    submitted.write_bytes(original)

    prepared = prepare_case(submitted, settings)
    assert not isinstance(prepared, CaseResult), prepared
    case_id, _started_at = prepared
    stored = WorkflowStore(settings.workflow_db).load_extraction(case_id)
    submitted.write_bytes((DATA_DIR / "invoice_1002.txt").read_bytes())

    reparsed = extract_invoice_evidence(stored.source, settings.pdf_policy())
    assert stored.source.canonical_path != submitted.resolve()
    assert reparsed.invoice_number.normalized_value == "INV-1001"
    assert stored.source.canonical_path.read_bytes() == original
    assert (
        hashlib.sha256(stored.source.canonical_path.read_bytes()).hexdigest()
        == stored.source.sha256
    )


@pytest.mark.parametrize("mutation", ["changed", "missing"])
def test_verified_reader_rejects_changed_or_missing_snapshot(
    tmp_path: Path, settings: Settings, mutation: str
) -> None:
    submitted = tmp_path / "invoice.json"
    submitted.write_bytes((DATA_DIR / "invoice_1004.json").read_bytes())
    prepared = prepare_case(submitted, settings)
    assert not isinstance(prepared, CaseResult), prepared
    invoice = WorkflowStore(settings.workflow_db).load_extraction(prepared[0])

    if mutation == "changed":
        invoice.source.canonical_path.write_bytes(b'{"invoice_number":"invented"}')
    else:
        invoice.source.canonical_path.unlink()

    with pytest.raises(SourceEvidenceError) as caught:
        extract_invoice_evidence(invoice.source, settings.pdf_policy())

    assert caught.value.stop_reason == "SOURCE_HASH_MISMATCH"


def test_text_reader_parses_the_same_bytes_that_passed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"Invoice Number: INV-ORIGINAL\nTotal: $1.00\n"
    swapped = b"Invoice Number: INV-SWAPPED\nTotal: $999.00\n"
    submitted = tmp_path / "invoice.txt"
    submitted.write_bytes(original)
    source = source_store().snapshot_source(
        submitted,
        tmp_path / "archive",
        max_bytes=1024,
        pdf_policy=TEST_PDF_POLICY,
    )
    real_read_text = Path.read_text

    def swap_before_reopen(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == source.canonical_path:
            path.write_bytes(swapped)
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", swap_before_reopen)

    parsed = read_text_invoice(source)

    assert "INV-ORIGINAL" in parsed["raw_text"]
    assert "INV-SWAPPED" not in parsed["raw_text"]


def test_json_reader_parses_the_same_bytes_that_passed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b'{"invoice_number":"INV-ORIGINAL"}'
    swapped = b'{"invoice_number":"INV-SWAPPED"}'
    submitted = tmp_path / "invoice.json"
    submitted.write_bytes(original)
    source = source_store().snapshot_source(
        submitted,
        tmp_path / "archive",
        max_bytes=1024,
        pdf_policy=TEST_PDF_POLICY,
    )
    real_read_text = Path.read_text

    def swap_before_reopen(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == source.canonical_path:
            path.write_bytes(swapped)
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", swap_before_reopen)

    parsed = read_json_invoice(source)

    assert parsed["invoice_number"] == "INV-ORIGINAL"


def test_csv_reader_parses_the_same_bytes_that_passed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"invoice_number,total\nINV-ORIGINAL,1.00\n"
    swapped = b"invoice_number,total\nINV-SWAPPED,999.00\n"
    submitted = tmp_path / "invoice.csv"
    submitted.write_bytes(original)
    source = source_store().snapshot_source(
        submitted,
        tmp_path / "archive",
        max_bytes=1024,
        pdf_policy=TEST_PDF_POLICY,
    )
    real_open = Path.open

    def swap_before_reopen(
        path: Path,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if path == source.canonical_path and mode == "r":
            path.write_bytes(swapped)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_before_reopen)

    parsed = read_csv_invoice(source)

    assert parsed["rows"][1]["values"] == ["INV-ORIGINAL", "1.00"]


def test_xml_reader_parses_the_same_bytes_that_passed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"<invoice><invoice_number>INV-ORIGINAL</invoice_number></invoice>"
    swapped = b"<invoice><invoice_number>INV-SWAPPED</invoice_number></invoice>"
    submitted = tmp_path / "invoice.xml"
    submitted.write_bytes(original)
    source = source_store().snapshot_source(
        submitted,
        tmp_path / "archive",
        max_bytes=1024,
        pdf_policy=TEST_PDF_POLICY,
    )
    real_parse = ElementTree.parse

    def swap_before_reopen(path: Path, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == source.canonical_path:
            source.canonical_path.write_bytes(swapped)
        return real_parse(path, *args, **kwargs)

    monkeypatch.setattr(ElementTree, "parse", swap_before_reopen)

    parsed = read_xml_invoice(source)

    assert parsed["nodes"][1]["value"] == "INV-ORIGINAL"


def test_csv_inspection_counts_the_same_bytes_that_passed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"invoice_number,total\nINV-ORIGINAL,1.00\n"
    swapped = b"invoice_number,total\nINV-SWAPPED,999.00\nextra,row\n"
    target_hash = hashlib.sha256(original).hexdigest()
    target = (tmp_path / f"{target_hash}.csv").resolve()
    target.write_bytes(original)
    real_open = Path.open

    def swap_before_reopen(
        path: Path,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if path == target and mode == "r":
            path.write_bytes(swapped)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_before_reopen)

    inspected = source_store().inspect_snapshot(
        target,
        expected_hash=target_hash,
        size_bytes=len(original),
        pdf_policy=TEST_PDF_POLICY,
    )

    assert inspected.row_count == 2
