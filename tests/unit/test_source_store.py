"""Immutable content-addressed invoice source storage."""

import hashlib
from importlib import import_module
from pathlib import Path

import pytest

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import SourceEvidenceError
from invoice_agents.models import CaseResult
from invoice_agents.orchestration import prepare_case
from invoice_agents.tools.evidence import extract_invoice_evidence

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "invoices"


def source_store() -> object:
    """Load the intended API at test runtime so a missing module is a genuine RED test."""

    return import_module("invoice_agents.source_store")


def test_snapshot_is_content_addressed_and_reuses_verified_bytes(tmp_path: Path) -> None:
    submitted = tmp_path / "submitted.txt"
    submitted.write_bytes(b"immutable invoice bytes\n")
    archive = tmp_path / "archive"

    first = source_store().snapshot_source(submitted, archive, max_bytes=1024)
    second = source_store().snapshot_source(submitted, archive, max_bytes=1024)

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
        source_store().snapshot_source(submitted, tmp_path / "archive", max_bytes=4)

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
        source_store().snapshot_source(submitted, archive, max_bytes=1024)

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
        source_store().snapshot_source(submitted, archive, max_bytes=1024)

    assert caught.value.stop_reason == "SOURCE_HASH_MISMATCH"
    assert target.is_symlink()


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

    reparsed = extract_invoice_evidence(stored.source)
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
        extract_invoice_evidence(invoice.source)

    assert caught.value.stop_reason == "SOURCE_HASH_MISMATCH"
