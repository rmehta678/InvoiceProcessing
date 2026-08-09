"""Workflow migration 002: review sequencing, v2 verification, and required indexes."""

import hashlib
import json
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import invoice_agents.db.core as core_module
from invoice_agents.agents.decision_rules import blocking_evidence
from invoice_agents.config import Settings
from invoice_agents.db.core import (
    DatabaseKind,
    _migration_resources,
    connect_database,
    migrate_database,
    reconcile_legacy_authorization,
    seed_inventory,
    verify_database,
)
from invoice_agents.db.legacy_archive import LEGACY_NON_AUTHORIZING_DISPOSITION
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import DatabaseVerificationError, InvoiceAgentsError
from invoice_agents.hitl.service import record_human_decision
from invoice_agents.models import (
    CaseStatus,
    Critique,
    DecisionKind,
    ExtractedInvoice,
    HumanDecisionKind,
    Money,
    ReviewRequest,
    RiskAssessment,
    SourceArtifact,
)
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.comparison import (
    InventoryReader,
    apply_mapping_evidence,
    build_risk_assessment,
    compare_inventory_evidence,
    compute_invoice_totals,
    find_prior_invoice_candidates,
)
from invoice_agents.tools.evidence import extract_invoice_evidence

CASE_ID = "case_v1_legacy"
REVIEW_ID = "rev_v1_legacy"
LEGACY_AT = datetime(2026, 1, 1, tzinfo=UTC)
LEGACY_SOURCE = Path(__file__).resolve().parents[2] / "data" / "invoices" / "invoice_1001.txt"


def make_critique() -> Critique:
    return Critique(
        supported_findings=["legacy evidence"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=DecisionKind.HOLD,
        rationale=["legacy rationale"],
    )


def make_source(archive_dir: Path) -> SourceArtifact:
    return snapshot_source(LEGACY_SOURCE, archive_dir, max_bytes=10_485_760)


def make_review(review_id: str, created_at: datetime, source: SourceArtifact) -> ReviewRequest:
    return ReviewRequest(
        review_id=review_id,
        case_id=CASE_ID,
        status="PENDING",
        reasons=["legacy policy trigger"],
        amount=None,
        source=source,
        evidence_bundle={},
        agent_recommendation=DecisionKind.HOLD,
        agent_rationale=["legacy review"],
        critic=make_critique(),
        critic_disagreement_reason=None,
        questions=["Does the legacy evidence support this decision?"],
        created_at=created_at,
    )


def make_invoice(source: SourceArtifact) -> ExtractedInvoice:
    return extract_invoice_evidence(source)


def persist_bound_review_evidence(
    store: WorkflowStore,
    claim: ExecutionClaim,
    settings: Settings,
    source: SourceArtifact,
) -> tuple[ExtractedInvoice, RiskAssessment, Critique]:
    source_invoice = make_invoice(source)
    mappings, comparisons, unresolved = compare_inventory_evidence(
        source_invoice, InventoryReader(settings.inventory_db)
    )
    invoice = apply_mapping_evidence(source_invoice, mappings, unresolved)
    identity = find_prior_invoice_candidates(CASE_ID, invoice, store)
    risk = build_risk_assessment(
        invoice,
        comparisons,
        identity,
        compute_invoice_totals(invoice),
        settings,
    )
    case_critique = make_critique()
    store.save_extraction(CASE_ID, invoice, claim)
    store.save_identity(
        CASE_ID,
        [candidate.model_dump(mode="json") for candidate in identity],
        claim,
    )
    store.save_comparison(
        CASE_ID,
        "inventory",
        {
            "comparisons": [comparison.model_dump(mode="json") for comparison in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        },
        claim,
    )
    store.save_comparison(CASE_ID, "risk", risk.model_dump(mode="json"), claim)
    store.save_critique(CASE_ID, case_critique, claim)
    return invoice, risk, case_critique


def make_bound_review(
    review_id: str,
    created_at: datetime,
    invoice: ExtractedInvoice,
    risk: RiskAssessment,
    case_critique: Critique,
) -> ReviewRequest:
    return ReviewRequest(
        review_id=review_id,
        case_id=CASE_ID,
        status="PENDING",
        reasons=["legacy policy trigger"],
        amount=(
            Money(amount=invoice.declared_total, currency=invoice.currency.normalized_value)
            if invoice.declared_total is not None and invoice.currency.normalized_value is not None
            else None
        ),
        source=invoice.source,
        evidence_bundle={
            "invoice": invoice.model_dump(mode="json"),
            "financial": risk.financial.model_dump(mode="json"),
            "inventory": [item.model_dump(mode="json") for item in risk.inventory],
            "identity_candidates": [
                item.model_dump(mode="json") for item in risk.identity_candidates
            ],
            "dates": [item.model_dump(mode="json") for item in risk.dates],
            "suspicious_signals": risk.suspicious_signals,
            "unavailable_reconciliations": risk.unavailable_reconciliations,
            "blocking_evidence": [item.model_dump(mode="json") for item in blocking_evidence(risk)],
            "rendered_pages": [],
        },
        agent_recommendation=DecisionKind.HOLD,
        agent_rationale=["legacy review"],
        critic=case_critique,
        critic_disagreement_reason=None,
        questions=["Does the legacy evidence support this decision?"],
        created_at=created_at,
    )


def v1_review_payload(source: SourceArtifact) -> str:
    """A valid pre-schema-v2 review payload: no 'sequence' key at all."""

    payload = make_review(REVIEW_ID, LEGACY_AT, source).model_dump(mode="json")
    del payload["sequence"]
    return json.dumps(payload)


def build_v1_workflow_db(tmp_path: Path) -> tuple[Path, SourceArtifact]:
    """Apply only 001_initial.sql and populate case, review, and human-decision rows."""

    path = tmp_path / "workflow_v1.db"
    source = make_source(tmp_path / "sources")
    script = _migration_resources(DatabaseKind.WORKFLOW)[0].read_text(encoding="utf-8")
    at = LEGACY_AT.isoformat()
    with connect_database(path) as connection:
        connection.executescript(script)
        connection.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO schema_version(version, applied_at) VALUES (1, ?)", (at,))
        connection.execute(
            "INSERT INTO source_artifacts("
            "source_id, canonical_path, source_hash, source_format, size_bytes, modified_at, "
            "metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source.source_id,
                str(source.canonical_path),
                source.sha256,
                source.source_format,
                source.size_bytes,
                source.modified_at.isoformat(),
                source.model_dump_json(),
                at,
            ),
        )
        connection.execute(
            "INSERT INTO cases(case_id, source_id, status, stop_reason, started_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                CASE_ID,
                source.source_id,
                "NEEDS_HUMAN",
                "HUMAN_REVIEW_REQUESTED",
                at,
                at,
            ),
        )
        connection.execute(
            "INSERT INTO review_requests(review_id, case_id, status, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (REVIEW_ID, CASE_ID, "PENDING", v1_review_payload(source), at),
        )
        connection.execute(
            "INSERT INTO human_decisions("
            "decision_id, review_id, reviewer, decision, reason, payload_json, decided_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "hdec_v1_legacy",
                REVIEW_ID,
                "reviewer@example.com",
                "REQUEST_CORRECTION",
                "legacy correction request",
                "{}",
                at,
            ),
        )
        connection.commit()
    return path, source


def test_v1_database_requires_explicit_legacy_authorization_reconciliation(
    tmp_path: Path,
) -> None:
    path, _source = build_v1_workflow_db(tmp_path)
    inventory_db = tmp_path / "inventory.db"
    migrate_database(inventory_db, DatabaseKind.INVENTORY)
    seed_inventory(inventory_db)
    settings = Settings(workflow_db=path, inventory_db=inventory_db)
    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(path, DatabaseKind.WORKFLOW, settings=settings)
    assert excinfo.value.stop_reason == "DATABASE_VERSION_MISMATCH"
    with pytest.raises(DatabaseVerificationError) as migration_error:
        migrate_database(path, DatabaseKind.WORKFLOW)
    assert migration_error.value.stop_reason == "AUTHORIZATION_RECONCILIATION_REQUIRED"
    assert migration_error.value.details == {
        "review_request_count": 1,
        "human_decision_count": 1,
        "final_decision_count": 0,
        "payment_count": 0,
    }
    receipt = reconcile_legacy_authorization(
        path,
        reviewer="legacy-auditor@example.com",
        reason="legacy review rows do not carry generation-bound evidence",
        disposition=LEGACY_NON_AUTHORIZING_DISPOSITION,
        confirmed=True,
    )
    assert receipt.record_count == 2
    assert migrate_database(path, DatabaseKind.WORKFLOW) == [3]
    report = verify_database(path, DatabaseKind.WORKFLOW, settings=settings)
    assert report["schema_version"] == 3
    with connect_database(path, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM review_requests").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM human_decisions").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM legacy_authorization_quarantine").fetchone()[0]
            == 2
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_second_review_cycle_is_sequenced_and_duplicates_are_rejected(
    settings: Settings,
) -> None:
    path = settings.workflow_db
    source = make_source(settings.source_archive_dir)
    store = WorkflowStore(settings)
    store.register_source(source)
    store.create_case(CASE_ID, source, LEGACY_AT)
    claim = store.claim_case_execution(
        CASE_ID, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    invoice, risk, case_critique = persist_bound_review_evidence(store, claim, settings, source)
    first = store.save_review(
        make_bound_review(
            "rev_v3_first", datetime(2026, 1, 15, tzinfo=UTC), invoice, risk, case_critique
        ),
        claim,
    )
    saved = store.save_review(
        make_bound_review(
            "rev_v2_cycle", datetime(2026, 2, 1, tzinfo=UTC), invoice, risk, case_critique
        ),
        claim,
    )
    store.release_case_execution(claim)
    assert first.sequence == 1
    assert saved.sequence == 2
    latest = store.load_case_review(CASE_ID)
    assert latest is not None
    assert latest.review_id == "rev_v2_cycle"
    assert latest.sequence == 2
    ordered = store.list_reviews(pending_only=False)
    assert [review.review_id for review in ordered] == ["rev_v3_first", "rev_v2_cycle"]
    assert [review.sequence for review in ordered] == [1, 2]
    with connect_database(path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO review_requests("
            "review_id, case_id, sequence, status, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "rev_duplicate",
                CASE_ID,
                2,
                "PENDING",
                "{}",
                datetime(2026, 2, 2, tzinfo=UTC).isoformat(),
            ),
        )


def test_missing_required_indexes_fail_verification(workflow_db: Path, inventory_db: Path) -> None:
    settings = Settings(workflow_db=workflow_db, inventory_db=inventory_db)
    with connect_database(workflow_db) as connection:
        connection.execute("DROP INDEX idx_events_case_created")
        connection.commit()
    with pytest.raises(DatabaseVerificationError, match="idx_events_case_created") as workflow_exc:
        verify_database(workflow_db, DatabaseKind.WORKFLOW, settings=settings)
    assert workflow_exc.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"

    with connect_database(inventory_db) as connection:
        connection.execute("DROP INDEX idx_item_aliases_sku")
        connection.commit()
    with pytest.raises(DatabaseVerificationError, match="idx_item_aliases_sku") as inventory_exc:
        verify_database(inventory_db, DatabaseKind.INVENTORY)
    assert inventory_exc.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"


@pytest.mark.parametrize(
    ("versions", "primary_key"),
    [
        ([3], True),
        ([1, 3], True),
        ([1, 1], False),
        ([1, 2, 3, 4], False),
    ],
    ids=["missing-prefix", "sparse", "duplicate", "unknown"],
)
def test_migration_rejects_noncontiguous_history_before_any_database_mutation(
    tmp_path: Path,
    versions: list[int],
    primary_key: bool,
) -> None:
    path = tmp_path / "invalid-history.db"
    with sqlite3.connect(path) as connection:
        key_sql = " PRIMARY KEY" if primary_key else ""
        connection.execute(
            f"CREATE TABLE schema_version (version INTEGER{key_sql}, applied_at TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            [
                (version, f"2026-08-08T00:00:0{index}+00:00")
                for index, version in enumerate(versions)
            ],
        )
        connection.execute("CREATE TABLE untouched (value BLOB)")
        connection.execute("INSERT INTO untouched(value) VALUES (?)", (b"opaque-before-migration",))
        connection.commit()
    before_bytes = path.read_bytes()
    before_digest = hashlib.sha256(before_bytes).hexdigest()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert path.read_bytes() == before_bytes
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_digest


def test_legacy_migration_history_is_a_unique_set_prefix_not_insertion_order() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            "CREATE TABLE schema_version("
            "insertion_id INTEGER PRIMARY KEY, version INTEGER UNIQUE, applied_at TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            ((2, "legacy-second"), (1, "legacy-first")),
        )

        assert core_module._read_migration_history(
            connection,
            kind=DatabaseKind.WORKFLOW,
            packaged_versions=(1, 2, 3),
        ) == (1, 2)
    finally:
        connection.close()


def _packaged_workflow_hashes() -> dict[int, str]:
    return {
        int(resource.name.split("_", 1)[0]): hashlib.sha256(resource.read_bytes()).hexdigest()
        for resource in _migration_resources(DatabaseKind.WORKFLOW)
    }


def _durable_history_rows(path: Path) -> list[tuple[int, int, str, str]]:
    with connect_database(path, read_only=True) as connection:
        return [
            tuple(row)
            for row in connection.execute(
                "SELECT ordinal, version, migration_sha256, applied_at "
                "FROM schema_migration_history ORDER BY ordinal"
            )
        ]


def _remove_durable_history(path: Path) -> None:
    with connect_database(path) as connection:
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'trg_schema_migration_history_%'"
        ).fetchall():
            connection.execute(f'DROP TRIGGER "{row["name"]}"')
        connection.execute("DROP TABLE schema_migration_history")
        connection.commit()


def _retrofit_settings(tmp_path: Path, workflow_db: Path) -> Settings:
    inventory_db = tmp_path / "retrofit-inventory.db"
    migrate_database(inventory_db, DatabaseKind.INVENTORY)
    seed_inventory(inventory_db)
    return Settings(workflow_db=workflow_db, inventory_db=inventory_db)


def _directory_file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def _run_rival_database_operation(path: Path, operation: str) -> str:
    script = """
import sqlite3
import sys

path, operation = sys.argv[1:]
connection = sqlite3.connect(path, timeout=0.1)
try:
    if operation == "wal":
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        print(f"CHANGED:{mode}")
    else:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE schema_version SET applied_at = applied_at WHERE version = 1"
        )
        connection.commit()
        print("CHANGED:write")
except sqlite3.OperationalError as exc:
    print(f"LOCKED:{exc}")
finally:
    connection.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path), operation],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip()


def test_migration_003_backfills_digest_bound_immutable_durable_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable-history.db"

    assert migrate_database(path, DatabaseKind.WORKFLOW) == [1, 2, 3]

    rows = _durable_history_rows(path)
    hashes = _packaged_workflow_hashes()
    assert [(ordinal, version, digest) for ordinal, version, digest, _at in rows] == [
        (version, version, hashes[version]) for version in (1, 2, 3)
    ]
    for _ordinal, _version, _digest, applied_at in rows:
        parsed = datetime.fromisoformat(applied_at)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == UTC.utcoffset(parsed)
        assert parsed.isoformat() == applied_at
    with connect_database(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="MIGRATION_HISTORY_IMMUTABLE"):
            connection.execute(
                "UPDATE schema_migration_history SET migration_sha256 = ? WHERE version = 1",
                ("f" * 64,),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="MIGRATION_HISTORY_IMMUTABLE"):
            connection.execute("DELETE FROM schema_migration_history WHERE version = 1")


def test_existing_legitimate_v3_history_is_retrofitted_without_version_004(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing-v3-history.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    settings = _retrofit_settings(tmp_path, path)

    assert migrate_database(path, DatabaseKind.WORKFLOW, settings=settings) == []
    assert [row[:3] for row in _durable_history_rows(path)] == [
        (version, version, _packaged_workflow_hashes()[version]) for version in (1, 2, 3)
    ]
    assert verify_database(path, DatabaseKind.WORKFLOW, settings=settings)["schema_version"] == 3
    assert not any(
        resource.name.startswith("004_") for resource in _migration_resources(DatabaseKind.WORKFLOW)
    )


def test_legacy_v3_retrofit_rejects_main_file_wal_header_without_artifacts_or_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-v3-main-only-wal.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    settings = _retrofit_settings(tmp_path, path)
    with connect_database(path) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    assert path.read_bytes()[18:20] == b"\x02\x02"
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
    before = _directory_file_hashes(tmp_path)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW, settings=settings)

    assert excinfo.value.stop_reason == "WORKFLOW_WAL_MODE_UNSUPPORTED"
    assert _directory_file_hashes(tmp_path) == before


def test_legacy_v3_retrofit_rejects_wal_without_touching_existing_sidecars(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-v3-existing-wal-sidecars.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    settings = _retrofit_settings(tmp_path, path)
    keeper = sqlite3.connect(path)
    try:
        assert keeper.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
        keeper.execute("PRAGMA wal_autocheckpoint = 0")
        keeper.execute("UPDATE schema_version SET applied_at = applied_at WHERE version = 1")
        keeper.commit()
        assert path.read_bytes()[18:20] == b"\x02\x02"
        assert Path(f"{path}-wal").is_file()
        assert Path(f"{path}-shm").is_file()
        before = _directory_file_hashes(tmp_path)

        with pytest.raises(DatabaseVerificationError) as excinfo:
            migrate_database(path, DatabaseKind.WORKFLOW, settings=settings)

        assert excinfo.value.stop_reason == "WORKFLOW_WAL_MODE_UNSUPPORTED"
        assert _directory_file_hashes(tmp_path) == before
    finally:
        keeper.close()


@pytest.mark.parametrize("operation", ["wal", "write"])
def test_legacy_v3_retrofit_holds_cross_process_sqlite_lock_before_begin_immediate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    path = tmp_path / f"legacy-v3-cross-process-{operation}.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    settings = _retrofit_settings(tmp_path, path)
    real_connect = core_module.connect_database
    rival_results: list[str] = []

    class RivalProbeConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self.connection, name)

        def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
            if sql == "BEGIN IMMEDIATE" and not rival_results:
                rival_results.append(_run_rival_database_operation(path, operation))
            return self.connection.execute(sql, parameters)

    @contextmanager
    def observed_connect(target: Path, *, read_only: bool = False) -> Iterator[Any]:
        with real_connect(target, read_only=read_only) as connection:
            if target.resolve() == path.resolve() and not read_only:
                yield RivalProbeConnection(connection)
            else:
                yield connection

    monkeypatch.setattr(core_module, "connect_database", observed_connect)

    assert migrate_database(path, DatabaseKind.WORKFLOW, settings=settings) == []
    assert len(rival_results) == 1
    assert rival_results[0].startswith("LOCKED:database is locked")
    assert [row[:3] for row in _durable_history_rows(path)] == [
        (version, version, _packaged_workflow_hashes()[version]) for version in (1, 2, 3)
    ]


def test_legacy_v3_retrofit_aborts_when_same_process_raw_sqlite_switches_to_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "legacy-v3-same-process-wal-race.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    settings = _retrofit_settings(tmp_path, path)
    real_connect = core_module.connect_database
    keeper: sqlite3.Connection | None = None

    @contextmanager
    def racing_connect(target: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
        nonlocal keeper
        if target.resolve() == path.resolve() and not read_only and keeper is None:
            keeper = sqlite3.connect(path)
            assert keeper.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
            keeper.execute("UPDATE schema_version SET applied_at = applied_at WHERE version = 1")
            keeper.commit()
            assert Path(f"{path}-wal").is_file()
            assert Path(f"{path}-shm").is_file()
        with real_connect(target, read_only=read_only) as connection:
            yield connection

    monkeypatch.setattr(core_module, "connect_database", racing_connect)

    try:
        with pytest.raises(DatabaseVerificationError) as excinfo:
            migrate_database(path, DatabaseKind.WORKFLOW, settings=settings)

        assert excinfo.value.stop_reason == "WORKFLOW_WAL_MODE_UNSUPPORTED"
        assert keeper is not None
        assert Path(f"{path}-wal").is_file()
        assert Path(f"{path}-shm").is_file()
        assert (
            keeper.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'table' AND name = 'schema_migration_history'"
            ).fetchone()[0]
            == 0
        )
    finally:
        if keeper is not None:
            keeper.close()


def test_legacy_v3_retrofit_rejects_missing_required_trigger_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "false-v3-missing-trigger.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    settings = _retrofit_settings(tmp_path, path)
    with connect_database(path) as connection:
        connection.execute("DROP TRIGGER trg_payments_authorization_insert")
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW, settings=settings)

    assert excinfo.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"
    assert path.read_bytes() == before


def test_legacy_v3_retrofit_requires_explicit_authorization_context_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-v3-context-required.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW)

    assert excinfo.value.stop_reason == "DATABASE_AUTHORIZATION_CONTEXT_REQUIRED"
    assert path.read_bytes() == before


def test_legacy_v3_retrofit_rejects_invalid_authorization_before_history_install(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-v3-invalid-authorization.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    settings = _retrofit_settings(tmp_path, path)
    at = "2026-08-09T12:00:00+00:00"
    with connect_database(path) as connection:
        connection.execute(
            "INSERT INTO source_artifacts(source_id, canonical_path, source_hash, source_format, "
            "size_bytes, modified_at, metadata_json, created_at) "
            "VALUES ('src_invalid_retrofit', '/invalid/retrofit.txt', ?, 'txt', 1, ?, '{}', ?)",
            ("a" * 64, at, at),
        )
        connection.execute(
            "INSERT INTO cases(case_id, source_id, status, started_at, updated_at) "
            "VALUES ('case_invalid_retrofit', 'src_invalid_retrofit', 'INCOMPLETE', ?, ?)",
            (at, at),
        )
        connection.execute(
            "INSERT INTO review_requests(review_id, case_id, sequence, status, payload_json, "
            "created_at, execution_generation, evidence_snapshot_digest) "
            "VALUES ('review_invalid_retrofit', 'case_invalid_retrofit', 1, 'PENDING', "
            "'{}', ?, 1, ?)",
            (at, "b" * 64),
        )
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW, settings=settings)

    assert excinfo.value.stop_reason == "DATABASE_AUTHORIZATION_PROVENANCE_INVALID"
    assert excinfo.value.details["invalid_review_count"] == 1
    assert path.read_bytes() == before


def test_legacy_v3_retrofit_revalidates_contract_after_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "legacy-v3-retrofit-race.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    _remove_durable_history(path)
    settings = _retrofit_settings(tmp_path, path)
    real_preflight = core_module._preflight_existing_migration_history
    raced_bytes: list[bytes] = []

    def mutate_after_preflight(*args: object, **kwargs: object) -> object:
        result = real_preflight(*args, **kwargs)  # type: ignore[arg-type]
        with connect_database(path) as connection:
            connection.execute("DROP TRIGGER trg_payments_authorization_insert")
            connection.commit()
        raced_bytes.append(path.read_bytes())
        return result

    monkeypatch.setattr(
        core_module,
        "_preflight_existing_migration_history",
        mutate_after_preflight,
    )

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW, settings=settings)

    assert excinfo.value.stop_reason == "DATABASE_CHANGED_DURING_VERIFICATION"
    assert len(raced_bytes) == 1
    assert path.read_bytes() == raced_bytes[0]


def test_legacy_v3_retrofit_rereads_durable_history_after_write_lock_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "legacy-v3-durable-history-race.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    settings = _retrofit_settings(tmp_path, path)
    with connect_database(path) as connection:
        connection.execute("DROP TABLE legacy_authorization_quarantine")
        connection.execute("DROP TABLE legacy_authorization_reconciliations")
        connection.commit()

    real_connect = core_module.connect_database
    race_completed = False
    raced_bytes: list[bytes] = []
    post_race_mutations: list[str] = []

    class RacingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self.connection, name)

        def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
            nonlocal race_completed
            if sql == "BEGIN IMMEDIATE" and not race_completed:
                with real_connect(path) as rival:
                    trigger_name = "trg_schema_migration_history_immutable_update"
                    trigger_sql = str(
                        rival.execute(
                            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                            (trigger_name,),
                        ).fetchone()[0]
                    )
                    rival.execute(f"DROP TRIGGER {trigger_name}")
                    rival.execute(
                        "UPDATE schema_migration_history SET applied_at = ? WHERE version = 2",
                        ("2030-01-01T00:00:00+00:00",),
                    )
                    rival.execute(trigger_sql)
                    rival.commit()
                raced_bytes.append(path.read_bytes())
                race_completed = True
            elif race_completed and sql.lstrip().upper().startswith(
                ("ALTER ", "CREATE ", "DELETE ", "DROP ", "INSERT ", "REPLACE ", "UPDATE ")
            ):
                post_race_mutations.append(sql)
            return self.connection.execute(sql, parameters)

    @contextmanager
    def racing_connect(target: Path, *, read_only: bool = False) -> Iterator[Any]:
        with real_connect(target, read_only=read_only) as connection:
            if target.resolve() == path.resolve() and not read_only:
                yield RacingConnection(connection)
            else:
                yield connection

    monkeypatch.setattr(core_module, "connect_database", racing_connect)

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW, settings=settings)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert race_completed
    assert post_race_mutations == []
    assert path.read_bytes() == raced_bytes[0]
    with real_connect(path, read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT applied_at FROM schema_migration_history WHERE version = 2"
            ).fetchone()[0]
            == "2030-01-01T00:00:00+00:00"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'table' AND name IN ("
                "'legacy_authorization_quarantine', "
                "'legacy_authorization_reconciliations')"
            ).fetchone()[0]
            == 0
        )


def test_durable_history_verifier_rejects_noncanonical_applied_at(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable-history-noncanonical-applied-at.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    with connect_database(path) as connection:
        trigger_name = "trg_schema_migration_history_immutable_update"
        trigger_sql = str(
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                (trigger_name,),
            ).fetchone()[0]
        )
        connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE schema_migration_history SET applied_at = ? WHERE version = 2",
            ("not-a-canonical-applied-at",),
        )
        connection.execute("PRAGMA ignore_check_constraints = OFF")
        connection.execute(trigger_sql)
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "corruption",
    ["ordinal-gap", "version-order", "duplicate-version", "digest", "timestamp", "missing"],
)
def test_malformed_durable_history_fails_before_any_write(
    tmp_path: Path,
    corruption: str,
) -> None:
    path = tmp_path / f"malformed-durable-{corruption}.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    hashes = _packaged_workflow_hashes()
    valid_at = "2026-08-09T12:00:00+00:00"
    rows: list[tuple[int, int, str, str]] = [
        (1, 1, hashes[1], valid_at),
        (2, 2, hashes[2], valid_at),
        (3, 3, hashes[3], valid_at),
    ]
    if corruption == "ordinal-gap":
        rows[1] = (4, 2, hashes[2], valid_at)
    elif corruption == "version-order":
        rows[1], rows[2] = (2, 3, hashes[3], valid_at), (3, 2, hashes[2], valid_at)
    elif corruption == "duplicate-version":
        rows[2] = (3, 2, hashes[2], valid_at)
    elif corruption == "digest":
        rows[1] = (2, 2, "f" * 64, valid_at)
    elif corruption == "timestamp":
        rows[1] = (2, 2, hashes[2], "2026-08-09 12:00:00Z")
    else:
        rows.pop()
    with connect_database(path) as connection:
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'trg_schema_migration_history_%'"
        ).fetchall():
            connection.execute(f'DROP TRIGGER "{row["name"]}"')
        connection.execute("DROP TABLE schema_migration_history")
        connection.execute(
            "CREATE TABLE schema_migration_history("
            "ordinal INTEGER, version INTEGER, migration_sha256 TEXT, applied_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO schema_migration_history VALUES (?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert path.read_bytes() == before


def test_future_synthetic_migration_appends_one_digest_bound_history_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "synthetic-future-history.db"
    migrate_database(path, DatabaseKind.WORKFLOW)
    packaged = _migration_resources(DatabaseKind.WORKFLOW)
    synthetic_sql = "CREATE TABLE synthetic_migration_probe(value INTEGER);\n"

    class SyntheticMigration:
        name = "004_synthetic_history_probe.sql"

        def read_text(self, encoding: str = "utf-8") -> str:
            assert encoding == "utf-8"
            return synthetic_sql

        def read_bytes(self) -> bytes:
            return synthetic_sql.encode("utf-8")

    original_resources = core_module._migration_resources

    def resources(kind: DatabaseKind) -> list[object]:
        if kind is DatabaseKind.WORKFLOW:
            return [*packaged, SyntheticMigration()]
        return list(original_resources(kind))

    monkeypatch.setattr(core_module, "_migration_resources", resources)

    assert migrate_database(path, DatabaseKind.WORKFLOW) == [4]
    rows = _durable_history_rows(path)
    assert rows[-1][:3] == (
        4,
        4,
        hashlib.sha256(synthetic_sql.encode("utf-8")).hexdigest(),
    )
    assert len(rows) == 4
    with connect_database(path) as connection:
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'trg_schema_migration_history_%'"
        ).fetchall():
            connection.execute(f'DROP TRIGGER "{row["name"]}"')
        connection.execute("DROP TABLE schema_migration_history")
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert path.read_bytes() == before


def test_migration_rejects_schema_objects_without_history_using_stable_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-history.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE untouched (value BLOB)")
        connection.execute("INSERT INTO untouched(value) VALUES (?)", (b"opaque",))
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW)

    assert excinfo.value.stop_reason == "MIGRATION_HISTORY_INVALID"
    assert path.read_bytes() == before


@pytest.mark.parametrize("wal_database", ["workflow", "inventory"])
def test_atomic_human_decision_preflight_rejects_wal_without_mutation(
    wal_database: str, workflow_db: Path, inventory_db: Path
) -> None:
    settings = Settings(
        workflow_db=workflow_db,
        inventory_db=inventory_db,
        source_archive_dir=workflow_db.parent / "sources",
    )
    store = WorkflowStore(settings)
    source = make_source(settings.source_archive_dir)
    store.register_source(source)
    store.create_case(CASE_ID, source, LEGACY_AT)
    claim = store.claim_case_execution(
        CASE_ID, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    invoice, risk, case_critique = persist_bound_review_evidence(store, claim, settings, source)
    review = store.save_review(
        make_bound_review("rev_journal_mode", LEGACY_AT, invoice, risk, case_critique), claim
    )
    store.release_case_execution(claim)
    target = workflow_db if wal_database == "workflow" else inventory_db
    with connect_database(target) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    with connect_database(workflow_db, read_only=True) as connection:
        review_before = tuple(
            connection.execute(
                "SELECT status, payload_json, resolved_at FROM review_requests WHERE review_id = ?",
                (review.review_id,),
            ).fetchone()
        )
        decisions_before = connection.execute(
            "SELECT * FROM human_decisions ORDER BY decision_id"
        ).fetchall()
    with connect_database(inventory_db, read_only=True) as connection:
        aliases_before = connection.execute(
            "SELECT * FROM item_aliases ORDER BY alias_normalized"
        ).fetchall()

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.REJECT,
            "rollback journal mode is required for two-file atomicity",
            store,
            inventory_db,
        )

    assert excinfo.value.stop_reason == "ATOMIC_JOURNAL_MODE_REQUIRED"
    with connect_database(workflow_db, read_only=True) as connection:
        assert (
            tuple(
                connection.execute(
                    "SELECT status, payload_json, resolved_at FROM review_requests WHERE review_id = ?",
                    (review.review_id,),
                ).fetchone()
            )
            == review_before
        )
        assert (
            connection.execute("SELECT * FROM human_decisions ORDER BY decision_id").fetchall()
            == decisions_before
        )
    with connect_database(inventory_db, read_only=True) as connection:
        assert (
            connection.execute("SELECT * FROM item_aliases ORDER BY alias_normalized").fetchall()
            == aliases_before
        )
