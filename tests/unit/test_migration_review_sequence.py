"""Workflow migration 002: review sequencing, v2 verification, and required indexes."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
        questions=[],
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
        questions=[],
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
