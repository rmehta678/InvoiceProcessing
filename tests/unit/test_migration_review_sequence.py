"""Workflow migration 002: review sequencing, v2 verification, and required indexes."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from invoice_agents.db.core import (
    DatabaseKind,
    _migration_resources,
    connect_database,
    migrate_database,
    verify_database,
)
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import DatabaseVerificationError, InvoiceAgentsError
from invoice_agents.hitl.service import record_human_decision
from invoice_agents.models import (
    CaseStatus,
    Critique,
    DecisionKind,
    HumanDecisionKind,
    ReviewRequest,
    SourceArtifact,
)

CASE_ID = "case_v1_legacy"
REVIEW_ID = "rev_v1_legacy"
LEGACY_AT = datetime(2026, 1, 1, tzinfo=UTC)


def make_critique() -> Critique:
    return Critique(
        supported_findings=["legacy evidence"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=DecisionKind.HOLD,
        rationale=["legacy rationale"],
    )


def make_source() -> SourceArtifact:
    return SourceArtifact(
        source_id="src_v1_legacy",
        canonical_path=Path("invoice_legacy.txt"),
        sha256="f" * 64,
        source_format="txt",
        size_bytes=1,
        modified_at=LEGACY_AT,
    )


def make_review(review_id: str, created_at: datetime) -> ReviewRequest:
    return ReviewRequest(
        review_id=review_id,
        case_id=CASE_ID,
        status="PENDING",
        reasons=["legacy policy trigger"],
        amount=None,
        source=make_source(),
        evidence_bundle={},
        agent_recommendation=DecisionKind.HOLD,
        agent_rationale=["legacy review"],
        critic=make_critique(),
        questions=[],
        created_at=created_at,
    )


def v1_review_payload() -> str:
    """A valid pre-schema-v2 review payload: no 'sequence' key at all."""

    payload = make_review(REVIEW_ID, LEGACY_AT).model_dump(mode="json")
    del payload["sequence"]
    return json.dumps(payload)


def build_v1_workflow_db(tmp_path: Path) -> Path:
    """Apply only 001_initial.sql and populate case, review, and human-decision rows."""

    path = tmp_path / "workflow_v1.db"
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
            ("src_v1_legacy", "invoice_legacy.txt", "f" * 64, "txt", 1, at, "{}", at),
        )
        connection.execute(
            "INSERT INTO cases(case_id, source_id, status, stop_reason, started_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (CASE_ID, "src_v1_legacy", "NEEDS_HUMAN", "HUMAN_REVIEW_REQUESTED", at, at),
        )
        connection.execute(
            "INSERT INTO review_requests(review_id, case_id, status, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (REVIEW_ID, CASE_ID, "PENDING", v1_review_payload(), at),
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
    return path


def test_v1_database_is_rejected_until_migrated_and_rows_gain_sequence(tmp_path: Path) -> None:
    path = build_v1_workflow_db(tmp_path)
    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(path, DatabaseKind.WORKFLOW)
    assert excinfo.value.stop_reason == "DATABASE_VERSION_MISMATCH"
    assert migrate_database(path, DatabaseKind.WORKFLOW) == [2, 3]
    report = verify_database(path, DatabaseKind.WORKFLOW)
    assert report["schema_version"] == 3
    with connect_database(path, read_only=True) as connection:
        row = connection.execute(
            "SELECT sequence FROM review_requests WHERE review_id = ?", (REVIEW_ID,)
        ).fetchone()
        assert int(row["sequence"]) == 1
        # The human_decisions -> review_requests foreign key survives the table rebuild.
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    review = WorkflowStore(path).load_case_review(CASE_ID)
    assert review is not None
    assert review.review_id == REVIEW_ID
    assert review.sequence == 1


def test_second_review_cycle_is_sequenced_and_duplicates_are_rejected(tmp_path: Path) -> None:
    path = build_v1_workflow_db(tmp_path)
    migrate_database(path, DatabaseKind.WORKFLOW)
    store = WorkflowStore(path)
    claim = store.claim_case_execution(
        CASE_ID, frozenset({CaseStatus.NEEDS_HUMAN}), lease_seconds=60
    )
    saved = store.save_review(make_review("rev_v2_cycle", datetime(2026, 2, 1, tzinfo=UTC)), claim)
    store.release_case_execution(claim)
    assert saved.sequence == 2
    latest = store.load_case_review(CASE_ID)
    assert latest is not None
    assert latest.review_id == "rev_v2_cycle"
    assert latest.sequence == 2
    ordered = store.list_reviews(pending_only=False)
    assert [review.review_id for review in ordered] == [REVIEW_ID, "rev_v2_cycle"]
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
    with connect_database(workflow_db) as connection:
        connection.execute("DROP INDEX idx_events_case_created")
        connection.commit()
    with pytest.raises(DatabaseVerificationError, match="idx_events_case_created") as workflow_exc:
        verify_database(workflow_db, DatabaseKind.WORKFLOW)
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
    store = WorkflowStore(workflow_db)
    source = make_source()
    store.register_source(source)
    store.create_case(CASE_ID, source, LEGACY_AT)
    claim = store.claim_case_execution(
        CASE_ID, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    review = store.save_review(make_review("rev_journal_mode", LEGACY_AT), claim)
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
