"""Persisted review decisions, cross-database atomicity, and payment idempotency."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from invoice_agents import cli
from invoice_agents.config import Settings
from invoice_agents.db.core import (
    REQUIRED_WORKFLOW_TRIGGERS,
    DatabaseKind,
    connect_database,
    verify_database,
)
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import DatabaseVerificationError, InvoiceAgentsError
from invoice_agents.hitl.service import create_review_request, record_human_decision
from invoice_agents.models import (
    CanonicalMapping,
    CaseResult,
    CaseStatus,
    Critique,
    DecisionKind,
    FinalDecision,
    HumanDecisionKind,
    IdentityCandidate,
    Money,
    PaymentResult,
    PaymentStatus,
    ReviewRequest,
)
from invoice_agents.payment.service import mock_payment
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
from invoice_agents.ui.runs import RunRegistry
from invoice_agents.ui.sse import terminal_payload

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load(invoice_dir: Path, name: str, archive: Path):  # type: ignore[no-untyped-def]
    source = snapshot_source(invoice_dir / name, archive, max_bytes=10_485_760)
    return extract_invoice_evidence(source)


def persist_case(
    store: WorkflowStore,
    case_id: str,
    invoice: Any,
    *,
    started_at: datetime | None = None,
) -> None:
    store.register_source(invoice.source)
    store.create_case(case_id, invoice.source, started_at or datetime.now(UTC))
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.save_extraction(case_id, invoice, claim)
    store.release_case_execution(claim)


def critique(disposition: DecisionKind = DecisionKind.HOLD) -> Critique:
    return Critique(
        supported_findings=["deterministic evidence reviewed"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=disposition,
        rationale=["review policy applies"],
    )


def record_payment_evidence(
    store: WorkflowStore,
    case_id: str,
    invoice: Any,
    settings: Settings,
    disposition: DecisionKind = DecisionKind.APPROVE,
    *,
    authorize_review: bool = False,
) -> ExecutionClaim:
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    promoted = store.promote_predecessor_extraction(claim)
    mappings, comparisons, unresolved = compare_inventory_evidence(
        promoted, InventoryReader(settings.inventory_db)
    )
    enriched = apply_mapping_evidence(promoted, mappings, unresolved)
    store.save_extraction(case_id, enriched, claim)
    identity = find_prior_invoice_candidates(case_id, enriched, store)
    risk = build_risk_assessment(
        enriched, comparisons, identity, compute_invoice_totals(enriched), settings
    )
    store.save_identity(
        case_id,
        [item.model_dump(mode="json") for item in identity],
        claim,
    )
    store.save_comparison(
        case_id,
        "inventory",
        {
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        },
        claim,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    case_critique = critique(disposition)
    store.save_critique(case_id, case_critique, claim)
    if authorize_review:
        review = create_review_request(
            case_id,
            enriched,
            risk,
            case_critique,
            DecisionKind.HOLD,
            ["explicit fixture authorization for representation-specific evidence"],
            store,
            claim,
        )
        record_human_decision(
            review.review_id,
            "payment-fixture-reviewer@example.com",
            HumanDecisionKind.APPROVE,
            "the representation-specific evidence is explicitly authorized",
            store,
            settings.inventory_db,
            addressed_blocker_ids=[
                str(item["blocker_id"]) for item in review.evidence_bundle["blocking_evidence"]
            ],
        )
    return claim


def approve_final(store: WorkflowStore, case_id: str, claim: ExecutionClaim) -> ExecutionClaim:
    review = store.load_case_review(case_id)
    invoice = store.load_current_extraction(claim)
    store.save_final_decision(
        case_id,
        FinalDecision(
            decision=DecisionKind.APPROVE,
            reasons=["approved evidence"],
            evidence=[reference for line in invoice.lines for reference in line.evidence[:1]],
            critic_disposition=DecisionKind.APPROVE,
            human_outcome=review.human_decision if review is not None else None,
            payment_eligible=True,
        ),
        claim,
    )
    return claim


def pending_review(
    source: Path,
    case_id: str,
    settings: Settings,
    *,
    identity_candidates: list[IdentityCandidate] | None = None,
    started_at: datetime | None = None,
) -> ReviewRequest:
    """Persist a review through the real extraction/comparison path."""

    store = WorkflowStore(settings)
    invoice = load(source.parent, source.name, settings.source_archive_dir)
    persist_case(store, case_id, invoice, started_at=started_at)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    promoted = store.promote_predecessor_extraction(claim)
    mappings, comparisons, unresolved = compare_inventory_evidence(
        promoted, InventoryReader(settings.inventory_db)
    )
    enriched = apply_mapping_evidence(promoted, mappings, unresolved)
    store.save_extraction(case_id, enriched, claim)
    identity = identity_candidates or find_prior_invoice_candidates(case_id, enriched, store)
    risk = build_risk_assessment(
        enriched,
        comparisons,
        identity,
        compute_invoice_totals(enriched),
        settings,
    )
    store.save_identity(
        case_id,
        [item.model_dump(mode="json") for item in identity],
        claim,
    )
    store.save_comparison(
        case_id,
        "inventory",
        {
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        },
        claim,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    case_critique = critique()
    store.save_critique(case_id, case_critique, claim)
    review = create_review_request(
        case_id,
        enriched,
        risk,
        case_critique,
        DecisionKind.HOLD,
        ["evidence requires a human decision"],
        store,
        claim,
        extra_reasons=["atomic-decision regression"],
    )
    store.release_case_execution(claim)
    return review


def persisted_decision_state(
    workflow_db: Path, inventory_db: Path, review_id: str
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]], tuple[Any, ...] | None]:
    """Return byte-exact persisted decision inputs/outputs for mutation assertions."""

    with connect_database(inventory_db, read_only=True) as connection:
        aliases = [
            tuple(row)
            for row in connection.execute(
                "SELECT alias_normalized, sku, source, approved_by, approved_at "
                "FROM item_aliases ORDER BY alias_normalized"
            ).fetchall()
        ]
    with connect_database(workflow_db, read_only=True) as connection:
        decisions = [
            tuple(row)
            for row in connection.execute(
                "SELECT decision_id, review_id, reviewer, decision, reason, payload_json, "
                "decided_at FROM human_decisions ORDER BY decision_id"
            ).fetchall()
        ]
        row = connection.execute(
            "SELECT status, payload_json, resolved_at FROM review_requests WHERE review_id = ?",
            (review_id,),
        ).fetchone()
    return aliases, decisions, tuple(row) if row is not None else None


def mapping(raw_item: str, sku: str = "SKU-WIDGET-A") -> CanonicalMapping:
    return CanonicalMapping(raw_item=raw_item, sku=sku, basis="human_decision")


def test_review_request_and_human_decision_are_persisted(
    invoice_dir: Path,
    inventory_db: Path,
    workflow_db: Path,
    settings: Settings,
) -> None:
    store = WorkflowStore(settings)
    review = pending_review(
        invoice_dir / "invoice_1002.txt",
        "case_review",
        settings,
    )
    assert review.status == "PENDING"
    resolved = record_human_decision(
        review.review_id,
        "vp@example.com",
        HumanDecisionKind.REJECT,
        "quantity is not authorized",
        store,
        inventory_db,
    )
    assert resolved.status == "RESOLVED"
    assert resolved.human_decision is not None
    assert resolved.human_decision.reviewer == "vp@example.com"
    assert resolved.agent_recommendation is DecisionKind.HOLD


def test_nonexistent_review_decision_leaves_both_databases_unchanged(
    settings: Settings,
) -> None:
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, "rev_does_not_exist"
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            "rev_does_not_exist",
            "reviewer@example.com",
            HumanDecisionKind.REJECT,
            "the evidence does not support payment",
            WorkflowStore(settings),
            settings.inventory_db,
        )

    assert excinfo.value.stop_reason == "REVIEW_NOT_FOUND"
    assert (
        persisted_decision_state(settings.workflow_db, settings.inventory_db, "rev_does_not_exist")
        == before
    )


@pytest.fixture
def mapping_review(settings: Settings) -> ReviewRequest:
    return pending_review(
        FIXTURE_DIR / "invoice_2001_bulk_alias.txt", "case_mapping_review", settings
    )


def test_mapping_payload_on_non_mapping_decision_is_rejected_without_mutation(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.REJECT,
            "rejecting while a stale client submits mapping fields",
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=[mapping("WidgetA (bulk)")],
        )

    assert excinfo.value.stop_reason == "HUMAN_MAPPING_INVALID"
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_mapping_alias_absent_from_unresolved_review_evidence_is_rejected_without_mutation(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.ESTABLISH_MAPPING,
            "this alias was not in the review package",
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=[mapping("invented alias")],
        )

    assert excinfo.value.stop_reason == "MAPPING_ALIAS_NOT_IN_REVIEW"
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_unknown_mapping_sku_is_rejected_without_mutation(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.ESTABLISH_MAPPING,
            "the target must exist in inventory",
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=[mapping("WidgetA (bulk)", "SKU-DOES-NOT-EXIST")],
        )

    assert excinfo.value.stop_reason == "MAPPING_SKU_NOT_FOUND"
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_unknown_blocker_id_is_rejected_without_mutation(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.APPROVE,
            "the submitted blocker did not come from this review",
            WorkflowStore(settings),
            settings.inventory_db,
            addressed_blocker_ids=["inventory:invented:UNKNOWN"],
        )

    assert excinfo.value.stop_reason == "BLOCKER_AUTHORIZATION_INVALID"
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_superseded_case_on_non_supersession_decision_is_rejected_without_mutation(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.REJECT,
            "a stale form included an unrelated supersession field",
            WorkflowStore(settings),
            settings.inventory_db,
            superseded_case_id="case_unrelated",
        )

    assert excinfo.value.stop_reason == "SUPERSEDED_CASE_INVALID"
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_workflow_write_failure_rolls_back_alias_and_decision_across_both_files(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    # A real SQLite trigger fails the final workflow write after alias/human-decision inserts.
    with connect_database(settings.workflow_db) as connection:
        connection.executescript(
            "CREATE TRIGGER fail_review_resolution BEFORE UPDATE OF status ON review_requests "
            "WHEN NEW.status = 'RESOLVED' BEGIN "
            "SELECT RAISE(ABORT, 'injected workflow review failure'); END;"
        )
        connection.commit()
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected workflow review failure"):
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.ESTABLISH_MAPPING,
            "the bulk name is an approved alias",
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=[mapping("WidgetA (bulk)")],
        )

    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_semantic_replay_ignores_timestamp_and_normalized_whitespace(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    blockers = list(mapping_review.evidence_bundle["blocking_evidence"])
    first = record_human_decision(
        mapping_review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.ESTABLISH_MAPPING,
        "bulk aliases are authorized",
        WorkflowStore(settings),
        settings.inventory_db,
        mappings=[mapping("WidgetA (bulk)")],
        addressed_blocker_ids=[str(item["blocker_id"]) for item in blockers],
    )
    before_retry = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    replayed = record_human_decision(
        mapping_review.review_id,
        " reviewer@example.com ",
        HumanDecisionKind.ESTABLISH_MAPPING,
        "  bulk   aliases\nare authorized  ",
        WorkflowStore(settings),
        settings.inventory_db,
        mappings=[mapping("WidgetA (bulk)")],
        addressed_blocker_ids=[str(item["blocker_id"]) for item in reversed(blockers)],
    )

    assert replayed == first
    assert replayed.human_decision is not None
    assert first.human_decision is not None
    assert replayed.human_decision.decided_at == first.human_decision.decided_at
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before_retry
    )


def test_every_semantically_different_retry_fails_without_mutation(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    resolved = record_human_decision(
        mapping_review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.ESTABLISH_MAPPING,
        "approved mapping evidence",
        WorkflowStore(settings),
        settings.inventory_db,
        mappings=[mapping("WidgetA (bulk)")],
    )
    assert resolved.status == "RESOLVED"
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )
    different_inputs = [
        {"reviewer": "other@example.com"},
        {"decision": HumanDecisionKind.REJECT},
        {"reason": "different evidence"},
        {"mappings": [mapping("WidgetA (bulk)", "SKU-WIDGET-B")]},
        {"mappings": [mapping("Widget A bulk")]},
        {"superseded_case_id": "case_other"},
        {"addressed_blocker_ids": ["inventory:stale:UNKNOWN"]},
    ]
    for changes in different_inputs:
        inputs: dict[str, Any] = {
            "reviewer": "reviewer@example.com",
            "decision": HumanDecisionKind.ESTABLISH_MAPPING,
            "reason": "approved mapping evidence",
            "mappings": [mapping("WidgetA (bulk)")],
            "superseded_case_id": None,
            "addressed_blocker_ids": [],
        }
        inputs.update(changes)
        with pytest.raises(InvoiceAgentsError) as excinfo:
            record_human_decision(
                mapping_review.review_id,
                inputs["reviewer"],
                inputs["decision"],
                inputs["reason"],
                WorkflowStore(settings),
                settings.inventory_db,
                mappings=inputs["mappings"],
                superseded_case_id=inputs["superseded_case_id"],
                addressed_blocker_ids=inputs["addressed_blocker_ids"],
            )
        assert excinfo.value.stop_reason == "REVIEW_ALREADY_RESOLVED"
        assert (
            persisted_decision_state(
                settings.workflow_db, settings.inventory_db, mapping_review.review_id
            )
            == before
        )


@pytest.mark.parametrize(
    ("reviewer", "reason"),
    [
        ("   ", "approved mapping evidence"),
        ("reviewer@example.com", " \n\t "),
    ],
    ids=["blank-reviewer", "blank-reason"],
)
def test_resolved_blank_field_retry_is_classified_as_already_resolved_before_validation(
    reviewer: str,
    reason: str,
    settings: Settings,
    mapping_review: ReviewRequest,
) -> None:
    record_human_decision(
        mapping_review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.ESTABLISH_MAPPING,
        "approved mapping evidence",
        WorkflowStore(settings),
        settings.inventory_db,
        mappings=[mapping("WidgetA (bulk)")],
    )
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            mapping_review.review_id,
            reviewer,
            HumanDecisionKind.ESTABLISH_MAPPING,
            reason,
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=[mapping("WidgetA (bulk)")],
        )

    assert excinfo.value.stop_reason == "REVIEW_ALREADY_RESOLVED"
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


@pytest.mark.parametrize("exact_replay", [True, False], ids=["exact", "different"])
def test_resolved_retry_uses_only_workflow_state_when_inventory_is_wal(
    exact_replay: bool, settings: Settings
) -> None:
    review = pending_review(Path("data/invoices/invoice_1002.txt"), "case_wal_replay", settings)
    resolved = record_human_decision(
        review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.REJECT,
        "the evidence does not support payment",
        WorkflowStore(settings),
        settings.inventory_db,
    )
    with connect_database(settings.inventory_db) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    before = persisted_decision_state(settings.workflow_db, settings.inventory_db, review.review_id)

    if exact_replay:
        replayed = record_human_decision(
            review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.REJECT,
            "the evidence does not support payment",
            WorkflowStore(settings),
            settings.inventory_db,
        )
        assert replayed == resolved
    else:
        with pytest.raises(InvoiceAgentsError) as excinfo:
            record_human_decision(
                review.review_id,
                "other@example.com",
                HumanDecisionKind.REJECT,
                "the evidence does not support payment",
                WorkflowStore(settings),
                settings.inventory_db,
            )
        assert excinfo.value.stop_reason == "REVIEW_ALREADY_RESOLVED"
    assert (
        persisted_decision_state(settings.workflow_db, settings.inventory_db, review.review_id)
        == before
    )


@pytest.mark.parametrize("exact_replay", [True, False], ids=["exact", "different"])
def test_resolved_retry_never_opens_an_unavailable_inventory_path(
    exact_replay: bool, settings: Settings, tmp_path: Path
) -> None:
    review = pending_review(
        Path("data/invoices/invoice_1002.txt"), "case_missing_inventory_replay", settings
    )
    resolved = record_human_decision(
        review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.REJECT,
        "the evidence does not support payment",
        WorkflowStore(settings),
        settings.inventory_db,
    )
    unavailable_inventory = tmp_path / "missing-parent" / "inventory.db"
    assert not unavailable_inventory.parent.exists()
    before = persisted_decision_state(settings.workflow_db, settings.inventory_db, review.review_id)

    if exact_replay:
        replayed = record_human_decision(
            review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.REJECT,
            "the evidence does not support payment",
            WorkflowStore(settings),
            unavailable_inventory,
        )
        assert replayed == resolved
    else:
        with pytest.raises(InvoiceAgentsError) as excinfo:
            record_human_decision(
                review.review_id,
                "other@example.com",
                HumanDecisionKind.REJECT,
                "the evidence does not support payment",
                WorkflowStore(settings),
                unavailable_inventory,
            )
        assert excinfo.value.stop_reason == "REVIEW_ALREADY_RESOLVED"
    assert not unavailable_inventory.parent.exists()
    assert (
        persisted_decision_state(settings.workflow_db, settings.inventory_db, review.review_id)
        == before
    )


def test_replay_and_preflight_share_incompatible_human_field_validation(
    settings: Settings,
) -> None:
    review = pending_review(
        Path("data/invoices/invoice_1002.txt"),
        "case_invalid_resolved_replay",
        settings,
    )
    resolved = record_human_decision(
        review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.REJECT,
        "the evidence does not support payment",
        WorkflowStore(settings),
        settings.inventory_db,
    )
    assert resolved.human_decision is not None
    review_payload = resolved.model_dump(mode="json")
    human_payload = resolved.human_decision.model_dump(mode="json")
    review_payload["human_decision"]["superseded_case_id"] = "case_unrelated"
    human_payload["superseded_case_id"] = "case_unrelated"
    with connect_database(settings.workflow_db) as connection:
        for trigger in (
            "trg_resolved_review_immutable_update",
            "trg_human_decisions_immutable_update",
        ):
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(
            "UPDATE review_requests SET payload_json = ? WHERE review_id = ?",
            (json.dumps(review_payload), review.review_id),
        )
        connection.execute(
            "UPDATE human_decisions SET payload_json = ? WHERE review_id = ?",
            (json.dumps(human_payload), review.review_id),
        )
        for trigger in (
            "trg_resolved_review_immutable_update",
            "trg_human_decisions_immutable_update",
        ):
            connection.execute(REQUIRED_WORKFLOW_TRIGGERS[trigger])
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as replay_error:
        record_human_decision(
            review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.REJECT,
            "the evidence does not support payment",
            WorkflowStore(settings),
            settings.inventory_db,
            superseded_case_id="case_unrelated",
        )
    assert replay_error.value.stop_reason == "SUPERSEDED_CASE_INVALID"

    with pytest.raises(DatabaseVerificationError) as preflight_error:
        verify_database(settings.workflow_db, DatabaseKind.WORKFLOW, settings=settings)
    assert preflight_error.value.stop_reason == "DATABASE_AUTHORIZATION_PROVENANCE_INVALID"
    assert preflight_error.value.details["invalid_review_count"] == 1
    assert preflight_error.value.details["invalid_human_decision_count"] == 1


def test_mapping_replay_and_preflight_require_exact_persisted_alias_provenance(
    settings: Settings,
    mapping_review: ReviewRequest,
) -> None:
    record_human_decision(
        mapping_review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.ESTABLISH_MAPPING,
        "the bulk alias is authorized",
        WorkflowStore(settings),
        settings.inventory_db,
        mappings=[mapping("WidgetA (bulk)")],
    )
    with connect_database(settings.inventory_db) as connection:
        connection.execute(
            "UPDATE item_aliases SET approved_by = 'forged@example.com' WHERE source = ?",
            (f"human_review:{mapping_review.review_id}",),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as replay_error:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.ESTABLISH_MAPPING,
            "the bulk alias is authorized",
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=[mapping("WidgetA (bulk)")],
        )
    assert replay_error.value.stop_reason == "HUMAN_MAPPING_PROVENANCE_INVALID"

    with pytest.raises(DatabaseVerificationError) as preflight_error:
        verify_database(settings.workflow_db, DatabaseKind.WORKFLOW, settings=settings)
    assert preflight_error.value.stop_reason == "DATABASE_AUTHORIZATION_PROVENANCE_INVALID"
    assert preflight_error.value.details["invalid_review_count"] == 1
    assert preflight_error.value.details["invalid_human_decision_count"] == 1


def test_mapping_replay_and_preflight_reject_extra_aliases_claiming_review_provenance(
    settings: Settings,
    mapping_review: ReviewRequest,
) -> None:
    resolved = record_human_decision(
        mapping_review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.ESTABLISH_MAPPING,
        "the bulk alias is authorized",
        WorkflowStore(settings),
        settings.inventory_db,
        mappings=[mapping("WidgetA (bulk)")],
    )
    assert resolved.human_decision is not None
    with connect_database(settings.inventory_db) as connection:
        connection.execute(
            "INSERT INTO item_aliases("
            "alias_normalized, sku, source, approved_by, approved_at) VALUES (?, ?, ?, ?, ?)",
            (
                "forged extra alias",
                "SKU-WIDGET-A",
                f"human_review:{mapping_review.review_id}",
                resolved.human_decision.reviewer,
                resolved.human_decision.decided_at.isoformat(),
            ),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as replay_error:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.ESTABLISH_MAPPING,
            "the bulk alias is authorized",
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=[mapping("WidgetA (bulk)")],
        )
    assert replay_error.value.stop_reason == "HUMAN_MAPPING_PROVENANCE_INVALID"

    with pytest.raises(DatabaseVerificationError) as preflight_error:
        verify_database(settings.workflow_db, DatabaseKind.WORKFLOW, settings=settings)
    assert preflight_error.value.stop_reason == "DATABASE_AUTHORIZATION_PROVENANCE_INVALID"
    assert preflight_error.value.details["invalid_review_count"] == 1
    assert preflight_error.value.details["invalid_human_decision_count"] == 1


def test_pending_blank_field_still_fails_validation_before_inventory_access(
    settings: Settings, mapping_review: ReviewRequest, tmp_path: Path
) -> None:
    unavailable_inventory = tmp_path / "missing-parent" / "inventory.db"
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.ESTABLISH_MAPPING,
            "   ",
            WorkflowStore(settings),
            unavailable_inventory,
            mappings=[mapping("WidgetA (bulk)")],
        )

    assert excinfo.value.stop_reason == "HUMAN_DECISION_INVALID"
    assert not unavailable_inventory.parent.exists()
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_conflicting_targets_for_one_normalized_alias_are_rejected_without_mutation(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    before = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        record_human_decision(
            mapping_review.review_id,
            "reviewer@example.com",
            HumanDecisionKind.ESTABLISH_MAPPING,
            "one alias cannot authorize two inventory identities",
            WorkflowStore(settings),
            settings.inventory_db,
            mappings=[
                mapping("WidgetA (bulk)", "SKU-WIDGET-A"),
                mapping("Widget A bulk", "SKU-WIDGET-B"),
            ],
        )

    assert excinfo.value.stop_reason == "HUMAN_MAPPING_INVALID"
    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_only_one_of_two_competing_human_decisions_commits(settings: Settings) -> None:
    review = pending_review(
        Path("data/invoices/invoice_1002.txt"), "case_competing_review", settings
    )
    barrier = threading.Barrier(2)

    def decide(kind: HumanDecisionKind) -> ReviewRequest | InvoiceAgentsError:
        barrier.wait(timeout=5)
        try:
            return record_human_decision(
                review.review_id,
                f"{kind.value.lower()}@example.com",
                kind,
                f"competing {kind.value.lower()} decision",
                WorkflowStore(settings),
                settings.inventory_db,
            )
        except InvoiceAgentsError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(decide, [HumanDecisionKind.REJECT, HumanDecisionKind.REQUEST_CORRECTION])
        )

    committed = [outcome for outcome in outcomes if isinstance(outcome, ReviewRequest)]
    rejected = [outcome for outcome in outcomes if isinstance(outcome, InvoiceAgentsError)]
    assert len(committed) == 1
    assert len(rejected) == 1
    assert rejected[0].stop_reason == "REVIEW_ALREADY_RESOLVED"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM human_decisions WHERE review_id = ?", (review.review_id,)
        ).fetchone()[0]
    assert count == 1
    stored = WorkflowStore(settings).load_review(review.review_id)
    assert stored.human_decision == committed[0].human_decision


def test_supersession_requires_distinct_earlier_review_candidate_for_same_invoice_vendor(
    settings: Settings,
) -> None:
    store = WorkflowStore(settings)
    prior_at = datetime(2026, 1, 1, tzinfo=UTC)
    current_at = datetime(2026, 2, 1, tzinfo=UTC)
    later_at = datetime(2026, 3, 1, tzinfo=UTC)
    original = load(Path("data/invoices"), "invoice_1004.json", settings.source_archive_dir)
    revised = load(Path("data/invoices"), "invoice_1004_revised.json", settings.source_archive_dir)
    unrelated = load(Path("data/invoices"), "invoice_1001.txt", settings.source_archive_dir)
    persist_case(store, "case_prior_revision", original, started_at=prior_at)
    persist_case(store, "case_current_revision", revised, started_at=current_at)
    persist_case(store, "case_later_revision", original, started_at=later_at)
    persist_case(store, "case_unrelated", unrelated, started_at=prior_at)

    claim = store.claim_case_execution(
        "case_current_revision",
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    promoted = store.promote_predecessor_extraction(claim)
    mappings, comparisons, unresolved = compare_inventory_evidence(
        promoted, InventoryReader(settings.inventory_db)
    )
    enriched = apply_mapping_evidence(promoted, mappings, unresolved)
    store.save_extraction("case_current_revision", enriched, claim)
    candidates = find_prior_invoice_candidates("case_current_revision", enriched, store)
    assert {candidate.case_id for candidate in candidates} == {
        "case_prior_revision",
        "case_later_revision",
    }
    risk = build_risk_assessment(
        enriched,
        comparisons,
        candidates,
        compute_invoice_totals(enriched),
        settings,
    )
    store.save_identity(
        "case_current_revision",
        [item.model_dump(mode="json") for item in candidates],
        claim,
    )
    store.save_comparison(
        "case_current_revision",
        "inventory",
        {
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
            "unresolved_candidates": {
                item: result.model_dump(mode="json") for item, result in unresolved.items()
            },
        },
        claim,
    )
    store.save_comparison("case_current_revision", "risk", risk.model_dump(mode="json"), claim)
    case_critique = critique()
    store.save_critique("case_current_revision", case_critique, claim)
    review = create_review_request(
        "case_current_revision",
        enriched,
        risk,
        case_critique,
        DecisionKind.HOLD,
        ["possible revision requires a human ruling"],
        store,
        claim,
    )
    store.release_case_execution(claim)
    before = persisted_decision_state(settings.workflow_db, settings.inventory_db, review.review_id)
    for invalid_case_id in (
        "case_current_revision",
        "case_unrelated",
        "case_does_not_exist",
        "case_later_revision",
    ):
        with pytest.raises(InvoiceAgentsError) as excinfo:
            record_human_decision(
                review.review_id,
                "reviewer@example.com",
                HumanDecisionKind.SUPERSEDE_REVISION,
                "selecting the superseded revision",
                store,
                settings.inventory_db,
                superseded_case_id=invalid_case_id,
            )
        assert excinfo.value.stop_reason == "SUPERSEDED_CASE_INVALID"
        assert (
            persisted_decision_state(settings.workflow_db, settings.inventory_db, review.review_id)
            == before
        )

    resolved = record_human_decision(
        review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.SUPERSEDE_REVISION,
        "the earlier R0 case is superseded by R1",
        store,
        settings.inventory_db,
        superseded_case_id="case_prior_revision",
    )
    assert resolved.human_decision is not None
    assert resolved.human_decision.superseded_case_id == "case_prior_revision"


def test_cli_accepts_repeatable_address_blocker_options(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = pending_review(Path("data/invoices/invoice_1002.txt"), "case_cli_blockers", settings)
    blocker_ids = [
        str(entry["blocker_id"]) for entry in review.evidence_bundle["blocking_evidence"]
    ]
    assert blocker_ids
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    args = [
        "review",
        "decide",
        review.review_id,
        "--reviewer",
        "cli-reviewer@example.com",
        "--decision",
        "APPROVE",
        "--reason",
        "explicit blocker authorization",
    ]
    for blocker_id in blocker_ids:
        args.extend(["--address-blocker", blocker_id])

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    stored = WorkflowStore(settings).load_review(review.review_id)
    assert stored.human_decision is not None
    assert stored.human_decision.addressed_blocker_ids == blocker_ids


def test_mock_payment_pays_once_across_duplicate_representations(
    invoice_dir: Path, settings: Settings
) -> None:
    workflow_db = settings.workflow_db
    store = WorkflowStore(settings)
    text = load(invoice_dir, "invoice_1011.txt", workflow_db.parent / "sources")
    pdf = load(invoice_dir, "invoice_1011.pdf", workflow_db.parent / "sources")
    persist_case(store, "case_text", text)
    text_claim = record_payment_evidence(store, "case_text", text, settings)
    text = store.load_current_extraction(text_claim)
    approve_final(store, "case_text", text_claim)
    first = mock_payment("case_text", text, store, workflow_db, text_claim)
    assert first.status is PaymentStatus.PAID
    persist_case(store, "case_pdf", pdf)
    pdf_claim = record_payment_evidence(store, "case_pdf", pdf, settings, authorize_review=True)
    pdf = store.load_current_extraction(pdf_claim)
    approve_final(store, "case_pdf", pdf_claim)
    duplicate = mock_payment("case_pdf", pdf, store, workflow_db, pdf_claim)
    assert duplicate.status is PaymentStatus.DUPLICATE
    assert duplicate.case_id == "case_pdf"
    assert duplicate.duplicate_of == first.payment_id
    with connect_database(workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 1


def duplicate_terminal_fixture(
    invoice_dir: Path, settings: Settings
) -> tuple[WorkflowStore, CaseResult, PaymentResult, PaymentResult]:
    store = WorkflowStore(settings)
    text = load(invoice_dir, "invoice_1011.txt", settings.source_archive_dir)
    pdf = load(invoice_dir, "invoice_1011.pdf", settings.source_archive_dir)
    persist_case(store, "case_original_payment", text)
    original_claim = record_payment_evidence(store, "case_original_payment", text, settings)
    text = store.load_current_extraction(original_claim)
    approve_final(store, "case_original_payment", original_claim)
    paid = mock_payment("case_original_payment", text, store, settings.workflow_db, original_claim)
    persist_case(store, "case_duplicate_attempt", pdf)
    duplicate_claim = record_payment_evidence(
        store,
        "case_duplicate_attempt",
        pdf,
        settings,
        authorize_review=True,
    )
    pdf = store.load_current_extraction(duplicate_claim)
    approve_final(store, "case_duplicate_attempt", duplicate_claim)
    duplicate = mock_payment(
        "case_duplicate_attempt", pdf, store, settings.workflow_db, duplicate_claim
    )
    assert duplicate.status is PaymentStatus.DUPLICATE
    final = store.load_current_final_decision(duplicate_claim)
    assert final is not None
    started_at = store.load_authoritative_case_started_at(duplicate_claim)
    attempted = CaseResult(
        case_id="case_duplicate_attempt",
        source_id=pdf.source.source_id,
        status=CaseStatus.SUCCEEDED,
        stop_reason="APPROVED_PAYMENT_RECORDED",
        final_decision=final,
        payment=duplicate,
        started_at=started_at,
        finished_at=max(datetime.now(UTC), started_at),
    )
    return store, attempted, paid, duplicate


def persist_finished_terminal_envelope(settings: Settings, result: CaseResult) -> None:
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET status = ?, stop_reason = ?, result_json = ?, "
            "execution_state = 'FINISHED', lease_expires_at = NULL WHERE case_id = ?",
            (
                str(result.status),
                result.stop_reason,
                result.model_dump_json(),
                result.case_id,
            ),
        )
        connection.commit()


def test_terminal_merge_reconstructs_cross_case_duplicate_from_paid_ledger(
    invoice_dir: Path, settings: Settings
) -> None:
    """A valid duplicate result derives from the one immutable original PAID row."""

    store, attempted, paid, duplicate = duplicate_terminal_fixture(invoice_dir, settings)

    merged = store.merge_relational_case_evidence(attempted)

    assert merged.payment == duplicate
    assert merged.payment is not None
    assert merged.payment.case_id == "case_duplicate_attempt"
    assert merged.payment.status is PaymentStatus.DUPLICATE
    assert merged.payment.payment_id == paid.payment_id
    assert merged.payment.duplicate_of == paid.payment_id
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 1


def test_duplicate_reconstruction_propagates_missing_evidence_authority(
    invoice_dir: Path, settings: Settings
) -> None:
    """Missing Settings is not evidence corruption and must retain its exact stop reason."""

    _store, attempted, _paid, _duplicate = duplicate_terminal_fixture(invoice_dir, settings)
    path_only_store = WorkflowStore(settings.workflow_db)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        path_only_store.merge_relational_case_evidence(attempted)

    assert excinfo.value.stop_reason == "EVIDENCE_AUTHORITY_MISSING"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_finished_duplicate_sse_requires_settings_without_mutating_persisted_state(
    invoice_dir: Path, settings: Settings
) -> None:
    store, attempted, _paid, _duplicate = duplicate_terminal_fixture(invoice_dir, settings)
    persist_finished_terminal_envelope(settings, attempted)
    with connect_database(settings.workflow_db, read_only=True) as connection:
        before = tuple(
            connection.execute(
                "SELECT status, stop_reason, result_json, execution_state, lease_expires_at "
                "FROM cases WHERE case_id = ?",
                (attempted.case_id,),
            ).fetchone()
        )

    missing_authority = terminal_payload(
        settings.workflow_db,
        attempted.case_id,
        RunRegistry(),
    )
    trusted = terminal_payload(
        settings.workflow_db,
        attempted.case_id,
        RunRegistry(),
        settings,
    )

    assert missing_authority is not None
    assert missing_authority["stop_reason"] == "EVIDENCE_AUTHORITY_MISSING"
    assert trusted is not None
    assert trusted["status"] == "SUCCEEDED"
    assert trusted["stop_reason"] == "APPROVED_PAYMENT_RECORDED"
    assert store.load_result(attempted.case_id) == attempted
    with connect_database(settings.workflow_db, read_only=True) as connection:
        after = tuple(
            connection.execute(
                "SELECT status, stop_reason, result_json, execution_state, lease_expires_at "
                "FROM cases WHERE case_id = ?",
                (attempted.case_id,),
            ).fetchone()
        )
    assert after == before


def test_finished_duplicate_sse_keeps_persisted_contradiction_classification(
    invoice_dir: Path, settings: Settings
) -> None:
    _store, attempted, _paid, duplicate = duplicate_terminal_fixture(invoice_dir, settings)
    contradictory = attempted.model_copy(
        update={"payment": duplicate.model_copy(update={"idempotency_key": "0" * 64})},
        deep=True,
    )
    persist_finished_terminal_envelope(settings, contradictory)
    with connect_database(settings.workflow_db, read_only=True) as connection:
        before = connection.execute(
            "SELECT result_json FROM cases WHERE case_id = ?", (attempted.case_id,)
        ).fetchone()[0]

    payload = terminal_payload(
        settings.workflow_db,
        attempted.case_id,
        RunRegistry(),
        settings,
    )

    assert payload is not None
    assert payload["stop_reason"] == "PERSISTED_RESULT_INVALID"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        after = connection.execute(
            "SELECT result_json FROM cases WHERE case_id = ?", (attempted.case_id,)
        ).fetchone()[0]
    assert after == before


def test_terminal_sse_recovery_validates_cross_case_duplicate_with_request_settings(
    invoice_dir: Path, settings: Settings
) -> None:
    """Post-startup lease recovery retains a validated cross-case duplicate."""

    store, attempted, _paid, duplicate = duplicate_terminal_fixture(invoice_dir, settings)
    interrupted = attempted.model_copy(
        update={
            "status": CaseStatus.INCOMPLETE,
            "stop_reason": "INTERRUPTED_AFTER_DUPLICATE",
        },
        deep=True,
    )
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET status = ?, stop_reason = ?, result_json = ?, "
            "lease_expires_at = ? WHERE case_id = ?",
            (
                str(interrupted.status),
                interrupted.stop_reason,
                interrupted.model_dump_json(),
                "2000-01-01T00:00:00+00:00",
                attempted.case_id,
            ),
        )
        connection.commit()

    with connect_database(settings.workflow_db, read_only=True) as connection:
        before = tuple(
            connection.execute(
                "SELECT status, stop_reason, result_json, execution_state, lease_expires_at "
                "FROM cases WHERE case_id = ?",
                (attempted.case_id,),
            ).fetchone()
        )

    missing_authority = terminal_payload(
        settings.workflow_db,
        attempted.case_id,
        RunRegistry(),
    )

    assert missing_authority is not None
    assert missing_authority["stop_reason"] == "EVIDENCE_AUTHORITY_MISSING"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        after_missing_authority = tuple(
            connection.execute(
                "SELECT status, stop_reason, result_json, execution_state, lease_expires_at "
                "FROM cases WHERE case_id = ?",
                (attempted.case_id,),
            ).fetchone()
        )
    assert after_missing_authority == before

    payload = terminal_payload(
        settings.workflow_db,
        attempted.case_id,
        RunRegistry(),
        settings,
    )

    assert payload is not None
    assert payload["status"] == "INCOMPLETE"
    assert payload["stop_reason"] == "ORPHANED_EXECUTION"
    recovered = store.load_result(attempted.case_id)
    assert recovered is not None
    assert recovered.payment == duplicate


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_original",
        "wrong_payment_reference",
        "wrong_idempotency_key",
        "wrong_amount",
        "wrong_currency",
        "original_source",
        "original_generation",
        "original_status",
        "missing_ledger",
        "multiple_case_ledger_rows",
    ],
)
def test_terminal_merge_rejects_contradictory_duplicate_authority(
    corruption: str,
    invoice_dir: Path,
    settings: Settings,
) -> None:
    store, attempted, paid, duplicate = duplicate_terminal_fixture(invoice_dir, settings)
    assert paid.payment_id is not None
    assert duplicate.amount is not None
    if corruption == "missing_original":
        attempted.payment = duplicate.model_copy(
            update={"payment_id": "pay_missing", "duplicate_of": "pay_missing"}
        )
    elif corruption == "wrong_payment_reference":
        attempted.payment = duplicate.model_copy(update={"payment_id": "pay_wrong"})
    elif corruption == "wrong_idempotency_key":
        attempted.payment = duplicate.model_copy(update={"idempotency_key": "0" * 64})
    elif corruption == "wrong_amount":
        attempted.payment = duplicate.model_copy(
            update={
                "amount": Money(
                    amount=duplicate.amount.amount + Decimal("1"),
                    currency=duplicate.amount.currency,
                )
            }
        )
    elif corruption == "wrong_currency":
        attempted.payment = duplicate.model_copy(
            update={"amount": Money(amount=duplicate.amount.amount, currency="EUR")}
        )
    else:
        with connect_database(settings.workflow_db) as connection:
            for trigger in (
                "trg_payments_authorization_insert",
                "trg_payments_snapshot_digest_insert",
                "trg_payments_immutable_update",
                "trg_payments_snapshot_digest_update",
                "trg_paid_payments_immutable_update",
                "trg_payments_immutable_delete",
                "trg_paid_payments_immutable_delete",
            ):
                connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            if corruption == "original_source":
                connection.execute(
                    "UPDATE payments SET source_id = 'src_wrong' WHERE payment_id = ?",
                    (paid.payment_id,),
                )
            elif corruption == "original_generation":
                connection.execute(
                    "UPDATE payments SET decision_generation = decision_generation + 1 "
                    "WHERE payment_id = ?",
                    (paid.payment_id,),
                )
            elif corruption == "original_status":
                connection.execute(
                    "UPDATE payments SET status = 'FAILED', error = 'forged failure' "
                    "WHERE payment_id = ?",
                    (paid.payment_id,),
                )
            elif corruption == "missing_ledger":
                connection.execute("DELETE FROM payments WHERE payment_id = ?", (paid.payment_id,))
            else:
                for index, key in enumerate(("a" * 64, "b" * 64), start=1):
                    connection.execute(
                        "INSERT INTO payments(payment_id, case_id, idempotency_key, vendor, "
                        "amount, currency, status, error, created_at, decision_generation, "
                        "evidence_snapshot_digest, source_id, invoice_number, review_id) "
                        "SELECT ?, 'case_duplicate_attempt', ?, vendor, amount, currency, "
                        "status, error, created_at, decision_generation, "
                        "evidence_snapshot_digest, source_id, invoice_number, review_id "
                        "FROM payments WHERE payment_id = ?",
                        (f"pay_extra_{index}", key, paid.payment_id),
                    )
            connection.commit()

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.merge_relational_case_evidence(attempted)

    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"


def test_rejected_and_injected_failure_never_report_payment_success(
    invoice_dir: Path, settings: Settings
) -> None:
    workflow_db = settings.workflow_db
    store = WorkflowStore(settings)
    rejected_invoice = load(invoice_dir, "invoice_1001.txt", workflow_db.parent / "sources")
    persist_case(store, "case_rejected", rejected_invoice)
    rejected_claim = record_payment_evidence(
        store, "case_rejected", rejected_invoice, settings, DecisionKind.REJECT
    )
    rejected_invoice = store.load_current_extraction(rejected_claim)
    store.save_final_decision(
        "case_rejected",
        FinalDecision(
            decision=DecisionKind.REJECT,
            reasons=["rejected"],
            evidence=[
                reference for line in rejected_invoice.lines for reference in line.evidence[:1]
            ],
            critic_disposition=DecisionKind.REJECT,
            payment_eligible=False,
        ),
        rejected_claim,
    )
    rejected = mock_payment("case_rejected", rejected_invoice, store, workflow_db, rejected_claim)
    assert rejected.status is PaymentStatus.NOT_ELIGIBLE

    failed_invoice = load(invoice_dir, "invoice_1004.json", workflow_db.parent / "sources")
    persist_case(store, "case_failed_payment", failed_invoice)
    failed_claim = record_payment_evidence(store, "case_failed_payment", failed_invoice, settings)
    failed_invoice = store.load_current_extraction(failed_claim)
    approve_final(store, "case_failed_payment", failed_claim)
    failed = mock_payment(
        "case_failed_payment",
        failed_invoice,
        store,
        workflow_db,
        failed_claim,
        inject_failure=True,
    )
    assert failed.status is PaymentStatus.FAILED
    assert failed.error == "injected mock-payment failure"


def test_payment_rejects_missing_risk_snapshot(invoice_dir: Path, settings: Settings) -> None:
    store = WorkflowStore(settings)
    invoice = load(invoice_dir, "invoice_1001.txt", settings.source_archive_dir)
    persist_case(store, "case_missing_payment_risk", invoice)
    claim = record_payment_evidence(store, "case_missing_payment_risk", invoice, settings)
    invoice = store.load_current_extraction(claim)
    approve_final(store, "case_missing_payment_risk", claim)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "DROP TRIGGER IF EXISTS trg_comparison_results_immutable_after_final_delete"
        )
        connection.execute(
            "DELETE FROM comparison_results WHERE case_id = ? AND comparison_type = 'risk'",
            ("case_missing_payment_risk",),
        )
        connection.commit()

    result = mock_payment(
        "case_missing_payment_risk",
        invoice,
        store,
        settings.workflow_db,
        claim,
    )

    assert result.status is PaymentStatus.NOT_ELIGIBLE
    assert result.error is not None
    assert "evidence snapshot is invalid" in result.error
    assert "missing=['risk']" in result.error
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0


def test_payment_rejects_final_decision_from_prior_execution_generation(
    invoice_dir: Path, settings: Settings
) -> None:
    store = WorkflowStore(settings)
    invoice = load(invoice_dir, "invoice_1001.txt", settings.source_archive_dir)
    persist_case(store, "case_stale_payment_decision", invoice)
    stale = record_payment_evidence(store, "case_stale_payment_decision", invoice, settings)
    invoice = store.load_current_extraction(stale)
    approve_final(store, "case_stale_payment_decision", stale)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            ("2000-01-01T00:00:00+00:00", "case_stale_payment_decision"),
        )
        connection.commit()
    current = store.claim_case_execution(
        "case_stale_payment_decision",
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    assert current.generation == stale.generation + 1
    store.adopt_latest_evidence(current)

    result = mock_payment(
        "case_stale_payment_decision",
        invoice,
        store,
        settings.workflow_db,
        current,
    )

    assert result.status is PaymentStatus.NOT_ELIGIBLE
    assert result.error == "final decision is missing or stale"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0
