"""Persisted review decisions, cross-database atomicity, and payment idempotency."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from invoice_agents import cli
from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.hitl.service import create_review_request, record_human_decision
from invoice_agents.models import (
    CanonicalMapping,
    CaseStatus,
    Critique,
    DecisionKind,
    FinalDecision,
    HumanDecisionKind,
    IdentityCandidate,
    IdentityRelationship,
    PaymentStatus,
    ReviewRequest,
)
from invoice_agents.payment.service import mock_payment
from invoice_agents.source_store import snapshot_source
from invoice_agents.tools.comparison import (
    InventoryReader,
    build_risk_assessment,
    compare_inventory,
    compute_invoice_totals,
)
from invoice_agents.tools.evidence import extract_invoice_evidence

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
    store.adopt_latest_evidence(claim)
    comparisons, _ = compare_inventory(invoice, InventoryReader(settings.inventory_db))
    risk = build_risk_assessment(
        invoice, comparisons, [], compute_invoice_totals(invoice), settings
    )
    store.save_identity(case_id, [], claim)
    store.save_comparison(
        case_id,
        "inventory",
        {"comparisons": [item.model_dump(mode="json") for item in comparisons]},
        claim,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    case_critique = critique(disposition)
    store.save_critique(case_id, case_critique, claim)
    if authorize_review:
        review = create_review_request(
            case_id,
            invoice,
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
    store.save_final_decision(
        case_id,
        FinalDecision(
            decision=DecisionKind.APPROVE,
            reasons=["approved evidence"],
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

    store = WorkflowStore(settings.workflow_db)
    invoice = load(source.parent, source.name, settings.source_archive_dir)
    persist_case(store, case_id, invoice, started_at=started_at)
    claim = store.claim_case_execution(
        case_id, frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.adopt_latest_evidence(claim)
    comparisons, _ = compare_inventory(invoice, InventoryReader(settings.inventory_db))
    risk = build_risk_assessment(
        invoice,
        comparisons,
        identity_candidates or [],
        compute_invoice_totals(invoice),
        settings,
    )
    store.save_identity(case_id, identity_candidates or [], claim)
    store.save_comparison(
        case_id,
        "inventory",
        {"comparisons": [item.model_dump(mode="json") for item in comparisons]},
        claim,
    )
    store.save_comparison(case_id, "risk", risk.model_dump(mode="json"), claim)
    case_critique = critique()
    store.save_critique(case_id, case_critique, claim)
    review = create_review_request(
        case_id,
        invoice,
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


def replace_review_evidence(
    workflow_db: Path,
    review: ReviewRequest,
    *,
    inventory: list[dict[str, Any]] | None = None,
    blocking_evidence: list[dict[str, Any]] | None = None,
) -> ReviewRequest:
    bundle = review.evidence_bundle.copy()
    if inventory is not None:
        bundle["inventory"] = inventory
    if blocking_evidence is not None:
        bundle["blocking_evidence"] = blocking_evidence
    changed = review.model_copy(update={"evidence_bundle": bundle}, deep=True)
    with connect_database(workflow_db) as connection:
        connection.execute(
            "UPDATE review_requests SET payload_json = ? WHERE review_id = ?",
            (changed.model_dump_json(), changed.review_id),
        )
        connection.commit()
    return changed


def test_review_request_and_human_decision_are_persisted(
    invoice_dir: Path,
    inventory_db: Path,
    workflow_db: Path,
    settings: Settings,
) -> None:
    store = WorkflowStore(workflow_db)
    inv = load(invoice_dir, "invoice_1002.txt", workflow_db.parent / "sources")
    persist_case(store, "case_review", inv)
    claim = store.claim_case_execution(
        "case_review", frozenset({CaseStatus.INCOMPLETE}), lease_seconds=60
    )
    store.adopt_latest_evidence(claim)
    comparisons, _ = compare_inventory(inv, InventoryReader(inventory_db))
    risk = build_risk_assessment(inv, comparisons, [], compute_invoice_totals(inv), settings)
    store.save_identity("case_review", [], claim)
    store.save_comparison(
        "case_review",
        "inventory",
        {"comparisons": [item.model_dump(mode="json") for item in comparisons]},
        claim,
    )
    store.save_comparison("case_review", "risk", risk.model_dump(mode="json"), claim)
    case_critique = critique()
    store.save_critique("case_review", case_critique, claim)
    review = create_review_request(
        "case_review",
        inv,
        risk,
        case_critique,
        DecisionKind.HOLD,
        ["stock exceeds"],
        store,
        claim,
    )
    store.release_case_execution(claim)
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
            WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
            settings.inventory_db,
            mappings=[mapping("WidgetA (bulk)")],
        )

    assert (
        persisted_decision_state(
            settings.workflow_db, settings.inventory_db, mapping_review.review_id
        )
        == before
    )


def test_semantic_replay_ignores_timestamp_order_and_normalized_whitespace(
    settings: Settings, mapping_review: ReviewRequest
) -> None:
    first_inventory = list(mapping_review.evidence_bundle["inventory"])
    second_inventory = dict(first_inventory[0])
    second_inventory["raw_items"] = ["WidgetA (wholesale)"]
    blockers = [
        {
            "blocker_id": "inventory:bulk:UNKNOWN",
            "kind": "inventory",
            "evidence_id": "bulk",
            "description": "bulk alias requires authorization",
        },
        {
            "blocker_id": "financial:declared-total-delta",
            "kind": "financial",
            "evidence_id": "declared-total-delta",
            "description": "declared total differs",
        },
    ]
    mapping_review = replace_review_evidence(
        settings.workflow_db,
        mapping_review,
        inventory=[*first_inventory, second_inventory],
        blocking_evidence=blockers,
    )
    first = record_human_decision(
        mapping_review.review_id,
        "reviewer@example.com",
        HumanDecisionKind.ESTABLISH_MAPPING,
        "bulk aliases are authorized",
        WorkflowStore(settings.workflow_db),
        settings.inventory_db,
        mappings=[mapping("WidgetA (bulk)"), mapping("WidgetA (wholesale)", "SKU-WIDGET-B")],
        addressed_blocker_ids=[blockers[0]["blocker_id"], blockers[1]["blocker_id"]],
    )
    before_retry = persisted_decision_state(
        settings.workflow_db, settings.inventory_db, mapping_review.review_id
    )

    replayed = record_human_decision(
        mapping_review.review_id,
        " reviewer@example.com ",
        HumanDecisionKind.ESTABLISH_MAPPING,
        "  bulk   aliases\nare authorized  ",
        WorkflowStore(settings.workflow_db),
        settings.inventory_db,
        mappings=[
            mapping("WidgetA (wholesale)", "SKU-WIDGET-B"),
            mapping("WidgetA (bulk)"),
        ],
        addressed_blocker_ids=[blockers[1]["blocker_id"], blockers[0]["blocker_id"]],
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
        WorkflowStore(settings.workflow_db),
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
                WorkflowStore(settings.workflow_db),
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
        WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
        WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
                WorkflowStore(settings.workflow_db),
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
        WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
                WorkflowStore(settings.workflow_db),
                unavailable_inventory,
            )
        assert excinfo.value.stop_reason == "REVIEW_ALREADY_RESOLVED"
    assert not unavailable_inventory.parent.exists()
    assert (
        persisted_decision_state(settings.workflow_db, settings.inventory_db, review.review_id)
        == before
    )


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
            WorkflowStore(settings.workflow_db),
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
            WorkflowStore(settings.workflow_db),
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
                WorkflowStore(settings.workflow_db),
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
    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.human_decision == committed[0].human_decision


def test_supersession_requires_distinct_earlier_review_candidate_for_same_invoice_vendor(
    settings: Settings,
) -> None:
    store = WorkflowStore(settings.workflow_db)
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

    def candidate(case_id: str) -> IdentityCandidate:
        return IdentityCandidate(
            case_id=case_id,
            source_id=original.source.source_id,
            invoice_number=original.invoice_number.normalized_value,
            vendor=original.vendor.normalized_value,
            source_hash=original.source.sha256,
            revision=original.revision.normalized_value if original.revision else None,
            source_format=original.source.source_format,
            relationship=IdentityRelationship.POSSIBLE_REVISION,
            explanation="same invoice/vendor with different revision evidence",
        )

    comparisons, _ = compare_inventory(revised, InventoryReader(settings.inventory_db))
    risk = build_risk_assessment(
        revised,
        comparisons,
        [candidate("case_prior_revision"), candidate("case_later_revision")],
        compute_invoice_totals(revised),
        settings,
    )
    claim = store.claim_case_execution(
        "case_current_revision",
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    store.adopt_latest_evidence(claim)
    candidates = [candidate("case_prior_revision"), candidate("case_later_revision")]
    store.save_identity("case_current_revision", candidates, claim)
    store.save_comparison(
        "case_current_revision",
        "inventory",
        {"comparisons": [item.model_dump(mode="json") for item in comparisons]},
        claim,
    )
    store.save_comparison("case_current_revision", "risk", risk.model_dump(mode="json"), claim)
    case_critique = critique()
    store.save_critique("case_current_revision", case_critique, claim)
    review = create_review_request(
        "case_current_revision",
        revised,
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
    stored = WorkflowStore(settings.workflow_db).load_review(review.review_id)
    assert stored.human_decision is not None
    assert stored.human_decision.addressed_blocker_ids == blocker_ids


def test_mock_payment_pays_once_across_duplicate_representations(
    invoice_dir: Path, settings: Settings
) -> None:
    workflow_db = settings.workflow_db
    store = WorkflowStore(workflow_db)
    text = load(invoice_dir, "invoice_1011.txt", workflow_db.parent / "sources")
    pdf = load(invoice_dir, "invoice_1011.pdf", workflow_db.parent / "sources")
    persist_case(store, "case_text", text)
    text_claim = record_payment_evidence(store, "case_text", text, settings)
    approve_final(store, "case_text", text_claim)
    first = mock_payment("case_text", text, store, workflow_db, text_claim)
    assert first.status is PaymentStatus.PAID
    persist_case(store, "case_pdf", pdf)
    pdf_claim = record_payment_evidence(store, "case_pdf", pdf, settings, authorize_review=True)
    approve_final(store, "case_pdf", pdf_claim)
    duplicate = mock_payment("case_pdf", pdf, store, workflow_db, pdf_claim)
    assert duplicate.status is PaymentStatus.DUPLICATE
    assert duplicate.case_id == "case_pdf"
    assert duplicate.duplicate_of == first.payment_id
    with connect_database(workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 1


def test_rejected_and_injected_failure_never_report_payment_success(
    invoice_dir: Path, settings: Settings
) -> None:
    workflow_db = settings.workflow_db
    store = WorkflowStore(workflow_db)
    rejected_invoice = load(invoice_dir, "invoice_1001.txt", workflow_db.parent / "sources")
    persist_case(store, "case_rejected", rejected_invoice)
    rejected_claim = record_payment_evidence(
        store, "case_rejected", rejected_invoice, settings, DecisionKind.REJECT
    )
    store.save_final_decision(
        "case_rejected",
        FinalDecision(
            decision=DecisionKind.REJECT,
            reasons=["rejected"],
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
    store = WorkflowStore(settings.workflow_db)
    invoice = load(invoice_dir, "invoice_1001.txt", settings.source_archive_dir)
    persist_case(store, "case_missing_payment_risk", invoice)
    claim = record_payment_evidence(store, "case_missing_payment_risk", invoice, settings)
    approve_final(store, "case_missing_payment_risk", claim)
    with connect_database(settings.workflow_db) as connection:
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
    assert result.error == "latest risk assessment is missing or stale"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0


def test_payment_rejects_final_decision_from_prior_execution_generation(
    invoice_dir: Path, settings: Settings
) -> None:
    store = WorkflowStore(settings.workflow_db)
    invoice = load(invoice_dir, "invoice_1001.txt", settings.source_archive_dir)
    persist_case(store, "case_stale_payment_decision", invoice)
    stale = record_payment_evidence(store, "case_stale_payment_decision", invoice, settings)
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
